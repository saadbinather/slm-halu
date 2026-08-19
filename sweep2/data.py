import os
import json
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch_geometric.data import Data

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device for trace extraction: {device}")

# 1. Load Tokenizer & Model
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Changed torch_dtype to dtype to fix the transformers deprecation warning
qwen = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    attn_implementation="eager"
).to(device)

# 2. Load 1,000 HaluEval QA Entries
data_path = "../halueval_data/qa_data.json"
if not os.path.exists(data_path):
    data_path = "../halueval_data/qa_data.jsonl"

entries = []
decoder = json.JSONDecoder()

with open(data_path, "r", encoding="utf-8") as f:
    content = f.read()

idx = 0
length = len(content)

while idx < length and len(entries) < 1000:
    while idx < length and content[idx].isspace():
        idx += 1
    if idx >= length:
        break
    try:
        obj, end_idx = decoder.raw_decode(content, idx)
        entries.append(obj)
        idx = end_idx
    except json.JSONDecodeError:
        idx += 1

print(f"Successfully loaded {len(entries)} HaluEval entries!")

output_dir = "halueval_traces_1k"
os.makedirs(output_dir, exist_ok=True)

# 3. Extraction Function (Forward pass + Answer Token Slicing)
def extract_answer_trace(question, answer, label):
    q_text = f"Question: {question}\nAnswer:"
    full_text = f"Question: {question}\nAnswer: {answer}"
    
    q_len = tokenizer(q_text, return_tensors="pt").input_ids.shape[1]
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = qwen(**inputs, output_hidden_states=True)
        
    # Extract activation representations across all layers for Answer tokens only
    node_features = []
    for layer_hs in outputs.hidden_states:
        ans_hs = layer_hs[0, q_len:, :].cpu()
        if ans_hs.size(0) == 0:  # Fallback if empty slice
            ans_hs = layer_hs[0, -1:, :].cpu()
        layer_feat = ans_hs.mean(dim=0)
        node_features.append(layer_feat)
        
    x = torch.stack(node_features).float()
    
    # Layer sequential edges
    num_layers = x.size(0)
    src, dst = list(range(num_layers - 1)), list(range(1, num_layers))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    
    del inputs, outputs
    torch.mps.empty_cache()
    gc.collect()
    
    return data

# 4. Extract 1,000 Factual + 1,000 Hallucinated Traces
print("Extracting layer-activation graphs...")

for idx, entry in tqdm(enumerate(entries), total=len(entries)):
    q = entry['question']
    r_ans = str(entry['right_answer']).strip()
    h_ans = str(entry['hallucinated_answer']).strip()
    
    # Right Answer Trace -> Label 0 (Factual)
    g_factual = extract_answer_trace(q, r_ans, label=0)
    torch.save(g_factual, os.path.join(output_dir, f"trace_{idx}_factual.pt"))
    
    # Hallucinated Answer Trace -> Label 1 (Hallucinated)
    g_halu = extract_answer_trace(q, h_ans, label=1)
    # Fixed typo: outpu. _dir is now output_dir
    torch.save(g_halu, os.path.join(output_dir, f"trace_{idx}_halu.pt"))

print(f"\nSaved 2,000 balanced graphs to '{output_dir}' directory!")