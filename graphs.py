import os
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch_geometric.data import Data

# 1. Mac Apple Silicon MPS Device Setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon GPU (MPS engine).")
else:
    device = torch.device("cpu")
    print("⚠️ MPS engine not found. Running on CPU.")

# 2. Configs
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
SAVE_DIR = "extracted_graphs"
TAU_THRESHOLD = 0.05
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Loading Model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": device}
)
model.eval()

def extract_traces(prompt, response, target_layer=12):
    """Extracts internal attention matrices and layer activations from SLM."""
    full_text = f"Question: {prompt}\nAnswer: {response}"
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            **inputs, 
            output_attentions=True, 
            output_hidden_states=True
        )
    
    # Extract attentions [L, H, N, N] and activations [N, D]
    attentions = torch.stack([attn[0] for attn in outputs.attentions]).cpu()
    activations = outputs.hidden_states[target_layer][0].cpu()
    
    # MPS Memory Clean Up
    del outputs, inputs
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    
    return attentions, activations

def build_charm_graph(attentions, activations, tau=0.05):
    """Converts extracted tensors into PyTorch Geometric Data format."""
    num_layers, num_heads, N, _ = attentions.shape
    
    # Flatten edge features across layers and heads
    edge_features_full = attentions.permute(2, 3, 0, 1).reshape(N, N, num_layers * num_heads)
    
    # Node Features: Self-attention diagonals + Layer Activations
    diag_indices = torch.arange(N)
    self_attentions = edge_features_full[diag_indices, diag_indices, :]
    node_features = torch.cat([self_attentions, activations], dim=-1)
    
    # Tau-Sparsification Rule (CHARM)
    max_attn_scores, _ = edge_features_full.max(dim=-1)
    mask = (max_attn_scores > tau) & (torch.tril(torch.ones(N, N), diagonal=-1).bool())
    edge_index = mask.nonzero(as_tuple=False).t()
    
    if edge_index.shape[1] == 0:  # Fallback for ultra-sparse graphs
        edge_index = (torch.tril(torch.ones(N, N), diagonal=-1).bool()).nonzero(as_tuple=False).t()

    edge_attr = edge_features_full[edge_index[0], edge_index[1], :]
    
    return Data(
        x=node_features.float(),
        edge_index=edge_index,
        edge_attr=edge_attr.float(),
        num_nodes=N
    )

if __name__ == "__main__":
    print("Loading HaluEval QA dataset...")
    # First 10 samples for fast baseline test
    halu_ds = load_dataset("fka/halueval", "qa_samples", split="train[:10]") 
    
    count = 0
    print("\n--- Extracting Traces & Saving Graphs to Disk ---")
    for idx, sample in enumerate(tqdm(halu_ds, desc="Processing Samples")):
        q = sample['question']
        
        # 1. Non-Hallucinated Sample (Label = 0)
        atts_r, acts_r = extract_traces(q, sample['right_answer'])
        g_right = build_charm_graph(atts_r, acts_r, tau=TAU_THRESHOLD)
        g_right.y = torch.tensor([0.0])
        torch.save(g_right, os.path.join(SAVE_DIR, f"graph_{count}_label_0.pt"))
        count += 1
        
        # 2. Hallucinated Sample (Label = 1)
        atts_h, acts_h = extract_traces(q, sample['hallucinated_answer'])
        g_hallu = build_charm_graph(atts_h, acts_h, tau=TAU_THRESHOLD)
        g_hallu.y = torch.tensor([1.0])
        torch.save(g_hallu, os.path.join(SAVE_DIR, f"graph_{count}_label_1.pt"))
        count += 1

    print(f"\n✅ Extraction Complete! {count} attributed graph files saved in '{SAVE_DIR}/'.")