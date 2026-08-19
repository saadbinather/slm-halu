import os
import gc
import torch
import random
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Setup Device & Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    attn_implementation="eager"
).to(device)

# 2. Config & Output Directories
base_dir = "MetaQA"
hops = ["1-hop", "2-hop", "3-hop"]
samples_per_hop = 500
out_dir = "metaqa_traces"
os.makedirs(out_dir, exist_ok=True)

stats = {
    "1-hop": {"total": 0, "hallucinated": 0},
    "2-hop": {"total": 0, "hallucinated": 0},
    "3-hop": {"total": 0, "hallucinated": 0}
}

def load_metaqa_samples(file_path, n_samples):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.seed(42)
    sampled_lines = random.sample(lines, min(n_samples, len(lines)))
    data = []
    for line in sampled_lines:
        q, a = line.strip().split('\t')
        valid_answers = [ans.lower().strip() for ans in a.split('|')]
        data.append((q, valid_answers, valid_answers[0])) # Store all valid, and first as default gold
    return data

# 3. Main Processing Loop
trace_counter = 0

for hop in hops:
    print(f"\n--- Processing {hop} ---")
    file_path = os.path.join(base_dir, hop, "vanilla", "qa_test.txt")
    qa_pairs = load_metaqa_samples(file_path, samples_per_hop)
    
    for q, valid_answers, primary_gold in tqdm(qa_pairs, desc=f"Extracting {hop}"):
        
        # ==========================================
        # RUN 1: SLM GENERATION (Predict & Label)
        # ==========================================
        prompt_gen = f"Question: {q}\nAnswer strictly in 1 to 5 words:"
        inputs_gen = tokenizer(prompt_gen, return_tensors="pt").to(device)
        input_len = inputs_gen.input_ids.shape[1]
        
        with torch.no_grad():
            outputs_gen = model.generate(
                **inputs_gen,
                max_new_tokens=15,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        gen_tokens = outputs_gen.sequences[0][input_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()
        
        # Verify Hallucination (String Matching against all valid answers)
        is_halu = 1 # Default to Hallucination
        for ans in valid_answers:
            if ans in gen_text:
                is_halu = 0
                break
                
        # Update Stats
        stats[hop]["total"] += 1
        stats[hop]["hallucinated"] += is_halu
        
        # Slice Generated Hidden States & Attentions (take mean to save Memory)
        hidden_gen = [h.squeeze(0).mean(dim=0).cpu() for h in outputs_gen.hidden_states[0]]
        # Attentions format: Tuple of (layers) -> shape [batch, heads, seq, seq]
        # We mean across heads and seq length to save space, keeping layer-wise representation
        attn_gen = [a.squeeze(0).mean(dim=(0, 1, 2)).cpu() for a in outputs_gen.attentions[0]]

        trace_gen = {
            "question": q,
            "generated_text": gen_text,
            "gold_answers": valid_answers,
            "hop_level": hop,
            "run_type": "generation",
            "label": is_halu,
            "hidden_states": hidden_gen,
            "attentions": attn_gen
        }
        
        torch.save(trace_gen, os.path.join(out_dir, f"trace_{trace_counter}_gen.pt"))
        
        # Cleanup Run 1
        del inputs_gen, outputs_gen
        torch.mps.empty_cache()

        # ==========================================
        # RUN 2: GOLD ANSWER / TEACHER FORCING
        # ==========================================
        prompt_gold = f"Question: {q}\nAnswer strictly in 1 to 5 words: {primary_gold}"
        inputs_gold = tokenizer(prompt_gold, return_tensors="pt").to(device)
        
        # Find where the gold answer starts to slice tensors
        # Prompt without gold answer to get exact token boundary
        boundary_len = tokenizer(f"Question: {q}\nAnswer strictly in 1 to 5 words:", return_tensors="pt").input_ids.shape[1]
        
        with torch.no_grad():
            outputs_gold = model(**inputs_gold, output_hidden_states=True, output_attentions=True)
            
        # Extract Hidden States & Attentions specifically for the Answer Tokens
        # hidden_states is tuple of 37 tensors of shape [1, seq_len, hidden_dim]
        hidden_gold = [h[:, boundary_len:, :].squeeze(0).mean(dim=0).cpu() for h in outputs_gold.hidden_states]
        attn_gold = [a[:, :, boundary_len:, :].squeeze(0).mean(dim=(0, 1, 2)).cpu() for a in outputs_gold.attentions]
        
        trace_gold = {
            "question": q,
            "gold_answer": primary_gold,
            "hop_level": hop,
            "run_type": "gold_forced",
            "label": 0, # STRICTLY NON-HALLUCINATED
            "hidden_states": hidden_gold,
            "attentions": attn_gold
        }
        
        torch.save(trace_gold, os.path.join(out_dir, f"trace_{trace_counter}_gold.pt"))
        trace_counter += 1
        
        # Cleanup Run 2
        del inputs_gold, outputs_gold
        torch.mps.empty_cache()
        gc.collect()

# 4. Print Hop-wise Hallucination Report
print("\n" + "="*40)
print("HOP-WISE HALLUCINATION REPORT")
print("="*40)
for hop in hops:
    t = stats[hop]["total"]
    h = stats[hop]["hallucinated"]
    rate = (h / t) * 100 if t > 0 else 0
    print(f"{hop.upper()}: {h}/{t} Hallucinated ({rate:.2f}%)")
print("="*40)
print(f"Extraction complete! Saved {trace_counter * 2} trace files in '{out_dir}/'")