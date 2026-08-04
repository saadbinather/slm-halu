import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)

folder = "movie_traces_train"
files = [f for f in os.listdir(folder) if f.endswith('.pt')]

factual_count, halu_count = 0, 0

print("Labeling 2000 samples with LLM Verifier...")
for file in tqdm(files):
    path = os.path.join(folder, file)
    data = torch.load(path, map_location="cpu")
    
    gold = str(data['gold_answer']).strip().lower()
    gen = str(data['generated_answer']).strip().lower()
    
    # Fast path: exact substring match
    if gold in gen:
        label = 0
    else:
        # LLM Verification
        prompt = (
            f"Question: {data['question']}\n"
            f"Gold Answer: {data['gold_answer']}\n"
            f"Generated Answer: {data['generated_answer']}\n"
            f"Does the Generated Answer contain the correct info from Gold Answer? Answer ONLY Yes or No:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3)
        verdict = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        
        label = 0 if "yes" in verdict else 1

    # Update saved .pt file with label
    data['label'] = label
    torch.save(data, path)
    
    if label == 0: factual_count += 1
    else: halu_count += 1

print(f"\n Done Labeling!")
print(f"Factual (0): {factual_count} | Hallucinated (1): {halu_count}")