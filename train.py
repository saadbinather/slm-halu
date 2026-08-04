import torch
import pandas as pd
import os
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Setup Device (Apple Silicon MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Model & Tokenizer
model_id = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Yahan change kiya hai: device_map hata kar .to(device) lagaya hai
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, # Yeh RAM ko crash hone se bachayega
    attn_implementation="eager"  # <-- YEH LINE ADD KAREIN
).to(device)



# 3. Load & Split Dataset
print("Loading Movies Dataset...")
train_url = "https://raw.githubusercontent.com/technion-cs-nlp/LLMsKnow/main/data/movie_qa_train.csv"
df_train = pd.read_csv(train_url)

# 2000 samples for training
df_train_small = df_train.sample(n=2000, random_state=42).reset_index(drop=True)
print(f"Dataset ready! Training on {len(df_train_small)} samples.")

# 4. Trace Extraction Loop
os.makedirs("movie_traces_train2", exist_ok=True)
print("Starting extraction loop...")

for i, row in tqdm(df_train_small.iterrows(), total=len(df_train_small)):
    question = row['Question']
    gold_answer = row['Answer']
    
    prompt = f"Question: {question}\nAnswer strictly in 1 to 3 words:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15, 
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    # Get Generated Text
    gen_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # Extract Traces & Move to CPU instantly
    hidden_states = [h.cpu() for h in outputs.hidden_states[0]]
    attentions = [a.cpu() for a in outputs.attentions[0]]
    
    # Save to disk
    sample_data = {
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": generated_text,
        "hidden_states": hidden_states,
        "attentions": attentions
    }
    
    torch.save(sample_data, f"movie_traces_train2/sample_{i}.pt")
    
    # Strict Cleanup for Mac VRAM
    del inputs, outputs, sample_data, hidden_states, attentions
    torch.mps.empty_cache()
    gc.collect()

print("\n✅ All 2000 train traces extracted and saved locally!")