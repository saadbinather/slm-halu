import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set device to Apple Silicon (MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Aise load karein:
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to("mps")

# QnA Setup
prompt = "What is the capital of Pakistan? Short answer."
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

# Generate response
print("Generating...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)

# Extract and print only the generated text
generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\nResponse:", generated_text)