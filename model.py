"""
Load the LLM(s) used in the CHARM paper (arXiv:2509.24770) and extract
attention matrices + residual-stream activations needed to build the
computational trace graphs described in Section 3.

Models used in the paper (Section D.3):
- meta-llama/Llama-2-7b-chat-hf   (gated -> needs HF token + license accept)
- mistralai/Mistral-7B-Instruct-v0.3
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pick one:
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Needed only for gated models like Llama-2. Get one from https://huggingface.co/settings/tokens
HF_TOKEN = None  # e.g. "hf_xxx..."


def load_model(model_name: str, hf_token: str | None = None):
    # Pick device: Apple Silicon GPU (mps) if available, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16   # mps supports fp16
    else:
        device = "cpu"
        dtype = torch.float32   # cpu: fp16 is slow/unstable, use fp32

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype=dtype,
        attn_implementation="eager",  # needed so output_attentions actually returns real attn maps
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def get_traces(prompt: str, response: str, tokenizer, model):
    """Teacher-force prompt+response through the model and return attentions + hidden states."""
    text = prompt + response
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(
            **inputs,
            output_attentions=True,   # -> tuple of (L) tensors, each [batch, H, n, n]
            output_hidden_states=True,  # -> tuple of (L+1) tensors, each [batch, n, d]
        )

    attentions = out.attentions        # per-layer attention matrices A^{l,h}
    hidden_states = out.hidden_states  # per-layer residual stream activations a^l_i
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, attentions, hidden_states


if __name__ == "__main__":
    tok, model = load_model(MODEL_NAME, HF_TOKEN)

    prompt = "Question: What is the capital of France?\nAnswer:"
    response = " The capital of France is Paris."

    tokens, attentions, hidden_states = get_traces(prompt, response, tok, model)

    print(f"Num tokens: {len(tokens)}")
    print(f"Num attention layers: {len(attentions)}, shape per layer: {attentions[0].shape}")
    print(f"Num hidden-state layers: {len(hidden_states)}, shape per layer: {hidden_states[0].shape}")