
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 1. GNN Architecture ---
class HallucinationGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 2)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# --- 2. Load Model & Tokenizer ---
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
qwen = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager"
).to(device)

# --- 3. Extract & Label 500 Test Traces ---
test_url = "https://raw.githubusercontent.com/technion-cs-nlp/LLMsKnow/main/data/movie_qa_test.csv"
df_test = pd.read_csv(test_url).sample(n=500, random_state=100).reset_index(drop=True)

test_graphs = []
print("Extracting traces & labeling 500 test samples...")

for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
    q, gold = row['Question'], str(row['Answer']).strip()
    prompt = f"Question: {q}\nAnswer strictly in 1 to 3 words:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = qwen.generate(
            **inputs, max_new_tokens=15, 
            output_hidden_states=True, return_dict_in_generate=True
        )
    
    gen_tokens = out.sequences[0][inputs.input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    # LLM-as-a-judge
    if gold.lower() in gen_text.lower():
        label = 0
    else:
        judge_prompt = f"Question: {q}\nGold Answer: {gold}\nGenerated Answer: {gen_text}\nDoes the Generated Answer contain the correct info from Gold Answer? Answer ONLY Yes or No:"
        j_inputs = tokenizer(judge_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            j_out = qwen.generate(**j_inputs, max_new_tokens=3)
        verdict = tokenizer.decode(j_out[0][j_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        label = 0 if "yes" in verdict else 1
        
    # Build Graph
    hidden_states = [h.cpu() for h in out.hidden_states[0]]
    node_features = [h.squeeze(0).mean(dim=0) for h in hidden_states]
    x = torch.stack(node_features).float()
    
    num_layers = x.size(0)
    src, dst = list(range(num_layers - 1)), list(range(1, num_layers))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    
    graph_data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    test_graphs.append(graph_data)
    
    del inputs, out, hidden_states
    torch.mps.empty_cache()
    gc.collect()

gnn = HallucinationGNN(in_channels=test_graphs[0].x.size(1)).to(device)
gnn.load_state_dict(torch.load("gnn_model.pt", map_location=device))
gnn.eval()

test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = gnn(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(out, dim=1)[:, 1] # Probability for Hallucination (Class 1)
        preds = out.argmax(dim=1)
        
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

# Calculate Metrics
auroc = roc_auc_score(y_true, y_probs)
aupr = average_precision_score(y_true, y_probs)

print("\n--- Unseen Test Set Results ---")
print(classification_report(y_true, y_pred, target_names=["Factual (0)", "Hallucinated (1)"]))
print(f"ROC-AUC Score (AUROC): {auroc:.4f}")
print(f"PR-AUC Score (AUPR):   {aupr:.4f}")