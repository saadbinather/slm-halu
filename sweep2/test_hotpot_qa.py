import os
import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Define & Load Trained GNN Architecture
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

gnn_model = HallucinationGNN(in_channels=2048).to(device)
gnn_model.load_state_dict(torch.load("gnn_halueval_model.pt", map_location=device, weights_only=True))
gnn_model.eval()
print("Loaded HaluEval-trained GNN model!")

# 2. Load Qwen LLM & Tokenizer
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
qwen = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager"
).to(device)

# 3. Load HotpotQA (Multihop Dataset)
print("Downloading/Loading HotpotQA from HuggingFace...")
dataset = load_dataset("hotpot_qa", "distractor", split="validation[:200]")

questions = [item['question'] for item in dataset]
true_answers = [item['answer'] for item in dataset]

# Create "Hallucinated" answers via negative sampling (shuffling true answers)
wrong_answers = true_answers.copy()
random.Random(42).shuffle(wrong_answers)
# Ensure no accidental matches
for i in range(len(wrong_answers)):
    if wrong_answers[i] == true_answers[i]:
        wrong_answers[i] = wrong_answers[(i+1) % len(wrong_answers)]

print(f"Prepared 200 Factual and 200 Hallucinated Multihop traces (Total: 400).")

# 4. Extraction & Prediction Helper
def extract_and_predict(question, answer, label):
    q_text = f"Question: {question}\nAnswer:"
    full_text = f"Question: {question}\nAnswer: {answer}"
    
    q_len = tokenizer(q_text, return_tensors="pt").input_ids.shape[1]
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = qwen(**inputs, output_hidden_states=True)
        
    node_features = []
    for layer_hs in outputs.hidden_states:
        ans_hs = layer_hs[0, q_len:, :].cpu()
        if ans_hs.size(0) == 0:
            ans_hs = layer_hs[0, -1:, :].cpu()
        node_features.append(ans_hs.mean(dim=0))
        
    x = torch.stack(node_features).float()
    num_layers = x.size(0)
    src, dst = list(range(num_layers - 1)), list(range(1, num_layers))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        out = gnn_model(batch.x, batch.edge_index, batch.batch)
        prob = F.softmax(out, dim=1)[:, 1].item()
        pred = out.argmax(dim=1).item()
        
    del inputs, outputs
    return prob, pred

# 5. Evaluate Loop
y_true, y_pred, y_probs = [], [], []

print("Extracting and Evaluating Multihop Traces...")
for i in tqdm(range(200)):
    q = questions[i]
    r_ans = true_answers[i]
    h_ans = wrong_answers[i]
    
    # 1. Factual Test (Label 0)
    prob_f, pred_f = extract_and_predict(q, r_ans, 0)
    y_true.append(0)
    y_pred.append(pred_f)
    y_probs.append(prob_f)
    
    # 2. Hallucinated Test (Label 1)
    prob_h, pred_h = extract_and_predict(q, h_ans, 1)
    y_true.append(1)
    y_pred.append(pred_h)
    y_probs.append(prob_h)
    
    gc.collect()
    torch.mps.empty_cache()

# 6. Final Results
auroc = roc_auc_score(y_true, y_probs)
aupr = average_precision_score(y_true, y_probs)

print("\n" + "="*60)
print("--- CROSS-DOMAIN MULTIHOP TEST RESULTS (HOTPOT QA) ---")
print("="*60)
print(classification_report(y_true, y_pred, target_names=["Factual (0)", "Hallucinated (1)"]))
print(f"ROC-AUC Score (AUROC): {auroc:.4f}")
print(f"PR-AUC Score (AUPR):   {aupr:.4f}")
print("="*60)
print("Hypothesis Check: Did the 0.97 F1-score drop significantly on multihop?")