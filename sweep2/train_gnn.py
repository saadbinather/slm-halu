import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Device config
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Traces
folder = "halueval_traces_1k"
files = [f for f in os.listdir(folder) if f.endswith('.pt')]

graph_list = []
print(f"Loading {len(files)} traces from {folder}...")

for file in files:
    path = os.path.join(folder, file)
    data = torch.load(path, map_location="cpu", weights_only=False)
    graph_list.append(data)

# 2. Train / Test Split (80/20)
train_data, test_data = train_test_split(
    graph_list, test_size=0.2, random_state=42, stratify=[g.y.item() for g in graph_list]
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Train Set: {len(train_data)} graphs")
print(f"Test Set: {len(test_data)} graphs")

# 3. Define GNN Architecture
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

input_dim = graph_list[0].x.size(1)
model = HallucinationGNN(in_channels=input_dim).to(device)

criterion = nn.CrossEntropyLoss() # No class weights needed now!
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 4. Training Loop
epochs = 30
print("\nStarting Training...")

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss / len(train_data):.4f}")

# Save the model
torch.save(model.state_dict(), "gnn_halueval_model.pt")
print("\nModel saved to 'gnn_halueval_model.pt'")

# 5. Evaluation on Unseen Test Split
model.eval()
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        
        y_true.extend(batch.y.squeeze().cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

auroc = roc_auc_score(y_true, y_probs)
aupr = average_precision_score(y_true, y_probs)

print("\n--- HaluEval Balanced Test Results (400 Unseen Samples) ---")
print(classification_report(y_true, y_pred, target_names=["Factual (0)", "Hallucinated (1)"]))
print(f"ROC-AUC (AUROC): {auroc:.4f}")
print(f"PR-AUC (AUPR):   {aupr:.4f}")