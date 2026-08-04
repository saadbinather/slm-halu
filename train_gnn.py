import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device for GNN training: {device}")

# 1. Convert .pt files to PyG Graph Objects
folder = "movie_traces_train"
files = [f for f in os.listdir(folder) if f.endswith('.pt')]

graph_list = []
print("Constructing Layer-wise Activation Graphs...")

for file in tqdm(files):
    path = os.path.join(folder, file)
    sample = torch.load(path, map_location="cpu")
    
    # Extract Hidden States (37 Layers)
    hidden_states = sample['hidden_states']  # List of [1, seq_len, hidden_dim]
    
    # Node Feature: Mean pooled representation for each layer
    # Nodes = Layers (37 nodes), Feature Dim = Hidden Dim (e.g. 2048/2816)
    node_features = [h.squeeze(0).mean(dim=0) for h in hidden_states]
    x = torch.stack(node_features).float()  # Shape: [num_layers, hidden_dim]
    
    # Edges: Sequential flow between layers (0->1, 1->2, ..., 35->36)
    num_layers = x.size(0)
    source_nodes = list(range(num_layers - 1))
    target_nodes = list(range(1, num_layers))
    
    # Bidirectional sequential edges
    edge_index = torch.tensor([
        source_nodes + target_nodes,
        target_nodes + source_nodes
    ], dtype=torch.long)
    
    label = torch.tensor([sample['label']], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=label)
    graph_list.append(data)

# 2. Train / Test Split
train_data, test_data = train_test_split(graph_list, test_size=0.2, random_state=42, stratify=[g.y.item() for g in graph_list])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 3. Define GNN Architecture
class HallucinationGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 2)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global Graph Pooling (Combine layer representations into single graph vector)
        x = global_mean_pool(x, batch)
        
        out = self.fc(x)
        return out

input_dim = graph_list[0].x.size(1)
model = HallucinationGNN(in_channels=input_dim).to(device)

# Class Weighting for Imbalance (100 Factual vs 1900 Hallucinated)
# Class 0 Weight = 1900 / 100 = 19.0, Class 1 Weight = 1.0
class_weights = torch.tensor([19.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 4. Training Loop
print("\nStarting GNN Training...")
epochs = 30

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        
    train_loss = total_loss / len(train_data)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f}")

# 5. Evaluation
model.eval()
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

print("\n--- Final GNN Performance Results ---")
print(classification_report(y_true, y_pred, target_names=["Factual (0)", "Hallucinated (1)"]))
print(f"ROC-AUC Score: {roc_auc_score(y_true, y_probs):.4f}")

torch.save(model.state_dict(), "gnn_model.pt")