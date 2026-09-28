import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import subgraph
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

FEATURE_DIR = Path("processed_features")
GRAPH_DIR = Path("graphs")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# METRICS
# label 0 = correct
# label 1 = hallucinated
# =========================================================

def find_best_threshold(y_true, probs):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs >= threshold).astype(np.int64)

        score = balanced_accuracy_score(
            y_true,
            preds
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def compute_metrics(y_true, probs, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)

    preds = (probs >= threshold).astype(np.int64)

    return {
        "threshold": float(threshold),
        "auroc": float(
            roc_auc_score(y_true, probs)
        ),
        "auprc_hallucinated": float(
            average_precision_score(
                y_true,
                probs
            )
        ),
        "auprc_correct": float(
            average_precision_score(
                1 - y_true,
                1.0 - probs
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                preds,
                average="macro"
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                preds
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                preds
            )
        ),
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"AUROC:              {metrics['auroc']:.4f}")
    print(f"AUPRC hallucinated: {metrics['auprc_hallucinated']:.4f}")
    print(f"AUPRC correct:      {metrics['auprc_correct']:.4f}")
    print(f"Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Threshold:           {metrics['threshold']:.3f}")


# =========================================================
# LOAD t=0 FEATURE DATA
# =========================================================

def feature_files(split):
    folder = FEATURE_DIR / split

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(folder.glob("*.pt"))

    if not files:
        raise RuntimeError(f"No processed feature files in {folder}")

    return files


def load_t0_features(split):
    """
    Returns three views of exactly the same t=0 activations:

    final_layer:
        [N, 16]

    mean_layers:
        [N, 16]

    all_layers_flat:
        [N, 28*16] = [N, 448]
    """

    final_layer = []
    mean_layers = []
    all_layers_flat = []
    labels = []

    for path in tqdm(
        feature_files(split),
        desc=f"t=0 features {split}"
    ):
        trace = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        label = int(trace["label"])

        if label not in [0, 1]:
            continue

        times = [
            int(t)
            for t in trace["time_positions"]
        ]

        if 0 not in times:
            raise RuntimeError(
                f"{path.name} has no t=0"
            )

        t0_idx = times.index(0)

        # [28, 16]
        x = trace["features"][
            t0_idx
        ].float()

        final_layer.append(
            x[-1].numpy()
        )

        mean_layers.append(
            x.mean(dim=0).numpy()
        )

        all_layers_flat.append(
            x.reshape(-1).numpy()
        )

        labels.append(label)

    return {
        "final_layer":
            np.asarray(
                final_layer,
                dtype=np.float32
            ),

        "mean_layers":
            np.asarray(
                mean_layers,
                dtype=np.float32
            ),

        "all_layers_flat":
            np.asarray(
                all_layers_flat,
                dtype=np.float32
            ),

        "y":
            np.asarray(
                labels,
                dtype=np.int64
            ),
    }


# =========================================================
# LOGISTIC REGRESSION BASELINE
# =========================================================

def run_logreg(
    name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed
):
    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=seed
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    val_probs = model.predict_proba(
        X_val
    )[:, 1]

    threshold = find_best_threshold(
        y_val,
        val_probs
    )

    test_probs = model.predict_proba(
        X_test
    )[:, 1]

    val_metrics = compute_metrics(
        y_val,
        val_probs,
        threshold
    )

    test_metrics = compute_metrics(
        y_test,
        test_probs,
        threshold
    )

    print_metrics(
        "Validation",
        val_metrics
    )

    print_metrics(
        "Test",
        test_metrics
    )

    return {
        "name":
            name,

        "input_dim":
            int(X_train.shape[1]),

        "validation":
            val_metrics,

        "test":
            test_metrics,
    }


# =========================================================
# t=0 ALL-LAYER MLP
# =========================================================

class T0MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden1=256,
        hidden2=64,
        dropout=0.30
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden1
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden1,
                hidden2
            ),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(
                hidden2,
                1
            )
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict_mlp(
    model,
    X,
    device,
    batch_size
):
    model.eval()

    tensor_x = torch.tensor(
        X,
        dtype=torch.float32
    )

    loader = TorchLoader(
        TensorDataset(tensor_x),
        batch_size=batch_size,
        shuffle=False
    )

    probs = []

    for (batch_x,) in loader:
        batch_x = batch_x.to(device)

        logits = model(batch_x)

        probs.append(
            torch.sigmoid(
                logits
            ).cpu().numpy()
        )

    return np.concatenate(probs)


def run_t0_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    args,
    device
):
    print("\n" + "=" * 70)
    print("t=0 ALL-LAYER MLP")
    print("=" * 70)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)

    model = T0MLP(
        input_dim=X_train.shape[1]
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: {total_params:,}"
    )

    train_x = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    train_y = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    loader = TorchLoader(
        TensorDataset(
            train_x,
            train_y
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    num_neg = float(
        np.sum(y_train == 0)
    )

    num_pos = float(
        np.sum(y_train == 1)
    )

    pos_weight = (
        num_neg
        / max(num_pos, 1.0)
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    checkpoint_path = (
        MODEL_DIR
        / f"t0_all_baselines_mlp_seed{args.seed}.pt"
    )

    best_val = -1.0
    no_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1
    ):
        model.train()

        running_loss = 0.0
        seen = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()
            optimizer.step()

            n = batch_x.shape[0]

            running_loss += (
                loss.item() * n
            )

            seen += n

        train_loss = (
            running_loss
            / max(seen, 1)
        )

        val_probs = predict_mlp(
            model,
            X_val,
            device,
            args.batch_size
        )

        val_auroc = roc_auc_score(
            y_val,
            val_probs
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_AUROC={val_auroc:.4f}"
        )

        if val_auroc > best_val:
            best_val = float(
                val_auroc
            )

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_dim":
                        int(X_train.shape[1]),

                    "seed":
                        int(args.seed),

                    "trainable_parameters":
                        int(total_params),

                    "best_val_auroc":
                        best_val,
                },
                checkpoint_path
            )

        else:
            no_improvement += 1

        if (
            no_improvement
            >= args.patience
        ):
            print(
                f"\nEarly stopping after "
                f"{epoch} epochs."
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    val_probs = predict_mlp(
        model,
        X_val,
        device,
        args.batch_size
    )

    threshold = find_best_threshold(
        y_val,
        val_probs
    )

    test_probs = predict_mlp(
        model,
        X_test,
        device,
        args.batch_size
    )

    val_metrics = compute_metrics(
        y_val,
        val_probs,
        threshold
    )

    test_metrics = compute_metrics(
        y_test,
        test_probs,
        threshold
    )

    print_metrics(
        "Validation",
        val_metrics
    )

    print_metrics(
        "Test",
        test_metrics
    )

    return {
        "name":
            "t0_all_layer_mlp",

        "input_dim":
            int(X_train.shape[1]),

        "trainable_parameters":
            int(total_params),

        "validation":
            val_metrics,

        "test":
            test_metrics,
    }


# =========================================================
# LOAD + TRUNCATE t=0 GRAPHS
# =========================================================

def graph_files(split):
    folder = GRAPH_DIR / split

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing folder: {folder}"
        )

    files = sorted(
        folder.glob("*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No graphs found in {folder}"
        )

    return files


def truncate_to_t0(g):
    keep = (
        g.time_position == 0
    )

    new_edge_index, _ = subgraph(
        subset=keep,
        edge_index=g.edge_index,
        relabel_nodes=True,
        num_nodes=g.num_nodes
    )

    new_g = Data(
        x=g.x[keep].clone(),
        edge_index=new_edge_index.clone(),
        y=g.y.clone(),
        layer_index=g.layer_index[keep].clone(),
        time_position=g.time_position[keep].clone(),
    )

    new_g.sample_id = str(
        getattr(
            g,
            "sample_id",
            "unknown"
        )
    )

    return new_g


def load_t0_graphs(split):
    output = []

    for path in tqdm(
        graph_files(split),
        desc=f"t=0 graphs {split}"
    ):
        g = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        if int(g.y.item()) not in [0, 1]:
            continue

        output.append(
            truncate_to_t0(g)
        )

    return output


# =========================================================
# SHUFFLE t=0 DEPTH EDGES
# Degree-preserving double-edge swaps
# =========================================================

def stable_seed(sample_id, base_seed):
    raw = (
        f"{sample_id}|{base_seed}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        raw
    ).digest()

    value = int.from_bytes(
        digest[:8],
        "little",
        signed=False
    )

    return value % (2**32)


def undirected_edge_set(edge_index):
    edges = set()

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for u, v in zip(src, dst):
        if u == v:
            continue

        edges.add(
            (
                min(u, v),
                max(u, v)
            )
        )

    return edges


def shuffle_edges_degree_preserving(
    edge_index,
    num_nodes,
    seed,
    swaps_per_edge=10
):
    original = list(
        undirected_edge_set(
            edge_index
        )
    )

    if len(original) < 2:
        return edge_index.clone()

    edges = list(original)
    edge_set = set(original)

    rng = random.Random(seed)

    target = max(
        1,
        swaps_per_edge * len(edges)
    )

    successful = 0
    attempts = 0
    max_attempts = target * 50

    while (
        successful < target
        and attempts < max_attempts
    ):
        attempts += 1

        i, j = rng.sample(
            range(len(edges)),
            2
        )

        a, b = edges[i]
        c, d = edges[j]

        if len({a, b, c, d}) < 4:
            continue

        if rng.random() < 0.5:
            n1 = (a, d)
            n2 = (c, b)
        else:
            n1 = (a, c)
            n2 = (b, d)

        n1 = (
            min(n1[0], n1[1]),
            max(n1[0], n1[1])
        )

        n2 = (
            min(n2[0], n2[1]),
            max(n2[0], n2[1])
        )

        if (
            n1[0] == n1[1]
            or n2[0] == n2[1]
            or n1 == n2
        ):
            continue

        old1 = edges[i]
        old2 = edges[j]

        edge_set.remove(old1)
        edge_set.remove(old2)

        valid = (
            n1 not in edge_set
            and n2 not in edge_set
        )

        if not valid:
            edge_set.add(old1)
            edge_set.add(old2)
            continue

        edges[i] = n1
        edges[j] = n2

        edge_set.add(n1)
        edge_set.add(n2)

        successful += 1

    src = []
    dst = []

    for u, v in edges:
        src.extend([u, v])
        dst.extend([v, u])

    shuffled = torch.tensor(
        [src, dst],
        dtype=torch.long
    )

    if (
        shuffled.shape[1]
        != edge_index.shape[1]
    ):
        raise RuntimeError(
            "Edge shuffle changed edge count."
        )

    return shuffled


def make_shuffled_t0_graphs(
    graphs,
    seed,
    swaps_per_edge
):
    out = []

    for g in graphs:
        new_g = Data(
            x=g.x.clone(),
            edge_index=shuffle_edges_degree_preserving(
                g.edge_index,
                g.num_nodes,
                stable_seed(
                    g.sample_id,
                    seed
                ),
                swaps_per_edge
            ),
            y=g.y.clone(),
            layer_index=g.layer_index.clone(),
            time_position=g.time_position.clone(),
        )

        new_g.sample_id = str(
            g.sample_id
        )

        out.append(new_g)

    return out


# =========================================================
# GRAPHSAGE
# =========================================================

class T0GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        dropout=0.30
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.norm1 = nn.BatchNorm1d(
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.norm2 = nn.BatchNorm1d(
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim // 2,
                2
            )
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):
        x = self.conv1(
            x,
            edge_index
        )

        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(
            x,
            batch
        )

        return self.classifier(x)


def graph_class_weights(graphs):
    labels = np.asarray(
        [
            int(g.y.item())
            for g in graphs
        ],
        dtype=np.int64
    )

    counts = np.bincount(
        labels,
        minlength=2
    )

    total = len(labels)

    weights = np.asarray(
        [
            total / (
                2.0 * max(counts[0], 1)
            ),
            total / (
                2.0 * max(counts[1], 1)
            ),
        ],
        dtype=np.float32
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def train_graph_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    model.train()

    total_loss = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch
        )

        y = batch.y.view(-1).long()

        loss = criterion(
            logits,
            y
        )

        loss.backward()
        optimizer.step()

        n = y.shape[0]

        total_loss += (
            loss.item() * n
        )

        total += n

    return (
        total_loss
        / max(total, 1)
    )


@torch.no_grad()
def predict_graph(
    model,
    loader,
    device
):
    model.eval()

    probs = []
    labels = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch
        )

        p = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        y = (
            batch.y
            .view(-1)
            .long()
        )

        probs.extend(
            p.cpu().numpy().tolist()
        )

        labels.extend(
            y.cpu().numpy().tolist()
        )

    return (
        np.asarray(
            labels,
            dtype=np.int64
        ),
        np.asarray(
            probs,
            dtype=np.float64
        ),
    )


def run_graphsage(
    name,
    train_graphs,
    val_graphs,
    test_graphs,
    args,
    device
):
    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    set_seed(args.seed)

    train_loader = GeoLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = GeoLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = GeoLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False
    )

    input_dim = int(
        train_graphs[0].x.shape[1]
    )

    model = T0GraphSAGE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: {total_params:,}"
    )

    criterion = nn.CrossEntropyLoss(
        weight=graph_class_weights(
            train_graphs
        ).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace("-", "_")
    )

    checkpoint_path = (
        MODEL_DIR
        / f"{safe_name}_seed{args.seed}.pt"
    )

    best_val = -1.0
    no_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1
    ):
        train_loss = train_graph_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_y, val_probs = predict_graph(
            model,
            val_loader,
            device
        )

        val_auroc = roc_auc_score(
            val_y,
            val_probs
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_AUROC={val_auroc:.4f}"
        )

        if val_auroc > best_val:
            best_val = float(
                val_auroc
            )

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_val_auroc":
                        best_val,

                    "input_dim":
                        input_dim,

                    "hidden_dim":
                        args.hidden_dim,

                    "seed":
                        args.seed,

                    "trainable_parameters":
                        total_params,
                },
                checkpoint_path
            )

        else:
            no_improvement += 1

        if (
            no_improvement
            >= args.patience
        ):
            print(
                f"\nEarly stopping after "
                f"{epoch} epochs."
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    val_y, val_probs = predict_graph(
        model,
        val_loader,
        device
    )

    threshold = find_best_threshold(
        val_y,
        val_probs
    )

    test_y, test_probs = predict_graph(
        model,
        test_loader,
        device
    )

    val_metrics = compute_metrics(
        val_y,
        val_probs,
        threshold
    )

    test_metrics = compute_metrics(
        test_y,
        test_probs,
        threshold
    )

    print_metrics(
        "Validation",
        val_metrics
    )

    print_metrics(
        "Test",
        test_metrics
    )

    return {
        "name":
            name,

        "input_dim":
            input_dim,

        "trainable_parameters":
            int(total_params),

        "validation":
            val_metrics,

        "test":
            test_metrics,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    parser.add_argument(
        "--swaps-per-edge",
        type=int,
        default=10
    )

    args = parser.parse_args()

    set_seed(args.seed)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("t=0 FAIR BASELINE SUITE")
    print("=" * 70)

    print(
        "\nAll methods use ONLY pre-generation t=0 information."
    )

    print(
        "No generated token states are used."
    )

    # =====================================================
    # FEATURE-BASED BASELINES
    # =====================================================

    train = load_t0_features(
        "train"
    )

    val = load_t0_features(
        "val"
    )

    test = load_t0_features(
        "test"
    )

    print(
        f"\nTrain/Val/Test: "
        f"{len(train['y'])}/"
        f"{len(val['y'])}/"
        f"{len(test['y'])}"
    )

    print(
        f"All-layer flattened dimension: "
        f"{train['all_layers_flat'].shape[1]}"
    )

    results = []

    # -----------------------------------------------------
    # 1. Final-layer Logistic Regression
    # -----------------------------------------------------

    results.append(
        run_logreg(
            name="t0_final_layer_logreg",
            X_train=train["final_layer"],
            y_train=train["y"],
            X_val=val["final_layer"],
            y_val=val["y"],
            X_test=test["final_layer"],
            y_test=test["y"],
            seed=args.seed
        )
    )

    # -----------------------------------------------------
    # 2. Mean-over-layers Logistic Regression
    # -----------------------------------------------------

    results.append(
        run_logreg(
            name="t0_mean_layer_logreg",
            X_train=train["mean_layers"],
            y_train=train["y"],
            X_val=val["mean_layers"],
            y_val=val["y"],
            X_test=test["mean_layers"],
            y_test=test["y"],
            seed=args.seed
        )
    )

    # -----------------------------------------------------
    # 3. All-layer flattened MLP
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    results.append(
        run_t0_mlp(
            X_train=train["all_layers_flat"],
            y_train=train["y"],
            X_val=val["all_layers_flat"],
            y_val=val["y"],
            X_test=test["all_layers_flat"],
            y_test=test["y"],
            args=args,
            device=device
        )
    )

    # =====================================================
    # GRAPH BASELINES
    # =====================================================

    train_graphs = load_t0_graphs(
        "train"
    )

    val_graphs = load_t0_graphs(
        "val"
    )

    test_graphs = load_t0_graphs(
        "test"
    )

    # -----------------------------------------------------
    # 4. Real depth-chain EpiGraph
    # -----------------------------------------------------

    results.append(
        run_graphsage(
            name="t0_real_depth_epigraph",
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            args=args,
            device=device
        )
    )

    # -----------------------------------------------------
    # 5. Shuffled-depth GraphSAGE
    # -----------------------------------------------------

    shuffled_train = (
        make_shuffled_t0_graphs(
            train_graphs,
            args.seed,
            args.swaps_per_edge
        )
    )

    shuffled_val = (
        make_shuffled_t0_graphs(
            val_graphs,
            args.seed,
            args.swaps_per_edge
        )
    )

    shuffled_test = (
        make_shuffled_t0_graphs(
            test_graphs,
            args.seed,
            args.swaps_per_edge
        )
    )

    results.append(
        run_graphsage(
            name="t0_shuffled_depth_graphsage",
            train_graphs=shuffled_train,
            val_graphs=shuffled_val,
            test_graphs=shuffled_test,
            args=args,
            device=device
        )
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary_csv = (
        RESULT_DIR
        / f"step12_t0_all_baselines_seed{args.seed}.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "method",
                "input_dim",
                "trainable_parameters",
                "val_auroc",
                "test_auroc",
                "test_auprc_hallucinated",
                "test_auprc_correct",
                "test_macro_f1",
                "test_balanced_accuracy",
                "test_accuracy",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r["name"],
                    r.get(
                        "input_dim",
                        ""
                    ),
                    r.get(
                        "trainable_parameters",
                        ""
                    ),
                    r["validation"]["auroc"],
                    r["test"]["auroc"],
                    r["test"]["auprc_hallucinated"],
                    r["test"]["auprc_correct"],
                    r["test"]["macro_f1"],
                    r["test"]["balanced_accuracy"],
                    r["test"]["accuracy"],
                ]
            )

    summary_json = (
        RESULT_DIR
        / f"step12_t0_all_baselines_seed{args.seed}.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "seed":
                    args.seed,

                "scope":
                    "t=0 only; no generated-token states",

                "results":
                    results,
            },
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("t=0 BASELINE SUMMARY")
    print("=" * 70)

    print(
        "\nMethod                              Test AUROC"
    )

    for r in results:
        print(
            f"{r['name']:<35} "
            f"{r['test']['auroc']:.4f}"
        )

    print(
        f"\nSummary CSV:\n{summary_csv}"
    )

    print(
        f"\nSummary JSON:\n{summary_json}"
    )


if __name__ == "__main__":
    main()
