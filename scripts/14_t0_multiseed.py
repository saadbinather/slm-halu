import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import subgraph
from tqdm import tqdm


FEATURE_DIR = Path("processed_features")
GRAPH_DIR = Path("graphs")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


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
# t=0 ALL-LAYER MLP DATA
# =========================================================

def get_feature_files(split):
    folder = FEATURE_DIR / split

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing folder: {folder}"
        )

    files = sorted(
        folder.glob("*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No features found in {folder}"
        )

    return files


def load_t0_flat_features(split):
    X = []
    y = []

    for path in tqdm(
        get_feature_files(split),
        desc=f"Features {split}"
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

        t0_idx = times.index(0)

        # [28, 16] -> 448
        vector = (
            trace["features"][t0_idx]
            .float()
            .reshape(-1)
            .numpy()
        )

        X.append(vector)
        y.append(label)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64)
    )


# =========================================================
# t=0 GRAPHS
# =========================================================

def get_graph_files(split):
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

    edge_index, _ = subgraph(
        subset=keep,
        edge_index=g.edge_index,
        relabel_nodes=True,
        num_nodes=g.num_nodes
    )

    out = Data(
        x=g.x[keep].clone(),
        edge_index=edge_index.clone(),
        y=g.y.clone(),
        layer_index=g.layer_index[keep].clone(),
        time_position=g.time_position[keep].clone(),
    )

    out.sample_id = str(
        getattr(
            g,
            "sample_id",
            "unknown"
        )
    )

    return out


def load_t0_graphs(split):
    graphs = []

    for path in tqdm(
        get_graph_files(split),
        desc=f"Graphs {split}"
    ):
        g = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        if int(g.y.item()) not in [0, 1]:
            continue

        graphs.append(
            truncate_to_t0(g)
        )

    return graphs


# =========================================================
# DEGREE-PRESERVING SHUFFLE
# =========================================================

def stable_graph_seed(
    sample_id,
    base_seed
):
    raw = (
        f"{sample_id}|{base_seed}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        raw
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            "little",
            signed=False
        )
        % (2**32)
    )


def undirected_edges(edge_index):
    edges = set()

    for u, v in zip(
        edge_index[0].tolist(),
        edge_index[1].tolist()
    ):
        if u == v:
            continue

        edges.add(
            (
                min(u, v),
                max(u, v)
            )
        )

    return edges


def degree_preserving_shuffle(
    edge_index,
    seed,
    swaps_per_edge=10
):
    original = list(
        undirected_edges(
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

    success = 0
    attempts = 0
    max_attempts = target * 50

    while (
        success < target
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

        success += 1

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
            "Shuffle changed edge count."
        )

    return shuffled


def shuffled_graph_copy(
    g,
    seed,
    swaps_per_edge
):
    out = Data(
        x=g.x.clone(),
        edge_index=degree_preserving_shuffle(
            g.edge_index,
            stable_graph_seed(
                g.sample_id,
                seed
            ),
            swaps_per_edge
        ),
        y=g.y.clone(),
        layer_index=g.layer_index.clone(),
        time_position=g.time_position.clone(),
    )

    out.sample_id = str(
        g.sample_id
    )

    return out


# =========================================================
# MLP
# =========================================================

class T0MLP(nn.Module):

    def __init__(
        self,
        input_dim=448,
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

    loader = TorchLoader(
        TensorDataset(
            torch.tensor(
                X,
                dtype=torch.float32
            )
        ),
        batch_size=batch_size,
        shuffle=False
    )

    probs = []

    for (x,) in loader:
        x = x.to(device)

        probs.append(
            torch.sigmoid(
                model(x)
            ).cpu().numpy()
        )

    return np.concatenate(probs)


def run_mlp_seed(
    seed,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    args,
    device
):
    set_seed(seed)

    scaler = StandardScaler()

    train_x = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    val_x = scaler.transform(
        X_val
    ).astype(np.float32)

    test_x = scaler.transform(
        X_test
    ).astype(np.float32)

    model = T0MLP(
        input_dim=train_x.shape[1],
        dropout=args.dropout
    ).to(device)

    loader = TorchLoader(
        TensorDataset(
            torch.tensor(
                train_x,
                dtype=torch.float32
            ),
            torch.tensor(
                y_train,
                dtype=torch.float32
            )
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

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            num_neg / max(num_pos, 1.0),
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val = -1.0
    best_state = None
    no_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1
    ):
        model.train()

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            loss = criterion(
                model(batch_x),
                batch_y
            )

            loss.backward()
            optimizer.step()

        val_probs = predict_mlp(
            model,
            val_x,
            device,
            args.batch_size
        )

        val_auroc = roc_auc_score(
            y_val,
            val_probs
        )

        if val_auroc > best_val:
            best_val = float(
                val_auroc
            )

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

            no_improvement = 0

        else:
            no_improvement += 1

        if no_improvement >= args.patience:
            break

    model.load_state_dict(
        best_state
    )

    test_probs = predict_mlp(
        model,
        test_x,
        device,
        args.batch_size
    )

    return {
        "seed":
            int(seed),

        "val_auroc":
            float(best_val),

        "test_auroc":
            float(
                roc_auc_score(
                    y_test,
                    test_probs
                )
            ),
    }


# =========================================================
# GRAPHSAGE
# =========================================================

class T0GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim=18,
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


def graph_weights(graphs):
    labels = np.asarray(
        [
            int(g.y.item())
            for g in graphs
        ]
    )

    counts = np.bincount(
        labels,
        minlength=2
    )

    total = len(labels)

    return torch.tensor(
        [
            total / (
                2.0 * max(counts[0], 1)
            ),
            total / (
                2.0 * max(counts[1], 1)
            ),
        ],
        dtype=torch.float32
    )


@torch.no_grad()
def predict_graph(
    model,
    loader,
    device
):
    model.eval()

    labels = []
    probs = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch
        )

        probs.extend(
            torch.softmax(
                logits,
                dim=1
            )[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )

        labels.extend(
            batch.y
            .view(-1)
            .cpu()
            .numpy()
            .tolist()
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


def run_graph_seed(
    seed,
    train_graphs,
    val_graphs,
    test_graphs,
    args,
    device
):
    set_seed(seed)

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

    model = T0GraphSAGE(
        input_dim=int(
            train_graphs[0].x.shape[1]
        ),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=graph_weights(
            train_graphs
        ).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val = -1.0
    best_state = None
    no_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1
    ):
        model.train()

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            logits = model(
                batch.x.float(),
                batch.edge_index,
                batch.batch
            )

            y = (
                batch.y
                .view(-1)
                .long()
            )

            loss = criterion(
                logits,
                y
            )

            loss.backward()
            optimizer.step()

        val_y, val_probs = predict_graph(
            model,
            val_loader,
            device
        )

        val_auroc = roc_auc_score(
            val_y,
            val_probs
        )

        if val_auroc > best_val:
            best_val = float(
                val_auroc
            )

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

            no_improvement = 0

        else:
            no_improvement += 1

        if no_improvement >= args.patience:
            break

    model.load_state_dict(
        best_state
    )

    test_y, test_probs = predict_graph(
        model,
        test_loader,
        device
    )

    return {
        "seed":
            int(seed),

        "val_auroc":
            float(best_val),

        "test_auroc":
            float(
                roc_auc_score(
                    test_y,
                    test_probs
                )
            ),
    }


# =========================================================
# SUMMARY
# =========================================================

def summarize(rows):
    values = np.asarray(
        [
            r["test_auroc"]
            for r in rows
        ],
        dtype=np.float64
    )

    return {
        "mean_test_auroc":
            float(values.mean()),

        "std_test_auroc":
            float(
                values.std(ddof=1)
                if len(values) > 1
                else 0.0
            ),

        "min_test_auroc":
            float(values.min()),

        "max_test_auroc":
            float(values.max()),
    }


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15
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
        "--swaps-per-edge",
        type=int,
        default=10
    )

    args = parser.parse_args()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("t=0 MULTI-SEED ROBUSTNESS")
    print("=" * 70)

    print(
        f"\nSeeds: {args.seeds}"
    )

    print(
        f"Device: {device}"
    )

    # -----------------------------------------------------
    # Load once
    # -----------------------------------------------------

    X_train, y_train = load_t0_flat_features(
        "train"
    )

    X_val, y_val = load_t0_flat_features(
        "val"
    )

    X_test, y_test = load_t0_flat_features(
        "test"
    )

    real_train = load_t0_graphs(
        "train"
    )

    real_val = load_t0_graphs(
        "val"
    )

    real_test = load_t0_graphs(
        "test"
    )

    mlp_rows = []
    real_rows = []
    shuffled_rows = []

    # -----------------------------------------------------
    # Seeds
    # -----------------------------------------------------

    for seed in args.seeds:
        print("\n" + "=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        mlp_result = run_mlp_seed(
            seed,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            args,
            device
        )

        mlp_rows.append(
            mlp_result
        )

        print(
            f"MLP test AUROC: "
            f"{mlp_result['test_auroc']:.4f}"
        )

        real_result = run_graph_seed(
            seed,
            real_train,
            real_val,
            real_test,
            args,
            device
        )

        real_rows.append(
            real_result
        )

        print(
            f"Real EpiGraph test AUROC: "
            f"{real_result['test_auroc']:.4f}"
        )

        # Shuffled topology changes with seed as well.
        shuffled_train = [
            shuffled_graph_copy(
                g,
                seed,
                args.swaps_per_edge
            )
            for g in real_train
        ]

        shuffled_val = [
            shuffled_graph_copy(
                g,
                seed,
                args.swaps_per_edge
            )
            for g in real_val
        ]

        shuffled_test = [
            shuffled_graph_copy(
                g,
                seed,
                args.swaps_per_edge
            )
            for g in real_test
        ]

        shuffled_result = run_graph_seed(
            seed,
            shuffled_train,
            shuffled_val,
            shuffled_test,
            args,
            device
        )

        shuffled_rows.append(
            shuffled_result
        )

        print(
            f"Shuffled GraphSAGE test AUROC: "
            f"{shuffled_result['test_auroc']:.4f}"
        )

    # -----------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------

    summary = {
        "seeds":
            args.seeds,

        "t0_all_layer_mlp": {
            "runs":
                mlp_rows,

            "summary":
                summarize(
                    mlp_rows
                ),
        },

        "t0_real_depth_epigraph": {
            "runs":
                real_rows,

            "summary":
                summarize(
                    real_rows
                ),
        },

        "t0_shuffled_depth_graphsage": {
            "runs":
                shuffled_rows,

            "summary":
                summarize(
                    shuffled_rows
                ),
        },
    }

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)

    for key in [
        "t0_all_layer_mlp",
        "t0_real_depth_epigraph",
        "t0_shuffled_depth_graphsage"
    ]:
        stats = summary[
            key
        ]["summary"]

        print(
            f"{key:<35} "
            f"{stats['mean_test_auroc']:.4f} "
            f"+/- {stats['std_test_auroc']:.4f}"
        )

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    csv_path = (
        RESULT_DIR
        / "step14_t0_multiseed.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "seed",
                "mlp_test_auroc",
                "real_epigraph_test_auroc",
                "shuffled_graphsage_test_auroc",
            ]
        )

        for i, seed in enumerate(
            args.seeds
        ):
            writer.writerow(
                [
                    seed,
                    mlp_rows[i]["test_auroc"],
                    real_rows[i]["test_auroc"],
                    shuffled_rows[i]["test_auroc"],
                ]
            )

    json_path = (
        RESULT_DIR
        / "step14_t0_multiseed.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"\nCSV:\n{csv_path}"
    )

    print(
        f"\nJSON:\n{json_path}"
    )


if __name__ == "__main__":
    main()
