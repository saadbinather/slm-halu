import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    global_mean_pool,
)
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

GRAPH_DIR = Path("graphs")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Good default reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# LOAD GRAPHS
# =========================================================

def get_graph_files(split):
    folder = GRAPH_DIR / split

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(folder.glob("*.pt"))

    if not files:
        raise RuntimeError(f"No .pt graphs found in {folder}")

    return files


def load_graphs(split):
    files = get_graph_files(split)

    graphs = []

    for path in tqdm(
        files,
        desc=f"Loading {split}"
    ):
        graph = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        label = int(graph.y.item())

        if label not in [0, 1]:
            continue

        # Keep source filename/sample ID available.
        if not hasattr(graph, "sample_id"):
            graph.sample_id = path.stem

        graphs.append(graph)

    if not graphs:
        raise RuntimeError(f"No usable graphs in split: {split}")

    return graphs


# =========================================================
# METRICS
# label 1 = hallucinated
# label 0 = correct
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


def compute_metrics(
    y_true,
    probs,
    threshold=0.5
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )

    probs = np.asarray(
        probs,
        dtype=np.float64
    )

    preds = (
        probs >= threshold
    ).astype(np.int64)

    return {
        "threshold":
            float(threshold),

        "auroc":
            float(
                roc_auc_score(
                    y_true,
                    probs
                )
            ),

        "auprc_hallucinated":
            float(
                average_precision_score(
                    y_true,
                    probs
                )
            ),

        "auprc_correct":
            float(
                average_precision_score(
                    1 - y_true,
                    1.0 - probs
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    preds,
                    average="macro"
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    preds
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    preds
                )
            ),
    }


def print_metrics(title, metrics):
    print("\n" + title)
    print("-" * len(title))

    print(
        f"AUROC:              "
        f"{metrics['auroc']:.4f}"
    )

    print(
        f"AUPRC hallucinated: "
        f"{metrics['auprc_hallucinated']:.4f}"
    )

    print(
        f"AUPRC correct:      "
        f"{metrics['auprc_correct']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Balanced accuracy:  "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Accuracy:            "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Threshold:           "
        f"{metrics['threshold']:.3f}"
    )


# =========================================================
# EPIGRAPH MODEL
#
# Input graph:
#   node = (time, layer)
#   node features = PCA-16 + normalized layer + normalized time
#
# GraphSAGE/GCN message passing uses the real EpiGraph edge_index:
#   - depth edges
#   - temporal edges
#
# Then global mean pooling gives one graph embedding.
# =========================================================

class EpiGraphGNN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        dropout=0.30,
        model_type="sage"
    ):
        super().__init__()

        self.model_type = model_type

        if model_type == "sage":
            Conv = SAGEConv

        elif model_type == "gcn":
            Conv = GCNConv

        else:
            raise ValueError(
                f"Unknown model_type: {model_type}"
            )

        self.conv1 = Conv(
            input_dim,
            hidden_dim
        )

        self.norm1 = nn.BatchNorm1d(
            hidden_dim
        )

        self.conv2 = Conv(
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
        # Layer 1
        x = self.conv1(
            x,
            edge_index
        )

        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(
            x,
            edge_index
        )

        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Variable-size graph -> one embedding
        graph_embedding = global_mean_pool(
            x,
            batch
        )

        logits = self.classifier(
            graph_embedding
        )

        return logits


# =========================================================
# CLASS WEIGHTS
# =========================================================

def get_class_weights(graphs):
    labels = np.array(
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

    # Equivalent to sklearn class_weight="balanced"
    weights = np.array(
        [
            total / (2.0 * max(counts[0], 1)),
            total / (2.0 * max(counts[1], 1))
        ],
        dtype=np.float32
    )

    return (
        counts,
        torch.tensor(
            weights,
            dtype=torch.float32
        )
    )


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    model.train()

    total_loss = 0.0
    total_graphs = 0

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

        total_graphs += n

    return (
        total_loss
        / max(total_graphs, 1)
    )


# =========================================================
# EVALUATION
# =========================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device
):
    model.eval()

    all_probs = []
    all_labels = []
    all_sample_ids = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        labels = (
            batch.y
            .view(-1)
            .long()
        )

        all_probs.extend(
            probs.cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        # PyG normally batches string metadata into a list.
        ids = batch.sample_id

        if isinstance(ids, (list, tuple)):
            all_sample_ids.extend(
                [str(x) for x in ids]
            )
        else:
            all_sample_ids.append(
                str(ids)
            )

    return (
        np.asarray(
            all_labels,
            dtype=np.int64
        ),
        np.asarray(
            all_probs,
            dtype=np.float64
        ),
        all_sample_ids
    )


# =========================================================
# SAVE PREDICTIONS
# =========================================================

def save_predictions(
    path,
    sample_ids,
    labels,
    probs,
    threshold
):
    preds = (
        probs >= threshold
    ).astype(np.int64)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "sample_id",
                "label",
                "prob_hallucinated",
                "prediction",
            ]
        )

        for sid, y, p, pred in zip(
            sample_ids,
            labels,
            probs,
            preds
        ):
            writer.writerow(
                [
                    sid,
                    int(y),
                    float(p),
                    int(pred),
                ]
            )


# =========================================================
# MAIN TRAINING
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "sage",
            "gcn"
        ],
        default="sage"
    )

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
    print(
        "EPIGRAPH — STEP 08: MAIN GNN"
    )
    print("=" * 70)

    print(
        f"\nModel: {args.model.upper()}"
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    train_graphs = load_graphs(
        "train"
    )

    val_graphs = load_graphs(
        "val"
    )

    test_graphs = load_graphs(
        "test"
    )

    print(
        f"\nTrain graphs: {len(train_graphs)}"
    )

    print(
        f"Val graphs:   {len(val_graphs)}"
    )

    print(
        f"Test graphs:  {len(test_graphs)}"
    )

    input_dim = int(
        train_graphs[0].x.shape[1]
    )

    print(
        f"Node feature dimension: "
        f"{input_dim}"
    )

    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    counts, class_weights = (
        get_class_weights(
            train_graphs
        )
    )

    print(
        f"\nTrain labels:"
        f" correct={counts[0]},"
        f" hallucinated={counts[1]}"
    )

    print(
        "Class weights: "
        f"[correct={class_weights[0]:.4f}, "
        f"hallucinated={class_weights[1]:.4f}]"
    )

    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False
    )

    # =====================================================
    # DEVICE / MODEL
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    model = EpiGraphGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        model_type=args.model
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # =====================================================
    # TRAINING + EARLY STOPPING ON VAL AUROC
    # =====================================================

    best_val_auroc = -1.0
    epochs_without_improvement = 0

    checkpoint_path = (
        MODEL_DIR
        / f"epigraph_{args.model}.pt"
    )

    history = []

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_y, val_probs, _ = predict(
            model,
            val_loader,
            device
        )

        val_auroc = roc_auc_score(
            val_y,
            val_probs
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    float(train_loss),

                "val_auroc":
                    float(val_auroc),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_AUROC={val_auroc:.4f}"
        )

        if val_auroc > best_val_auroc:

            best_val_auroc = float(
                val_auroc
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "model_type":
                        args.model,

                    "input_dim":
                        input_dim,

                    "hidden_dim":
                        args.hidden_dim,

                    "dropout":
                        args.dropout,

                    "best_val_auroc":
                        best_val_auroc,

                    "seed":
                        args.seed,
                },
                checkpoint_path
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                f"\nEarly stopping after "
                f"{epoch} epochs."
            )

            break

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

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

    print(
        f"\nLoaded best checkpoint "
        f"(val AUROC="
        f"{checkpoint['best_val_auroc']:.4f})"
    )

    # =====================================================
    # VALIDATION:
    # choose threshold ONLY here
    # =====================================================

    val_y, val_probs, val_ids = predict(
        model,
        val_loader,
        device
    )

    threshold = find_best_threshold(
        val_y,
        val_probs
    )

    val_metrics = compute_metrics(
        val_y,
        val_probs,
        threshold
    )

    # =====================================================
    # TEST:
    # fixed threshold from validation
    # =====================================================

    test_y, test_probs, test_ids = predict(
        model,
        test_loader,
        device
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

    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    val_pred_path = (
        RESULT_DIR
        / f"step08_epigraph_{args.model}_val_predictions.csv"
    )

    test_pred_path = (
        RESULT_DIR
        / f"step08_epigraph_{args.model}_test_predictions.csv"
    )

    save_predictions(
        val_pred_path,
        val_ids,
        val_y,
        val_probs,
        threshold
    )

    save_predictions(
        test_pred_path,
        test_ids,
        test_y,
        test_probs,
        threshold
    )

    # =====================================================
    # SAVE RESULTS JSON
    # =====================================================

    result = {
        "model":
            args.model,

        "seed":
            args.seed,

        "input_dim":
            input_dim,

        "hidden_dim":
            args.hidden_dim,

        "dropout":
            args.dropout,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "class_counts": {
            "correct":
                int(counts[0]),

            "hallucinated":
                int(counts[1]),
        },

        "class_weights": {
            "correct":
                float(class_weights[0]),

            "hallucinated":
                float(class_weights[1]),
        },

        "best_val_auroc":
            float(
                checkpoint[
                    "best_val_auroc"
                ]
            ),

        "validation":
            val_metrics,

        "test":
            test_metrics,

        "history":
            history,

        "checkpoint":
            str(checkpoint_path),

        "val_predictions":
            str(val_pred_path),

        "test_predictions":
            str(test_pred_path),
    }

    result_path = (
        RESULT_DIR
        / f"step08_epigraph_{args.model}.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("STEP 08 COMPLETE")
    print("=" * 70)

    print(
        f"\nCheckpoint:\n"
        f"{checkpoint_path}"
    )

    print(
        f"\nResults:\n"
        f"{result_path}"
    )

    print(
        "\nCompare TEST AUROC against:"
    )

    print(
        "Single-state Logistic Regression: 0.7890"
    )

    print(
        "Flattened trajectory MLP:          0.7712"
    )

    print(
        "\nIf EpiGraph improves over graph-free "
        "baselines, next run shuffled-edge ablation."
    )


if __name__ == "__main__":
    main()
