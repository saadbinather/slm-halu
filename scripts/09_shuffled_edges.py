import argparse
import csv
import hashlib
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
from torch_geometric.nn import SAGEConv, global_mean_pool
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# STABLE PER-GRAPH SEED
# =========================================================

def stable_graph_seed(sample_id: str, base_seed: int) -> int:
    """
    Python's built-in hash() changes across processes.
    SHA256 gives us a stable deterministic seed.
    """
    text = f"{sample_id}|{base_seed}".encode("utf-8")
    digest = hashlib.sha256(text).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False
    )

    return value % (2**32)


# =========================================================
# GRAPH FILES
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
            f"No .pt graphs found in {folder}"
        )

    return files


# =========================================================
# DEGREE-PRESERVING UNDIRECTED EDGE SHUFFLE
# =========================================================

def edge_index_to_undirected_set(edge_index):
    """
    Original EpiGraph edges are stored bidirectionally.

    Convert:
        u -> v
        v -> u

    into one undirected pair:
        (min(u,v), max(u,v))
    """
    edges = set()

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for u, v in zip(src, dst):

        if u == v:
            continue

        a = min(u, v)
        b = max(u, v)

        edges.add((a, b))

    return edges


def degree_preserving_shuffle(
    edge_index,
    num_nodes,
    seed,
    swaps_per_edge=10
):
    """
    Randomize topology using double-edge swaps.

    Preserves:
      - number of nodes
      - number of undirected edges
      - degree of every node
      - bidirectional/undirected structure

    Destroys:
      - original layer/time neighborhood topology

    This is a stronger/fairer ablation than simply replacing
    all edges with arbitrary random pairs.
    """

    original_edges = edge_index_to_undirected_set(
        edge_index
    )

    edges = list(original_edges)
    edge_set = set(original_edges)

    if len(edges) < 2:
        return edge_index.clone()

    rng = random.Random(seed)

    target_swaps = max(
        1,
        swaps_per_edge * len(edges)
    )

    successful_swaps = 0
    attempts = 0

    # Give enough attempts even for constrained small graphs.
    max_attempts = target_swaps * 30

    while (
        successful_swaps < target_swaps
        and attempts < max_attempts
    ):
        attempts += 1

        i, j = rng.sample(
            range(len(edges)),
            2
        )

        a, b = edges[i]
        c, d = edges[j]

        # Four distinct nodes makes a clean 2-edge swap.
        if len({a, b, c, d}) < 4:
            continue

        # Randomly choose one of the two possible rewiring patterns.
        if rng.random() < 0.5:
            new1 = (a, d)
            new2 = (c, b)
        else:
            new1 = (a, c)
            new2 = (b, d)

        # Canonical undirected ordering.
        new1 = (
            min(new1[0], new1[1]),
            max(new1[0], new1[1])
        )

        new2 = (
            min(new2[0], new2[1]),
            max(new2[0], new2[1])
        )

        # No self-loops.
        if (
            new1[0] == new1[1]
            or new2[0] == new2[1]
        ):
            continue

        # No duplicate edge between the two proposals.
        if new1 == new2:
            continue

        old1 = edges[i]
        old2 = edges[j]

        # Temporarily remove old edges when checking duplicates.
        edge_set.remove(old1)
        edge_set.remove(old2)

        valid = (
            new1 not in edge_set
            and new2 not in edge_set
        )

        if not valid:
            edge_set.add(old1)
            edge_set.add(old2)
            continue

        # Accept swap.
        edges[i] = new1
        edges[j] = new2

        edge_set.add(new1)
        edge_set.add(new2)

        successful_swaps += 1

    # Rebuild bidirectional edge_index.
    src = []
    dst = []

    for u, v in edges:
        src.extend([u, v])
        dst.extend([v, u])

    shuffled_edge_index = torch.tensor(
        [src, dst],
        dtype=torch.long
    )

    # Defensive checks.
    if shuffled_edge_index.shape[1] != edge_index.shape[1]:
        raise RuntimeError(
            "Shuffled graph changed the number of directed edges."
        )

    if shuffled_edge_index.numel() > 0:
        if int(shuffled_edge_index.max()) >= num_nodes:
            raise RuntimeError(
                "Shuffled graph contains invalid node IDs."
            )

    return shuffled_edge_index


# =========================================================
# LOAD + SHUFFLE GRAPHS
# =========================================================

def load_shuffled_graphs(
    split,
    seed,
    swaps_per_edge
):
    files = get_graph_files(split)

    graphs = []

    for path in tqdm(
        files,
        desc=f"Shuffle {split}"
    ):
        graph = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        label = int(
            graph.y.item()
        )

        if label not in [0, 1]:
            continue

        sample_id = str(
            getattr(
                graph,
                "sample_id",
                path.stem
            )
        )

        graph_seed = stable_graph_seed(
            sample_id,
            seed
        )

        original_edge_index = (
            graph.edge_index.clone()
        )

        shuffled_edge_index = (
            degree_preserving_shuffle(
                edge_index=original_edge_index,
                num_nodes=int(graph.num_nodes),
                seed=graph_seed,
                swaps_per_edge=swaps_per_edge
            )
        )

        graph.edge_index = (
            shuffled_edge_index
        )

        # edge_type is not used by the current GraphSAGE model.
        # We remove it so nobody accidentally interprets stale
        # "depth/time" labels after topology has been shuffled.
        if hasattr(graph, "edge_type"):
            del graph.edge_type

        graph.sample_id = sample_id
        graph.is_shuffled = True

        graphs.append(graph)

    if not graphs:
        raise RuntimeError(
            f"No usable graphs in split {split}"
        )

    return graphs


# =========================================================
# METRICS
# label 1 = hallucinated
# label 0 = correct
# =========================================================

def find_best_threshold(y_true, probs):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(
        0.05,
        0.95,
        181
    ):
        preds = (
            probs >= threshold
        ).astype(np.int64)

        score = balanced_accuracy_score(
            y_true,
            preds
        )

        if score > best_score:
            best_score = score
            best_threshold = float(
                threshold
            )

    return best_threshold


def compute_metrics(
    y_true,
    probs,
    threshold
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


def print_metrics(
    title,
    metrics
):
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
# SAME MODEL AS STEP 08
# =========================================================

class EpiGraphGraphSAGE(nn.Module):

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


# =========================================================
# CLASS WEIGHTS
# =========================================================

def get_class_weights(graphs):
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
            total
            / (2.0 * max(counts[0], 1)),

            total
            / (2.0 * max(counts[1], 1)),
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
# TRAIN
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
# PREDICT
# =========================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device
):
    model.eval()

    probs_all = []
    labels_all = []
    ids_all = []

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

        probs_all.extend(
            probs.cpu().numpy().tolist()
        )

        labels_all.extend(
            labels.cpu().numpy().tolist()
        )

        ids = batch.sample_id

        if isinstance(
            ids,
            (list, tuple)
        ):
            ids_all.extend(
                [str(x) for x in ids]
            )
        else:
            ids_all.append(str(ids))

    return (
        np.asarray(
            labels_all,
            dtype=np.int64
        ),
        np.asarray(
            probs_all,
            dtype=np.float64
        ),
        ids_all
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
        default=10,
        help=(
            "Number of attempted successful "
            "double-edge swaps per undirected edge."
        )
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
        "EPIGRAPH — STEP 09: SHUFFLED-EDGE ABLATION"
    )
    print("=" * 70)

    print(
        "\nSame node features + same GraphSAGE, "
        "but graph topology is degree-preserving shuffled."
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Swaps per undirected edge: "
        f"{args.swaps_per_edge}"
    )

    # =====================================================
    # LOAD SHUFFLED GRAPHS
    # =====================================================

    train_graphs = load_shuffled_graphs(
        "train",
        args.seed,
        args.swaps_per_edge
    )

    val_graphs = load_shuffled_graphs(
        "val",
        args.seed,
        args.swaps_per_edge
    )

    test_graphs = load_shuffled_graphs(
        "test",
        args.seed,
        args.swaps_per_edge
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
    # CLASS WEIGHTS
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
    # LOADERS
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
    # MODEL
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    model = EpiGraphGraphSAGE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    checkpoint_path = (
        MODEL_DIR
        / f"epigraph_shuffled_sage_seed{args.seed}.pt"
    )

    best_val_auroc = -1.0
    epochs_without_improvement = 0

    history = []

    # =====================================================
    # TRAINING
    # =====================================================

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
                    int(epoch),

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

                    "swaps_per_edge":
                        args.swaps_per_edge,
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
    # LOAD BEST CHECKPOINT
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
    # VALIDATION THRESHOLD
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
    # TEST
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
    # SAVE
    # =====================================================

    val_pred_path = (
        RESULT_DIR
        / f"step09_shuffled_sage_seed{args.seed}_val_predictions.csv"
    )

    test_pred_path = (
        RESULT_DIR
        / f"step09_shuffled_sage_seed{args.seed}_test_predictions.csv"
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

    result = {
        "experiment":
            "degree_preserving_shuffled_edges",

        "seed":
            args.seed,

        "swaps_per_edge":
            args.swaps_per_edge,

        "model":
            "GraphSAGE",

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
        / f"step09_shuffled_sage_seed{args.seed}.json"
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
    print("STEP 09 COMPLETE")
    print("=" * 70)

    print(
        f"\nResults:\n"
        f"{result_path}"
    )

    print(
        "\nReference TEST AUROC:"
    )

    print(
        "Real EpiGraph GraphSAGE: 0.7916"
    )

    print(
        "Single-state LogReg:     0.7890"
    )

    print(
        "Flattened trajectory MLP:0.7712"
    )

    print(
        "\nCritical comparison:"
    )

    print(
        "Real EpiGraph vs shuffled-edge GraphSAGE"
    )


if __name__ == "__main__":
    main()
