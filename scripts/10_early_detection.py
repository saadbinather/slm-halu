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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import subgraph
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

GRAPH_DIR = Path("graphs")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

DEFAULT_CUTOFFS = [0, 1, 2, 4, 8]
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
# LOAD ORIGINAL GRAPHS
# =========================================================

def get_graph_files(split):
    folder = GRAPH_DIR / split

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(folder.glob("*.pt"))

    if not files:
        raise RuntimeError(f"No graph files found in {folder}")

    return files


def load_graphs(split):
    files = get_graph_files(split)

    graphs = []

    for path in tqdm(
        files,
        desc=f"Loading {split}"
    ):
        g = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        label = int(g.y.item())

        if label not in [0, 1]:
            continue

        if not hasattr(g, "sample_id"):
            g.sample_id = path.stem

        graphs.append(g)

    if not graphs:
        raise RuntimeError(
            f"No usable graphs for split={split}"
        )

    return graphs


# =========================================================
# TRUNCATE GRAPH TO EARLY-DETECTION CUTOFF
#
# cutoff=0:
#   only t=0 nodes
#
# cutoff=1:
#   t=0 and t=1
#
# cutoff=2:
#   t=0,1,2
#
# etc.
#
# If a sample does not contain a later checkpoint, we simply
# use every checkpoint that actually exists <= cutoff.
# =========================================================

def truncate_graph(g, cutoff):
    keep = (
        g.time_position <= cutoff
    )

    if int(keep.sum()) == 0:
        raise RuntimeError(
            f"{g.sample_id}: no nodes available "
            f"for cutoff={cutoff}"
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

    new_g.sample_id = str(g.sample_id)
    new_g.cutoff = int(cutoff)

    return new_g


def make_truncated_split(
    graphs,
    cutoff,
    desc
):
    output = []

    for g in tqdm(
        graphs,
        desc=desc,
        leave=False
    ):
        output.append(
            truncate_graph(
                g,
                cutoff
            )
        )

    return output


# =========================================================
# METRICS
# label 1 = hallucinated
# label 0 = correct
# =========================================================

def find_best_threshold(
    y_true,
    probs
):
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


# =========================================================
# SAME GRAPHSAGE AS STEP 08
# =========================================================

class EarlyEpiGraph(nn.Module):

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

        graph_embedding = (
            global_mean_pool(
                x,
                batch
            )
        )

        return self.classifier(
            graph_embedding
        )


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
            total / (
                2.0 * max(counts[0], 1)
            ),
            total / (
                2.0 * max(counts[1], 1)
            ),
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
# TRAIN / PREDICT
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


@torch.no_grad()
def predict(
    model,
    loader,
    device
):
    model.eval()

    all_probs = []
    all_labels = []
    all_ids = []

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

        ids = batch.sample_id

        if isinstance(
            ids,
            (list, tuple)
        ):
            all_ids.extend(
                [str(x) for x in ids]
            )
        else:
            all_ids.append(
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
        all_ids
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
# ONE CUTOFF EXPERIMENT
# =========================================================

def run_cutoff(
    cutoff,
    original_train,
    original_val,
    original_test,
    args,
    device
):
    print("\n" + "=" * 70)
    print(
        f"EARLY DETECTION — t <= {cutoff}"
    )
    print("=" * 70)

    # Reset seed before each cutoff so initialization
    # and loader randomness are matched as closely as possible.
    set_seed(args.seed)

    train_graphs = make_truncated_split(
        original_train,
        cutoff,
        f"Train t<={cutoff}"
    )

    val_graphs = make_truncated_split(
        original_val,
        cutoff,
        f"Val t<={cutoff}"
    )

    test_graphs = make_truncated_split(
        original_test,
        cutoff,
        f"Test t<={cutoff}"
    )

    train_nodes = np.asarray(
        [g.num_nodes for g in train_graphs]
    )

    val_nodes = np.asarray(
        [g.num_nodes for g in val_graphs]
    )

    test_nodes = np.asarray(
        [g.num_nodes for g in test_graphs]
    )

    print(
        f"\nTrain graphs: {len(train_graphs)} | "
        f"nodes min/mean/max = "
        f"{train_nodes.min()}/"
        f"{train_nodes.mean():.1f}/"
        f"{train_nodes.max()}"
    )

    print(
        f"Val graphs:   {len(val_graphs)} | "
        f"nodes min/mean/max = "
        f"{val_nodes.min()}/"
        f"{val_nodes.mean():.1f}/"
        f"{val_nodes.max()}"
    )

    print(
        f"Test graphs:  {len(test_graphs)} | "
        f"nodes min/mean/max = "
        f"{test_nodes.min()}/"
        f"{test_nodes.mean():.1f}/"
        f"{test_nodes.max()}"
    )

    input_dim = int(
        train_graphs[0].x.shape[1]
    )

    counts, class_weights = (
        get_class_weights(
            train_graphs
        )
    )

    print(
        f"\nTrain labels: "
        f"correct={counts[0]}, "
        f"hallucinated={counts[1]}"
    )

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

    model = EarlyEpiGraph(
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
        / f"epigraph_early_t{cutoff}_seed{args.seed}.pt"
    )

    best_val_auroc = -1.0
    no_improvement = 0

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

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "cutoff":
                        int(cutoff),

                    "input_dim":
                        int(input_dim),

                    "hidden_dim":
                        int(args.hidden_dim),

                    "dropout":
                        float(args.dropout),

                    "seed":
                        int(args.seed),

                    "best_val_auroc":
                        float(best_val_auroc),
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

    # -----------------------------------------------------
    # Load best model
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Validation threshold
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

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

    print("\nValidation")
    print("----------")
    print(
        f"AUROC:              "
        f"{val_metrics['auroc']:.4f}"
    )
    print(
        f"Balanced accuracy:  "
        f"{val_metrics['balanced_accuracy']:.4f}"
    )

    print("\nTest")
    print("----")
    print(
        f"AUROC:              "
        f"{test_metrics['auroc']:.4f}"
    )
    print(
        f"AUPRC hallucinated: "
        f"{test_metrics['auprc_hallucinated']:.4f}"
    )
    print(
        f"AUPRC correct:      "
        f"{test_metrics['auprc_correct']:.4f}"
    )
    print(
        f"Macro F1:           "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Balanced accuracy:  "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Accuracy:            "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Threshold:           "
        f"{test_metrics['threshold']:.3f}"
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    val_pred_path = (
        RESULT_DIR
        / f"step10_early_t{cutoff}_seed{args.seed}_val.csv"
    )

    test_pred_path = (
        RESULT_DIR
        / f"step10_early_t{cutoff}_seed{args.seed}_test.csv"
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
        "cutoff":
            int(cutoff),

        "seed":
            int(args.seed),

        "train_graphs":
            len(train_graphs),

        "val_graphs":
            len(val_graphs),

        "test_graphs":
            len(test_graphs),

        "train_nodes_mean":
            float(train_nodes.mean()),

        "val_nodes_mean":
            float(val_nodes.mean()),

        "test_nodes_mean":
            float(test_nodes.mean()),

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

        "checkpoint":
            str(checkpoint_path),

        "history":
            history,
    }

    result_path = (
        RESULT_DIR
        / f"step10_early_t{cutoff}_seed{args.seed}.json"
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

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=DEFAULT_CUTOFFS,
        help="Example: --cutoffs 0 1 2 4 8"
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

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    set_seed(args.seed)

    print("=" * 70)
    print(
        "EPIGRAPH — STEP 10: EARLY DETECTION"
    )
    print("=" * 70)

    print(
        f"\nCutoffs: {args.cutoffs}"
    )

    print(
        f"Seed: {args.seed}"
    )

    # =====================================================
    # LOAD ORIGINAL FULL GRAPHS ONCE
    # =====================================================

    original_train = load_graphs(
        "train"
    )

    original_val = load_graphs(
        "val"
    )

    original_test = load_graphs(
        "test"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # =====================================================
    # RUN EVERY CUTOFF
    # =====================================================

    all_results = []

    for cutoff in args.cutoffs:
        result = run_cutoff(
            cutoff=cutoff,
            original_train=original_train,
            original_val=original_val,
            original_test=original_test,
            args=args,
            device=device
        )

        all_results.append(
            result
        )

    # =====================================================
    # SAVE SUMMARY CSV + JSON
    # =====================================================

    summary_csv = (
        RESULT_DIR
        / f"step10_early_detection_seed{args.seed}.csv"
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
                "cutoff",
                "val_auroc",
                "test_auroc",
                "test_auprc_hallucinated",
                "test_auprc_correct",
                "test_macro_f1",
                "test_balanced_accuracy",
                "test_accuracy",
                "threshold",
            ]
        )

        for r in all_results:
            writer.writerow(
                [
                    r["cutoff"],
                    r["validation"]["auroc"],
                    r["test"]["auroc"],
                    r["test"]["auprc_hallucinated"],
                    r["test"]["auprc_correct"],
                    r["test"]["macro_f1"],
                    r["test"]["balanced_accuracy"],
                    r["test"]["accuracy"],
                    r["test"]["threshold"],
                ]
            )

    summary_json = (
        RESULT_DIR
        / f"step10_early_detection_seed{args.seed}.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "seed":
                    int(args.seed),

                "cutoffs":
                    args.cutoffs,

                "results":
                    all_results,
            },
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("EARLY-DETECTION SUMMARY")
    print("=" * 70)

    print(
        "\ncutoff | test AUROC | balanced acc"
    )

    for r in all_results:
        print(
            f"{r['cutoff']:>6} | "
            f"{r['test']['auroc']:.4f}     | "
            f"{r['test']['balanced_accuracy']:.4f}"
        )

    print(
        f"\nSummary CSV:\n{summary_csv}"
    )

    print(
        f"\nSummary JSON:\n{summary_json}"
    )


if __name__ == "__main__":
    main()
