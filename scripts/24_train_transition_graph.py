"""Train/evaluate PCA-16 Layer-Time Transition Graph classifiers.

Runs t=0 and t<=4 experiments across seeds 42-46. GraphSAGE matches the
original EpiGraph architecture; a node-MLP plus mean-pooling ablation uses the
same node features without message passing.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import subgraph
from tqdm import tqdm


GRAPH_ROOT = Path("processed_graphs/transitions_pca16")
MODEL_DIR = Path("models/transition_graph")
RESULT_DIR = Path("results/transition_graph")

SEEDS = (42, 43, 44, 45, 46)
EXPERIMENTS = {"t0": 0, "temporal_t4": 4}
INPUT_DIM = 22
HIDDEN_DIM = 64
DROPOUT = 0.30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15

ORIGINAL_T0_FIVE_SEED_AUROC = {"mean": 0.8001, "std": 0.0051}
ORIGINAL_T0_CUTOFF_AUROC = 0.8039


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json_dump(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def graph_files(split):
    files = sorted((GRAPH_ROOT / split).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No transition graphs found for {split}")
    return files


def filter_graph(graph, cutoff):
    keep = graph.time_position <= cutoff
    edge_index, _, edge_mask = subgraph(
        subset=keep,
        edge_index=graph.edge_index,
        relabel_nodes=True,
        num_nodes=graph.num_nodes,
        return_edge_mask=True,
    )
    filtered = Data(
        x=graph.x[keep].clone(),
        edge_index=edge_index.clone(),
        edge_type=graph.edge_type[edge_mask].clone(),
        edge_time_gap=graph.edge_time_gap[edge_mask].clone(),
        edge_time_delta=graph.edge_time_delta[edge_mask].clone(),
        y=graph.y.clone(),
        transition_index=graph.transition_index[keep].clone(),
        time_position=graph.time_position[keep].clone(),
    )
    filtered.sample_id = str(graph.sample_id)
    filtered.split = str(graph.split)
    return filtered


def validate_filtered(graph, experiment, cutoff):
    if graph.x.ndim != 2 or graph.x.shape[1] != INPUT_DIM:
        raise ValueError(f"{graph.sample_id}: expected 22 node features")
    if not torch.isfinite(graph.x).all():
        raise ValueError(f"{graph.sample_id}: node features contain NaN/Inf")
    if int(graph.y.item()) not in (0, 1):
        raise ValueError(f"{graph.sample_id}: invalid label")
    if graph.time_position.max().item() > cutoff:
        raise ValueError(f"{graph.sample_id}: future-checkpoint leakage")
    if int((graph.time_position == 0).sum()) != 27:
        raise ValueError(f"{graph.sample_id}: t=0 must contain 27 nodes")
    if experiment == "t0":
        if graph.num_nodes != 27 or not torch.all(graph.time_position == 0):
            raise ValueError(f"{graph.sample_id}: invalid t=0 filtering")
        if not torch.all(graph.edge_type == 0):
            raise ValueError(f"{graph.sample_id}: t=0 graph contains temporal edges")
    if experiment == "temporal_t4" and (graph.time_position == 8).any():
        raise ValueError(f"{graph.sample_id}: t=8 leaked into temporal_t4")


def load_graphs(split, experiment, cutoff):
    graphs = []
    coverage = {time: 0 for time in (0, 1, 2, 4)}
    for path in tqdm(graph_files(split), desc=f"Load {experiment} {split}"):
        source = torch.load(path, map_location="cpu", weights_only=False)
        filtered = filter_graph(source, cutoff)
        validate_filtered(filtered, experiment, cutoff)
        graphs.append(filtered)
        present = set(filtered.time_position.tolist())
        for time in coverage:
            coverage[time] += int(time in present)
    return graphs, coverage


def validate_splits(datasets):
    id_sets = {}
    for split, graphs in datasets.items():
        ids = [graph.sample_id for graph in graphs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in {split}")
        id_sets[split] = set(ids)
        if any(graph.split != split for graph in graphs):
            raise ValueError(f"Incorrect split metadata in {split}")
    if id_sets["train"] & id_sets["val"] or id_sets["train"] & id_sets["test"] or id_sets["val"] & id_sets["test"]:
        raise ValueError("Sample overlap detected between train/val/test")


class GraphSAGEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(INPUT_DIM, HIDDEN_DIM)
        self.norm1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.conv2 = SAGEConv(HIDDEN_DIM, HIDDEN_DIM)
        self.norm2 = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        x = self.dropout(torch.relu(self.norm1(self.conv1(x, edge_index))))
        x = self.dropout(torch.relu(self.norm2(self.conv2(x, edge_index))))
        return self.classifier(global_mean_pool(x, batch))


class NodeMLPClassifier(nn.Module):
    """Graph-free matched ablation: node MLP followed by mean pooling."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.norm1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.norm2 = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        del edge_index
        x = self.dropout(torch.relu(self.norm1(self.linear1(x))))
        x = self.dropout(torch.relu(self.norm2(self.linear2(x))))
        return self.classifier(global_mean_pool(x, batch))


def class_weights(graphs):
    labels = np.asarray([int(graph.y.item()) for graph in graphs], dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    weights = len(labels) / (2.0 * np.maximum(counts, 1))
    return counts, torch.tensor(weights, dtype=torch.float32)


def make_loader(graphs, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        labels = batch.y.view(-1).long()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.shape[0]
        total += labels.shape[0]
    return total_loss / max(total, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels = []
    probabilities = []
    sample_ids = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(batch.y.view(-1).long().cpu().tolist())
        ids = batch.sample_id
        sample_ids.extend([str(value) for value in ids] if isinstance(ids, (list, tuple)) else [str(ids)])
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        sample_ids,
    )


def find_best_threshold(labels, probabilities):
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(np.int64)
        score = balanced_accuracy_score(labels, predictions)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "auprc_hallucinated": float(average_precision_score(labels, probabilities)),
        "auprc_correct": float(average_precision_score(1 - labels, 1.0 - probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def save_predictions(path, sample_ids, labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prob_hallucinated", "prediction", "threshold"])
        for row in zip(sample_ids, labels, probabilities, predictions):
            writer.writerow([row[0], int(row[1]), float(row[2]), int(row[3]), threshold])
    os.replace(temporary, path)


def model_for(architecture):
    if architecture == "graphsage":
        return GraphSAGEClassifier()
    if architecture == "node_mlp":
        return NodeMLPClassifier()
    raise ValueError(architecture)


def run_one(datasets, experiment, architecture, seed, device, overwrite):
    run_name = f"{experiment}_{architecture}_seed{seed}"
    result_path = RESULT_DIR / f"{run_name}.json"
    checkpoint_path = MODEL_DIR / f"transition_{run_name}.pt"
    val_prediction_path = RESULT_DIR / f"{run_name}_val_predictions.csv"
    test_prediction_path = RESULT_DIR / f"{run_name}_test_predictions.csv"

    if result_path.exists() and not overwrite:
        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        print(f"Loaded existing run: {run_name}")
        return result

    set_seed(seed)
    train_loader = make_loader(datasets["train"], True, seed)
    val_loader = make_loader(datasets["val"], False, seed)
    test_loader = make_loader(datasets["test"], False, seed)
    counts, weights = class_weights(datasets["train"])
    model = model_for(architecture).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_auroc = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_labels, val_probabilities, _ = predict(model, val_loader, device)
        val_auroc = float(roc_auc_score(val_labels, val_probabilities))
        history.append({"epoch": epoch, "train_loss": float(loss), "val_auroc": val_auroc})
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "experiment": experiment,
                    "architecture": architecture,
                    "seed": seed,
                    "input_dim": INPUT_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT,
                    "best_val_auroc": best_val_auroc,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_labels, val_probabilities, val_ids = predict(model, val_loader, device)
    threshold = find_best_threshold(val_labels, val_probabilities)
    validation_metrics = metrics(val_labels, val_probabilities, threshold)

    # Test is evaluated only after validation has selected the epoch/threshold.
    test_labels, test_probabilities, test_ids = predict(model, test_loader, device)
    test_metrics = metrics(test_labels, test_probabilities, threshold)
    save_predictions(val_prediction_path, val_ids, val_labels, val_probabilities, threshold)
    save_predictions(test_prediction_path, test_ids, test_labels, test_probabilities, threshold)

    result = {
        "experiment": experiment,
        "architecture": architecture,
        "seed": seed,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "epochs_trained": len(history),
        "best_epoch": int(max(history, key=lambda row: row["val_auroc"])["epoch"]),
        "best_val_auroc": float(best_val_auroc),
        "class_counts": {"correct": int(counts[0]), "hallucinated": int(counts[1])},
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "validation_predictions": str(val_prediction_path),
        "test_predictions": str(test_prediction_path),
    }
    atomic_json_dump(result, result_path)
    print(
        f"{run_name}: epoch={result['best_epoch']} "
        f"val={best_val_auroc:.4f} test={test_metrics['auroc']:.4f} "
        f"threshold={threshold:.3f}"
    )
    return result


def aggregate_runs(runs):
    metric_names = ("auroc", "auprc", "balanced_accuracy", "macro_f1")
    summary = {}
    for experiment in EXPERIMENTS:
        summary[experiment] = {}
        architectures = sorted({run["architecture"] for run in runs if run["experiment"] == experiment})
        for architecture in architectures:
            selected = [run for run in runs if run["experiment"] == experiment and run["architecture"] == architecture]
            aggregate = {"seeds": [run["seed"] for run in selected], "per_seed": selected}
            for metric_name in metric_names:
                values = np.asarray([run["test"][metric_name] for run in selected], dtype=np.float64)
                aggregate[metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            aggregate["thresholds"] = [run["test"]["threshold"] for run in selected]
            aggregate["confusion_matrices"] = [run["test"]["confusion_matrix"] for run in selected]
            summary[experiment][architecture] = aggregate
    return summary


def write_summary_csv(summary, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "architecture", "metric", "mean", "std"])
        for experiment, architectures in summary.items():
            for architecture, values in architectures.items():
                for metric_name in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
                    writer.writerow([
                        experiment, architecture, metric_name,
                        values[metric_name]["mean"], values[metric_name]["std"],
                    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("At least one seed is required")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_runs = []
    dataset_report = {}
    for experiment, cutoff in EXPERIMENTS.items():
        datasets = {}
        coverage = {}
        for split in ("train", "val", "test"):
            datasets[split], coverage[split] = load_graphs(split, experiment, cutoff)
        validate_splits(datasets)
        dataset_report[experiment] = {
            "cutoff": cutoff,
            "counts": {split: len(graphs) for split, graphs in datasets.items()},
            "checkpoint_coverage": coverage,
            "no_split_overlap": True,
            "eos_nodes_present": False,
        }
        architectures = ["graphsage"] + ([] if args.skip_mlp else ["node_mlp"])
        for architecture in architectures:
            for seed in args.seeds:
                all_runs.append(
                    run_one(datasets, experiment, architecture, seed, device, args.overwrite)
                )

    aggregate = aggregate_runs(all_runs)
    report = {
        "module": 24,
        "graph_source": str(GRAPH_ROOT),
        "seeds": args.seeds,
        "dataset_validation": dataset_report,
        "hyperparameters": {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "selection": "best validation AUROC; threshold maximizes validation balanced accuracy",
        },
        "baselines": {
            "original_t0_five_seed_auroc": ORIGINAL_T0_FIVE_SEED_AUROC,
            "original_t0_cutoff_auroc_separate_experiment": ORIGINAL_T0_CUTOFF_AUROC,
        },
        "results": aggregate,
    }
    atomic_json_dump(report, RESULT_DIR / "summary.json")
    write_summary_csv(aggregate, RESULT_DIR / "summary.csv")
    print("\nAGGREGATED RESULTS")
    for experiment, architectures in aggregate.items():
        for architecture, values in architectures.items():
            print(
                f"{experiment} {architecture}: "
                f"AUROC={values['auroc']['mean']:.4f}+-{values['auroc']['std']:.4f}; "
                f"AUPRC={values['auprc']['mean']:.4f}+-{values['auprc']['std']:.4f}; "
                f"BalAcc={values['balanced_accuracy']['mean']:.4f}+-{values['balanced_accuracy']['std']:.4f}; "
                f"MacroF1={values['macro_f1']['mean']:.4f}+-{values['macro_f1']['std']:.4f}"
            )


if __name__ == "__main__":
    main()
