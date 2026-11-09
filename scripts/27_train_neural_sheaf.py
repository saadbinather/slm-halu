"""Train diagonal Neural Sheaf Diffusion on saved PCA-16 transition graphs.

For each undirected edge e=(u,v), learned diagonal incidence maps F[u,e] and
F[v,e] define L = delta.T @ delta. Diffusion uses the diagonal-normalized
sheaf Laplacian D^-1/2 L D^-1/2 without constructing a dense matrix.
Run --smoke-test first; default training is seed 42 only.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph
from tqdm import tqdm


GRAPH_ROOT = Path("processed_graphs/transitions_pca16")
MODEL_ROOT = Path("models/neural_sheaf")
RESULT_ROOT = Path("results/neural_sheaf")
COUNTS = {"train": 1858, "val": 398, "test": 399}
EXPERIMENTS = {"t0": 0, "temporal_t4": 4}
BASELINES = {
    "t0": {"mlp": 0.8097160603371784, "graphsage": 0.8071146245059289, "gcn": 0.8105106074050173},
    "temporal_t4": {"mlp": 0.8165685246430587, "graphsage": 0.8155924820521093, "gcn": 0.8142695813503267},
}
INPUT_DIM = 22
HIDDEN_DIM = 32
BATCH_SIZE = 64
DROPOUT = 0.30
LR = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 100
PATIENCE = 15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(tmp, path)


def filter_graph(graph, cutoff):
    keep = graph.time_position <= cutoff
    edge_index, _, edge_mask = subgraph(keep, graph.edge_index, relabel_nodes=True,
                                        num_nodes=graph.num_nodes, return_edge_mask=True)
    result = Data(x=graph.x[keep].clone(), edge_index=edge_index.clone(),
                  edge_type=graph.edge_type[edge_mask].clone(),
                  edge_time_gap=graph.edge_time_gap[edge_mask].clone(),
                  edge_time_delta=graph.edge_time_delta[edge_mask].clone(),
                  y=graph.y.clone(), time_position=graph.time_position[keep].clone(),
                  transition_index=graph.transition_index[keep].clone())
    result.sample_id = str(graph.sample_id)
    result.split = str(graph.split)
    return result


def validate_graph(graph, split, cutoff):
    if graph.split != split or graph.x.ndim != 2 or graph.x.shape[1] != INPUT_DIM:
        raise ValueError(f"{graph.sample_id}: split or input dimension mismatch")
    if int(graph.y.item()) not in (0, 1) or not torch.isfinite(graph.x).all():
        raise ValueError(f"{graph.sample_id}: invalid label or nonfinite features")
    if graph.num_nodes > 135 or graph.time_position.max() > cutoff:
        raise ValueError(f"{graph.sample_id}: future checkpoint leakage")
    if int((graph.time_position == 0).sum()) != 27:
        raise ValueError(f"{graph.sample_id}: t=0 does not have 27 nodes")
    if cutoff == 0 and (graph.num_nodes != 27 or (graph.edge_type != 0).any()):
        raise ValueError(f"{graph.sample_id}: invalid t=0 graph")
    src, dst = graph.edge_index
    if src.numel() != graph.edge_type.numel() or src.numel() != graph.edge_time_gap.numel():
        raise ValueError(f"{graph.sample_id}: edge metadata mismatch")
    if src.numel() and (src.min() < 0 or dst.max() >= graph.num_nodes):
        raise ValueError(f"{graph.sample_id}: invalid edge endpoint")
    depth, temporal = graph.edge_type == 0, graph.edge_type == 1
    if not torch.all(depth | temporal):
        raise ValueError(f"{graph.sample_id}: unexpected edge type")
    if not torch.all((graph.time_position[src[depth]] == graph.time_position[dst[depth]])
                     & ((graph.transition_index[src[depth]] - graph.transition_index[dst[depth]]).abs() == 1)
                     & (graph.edge_time_gap[depth] == 0)):
        raise ValueError(f"{graph.sample_id}: invalid depth edge")
    if not torch.all((graph.transition_index[src[temporal]] == graph.transition_index[dst[temporal]])
                     & ((graph.time_position[src[temporal]] - graph.time_position[dst[temporal]]).abs() == graph.edge_time_gap[temporal])
                     & (graph.edge_time_gap[temporal] > 0)):
        raise ValueError(f"{graph.sample_id}: invalid temporal edge")
    # The source graphs are bidirectional; each undirected edge must occur twice.
    forward = {(int(u), int(v), int(t), int(g)) for u, v, t, g in
               zip(src.tolist(), dst.tolist(), graph.edge_type.tolist(), graph.edge_time_gap.tolist())}
    if len(forward) != src.numel() or any((v, u, t, g) not in forward for u, v, t, g in forward):
        raise ValueError(f"{graph.sample_id}: edges are not unique bidirectional pairs")


def load_graphs(split, cutoff, limit=0):
    files = sorted((GRAPH_ROOT / split).glob("*.pt"))
    if len(files) != COUNTS[split]:
        raise ValueError(f"{split}: expected {COUNTS[split]} graphs, found {len(files)}")
    if limit:
        files = files[:limit]
    graphs = []
    coverage = {t: 0 for t in (0, 1, 2, 4)}
    for path in tqdm(files, desc=f"Load {split} t<={cutoff}"):
        source = torch.load(path, map_location="cpu", weights_only=False)
        # The source graph already excludes EOS checkpoints; verify retained
        # positions never match a source position marked EOS.
        eos_times = {int(t) for t, flag in zip(source.original_time_positions.tolist(), source.checkpoint_is_eos.tolist()) if flag == 1}
        if eos_times.intersection(source.time_position.tolist()):
            raise ValueError(f"{path}: EOS node present in source graph")
        graph = filter_graph(source, cutoff)
        validate_graph(graph, split, cutoff)
        for t in coverage:
            coverage[t] += int(t in graph.time_position)
        graphs.append(graph)
    return graphs, coverage


def validate_splits(datasets):
    sets = {}
    for split, graphs in datasets.items():
        ids = [g.sample_id for g in graphs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split}: duplicate sample IDs")
        sets[split] = set(ids)
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("train/val/test sample overlap")


class RestrictionMaps(nn.Module):
    """Predict two diagonal incidence maps per undirected edge."""

    def __init__(self, width):
        super().__init__()
        self.depth = nn.Sequential(nn.Linear(2 * width + 1, width), nn.ReLU(), nn.Linear(width, 2 * width))
        self.temporal = nn.Sequential(nn.Linear(2 * width + 1, width), nn.ReLU(), nn.Linear(width, 2 * width))
        # Identity-like start gives ordinary normalized diffusion initially.
        for network in (self.depth, self.temporal):
            nn.init.zeros_(network[-1].weight)
            nn.init.ones_(network[-1].bias)

    def forward(self, h, edge_index, edge_type, edge_gap):
        source, target = edge_index
        # One orientation per saved bidirectional pair, so L=delta.T delta
        # counts each undirected edge exactly once.
        keep = source < target
        u, v = source[keep], target[keep]
        kind, gap = edge_type[keep], edge_gap[keep]
        features = torch.cat((h[u], h[v], torch.log1p(gap.float()).unsqueeze(1)), dim=1)
        maps = h.new_empty((len(u), 2 * h.shape[1]))
        for value, network in ((0, self.depth), (1, self.temporal)):
            selected = kind == value
            if selected.any():
                maps[selected] = 2.0 * torch.tanh(network(features[selected]))
        return u, v, maps[:, :h.shape[1]], maps[:, h.shape[1]:]


class SheafDiffusion(nn.Module):
    """One explicit step h - alpha D^-1/2 delta.T delta D^-1/2 h."""

    def __init__(self, width):
        super().__init__()
        self.restrictions = RestrictionMaps(width)
        self.mix = nn.Linear(width, width)
        self.logit_step = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def laplacian(h, u, v, fu, fv):
        degree = torch.zeros_like(h)
        degree.index_add_(0, u, fu.square())
        degree.index_add_(0, v, fv.square())
        inv = degree.clamp_min(1e-6).rsqrt()
        z = inv * h
        disagreement = fu * z[u] - fv * z[v]
        result = torch.zeros_like(h)
        result.index_add_(0, u, fu * disagreement)
        result.index_add_(0, v, -fv * disagreement)
        return inv * result

    def forward(self, h, edge_index, edge_type, edge_gap):
        u, v, fu, fv = self.restrictions(h, edge_index, edge_type, edge_gap)
        signal = self.mix(h)
        return signal - torch.sigmoid(self.logit_step) * self.laplacian(signal, u, v, fu, fv)


class NeuralSheafClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.diffusion1 = SheafDiffusion(HIDDEN_DIM)
        self.diffusion2 = SheafDiffusion(HIDDEN_DIM)
        self.norm1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.norm2 = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                  nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 2))

    def forward(self, batch):
        h = torch.relu(self.input(batch.x.float()))
        h = self.dropout(torch.relu(self.norm1(self.diffusion1(h, batch.edge_index, batch.edge_type, batch.edge_time_gap))))
        h = self.dropout(torch.relu(self.norm2(self.diffusion2(h, batch.edge_index, batch.edge_type, batch.edge_time_gap))))
        return self.head(global_mean_pool(h, batch.batch))


def make_loader(graphs, shuffle=False, seed=42, batch_size=BATCH_SIZE):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities, ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        probabilities.extend(torch.softmax(model(batch), dim=1)[:, 1].cpu().tolist())
        labels.extend(batch.y.view(-1).cpu().tolist())
        ids.extend(str(s) for s in batch.sample_id)
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities), ids


def select_threshold(labels, probabilities):
    choices = np.linspace(0.05, 0.95, 181)
    return float(max(choices, key=lambda t: balanced_accuracy_score(labels, probabilities >= t)))


def score(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    return {"auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
            "threshold": threshold}


def save_predictions(path, ids, labels, probabilities, threshold):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "label", "prob_hallucinated", "prediction", "threshold"))
        for sid, label, probability in zip(ids, labels, probabilities):
            writer.writerow((sid, int(label), float(probability), int(probability >= threshold), threshold))
    os.replace(tmp, path)


def smoke_test(device):
    set_seed(42)
    examples = []
    for split in ("train", "val"):
        path = sorted((GRAPH_ROOT / split).glob("*.pt"))[0]
        source = torch.load(path, map_location="cpu", weights_only=False)
        for cutoff in (0, 4):
            graph = filter_graph(source, cutoff)
            validate_graph(graph, split, cutoff)
            examples.append(graph)
    batch = next(iter(make_loader(examples, batch_size=4))).to(device)
    model = NeuralSheafClassifier().to(device)
    model.train()
    # Verify the normalized sheaf operator is PSD: <h,Lh> >= 0.
    h = torch.randn(batch.num_nodes, HIDDEN_DIM, device=device)
    u, v, fu, fv = model.diffusion1.restrictions(h, batch.edge_index, batch.edge_type, batch.edge_time_gap)
    energy = (h * model.diffusion1.laplacian(h, u, v, fu, fv)).sum().item()
    if energy < -1e-3:
        raise RuntimeError(f"Sheaf Laplacian has negative energy: {energy}")
    output = model(batch)
    loss = nn.CrossEntropyLoss()(output, batch.y.view(-1))
    loss.backward()
    if output.shape != (4, 2) or not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise RuntimeError("Nonfinite smoke-test output")
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Missing or nonfinite sheaf gradients")
    print(f"Smoke test passed: four graphs, PSD energy={energy:.4f}, finite outputs/gradients", flush=True)


def run_one(datasets, experiment, seed, device, coverage, overwrite=False):
    name = f"{experiment}_seed{seed}"
    result_path = RESULT_ROOT / f"{name}.json"
    checkpoint_path = MODEL_ROOT / f"{name}.pt"
    val_path = RESULT_ROOT / f"{name}_val_predictions.csv"
    test_path = RESULT_ROOT / f"{name}_test_predictions.csv"
    if not overwrite and all(p.exists() for p in (result_path, checkpoint_path, val_path, test_path)):
        with result_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("experiment") != experiment or existing.get("seed") != seed:
            raise ValueError(f"{result_path}: mismatched existing run")
        print(f"Skipping completed {name}: AUROC={existing['test']['auroc']:.4f}", flush=True)
        return existing

    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_loader = make_loader(datasets["train"], True, seed)
    val_loader = make_loader(datasets["val"])
    test_loader = make_loader(datasets["test"])
    labels_train = np.asarray([int(g.y.item()) for g in datasets["train"]])
    class_counts = np.bincount(labels_train, minlength=2)
    weights = torch.tensor(len(labels_train) / (2 * class_counts), dtype=torch.float32, device=device)
    model = NeuralSheafClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_auc, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1).long())
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: nonfinite loss at epoch {epoch}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"{name}: nonfinite gradients at epoch {epoch}")
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        val_labels, val_probs, val_ids = predict(model, val_loader, device)
        auc = float(roc_auc_score(val_labels, val_probs))
        history.append({"epoch": epoch, "train_loss": total_loss / len(datasets["train"]), "val_auroc": auc})
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            tmp = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"model_state_dict": model.state_dict(), "experiment": experiment,
                        "seed": seed, "epoch": epoch, "val_auroc": auc,
                        "architecture": "diagonal_normalized_neural_sheaf_diffusion"}, tmp)
            os.replace(tmp, checkpoint_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"{name} epoch={epoch} val_AUROC={auc:.4f} best={best_auc:.4f}", flush=True)
        if stale >= PATIENCE:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_labels, val_probs, val_ids = predict(model, val_loader, device)
    threshold = select_threshold(val_labels, val_probs)
    test_labels, test_probs, test_ids = predict(model, test_loader, device)
    for split, labels, ids in (("val", val_labels, val_ids), ("test", test_labels, test_ids)):
        if len(ids) != len(datasets[split]) or ids != [g.sample_id for g in datasets[split]]:
            raise ValueError(f"{name}: {split} predictions misaligned")
        if not np.array_equal(labels, [int(g.y.item()) for g in datasets[split]]):
            raise ValueError(f"{name}: {split} labels misaligned")
    save_predictions(val_path, val_ids, val_labels, val_probs, threshold)
    save_predictions(test_path, test_ids, test_labels, test_probs, threshold)
    runtime = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
    result = {"experiment": experiment, "seed": seed, "architecture": "diagonal_normalized_neural_sheaf_diffusion",
              "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "batch_size": BATCH_SIZE,
              "dropout": DROPOUT, "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
              "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "best_epoch": best_epoch,
              "epochs_trained": epoch, "best_val_auroc": best_auc,
              "validation": score(val_labels, val_probs, threshold),
              "test": score(test_labels, test_probs, threshold),
              "counts": {split: len(graphs) for split, graphs in datasets.items()},
              "checkpoint_coverage": coverage, "class_counts": class_counts.tolist(),
              "runtime_seconds": runtime, "peak_gpu_memory_mib": peak_mib,
              "checkpoint": str(checkpoint_path), "val_predictions": str(val_path),
              "test_predictions": str(test_path), "history": history}
    atomic_json(result, result_path)
    print(f"{name}: test AUROC={result['test']['auroc']:.4f}, runtime={runtime:.1f}s, peak_gpu_mib={peak_mib}", flush=True)
    return result


def write_summary(runs, device):
    grouped = {}
    for experiment in EXPERIMENTS:
        selected = [r for r in runs if r["experiment"] == experiment]
        if not selected:
            continue
        grouped[experiment] = {"per_seed": selected}
        for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
            values = np.asarray([r["test"][metric] for r in selected])
            grouped[experiment][metric] = {"mean": float(values.mean()),
                                           "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
    atomic_json({"module": 27, "device": str(device), "sheaf_operator":
                 "h - sigmoid(alpha) D^-1/2 delta.T delta D^-1/2 h; learned diagonal endpoint restrictions; separate depth/temporal networks; log1p(time gap)",
                 "baseline_auroc": BASELINES, "results": grouped}, RESULT_ROOT / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Default runs feasibility seed 42 only")
    parser.add_argument("--experiment", choices=["t0", "temporal_t4", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("Provide unique seeds")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        smoke_test(device)
        return
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    experiments = EXPERIMENTS if args.experiment == "all" else {args.experiment: EXPERIMENTS[args.experiment]}
    runs = []
    for name, cutoff in experiments.items():
        datasets, coverage = {}, {}
        for split in COUNTS:
            datasets[split], coverage[split] = load_graphs(split, cutoff)
        validate_splits(datasets)
        for seed in args.seeds:
            runs.append(run_one(datasets, name, seed, device, coverage, args.overwrite))
    write_summary(runs, device)


if __name__ == "__main__":
    main()
