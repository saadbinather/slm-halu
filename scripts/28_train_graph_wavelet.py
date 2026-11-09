"""Train polynomial diffusion-wavelet networks on saved transition graphs.

For a symmetric, relation-weighted adjacency A, P=(I+D^-1/2 A D^-1/2)/2
is a lazy graph diffusion. The wavelet bank is (I-P), (P-P²), (P²-P⁴)
and low-pass P⁴. These are spectral polynomials evaluated with sparse edges.
The single-scale ablation uses four copies of P x with identical capacity.
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
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph
from tqdm import tqdm

GRAPH_ROOT = Path("processed_graphs/transitions_pca16")
MODEL_ROOT = Path("models/graph_wavelet")
RESULT_ROOT = Path("results/graph_wavelet")
COUNTS = {"train": 1858, "val": 398, "test": 399}
EXPERIMENTS = {"t0": 0, "temporal_t4": 4}
VARIANTS = ("multiscale", "single_scale")
SEEDS = (42, 43, 44, 45, 46)
BASELINES = {
    "t0": {"mlp": 0.8097160603371784, "graphsage": 0.8071146245059289,
           "gcn": 0.8105106074050173, "neural_sheaf": 0.7922239251431799},
    "temporal_t4": {"mlp": 0.8165685246430587, "graphsage": 0.8155924820521093,
                    "gcn": 0.8142695813503267, "neural_sheaf": 0.8169718480277487},
}
WIDTH = 32
BATCH_SIZE = 64
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


def filter_graph(source, cutoff):
    keep = source.time_position <= cutoff
    edge_index, _, edge_mask = subgraph(keep, source.edge_index, relabel_nodes=True,
                                        num_nodes=source.num_nodes, return_edge_mask=True)
    graph = Data(x=source.x[keep].clone(), y=source.y.clone(), edge_index=edge_index,
                 edge_type=source.edge_type[edge_mask].clone(),
                 edge_time_gap=source.edge_time_gap[edge_mask].clone(),
                 time_position=source.time_position[keep].clone(),
                 transition_index=source.transition_index[keep].clone())
    graph.sample_id = str(source.sample_id)
    graph.split = str(source.split)
    return graph


def validate_graph(graph, source, split, cutoff):
    if graph.split != split or tuple(graph.x.shape)[1:] != (22,) or not torch.isfinite(graph.x).all():
        raise ValueError(f"{graph.sample_id}: invalid split/features")
    if int(graph.y.item()) not in (0, 1) or int((graph.time_position == 0).sum()) != 27:
        raise ValueError(f"{graph.sample_id}: invalid label or t=0 nodes")
    if graph.time_position.max() > cutoff or graph.num_nodes > 135:
        raise ValueError(f"{graph.sample_id}: future checkpoint leakage")
    eos = {int(t) for t, flag in zip(source.original_time_positions.tolist(), source.checkpoint_is_eos.tolist()) if flag == 1}
    if eos.intersection(graph.time_position.tolist()):
        raise ValueError(f"{graph.sample_id}: EOS nodes present")
    u, v = graph.edge_index
    if u.numel() != graph.edge_type.numel() or u.numel() != graph.edge_time_gap.numel():
        raise ValueError(f"{graph.sample_id}: edge metadata mismatch")
    if u.numel() and (u.min() < 0 or v.max() >= graph.num_nodes):
        raise ValueError(f"{graph.sample_id}: edge endpoint out of range")
    depth, temporal = graph.edge_type == 0, graph.edge_type == 1
    if not torch.all(depth | temporal):
        raise ValueError(f"{graph.sample_id}: unknown edge relation")
    if cutoff == 0 and (graph.num_nodes != 27 or temporal.any()):
        raise ValueError(f"{graph.sample_id}: invalid t=0 graph")
    if not torch.all((graph.time_position[u[depth]] == graph.time_position[v[depth]])
                     & ((graph.transition_index[u[depth]] - graph.transition_index[v[depth]]).abs() == 1)
                     & (graph.edge_time_gap[depth] == 0)):
        raise ValueError(f"{graph.sample_id}: bad depth edge")
    if not torch.all((graph.transition_index[u[temporal]] == graph.transition_index[v[temporal]])
                     & ((graph.time_position[u[temporal]] - graph.time_position[v[temporal]]).abs() == graph.edge_time_gap[temporal])
                     & (graph.edge_time_gap[temporal] > 0)):
        raise ValueError(f"{graph.sample_id}: bad temporal edge")


def load_graphs(split, cutoff):
    files = sorted((GRAPH_ROOT / split).glob("*.pt"))
    if len(files) != COUNTS[split]:
        raise ValueError(f"{split}: expected {COUNTS[split]}, got {len(files)} files")
    graphs = []
    coverage = {t: 0 for t in (0, 1, 2, 4)}
    for path in tqdm(files, desc=f"Load {split} t<={cutoff}", mininterval=2):
        source = torch.load(path, map_location="cpu", weights_only=False)
        graph = filter_graph(source, cutoff)
        validate_graph(graph, source, split, cutoff)
        for t in coverage:
            coverage[t] += int(t in graph.time_position)
        graphs.append(graph)
    return graphs, coverage


def validate_splits(datasets):
    ids = {split: [graph.sample_id for graph in graphs] for split, graphs in datasets.items()}
    if any(len(values) != len(set(values)) for values in ids.values()):
        raise ValueError("Duplicate sample ID within a split")
    if set(ids["train"]) & set(ids["val"]) or set(ids["train"]) & set(ids["test"]) or set(ids["val"]) & set(ids["test"]):
        raise ValueError("Train/validation/test overlap")


class LazyDiffusion(nn.Module):
    """Sparse symmetric normalized diffusion with positive relation weights."""

    def __init__(self):
        super().__init__()
        self.log_relation_weight = nn.Parameter(torch.zeros(2))

    def forward(self, x, edge_index, edge_type, edge_gap):
        u, v = edge_index
        relation_weight = torch.nn.functional.softplus(self.log_relation_weight)
        gap_weight = torch.where(edge_type == 0, torch.ones_like(edge_gap, dtype=x.dtype),
                                 edge_gap.float().clamp_min(1).reciprocal())
        weight = relation_weight[edge_type] * gap_weight
        degree = x.new_zeros(x.shape[0])
        degree.index_add_(0, v, weight)
        inv = degree.clamp_min(1e-12).rsqrt()
        neighbor = x.new_zeros(x.shape)
        neighbor.index_add_(0, v, (weight * inv[u] * inv[v]).unsqueeze(1) * x[u])
        return 0.5 * (x + neighbor)


class GraphWaveletLayer(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.diffusion = LazyDiffusion()
        self.filters = nn.ModuleList(nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(4))
        self.bias = nn.Parameter(torch.zeros(WIDTH))

    def forward(self, x, edge_index, edge_type, edge_gap):
        step = lambda h: self.diffusion(h, edge_index, edge_type, edge_gap)
        p1 = step(x)
        if self.variant == "multiscale":
            p2 = step(p1)
            p4 = step(step(p2))
            bands = (x - p1, p1 - p2, p2 - p4, p4)
        else:
            # Same four matrices and parameter count, but no scale diversity.
            bands = (p1, p1, p1, p1)
        return sum(linear(band) for linear, band in zip(self.filters, bands)) + self.bias


class GraphWaveletClassifier(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.input = nn.Linear(22, WIDTH)
        self.wavelet1 = GraphWaveletLayer(variant)
        self.wavelet2 = GraphWaveletLayer(variant)
        self.norm1 = nn.BatchNorm1d(WIDTH)
        self.norm2 = nn.BatchNorm1d(WIDTH)
        self.dropout = nn.Dropout(0.30)
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Dropout(0.30), nn.Linear(WIDTH, 2))

    def forward(self, graph):
        x = torch.relu(self.input(graph.x.float()))
        x = self.dropout(torch.relu(self.norm1(self.wavelet1(x, graph.edge_index, graph.edge_type, graph.edge_time_gap))))
        x = self.dropout(torch.relu(self.norm2(self.wavelet2(x, graph.edge_index, graph.edge_type, graph.edge_time_gap))))
        return self.head(global_mean_pool(x, graph.batch))


def loader(graphs, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    labels, scores, ids = [], [], []
    for graph in batches:
        graph = graph.to(device)
        scores.extend(torch.softmax(model(graph), dim=1)[:, 1].cpu().tolist())
        labels.extend(graph.y.view(-1).cpu().tolist())
        ids.extend(str(value) for value in graph.sample_id)
    return np.asarray(labels, dtype=np.int64), np.asarray(scores), ids


def threshold_for(labels, scores):
    grid = np.linspace(0.05, 0.95, 181)
    return float(max(grid, key=lambda threshold: balanced_accuracy_score(labels, scores >= threshold)))


def metrics(labels, scores, threshold):
    decisions = scores >= threshold
    return {"auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
            "macro_f1": float(f1_score(labels, decisions, average="macro")), "threshold": threshold}


def save_predictions(path, ids, labels, scores, threshold):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "label", "prob_hallucinated", "prediction", "threshold"))
        for sample_id, label, score in zip(ids, labels, scores):
            writer.writerow((sample_id, int(label), float(score), int(score >= threshold), threshold))
    os.replace(tmp, path)


def smoke_test(device):
    set_seed(42)
    graphs = []
    for split in ("train", "val"):
        source = torch.load(sorted((GRAPH_ROOT / split).glob("*.pt"))[0], map_location="cpu", weights_only=False)
        for cutoff in (0, 4):
            graph = filter_graph(source, cutoff)
            validate_graph(graph, source, split, cutoff)
            graphs.append(graph)
    batch = next(iter(DataLoader(graphs, batch_size=4))).to(device)
    for variant in VARIANTS:
        model = GraphWaveletClassifier(variant).to(device)
        model.train()
        output = model(batch)
        loss = nn.CrossEntropyLoss()(output, batch.y.view(-1))
        loss.backward()
        if output.shape != (4, 2) or not torch.isfinite(output).all() or not torch.isfinite(loss):
            raise RuntimeError(f"{variant}: nonfinite output")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"{variant}: nonfinite gradient")
        # The filter bank telescopes exactly to the original signal.
        if variant == "multiscale":
            x = torch.randn(batch.num_nodes, WIDTH, device=device)
            op = model.wavelet1.diffusion
            p1 = op(x, batch.edge_index, batch.edge_type, batch.edge_time_gap)
            p2 = op(p1, batch.edge_index, batch.edge_type, batch.edge_time_gap)
            p4 = op(op(p2, batch.edge_index, batch.edge_type, batch.edge_time_gap), batch.edge_index, batch.edge_type, batch.edge_time_gap)
            if not torch.allclose((x-p1)+(p1-p2)+(p2-p4)+p4, x, atol=1e-5):
                raise RuntimeError("Wavelet band reconstruction failed")
    print("Smoke test passed: multiscale and single-scale models, finite gradients, wavelet reconstruction", flush=True)


def run_one(datasets, coverage, experiment, variant, seed, device, overwrite):
    name = f"{experiment}_{variant}_seed{seed}"
    result_path = RESULT_ROOT / f"{name}.json"
    model_path = MODEL_ROOT / f"{name}.pt"
    val_path = RESULT_ROOT / f"{name}_val_predictions.csv"
    test_path = RESULT_ROOT / f"{name}_test_predictions.csv"
    if not overwrite and all(path.exists() for path in (result_path, model_path, val_path, test_path)):
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if (result["experiment"], result["variant"], result["seed"]) != (experiment, variant, seed):
            raise ValueError(f"{name}: saved result metadata mismatch")
        print(f"Skipping {name}: AUROC={result['test']['auroc']:.4f}", flush=True)
        return result
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_loader = loader(datasets["train"], True, seed)
    val_loader = loader(datasets["val"])
    test_loader = loader(datasets["test"])
    class_counts = np.bincount([int(graph.y.item()) for graph in datasets["train"]], minlength=2)
    weights = torch.as_tensor(len(datasets["train"]) / (2 * class_counts), device=device, dtype=torch.float32)
    model = GraphWaveletClassifier(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_auc, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for graph in train_loader:
            graph = graph.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(graph), graph.y.view(-1).long())
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: nonfinite loss at epoch {epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"{name}: nonfinite gradient at epoch {epoch}")
            optimizer.step()
            total_loss += loss.item() * graph.num_graphs
        val_y, val_scores, val_ids = predict(model, val_loader, device)
        val_auc = float(roc_auc_score(val_y, val_scores))
        history.append({"epoch": epoch, "train_loss": total_loss / len(datasets["train"]), "val_auroc": val_auc})
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            tmp = model_path.with_suffix(".pt.tmp")
            torch.save({"model_state_dict": model.state_dict(), "experiment": experiment,
                        "variant": variant, "seed": seed, "epoch": epoch, "val_auroc": val_auc}, tmp)
            os.replace(tmp, model_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name} epoch={epoch} val_AUROC={val_auc:.4f} best={best_auc:.4f}", flush=True)
        if stale >= PATIENCE:
            break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
    val_y, val_scores, val_ids = predict(model, val_loader, device)
    threshold = threshold_for(val_y, val_scores)
    test_y, test_scores, test_ids = predict(model, test_loader, device)
    for split, labels, ids in (("val", val_y, val_ids), ("test", test_y, test_ids)):
        if len(ids) != len(datasets[split]) or ids != [g.sample_id for g in datasets[split]]:
            raise ValueError(f"{name}: {split} prediction IDs misaligned")
        if not np.array_equal(labels, [int(g.y.item()) for g in datasets[split]]):
            raise ValueError(f"{name}: {split} prediction labels misaligned")
    save_predictions(val_path, val_ids, val_y, val_scores, threshold)
    save_predictions(test_path, test_ids, test_y, test_scores, threshold)
    runtime = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    result = {"experiment": experiment, "variant": variant, "seed": seed,
              "best_epoch": best_epoch, "epochs_trained": epoch, "best_val_auroc": best_auc,
              "validation": metrics(val_y, val_scores, threshold), "test": metrics(test_y, test_scores, threshold),
              "counts": {split: len(graphs) for split, graphs in datasets.items()},
              "checkpoint_coverage": coverage, "runtime_seconds": runtime,
              "peak_gpu_memory_mib": peak_mib, "checkpoint": str(model_path),
              "val_predictions": str(val_path), "test_predictions": str(test_path), "history": history}
    atomic_json(result, result_path)
    print(f"{name}: test_AUROC={result['test']['auroc']:.4f}, runtime={runtime:.1f}s, peak_gpu_mib={peak_mib}", flush=True)
    return result


def summarize(runs, device):
    result = {}
    for experiment in EXPERIMENTS:
        result[experiment] = {}
        for variant in VARIANTS:
            selected = [run for run in runs if run["experiment"] == experiment and run["variant"] == variant]
            if not selected:
                continue
            entry = {"per_seed": selected, "seeds": [run["seed"] for run in selected]}
            for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
                values = np.asarray([run["test"][metric] for run in selected])
                entry[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
            result[experiment][variant] = entry
    for experiment in EXPERIMENTS:
        if all(v in result[experiment] for v in VARIANTS):
            result[experiment]["multiscale_minus_single_scale_auroc"] = (
                result[experiment]["multiscale"]["auroc"]["mean"] - result[experiment]["single_scale"]["auroc"]["mean"])
    atomic_json({"module": 28, "device": str(device), "filter": "P=(I+D^-1/2 A D^-1/2)/2; bands I-P, P-P^2, P^2-P^4, P^4",
                 "edge_weights": "positive learned depth/temporal weights; temporal divided by time gap",
                 "ablation": "four repeated P features with identical parameter count", "baselines": BASELINES,
                 "results": result}, RESULT_ROOT / "summary.json")
    with (RESULT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("experiment", "variant", "metric", "mean", "sd"))
        for experiment in EXPERIMENTS:
            for variant in VARIANTS:
                if variant in result[experiment]:
                    for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
                        value = result[experiment][variant][metric]
                        writer.writerow((experiment, variant, metric, value["mean"], value["sd"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
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
    runs = []
    for experiment, cutoff in EXPERIMENTS.items():
        datasets, coverage = {}, {}
        for split in COUNTS:
            datasets[split], coverage[split] = load_graphs(split, cutoff)
        validate_splits(datasets)
        for variant in VARIANTS:
            for seed in args.seeds:
                runs.append(run_one(datasets, coverage, experiment, variant, seed, device, args.overwrite))
        summarize(runs, device)
    summarize(runs, device)


if __name__ == "__main__":
    main()
