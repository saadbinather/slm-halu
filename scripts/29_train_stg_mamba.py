"""STG-Mamba-inspired transition classifier with a custom selective SSM.

The standard mamba_ssm/causal_conv1d packages are unavailable in this Windows
environment. This script uses the Mamba selective SSM equations with input-
dependent delta, B and C and exact diagonal zero-order-hold discretization.
It does not claim to reproduce the fused official Mamba block/kernel.
"""

import argparse
import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm


# Reuse Module 28's graph filtering and evaluation helpers without changing it.
_spec = importlib.util.spec_from_file_location("epigraph_module28", Path(__file__).with_name("28_train_graph_wavelet.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

GRAPH_ROOT = Path("processed_graphs/transitions_pca16")
MODEL_ROOT = Path("models/stg_mamba")
RESULT_ROOT = Path("results/stg_mamba")
COUNTS = {"train": 1858, "val": 398, "test": 399}
EXPERIMENTS = {"t0": 0, "temporal_t4": 4}
VARIANTS = {"t0": ("spatial_only", "pointwise_only"),
            "temporal_t4": ("full", "no_spatial", "no_temporal")}
SEEDS = (42, 43, 44, 45, 46)
TIMES = (0, 1, 2, 4)
WIDTH = 32
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
BASELINES = {"t0": {"mlp": 0.8097160603371784, "graphsage": 0.8071146245059289,
                     "gcn": 0.8105106074050173, "neural_sheaf": 0.7922239251431799},
             "temporal_t4": {"mlp": 0.8165685246430587, "graphsage": 0.8155924820521093,
                             "gcn": 0.8142695813503267, "neural_sheaf": 0.8169718480277487}}


def load_graphs(split, cutoff):
    files = sorted((GRAPH_ROOT / split).glob("*.pt"))
    if len(files) != COUNTS[split]:
        raise ValueError(f"{split}: expected {COUNTS[split]} graphs, found {len(files)}")
    graphs = []
    coverage = {time: 0 for time in TIMES}
    for path in tqdm(files, desc=f"Load {split} t<={cutoff}", mininterval=2):
        source = torch.load(path, map_location="cpu", weights_only=False)
        saved_times = set(int(t) for t in source.original_time_positions.tolist())
        raw_mask = source.checkpoint_availability_mask.tolist()
        if len(raw_mask) != 5 or raw_mask != [t in saved_times for t in (0, 1, 2, 4, 8)]:
            raise ValueError(f"{path}: checkpoint availability metadata mismatch")
        graph = base.filter_graph(source, cutoff)
        base.validate_graph(graph, source, split, cutoff)
        times = graph.time_position.tolist()
        if times != sorted(times) or any(times.count(t) not in (0, 27) for t in TIMES):
            raise ValueError(f"{path}: checkpoint ordering or layer count mismatch")
        if graph.transition_index.tolist() != list(range(27)) * (graph.num_nodes // 27):
            raise ValueError(f"{path}: transition nodes are not in layer order")
        if len(set(zip(times, graph.transition_index.tolist()))) != graph.num_nodes:
            raise ValueError(f"{path}: duplicate time/layer node")
        for t in TIMES:
            coverage[t] += int(t in times)
        graphs.append(graph)
    return graphs, coverage


class SpatialDepth(nn.Module):
    """Two depth-only normalized neighbor layers; pointwise control shares weights."""

    def __init__(self, graph_aware):
        super().__init__()
        self.graph_aware = graph_aware
        self.self_layers = nn.ModuleList(nn.Linear(WIDTH, WIDTH) for _ in range(2))
        self.neighbor_layers = nn.ModuleList(nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in range(2))

    def forward(self, x, edge_index, edge_type):
        for self_linear, neighbor_linear, norm in zip(self.self_layers, self.neighbor_layers, self.norms):
            if self.graph_aware:
                u, v = edge_index[:, edge_type == 0]
                neighbors = torch.zeros_like(x)
                neighbors.index_add_(0, v, x[u])
                degree = x.new_zeros((x.shape[0], 1))
                degree.index_add_(0, v, torch.ones((len(v), 1), device=x.device))
                neighbors = neighbors / degree.clamp_min(1)
            else:
                neighbors = x  # Same trained matrices, without graph messages.
            x = torch.relu(norm(self_linear(x) + neighbor_linear(neighbors)))
        return x


class SelectiveTemporalSSM(nn.Module):
    """Diagonal selective SSM, with input-dependent delta/B/C and causal scan.

    A=-exp(log_A)<0; a_t=exp(A*delta_t*gap_t);
    b_t=((a_t-1)/A)*B(x_t); s_t=a_t*s_(t-1)+b_t*x_t;
    y_t=sigmoid(z(x_t))*(C(x_t)*s_t+D*x_t).
    Missing checkpoints do not update state or yield graph nodes.
    """

    def __init__(self):
        super().__init__()
        self.delta = nn.Linear(WIDTH, WIDTH)
        self.B = nn.Linear(WIDTH, WIDTH)
        self.C = nn.Linear(WIDTH, WIDTH)
        self.z = nn.Linear(WIDTH, WIDTH)
        self.log_A = nn.Parameter(torch.full((WIDTH,), -1.0))
        self.D = nn.Parameter(torch.ones(WIDTH))
        self.output = nn.Linear(WIDTH, WIDTH)

    def forward(self, x, mask):
        # x [graphs, layers, checkpoints, width]; mask [graphs, layers, checkpoints].
        state = torch.zeros_like(x[:, :, 0, :])
        last_time = x.new_zeros(x.shape[:2])
        outputs = []
        A = -torch.exp(self.log_A)
        for slot, time_value in enumerate(TIMES):
            current = x[:, :, slot, :]
            valid = mask[:, :, slot]
            gap = (time_value - last_time).clamp_min(1).unsqueeze(-1)
            delta = torch.nn.functional.softplus(self.delta(current)) * gap
            a_bar = torch.exp(A * delta)
            b_bar = torch.expm1(A * delta) / A * self.B(current)
            candidate = a_bar * state + b_bar * current
            state = torch.where(valid.unsqueeze(-1), candidate, state)
            last_time = torch.where(valid, torch.full_like(last_time, time_value), last_time)
            output = torch.sigmoid(self.z(current)) * (self.C(current) * state + self.D * current)
            outputs.append(torch.where(valid.unsqueeze(-1), self.output(output), torch.zeros_like(output)))
        return torch.stack(outputs, dim=2)


def pack_sequences(x, graph):
    """Temporary masked workspace; padded cells never become graph nodes."""
    graph_id = graph.batch
    # PyG increments attributes ending in "_index" during batching. The
    # validated source stores 27 ordered layer nodes per checkpoint, so derive
    # positions from node order instead of the collated transition_index.
    layer = torch.arange(x.shape[0], device=x.device) % 27
    slot = torch.searchsorted(torch.tensor(TIMES, device=x.device), graph.time_position.long())
    if not torch.all(torch.tensor(TIMES, device=x.device)[slot] == graph.time_position):
        raise ValueError("Unexpected checkpoint position")
    packed = x.new_zeros((graph.num_graphs, 27, len(TIMES), WIDTH))
    mask = torch.zeros((graph.num_graphs, 27, len(TIMES)), device=x.device, dtype=torch.bool)
    packed[graph_id, layer, slot] = x
    mask[graph_id, layer, slot] = True
    return packed, mask, (graph_id, layer, slot)


class STGMambaClassifier(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.input = nn.Linear(22, WIDTH)
        self.spatial = SpatialDepth(variant not in ("no_spatial", "pointwise_only"))
        self.temporal = SelectiveTemporalSSM() if variant in ("full", "no_spatial") else None
        self.gate = nn.Linear(2 * WIDTH, WIDTH) if self.temporal is not None else None
        self.dropout = nn.Dropout(0.30)
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Dropout(0.30), nn.Linear(WIDTH, 2))

    def forward(self, graph):
        x = torch.relu(self.input(graph.x.float()))
        spatial = self.spatial(x, graph.edge_index, graph.edge_type)
        if self.temporal is None:
            fused = spatial
        else:
            packed, mask, index = pack_sequences(spatial, graph)
            temporal = self.temporal(packed, mask)[index]
            gate = torch.sigmoid(self.gate(torch.cat((spatial, temporal), dim=-1)))
            fused = gate * spatial + (1 - gate) * temporal
        return self.head(global_mean_pool(self.dropout(fused), graph.batch))


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


def smoke_test(device):
    base.set_seed(42)
    examples = []
    for split in ("train", "val"):
        source = torch.load(sorted((GRAPH_ROOT / split).glob("*.pt"))[0], map_location="cpu", weights_only=False)
        for cutoff in (0, 4):
            graph = base.filter_graph(source, cutoff)
            base.validate_graph(graph, source, split, cutoff)
            examples.append(graph)
    batch = next(iter(DataLoader(examples, batch_size=4))).to(device)
    for variant in ("full", "no_spatial", "no_temporal", "spatial_only", "pointwise_only"):
        model = STGMambaClassifier(variant).to(device)
        output = model(batch)
        loss = nn.CrossEntropyLoss()(output, batch.y.view(-1))
        loss.backward()
        if output.shape != (4, 2) or not torch.isfinite(output).all() or not torch.isfinite(loss):
            raise RuntimeError(f"{variant}: invalid output")
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"{variant}: missing or nonfinite gradient")
    # Selection and gap length must alter the recurrence, while masked cells
    # cannot affect outputs at real checkpoints.
    ssm = SelectiveTemporalSSM().to(device)
    sample = torch.randn((1, 27, 4, WIDTH), device=device)
    mask = torch.tensor([True, True, False, True], device=device).view(1, 1, 4).expand(1, 27, 4)
    reference = ssm(sample, mask)
    changed_missing = sample.clone()
    changed_missing[:, :, 2] += 1000
    if not torch.allclose(reference, ssm(changed_missing, mask), atol=1e-5):
        raise RuntimeError("Missing checkpoint affected SSM output")
    if torch.allclose(reference, ssm(sample * 1.01, mask)):
        raise RuntimeError("Selective SSM ignored input")
    if torch.allclose(reference, ssm(sample, torch.ones_like(mask))):
        raise RuntimeError("SSM ignored checkpoint mask/gap")
    print("Smoke test passed: five variants, finite gradients, selective recurrence, temporal masks", flush=True)


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
            raise ValueError(f"{name}: saved metadata mismatch")
        print(f"Skipping {name}: AUROC={result['test']['auroc']:.4f}", flush=True)
        return result
    base.set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_loader = loader(datasets["train"], True, seed)
    val_loader = loader(datasets["val"])
    test_loader = loader(datasets["test"])
    class_counts = np.bincount([int(graph.y.item()) for graph in datasets["train"]], minlength=2)
    weights = torch.as_tensor(len(datasets["train"]) / (2 * class_counts), device=device, dtype=torch.float32)
    model = STGMambaClassifier(variant).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
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
        auc = float(roc_auc_score(val_y, val_scores))
        history.append({"epoch": epoch, "train_loss": total_loss / len(datasets["train"]), "val_auroc": auc})
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            tmp = model_path.with_suffix(".pt.tmp")
            torch.save({"model_state_dict": model.state_dict(), "experiment": experiment,
                        "variant": variant, "seed": seed, "epoch": epoch, "val_auroc": auc}, tmp)
            os.replace(tmp, model_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name} epoch={epoch} val_AUROC={auc:.4f} best={best_auc:.4f}", flush=True)
        if stale >= PATIENCE:
            break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
    val_y, val_scores, val_ids = predict(model, val_loader, device)
    threshold = base.threshold_for(val_y, val_scores)
    test_y, test_scores, test_ids = predict(model, test_loader, device)
    for split, labels, ids in (("val", val_y, val_ids), ("test", test_y, test_ids)):
        if len(ids) != len(datasets[split]) or ids != [g.sample_id for g in datasets[split]]:
            raise ValueError(f"{name}: {split} IDs misaligned")
        if not np.array_equal(labels, [int(g.y.item()) for g in datasets[split]]):
            raise ValueError(f"{name}: {split} labels misaligned")
    base.save_predictions(val_path, val_ids, val_y, val_scores, threshold)
    base.save_predictions(test_path, test_ids, test_y, test_scores, threshold)
    runtime = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    result = {"experiment": experiment, "variant": variant, "seed": seed,
              "architecture": "custom_input_selective_diagonal_ssm", "parameter_count": parameter_count,
              "temporal_active": experiment != "t0" and model.temporal is not None,
              "input_dim": 22, "hidden_dim": WIDTH, "batch_size": BATCH_SIZE,
              "learning_rate": 0.001, "weight_decay": 0.0001, "dropout": 0.30,
              "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
              "best_epoch": best_epoch, "epochs_trained": epoch, "best_val_auroc": best_auc,
              "validation": base.metrics(val_y, val_scores, threshold),
              "test": base.metrics(test_y, test_scores, threshold),
              "counts": {split: len(graphs) for split, graphs in datasets.items()},
              "checkpoint_coverage": coverage, "runtime_seconds": runtime,
              "peak_gpu_memory_mib": peak_mib, "checkpoint": str(model_path),
              "val_predictions": str(val_path), "test_predictions": str(test_path), "history": history}
    base.atomic_json(result, result_path)
    print(f"{name}: test_AUROC={result['test']['auroc']:.4f}, runtime={runtime:.1f}s, peak_gpu_mib={peak_mib}", flush=True)
    return result


def summarize(runs, device):
    grouped = {}
    for experiment in EXPERIMENTS:
        grouped[experiment] = {}
        for variant in VARIANTS[experiment]:
            selected = [r for r in runs if r["experiment"] == experiment and r["variant"] == variant]
            if not selected:
                continue
            entry = {"seeds": [r["seed"] for r in selected], "per_seed": selected,
                     "parameter_count": selected[0]["parameter_count"]}
            for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
                values = np.asarray([r["test"][metric] for r in selected])
                entry[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
            grouped[experiment][variant] = entry
    base.atomic_json({"module": 29, "device": str(device), "implementation":
                      "custom selective diagonal SSM (not official fused Mamba); input-dependent delta/B/C, exact ZOH, gap-scaled delta, masked scan",
                      "baseline_auroc": BASELINES, "results": grouped}, RESULT_ROOT / "summary.json")
    with (RESULT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("experiment", "variant", "metric", "mean", "sd"))
        for experiment, variants in grouped.items():
            for variant, item in variants.items():
                for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
                    writer.writerow((experiment, variant, metric, item[metric]["mean"], item[metric]["sd"]))


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
        base.validate_splits(datasets)
        for variant in VARIANTS[experiment]:
            for seed in args.seeds:
                runs.append(run_one(datasets, coverage, experiment, variant, seed, device, args.overwrite))
        summarize(runs, device)
    summarize(runs, device)


if __name__ == "__main__":
    main()
