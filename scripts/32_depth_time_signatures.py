"""Depth and checkpoint-time path signatures from saved PCA-16 graphs.

Reuses Module 30's train-fitted 16->5 PCA, node scaler, degree-3 signature,
training convention, and exact depth-only results. All paths use only saved
non-EOS nodes at t=0,1,2,4. No missing state is padded into a signature.
"""

import argparse
import csv
import importlib.util
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_spec = importlib.util.spec_from_file_location("epigraph_module30", Path(__file__).with_name("30_train_path_signature.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

GRAPH_ROOT = Path("processed_graphs/transitions_pca16")
MODEL_ROOT = Path("models/depth_time_signatures")
RESULT_ROOT = Path("results/depth_time_signatures")
OLD_MODEL_ROOT = Path("models/path_signature")
OLD_RESULT_ROOT = Path("results/path_signature")
SPLITS = ("train", "val", "test")
COUNTS = {"train": 1858, "val": 398, "test": 399}
TIMES = (0, 1, 2, 4)
SEEDS = (42, 43, 44, 45, 46)
CONFIGS = ("depth_only", "temporal_only", "fusion", "fusion_no_temporal",
           "availability_only", "subset_true", "subset_shuffled")
NEW_CONFIGS = CONFIGS[1:]
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
BOOTSTRAP_REPLICATES = 2000
BASELINES = {"depth_t0_order3": 0.811284988303622,
             "transition_mlp_t4": 0.8165685246430587,
             "neural_sheaf_t4": 0.8169718480277487,
             "original_layer_time_graph_t4_user_reference": 0.7907}


def require_existing_projection():
    pca_path = OLD_MODEL_ROOT / "train_projection_16_to_5.joblib"
    scaler_path = OLD_MODEL_ROOT / "projected_node_scaler.joblib"
    if not pca_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Module 30 train projection/scaler missing; refusing to refit PCA")
    pca, scaler = joblib.load(pca_path), joblib.load(scaler_path)
    if getattr(pca, "training_split_", None) != "train" or getattr(scaler, "training_split_", None) != "train":
        raise ValueError("Module 30 projection lacks train-only provenance")
    if int(pca.n_features_in_) != 16 or int(pca.n_components_) != 5 or int(scaler.n_features_in_) != 5:
        raise ValueError("Module 30 projection dimensions mismatch")
    return pca, scaler


def load_graph_data():
    data, seen = {}, set()
    for split in SPLITS:
        files = sorted((GRAPH_ROOT / split).glob("*.pt"))
        if len(files) != COUNTS[split]:
            raise ValueError(f"{split}: expected {COUNTS[split]} files, found {len(files)}")
        samples, labels, ids, lengths = [], [], [], []
        for path in tqdm(files, desc=f"Load {split} t<=4", mininterval=2):
            graph = torch.load(path, map_location="cpu", weights_only=False)
            sample_id = str(graph.sample_id)
            if graph.split != split or sample_id in seen:
                raise ValueError(f"{path}: split mismatch or repeated ID")
            seen.add(sample_id)
            saved_times = [int(t) for t in graph.original_time_positions.tolist()]
            eos_times = {t for t, flag in zip(saved_times, graph.checkpoint_is_eos.tolist()) if flag == 1}
            if graph.checkpoint_availability_mask.tolist() != [t in saved_times for t in (0, 1, 2, 4, 8)]:
                raise ValueError(f"{path}: saved availability mask mismatch")
            keep = graph.time_position <= 4
            selected_times = graph.time_position[keep].tolist()
            times = sorted(set(selected_times))
            if not times or times[0] != 0 or any(t not in TIMES for t in times) or eos_times.intersection(times):
                raise ValueError(f"{path}: invalid time or EOS checkpoint")
            if selected_times != sorted(selected_times) or any(selected_times.count(t) != 27 for t in times):
                raise ValueError(f"{path}: missing or unordered transition nodes")
            if graph.transition_index[keep].tolist() != list(range(27)) * len(times):
                raise ValueError(f"{path}: layer order mismatch")
            x = graph.x[keep].float().numpy().reshape(len(times), 27, 22)
            if not np.isfinite(x).all() or not np.allclose(x[:, :, 20], np.linspace(0, 1, 27)[None, :], atol=1e-6):
                raise ValueError(f"{path}: invalid features or layer positions")
            if not np.allclose(x[:, :, 21], np.asarray(times)[:, None] / 8.0, atol=1e-6):
                raise ValueError(f"{path}: saved time-position feature mismatch")
            label = int(graph.y.item())
            if label not in (0, 1):
                raise ValueError(f"{path}: invalid label")
            samples.append({"times": times, "nodes": x})
            labels.append(label)
            ids.append(sample_id)
            lengths.append(len(times))
        data[split] = {"samples": samples, "labels": np.asarray(labels, dtype=np.int64),
                       "ids": ids, "lengths": np.asarray(lengths, dtype=np.int64)}
    return data


def compute_signature_features(data, pca, node_scaler):
    features = {}
    for split in SPLITS:
        samples = data[split]["samples"]
        count = len(samples)
        depth0 = np.empty((count, 258), dtype=np.float64)
        depth_mean = np.empty_like(depth0)
        temporal = np.empty_like(depth0)
        temporal_shuffled = np.empty_like(depth0)
        for length in (1, 2, 3, 4):
            indices = np.flatnonzero(data[split]["lengths"] == length)
            for start in tqdm(range(0, len(indices), 128), desc=f"Signatures {split} length={length}", leave=False):
                chosen = indices[start:start + 128]
                if not len(chosen):
                    continue
                nodes = np.stack([samples[i]["nodes"] for i in chosen])  # [batch, time, layer, 22]
                times = np.stack([samples[i]["times"] for i in chosen])
                projected = node_scaler.transform(pca.transform(nodes[:, :, :, :16].reshape(-1, 16)))
                projected = projected.reshape(len(chosen), length, 27, 5)
                # Depth path: 27 true layer positions at each available time.
                depth_path = np.concatenate((projected, nodes[:, :, :, 20:21]), axis=-1)
                depth_sig = base.signature(depth_path.reshape(-1, 27, 6), 3).numpy().reshape(len(chosen), length, 258)
                depth0[chosen] = depth_sig[:, 0]
                depth_mean[chosen] = depth_sig.mean(axis=1)
                # Temporal path: actual checkpoint positions (0,1,2,4)/4;
                # missing checkpoints never enter these variable-length paths.
                coordinates = np.transpose(projected, (0, 2, 1, 3))
                time_coordinate = np.broadcast_to((times / 4.0)[:, None, :, None], (len(chosen), 27, length, 1))
                temporal_path = np.concatenate((coordinates, time_coordinate), axis=-1)
                temporal_sig = base.signature(temporal_path.reshape(-1, length, 6), 3).numpy().reshape(len(chosen), 27, 258)
                temporal[chosen] = temporal_sig.mean(axis=1)
                if length >= 3:
                    # Deterministic fixed shuffle of post-t=0 checkpoints;
                    # t=0 remains the anchor and time travels with its state.
                    permutation = [0, 2, 1] if length == 3 else [0, 2, 3, 1]
                    shuffled = temporal_path[:, :, permutation, :].copy()
                    shuffled_sig = base.signature(shuffled.reshape(-1, length, 6), 3).numpy().reshape(len(chosen), 27, 258)
                    temporal_shuffled[chosen] = shuffled_sig.mean(axis=1)
                else:
                    temporal_shuffled[chosen] = np.nan  # excluded from shuffle control
        if not all(np.isfinite(array).all() for array in (depth0, depth_mean, temporal)):
            raise ValueError(f"{split}: nonfinite signature")
        features[split] = {"depth0": depth0, "depth_mean": depth_mean,
                           "temporal": temporal, "temporal_shuffled": temporal_shuffled,
                           "availability": np.stack([[int(t in sample["times"]) for t in (1, 2, 4)] for sample in samples]).astype(np.float32)}
    return features


def train_scaler(name, train_values):
    path = MODEL_ROOT / f"{name}_train_scaler.joblib"
    if path.exists():
        scaler = joblib.load(path)
        if getattr(scaler, "training_split_", None) != "train" or scaler.n_features_in_ != train_values.shape[1]:
            raise ValueError(f"{path}: invalid existing scaler")
    else:
        scaler = StandardScaler().fit(train_values)
        scaler.training_split_ = "train"
        scaler.training_examples_ = train_values.shape[0]
        temporary = path.with_suffix(".joblib.tmp")
        joblib.dump(scaler, temporary)
        os.replace(temporary, path)
    return scaler


def prepare_inputs(data, raw):
    subset = {split: np.flatnonzero(data[split]["lengths"] >= 3) for split in SPLITS}
    scalers = {
        "depth": train_scaler("depth_mean", raw["train"]["depth_mean"]),
        "temporal": train_scaler("temporal", raw["train"]["temporal"]),
        "subset_depth": train_scaler("subset_depth", raw["train"]["depth_mean"][subset["train"]]),
        "subset_temporal": train_scaler("subset_temporal", raw["train"]["temporal"][subset["train"]]),
        "subset_shuffled": train_scaler("subset_shuffled", raw["train"]["temporal_shuffled"][subset["train"]]),
    }
    inputs = {}
    for config in NEW_CONFIGS:
        inputs[config] = {}
        for split in SPLITS:
            indices = subset[split] if config.startswith("subset_") else np.arange(COUNTS[split])
            depth = raw[split]["depth_mean"][indices]
            temporal = raw[split]["temporal"][indices]
            if config == "temporal_only":
                value = (scalers["temporal"].transform(temporal).astype(np.float32),)
            elif config in ("fusion", "fusion_no_temporal"):
                d = scalers["depth"].transform(depth).astype(np.float32)
                t = scalers["temporal"].transform(temporal).astype(np.float32) if config == "fusion" else d.copy()
                value = (d, t)
            elif config == "availability_only":
                value = (raw[split]["availability"].astype(np.float32),)
            else:
                d = scalers["subset_depth"].transform(depth).astype(np.float32)
                temporal_raw = raw[split]["temporal_shuffled"][indices] if config == "subset_shuffled" else temporal
                key = "subset_shuffled" if config == "subset_shuffled" else "subset_temporal"
                t = scalers[key].transform(temporal_raw).astype(np.float32)
                value = (d, t)
            if any(not np.isfinite(array).all() for array in value):
                raise ValueError(f"{config}/{split}: nonfinite inputs")
            inputs[config][split] = {"x": value, "indices": indices,
                                     "labels": data[split]["labels"][indices],
                                     "ids": [data[split]["ids"][i] for i in indices]}
    return inputs, subset


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = nn.Sequential(nn.Linear(258, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.30))
        self.temporal = nn.Sequential(nn.Linear(258, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.30))
        self.gate = nn.Linear(128, 64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.30), nn.Linear(32, 2))

    def forward(self, depth, temporal):
        d, t = self.depth(depth), self.temporal(temporal)
        gate = torch.sigmoid(self.gate(torch.cat((d, t), dim=1)))
        return self.head(gate * d + (1 - gate) * t)


class AvailabilityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


def make_model(config):
    if config == "temporal_only":
        return base.SignatureMLP(258)
    if config == "availability_only":
        return AvailabilityMLP()
    return GatedFusion()


@torch.no_grad()
def predict(model, part, device):
    model.eval()
    arrays = [torch.from_numpy(value) for value in part["x"]]
    scores = []
    for batch in DataLoader(TensorDataset(*arrays), batch_size=BATCH_SIZE):
        scores.extend(torch.softmax(model(*(value.to(device) for value in batch)), dim=1)[:, 1].cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def run_one(inputs, config, seed, device, overwrite):
    name = f"{config}_seed{seed}"
    result_path = RESULT_ROOT / f"{name}.json"
    model_path = MODEL_ROOT / f"{name}.pt"
    val_path = RESULT_ROOT / f"{name}_val_predictions.csv"
    test_path = RESULT_ROOT / f"{name}_test_predictions.csv"
    if not overwrite and all(path.exists() for path in (result_path, model_path, val_path, test_path)):
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result["config"] != config or result["seed"] != seed:
            raise ValueError(f"{result_path}: saved run mismatch")
        print(f"Skipping {name}: AUROC={result['test']['auroc']:.4f}", flush=True)
        return result
    base.seed_all(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = make_model(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    train = inputs[config]["train"]
    counts = np.bincount(train["labels"], minlength=2)
    weights = torch.as_tensor(len(train["labels"]) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    train_loader = DataLoader(TensorDataset(*(torch.from_numpy(value) for value in train["x"]),
                                            torch.from_numpy(train["labels"])),
                              batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_auc, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            *values, labels = batch
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*(value.to(device) for value in values)), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: nonfinite loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"{name}: nonfinite gradient")
            optimizer.step()
            total += loss.item() * len(labels)
        val = inputs[config]["val"]
        auc = float(roc_auc_score(val["labels"], predict(model, val, device)))
        history.append({"epoch": epoch, "train_loss": total / len(train["labels"]), "val_auroc": auc})
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            temporary = model_path.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed,
                        "best_epoch": epoch, "val_auroc": auc}, temporary)
            os.replace(temporary, model_path)
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["state_dict"])
    val, test = inputs[config]["val"], inputs[config]["test"]
    val_scores = predict(model, val, device)
    threshold = base.select_threshold(val["labels"], val_scores)
    test_scores = predict(model, test, device)
    base.save_predictions(val_path, val["ids"], val["labels"], val_scores, threshold)
    base.save_predictions(test_path, test["ids"], test["labels"], test_scores, threshold)
    runtime = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    result = {"config": config, "seed": seed, "parameter_count": parameter_count,
              "best_epoch": best_epoch, "epochs_trained": epoch, "best_val_auroc": best_auc,
              "validation": base.metrics(val["labels"], val_scores, threshold),
              "test": base.metrics(test["labels"], test_scores, threshold),
              "counts": {split: len(inputs[config][split]["labels"]) for split in SPLITS},
              "runtime_seconds": runtime, "peak_gpu_memory_mib": peak_mib,
              "predictions": str(test_path), "checkpoint": str(model_path), "history": history}
    base.atomic_json(result, result_path)
    print(f"{name}: test_AUROC={result['test']['auroc']:.4f}, runtime={runtime:.1f}s", flush=True)
    return result


def load_depth_only(data):
    runs = []
    for seed in SEEDS:
        result_path = OLD_RESULT_ROOT / f"order3_seed{seed}.json"
        prediction_path = OLD_RESULT_ROOT / f"order3_seed{seed}_test_predictions.csv"
        model_path = OLD_MODEL_ROOT / f"order3_seed{seed}.pt"
        if any(not path.exists() for path in (result_path, prediction_path, model_path)):
            raise FileNotFoundError(f"Module 30 order-3 seed {seed} is incomplete")
        with result_path.open(encoding="utf-8") as handle:
            old = json.load(handle)
        with prediction_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if [row["sample_id"] for row in rows] != data["test"]["ids"] or [int(row["label"]) for row in rows] != data["test"]["labels"].tolist():
            raise ValueError(f"{prediction_path}: depth-only test predictions misaligned")
        runs.append({"config": "depth_only", "seed": seed, "test": old["test"],
                     "validation": old["validation"], "parameter_count": old["parameter_count"],
                     "predictions": str(prediction_path), "checkpoint": str(model_path),
                     "reused_from": str(result_path)})
    return runs


def read_predictions(run, expected_ids, expected_labels):
    with Path(run["predictions"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row["sample_id"] for row in rows] != expected_ids or [int(row["label"]) for row in rows] != expected_labels.tolist():
        raise ValueError(f"{run['predictions']}: predictions do not align with test examples")
    return np.asarray([float(row["prob_hallucinated"]) for row in rows])


def bootstrap_paired(labels, left, right, replicates=BOOTSTRAP_REPLICATES):
    rng = np.random.default_rng(32_4242)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    indices = np.concatenate((rng.choice(positive, (replicates, len(positive)), replace=True),
                              rng.choice(negative, (replicates, len(negative)), replace=True)), axis=1)
    labels_boot = labels[indices]
    pos_count, neg_count = len(positive), len(negative)

    def aurocs(values):
        ranked = rankdata(values[indices], axis=1, method="average")
        return ((ranked * labels_boot).sum(axis=1) - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)

    draws = np.mean(np.stack([aurocs(a) - aurocs(b) for a, b in zip(left, right)]), axis=0)
    low, high = np.percentile(draws, [2.5, 97.5])
    observed = np.asarray([roc_auc_score(labels, a) - roc_auc_score(labels, b) for a, b in zip(left, right)])
    return {"per_seed_difference": observed.tolist(), "mean_difference": float(observed.mean()),
            "paired_bootstrap_95_ci": [float(low), float(high)], "reliably_positive": bool(low > 0)}


def compare_and_stratify(data, runs, inputs):
    labels = data["test"]["labels"]
    ids = data["test"]["ids"]
    predictions = {config: np.stack([read_predictions(run, ids, labels) for run in runs[config]])
                   for config in CONFIGS[:5]}
    pairs = (("fusion", "depth_only"), ("fusion", "fusion_no_temporal"),
             ("fusion", "temporal_only"), ("fusion", "availability_only"))
    comparisons = {f"{a}_minus_{b}": bootstrap_paired(labels, predictions[a], predictions[b]) for a, b in pairs}
    subset = inputs["subset_true"]["test"]
    left = np.stack([read_predictions(run, subset["ids"], subset["labels"]) for run in runs["subset_true"]])
    right = np.stack([read_predictions(run, subset["ids"], subset["labels"]) for run in runs["subset_shuffled"]])
    comparisons["subset_true_minus_shuffled"] = bootstrap_paired(subset["labels"], left, right)
    strata = {}
    for length in (2, 3, 4):
        chosen = np.flatnonzero(data["test"]["lengths"] == length)
        y = labels[chosen]
        entry = {"test_examples": len(chosen), "correct": int((y == 0).sum()), "hallucinated": int((y == 1).sum())}
        if len(np.unique(y)) == 2:
            for config in ("depth_only", "temporal_only", "fusion", "fusion_no_temporal", "availability_only"):
                values = np.asarray([roc_auc_score(y, scores[chosen]) for scores in predictions[config]])
                entry[config] = {"auroc_mean": float(values.mean()), "auroc_sd": float(values.std(ddof=1))}
            for alternative in ("depth_only", "fusion_no_temporal"):
                entry[f"fusion_minus_{alternative}"] = bootstrap_paired(
                    y, predictions["fusion"][:, chosen], predictions[alternative][:, chosen]
                )
        else:
            entry["auroc_status"] = "unavailable: one class in stratum"
        strata[str(length)] = entry
    return comparisons, strata


def summarize(data, runs, inputs, device):
    groups = {}
    for config in CONFIGS:
        selected = runs[config]
        entry = {"seeds": [run["seed"] for run in selected], "per_seed": selected,
                 "parameter_count": selected[0]["parameter_count"]}
        for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
            values = np.asarray([run["test"][metric] for run in selected])
            entry[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
        groups[config] = entry
    comparisons, strata = compare_and_stratify(data, runs, inputs)
    lengths = {split: {str(length): int((data[split]["lengths"] == length).sum()) for length in (1, 2, 3, 4)} for split in SPLITS}
    report = {"module": 32, "device": str(device), "counts": COUNTS, "signature_lengths": lengths,
              "projection": "reused Module 30 train-only PCA 16->5 and projected-node scaler",
              "path": "degree-3 Chen signatures, five projected transition coordinates plus normalized layer/time position; only real non-EOS checkpoints",
              "temporal_shuffle": "only K>=3; t=0 fixed first; K=3 perm [0,2,1], K=4 perm [0,2,3,1]",
              "availability_control": "binary availability of t=1,2,4 only",
              "baselines": BASELINES, "aggregate": groups, "paired_comparisons": comparisons,
              "test_by_signature_length": strata}
    base.atomic_json(report, RESULT_ROOT / "summary.json")
    with (RESULT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("config", "metric", "mean", "sd"))
        for config in CONFIGS:
            for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
                item = groups[config][metric]
                writer.writerow((config, metric, item["mean"], item["sd"]))
    return report


def smoke_test():
    base.smoke_test()
    # A short temporal path has 2 real points; a longer one changes the
    # signature, and the time coordinate records actual checkpoint spacing.
    two = np.array([[[1., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0.25]]])
    three = np.array([[[1., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0.25], [2., 1., 0., 0., 0., 0.5]]])
    if base.signature(two, 3).shape != (1, 258) or torch.allclose(base.signature(two, 3), base.signature(three, 3)):
        raise RuntimeError("Temporal signature length/position test failed")
    fusion = GatedFusion()
    output = fusion(torch.randn(4, 258), torch.randn(4, 258))
    output.sum().backward()
    if output.shape != (4, 2) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in fusion.parameters()):
        raise RuntimeError("Fusion output or gradients invalid")
    print("Module 32 smoke test passed: signature lengths, checkpoint coordinate, fusion gradients", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    pca, node_scaler = require_existing_projection()
    data = load_graph_data()
    raw = compute_signature_features(data, pca, node_scaler)
    inputs, _ = prepare_inputs(data, raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {"depth_only": load_depth_only(data)}
    for config in NEW_CONFIGS:
        runs[config] = [run_one(inputs, config, seed, device, args.overwrite) for seed in SEEDS]
    report = summarize(data, runs, inputs, device)
    for config in CONFIGS:
        item = report["aggregate"][config]["auroc"]
        print(f"{config}: AUROC={item['mean']:.4f}+/-{item['sd']:.4f}", flush=True)


if __name__ == "__main__":
    main()
