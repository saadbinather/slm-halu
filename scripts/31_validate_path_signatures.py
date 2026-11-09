"""Order and robustness study for Module 30's t=0 degree-3 signatures.

Reuses Module 30's original, geometry and fixed-permutation-30 runs exactly.
New orderings use the same train-fitted 5D PCA projection, zero basepoint,
degree-3 signature and classifier. Mean pooling is the order-insensitive
capacity-matched control. All comparisons use paired test-example bootstrap.
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

_spec = importlib.util.spec_from_file_location("epigraph_module30", Path(__file__).with_name("30_train_path_signature.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

MODEL_ROOT = Path("models/path_signature_ordering")
RESULT_ROOT = Path("results/path_signature_ordering")
OLD_MODEL_ROOT = Path("models/path_signature")
OLD_RESULT_ROOT = Path("results/path_signature")
SEEDS = (42, 43, 44, 45, 46)
SPLITS = ("train", "val", "test")
FIXED_PERMUTATION_IDS = (30, 31, 32, 33, 34)
CONFIGS = ("original", "reverse", "fixed_perm_30", "fixed_perm_31", "fixed_perm_32",
           "fixed_perm_33", "fixed_perm_34", "per_example", "mean_pool", "geometry")
NEW_CONFIGS = ("reverse", "fixed_perm_31", "fixed_perm_32", "fixed_perm_33",
               "fixed_perm_34", "per_example", "mean_pool")
REUSE = {"original": "order3", "fixed_perm_30": "order3_shuffled", "geometry": "order3_geometry"}
BOOTSTRAP_REPLICATES = 2000


def fixed_permutations():
    permutations = {30: base.SHUFFLE.copy()}
    for number in FIXED_PERMUTATION_IDS[1:]:
        permutations[number] = np.random.default_rng(number).permutation(27)
    values = [tuple(permutation.tolist()) for permutation in permutations.values()]
    if len(set(values)) != len(values) or any(list(value) in (list(range(27)), list(range(26, -1, -1))) for value in values):
        raise ValueError("Fixed permutations must be distinct from each other and from true/reverse order")
    return permutations


def require_existing_preprocessing():
    required = [OLD_MODEL_ROOT / "train_projection_16_to_5.joblib",
                OLD_MODEL_ROOT / "projected_node_scaler.joblib"]
    if any(not path.exists() for path in required):
        raise FileNotFoundError("Module 30's train-fitted projection is required; refusing to refit")
    pca, scaler = (joblib.load(path) for path in required)
    if getattr(pca, "training_split_", None) != "train" or getattr(scaler, "training_split_", None) != "train":
        raise ValueError("Module 30 projection lacks train-only provenance")


def load_reused(data):
    runs = {}
    for config, old_name in REUSE.items():
        runs[config] = []
        for seed in SEEDS:
            result_path = OLD_RESULT_ROOT / f"{old_name}_seed{seed}.json"
            prediction_path = OLD_RESULT_ROOT / f"{old_name}_seed{seed}_test_predictions.csv"
            model_path = OLD_MODEL_ROOT / f"{old_name}_seed{seed}.pt"
            if any(not path.exists() for path in (result_path, prediction_path, model_path)):
                raise FileNotFoundError(f"Incomplete Module 30 run: {old_name} seed {seed}")
            with result_path.open(encoding="utf-8") as handle:
                old = json.load(handle)
            if old["config"] != old_name or old["seed"] != seed:
                raise ValueError(f"{result_path}: mismatched source run")
            with prediction_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if [row["sample_id"] for row in rows] != data["test"]["ids"] or [int(row["label"]) for row in rows] != data["test"]["labels"].tolist():
                raise ValueError(f"{prediction_path}: test prediction alignment failed")
            runs[config].append({"config": config, "seed": seed, "test": old["test"],
                                 "validation": old["validation"], "feature_dim": old["feature_dim"],
                                 "parameter_count": old["parameter_count"], "runtime_seconds": old["runtime_seconds"],
                                 "peak_gpu_memory_mib": old["peak_gpu_memory_mib"],
                                 "predictions": str(prediction_path), "checkpoint": str(model_path),
                                 "reused_from": str(result_path)})
    return runs


def permutation_for_examples(split, seed, count):
    # Separate reproducible random streams by split and seed. Each row is an
    # independent permutation; no label or model prediction enters selection.
    split_number = {"train": 0, "val": 1, "test": 2}[split]
    rng = np.random.default_rng(31_000 + 100 * seed + split_number)
    return np.stack([rng.permutation(27) for _ in range(count)])


def signature_features(paths, config, seed, permutations):
    raw = {}
    for split in SPLITS:
        points = paths[split]
        if config == "reverse":
            ordered = points[:, ::-1, :].copy()
        elif config.startswith("fixed_perm_"):
            ordered = points[:, permutations[int(config.rsplit("_", 1)[1])], :]
        elif config == "per_example":
            indices = permutation_for_examples(split, seed, len(points))
            ordered = np.take_along_axis(points, indices[:, :, None], axis=1)
        else:
            raise ValueError(config)
        raw[split] = base.signature(ordered, 3).numpy()
        if raw[split].shape != (len(points), 258) or not np.isfinite(raw[split]).all():
            raise ValueError(f"{config}/{split}: invalid signature shape or value")
    return raw


def mean_pool_features(paths):
    # The same projected 6D nodes enter a shared node MLP, then mean pooling.
    # No position-order operation is applied; permuting nodes cannot change it.
    return {split: np.asarray(points, dtype=np.float64) for split, points in paths.items()}


def scale_features(raw, config, seed):
    suffix = f"_{seed}" if config == "per_example" else ""
    path = MODEL_ROOT / f"{config}{suffix}_train_scaler.joblib"
    width = raw["train"].shape[-1]
    if path.exists():
        scaler = joblib.load(path)
        if getattr(scaler, "training_split_", None) != "train" or scaler.n_features_in_ != width:
            raise ValueError(f"{path}: invalid saved train scaler")
    else:
        train = raw["train"].reshape(-1, width) if config == "mean_pool" else raw["train"]
        scaler = StandardScaler().fit(train)
        scaler.training_split_ = "train"
        scaler.training_examples_ = base.COUNTS["train"]
        temporary = path.with_suffix(".joblib.tmp")
        joblib.dump(scaler, temporary)
        os.replace(temporary, path)
    scaled = {}
    for split in SPLITS:
        value = raw[split]
        transformed = scaler.transform(value.reshape(-1, width)).reshape(value.shape).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError(f"{config}/{split}: nonfinite standardized features")
        scaled[split] = transformed
    return scaled


class MeanPoolMLP(nn.Module):
    """Order-insensitive node MLP; 18,970 parameters vs 18,850 signature MLP."""

    def __init__(self):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(6, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.30),
                                  nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.30))
        self.head = nn.Sequential(nn.Linear(128, 8), nn.ReLU(), nn.Dropout(0.30), nn.Linear(8, 2))

    def forward(self, x):
        return self.head(self.node(x).mean(dim=1))


def run_one(data, paths, config, seed, permutations, device, overwrite):
    name = f"{config}_seed{seed}"
    result_path = RESULT_ROOT / f"{name}.json"
    checkpoint_path = MODEL_ROOT / f"{name}.pt"
    val_path = RESULT_ROOT / f"{name}_val_predictions.csv"
    test_path = RESULT_ROOT / f"{name}_test_predictions.csv"
    if not overwrite and all(path.exists() for path in (result_path, checkpoint_path, val_path, test_path)):
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result["config"] != config or result["seed"] != seed:
            raise ValueError(f"{result_path}: saved result mismatch")
        print(f"Skipping {name}: AUROC={result['test']['auroc']:.4f}", flush=True)
        return result

    raw = mean_pool_features(paths) if config == "mean_pool" else signature_features(paths, config, seed, permutations)
    features = scale_features(raw, config, seed)
    base.seed_all(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = MeanPoolMLP().to(device) if config == "mean_pool" else base.SignatureMLP(258).to(device)
    params = sum(p.numel() for p in model.parameters())
    counts = np.bincount(data["train"]["labels"], minlength=2)
    weights = torch.as_tensor(len(data["train"]["labels"]) / (2 * counts), device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(features["train"]),
                                            torch.from_numpy(data["train"]["labels"])),
                              batch_size=64, shuffle=True, generator=generator)
    best_auc, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, 101):
        model.train()
        loss_sum = 0.0
        for batch, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: nonfinite loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"{name}: nonfinite gradient")
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        val_scores = base.predict(model, features["val"], device)
        auc = float(roc_auc_score(data["val"]["labels"], val_scores))
        history.append({"epoch": epoch, "train_loss": loss_sum / len(data["train"]["labels"]), "val_auroc": auc})
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed,
                        "best_epoch": epoch, "val_auroc": auc}, temporary)
            os.replace(temporary, checkpoint_path)
        else:
            stale += 1
        if stale >= 15:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"])
    val_scores = base.predict(model, features["val"], device)
    threshold = base.select_threshold(data["val"]["labels"], val_scores)
    test_scores = base.predict(model, features["test"], device)
    base.save_predictions(val_path, data["val"]["ids"], data["val"]["labels"], val_scores, threshold)
    base.save_predictions(test_path, data["test"]["ids"], data["test"]["labels"], test_scores, threshold)
    runtime = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    result = {"config": config, "seed": seed, "feature_dim": int(features["train"].shape[-1]),
              "parameter_count": params, "best_epoch": best_epoch, "epochs_trained": epoch,
              "best_val_auroc": best_auc, "validation": base.metrics(data["val"]["labels"], val_scores, threshold),
              "test": base.metrics(data["test"]["labels"], test_scores, threshold),
              "counts": base.COUNTS, "runtime_seconds": runtime, "peak_gpu_memory_mib": peak_mib,
              "predictions": str(test_path), "checkpoint": str(checkpoint_path), "history": history}
    base.atomic_json(result, result_path)
    print(f"{name}: AUROC={result['test']['auroc']:.4f}, runtime={runtime:.1f}s", flush=True)
    return result


def prediction_matrix(data, runs):
    labels = data["test"]["labels"]
    ids = data["test"]["ids"]
    predictions = {}
    for config, selected in runs.items():
        rows_by_seed = []
        for run in selected:
            with Path(run["predictions"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if [row["sample_id"] for row in rows] != ids or [int(row["label"]) for row in rows] != labels.tolist():
                raise ValueError(f"{config} seed {run['seed']}: test prediction alignment failed")
            rows_by_seed.append([float(row["prob_hallucinated"]) for row in rows])
        predictions[config] = np.asarray(rows_by_seed, dtype=np.float64)
    return predictions


def bootstrap_comparisons(data, runs, replicates):
    labels = data["test"]["labels"]
    predictions = prediction_matrix(data, runs)
    rng = np.random.default_rng(31_4242)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    # Same stratified test-example resamples for every model and seed.
    samples = np.concatenate((rng.choice(positive, (replicates, len(positive)), replace=True),
                              rng.choice(negative, (replicates, len(negative)), replace=True)), axis=1)
    labels_boot = labels[samples]
    npos, nneg = len(positive), len(negative)

    def boot_auc(values):
        ranked = rankdata(values[samples], axis=1, method="average")
        positive_rank_sum = (ranked * labels_boot).sum(axis=1)
        return (positive_rank_sum - npos * (npos + 1) / 2) / (npos * nneg)

    reference_boot = np.stack([boot_auc(values) for values in predictions["original"]])
    reference_auc = np.asarray([run["test"]["auroc"] for run in runs["original"]])
    comparisons = {}
    for config in CONFIGS:
        if config == "original":
            continue
        candidate_auc = np.asarray([run["test"]["auroc"] for run in runs[config]])
        candidate_boot = np.stack([boot_auc(values) for values in predictions[config]])
        draws = (reference_boot - candidate_boot).mean(axis=0)
        low, high = np.percentile(draws, [2.5, 97.5])
        differences = reference_auc - candidate_auc
        comparisons[config] = {"per_seed_original_minus_alternative": differences.tolist(),
                               "mean_original_minus_alternative": float(differences.mean()),
                               "paired_bootstrap_95_ci": [float(low), float(high)],
                               "reliably_positive": bool(low > 0)}
    return comparisons


def summarize(data, runs, permutations, device, replicates):
    aggregate = {}
    for config in CONFIGS:
        selected = runs.get(config, [])
        if not selected:
            continue
        entry = {"seeds": [run["seed"] for run in selected], "per_seed": selected,
                 "parameter_count": selected[0]["parameter_count"]}
        for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
            values = np.asarray([run["test"][metric] for run in selected])
            entry[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
        aggregate[config] = entry
    comparisons = bootstrap_comparisons(data, runs, replicates)
    report = {"module": 31, "device": str(device), "counts": base.COUNTS,
              "projection": "reused Module 30 train-fitted PCA 16->5 and train-fitted projected-node scaler",
              "signature": "degree 3, six coordinates including normalized layer position, zero basepoint",
              "fixed_permutations": {str(number): value.tolist() for number, value in permutations.items()},
              "per_example_shuffle": "independent per example and split, RNG seed 31000 + 100*training_seed + split_index",
              "bootstrap": {"replicates": replicates, "scheme": "paired stratified test-example resampling; same indices for all seeds/models"},
              "aggregate": aggregate, "comparisons_to_original": comparisons}
    base.atomic_json(report, RESULT_ROOT / "summary.json")
    with (RESULT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("config", "auroc_mean", "auroc_sd", "auprc_mean", "balanced_accuracy_mean", "macro_f1_mean",
                         "original_minus_config", "ci_low", "ci_high"))
        for config in CONFIGS:
            item, comparison = aggregate[config], comparisons.get(config, {})
            ci = comparison.get("paired_bootstrap_95_ci", ("", ""))
            writer.writerow((config, item["auroc"]["mean"], item["auroc"]["sd"], item["auprc"]["mean"],
                             item["balanced_accuracy"]["mean"], item["macro_f1"]["mean"],
                             comparison.get("mean_original_minus_alternative", 0.0), *ci))
    return report


def smoke_test():
    base.smoke_test()
    permutations = fixed_permutations()
    points = np.arange(27 * 6, dtype=np.float64).reshape(1, 27, 6)
    if base.signature(points, 3).shape != (1, 258):
        raise RuntimeError("Degree-3 signature shape mismatch")
    for number, permutation in permutations.items():
        if len(np.unique(permutation)) != 27 or np.array_equal(permutation, np.arange(27)):
            raise RuntimeError(f"Invalid fixed permutation {number}")
    per_example = permutation_for_examples("train", 42, 8)
    if any(len(np.unique(row)) != 27 for row in per_example) or len({tuple(row) for row in per_example}) == 1:
        raise RuntimeError("Per-example shuffling is not independent")
    if not np.array_equal(per_example, permutation_for_examples("train", 42, 8)):
        raise RuntimeError("Per-example shuffling is not reproducible")
    model = MeanPoolMLP()
    model.eval()
    x = torch.randn((4, 27, 6))
    if not torch.allclose(model(x), model(x[:, torch.randperm(27)]), atol=1e-6):
        raise RuntimeError("Mean-pooling control changed under permutation")
    model(x).sum().backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Mean-pooling gradients missing or nonfinite")
    print("Module 31 smoke test passed: permutations, signature shape, mean-pool invariance, finite gradients", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    if args.smoke_test:
        smoke_test()
        return
    require_existing_preprocessing()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    data = base.load_data()
    paths, _ = base.load_or_fit_projection(data)
    permutations = fixed_permutations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = load_reused(data)
    for config in NEW_CONFIGS:
        runs[config] = [run_one(data, paths, config, seed, permutations, device, args.overwrite) for seed in SEEDS]
    report = summarize(data, runs, permutations, device, args.bootstrap)
    for config in CONFIGS:
        entry = report["aggregate"][config]["auroc"]
        comparison = report["comparisons_to_original"].get(config, {})
        print(f"{config}: AUROC={entry['mean']:.4f}+/-{entry['sd']:.4f}; CI={comparison.get('paired_bootstrap_95_ci')}", flush=True)


if __name__ == "__main__":
    main()
