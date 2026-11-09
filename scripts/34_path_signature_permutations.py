"""Module 34: natural-order vs 100 fixed layer-order permutations.

Uses Module 30's existing train-only 16->5 projection and degree-3 signature,
Module 31's train-only signature scaling convention, and identical MLP/training
protocol. The first five permutations (RNG seeds 30..34) and natural order
reuse complete Module 30/31 runs; new runs are stored under Module 34 paths.
No model fitting occurs until this script is explicitly run without --check.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent


def import_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = import_module("epigraph_module30_for34", "30_train_path_signature.py")
ordering = import_module("epigraph_module31_for34", "31_validate_path_signatures.py")

MODEL_ROOT = ROOT / "models/path_signature_permutations"
RESULT_ROOT = ROOT / "results/path_signature_permutations"
CACHE_PATH = MODEL_ROOT / "projected_t0_paths.npz"
CACHE_META = MODEL_ROOT / "projected_t0_paths.json"
SEEDS = (42, 43, 44, 45, 46)
SPLITS = ("train", "val", "test")
FIRST_ID = 30
MAX_PERMUTATIONS = 100


def provenance_digest():
    digest = hashlib.sha256()
    for name in ("train_projection_16_to_5.joblib", "projected_node_scaler.joblib"):
        path = ROOT / "models/path_signature" / name
        if not path.exists():
            raise FileNotFoundError(f"Module 30 train-fitted preprocessing missing: {path}")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    for split in SPLITS:
        files = sorted((ROOT / "processed_graphs/transitions_pca16" / split).glob("*.pt"))
        if len(files) != base.COUNTS[split]:
            raise ValueError(f"{split}: expected {base.COUNTS[split]} graphs, found {len(files)}")
        for path in files:
            stat = path.stat()
            digest.update(f"{split}/{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def require_train_projection():
    paths = [ROOT / "models/path_signature/train_projection_16_to_5.joblib",
             ROOT / "models/path_signature/projected_node_scaler.joblib"]
    pca, scaler = [joblib.load(path) for path in paths]
    if getattr(pca, "training_split_", None) != "train" or getattr(scaler, "training_split_", None) != "train":
        raise ValueError("Module 30 projection/scaler lacks train-only provenance")
    if int(pca.n_features_in_) != 16 or int(pca.n_components_) != 5 or int(scaler.n_features_in_) != 5:
        raise ValueError("Module 30 projection/scaler dimension mismatch")


def load_paths():
    require_train_projection()
    fingerprint = provenance_digest()
    if CACHE_PATH.exists() != CACHE_META.exists():
        raise RuntimeError("Partial projected-path cache; remove both cache files before rerun")
    if CACHE_PATH.exists():
        metadata = json.loads(CACHE_META.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            raise RuntimeError("Cached paths are stale; remove Module 34 cache files before rerun")
        with np.load(CACHE_PATH, allow_pickle=False) as saved:
            data = {split: {"labels": saved[f"{split}_labels"],
                            "ids": saved[f"{split}_ids"].tolist()} for split in SPLITS}
            paths = {split: saved[f"{split}_paths"] for split in SPLITS}
    else:
        data = base.load_data()  # validates t=0, layer order, labels and split isolation
        paths, _ = base.load_or_fit_projection(data)  # existing files required above
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for split in SPLITS:
            arrays[f"{split}_paths"] = paths[split]
            arrays[f"{split}_labels"] = data[split]["labels"]
            arrays[f"{split}_ids"] = np.asarray(data[split]["ids"])
        temporary = CACHE_PATH.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, CACHE_PATH)
        base.atomic_json({"fingerprint": fingerprint, "source": "Module 30 train-fitted PCA/scaler",
                          "counts": base.COUNTS, "t": 0}, CACHE_META)
    for split in SPLITS:
        if paths[split].shape != (base.COUNTS[split], 27, 6) or len(data[split]["ids"]) != base.COUNTS[split]:
            raise ValueError(f"{split}: invalid cached path dimensions")
        if not np.isfinite(paths[split]).all() or set(data[split]["labels"].tolist()) != {0, 1}:
            raise ValueError(f"{split}: nonfinite paths or invalid labels")
    if any(set(data[a]["ids"]) & set(data[b]["ids"]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Train/validation/test sample overlap")
    return data, paths


def permutations(count):
    result = {}
    for number in range(FIRST_ID, FIRST_ID + count):
        perm = base.SHUFFLE.copy() if number == 30 else np.random.default_rng(number).permutation(27)
        if sorted(perm.tolist()) != list(range(27)) or np.array_equal(perm, np.arange(27)):
            raise ValueError(f"Invalid permutation seed {number}")
        result[number] = perm
    if len({tuple(p.tolist()) for p in result.values()}) != count:
        raise ValueError("Duplicate fixed permutations")
    return result


def source_paths(number, seed):
    if number is None:
        prefix = ROOT / "results/path_signature" / f"order3_seed{seed}"
        model = ROOT / "models/path_signature" / f"order3_seed{seed}.pt"
        expected = "order3"
    elif number == 30:
        prefix = ROOT / "results/path_signature" / f"order3_shuffled_seed{seed}"
        model = ROOT / "models/path_signature" / f"order3_shuffled_seed{seed}.pt"
        expected = "order3_shuffled"
    elif number < 35:
        prefix = ROOT / "results/path_signature_ordering" / f"fixed_perm_{number}_seed{seed}"
        model = ROOT / "models/path_signature_ordering" / f"fixed_perm_{number}_seed{seed}.pt"
        expected = f"fixed_perm_{number}"
    else:
        prefix = RESULT_ROOT / f"fixed_perm_{number}_seed{seed}"
        model = MODEL_ROOT / f"fixed_perm_{number}_seed{seed}.pt"
        expected = f"fixed_perm_{number}"
    return prefix.with_suffix(".json"), Path(str(prefix) + "_test_predictions.csv"), model, expected


def read_run(number, seed, data):
    result_path, prediction_path, model_path, expected = source_paths(number, seed)
    if any(not p.exists() for p in (result_path, prediction_path, model_path)):
        raise FileNotFoundError(f"Incomplete run: {expected} seed {seed}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["config"] != expected or result["seed"] != seed or result["feature_dim"] != 258:
        raise ValueError(f"{result_path}: incompatible saved run")
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [r["sample_id"] for r in rows] != data["test"]["ids"] or [int(r["label"]) for r in rows] != data["test"]["labels"].tolist():
        raise ValueError(f"{prediction_path}: prediction IDs or labels do not align")
    scores = np.asarray([float(r["prob_hallucinated"]) for r in rows])
    from sklearn.metrics import roc_auc_score
    if abs(float(roc_auc_score(data["test"]["labels"], scores)) - result["test"]["auroc"]) > 1e-8:
        raise ValueError(f"{result_path}: saved AUROC disagrees with predictions")
    return {"seed": seed, "test": result["test"], "result_file": str(result_path),
            "prediction_file": str(prediction_path), "scores": scores}


def bootstrap_auc_ci(labels, seed_score_matrix, replicates, seed):
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    samples = np.concatenate((rng.choice(positive, (replicates, len(positive)), replace=True),
                              rng.choice(negative, (replicates, len(negative)), replace=True)), axis=1)
    npos, nneg = len(positive), len(negative)
    rank_sum = np.stack([(rankdata(scores[samples], axis=1)[:, :npos]).sum(axis=1)
                         for scores in seed_score_matrix])
    auc = (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)
    return np.percentile(auc.mean(axis=0), [2.5, 97.5]).tolist()


def summarize(data, count, requested_seeds, complete_only=True):
    if tuple(requested_seeds) != SEEDS:
        raise ValueError("Permutation-test summary requires all five seeds 42-46")
    natural = [read_run(None, seed, data) for seed in SEEDS]
    natural_auc = np.asarray([r["test"]["auroc"] for r in natural])
    observed = float(natural_auc.mean())
    entries = []
    fixed = permutations(count)
    for number in range(FIRST_ID, FIRST_ID + count):
        files = [source_paths(number, seed)[0] for seed in SEEDS]
        if not all(path.exists() for path in files):
            if complete_only:
                raise FileNotFoundError(f"Permutation {number} lacks a five-seed result")
            continue
        runs = [read_run(number, seed, data) for seed in SEEDS]
        values = np.asarray([r["test"]["auroc"] for r in runs])
        entries.append({"id": number, "permutation": fixed[number].tolist(),
                        "auroc_mean": float(values.mean()), "auroc_sd": float(values.std(ddof=1)),
                        "per_seed_auroc": values.tolist(), "result_files": [r["result_file"] for r in runs]})
    if not entries:
        raise ValueError("No complete permutation results")
    means = np.asarray([entry["auroc_mean"] for entry in entries])
    rng = np.random.default_rng(34042)
    bootstrap_null = means[rng.integers(0, len(means), size=(10000, len(means)))].mean(axis=1)
    result = {"module": 34, "permutations_requested": count, "permutations_complete": len(entries),
              "seeds": list(SEEDS), "counts": base.COUNTS,
              "method": "degree-3 zero-basepoint path signature; fixed shared permutation; train-only scaler; same Module 30 MLP/training",
              "p_value": (1 + int(np.sum(means >= observed))) / (len(means) + 1),
              "p_value_definition": "one-sided Monte Carlo (1 + count(permutation mean AUROC >= natural mean AUROC))/(N+1)",
              "natural": {"auroc_mean": observed, "auroc_sd": float(natural_auc.std(ddof=1)),
                          "per_seed_auroc": natural_auc.tolist(),
                          "paired_test_bootstrap_95_ci": bootstrap_auc_ci(data["test"]["labels"],
                              np.stack([r["scores"] for r in natural]), 2000, 34100),
                          "result_files": [r["result_file"] for r in natural]},
              "permutation_distribution": {"mean_auroc": float(means.mean()),
                                           "sd_across_permutations": float(means.std(ddof=1)) if len(means) > 1 else None,
                                           "central_95_percentiles": np.percentile(means, [2.5, 97.5]).tolist(),
                                           "bootstrap_95_ci_of_mean": np.percentile(bootstrap_null, [2.5, 97.5]).tolist()},
              "permutations": entries}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    base.atomic_json(result, RESULT_ROOT / "summary.json")
    with (RESULT_ROOT / "permutations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("permutation_id", "mean_auroc", "sd_across_seeds", "seed42", "seed43", "seed44", "seed45", "seed46"))
        for item in entries:
            writer.writerow((item["id"], item["auroc_mean"], item["auroc_sd"], *item["per_seed_auroc"]))
    print(f"Natural={observed:.4f}; permutation mean={means.mean():.4f}; N={len(entries)}; one-sided p={result['p_value']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=100, help="First N fixed RNG permutations, N=1..100")
    parser.add_argument("--check", action="store_true", help="Validate configuration and saved preprocessing only; no training or writes")
    args = parser.parse_args()
    if not 1 <= args.permutations <= MAX_PERMUTATIONS:
        parser.error("--permutations must be in 1..100")
    selected = permutations(args.permutations)
    require_train_projection()
    if args.check:
        print(f"Validated {len(selected)} unique fixed permutations and train-only Module 30 projection; no training")
        return
    data, paths = load_paths()
    # Directly reuse historical runs only when their exact permutation and all
    # five models/predictions are present. Their result files remain untouched.
    for number in list(selected)[:5]:
        for seed in SEEDS:
            read_run(number, seed, data)
    for seed in SEEDS:
        read_run(None, seed, data)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    base.MODEL_ROOT = MODEL_ROOT
    base.RESULT_ROOT = RESULT_ROOT
    ordering.MODEL_ROOT = MODEL_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for number, permutation in selected.items():
        if number < 35:
            print(f"Reusing fixed permutation {number} from Modules 30/31", flush=True)
            continue
        existing = [source_paths(number, seed)[0].exists() for seed in SEEDS]
        if all(existing):
            for seed in SEEDS:
                read_run(number, seed, data)
            print(f"Skipping completed permutation {number}", flush=True)
            continue
        config = f"fixed_perm_{number}"
        raw = {split: base.signature(points[:, permutation, :], 3).numpy() for split, points in paths.items()}
        if any(value.shape != (base.COUNTS[split], 258) or not np.isfinite(value).all() for split, value in raw.items()):
            raise ValueError(f"{config}: invalid signature features")
        features = ordering.scale_features(raw, config, SEEDS[0])  # fits on train only
        for seed in SEEDS:
            if source_paths(number, seed)[0].exists():
                read_run(number, seed, data)
                continue
            base.run_one(data, features, config, 258, seed, device, overwrite=False)
        if (number - FIRST_ID + 1) % 10 == 0:
            summarize(data, number - FIRST_ID + 1, SEEDS)
    summarize(data, args.permutations, SEEDS)


if __name__ == "__main__":
    main()
