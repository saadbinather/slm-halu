"""Matched Module 26 transition+geometry node MLP on Module 35 TriviaQA.

Inputs: 27 t=0 PCA-16 transitions plus four Module-21 raw geometry values.
Train-only node StandardScaler, 20->16 projection, positional coordinate,
64/64 node MLP, mean pool, 64->32->2 head. No generation or extraction.
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models/triviaqa_mlp"
RESULT_ROOT = ROOT / "results/triviaqa_mlp"
DATASET_DISPLAY = "TriviaQA"
MODULE_NUMBER = 36


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("35_triviaqa_generalization.py", "epigraph35_for_mlp")
module21 = load_module("21_extract_transitions.py", "epigraph21_for_mlp")
module26 = load_module("26_compare_state_transition_mlp.py", "epigraph26_for_mlp")
SEEDS = base.SEEDS
SPLITS = base.SPLITS
COUNTS = base.COUNTS


def load_raw():
    sample = base.load_sample()
    preprocessing_path = base.RESULT / "models/within_preprocessing.joblib"
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"{DATASET_DISPLAY} within-dataset PCA is missing; finish its Path Signature run first")
    preprocessing = joblib.load(preprocessing_path)
    pca = preprocessing["pca16"]
    if pca.n_features_in_ != 1536 or pca.n_components != 16 or getattr(pca, "training_split_", None) != "train":
        raise ValueError("Module 35 PCA16 is not the train-fitted 1536->16 projection")
    if preprocessing["sample_ids"] != {split: [r["sample_id"] for r in sample["rows"][split]] for split in SPLITS}:
        raise ValueError("Module 35 PCA belongs to a different sample")
    result = {}
    for split in SPLITS:
        features = []
        for row in tqdm(sample["rows"][split], desc=f"MLP features {split}"):
            hidden = torch.load(base.path_for("hidden_states", split, row["sample_id"], ".pt"),
                                map_location="cpu", weights_only=False)
            transition = torch.load(base.path_for("transitions", split, row["sample_id"], ".pt"),
                                    map_location="cpu", weights_only=False)
            base.validate_hidden(hidden, row)
            raw = transition["transition_vectors"]
            if transition["sample_id"] != row["sample_id"] or raw.shape != (27, 1536):
                raise ValueError("Transition/sample mismatch")
            if not torch.equal(raw, hidden["hidden_states"][1:] - hidden["hidden_states"][:-1]):
                raise ValueError("Saved transitions disagree with hidden states")
            reduced = pca.transform(raw.numpy()).astype(np.float32)
            trace = {"sample_id": row["sample_id"], "split": split, "label": 0,
                     "label_name": "placeholder", "time_positions": [0], "num_layers": 28,
                     "hidden_dim": 1536, "hidden_states": hidden["hidden_states"].unsqueeze(0),
                     "num_generated_tokens": hidden["num_generated_tokens"]}
            geometry = module21.extract_transitions(trace, None, None)["geometric_features"][0, :, :4].numpy()
            x = np.concatenate((reduced, geometry), axis=1).astype(np.float32)
            if x.shape != (27, 20) or not np.isfinite(x).all():
                raise ValueError("Expected finite [27,20] PCA+geometry features")
            features.append(x)
        result[split] = np.stack(features)
    return sample, result


def load_scaled():
    sample = base.load_sample()
    cache_path = MODEL_ROOT / "node_features.npz"
    scaler_path = MODEL_ROOT / "train_node_scaler.joblib"
    if cache_path.exists() and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if scaler.n_features_in_ != 20 or getattr(scaler, "training_split_", None) != "train":
            raise ValueError("Saved MLP scaler lacks train-only provenance")
        with np.load(cache_path, allow_pickle=False) as saved:
            if any(saved[f"{split}_ids"].tolist() != [r["sample_id"] for r in sample["rows"][split]] for split in SPLITS):
                raise ValueError("Cached MLP features belong to another sample")
            features = {split: saved[split] for split in SPLITS}
    else:
        if cache_path.exists() or scaler_path.exists():
            print("Incomplete MLP feature cache; rebuilding from saved Module 35 transitions")
        sample, raw = load_raw()
        scaler = StandardScaler().fit(raw["train"].reshape(-1, 20))
        scaler.training_split_ = "train"
        scaler.training_examples_ = COUNTS["train"]
        scaler.feature_policy_ = "Module 26 C: one scaler per input channel pooled across train nodes"
        features = {split: scaler.transform(raw[split].reshape(-1, 20)).reshape(COUNTS[split], 27, 20).astype(np.float32)
                    for split in SPLITS}
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = scaler_path.with_suffix(".joblib.tmp")
        joblib.dump(scaler, temporary)
        os.replace(temporary, scaler_path)
        temporary = cache_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **features,
                                **{f"{split}_ids": np.asarray([r["sample_id"] for r in sample["rows"][split]]) for split in SPLITS})
        os.replace(temporary, cache_path)
    for split in SPLITS:
        if features[split].shape != (COUNTS[split], 27, 20) or not np.isfinite(features[split]).all():
            raise ValueError(f"{split}: invalid scaled MLP features")
    return sample, features


def run_one(variant, seed, features, data, device):
    name = f"{variant}_seed{seed}"
    result_path = RESULT_ROOT / f"{name}.json"
    model_path = MODEL_ROOT / f"{name}.pt"
    state_path = MODEL_ROOT / f"{name}_train_state.pt"
    val_path = RESULT_ROOT / f"{name}_val_predictions.csv"
    test_path = RESULT_ROOT / f"{name}_test_predictions.csv"
    if all(path.exists() for path in (result_path, model_path, val_path, test_path)):
        result = base.read_json(result_path)
        if result.get("variant") != variant or result.get("seed") != seed:
            raise ValueError("Completed run metadata mismatch")
        print(f"Skipping completed {name}")
        return
    if result_path.exists() or (val_path.exists() or test_path.exists()) and not state_path.exists():
        raise RuntimeError(f"{name}: incomplete final artifacts require inspection")
    module26.set_seed(seed)
    started = time.perf_counter()
    model = module26.MatchedNodeMLP(20).to(device)
    labels_train = data["train"]["labels"]
    counts = np.bincount(labels_train, minlength=2)
    if counts.min() == 0:
        raise ValueError("Both labels required for weighted cross-entropy")
    weights = torch.as_tensor(len(labels_train) / (2 * counts), device=device, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(features["train"]), torch.from_numpy(labels_train)),
                        batch_size=64, shuffle=True, generator=generator)
    best, best_epoch, stale, history, start_epoch = -1.0, 0, 0, [], 1
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["seed"] != seed or state["variant"] != variant:
            raise ValueError("Training state belongs to a different run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry[key] = value.to(device)
        best, best_epoch, stale, history = state["best"], state["best_epoch"], state["stale"], state["history"]
        start_epoch = state["epoch"] + 1
        generator.set_state(state["generator_state"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if not model_path.exists():
            raise FileNotFoundError("Best model checkpoint missing on resume")
    for epoch in range(start_epoch, 101) if stale < 15 else ():
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite MLP loss")
            loss.backward()
            optimizer.step()
        val_scores = module26.predict(model, features["val"], device)
        auc = float(roc_auc_score(data["val"]["labels"], val_scores))
        history.append({"epoch": epoch, "val_auroc": auc})
        if auc > best:
            best, best_epoch, stale = auc, epoch, 0
            base.atomic_torch({"state_dict": model.state_dict(), "seed": seed, "variant": variant,
                               "best_epoch": epoch, "val_auroc": auc, "input_dim": 20}, model_path)
        else:
            stale += 1
        base.atomic_torch({"seed": seed, "variant": variant, "epoch": epoch,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "best": best, "best_epoch": best_epoch, "stale": stale, "history": history,
                           "generator_state": generator.get_state(), "python_rng": random.getstate(),
                           "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, state_path)
        if stale >= 15:
            break
    epoch = history[-1]["epoch"]
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["state_dict"])
    val_scores = module26.predict(model, features["val"], device)
    threshold = module26.threshold_from_val(data["val"]["labels"], val_scores)
    test_scores = module26.predict(model, features["test"], device)
    base.save_predictions(val_path, data["val"]["ids"], data["val"]["labels"], val_scores, threshold)
    base.save_predictions(test_path, data["test"]["ids"], data["test"]["labels"], test_scores, threshold)
    base.atomic_json({"module": MODULE_NUMBER, "dataset": DATASET_DISPLAY,
                      "arm": "C_transition_geometry", "variant": variant,
                      "seed": seed, "input_shape": [27, 20],
                      "configuration": {"architecture": "Module 26 MatchedNodeMLP(20)",
                                        "pca_components": 16, "geometry_features": 4,
                                        "scaler": str(MODEL_ROOT / "train_node_scaler.joblib"),
                                        "batch_size": 64, "dropout": 0.30,
                                        "optimizer": "AdamW", "learning_rate": 0.001,
                                        "weight_decay": 0.0001, "max_epochs": 100,
                                        "patience": 15, "model_selection": "validation AUROC",
                                        "threshold_selection": "validation balanced accuracy"},
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "best_epoch": best_epoch, "epochs_trained": epoch,
                      "validation": module26.metrics(data["val"]["labels"], val_scores, threshold),
                      "test": module26.metrics(data["test"]["labels"], test_scores, threshold),
                      "runtime_seconds": time.perf_counter() - started,
                      "checkpoint": str(model_path), "test_predictions": str(test_path)}, result_path)
    if state_path.exists():
        state_path.unlink()
    print(f"{name}: test AUROC={roc_auc_score(data['test']['labels'], test_scores):.4f}")


def read_predictions(path, ids, labels):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row["sample_id"] for row in rows] != ids or [int(row["label"]) for row in rows] != labels.tolist():
        raise ValueError(f"{path}: IDs or labels disagree with Module 35 manifest")
    scores = np.asarray([float(row["prob_incorrect"]) for row in rows])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{path}: invalid prediction probabilities")
    return scores


def evaluate(variant):
    _, data = base.labeled_data(variant)
    results = []
    for seed in SEEDS:
        result_path = RESULT_ROOT / f"{variant}_seed{seed}.json"
        result = base.read_json(result_path)
        if result.get("seed") != seed or result.get("variant") != variant:
            raise ValueError(f"{result_path}: wrong run metadata")
        scores = read_predictions(RESULT_ROOT / f"{variant}_seed{seed}_test_predictions.csv",
                                  data["test"]["ids"], data["test"]["labels"])
        if abs(roc_auc_score(data["test"]["labels"], scores) - result["test"]["auroc"]) > 1e-8:
            raise ValueError(f"{result_path}: AUROC disagrees with predictions")
        results.append(result)
    aggregate = {"variant": variant, "seeds": list(SEEDS), "counts": COUNTS,
                 "per_seed_auroc": [r["test"]["auroc"] for r in results]}
    for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
        values = np.asarray([r["test"][metric] for r in results])
        aggregate[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    base.atomic_json(aggregate, RESULT_ROOT / f"{variant}_aggregate.json")
    print(f"{variant}: MLP AUROC {aggregate['auroc']['mean']:.4f} +/- {aggregate['auroc']['sd']:.4f}")
    return aggregate


def report():
    base.load_sample()
    summary = {"module": MODULE_NUMBER, "dataset": DATASET_DISPLAY,
               "comparison": f"{DATASET_DISPLAY} within-dataset degree-3 Path Signature versus Module 26 C transition+geometry MLP",
               "bootstrap": "2000 stratified resamples of the same 399 test examples, paired over seeds",
               "results": {}}
    lines = [f"# {DATASET_DISPLAY}: Path Signature versus matched transition + geometry MLP", "",
             "The same 1,858/398/399 Module 35 questions and labels are used. Test is held out from official validation.",
             "The MLP has the original Module 26 architecture and 27 PCA-16 + four raw geometry inputs; "
             f"its train-only 20-channel scaler and {DATASET_DISPLAY} PCA are reused for both label variants.", "",
             "| Labels | Path Signature AUROC | MLP AUROC | MLP AUPRC | MLP balanced accuracy | MLP Macro-F1 | Path minus MLP (paired 95% CI) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in ("exact", "semantic"):
        aggregate = evaluate(variant)
        _, data = base.labeled_data(variant)
        labels, ids = data["test"]["labels"], data["test"]["ids"]
        mlp = np.stack([read_predictions(RESULT_ROOT / f"{variant}_seed{seed}_test_predictions.csv", ids, labels)
                        for seed in SEEDS])
        signature = np.stack([read_predictions(base.prediction_path("within", variant, seed), ids, labels)
                              for seed in SEEDS])
        draws = base.bootstrap_auc(labels, {"mlp": mlp, "signature": signature})
        difference = draws["signature"] - draws["mlp"]
        ps_values = np.asarray([base.read_json(base.evaluation_path("within", variant, seed))["test"]["auroc"]
                                for seed in SEEDS])
        delta = float(ps_values.mean() - aggregate["auroc"]["mean"])
        ci = np.percentile(difference, [2.5, 97.5]).tolist()
        summary["results"][variant] = {"mlp": aggregate, "path_signature":
            {"auroc_mean": float(ps_values.mean()), "auroc_sd": float(ps_values.std(ddof=1))},
            "path_minus_mlp_auroc": delta,
            "paired_per_seed_auroc_difference": (ps_values - np.asarray(aggregate["per_seed_auroc"])).tolist(),
            "paired_95_ci": ci,
            "mlp_minus_path_auroc": -delta,
            "mlp_minus_path_paired_95_ci": [-ci[1], -ci[0]]}
        lines.append(f"| {variant} | {ps_values.mean():.4f} +/- {ps_values.std(ddof=1):.4f} | "
                     f"{aggregate['auroc']['mean']:.4f} +/- {aggregate['auroc']['sd']:.4f} | "
                     f"{aggregate['auprc']['mean']:.4f} | {aggregate['balanced_accuracy']['mean']:.4f} | "
                     f"{aggregate['macro_f1']['mean']:.4f} | "
                     f"{delta:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines.extend(["", "Positive differences favor Path Signatures; negative differences favor the MLP. "
                  "The paired CI accounts for shared test examples and averages across the same five seeds. "
                  "Labels indicate reference correctness, not verified factual hallucination."])
    base.atomic_json(summary, RESULT_ROOT / "summary.json")
    (RESULT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {RESULT_ROOT / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "report", "validate"))
    parser.add_argument("--variant", choices=("exact", "semantic"), default="exact")
    args = parser.parse_args()
    if args.stage == "validate":
        base.validate(argparse.Namespace(scope="all", variant=args.variant))
        _, features = load_scaled()
        print({split: value.shape for split, value in features.items()})
        return
    if args.stage == "train":
        _, data = base.labeled_data(args.variant)
        _, features = load_scaled()
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in SEEDS:
            run_one(args.variant, seed, features, data, device)
    elif args.stage == "evaluate":
        evaluate(args.variant)
    else:
        report()


if __name__ == "__main__":
    main()
