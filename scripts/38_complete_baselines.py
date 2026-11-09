"""Module 38: saved t=0 Hidden-State MLP, CNN and Transformer baselines.

No language-model inference or hidden-state extraction is performed. Target
preprocessing is fitted on the corresponding dataset's training split only.
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
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/essential_baselines"
RESULTS = ROOT / "results/essential_baselines"
SPLITS = ("train", "val", "test")
DATASETS = ("triviaqa", "hotpotqa")
VARIANTS = ("exact", "semantic")
ARCHITECTURES = ("hidden_mlp", "cnn", "transformer")
SEEDS = (42, 43, 44, 45, 46)
COUNTS = {"train": 1858, "val": 398, "test": 399}
BATCH_SIZE, EPOCHS, PATIENCE = 64, 100, 15
LR, WEIGHT_DECAY = 1e-3, 1e-4


def import_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod26 = import_script("26_compare_state_transition_mlp.py", "epigraph38_m26")
mod33 = import_script("33_train_sequence_baselines.py", "epigraph38_m33")
trivia_mlp = import_script("36_triviaqa_mlp.py", "epigraph38_trivia_mlp")
hot_mlp = import_script("37_hotpotqa_mlp.py", "epigraph38_hot_mlp")


def context(dataset):
    if dataset == "triviaqa":
        return trivia_mlp.base, trivia_mlp
    return hot_mlp.hot.base, hot_mlp.mlp


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def atomic_torch(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_joblib(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def atomic_npz(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **value)
    os.replace(temporary, path)


def ids_for(sample):
    return {split: [row["sample_id"] for row in sample["rows"][split]] for split in SPLITS}


def load_labels(base, variant):
    """Verify saved labels and manifest alignment without reauditing all traces."""
    sample = base.load_sample()
    data = {}
    for split in SPLITS:
        ids, values = [], []
        for row in sample["rows"][split]:
            sid = row["sample_id"]
            path = base.path_for(f"labels/{variant}", split, sid, ".json")
            decision = base.read_json(path)
            if (decision.get("sample_id"), decision.get("split"), decision.get("variant")) != (sid, split, variant):
                raise ValueError(f"{path}: label metadata mismatch")
            label = decision.get("label")
            if label not in (0, 1) or decision.get("uncertain"):
                raise ValueError(f"{path}: missing or uncertain correctness label")
            ids.append(sid)
            values.append(int(label))
        data[split] = {"ids": ids, "labels": np.asarray(values, dtype=np.int64)}
        if len(ids) != COUNTS[split] or set(values) != {0, 1}:
            raise ValueError(f"{split}: incorrect sample size or single correctness class")
    return sample, data


def read_cache(path, sample, shape):
    if not path.exists():
        return None
    ids = ids_for(sample)
    with np.load(path, allow_pickle=False) as saved:
        if any(saved[f"{split}_ids"].tolist() != ids[split] for split in SPLITS):
            raise ValueError(f"{path}: cache IDs disagree with sample manifest")
        arrays = {split: saved[split] for split in SPLITS}
    for split, values in arrays.items():
        if values.shape != (COUNTS[split], *shape) or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid {split} features {values.shape}")
    return arrays


def write_cache(path, sample, arrays):
    ids = ids_for(sample)
    atomic_npz({**arrays, **{f"{split}_ids": np.asarray(ids[split]) for split in SPLITS}}, path)


def sequence_features(dataset, sample):
    """Module 33 [27,22] policy, using target train PCA16 and t=0 geometry."""
    model_dir = MODELS / dataset / "preprocessing"
    cache = model_dir / "sequence_t0_22.npz"
    scaler_path = model_dir / "sequence_geometry_scaler.joblib"
    saved = read_cache(cache, sample, (27, 22))
    if saved is not None:
        scaler = joblib.load(scaler_path)
        if (getattr(scaler, "training_split_", None) != "train" or scaler.n_features_in_ != 4
                or getattr(scaler, "sample_ids_", None) != ids_for(sample)):
            raise ValueError("Sequence geometry scaler lacks train-only provenance")
        return saved
    base, mlp = context(dataset)
    source_sample, raw = mlp.load_raw()  # saved transitions/hidden states; existing target PCA16
    if ids_for(source_sample) != ids_for(sample):
        raise ValueError("Sequence inputs use different sample IDs")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if (getattr(scaler, "training_split_", None) != "train" or scaler.n_features_in_ != 4
                or getattr(scaler, "sample_ids_", None) != ids_for(sample)):
            raise ValueError("Sequence geometry scaler lacks train-only provenance")
    else:
        scaler = StandardScaler().fit(raw["train"][:, :, 16:20].reshape(-1, 4))
        scaler.training_split_ = "train"
        scaler.training_examples_ = COUNTS["train"]
        scaler.sample_ids_ = ids_for(sample)
        atomic_joblib(scaler, scaler_path)
    arrays = {}
    for split in SPLITS:
        pca = raw[split][:, :, :16]
        geo = scaler.transform(raw[split][:, :, 16:20].reshape(-1, 4)).reshape(-1, 27, 4)
        position = np.broadcast_to(np.linspace(0, 1, 27, dtype=np.float32)[None, :, None],
                                   (COUNTS[split], 27, 1))
        zero_time = np.zeros_like(position)
        arrays[split] = np.concatenate((pca, geo, position, zero_time), axis=2).astype(np.float32)
    write_cache(cache, sample, arrays)
    return arrays


def hidden_states(base, row, split):
    path = base.path_for("hidden_states", split, row["sample_id"], ".pt")
    hidden = torch.load(path, map_location="cpu", weights_only=False)
    base.validate_hidden(hidden, row)
    values = hidden["hidden_states"].numpy()
    if values.shape != (28, 1536) or not np.isfinite(values).all():
        raise ValueError(f"{path}: invalid t=0 hidden states")
    return values


def hidden_features(dataset, sample):
    """Module 26 A: target train PCA16, then train-only node StandardScaler."""
    base, _ = context(dataset)
    model_dir = MODELS / dataset / "preprocessing"
    cache = model_dir / "hidden_t0_16.npz"
    pca_path = model_dir / "hidden_pca16.joblib"
    scaler_path = model_dir / "hidden_node_scaler.joblib"
    saved = read_cache(cache, sample, (28, 16))
    if saved is not None:
        pca, scaler = joblib.load(pca_path), joblib.load(scaler_path)
        if getattr(pca, "training_split_", None) != "train" or getattr(scaler, "training_split_", None) != "train":
            raise ValueError("Hidden preprocessing lacks train-only provenance")
        if pca.n_features_in_ != 1536 or pca.n_components != 16 or scaler.n_features_in_ != 16:
            raise ValueError("Hidden preprocessing dimensions changed")
        if pca.sample_ids_ != ids_for(sample) or getattr(scaler, "sample_ids_", None) != ids_for(sample):
            raise ValueError("Hidden PCA belongs to different sample")
        return saved
    if pca_path.exists():
        pca = joblib.load(pca_path)
        if (getattr(pca, "training_split_", None) != "train" or pca.sample_ids_ != ids_for(sample)
                or pca.n_features_in_ != 1536 or pca.n_components != 16):
            raise ValueError("Saved hidden PCA provenance or dimensions mismatch")
    else:
        pca = IncrementalPCA(n_components=16)
        buffer, rows = [], 0
        for row in tqdm(sample["rows"]["train"], desc=f"{dataset} train hidden PCA"):
            values = hidden_states(base, row, "train")
            buffer.append(values)
            rows += 28
            if rows >= 8192:
                pca.partial_fit(np.concatenate(buffer))
                buffer, rows = [], 0
        if buffer:
            pca.partial_fit(np.concatenate(buffer))
        pca.training_split_ = "train"
        pca.sample_ids_ = ids_for(sample)
        atomic_joblib(pca, pca_path)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if (getattr(scaler, "training_split_", None) != "train" or scaler.n_features_in_ != 16
                or getattr(scaler, "sample_ids_", None) != ids_for(sample)):
            raise ValueError("Saved hidden scaler provenance or dimensions mismatch")
    else:
        scaler = StandardScaler()
        for row in tqdm(sample["rows"]["train"], desc=f"{dataset} hidden scaler"):
            values = pca.transform(hidden_states(base, row, "train")).astype(np.float32)
            scaler.partial_fit(values)
        scaler.training_split_ = "train"
        scaler.sample_ids_ = ids_for(sample)
        atomic_joblib(scaler, scaler_path)
    arrays = {}
    for split in SPLITS:
        features = []
        for row in tqdm(sample["rows"][split], desc=f"{dataset} hidden features {split}"):
            projected = pca.transform(hidden_states(base, row, split)).astype(np.float32)
            features.append(scaler.transform(projected).astype(np.float32))
        arrays[split] = np.stack(features)
    write_cache(cache, sample, arrays)
    return arrays


def feature_data(dataset, architecture, sample):
    return hidden_features(dataset, sample) if architecture == "hidden_mlp" else sequence_features(dataset, sample)


def paths(dataset, variant, architecture, seed):
    name = f"{variant}_seed{seed}"
    model_dir = MODELS / dataset / architecture
    result_dir = RESULTS / dataset / architecture
    return {"best": model_dir / f"{name}_best.pt", "state": model_dir / f"{name}_state.pt",
            "json": result_dir / f"{name}.json", "val": result_dir / f"{name}_val_predictions.csv",
            "test": result_dir / f"{name}_test_predictions.csv"}


def make_model(architecture):
    return mod26.MatchedNodeMLP(16) if architecture == "hidden_mlp" else mod33.Classifier(architecture)


def predict(model, x, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=BATCH_SIZE):
            probs.extend(model(batch.to(device)).softmax(dim=1)[:, 1].cpu().tolist())
    return np.asarray(probs, dtype=np.float64)


def save_predictions(path, ids, labels, probabilities, threshold):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "label", "prob_incorrect", "prediction", "threshold"))
        for sid, label, p in zip(ids, labels, probabilities):
            writer.writerow((sid, int(label), float(p), int(p >= threshold), threshold))
    os.replace(temporary, path)


def train_one(dataset, variant, architecture, seed, features, labels, device):
    p = paths(dataset, variant, architecture, seed)
    final = (p["best"], p["json"], p["val"], p["test"])
    if all(path.exists() for path in final):
        result = json.loads(p["json"].read_text(encoding="utf-8"))
        if (result["dataset"], result["variant"], result["architecture"], result["seed"]) != (dataset, variant, architecture, seed):
            raise ValueError(f"{p['json']}: completed run mismatch")
        print(f"Skip completed {dataset}/{variant}/{architecture}/seed{seed}")
        return
    if p["json"].exists() or p["val"].exists() or p["test"].exists():
        raise RuntimeError(f"Partial final outputs for {p['json']}; inspect before retrying")
    mod33.seed_everything(seed)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = make_model(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_y = labels["train"]["labels"]
    counts = np.bincount(train_y, minlength=2)
    if counts.min() == 0:
        raise ValueError("Both training correctness classes are required")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(train_y) / (2 * counts), dtype=torch.float32, device=device))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(features["train"]), torch.from_numpy(train_y)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    best, best_epoch, stale, history, start_epoch = -1.0, 0, 0, [], 1
    if p["state"].exists():
        state = torch.load(p["state"], map_location="cpu", weights_only=False)
        if (state["dataset"], state["variant"], state["architecture"], state["seed"]) != (dataset, variant, architecture, seed):
            raise ValueError("Training state belongs to another run")
        if not p["best"].exists():
            raise FileNotFoundError("Best checkpoint missing on resume")
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
    for epoch in range(start_epoch, EPOCHS + 1) if stale < PATIENCE else ():
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            optimizer.step()
        val_p = predict(model, features["val"], device)
        auc = float(roc_auc_score(labels["val"]["labels"], val_p))
        history.append({"epoch": epoch, "val_auroc": auc})
        if auc > best:
            best, best_epoch, stale = auc, epoch, 0
            atomic_torch({"state_dict": model.state_dict(), "dataset": dataset, "variant": variant,
                          "architecture": architecture, "seed": seed, "epoch": epoch, "val_auroc": auc}, p["best"])
        else:
            stale += 1
        atomic_torch({"dataset": dataset, "variant": variant, "architecture": architecture, "seed": seed,
                      "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "best": best, "best_epoch": best_epoch, "stale": stale, "history": history,
                      "generator_state": generator.get_state(), "python_rng": random.getstate(),
                      "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, p["state"])
        if stale >= PATIENCE:
            break
    if not history:
        raise RuntimeError("No saved training history")
    model.load_state_dict(torch.load(p["best"], map_location=device, weights_only=False)["state_dict"])
    val_p = predict(model, features["val"], device)
    threshold = mod33.threshold_from_validation(labels["val"]["labels"], val_p)
    test_p = predict(model, features["test"], device)
    save_predictions(p["val"], labels["val"]["ids"], labels["val"]["labels"], val_p, threshold)
    save_predictions(p["test"], labels["test"]["ids"], labels["test"]["labels"], test_p, threshold)
    result = {"module": 38, "dataset": dataset, "variant": variant, "architecture": architecture,
              "seed": seed, "input_shape": list(features["train"].shape[1:]),
              "parameter_count": sum(v.numel() for v in model.parameters()),
              "protocol": {"batch_size": BATCH_SIZE, "dropout": 0.30, "optimizer": "AdamW",
                           "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "max_epochs": EPOCHS,
                           "patience": PATIENCE, "epoch_selection": "validation AUROC",
                           "threshold_selection": "validation balanced accuracy"},
              "best_epoch": best_epoch, "epochs_trained": history[-1]["epoch"],
              "validation": mod33.metrics(labels["val"]["labels"], val_p, threshold),
              "test": mod33.metrics(labels["test"]["labels"], test_p, threshold),
              "runtime_seconds": time.perf_counter() - started,
              "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "best_checkpoint": str(p["best"]), "test_predictions": str(p["test"]), "history": history}
    atomic_json(result, p["json"])
    p["state"].unlink(missing_ok=True)
    print(f"{dataset}/{variant}/{architecture}/seed{seed}: AUROC {result['test']['auroc']:.4f}")


def read_scores(path, ids, labels):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [r["sample_id"] for r in rows] != ids or [int(r["label"]) for r in rows] != labels.tolist():
        raise ValueError(f"{path}: sample IDs or labels differ")
    values = np.asarray([float(r["prob_incorrect"]) for r in rows])
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"{path}: invalid probabilities")
    return values


def evaluate(dataset, variant, architecture):
    base, _ = context(dataset)
    _, data = load_labels(base, variant)
    runs, scores, signature = [], [], []
    for seed in SEEDS:
        p = paths(dataset, variant, architecture, seed)
        if not all(p[key].exists() for key in ("json", "best", "val", "test")):
            raise FileNotFoundError(f"Incomplete {dataset}/{variant}/{architecture}/seed{seed}")
        run = json.loads(p["json"].read_text(encoding="utf-8"))
        if (run["dataset"], run["variant"], run["architecture"], run["seed"]) != (dataset, variant, architecture, seed):
            raise ValueError(f"{p['json']}: wrong run metadata")
        prediction = read_scores(p["test"], data["test"]["ids"], data["test"]["labels"])
        if abs(roc_auc_score(data["test"]["labels"], prediction) - run["test"]["auroc"]) > 1e-8:
            raise ValueError(f"{p['json']}: AUROC differs from predictions")
        runs.append(run)
        scores.append(prediction)
        signature.append(read_scores(base.prediction_path("within", variant, seed), data["test"]["ids"], data["test"]["labels"]))
    metric_names = ("auroc", "auprc", "balanced_accuracy", "macro_f1")
    aggregate = {m: {"mean": float(np.mean([r["test"][m] for r in runs])),
                     "sd": float(np.std([r["test"][m] for r in runs], ddof=1))} for m in metric_names}
    baseline_auc = np.asarray([r["test"]["auroc"] for r in runs])
    signature_auc = np.asarray([roc_auc_score(data["test"]["labels"], s) for s in signature])
    draws = base.bootstrap_auc(data["test"]["labels"], {"baseline": np.stack(scores),
                                                          "signature": np.stack(signature)}, replicates=2000)
    ci = np.percentile(draws["baseline"] - draws["signature"], [2.5, 97.5]).tolist()
    summary = {"dataset": dataset, "variant": variant, "architecture": architecture, "seeds": list(SEEDS),
               "counts": COUNTS, "parameter_count": runs[0]["parameter_count"], "metrics": aggregate,
               "per_seed_auroc": baseline_auc.tolist(),
               "path_signature_auroc": {"mean": float(signature_auc.mean()), "sd": float(signature_auc.std(ddof=1))},
               "baseline_minus_path_signature_auroc": float(np.mean(baseline_auc - signature_auc)),
               "paired_per_seed_difference": (baseline_auc - signature_auc).tolist(),
               "paired_95_ci": ci, "bootstrap": "2000 stratified paired resamples of identical test examples; seed means"}
    atomic_json(summary, RESULTS / dataset / architecture / f"{variant}_aggregate.json")
    print(f"{dataset}/{variant}/{architecture}: {aggregate['auroc']['mean']:.4f} ± {aggregate['auroc']['sd']:.4f}; vs Path Signature {summary['baseline_minus_path_signature_auroc']:+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    return summary


def existing_mean(path, keys):
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    if isinstance(obj, dict) and "mean" in obj and "sd" in obj:
        return obj
    return None


def report():
    """Read only completed results; preserve genuinely missing cells."""
    table = {}
    for architecture in ARCHITECTURES:
        table[architecture] = {}
        for dataset in DATASETS:
            for variant in VARIANTS:
                path = RESULTS / dataset / architecture / f"{variant}_aggregate.json"
                table[architecture][f"{dataset}_{variant}"] = existing_mean(path, ("metrics", "auroc"))
    # Historical NQ results are read from saved summaries, never rerun.
    nq = {}
    nq["hidden_mlp"] = existing_mean(ROOT / "results/state_transition_ablation/summary.json", ("aggregate", "A_hidden", "auroc"))
    nq["cnn"] = existing_mean(ROOT / "results/sequence_baselines/summary.json", ("results", "cnn", "auroc"))
    nq["transformer"] = existing_mean(ROOT / "results/sequence_baselines/summary.json", ("results", "transformer", "auroc"))
    rows = {**{key: {"nq": nq[key], **value} for key, value in table.items()}}
    # Existing matched transition+geometry MLP and degree-3 signature context.
    rows["transition_geometry_mlp"] = {"nq": existing_mean(ROOT / "results/state_transition_ablation/summary.json", ("aggregate", "C_transition_geometry", "auroc"))}
    rows["path_signature_degree3"] = {"nq": existing_mean(ROOT / "results/path_signature/summary.json", ("results", "order3", "auroc"))}
    for dataset in DATASETS:
        for variant in VARIANTS:
            mlp = ROOT / f"results/{dataset}_mlp/{variant}_aggregate.json"
            rows["transition_geometry_mlp"][f"{dataset}_{variant}"] = existing_mean(mlp, ("auroc",))
            base, _ = context(dataset)
            ps_values = []
            for seed in SEEDS:
                path = base.evaluation_path("within", variant, seed)
                if not path.exists():
                    break
                ps_values.append(json.loads(path.read_text(encoding="utf-8"))["test"]["auroc"])
            rows["path_signature_degree3"][f"{dataset}_{variant}"] = (
                {"mean": float(np.mean(ps_values)), "sd": float(np.std(ps_values, ddof=1))}
                if len(ps_values) == len(SEEDS) else None)
    columns = ("nq", "triviaqa_exact", "triviaqa_semantic", "hotpotqa_exact", "hotpotqa_semantic")
    labels = {"hidden_mlp": "Hidden-State MLP", "cnn": "1D-CNN", "transformer": "Transformer Encoder",
              "transition_geometry_mlp": "Transition + Geometry MLP", "path_signature_degree3": "Path Signature (degree 3)"}
    def fmt(value):
        return f"{value['mean']:.4f} ± {value['sd']:.4f}" if value is not None else "—"
    lines = ["# Module 38: Essential baseline comparison", "", "NQ uses hybrid correctness labels; TriviaQA and HotpotQA use separate exact and semantic labels. All results are t=0 and use five seeds where available. NQ and existing MLP/Path Signature rows are historical saved experiments.", "",
             "| Model | NQ AUROC | TriviaQA Exact AUROC | TriviaQA Semantic AUROC | HotpotQA Exact AUROC | HotpotQA Semantic AUROC |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("hidden_mlp", "transition_geometry_mlp", "cnn", "transformer", "path_signature_degree3"):
        lines.append("| " + labels[name] + " | " + " | ".join(fmt(rows[name].get(column)) for column in columns) + " |")
    lines += ["", "## Paired comparisons with degree-3 Path Signature", "",
              "Positive differences favor the named baseline. Confidence intervals use 2,000 stratified paired bootstrap resamples of the same test examples, averaged over seeds.", "",
              "| Dataset | Labels | Baseline | Baseline minus Path Signature AUROC | Paired 95% CI |",
              "|---|---|---|---:|---:|"]
    comparisons = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for architecture in ARCHITECTURES:
                path = RESULTS / dataset / architecture / f"{variant}_aggregate.json"
                if not path.exists():
                    continue
                obj = json.loads(path.read_text(encoding="utf-8"))
                comparisons[f"{dataset}_{variant}_{architecture}"] = obj
                ci = obj["paired_95_ci"]
                lines.append(f"| {dataset} | {variant} | {labels[architecture]} | {obj['baseline_minus_path_signature_auroc']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    atomic_json({"module": 38, "main_table": rows, "comparisons": comparisons, "missing_means_untested_or_incomplete": True}, RESULTS / "summary.json")
    report_path = RESULTS / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {report_path}")


def hybrid_hidden_report():
    """Print the paper's three-model hybrid table from completed saved runs."""
    sources = {
        "nq_hidden": ROOT / "results/state_transition_ablation/summary.json",
        "nq_signature": ROOT / "results/path_signature/summary.json",
        "triviaqa_hidden": RESULTS / "triviaqa/hidden_mlp/semantic_aggregate.json",
        "hotpotqa_hidden": RESULTS / "hotpotqa/hidden_mlp/semantic_aggregate.json",
        "triviaqa_mlp": ROOT / "results/triviaqa_mlp/semantic_aggregate.json",
        "hotpotqa_mlp": ROOT / "results/hotpotqa_mlp/semantic_aggregate.json",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Complete both hybrid Hidden-State MLP runs and evaluations first: " + ", ".join(missing))
    def fmt(value):
        return f"{value['mean']:.4f} ± {value['sd']:.4f}"
    hidden = json.loads(sources["nq_hidden"].read_text(encoding="utf-8"))
    signature = json.loads(sources["nq_signature"].read_text(encoding="utf-8"))
    values = {}
    for dataset in DATASETS:
        aggregate = json.loads(sources[f"{dataset}_hidden"].read_text(encoding="utf-8"))
        if (aggregate.get("dataset"), aggregate.get("variant"), aggregate.get("architecture"),
                aggregate.get("seeds"), aggregate.get("counts")) != (
                    dataset, "semantic", "hidden_mlp", list(SEEDS), COUNTS):
            raise ValueError(f"{sources[f'{dataset}_hidden']}: incompatible run metadata")
        values[dataset] = aggregate
    lines = ["# Hybrid correctness: final Hidden-State MLP comparison", "",
             "All cells use five-seed test AUROC mean ± sample SD. NQ, TriviaQA and HotpotQA are separate datasets; TriviaQA and HotpotQA 'semantic' labels are hybrid exact/alias + judge labels.", "",
             "| Model | NQ Hybrid | TriviaQA Hybrid | HotpotQA Hybrid |",
             "|---|---:|---:|---:|",
             "| Hidden-State MLP | " + " | ".join((
                 fmt(hidden["aggregate"]["A_hidden"]["auroc"]),
                 fmt(values["triviaqa"]["metrics"]["auroc"]),
                 fmt(values["hotpotqa"]["metrics"]["auroc"]))) + " |",
             "| Transition + Geometry MLP | " + " | ".join((
                 fmt(hidden["aggregate"]["C_transition_geometry"]["auroc"]),
                 fmt(json.loads(sources["triviaqa_mlp"].read_text(encoding="utf-8"))["auroc"]),
                 fmt(json.loads(sources["hotpotqa_mlp"].read_text(encoding="utf-8"))["auroc"]))) + " |",
             "| Path Signature (degree 3) | " + " | ".join((
                 fmt(signature["results"]["order3"]["auroc"]),
                 fmt(values["triviaqa"]["path_signature_auroc"]),
                 fmt(values["hotpotqa"]["path_signature_auroc"]))) + " |",
             "", "## New Hidden-State MLP metrics and paired comparison", "",
             "Positive AUROC differences favor Hidden-State MLP. The 95% intervals use paired stratified bootstrap samples of the same test questions, averaged across seeds.", "",
             "| Dataset | AUPRC | Balanced accuracy | Macro-F1 | Hidden-State MLP minus Path Signature AUROC | Paired 95% CI |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        item = values[dataset]
        metrics = item["metrics"]
        ci = item["paired_95_ci"]
        lines.append(f"| {dataset} | {fmt(metrics['auprc'])} | {fmt(metrics['balanced_accuracy'])} | "
                     f"{fmt(metrics['macro_f1'])} | {item['baseline_minus_path_signature_auroc']:+.4f} | "
                     f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += ["", "Sources:"] + [f"- {name}: `{path.relative_to(ROOT)}`" for name, path in sources.items()]
    output = RESULTS / "hybrid_hidden_mlp_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "train", "evaluate", "report", "hybrid-report"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--model", choices=ARCHITECTURES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    if args.stage == "report":
        report()
        return
    if args.stage == "hybrid-report":
        hybrid_hidden_report()
        return
    if not args.dataset or not args.variant or not args.model:
        parser.error("--dataset, --variant and --model are required for validate/train/evaluate")
    if args.stage == "evaluate":
        evaluate(args.dataset, args.variant, args.model)
        return
    base, _ = context(args.dataset)
    sample, labels = load_labels(base, args.variant)
    if args.stage == "validate":
        print(f"Validated {args.dataset}/{args.variant} sample and labels: {sample['counts']}")
        if args.model == "hidden_mlp":
            for split in SPLITS:
                row = sample["rows"][split][0]
                print(f"{split} hidden shape: {hidden_states(base, row, split).shape}")
        else:
            for split in SPLITS:
                row = sample["rows"][split][0]
                item = torch.load(base.path_for("transitions", split, row["sample_id"], ".pt"), map_location="cpu", weights_only=False)
                print(f"{split} transition shape: {tuple(item['transition_vectors'].shape)}")
        return
    if any(seed not in SEEDS for seed in args.seeds) or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must be unique members of 42..46")
    features = feature_data(args.dataset, args.model, sample)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        train_one(args.dataset, args.variant, args.model, seed, features, labels, device)


if __name__ == "__main__":
    main()
