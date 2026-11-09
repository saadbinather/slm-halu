"""Matched t=0 node-MLP ablation: hidden states versus layer transitions.

Uses saved, train-fitted PCA-16 representations. Each representation gets a
train-only StandardScaler and a learned 16-dimensional input projection. The
same two-layer node MLP, mean pooling and validation protocol is used throughout.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


STATE_ROOT = Path("processed_features")
TRANSITION_ROOT = Path("processed_features/transitions_pca16")
MODEL_ROOT = Path("models/state_transition_ablation")
RESULT_ROOT = Path("results/state_transition_ablation")
SPLITS = ("train", "val", "test")
ARMS = ("A_hidden", "B_transition", "C_transition_geometry", "D_hidden_transition")
SEEDS = (42, 43, 44, 45, 46)
COUNTS = {"train": 1858, "val": 398, "test": 399}
PREVIOUS = {"hidden_flat_mlp": (0.7961, 0.0081), "transition_node_mlp": (0.8097, 0.0049)}


def atomic_save_json(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temp, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    data = {}
    all_ids = set()
    for split in SPLITS:
        state_files = sorted((STATE_ROOT / split).glob("*.pt"))
        transition_files = sorted((TRANSITION_ROOT / split).glob("*.pt"))
        if len(state_files) != COUNTS[split] or [p.name for p in state_files] != [p.name for p in transition_files]:
            raise ValueError(f"{split}: missing or misaligned saved features")
        states, transitions, geometry, labels, ids = [], [], [], [], []
        for state_path, transition_path in tqdm(zip(state_files, transition_files), total=len(state_files), desc=f"Load {split}"):
            s = torch.load(state_path, map_location="cpu", weights_only=False)
            t = torch.load(transition_path, map_location="cpu", weights_only=False)
            if s["sample_id"] != t["sample_id"] or s["split"] != split or t["split"] != split or int(s["label"]) != int(t["label"]):
                raise ValueError(f"{state_path.name}: ID, label or split mismatch")
            sid = str(s["sample_id"])
            if sid in all_ids:
                raise ValueError(f"Repeated ID across splits: {sid}")
            all_ids.add(sid)
            a = s["features"][list(s["time_positions"]).index(0)].float().numpy()
            b = t["transition_features"][list(t["time_positions"]).index(0)].float().numpy()
            g = t["geometric_features"][list(t["time_positions"]).index(0), :, :4].float().numpy()
            if a.shape != (28, 16) or b.shape != (27, 16) or g.shape != (27, 4):
                raise ValueError(f"{sid}: unexpected t=0 feature dimensions")
            if not all(np.isfinite(x).all() for x in (a, b, g)):
                raise ValueError(f"{sid}: nonfinite features")
            states.append(a)
            transitions.append(b)
            geometry.append(g)
            labels.append(int(s["label"]))
            ids.append(sid)
        data[split] = {
            "states": np.stack(states), "transitions": np.stack(transitions),
            "geometry": np.stack(geometry), "labels": np.asarray(labels, dtype=np.int64), "ids": ids,
        }
    return data


def arm_features(part, arm):
    state, trans, geo = part["states"], part["transitions"], part["geometry"]
    if arm == "A_hidden":
        return state
    if arm == "B_transition":
        return trans
    if arm == "C_transition_geometry":
        return np.concatenate((trans, geo), axis=2)
    if arm == "D_hidden_transition":
        # Pair source-layer state h_l with its outgoing transition h_(l+1)-h_l.
        return np.concatenate((state[:, :-1, :], trans), axis=2)
    raise ValueError(arm)


def scaled_arm(data, arm):
    raw = {split: arm_features(data[split], arm) for split in SPLITS}
    scaler = StandardScaler()
    train = raw["train"]
    scaler.fit(train.reshape(-1, train.shape[-1]))
    scaled = {}
    for split, values in raw.items():
        transformed = scaler.transform(values.reshape(-1, values.shape[-1]))
        scaled[split] = np.asarray(transformed.reshape(values.shape), dtype=np.float32)
        if not np.isfinite(scaled[split]).all():
            raise ValueError(f"{arm}/{split}: scaled features are nonfinite")
    scaler.training_split_ = "train"
    scaler.training_examples_ = len(train)
    scaler.feature_policy_ = "one scaler per input channel, pooled across train nodes"
    joblib.dump(scaler, MODEL_ROOT / f"{arm}_scaler.joblib")
    return scaled


class MatchedNodeMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.30):
        super().__init__()
        self.projection = nn.Linear(input_dim, 16)
        self.linear1 = nn.Linear(17, 64)
        self.norm1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 64)
        self.norm2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, features):
        batch, count, width = features.shape
        x = self.projection(features.reshape(-1, width))
        positions = torch.linspace(0, 1, count, device=x.device).repeat(batch).unsqueeze(1)
        x = torch.cat((x, positions), dim=1)
        x = self.dropout(torch.relu(self.norm1(self.linear1(x))))
        x = self.dropout(torch.relu(self.norm2(self.linear2(x))))
        return self.classifier(x.reshape(batch, count, 64).mean(dim=1))


def threshold_from_val(labels, probabilities):
    best = (float("-inf"), 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        score = balanced_accuracy_score(labels, probabilities >= threshold)
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def metrics(labels, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "threshold": threshold,
    }


@torch.no_grad()
def predict(model, features, device):
    model.eval()
    probabilities = []
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=64):
        probabilities.extend(torch.softmax(model(batch.to(device)), dim=1)[:, 1].cpu().tolist())
    return np.asarray(probabilities, dtype=np.float64)


def save_predictions(path, ids, labels, probabilities, threshold):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "label", "prob_hallucinated", "prediction"))
        for sid, label, probability in zip(ids, labels, probabilities):
            writer.writerow((sid, int(label), float(probability), int(probability >= threshold)))
    os.replace(temp, path)


def run_seed(data, scaled, arm, seed, args, device):
    name = f"{arm}_seed{seed}"
    result_path = RESULT_ROOT / f"{name}.json"
    model_path = MODEL_ROOT / f"{name}.pt"
    prediction_path = RESULT_ROOT / f"{name}_test_predictions.csv"
    if result_path.exists() and model_path.exists() and prediction_path.exists() and not args.overwrite:
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        saved = list(csv.DictReader(prediction_path.open(newline="", encoding="utf-8")))
        if [row["sample_id"] for row in saved] != data["test"]["ids"]:
            raise ValueError(f"{name}: existing predictions do not match test IDs")
        print(f"{name}: existing AUROC={result['test']['auroc']:.4f}", flush=True)
        return result

    set_seed(seed)
    model = MatchedNodeMLP(scaled["train"].shape[-1]).to(device)
    labels_train = data["train"]["labels"]
    counts = np.bincount(labels_train, minlength=2)
    weights = torch.as_tensor(len(labels_train) / (2.0 * counts), device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(scaled["train"]), torch.from_numpy(labels_train)), batch_size=64, shuffle=True, generator=generator)
    best_auroc, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        val_probability = predict(model, scaled["val"], device)
        auroc = roc_auc_score(data["val"]["labels"], val_probability)
        if auroc > best_auroc:
            best_auroc, best_epoch, stale = float(auroc), epoch, 0
            temp = model_path.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed, "best_epoch": epoch, "val_auroc": best_auroc, "input_dim": scaled["train"].shape[-1]}, temp)
            os.replace(temp, model_path)
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["state_dict"])
    val_probability = predict(model, scaled["val"], device)
    threshold = threshold_from_val(data["val"]["labels"], val_probability)
    test_probability = predict(model, scaled["test"], device)
    result = {
        "arm": arm, "seed": seed, "best_epoch": best_epoch, "epochs_trained": epoch,
        "validation": metrics(data["val"]["labels"], val_probability, threshold),
        "test": metrics(data["test"]["labels"], test_probability, threshold),
        "model": str(model_path), "predictions": str(prediction_path),
    }
    save_predictions(prediction_path, data["test"]["ids"], data["test"]["labels"], test_probability, threshold)
    atomic_save_json(result, result_path)
    print(f"{name}: best_epoch={best_epoch} test_AUROC={result['test']['auroc']:.4f}", flush=True)
    return result


def bootstrap_differences(data, runs, seeds, replicates):
    labels = data["test"]["labels"]
    ids = data["test"]["ids"]
    predictions = {}
    for arm in ARMS:
        values = []
        for seed in seeds:
            with (RESULT_ROOT / f"{arm}_seed{seed}_test_predictions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if [row["sample_id"] for row in rows] != ids or [int(row["label"]) for row in rows] != labels.tolist():
                raise ValueError(f"{arm}/seed{seed}: prediction alignment failure")
            values.append([float(row["prob_hallucinated"]) for row in rows])
        predictions[arm] = np.asarray(values)

    rng = np.random.default_rng(260042)
    samples = []
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            samples.append(indices)

    paired = {}
    baseline = np.asarray([runs["A_hidden"][i]["test"]["auroc"] for i in range(len(seeds))])
    for arm in ARMS[1:]:
        candidate = np.asarray([runs[arm][i]["test"]["auroc"] for i in range(len(seeds))])
        differences = candidate - baseline
        draws = [
            float(np.mean([
                roc_auc_score(labels[idx], predictions[arm][i, idx]) - roc_auc_score(labels[idx], predictions["A_hidden"][i, idx])
                for i in range(len(seeds))
            ])) for idx in samples
        ]
        low, high = np.percentile(draws, [2.5, 97.5])
        paired[arm] = {
            "per_seed_auroc_difference": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "bootstrap_95_ci": [float(low), float(high)],
            "reliably_positive": bool(low > 0),
        }
    return paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.seeds or args.bootstrap < 100 or args.epochs < 1 or args.patience < 1:
        parser.error("invalid seeds, bootstrap count, epochs or patience")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    runs = {}
    for arm in ARMS:
        scaled = scaled_arm(data, arm)
        runs[arm] = [run_seed(data, scaled, arm, seed, args, device) for seed in args.seeds]
    paired = bootstrap_differences(data, runs, args.seeds, args.bootstrap)
    aggregate = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
            values = np.asarray([run["test"][metric] for run in runs[arm]])
            aggregate[arm][metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    report = {
        "module": 26, "seeds": args.seeds, "counts": COUNTS, "t": 0,
        "representation": {
            "A_hidden": "28 PCA-16 states + normalized position",
            "B_transition": "27 PCA-16 transitions + normalized position",
            "C_transition_geometry": "27 PCA-16 transitions + four raw geometric features + normalized position",
            "D_hidden_transition": "27 paired source states and transitions + normalized position",
        },
        "training": {"model": "16D learned input projection, node MLP 17->64->64, mean pooling, 64->32->2", "batch_size": 64, "dropout": 0.30, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "patience": args.patience, "max_epochs": args.epochs, "selection": "best validation AUROC, validation balanced-accuracy threshold", "scaling": "per-arm StandardScaler fitted on training nodes only", "pca": "previously fitted on train only"},
        "limitations": "A has 28 nodes; B/C/D have 27. Input projection has arm-dependent parameter count. Older hidden MLP was flattened rather than node pooled; older transition node MLP used different 22D input features.",
        "previous_auroc": PREVIOUS, "per_seed": runs, "aggregate": aggregate,
        "paired_bootstrap": {"replicates": args.bootstrap, "resampling": "same test examples across all arms and seeds", "comparisons_to_A": paired},
    }
    atomic_save_json(report, RESULT_ROOT / "summary.json")
    with (RESULT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "auroc_mean", "auroc_sd", "auprc_mean", "balanced_accuracy_mean", "macro_f1_mean", "delta_vs_A", "bootstrap_ci_low", "bootstrap_ci_high"))
        for arm in ARMS:
            comparison = paired.get(arm, {})
            ci = comparison.get("bootstrap_95_ci", ("", ""))
            writer.writerow((arm, aggregate[arm]["auroc"]["mean"], aggregate[arm]["auroc"]["sd"], aggregate[arm]["auprc"]["mean"], aggregate[arm]["balanced_accuracy"]["mean"], aggregate[arm]["macro_f1"]["mean"], comparison.get("mean_difference", 0.0), *ci))
    for arm in ARMS:
        a = aggregate[arm]["auroc"]
        print(f"{arm}: AUROC {a['mean']:.4f} +/- {a['sd']:.4f}; paired={paired.get(arm, {}).get('bootstrap_95_ci')}", flush=True)


if __name__ == "__main__":
    main()
