"""Module 33: t=0 sequence baselines on saved PCA-16 transition graphs.

All models see the same ordered [27, 22] node matrix: 16 train-fitted PCA
coordinates, four train-standardized geometry coordinates, layer position,
and the zero t=0 time coordinate. No preprocessing is fitted here.
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = ROOT / "processed_graphs/transitions_pca16"
MODEL_ROOT = ROOT / "models/sequence_baselines"
RESULT_ROOT = ROOT / "results/sequence_baselines"
COUNTS = {"train": 1858, "val": 398, "test": 399}
ARCHITECTURES = ("cnn", "bilstm", "transformer")
SEEDS = (42, 43, 44, 45, 46)
WIDTH = 64
BATCH_SIZE = 64
DROPOUT = 0.30
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
REFERENCE = {
    "path_signature_order3_t0": {"auroc_mean": 0.811284988303622,
        "source": "results/path_signature/summary.json", "historical": True},
    "matched_transition_geometry_mlp_t0": {"auroc_mean": 0.8094942324755989,
        "source": "results/state_transition_ablation/summary.json", "historical": True},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    os.replace(tmp, path)


def load_data():
    data, seen = {}, set()
    for split, count in COUNTS.items():
        files = sorted((GRAPH_ROOT / split).glob("*.pt"))
        if len(files) != count:
            raise ValueError(f"{split}: expected {count} graphs, found {len(files)}")
        xs, ys, ids = [], [], []
        for path in files:
            graph = torch.load(path, map_location="cpu", weights_only=False)
            sid = str(graph.sample_id)
            if graph.split != split or sid in seen:
                raise ValueError(f"{path}: split mismatch or duplicate sample ID")
            seen.add(sid)
            keep = graph.time_position == 0
            x = graph.x[keep].float()
            if x.shape != (27, 22) or graph.transition_index[keep].tolist() != list(range(27)):
                raise ValueError(f"{path}: t=0 layer order or shape mismatch")
            if not torch.isfinite(x).all() or not torch.allclose(x[:, 20], torch.linspace(0, 1, 27), atol=1e-6):
                raise ValueError(f"{path}: nonfinite or invalid layer positions")
            if not torch.all(x[:, 21] == 0):
                raise ValueError(f"{path}: future time coordinate at t=0")
            eos_times = graph.original_time_positions[graph.checkpoint_is_eos == 1].tolist()
            if 0 in eos_times:
                raise ValueError(f"{path}: t=0 marked EOS")
            label = int(graph.y.item())
            if label not in (0, 1):
                raise ValueError(f"{path}: invalid label")
            xs.append(x)
            ys.append(label)
            ids.append(sid)
        data[split] = {"x": torch.stack(xs), "y": torch.tensor(ys, dtype=torch.long), "ids": ids}
    if set(data["train"]["y"].tolist()) != {0, 1} or set(data["val"]["y"].tolist()) != {0, 1} or set(data["test"]["y"].tolist()) != {0, 1}:
        raise ValueError("Both labels are required in every split")
    return data


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(22, WIDTH)
        self.blocks = nn.Sequential(nn.Conv1d(WIDTH, WIDTH, 3, padding=1), nn.BatchNorm1d(WIDTH), nn.ReLU(),
                                    nn.Dropout(DROPOUT), nn.Conv1d(WIDTH, WIDTH, 3, padding=1),
                                    nn.BatchNorm1d(WIDTH), nn.ReLU(), nn.Dropout(DROPOUT))

    def forward(self, x):
        return self.blocks(self.input(x).transpose(1, 2)).mean(dim=2)


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(22, WIDTH)
        self.lstm = nn.LSTM(WIDTH, WIDTH // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, x):
        y, _ = self.lstm(self.input(x))
        return self.norm(y).mean(dim=1)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(22, WIDTH)
        self.position = nn.Embedding(27, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, nhead=4, dim_feedforward=128,
                                           dropout=DROPOUT, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)

    def forward(self, x):
        positions = torch.arange(27, device=x.device)
        return self.encoder(self.input(x) + self.position(positions)).mean(dim=1)


class Classifier(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.encoder = {"cnn": CNN, "bilstm": BiLSTM, "transformer": Transformer}[architecture]()
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(WIDTH, 32),
                                  nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(32, 2))

    def forward(self, x):
        return self.head(self.encoder(x))


def loader(split, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TensorDataset(split["x"], split["y"]), batch_size=BATCH_SIZE,
                      shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    labels, probabilities = [], []
    for x, y in batches:
        prob = model(x.to(device)).softmax(dim=1)[:, 1]
        probabilities.extend(prob.cpu().tolist())
        labels.extend(y.tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def threshold_from_validation(y, p):
    choices = np.linspace(.05, .95, 181)
    return float(max(choices, key=lambda t: balanced_accuracy_score(y, p >= t)))


def metrics(y, p, threshold):
    predicted = (p >= threshold).astype(np.int64)
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro")),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "threshold": threshold}


def save_predictions(path, ids, y, p, threshold):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "label", "prob_hallucinated", "prediction", "threshold"))
        for sid, label, prob in zip(ids, y, p):
            writer.writerow((sid, int(label), float(prob), int(prob >= threshold), threshold))
    os.replace(tmp, path)


def run_one(data, architecture, seed, device):
    name = f"t0_{architecture}_seed{seed}"
    result_path = RESULT_ROOT / f"{name}.json"
    model_path = MODEL_ROOT / f"{name}.pt"
    val_path = RESULT_ROOT / f"{name}_val_predictions.csv"
    test_path = RESULT_ROOT / f"{name}_test_predictions.csv"
    paths = (result_path, model_path, val_path, test_path)
    if all(path.exists() for path in paths):
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("architecture") != architecture or result.get("seed") != seed:
            raise ValueError(f"{result_path}: mismatched saved run")
        return result
    if any(path.exists() for path in paths):
        raise FileExistsError(f"{name}: partial run exists; inspect/remove its files before rerunning")
    seed_everything(seed)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = Classifier(architecture).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    counts = np.bincount(data["train"]["y"].numpy(), minlength=2)
    weights = torch.tensor(len(data["train"]["y"]) / (2 * counts), device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_batches = loader(data["train"], True, seed)
    val_batches = loader(data["val"])
    test_batches = loader(data["test"])
    best, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
        val_y, val_p = predict(model, val_batches, device)
        auc = float(roc_auc_score(val_y, val_p))
        history.append({"epoch": epoch, "train_loss": total_loss / len(data["train"]["y"]), "val_auroc": auc})
        if auc > best:
            best, best_epoch, stale = auc, epoch, 0
            tmp = model_path.with_suffix(".pt.tmp")
            torch.save({"model_state_dict": model.state_dict(), "architecture": architecture,
                        "seed": seed, "best_epoch": epoch, "parameter_count": parameter_count}, tmp)
            os.replace(tmp, model_path)
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
    val_y, val_p = predict(model, val_batches, device)
    threshold = threshold_from_validation(val_y, val_p)
    test_y, test_p = predict(model, test_batches, device)
    if not np.array_equal(val_y, data["val"]["y"].numpy()) or not np.array_equal(test_y, data["test"]["y"].numpy()):
        raise ValueError("Prediction order or labels changed")
    save_predictions(val_path, data["val"]["ids"], val_y, val_p, threshold)
    save_predictions(test_path, data["test"]["ids"], test_y, test_p, threshold)
    result = {"module": 33, "architecture": architecture, "seed": seed, "parameter_count": parameter_count,
              "input": "t=0, 27 ordered nodes x 22 saved train-preprocessed features",
              "counts": COUNTS, "best_epoch": best_epoch, "epochs_trained": epoch,
              "validation": metrics(val_y, val_p, threshold), "test": metrics(test_y, test_p, threshold),
              "runtime_seconds": time.perf_counter() - started,
              "peak_gpu_memory_mib": (torch.cuda.max_memory_allocated(device) / 1048576 if device.type == "cuda" else None),
              "model": str(model_path), "val_predictions": str(val_path), "test_predictions": str(test_path),
              "history": history}
    atomic_json(result, result_path)
    print(f"{name}: test AUROC={result['test']['auroc']:.4f}, parameters={parameter_count}", flush=True)
    return result


def write_summary():
    aggregate = {}
    for architecture in ARCHITECTURES:
        files = [RESULT_ROOT / f"t0_{architecture}_seed{seed}.json" for seed in SEEDS]
        runs = []
        for path in files:
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    runs.append(json.load(handle))
        if not runs:
            continue
        aggregate[architecture] = {"seeds": [run["seed"] for run in runs], "per_seed_files": [str(p) for p in files if p.exists()],
                                   "parameter_count": runs[0]["parameter_count"]}
        for key in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
            values = np.asarray([run["test"][key] for run in runs])
            aggregate[architecture][key] = {"mean": float(values.mean()),
                                             "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
    atomic_json({"module": 33, "preprocessing": "saved train-fitted PCA-16 and train-fitted geometry scaler; no new fit",
                 "protocol": {"batch_size": BATCH_SIZE, "dropout": DROPOUT, "lr": LR,
                              "weight_decay": WEIGHT_DECAY, "max_epochs": EPOCHS, "patience": PATIENCE,
                              "selection": "validation AUROC; threshold from validation balanced accuracy"},
                 "historical_references": REFERENCE, "results": aggregate}, RESULT_ROOT / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES + ("all",), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--validate-only", action="store_true", help="Check saved data and model shapes without training or writing")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")
    data = load_data()
    selected = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    if args.validate_only:
        with torch.no_grad():
            for architecture in selected:
                model = Classifier(architecture).eval()
                output = model(data["train"]["x"][:2])
                if output.shape != (2, 2) or not torch.isfinite(output).all():
                    raise RuntimeError(f"{architecture}: invalid forward pass")
                print(f"{architecture}: input=(27,22), output=(2,2), parameters={sum(p.numel() for p in model.parameters())}")
        print(f"Validated saved splits: {COUNTS}; no training or writes")
        return
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for architecture in selected:
        for seed in args.seeds:
            run_one(data, architecture, seed, device)
    write_summary()


if __name__ == "__main__":
    main()
