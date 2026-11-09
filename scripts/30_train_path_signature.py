"""Pre-generation path-signature probes for 27 ordered layer transitions.

Uses Chen's identity to compute exact truncated signatures of the piecewise
linear path, with an explicit zero basepoint. Five train-fitted PCA coordinates
plus layer position give signature widths 42 (order 2) and 258 (order 3).
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

GRAPH_ROOT = Path("processed_graphs/transitions_pca16")
MODEL_ROOT = Path("models/path_signature")
RESULT_ROOT = Path("results/path_signature")
SPLITS = ("train", "val", "test")
COUNTS = {"train": 1858, "val": 398, "test": 399}
SEEDS = (42, 43, 44, 45, 46)
CONFIGS = ("order2", "order3", "order3_geometry", "order3_shuffled")
SIGNATURE_DIM = 6
SHUFFLE = np.random.default_rng(30).permutation(27)
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
BASELINES = {"hidden_state_mlp_module26": [0.787174316366863, 0.0048264514829448575],
             "transition_geometry_mlp_module26": [0.8094942324755989, 0.005554350240756528],
             "transition_gcn_module25": [0.8105106074050173, 0.004144763511565718]}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def load_data():
    data, all_ids = {}, set()
    for split in SPLITS:
        files = sorted((GRAPH_ROOT / split).glob("*.pt"))
        if len(files) != COUNTS[split]:
            raise ValueError(f"{split}: expected {COUNTS[split]} graphs, found {len(files)}")
        features, labels, ids = [], [], []
        for path in tqdm(files, desc=f"Load {split} t=0", mininterval=2):
            graph = torch.load(path, map_location="cpu", weights_only=False)
            sample_id = str(graph.sample_id)
            if graph.split != split or sample_id in all_ids:
                raise ValueError(f"{path}: split mismatch or duplicate ID")
            all_ids.add(sample_id)
            keep = graph.time_position == 0
            x = graph.x[keep].float().numpy()
            indices = graph.transition_index[keep].tolist()
            if x.shape != (27, 22) or indices != list(range(27)):
                raise ValueError(f"{path}: expected 27 ordered 22D transition nodes")
            if not np.isfinite(x).all() or not np.allclose(x[:, 20], np.linspace(0, 1, 27), atol=1e-6):
                raise ValueError(f"{path}: nonfinite features or wrong layer position")
            if not np.allclose(x[:, 21], 0, atol=1e-7):
                raise ValueError(f"{path}: t=0 time coordinate is not zero")
            if graph.original_time_positions[graph.checkpoint_is_eos == 1].tolist() and 0 in graph.original_time_positions[graph.checkpoint_is_eos == 1].tolist():
                raise ValueError(f"{path}: t=0 marked EOS")
            label = int(graph.y.item())
            if label not in (0, 1):
                raise ValueError(f"{path}: invalid label")
            features.append(x)
            labels.append(label)
            ids.append(sample_id)
        data[split] = {"nodes": np.stack(features), "labels": np.asarray(labels, dtype=np.int64), "ids": ids}
    return data


def load_or_fit_projection(data):
    pca_path = MODEL_ROOT / "train_projection_16_to_5.joblib"
    scaler_path = MODEL_ROOT / "projected_node_scaler.joblib"
    if pca_path.exists() and scaler_path.exists():
        pca, scaler = joblib.load(pca_path), joblib.load(scaler_path)
        if getattr(pca, "training_split_", None) != "train" or getattr(scaler, "training_split_", None) != "train":
            raise ValueError("Saved projection lacks train-only provenance")
    else:
        train = data["train"]["nodes"][:, :, :16].reshape(-1, 16)
        pca = PCA(n_components=5, svd_solver="full")
        pca.fit(train)
        scaler = StandardScaler().fit(pca.transform(train))
        pca.training_split_ = scaler.training_split_ = "train"
        pca.training_examples_ = scaler.training_examples_ = COUNTS["train"]
        for obj, path in ((pca, pca_path), (scaler, scaler_path)):
            temporary = path.with_suffix(path.suffix + ".tmp")
            joblib.dump(obj, temporary)
            os.replace(temporary, path)
    if int(pca.n_features_in_) != 16 or int(pca.n_components_) != 5 or int(scaler.n_features_in_) != 5:
        raise ValueError("Projection dimensions mismatch")
    paths = {}
    for split in SPLITS:
        nodes = data[split]["nodes"]
        projected = scaler.transform(pca.transform(nodes[:, :, :16].reshape(-1, 16)))
        projected = projected.reshape(len(nodes), 27, 5)
        paths[split] = np.concatenate((projected, nodes[:, :, 20:21]), axis=2).astype(np.float64)
        if not np.isfinite(paths[split]).all():
            raise ValueError(f"{split}: nonfinite projected path")
    return paths, float(pca.explained_variance_ratio_.sum())


def signature(points, order, basepoint=True):
    """Exact degree-2/3 signature of a batch of piecewise linear paths.

    Each segment v contributes exp(v)=(1,v,v⊗v/2,v⊗v⊗v/6), multiplied via
    Chen's identity. A zero basepoint makes the initial state observable.
    """
    points = torch.as_tensor(points, dtype=torch.float64)
    if points.ndim != 3 or order not in (2, 3):
        raise ValueError("Expected [examples, points, channels] and order 2/3")
    if basepoint:
        points = torch.cat((torch.zeros_like(points[:, :1]), points), dim=1)
    increments = points[:, 1:] - points[:, :-1]
    batch, _, dim = increments.shape
    s1 = points.new_zeros((batch, dim))
    s2 = points.new_zeros((batch, dim, dim))
    s3 = points.new_zeros((batch, dim, dim, dim)) if order == 3 else None
    for index in range(increments.shape[1]):
        v = increments[:, index]
        vv = torch.einsum("bi,bj->bij", v, v)
        if order == 3:
            s3 = s3 + torch.einsum("bij,bk->bijk", s2, v) \
                + 0.5 * torch.einsum("bi,bj,bk->bijk", s1, v, v) \
                + torch.einsum("bi,bj,bk->bijk", v, v, v) / 6.0
        s2 = s2 + torch.einsum("bi,bj->bij", s1, v) + 0.5 * vv
        s1 = s1 + v
    parts = [s1.reshape(batch, -1), s2.reshape(batch, -1)]
    if order == 3:
        parts.append(s3.reshape(batch, -1))
    return torch.cat(parts, dim=1)


def smoke_test():
    v = torch.tensor([1.0, 2.0], dtype=torch.float64)
    one = signature(v.view(1, 1, 2), 3)[0]
    expected = torch.cat((v, torch.outer(v, v).flatten() / 2,
                          torch.einsum("i,j,k->ijk", v, v, v).flatten() / 6))
    if not torch.allclose(one, expected, atol=1e-12):
        raise RuntimeError("Straight-line signature is incorrect")
    subdivided = signature(torch.stack((v / 2, v)).unsqueeze(0), 3)[0]
    if not torch.allclose(one, subdivided, atol=1e-12):
        raise RuntimeError("Subdivision changed a straight-line signature")
    axis = signature(torch.tensor([[[1., 0.], [1., 1.]]], dtype=torch.float64), 2)[0]
    if not torch.allclose(axis[2:].reshape(2, 2), torch.tensor([[0.5, 1.0], [0.0, 0.5]], dtype=torch.float64)):
        raise RuntimeError("Chen concatenation/order test failed")
    # Raw signatures are translation invariant; adding a zero basepoint is an
    # intentional design choice that preserves the first transition value.
    points = torch.tensor([[[1., 0.], [1., 1.]]], dtype=torch.float64)
    shift = torch.tensor([[[3., 2.]]], dtype=torch.float64)
    if not torch.allclose(signature(points, 3, False), signature(points + shift, 3, False)):
        raise RuntimeError("Raw signature translation invariance failed")
    if torch.allclose(signature(points, 3), signature(points + shift, 3)):
        raise RuntimeError("Zero basepoint failed to encode initial value")
    model = SignatureMLP(258)
    result = model(torch.randn(4, 258))
    result.sum().backward()
    if result.shape != (4, 2) or not torch.isfinite(result).all() or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Classifier output or gradients invalid")
    print("Smoke test passed: straight segment, subdivision, path order, translation, finite gradients", flush=True)


def prepare_features(data, paths, config):
    order = 2 if config == "order2" else 3
    raw = {}
    for split in SPLITS:
        points = paths[split][:, SHUFFLE, :] if config == "order3_shuffled" else paths[split]
        values = signature(points, order).numpy()
        if config == "order3_geometry":
            geometry = data[split]["nodes"][:, :, 16:20].astype(np.float64)
            summary = np.concatenate((geometry.mean(axis=1), geometry.std(axis=1)), axis=1)
            values = np.concatenate((values, summary), axis=1)
        raw[split] = values
    width = {"order2": 42, "order3": 258, "order3_geometry": 266, "order3_shuffled": 258}[config]
    if any(value.shape[1] != width or not np.isfinite(value).all() for value in raw.values()):
        raise ValueError(f"{config}: wrong dimension or nonfinite signature")
    scaler_path = MODEL_ROOT / f"{config}_train_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if getattr(scaler, "training_split_", None) != "train" or scaler.n_features_in_ != width:
            raise ValueError(f"{config}: invalid existing scaler")
    else:
        scaler = StandardScaler().fit(raw["train"])
        scaler.training_split_ = "train"
        scaler.training_examples_ = COUNTS["train"]
        temporary = scaler_path.with_suffix(".joblib.tmp")
        joblib.dump(scaler, temporary)
        os.replace(temporary, scaler_path)
    transformed = {split: scaler.transform(raw[split]).astype(np.float32) for split in SPLITS}
    if any(not np.isfinite(value).all() for value in transformed.values()):
        raise ValueError(f"{config}: nonfinite standardized signature")
    return transformed, width


class SignatureMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.30),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.30), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, features, device):
    model.eval()
    scores = []
    for (batch,) in DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=BATCH_SIZE):
        scores.extend(torch.softmax(model(batch.to(device)), dim=1)[:, 1].cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def select_threshold(labels, scores):
    grid = np.linspace(0.05, 0.95, 181)
    return float(max(grid, key=lambda threshold: balanced_accuracy_score(labels, scores >= threshold)))


def metrics(labels, scores, threshold):
    decisions = scores >= threshold
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
            "macro_f1": float(f1_score(labels, decisions, average="macro")),
            "threshold": threshold}


def save_predictions(path, ids, labels, scores, threshold):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "label", "prob_hallucinated", "prediction", "threshold"))
        for sample_id, label, score in zip(ids, labels, scores):
            writer.writerow((sample_id, int(label), float(score), int(score >= threshold), threshold))
    os.replace(temporary, path)


def run_one(data, features, config, width, seed, device, overwrite):
    name = f"{config}_seed{seed}"
    result_path = RESULT_ROOT / f"{name}.json"
    model_path = MODEL_ROOT / f"{name}.pt"
    val_path = RESULT_ROOT / f"{name}_val_predictions.csv"
    test_path = RESULT_ROOT / f"{name}_test_predictions.csv"
    if not overwrite and all(path.exists() for path in (result_path, model_path, val_path, test_path)):
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result["config"] != config or result["seed"] != seed or result["feature_dim"] != width:
            raise ValueError(f"{name}: saved metadata mismatch")
        print(f"Skipping {name}: AUROC={result['test']['auroc']:.4f}", flush=True)
        return result
    seed_all(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = SignatureMLP(width).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    labels_train = data["train"]["labels"]
    counts = np.bincount(labels_train, minlength=2)
    weights = torch.as_tensor(len(labels_train) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(features["train"]), torch.from_numpy(labels_train)),
                              batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    best_auc, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        for batch, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: nonfinite loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"{name}: nonfinite gradient")
            optimizer.step()
            total += loss.item() * len(labels)
        val_scores = predict(model, features["val"], device)
        auc = float(roc_auc_score(data["val"]["labels"], val_scores))
        history.append({"epoch": epoch, "train_loss": total / len(labels_train), "val_auroc": auc})
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            temporary = model_path.with_suffix(".pt.tmp")
            torch.save({"model_state_dict": model.state_dict(), "config": config, "seed": seed,
                        "feature_dim": width, "epoch": epoch, "val_auroc": auc}, temporary)
            os.replace(temporary, model_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name} epoch={epoch} val_AUROC={auc:.4f} best={best_auc:.4f}", flush=True)
        if stale >= PATIENCE:
            break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
    val_scores = predict(model, features["val"], device)
    threshold = select_threshold(data["val"]["labels"], val_scores)
    test_scores = predict(model, features["test"], device)
    save_predictions(val_path, data["val"]["ids"], data["val"]["labels"], val_scores, threshold)
    save_predictions(test_path, data["test"]["ids"], data["test"]["labels"], test_scores, threshold)
    runtime = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    result = {"config": config, "seed": seed, "feature_dim": width, "parameter_count": parameter_count,
              "best_epoch": best_epoch, "epochs_trained": epoch, "best_val_auroc": best_auc,
              "validation": metrics(data["val"]["labels"], val_scores, threshold),
              "test": metrics(data["test"]["labels"], test_scores, threshold),
              "counts": COUNTS, "runtime_seconds": runtime, "peak_gpu_memory_mib": peak_mib,
              "checkpoint": str(model_path), "val_predictions": str(val_path),
              "test_predictions": str(test_path), "history": history}
    atomic_json(result, result_path)
    print(f"{name}: test_AUROC={result['test']['auroc']:.4f}, runtime={runtime:.1f}s, peak_gpu_mib={peak_mib}", flush=True)
    return result


def summarize(runs, device, projection_variance):
    groups = {}
    for config in CONFIGS:
        selected = [r for r in runs if r["config"] == config]
        if not selected:
            continue
        group = {"per_seed": selected, "seeds": [r["seed"] for r in selected],
                 "feature_dim": selected[0]["feature_dim"], "parameter_count": selected[0]["parameter_count"]}
        for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
            values = np.asarray([r["test"][metric] for r in selected])
            group[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
        groups[config] = group
    if "order3" in groups and "order3_shuffled" in groups:
        groups["order3_minus_shuffled_auroc"] = groups["order3"]["auroc"]["mean"] - groups["order3_shuffled"]["auroc"]["mean"]
    atomic_json({"module": 30, "device": str(device), "implementation": "PyTorch Chen identity, exact truncated piecewise-linear signature",
                 "path": "zero basepoint; five train-PCA transition coordinates scaled on train; normalized layer position",
                 "projection_explained_variance": projection_variance,
                 "geometry": "four saved standardized geometric features summarized by mean and standard deviation",
                 "shuffle_permutation": SHUFFLE.tolist(), "baselines_cross_experiment": BASELINES,
                 "results": groups}, RESULT_ROOT / "summary.json")
    with (RESULT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("config", "metric", "mean", "sd"))
        for config in CONFIGS:
            if config in groups:
                for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
                    writer.writerow((config, metric, groups[config][metric]["mean"], groups[config][metric]["sd"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("Provide unique seeds")
    if args.smoke_test:
        smoke_test()
        return
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    paths, variance = load_or_fit_projection(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; train-only projection explained variance={variance:.4%}", flush=True)
    runs = []
    for config in CONFIGS:
        features, width = prepare_features(data, paths, config)
        for seed in args.seeds:
            runs.append(run_one(data, features, config, width, seed, device, args.overwrite))
        summarize(runs, device, variance)
    summarize(runs, device, variance)


if __name__ == "__main__":
    main()
