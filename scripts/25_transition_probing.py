"""Module 25: validation-selected t=0 transition logistic probes.

Compares raw hidden-state layers, raw layer transitions, transition geometry,
and concatenated PCA-16 transitions. All scaling and model selection use only
train/validation data; test data is evaluated after selection.
"""

import argparse
import csv
import gc
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


RAW_ROOT = Path("raw_traces")
TRANSITION_ROOT = Path("processed_features/transitions")
PCA16_ROOT = Path("processed_features/transitions_pca16")
MODEL_ROOT = Path("models/transition_probing")
RESULT_ROOT = Path("results/transition_probing")

SPLITS = ("train", "val", "test")
SEEDS = (42, 43, 44, 45, 46)
C_GRID = (0.01, 0.1, 1.0, 10.0)
EXPECTED_COUNTS = {"train": 1858, "val": 398, "test": 399}
PREVIOUS_FINAL_LAYER_AUROC = 0.7890215374687425


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def atomic_json_dump(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def atomic_joblib_dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def source_files(root, split):
    files = sorted((root / split).glob("*.pt"))
    if len(files) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"{root / split}: expected {EXPECTED_COUNTS[split]} files, got {len(files)}"
        )
    return files


def build_manifest():
    manifest = {}
    split_ids = {}
    for split in SPLITS:
        roots = {
            "raw": source_files(RAW_ROOT, split),
            "transition": source_files(TRANSITION_ROOT, split),
            "pca16": source_files(PCA16_ROOT, split),
        }
        names = {key: [path.name for path in paths] for key, paths in roots.items()}
        if names["raw"] != names["transition"] or names["raw"] != names["pca16"]:
            raise ValueError(f"File alignment mismatch in split {split}")

        rows = []
        ids = set()
        for paths in zip(roots["raw"], roots["transition"], roots["pca16"]):
            raw = load_pt(paths[0])
            transition = load_pt(paths[1])
            pca16 = load_pt(paths[2])
            sample_ids = {str(item["sample_id"]) for item in (raw, transition, pca16)}
            labels = {int(item["label"]) for item in (raw, transition, pca16)}
            splits = {str(item["split"]) for item in (raw, transition, pca16)}
            if len(sample_ids) != 1 or len(labels) != 1 or splits != {split}:
                raise ValueError(f"Metadata alignment failure for {paths[0].name}")
            sample_id = sample_ids.pop()
            label = labels.pop()
            if sample_id in ids or label not in (0, 1):
                raise ValueError(f"Invalid/duplicate sample {sample_id}")
            if 0 not in [int(t) for t in raw["time_positions"]]:
                raise ValueError(f"{sample_id} has no t=0 raw state")
            if 0 not in [int(t) for t in transition["time_positions"]]:
                raise ValueError(f"{sample_id} has no t=0 transition")
            ids.add(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "raw": paths[0],
                    "transition": paths[1],
                    "pca16": paths[2],
                }
            )
        manifest[split] = rows
        split_ids[split] = ids

    if split_ids["train"] & split_ids["val"] or split_ids["train"] & split_ids["test"] or split_ids["val"] & split_ids["test"]:
        raise ValueError("Sample overlap between train/validation/test")
    return manifest


def labels_and_ids(rows):
    return (
        np.asarray([row["label"] for row in rows], dtype=np.int64),
        [row["sample_id"] for row in rows],
    )


def load_raw_layers(rows, description):
    output = []
    for row in tqdm(rows, desc=description):
        item = load_pt(row["raw"])
        times = [int(t) for t in item["time_positions"]]
        tensor = item["hidden_states"][times.index(0)].float()
        if tensor.shape != (28, 1536) or not torch.isfinite(tensor).all():
            raise ValueError(f"{row['sample_id']}: invalid raw layer tensor")
        output.append(tensor.numpy())
    return np.asarray(output, dtype=np.float32)


def load_raw_transitions(rows, description):
    vectors = []
    geometry = []
    for row in tqdm(rows, desc=description):
        item = load_pt(row["transition"])
        times = [int(t) for t in item["time_positions"]]
        index = times.index(0)
        transition = item["transition_vectors"][index].float()
        geometric = item["geometric_features"][index, :, :4].float()
        if transition.shape != (27, 1536) or geometric.shape != (27, 4):
            raise ValueError(f"{row['sample_id']}: invalid transition tensor")
        if not torch.isfinite(transition).all() or not torch.isfinite(geometric).all():
            raise ValueError(f"{row['sample_id']}: non-finite transition features")
        vectors.append(transition.numpy())
        geometry.append(geometric.numpy())
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(geometry, dtype=np.float32),
    )


def load_pca16_transitions(rows, description):
    vectors = []
    geometry = []
    for row in tqdm(rows, desc=description):
        item = load_pt(row["pca16"])
        times = [int(t) for t in item["time_positions"]]
        index = times.index(0)
        pca = item["transition_features"][index].float()
        geometric = item["geometric_features"][index, :, :4].float()
        if pca.shape != (27, 16) or geometric.shape != (27, 4):
            raise ValueError(f"{row['sample_id']}: invalid PCA-16 transition tensor")
        if not torch.isfinite(pca).all() or not torch.isfinite(geometric).all():
            raise ValueError(f"{row['sample_id']}: non-finite PCA features")
        vectors.append(pca.numpy())
        geometry.append(geometric.numpy())
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(geometry, dtype=np.float32),
    )


def find_threshold(labels, probabilities):
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(np.int64)
        score = balanced_accuracy_score(labels, predictions)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "auprc_hallucinated": float(average_precision_score(labels, probabilities)),
        "auprc_correct": float(average_precision_score(1 - labels, 1.0 - probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).astype(int).tolist(),
    }


def fit_candidate(train_x, train_y, val_x, val_y, c_value, seed):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(train_x, train_y)
    probabilities = pipeline.predict_proba(val_x)[:, 1]
    return pipeline, probabilities, float(roc_auc_score(val_y, probabilities))


def select_c_and_evaluate(train_x, train_y, val_x, val_y, test_x, test_y, seed):
    candidates = []
    for c_value in C_GRID:
        model, val_probabilities, val_auroc = fit_candidate(
            train_x, train_y, val_x, val_y, c_value, seed
        )
        candidates.append((val_auroc, c_value, model, val_probabilities))
    # Grid order resolves exact ties toward stronger regularization (smaller C).
    _, best_c, model, val_probabilities = max(candidates, key=lambda row: row[0])
    threshold = find_threshold(val_y, val_probabilities)
    test_probabilities = model.predict_proba(test_x)[:, 1]
    return {
        "best_c": float(best_c),
        "validation": compute_metrics(val_y, val_probabilities, threshold),
        "test": compute_metrics(test_y, test_probabilities, threshold),
        "model": model,
        "val_probabilities": val_probabilities,
        "test_probabilities": test_probabilities,
        "c_validation_aurocs": {
            str(c_value): float(auroc) for auroc, c_value, _, _ in candidates
        },
    }


def save_predictions(path, ids, labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prob_hallucinated", "prediction", "threshold"])
        for sample_id, label, probability, prediction in zip(ids, labels, probabilities, predictions):
            writer.writerow([sample_id, int(label), float(probability), int(prediction), threshold])
    os.replace(temporary, path)


def table_row(index_name, index, result):
    return {
        index_name: int(index),
        "best_c": result["best_c"],
        "validation": result["validation"],
        "test": result["test"],
        "c_validation_aurocs": result["c_validation_aurocs"],
    }


def write_probe_table(path, index_name, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                index_name, "best_c", "val_auroc", "test_auroc", "test_auprc",
                "test_balanced_accuracy", "test_macro_f1", "threshold",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row[index_name], row["best_c"], row["validation"]["auroc"],
                    row["test"]["auroc"], row["test"]["auprc"],
                    row["test"]["balanced_accuracy"], row["test"]["macro_f1"],
                    row["test"]["threshold"],
                ]
            )


def deterministic_seed_summary(result, seeds):
    metrics = ("auroc", "auprc", "balanced_accuracy", "macro_f1")
    per_seed = []
    for seed in seeds:
        per_seed.append(
            {
                "seed": int(seed),
                "best_c": result["best_c"],
                "validation": result["validation"],
                "test": result["test"],
            }
        )
    aggregate = {}
    for metric in metrics:
        values = np.asarray([entry["test"][metric] for entry in per_seed])
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return {"per_seed": per_seed, "aggregate": aggregate}


def run_indexed_family(name, index_name, train, val, test, labels, ids, seeds, add_geometry=None):
    rows = []
    best_models = []
    for index in range(train.shape[1]):
        train_x = train[:, index, :]
        val_x = val[:, index, :]
        test_x = test[:, index, :]
        if add_geometry is not None:
            train_x = np.concatenate([train_x, add_geometry[0][:, index, :]], axis=1)
            val_x = np.concatenate([val_x, add_geometry[1][:, index, :]], axis=1)
            test_x = np.concatenate([test_x, add_geometry[2][:, index, :]], axis=1)
        result = select_c_and_evaluate(
            train_x, labels[0], val_x, labels[1], test_x, labels[2], seeds[0]
        )
        row = table_row(index_name, index, result)
        rows.append(row)
        best_models.append(result["model"])
        atomic_joblib_dump(
            result["model"], MODEL_ROOT / name / f"{index_name}_{index:02d}.joblib"
        )
        print(
            f"{name} {index_name}={index:02d} C={result['best_c']:g} "
            f"val={result['validation']['auroc']:.4f} test={result['test']['auroc']:.4f}"
        )

    best_index = max(range(len(rows)), key=lambda idx: rows[idx]["validation"]["auroc"])
    best = rows[best_index]
    model = best_models[best_index]
    train_x = train[:, best_index, :]
    val_x = val[:, best_index, :]
    test_x = test[:, best_index, :]
    if add_geometry is not None:
        train_x = np.concatenate([train_x, add_geometry[0][:, best_index, :]], axis=1)
        val_x = np.concatenate([val_x, add_geometry[1][:, best_index, :]], axis=1)
        test_x = np.concatenate([test_x, add_geometry[2][:, best_index, :]], axis=1)
    val_probabilities = model.predict_proba(val_x)[:, 1]
    test_probabilities = model.predict_proba(test_x)[:, 1]
    threshold = best["validation"]["threshold"]
    save_predictions(
        RESULT_ROOT / f"{name}_best_val_predictions.csv", ids[1], labels[1], val_probabilities, threshold
    )
    save_predictions(
        RESULT_ROOT / f"{name}_best_test_predictions.csv", ids[2], labels[2], test_probabilities, threshold
    )
    write_probe_table(RESULT_ROOT / f"{name}.csv", index_name, rows)
    return {
        "selection_rule": f"best {index_name} by validation AUROC only",
        "best": best,
        "seed_results": deterministic_seed_summary(
            {"best_c": best["best_c"], "validation": best["validation"], "test": best["test"]},
            seeds,
        ),
        "table": rows,
    }


def run_combined(name, train_x, val_x, test_x, labels, ids, seeds):
    result = select_c_and_evaluate(
        train_x, labels[0], val_x, labels[1], test_x, labels[2], seeds[0]
    )
    atomic_joblib_dump(result["model"], MODEL_ROOT / f"{name}.joblib")
    threshold = result["validation"]["threshold"]
    save_predictions(
        RESULT_ROOT / f"{name}_val_predictions.csv",
        ids[1], labels[1], result["val_probabilities"], threshold,
    )
    save_predictions(
        RESULT_ROOT / f"{name}_test_predictions.csv",
        ids[2], labels[2], result["test_probabilities"], threshold,
    )
    compact = {
        "input_dim": int(train_x.shape[1]),
        "best_c": result["best_c"],
        "validation": result["validation"],
        "test": result["test"],
        "c_validation_aurocs": result["c_validation_aurocs"],
    }
    compact["seed_results"] = deterministic_seed_summary(compact, seeds)
    print(
        f"{name}: dim={train_x.shape[1]} C={result['best_c']:g} "
        f"val={result['validation']['auroc']:.4f} test={result['test']['auroc']:.4f}"
    )
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("At least one seed is required")

    summary_path = RESULT_ROOT / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"Existing completed result found: {summary_path}; use --overwrite to rerun")
        return

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    labels = tuple(labels_and_ids(manifest[split])[0] for split in SPLITS)
    ids = tuple(labels_and_ids(manifest[split])[1] for split in SPLITS)

    print("\nLoading raw hidden-state layers")
    raw_layers = tuple(
        load_raw_layers(manifest[split], f"Raw layers {split}") for split in SPLITS
    )
    layer_result = run_indexed_family(
        "individual_layers_raw", "layer_index",
        raw_layers[0], raw_layers[1], raw_layers[2], labels, ids, args.seeds,
    )
    del raw_layers
    gc.collect()

    print("\nLoading raw transition vectors and geometry")
    raw_loaded = tuple(
        load_raw_transitions(manifest[split], f"Raw transitions {split}") for split in SPLITS
    )
    raw_transitions = tuple(item[0] for item in raw_loaded)
    raw_geometry = tuple(item[1] for item in raw_loaded)
    transition_result = run_indexed_family(
        "individual_transitions_raw", "transition_index",
        raw_transitions[0], raw_transitions[1], raw_transitions[2], labels, ids, args.seeds,
    )
    transition_geometry_result = run_indexed_family(
        "individual_transitions_raw_geometry", "transition_index",
        raw_transitions[0], raw_transitions[1], raw_transitions[2], labels, ids, args.seeds,
        add_geometry=raw_geometry,
    )
    del raw_loaded, raw_transitions, raw_geometry
    gc.collect()

    print("\nLoading PCA-16 transition vectors and geometry")
    pca_loaded = tuple(
        load_pca16_transitions(manifest[split], f"PCA-16 transitions {split}") for split in SPLITS
    )
    pca = tuple(item[0] for item in pca_loaded)
    geometry = tuple(item[1] for item in pca_loaded)
    flattened = tuple(array.reshape(array.shape[0], -1) for array in pca)
    flattened_geometry = tuple(
        np.concatenate(
            [pca_array.reshape(pca_array.shape[0], -1), geometry_array.reshape(geometry_array.shape[0], -1)],
            axis=1,
        )
        for pca_array, geometry_array in zip(pca, geometry)
    )
    all_transition_result = run_combined(
        "all_transitions_pca16", flattened[0], flattened[1], flattened[2], labels, ids, args.seeds
    )
    all_transition_geometry_result = run_combined(
        "all_transitions_pca16_geometry",
        flattened_geometry[0], flattened_geometry[1], flattened_geometry[2],
        labels, ids, args.seeds,
    )

    report = {
        "module": 25,
        "scope": "t=0 pre-generation logistic-regression probing",
        "seeds": args.seeds,
        "c_grid": list(C_GRID),
        "solver": "lbfgs",
        "seed_note": (
            "LBFGS logistic regression is deterministic; random_state does not alter fits. "
            "Per-seed values are therefore identical and standard deviations are zero."
        ),
        "data_validation": {
            "counts": EXPECTED_COUNTS,
            "aligned_ids_labels_across_sources": True,
            "no_split_overlap": True,
            "t0_present_for_all_examples": True,
            "scaling": "StandardScaler fit separately on training features only",
        },
        "fairness_note": (
            "The previous 0.7890 final-layer baseline used train-fitted PCA-16 and fixed C=1. "
            "This module's individual-layer baseline uses raw 1536-D states and validation-selected C; "
            "it is not an exact preprocessing match."
        ),
        "previous_final_layer_pca16_fixed_c_test_auroc": PREVIOUS_FINAL_LAYER_AUROC,
        "individual_layers_raw": layer_result,
        "individual_transitions_raw": transition_result,
        "individual_transitions_raw_geometry": transition_geometry_result,
        "all_transitions_pca16": all_transition_result,
        "all_transitions_pca16_geometry": all_transition_geometry_result,
    }
    atomic_json_dump(report, summary_path)

    comparison_path = RESULT_ROOT / "comparison.csv"
    with open(comparison_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "selected_index", "best_c", "val_auroc", "test_auroc", "test_auprc", "balanced_accuracy", "macro_f1"])
        entries = [
            ("best_raw_layer", layer_result["best"].get("layer_index"), layer_result["best"]),
            ("best_raw_transition", transition_result["best"].get("transition_index"), transition_result["best"]),
            ("best_raw_transition_geometry", transition_geometry_result["best"].get("transition_index"), transition_geometry_result["best"]),
            ("all_transitions_pca16", "all", all_transition_result),
            ("all_transitions_pca16_geometry", "all", all_transition_geometry_result),
        ]
        for name, selected_index, entry in entries:
            writer.writerow([
                name, selected_index, entry["best_c"], entry["validation"]["auroc"],
                entry["test"]["auroc"], entry["test"]["auprc"],
                entry["test"]["balanced_accuracy"], entry["test"]["macro_f1"],
            ])

    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
