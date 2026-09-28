import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


FEATURE_DIR = Path("processed_features")
RESULT_DIR = Path("results")

SEED = 42


def get_files(split):
    folder = FEATURE_DIR / split

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(folder.glob("*.pt"))

    if not files:
        raise RuntimeError(f"No .pt files found in {folder}")

    return files


def load_t0_all_layers(split):
    """
    Returns:
        X: [N, 28, 16]
        y: [N]
    """
    X = []
    y = []

    for path in tqdm(
        get_files(split),
        desc=f"Loading t=0 {split}"
    ):
        trace = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        label = int(trace["label"])

        if label not in [0, 1]:
            continue

        times = [
            int(t)
            for t in trace["time_positions"]
        ]

        if 0 not in times:
            raise RuntimeError(
                f"{path.name} has no t=0"
            )

        t0_idx = times.index(0)

        # [28, 16]
        t0 = (
            trace["features"][t0_idx]
            .float()
            .numpy()
        )

        X.append(t0)
        y.append(label)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64)
    )


def find_best_threshold(y_true, probs):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(
        0.05,
        0.95,
        181
    ):
        preds = (
            probs >= threshold
        ).astype(np.int64)

        score = balanced_accuracy_score(
            y_true,
            preds
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def compute_metrics(
    y_true,
    probs,
    threshold
):
    preds = (
        probs >= threshold
    ).astype(np.int64)

    return {
        "threshold":
            float(threshold),

        "auroc":
            float(
                roc_auc_score(
                    y_true,
                    probs
                )
            ),

        "auprc_hallucinated":
            float(
                average_precision_score(
                    y_true,
                    probs
                )
            ),

        "auprc_correct":
            float(
                average_precision_score(
                    1 - y_true,
                    1.0 - probs
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    preds,
                    average="macro"
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    preds
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    preds
                )
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    args = parser.parse_args()

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("t=0 LAYER-WISE HALLUCINATION PROBES")
    print("=" * 70)

    X_train, y_train = load_t0_all_layers(
        "train"
    )

    X_val, y_val = load_t0_all_layers(
        "val"
    )

    X_test, y_test = load_t0_all_layers(
        "test"
    )

    print(
        f"\nTrain: {X_train.shape}"
    )

    print(
        f"Val:   {X_val.shape}"
    )

    print(
        f"Test:  {X_test.shape}"
    )

    num_layers = X_train.shape[1]
    pca_dim = X_train.shape[2]

    print(
        f"\nLayers: {num_layers}"
    )

    print(
        f"PCA dims/layer: {pca_dim}"
    )

    results = []

    for layer_idx in range(
        num_layers
    ):
        # Each probe gets ONLY one layer at t=0.
        train_x = X_train[
            :,
            layer_idx,
            :
        ]

        val_x = X_val[
            :,
            layer_idx,
            :
        ]

        test_x = X_test[
            :,
            layer_idx,
            :
        ]

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=args.seed
                )
            )
        ])

        model.fit(
            train_x,
            y_train
        )

        val_probs = (
            model.predict_proba(
                val_x
            )[:, 1]
        )

        threshold = find_best_threshold(
            y_val,
            val_probs
        )

        test_probs = (
            model.predict_proba(
                test_x
            )[:, 1]
        )

        val_metrics = compute_metrics(
            y_val,
            val_probs,
            threshold
        )

        test_metrics = compute_metrics(
            y_test,
            test_probs,
            threshold
        )

        result = {
            # Human-readable layer numbering: L1 ... L28
            "layer":
                int(layer_idx + 1),

            # Zero-based tensor index as well.
            "layer_index":
                int(layer_idx),

            "validation":
                val_metrics,

            "test":
                test_metrics,
        }

        results.append(result)

        print(
            f"L{layer_idx + 1:02d} | "
            f"val AUROC={val_metrics['auroc']:.4f} | "
            f"test AUROC={test_metrics['auroc']:.4f}"
        )

    # -----------------------------------------------------
    # Best layer selected by VALIDATION only
    # -----------------------------------------------------

    best_by_val = max(
        results,
        key=lambda r: r["validation"]["auroc"]
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\nBest layer by VALIDATION AUROC: "
        f"L{best_by_val['layer']}"
    )

    print(
        f"Val AUROC:  "
        f"{best_by_val['validation']['auroc']:.4f}"
    )

    print(
        f"Test AUROC: "
        f"{best_by_val['test']['auroc']:.4f}"
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = (
        RESULT_DIR
        / f"step13_t0_layerwise_seed{args.seed}.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "layer",
                "layer_index",
                "val_auroc",
                "test_auroc",
                "test_auprc_hallucinated",
                "test_auprc_correct",
                "test_macro_f1",
                "test_balanced_accuracy",
                "test_accuracy",
                "threshold",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r["layer"],
                    r["layer_index"],
                    r["validation"]["auroc"],
                    r["test"]["auroc"],
                    r["test"]["auprc_hallucinated"],
                    r["test"]["auprc_correct"],
                    r["test"]["macro_f1"],
                    r["test"]["balanced_accuracy"],
                    r["test"]["accuracy"],
                    r["test"]["threshold"],
                ]
            )

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    json_path = (
        RESULT_DIR
        / f"step13_t0_layerwise_seed{args.seed}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "seed":
                    int(args.seed),

                "scope":
                    "t=0 only; one logistic-regression probe per transformer layer",

                "num_layers":
                    int(num_layers),

                "pca_dim":
                    int(pca_dim),

                "best_layer_by_validation":
                    best_by_val,

                "results":
                    results,
            },
            f,
            indent=2
        )

    print(
        f"\nCSV:\n{csv_path}"
    )

    print(
        f"\nJSON:\n{json_path}"
    )


if __name__ == "__main__":
    main()
