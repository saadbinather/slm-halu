from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


FEATURE_DIR = Path("processed_features")
RESULT_DIR = Path("results")

TIMES = [0, 1, 2, 4, 8]
SEED = 42


# =========================================================
# LOAD COMPLETE-CASE DATA
# =========================================================

def get_files(split):

    folder = FEATURE_DIR / split

    files = sorted(
        folder.glob("*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No files in {folder}"
        )

    return files


def load_complete_cases(split):

    X = []
    y = []
    ids = []

    for path in tqdm(
        get_files(split),
        desc=f"Loading {split}"
    ):

        trace = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        label = int(
            trace["label"]
        )

        if label not in [0, 1]:
            continue

        available_times = [
            int(t)
            for t in trace["time_positions"]
        ]

        # ---------------------------------------------
        # IMPORTANT:
        # only keep samples containing ALL checkpoints
        # ---------------------------------------------

        if not all(
            t in available_times
            for t in TIMES
        ):
            continue

        features = trace[
            "features"
        ].float()

        time_to_idx = {
            int(t): i
            for i, t in enumerate(
                available_times
            )
        }

        # Result:
        # [5 times, 28 layers, 16 PCA]
        sample = torch.stack(
            [
                features[
                    time_to_idx[t]
                ]
                for t in TIMES
            ],
            dim=0
        )

        X.append(
            sample.numpy()
        )

        y.append(label)

        ids.append(
            str(trace["sample_id"])
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    if len(X) == 0:
        raise RuntimeError(
            f"No complete cases in {split}"
        )

    return X, y, ids


# =========================================================
# ONE CELL = ONE LOGISTIC PROBE
# =========================================================

def fit_probe(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

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
                random_state=SEED
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    val_probs = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    test_probs = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    return (
        roc_auc_score(
            y_val,
            val_probs
        ),

        roc_auc_score(
            y_test,
            test_probs
        )
    )


# =========================================================
# PLOT
# =========================================================

def save_heatmap(
    matrix,
    title,
    path
):

    plt.figure(
        figsize=(8, 10)
    )

    im = plt.imshow(
        matrix,
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(
        im,
        label="AUROC"
    )

    plt.xticks(
        range(len(TIMES)),
        [f"t={t}" for t in TIMES]
    )

    plt.yticks(
        range(matrix.shape[0]),
        [
            f"L{i+1}"
            for i in range(
                matrix.shape[0]
            )
        ]
    )

    plt.xlabel(
        "Generation checkpoint"
    )

    plt.ylabel(
        "Transformer layer"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=250
    )

    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "LAYER × TIME EPISTEMIC PROBE"
    )
    print("=" * 70)

    X_train, y_train, _ = (
        load_complete_cases(
            "train"
        )
    )

    X_val, y_val, _ = (
        load_complete_cases(
            "val"
        )
    )

    X_test, y_test, _ = (
        load_complete_cases(
            "test"
        )
    )

    print(
        "\nComplete-case samples:"
    )

    print(
        f"Train: {len(y_train)}"
    )

    print(
        f"Val:   {len(y_val)}"
    )

    print(
        f"Test:  {len(y_test)}"
    )

    print(
        "\nShapes:"
    )

    print(
        f"Train: {X_train.shape}"
    )

    # Expected:
    # [N, 5, 28, 16]

    num_times = (
        X_train.shape[1]
    )

    num_layers = (
        X_train.shape[2]
    )

    val_matrix = np.zeros(
        (
            num_layers,
            num_times
        ),
        dtype=np.float32
    )

    test_matrix = np.zeros_like(
        val_matrix
    )

    rows = []

    # =====================================================
    # EVERY LAYER × TIME CELL
    # =====================================================

    for time_idx, time_value in enumerate(
        TIMES
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"TIME t={time_value}"
        )

        print(
            "=" * 60
        )

        for layer_idx in range(
            num_layers
        ):

            train_x = X_train[
                :,
                time_idx,
                layer_idx,
                :
            ]

            val_x = X_val[
                :,
                time_idx,
                layer_idx,
                :
            ]

            test_x = X_test[
                :,
                time_idx,
                layer_idx,
                :
            ]

            val_auc, test_auc = (
                fit_probe(
                    train_x,
                    y_train,
                    val_x,
                    y_val,
                    test_x,
                    y_test
                )
            )

            val_matrix[
                layer_idx,
                time_idx
            ] = val_auc

            test_matrix[
                layer_idx,
                time_idx
            ] = test_auc

            rows.append(
                {
                    "time":
                        int(time_value),

                    "layer":
                        int(
                            layer_idx + 1
                        ),

                    "layer_index":
                        int(layer_idx),

                    "val_auroc":
                        float(val_auc),

                    "test_auroc":
                        float(test_auc)
                }
            )

            print(
                f"L{layer_idx+1:02d} | "
                f"val={val_auc:.4f} | "
                f"test={test_auc:.4f}"
            )

    # =====================================================
    # BEST CELL:
    # SELECT USING VALIDATION ONLY
    # =====================================================

    best_flat = np.argmax(
        val_matrix
    )

    best_layer_idx, best_time_idx = (
        np.unravel_index(
            best_flat,
            val_matrix.shape
        )
    )

    best_time = TIMES[
        best_time_idx
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST CELL BY VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Layer: L{best_layer_idx + 1}"
    )

    print(
        f"Time:  t={best_time}"
    )

    print(
        f"Val AUROC:  "
        f"{val_matrix[best_layer_idx, best_time_idx]:.4f}"
    )

    print(
        f"Test AUROC: "
        f"{test_matrix[best_layer_idx, best_time_idx]:.4f}"
    )

    # =====================================================
    # BEST LAYER AT EACH TIME
    # =====================================================

    print(
        "\nBest validation layer at each time:"
    )

    for time_idx, t in enumerate(
        TIMES
    ):

        layer_idx = np.argmax(
            val_matrix[
                :,
                time_idx
            ]
        )

        print(
            f"t={t:<2} → "
            f"L{layer_idx+1:02d} | "
            f"val={val_matrix[layer_idx, time_idx]:.4f} | "
            f"test={test_matrix[layer_idx, time_idx]:.4f}"
        )

    # =====================================================
    # SAVE CSV
    # =====================================================

    csv_path = (
        RESULT_DIR
        / "step15_layer_time_heatmap.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "layer",
                "layer_index",
                "val_auroc",
                "test_auroc"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # =====================================================
    # SAVE JSON
    # =====================================================

    json_path = (
        RESULT_DIR
        / "step15_layer_time_heatmap.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "times":
                    TIMES,

                "complete_case_counts": {
                    "train":
                        int(
                            len(y_train)
                        ),

                    "val":
                        int(
                            len(y_val)
                        ),

                    "test":
                        int(
                            len(y_test)
                        )
                },

                "best_cell_by_validation": {
                    "layer":
                        int(
                            best_layer_idx
                            + 1
                        ),

                    "time":
                        int(
                            best_time
                        ),

                    "val_auroc":
                        float(
                            val_matrix[
                                best_layer_idx,
                                best_time_idx
                            ]
                        ),

                    "test_auroc":
                        float(
                            test_matrix[
                                best_layer_idx,
                                best_time_idx
                            ]
                        )
                },

                "rows":
                    rows
            },
            f,
            indent=2
        )

    # =====================================================
    # HEATMAPS
    # =====================================================

    save_heatmap(
        val_matrix,
        "Validation AUROC across layer × generation time",
        RESULT_DIR
        / "step15_layer_time_validation.png"
    )

    save_heatmap(
        test_matrix,
        "Test AUROC across layer × generation time",
        RESULT_DIR
        / "step15_layer_time_test.png"
    )

    print(
        f"\nCSV:\n{csv_path}"
    )

    print(
        "\nHeatmaps saved in results/"
    )


if __name__ == "__main__":
    main()