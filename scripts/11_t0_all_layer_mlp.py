import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

FEATURE_DIR = Path("processed_features")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# FILES
# =========================================================

def get_files(split):
    folder = FEATURE_DIR / split

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(folder.glob("*.pt"))

    if not files:
        raise RuntimeError(f"No .pt files found in {folder}")

    return files


def load_trace(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


# =========================================================
# BUILD FAIR t=0 ALL-LAYER DATASET
#
# Each sample:
#   t=0 only
#   all 28 transformer layers
#   16 PCA dims per layer
#
# Shape:
#   [28, 16] -> flatten -> 448-D
#
# This gives MLP exactly the same t=0 hidden-state information
# as the t=0 EpiGraph, but WITHOUT graph edges/message passing.
# =========================================================

def build_dataset(split):
    files = get_files(split)

    X = []
    y = []

    for path in tqdm(
        files,
        desc=f"t0 all-layer {split}"
    ):
        trace = load_trace(path)

        label = int(trace["label"])

        if label not in [0, 1]:
            continue

        features = trace["features"].float()
        times = [
            int(t)
            for t in trace["time_positions"]
        ]

        if 0 not in times:
            raise RuntimeError(
                f"{path.name} does not contain t=0"
            )

        t0_idx = times.index(0)

        # [28, 16]
        t0 = features[
            t0_idx
        ]

        # [448]
        vector = (
            t0
            .reshape(-1)
            .numpy()
        )

        X.append(vector)
        y.append(label)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64)
    )


# =========================================================
# METRICS
# =========================================================

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
    y_true = np.asarray(
        y_true,
        dtype=np.int64
    )

    probs = np.asarray(
        probs,
        dtype=np.float64
    )

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


def print_metrics(title, metrics):
    print("\n" + title)
    print("-" * len(title))

    print(
        f"AUROC:              "
        f"{metrics['auroc']:.4f}"
    )

    print(
        f"AUPRC hallucinated: "
        f"{metrics['auprc_hallucinated']:.4f}"
    )

    print(
        f"AUPRC correct:      "
        f"{metrics['auprc_correct']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Balanced accuracy:  "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Accuracy:            "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Threshold:           "
        f"{metrics['threshold']:.3f}"
    )


# =========================================================
# MLP
# Same hidden sizes as Step 07 trajectory MLP.
# =========================================================

class T0AllLayerMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden1=256,
        hidden2=64,
        dropout=0.30
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden1
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden1,
                hidden2
            ),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(
                hidden2,
                1
            )
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================================================
# PREDICT
# =========================================================

@torch.no_grad()
def predict(
    model,
    X,
    device,
    batch_size
):
    model.eval()

    x = torch.tensor(
        X,
        dtype=torch.float32
    )

    loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=False
    )

    probs = []

    for (batch_x,) in loader:
        batch_x = batch_x.to(device)

        logits = model(batch_x)

        probs.append(
            torch.sigmoid(
                logits
            ).cpu().numpy()
        )

    return np.concatenate(probs)


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    args = parser.parse_args()

    set_seed(args.seed)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "FAIR BASELINE — t=0 ALL-LAYER MLP"
    )
    print("=" * 70)

    # =====================================================
    # DATA
    # =====================================================

    X_train, y_train = build_dataset(
        "train"
    )

    X_val, y_val = build_dataset(
        "val"
    )

    X_test, y_test = build_dataset(
        "test"
    )

    print(
        f"\nTrain shape: {X_train.shape}"
    )

    print(
        f"Val shape:   {X_val.shape}"
    )

    print(
        f"Test shape:  {X_test.shape}"
    )

    print(
        "\nExpected input dimension = "
        "28 layers x 16 PCA = 448"
    )

    print(
        f"Actual input dimension = "
        f"{X_train.shape[1]}"
    )

    print(
        f"\nTrain labels: "
        f"correct={np.sum(y_train == 0)}, "
        f"hallucinated={np.sum(y_train == 1)}"
    )

    # =====================================================
    # TRAIN-ONLY SCALER
    # =====================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)

    scaler_path = (
        MODEL_DIR
        / f"t0_all_layer_mlp_scaler_seed{args.seed}.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    # =====================================================
    # DEVICE / MODEL
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    model = T0AllLayerMLP(
        input_dim=X_train.shape[1]
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{total_params:,}"
    )

    # =====================================================
    # WEIGHTED BCE
    # =====================================================

    num_negative = float(
        np.sum(y_train == 0)
    )

    num_positive = float(
        np.sum(y_train == 1)
    )

    pos_weight_value = (
        num_negative
        / max(num_positive, 1.0)
    )

    print(
        f"Positive-class weight: "
        f"{pos_weight_value:.4f}"
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight_value,
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # =====================================================
    # TRAIN LOADER
    # =====================================================

    train_x = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    train_y = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    train_loader = DataLoader(
        TensorDataset(
            train_x,
            train_y
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    checkpoint_path = (
        MODEL_DIR
        / f"t0_all_layer_mlp_seed{args.seed}.pt"
    )

    best_val_auroc = -1.0
    no_improvement = 0

    history = []

    # =====================================================
    # TRAIN
    # =====================================================

    for epoch in range(
        1,
        args.epochs + 1
    ):
        model.train()

        running_loss = 0.0
        seen = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            n = batch_x.shape[0]

            running_loss += (
                loss.item() * n
            )

            seen += n

        train_loss = (
            running_loss
            / max(seen, 1)
        )

        val_probs = predict(
            model,
            X_val,
            device,
            args.batch_size
        )

        val_auroc = roc_auc_score(
            y_val,
            val_probs
        )

        history.append(
            {
                "epoch":
                    int(epoch),

                "train_loss":
                    float(train_loss),

                "val_auroc":
                    float(val_auroc),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_AUROC={val_auroc:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = float(
                val_auroc
            )

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_dim":
                        int(X_train.shape[1]),

                    "best_val_auroc":
                        best_val_auroc,

                    "seed":
                        int(args.seed),

                    "trainable_parameters":
                        int(total_params),
                },
                checkpoint_path
            )

        else:
            no_improvement += 1

        if no_improvement >= args.patience:
            print(
                f"\nEarly stopping after "
                f"{epoch} epochs."
            )
            break

    # =====================================================
    # LOAD BEST
    # =====================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"\nLoaded best checkpoint "
        f"(val AUROC="
        f"{checkpoint['best_val_auroc']:.4f})"
    )

    # =====================================================
    # VALIDATION THRESHOLD
    # =====================================================

    val_probs = predict(
        model,
        X_val,
        device,
        args.batch_size
    )

    threshold = find_best_threshold(
        y_val,
        val_probs
    )

    val_metrics = compute_metrics(
        y_val,
        val_probs,
        threshold
    )

    # =====================================================
    # TEST
    # =====================================================

    test_probs = predict(
        model,
        X_test,
        device,
        args.batch_size
    )

    test_metrics = compute_metrics(
        y_test,
        test_probs,
        threshold
    )

    print_metrics(
        "Validation",
        val_metrics
    )

    print_metrics(
        "Test",
        test_metrics
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    result = {
        "experiment":
            "t0_all_layer_mlp",

        "description":
            (
                "Fair graph-free baseline using t=0 only, "
                "all 28 transformer layers, PCA-16 per layer."
            ),

        "seed":
            int(args.seed),

        "input_dim":
            int(X_train.shape[1]),

        "trainable_parameters":
            int(total_params),

        "best_val_auroc":
            float(
                checkpoint[
                    "best_val_auroc"
                ]
            ),

        "validation":
            val_metrics,

        "test":
            test_metrics,

        "history":
            history,

        "checkpoint":
            str(checkpoint_path),

        "scaler":
            str(scaler_path),
    }

    result_path = (
        RESULT_DIR
        / f"step11_t0_all_layer_mlp_seed{args.seed}.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("t=0 ALL-LAYER MLP COMPLETE")
    print("=" * 70)

    print(
        f"\nResults:\n{result_path}"
    )

    print(
        "\nCompare against:"
    )

    print(
        "t=0 EpiGraph GraphSAGE: 0.8039"
    )

    print(
        "final-layer t=0 LogReg: 0.7890"
    )


if __name__ == "__main__":
    main()
