import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

RAW_DIR = Path("raw_traces")
OUTPUT_DIR = Path("processed_features")
MODEL_DIR = Path("models")

DEFAULT_COMPONENTS = 16
FIT_BATCH_SIZE = 4096


# =========================================================
# LOAD ONE TRACE
# =========================================================

def load_trace(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


# =========================================================
# ITERATE TRACE FILES
# =========================================================

def get_trace_files(split):

    folder = RAW_DIR / split

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing folder: {folder}"
        )

    files = sorted(
        folder.glob("*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No .pt traces found in {folder}"
        )

    return files


# =========================================================
# FIT PCA ON TRAIN ONLY
# =========================================================

def fit_pca(
    n_components,
    batch_size
):

    train_files = get_trace_files(
        "train"
    )

    print("=" * 70)
    print("FITTING PCA — TRAIN ONLY")
    print("=" * 70)

    print(
        f"\nTraining traces: {len(train_files):,}"
    )

    print(
        f"PCA dimensions: 1536 -> {n_components}"
    )

    pca = IncrementalPCA(
        n_components=n_components
    )

    buffer = []

    buffered_rows = 0

    total_nodes = 0

    first_hidden_dim = None

    for path in tqdm(
        train_files,
        desc="Fitting PCA"
    ):

        trace = load_trace(path)

        hidden = trace[
            "hidden_states"
        ]

        # Expected:
        # [num_times, num_layers, hidden_dim]

        if hidden.ndim != 3:
            raise ValueError(
                f"{path} has unexpected shape "
                f"{hidden.shape}"
            )

        num_times, num_layers, hidden_dim = (
            hidden.shape
        )

        if first_hidden_dim is None:
            first_hidden_dim = hidden_dim

            print(
                f"\nDetected hidden dimension: "
                f"{hidden_dim}"
            )

        elif hidden_dim != first_hidden_dim:
            raise ValueError(
                f"Inconsistent hidden dimension in "
                f"{path}: {hidden_dim}"
            )

        # ---------------------------------------------
        # Flatten time × layer into PCA samples
        #
        # [T, L, 1536]
        #       ↓
        # [T*L, 1536]
        # ---------------------------------------------

        x = (
            hidden
            .reshape(
                -1,
                hidden_dim
            )
            .float()
            .numpy()
        )

        buffer.append(x)

        buffered_rows += x.shape[0]

        total_nodes += x.shape[0]

        # ---------------------------------------------
        # Partial PCA fit
        # ---------------------------------------------

        if buffered_rows >= batch_size:

            batch = np.concatenate(
                buffer,
                axis=0
            )

            pca.partial_fit(
                batch
            )

            buffer = []
            buffered_rows = 0

    # -------------------------------------------------
    # Remaining nodes
    # -------------------------------------------------

    if buffer:

        batch = np.concatenate(
            buffer,
            axis=0
        )

        # IncrementalPCA requires at least
        # n_components samples per partial_fit.
        if batch.shape[0] >= n_components:

            pca.partial_fit(
                batch
            )

        else:

            print(
                "\nWARNING: final PCA batch had "
                f"only {batch.shape[0]} rows and "
                "was skipped."
            )

    print(
        f"\nTotal TRAIN graph nodes used: "
        f"{total_nodes:,}"
    )

    explained = (
        pca.explained_variance_ratio_
    )

    print(
        f"Explained variance retained "
        f"by {n_components} PCs: "
        f"{explained.sum():.2%}"
    )

    print(
        "\nPer-component explained variance:"
    )

    for i, value in enumerate(
        explained,
        start=1
    ):

        print(
            f"PC{i:02d}: {value:.4%}"
        )

    return pca


# =========================================================
# TRANSFORM ONE SPLIT
# =========================================================

def transform_split(
    split,
    pca
):

    files = get_trace_files(
        split
    )

    output_folder = (
        OUTPUT_DIR / split
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    transformed = 0
    skipped = 0

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"TRANSFORMING {split.upper()}"
    )

    print(
        "=" * 70
    )

    for path in tqdm(
        files,
        desc=f"PCA {split}"
    ):

        output_path = (
            output_folder
            / path.name
        )

        # Resume support
        if output_path.exists():

            skipped += 1
            continue

        trace = load_trace(
            path
        )

        hidden = trace[
            "hidden_states"
        ].float()

        num_times, num_layers, hidden_dim = (
            hidden.shape
        )

        # ---------------------------------------------
        # Flatten nodes
        # ---------------------------------------------

        x = (
            hidden
            .reshape(
                -1,
                hidden_dim
            )
            .numpy()
        )

        # ---------------------------------------------
        # Apply TRAIN-fitted PCA
        # ---------------------------------------------

        x_pca = pca.transform(
            x
        )

        # ---------------------------------------------
        # Restore time × layer structure
        #
        # [T*L, 16]
        #      ↓
        # [T, L, 16]
        # ---------------------------------------------

        x_pca = x_pca.reshape(
            num_times,
            num_layers,
            -1
        )

        x_pca = torch.tensor(
            x_pca,
            dtype=torch.float32
        )

        # ---------------------------------------------
        # Preserve metadata
        # ---------------------------------------------

        processed = {

            "sample_id":
                trace["sample_id"],

            "split":
                trace["split"],

            "label":
                trace["label"],

            "label_name":
                trace["label_name"],

            "question":
                trace["question"],

            "gold_answer":
                trace["gold_answer"],

            "generated_answer":
                trace["generated_answer"],

            "prompt_length":
                trace["prompt_length"],

            "num_generated_tokens":
                trace["num_generated_tokens"],

            "time_positions":
                trace["time_positions"],

            "num_layers":
                trace["num_layers"],

            "original_hidden_dim":
                trace["hidden_dim"],

            "pca_dim":
                int(
                    x_pca.shape[-1]
                ),

            # Shape:
            # [num_times, 28, 16]
            "features":
                x_pca
        }

        torch.save(
            processed,
            output_path
        )

        transformed += 1

    print(
        f"\nNew files:       {transformed}"
    )

    print(
        f"Already existed: {skipped}"
    )

    print(
        f"Output: {output_folder}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--components",
        type=int,
        default=DEFAULT_COMPONENTS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=FIT_BATCH_SIZE
    )

    args = parser.parse_args()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # FIT TRAIN PCA
    # =====================================================

    pca = fit_pca(
        n_components=args.components,
        batch_size=args.batch_size
    )

    # =====================================================
    # SAVE PCA
    # =====================================================

    pca_path = (
        MODEL_DIR
        / f"pca_{args.components}d.joblib"
    )

    joblib.dump(
        pca,
        pca_path
    )

    print(
        f"\nSaved PCA model:\n{pca_path}"
    )

    # =====================================================
    # TRANSFORM ALL SPLITS
    # =====================================================

    for split in [
        "train",
        "val",
        "test"
    ]:

        transform_split(
            split=split,
            pca=pca
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PCA COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nPCA was fitted ONLY on TRAIN."
    )

    print(
        "The exact same PCA was applied "
        "to train / val / test."
    )


if __name__ == "__main__":
    main()