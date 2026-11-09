"""Fit train-only PCA-16/PCA-32 models for EpiGraph transition vectors.

The script reads Module 21 outputs one file at a time, fits independent
IncrementalPCA models using training transitions only, then transforms every
split without constructing graphs or training classifiers.
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


INPUT_ROOT = Path("processed_features/transitions")
OUTPUT_ROOTS = {
    16: Path("processed_features/transitions_pca16"),
    32: Path("processed_features/transitions_pca32"),
}
MODEL_DIR = Path("models/transition_pca")
COMPONENTS = (16, 32)
DEFAULT_BATCH_SIZE = 8192
EXPECTED_INPUT_DIM = 1536
EXPECTED_TRANSITIONS = 27
SPLITS = ("train", "val", "test")


def load_file(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def get_files(split):
    folder = INPUT_ROOT / split
    files = sorted(folder.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No Module 21 transition files found in {folder}")
    return files


def validate_source(item, path, expected_split):
    required = {
        "sample_id", "split", "label", "label_name", "time_positions",
        "checkpoint_availability_mask", "checkpoint_token_ids",
        "checkpoint_is_eos", "checkpoint_token_types", "eos_token_id",
        "layer_transition_indices", "layer_index_normalized",
        "time_normalized", "transition_vectors", "transition_magnitude",
        "relative_update_magnitude", "hidden_cosine_similarity",
        "update_alignment", "geometric_feature_names", "geometric_features",
    }
    missing = required.difference(item)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")

    vectors = item["transition_vectors"]
    if vectors.ndim != 3:
        raise ValueError(f"{path}: expected [time, transition, hidden], got {tuple(vectors.shape)}")
    if vectors.shape[0] != len(item["time_positions"]):
        raise ValueError(f"{path}: timestamp/tensor length mismatch")
    if vectors.shape[1:] != (EXPECTED_TRANSITIONS, EXPECTED_INPUT_DIM):
        raise ValueError(
            f"{path}: expected [T,{EXPECTED_TRANSITIONS},{EXPECTED_INPUT_DIM}], "
            f"got {tuple(vectors.shape)}"
        )
    if str(item["split"]) != expected_split:
        raise ValueError(f"{path}: split={item['split']!r}, expected {expected_split!r}")
    if not torch.isfinite(vectors).all():
        raise ValueError(f"{path}: transition vectors contain NaN/Inf")
    if not torch.isfinite(item["geometric_features"]).all():
        raise ValueError(f"{path}: geometric features contain NaN/Inf")


def model_path(components):
    return MODEL_DIR / f"transition_pca_{components}d.joblib"


def atomic_joblib_dump(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_model(pca, components, path):
    if int(pca.n_components) != components:
        raise ValueError(f"{path}: expected {components} components, got {pca.n_components}")
    if int(pca.n_features_in_) != EXPECTED_INPUT_DIM:
        raise ValueError(f"{path}: expected input dimension {EXPECTED_INPUT_DIM}")
    if not np.isfinite(pca.components_).all():
        raise ValueError(f"{path}: PCA components contain NaN/Inf")


def fit_models(components_to_fit, batch_size):
    """Fit independent PCA models in one streaming pass over train only."""
    models = {
        components: IncrementalPCA(n_components=components)
        for components in components_to_fit
    }
    train_files = get_files("train")
    buffer = []
    buffered_rows = 0
    total_rows = 0
    partial_fit_calls = 0

    def fit_buffer():
        nonlocal buffer, buffered_rows, partial_fit_calls
        batch = np.concatenate(buffer, axis=0)
        for pca in models.values():
            pca.partial_fit(batch)
        buffer = []
        buffered_rows = 0
        partial_fit_calls += 1

    print(f"\nFitting {sorted(components_to_fit)} on TRAIN only ({len(train_files):,} files)")
    for path in tqdm(train_files, desc="Fit transition PCA"):
        item = load_file(path)
        validate_source(item, path, "train")
        vectors = item["transition_vectors"].reshape(-1, EXPECTED_INPUT_DIM).numpy()
        buffer.append(vectors)
        buffered_rows += vectors.shape[0]
        total_rows += vectors.shape[0]
        if buffered_rows >= batch_size:
            fit_buffer()

    if buffer:
        # Every final fit batch must contain at least max(n_components) rows.
        if buffered_rows < max(components_to_fit):
            raise RuntimeError(
                f"Final PCA batch has only {buffered_rows} rows; increase or adjust batch size"
            )
        fit_buffer()

    for components, pca in models.items():
        pca.training_split_ = "train"
        pca.training_file_count_ = len(train_files)
        pca.training_row_count_ = total_rows
        pca.fit_batch_size_ = batch_size
        pca.partial_fit_calls_ = partial_fit_calls
        validate_model(pca, components, model_path(components))

    return models, total_rows


def load_or_fit_models(overwrite, batch_size):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    components_to_fit = []
    for components in COMPONENTS:
        path = model_path(components)
        if path.exists() and not overwrite:
            pca = joblib.load(path)
            validate_model(pca, components, path)
            if getattr(pca, "training_split_", None) != "train":
                raise ValueError(f"{path}: missing train-only provenance")
            models[components] = pca
            print(f"Loaded existing PCA-{components}: {path}")
        else:
            components_to_fit.append(components)

    if components_to_fit:
        fitted, total_rows = fit_models(components_to_fit, batch_size)
        for components, pca in fitted.items():
            path = model_path(components)
            atomic_joblib_dump(pca, path)
            reloaded = joblib.load(path)
            validate_model(reloaded, components, path)
            models[components] = reloaded
            variance = float(reloaded.explained_variance_ratio_.sum())
            print(f"Saved PCA-{components}: {path} (variance={variance:.6%})")
        print(f"Training transition rows used: {total_rows:,}")

    return models


def make_processed(source, reduced, components):
    # Preserve all Module 21 metadata and geometry, but omit the original
    # 1536-D vectors to avoid duplicating the 1.66 GiB source dataset.
    processed = {
        key: value for key, value in source.items() if key != "transition_vectors"
    }
    processed.update(
        {
            "source_transition_dim": EXPECTED_INPUT_DIM,
            "pca_dim": components,
            "transition_features": reduced.contiguous(),
            "pca_model": str(model_path(components)),
            "pca_fit_split": "train",
        }
    )
    return processed


def validate_processed(processed, source, components, output_path):
    expected_shape = (
        len(source["time_positions"]), EXPECTED_TRANSITIONS, components
    )
    if tuple(processed["transition_features"].shape) != expected_shape:
        raise ValueError(
            f"{output_path}: expected reduced shape {expected_shape}, "
            f"got {tuple(processed['transition_features'].shape)}"
        )
    if not torch.isfinite(processed["transition_features"]).all():
        raise ValueError(f"{output_path}: reduced features contain NaN/Inf")
    for key in (
        "sample_id", "split", "label", "label_name", "time_positions",
        "checkpoint_token_types", "eos_token_id",
    ):
        if processed[key] != source[key]:
            raise ValueError(f"{output_path}: metadata mismatch for {key}")
    for key in (
        "checkpoint_availability_mask", "checkpoint_token_ids",
        "checkpoint_is_eos", "layer_transition_indices",
        "layer_index_normalized", "time_normalized", "geometric_features",
        "transition_magnitude", "relative_update_magnitude",
        "hidden_cosine_similarity", "update_alignment",
    ):
        if not torch.equal(processed[key], source[key]):
            raise ValueError(f"{output_path}: tensor metadata mismatch for {key}")


def transform_all(models, overwrite, limit):
    summaries = {}
    for split in SPLITS:
        files = get_files(split)
        if limit > 0:
            files = files[:limit]
        split_summary = {
            components: {"new": 0, "existing": 0, "bytes": 0}
            for components in COMPONENTS
        }

        for components in COMPONENTS:
            (OUTPUT_ROOTS[components] / split).mkdir(parents=True, exist_ok=True)

        for path in tqdm(files, desc=f"Transform {split}"):
            source = load_file(path)
            validate_source(source, path, split)
            vectors = source["transition_vectors"]
            flat = vectors.reshape(-1, EXPECTED_INPUT_DIM).numpy()

            for components, pca in models.items():
                output_path = OUTPUT_ROOTS[components] / split / path.name
                if output_path.exists() and not overwrite:
                    existing = load_file(output_path)
                    validate_processed(existing, source, components, output_path)
                    split_summary[components]["existing"] += 1
                    split_summary[components]["bytes"] += output_path.stat().st_size
                    continue

                reduced_array = pca.transform(flat).reshape(
                    vectors.shape[0], EXPECTED_TRANSITIONS, components
                )
                reduced = torch.from_numpy(reduced_array.astype(np.float32, copy=False))
                processed = make_processed(source, reduced, components)
                validate_processed(processed, source, components, output_path)
                atomic_torch_save(processed, output_path)

                reloaded = load_file(output_path)
                validate_processed(reloaded, source, components, output_path)
                split_summary[components]["new"] += 1
                split_summary[components]["bytes"] += output_path.stat().st_size

        summaries[split] = {
            "selected": len(files),
            "components": split_summary,
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="Per-split transform limit; PCA still uses all train data")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace PCA models and outputs")
    args = parser.parse_args()
    if args.batch_size < max(COMPONENTS):
        parser.error(f"--batch-size must be at least {max(COMPONENTS)}")
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    models = load_or_fit_models(args.overwrite, args.batch_size)
    summaries = transform_all(models, args.overwrite, args.limit)

    print("\n" + "=" * 72)
    print("TRANSITION PCA SUMMARY")
    print("=" * 72)
    for components in COMPONENTS:
        pca = models[components]
        variance = float(pca.explained_variance_ratio_.sum())
        rows = int(getattr(pca, "training_row_count_", -1))
        print(f"PCA-{components}: explained_variance={variance:.6%}; train_rows={rows:,}")
    for split, summary in summaries.items():
        details = []
        for components in COMPONENTS:
            values = summary["components"][components]
            mib = values["bytes"] / (1024 ** 2)
            details.append(
                f"PCA-{components} new={values['new']} existing={values['existing']} size={mib:.1f} MiB"
            )
        print(f"{split}: selected={summary['selected']}; " + "; ".join(details))


if __name__ == "__main__":
    main()
