"""Extract full-dimensional layer-transition features from saved EpiGraph traces.

This module performs feature extraction only. It does not run a language model,
fit PCA, build graphs, or train a classifier.
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


RAW_DIR = Path("raw_traces")
GENERATIONS_DIR = Path("generations")
OUTPUT_ROOT = Path("processed_features/transitions")

EXPECTED_TIMES = [0, 1, 2, 4, 8]
MAX_TIME = 8
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN_DIM = 1536
EPS = 1e-12


def load_trace(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_ids(value):
    if isinstance(value, list):
        return [int(token_id) for token_id in value]
    if pd.isna(value):
        return []
    return [int(token_id) for token_id in json.loads(value)]


def load_generation_metadata(split):
    """Load token IDs needed for EOS annotation, without duplicating them."""
    path = GENERATIONS_DIR / f"{split}_generations_labeled.csv"
    if not path.exists():
        return {}, f"unavailable: missing {path}"

    df = pd.read_csv(path, usecols=["sample_id", "generated_ids", "model"])
    metadata = {}
    for row in df.itertuples(index=False):
        metadata[str(row.sample_id)] = {
            "generated_ids": parse_ids(row.generated_ids),
            "model": str(row.model),
        }
    return metadata, "generation CSV"


def resolve_eos_token_id(metadata, explicit_id):
    if explicit_id is not None:
        return int(explicit_id), "command line"
    if not metadata:
        return None, "unavailable: no generation metadata"

    model_ids = {item["model"] for item in metadata.values()}
    if len(model_ids) != 1:
        return None, "unavailable: generation rows contain multiple models"

    model_id = next(iter(model_ids))
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            return None, "unavailable: local tokenizer has no EOS token ID"
        return int(eos_id), f"local tokenizer for {model_id}"
    except Exception as exc:
        return None, f"unavailable: local tokenizer lookup failed ({exc})"


def validate_trace(trace, path, expected_split):
    required = {
        "sample_id", "split", "label", "label_name", "time_positions",
        "num_layers", "hidden_dim", "hidden_states", "num_generated_tokens",
    }
    missing = required.difference(trace)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")

    hidden = trace["hidden_states"]
    times = [int(t) for t in trace["time_positions"]]
    if hidden.ndim != 3:
        raise ValueError(f"{path}: expected [time, layer, hidden], got {tuple(hidden.shape)}")
    if hidden.shape[0] != len(times):
        raise ValueError(f"{path}: time metadata/tensor length mismatch")
    if list(times) != sorted(times) or any(t not in EXPECTED_TIMES for t in times):
        raise ValueError(f"{path}: unexpected checkpoint sequence {times}")
    if hidden.shape[1] != EXPECTED_LAYERS or int(trace["num_layers"]) != EXPECTED_LAYERS:
        raise ValueError(f"{path}: expected {EXPECTED_LAYERS} block outputs, got {hidden.shape[1]}")
    if hidden.shape[2] != EXPECTED_HIDDEN_DIM or int(trace["hidden_dim"]) != EXPECTED_HIDDEN_DIM:
        raise ValueError(f"{path}: expected hidden dimension {EXPECTED_HIDDEN_DIM}, got {hidden.shape[2]}")
    if str(trace["split"]) != expected_split:
        raise ValueError(f"{path}: split={trace['split']!r}, expected {expected_split!r}")
    if not torch.isfinite(hidden).all():
        raise ValueError(f"{path}: source hidden states contain NaN/Inf")


def annotate_checkpoints(times, sample_metadata, eos_token_id):
    token_ids = []
    is_eos = []
    token_types = []
    generated_ids = None if sample_metadata is None else sample_metadata["generated_ids"]

    for time_pos in times:
        if time_pos == 0:
            token_ids.append(-1)
            is_eos.append(0)
            token_types.append("prompt")
            continue

        if generated_ids is None or time_pos > len(generated_ids):
            token_ids.append(-1)
            is_eos.append(-1)
            token_types.append("unknown")
            continue

        token_id = int(generated_ids[time_pos - 1])
        token_ids.append(token_id)
        if eos_token_id is None:
            is_eos.append(-1)
            token_types.append("unknown_generated")
        elif token_id == eos_token_id:
            is_eos.append(1)
            token_types.append("eos")
        else:
            is_eos.append(0)
            token_types.append("answer_token")

    return token_ids, is_eos, token_types


def extract_transitions(trace, sample_metadata, eos_token_id):
    # Step 04 stored outputs.hidden_states[1:], i.e. consecutive decoder-block
    # outputs. The model's separate final RMSNorm output was not stored, so all
    # 27 adjacent differences compare like-for-like block representations.
    hidden = trace["hidden_states"].float()
    source = hidden[:, :-1, :]
    target = hidden[:, 1:, :]
    transitions = target - source

    transition_magnitude = torch.linalg.vector_norm(transitions, dim=-1)
    source_norm = torch.linalg.vector_norm(source, dim=-1)
    relative_update_magnitude = transition_magnitude / source_norm.clamp_min(EPS)
    hidden_cosine_similarity = F.cosine_similarity(source, target, dim=-1, eps=EPS)
    update_alignment = F.cosine_similarity(transitions, source, dim=-1, eps=EPS)

    num_times, num_transitions, hidden_dim = transitions.shape
    layer_index_normalized = torch.linspace(0.0, 1.0, num_transitions, dtype=torch.float32)
    times = [int(t) for t in trace["time_positions"]]
    time_normalized = torch.tensor(times, dtype=torch.float32) / float(MAX_TIME)

    geometry = torch.stack(
        [
            transition_magnitude,
            relative_update_magnitude,
            hidden_cosine_similarity,
            update_alignment,
            layer_index_normalized.unsqueeze(0).expand(num_times, -1),
            time_normalized.unsqueeze(1).expand(-1, num_transitions),
        ],
        dim=-1,
    )
    geometry = torch.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0)
    transitions = torch.nan_to_num(transitions, nan=0.0, posinf=0.0, neginf=0.0)

    availability_mask = torch.tensor(
        [checkpoint in times for checkpoint in EXPECTED_TIMES], dtype=torch.bool
    )
    token_ids, is_eos, token_types = annotate_checkpoints(
        times, sample_metadata, eos_token_id
    )
    transition_pairs = torch.stack(
        [torch.arange(num_transitions), torch.arange(1, num_transitions + 1)], dim=1
    ).long()

    return {
        "sample_id": str(trace["sample_id"]),
        "split": str(trace["split"]),
        "label": int(trace["label"]),
        "label_name": str(trace["label_name"]),
        "num_generated_tokens": int(trace["num_generated_tokens"]),
        "time_positions": times,
        "expected_time_positions": list(EXPECTED_TIMES),
        "checkpoint_availability_mask": availability_mask,
        "checkpoint_token_ids": torch.tensor(token_ids, dtype=torch.long),
        "checkpoint_is_eos": torch.tensor(is_eos, dtype=torch.int8),
        "checkpoint_token_types": token_types,
        "eos_token_id": eos_token_id,
        "num_source_layers": int(hidden.shape[1]),
        "num_layer_transitions": int(num_transitions),
        "hidden_dim": int(hidden_dim),
        "layer_transition_indices": transition_pairs,
        "layer_index_normalized": layer_index_normalized,
        "time_normalized": time_normalized,
        "transition_vectors": transitions.contiguous(),
        "transition_magnitude": geometry[..., 0].contiguous(),
        "relative_update_magnitude": geometry[..., 1].contiguous(),
        "hidden_cosine_similarity": geometry[..., 2].contiguous(),
        "update_alignment": geometry[..., 3].contiguous(),
        "geometric_feature_names": [
            "transition_magnitude", "relative_update_magnitude",
            "hidden_cosine_similarity", "update_alignment",
            "layer_index_normalized", "time_normalized",
        ],
        "geometric_features": geometry.contiguous(),
        "state_semantics": "consecutive Qwen decoder-block outputs; embeddings and final RMSNorm excluded",
    }


def atomic_torch_save(value, output_path):
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, output_path)


def validate_output(processed, source_trace):
    expected = source_trace["hidden_states"].float()[:, 1:, :] - source_trace["hidden_states"].float()[:, :-1, :]
    if processed["transition_vectors"].shape != expected.shape:
        raise ValueError("transition tensor has an unexpected shape")
    if not torch.equal(processed["transition_vectors"], expected):
        raise ValueError("stored transitions do not exactly equal h[l+1] - h[l]")
    tensors = [
        processed["transition_vectors"], processed["geometric_features"],
        processed["transition_magnitude"], processed["relative_update_magnitude"],
        processed["hidden_cosine_similarity"], processed["update_alignment"],
    ]
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("output contains NaN/Inf")


def process_split(args, split):
    input_dir = RAW_DIR / split
    files = sorted(input_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No raw traces found in {input_dir}")
    if args.limit > 0:
        files = files[: args.limit]

    metadata, metadata_status = load_generation_metadata(split)
    eos_token_id, eos_status = resolve_eos_token_id(metadata, args.eos_token_id)
    output_dir = OUTPUT_ROOT / split
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    extracted = skipped = failed = 0
    bytes_written = 0
    sample_validations = 0

    print(f"\n{split.upper()}: {len(files):,} raw traces")
    print(f"EOS metadata: {eos_status}; token ID={eos_token_id}")
    print(f"Token metadata: {metadata_status}")

    for path in tqdm(files, desc=f"Transitions {split}"):
        output_path = output_dir / path.name
        if output_path.exists() and not args.overwrite and not args.dry_run:
            existing = load_trace(output_path)
            for checkpoint in existing["time_positions"]:
                counts[int(checkpoint)] += 1
            skipped += 1
            bytes_written += output_path.stat().st_size
            continue

        try:
            trace = load_trace(path)
            validate_trace(trace, path, split)
            sample_id = str(trace["sample_id"])
            processed = extract_transitions(trace, metadata.get(sample_id), eos_token_id)
            validate_output(processed, trace)
            sample_validations += 1
            for checkpoint in processed["time_positions"]:
                counts[int(checkpoint)] += 1

            if not args.dry_run:
                atomic_torch_save(processed, output_path)
                reloaded = load_trace(output_path)
                if reloaded["sample_id"] != sample_id:
                    raise ValueError("saved output failed reload validation")
                bytes_written += output_path.stat().st_size
            extracted += 1
        except Exception as exc:
            failed += 1
            print(f"\nFAILED {path.name}: {exc}")

    return {
        "split": split,
        "selected": len(files),
        "extracted": extracted,
        "skipped_existing": skipped,
        "failed": failed,
        "validated": sample_validations,
        "checkpoint_coverage": {str(t): counts[t] for t in EXPECTED_TIMES},
        "bytes": bytes_written,
        "eos_token_id": eos_token_id,
        "eos_status": eos_status,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Per-split limit; 0 processes all traces")
    parser.add_argument("--dry-run", action="store_true", help="Compute and validate without writing files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing transition files atomically")
    parser.add_argument("--eos-token-id", type=int, default=None, help="Explicit EOS ID; otherwise use the cached tokenizer")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    summaries = [process_split(args, split) for split in splits]

    print("\n" + "=" * 70)
    print("TRANSITION EXTRACTION SUMMARY")
    print("=" * 70)
    for summary in summaries:
        mib = summary["bytes"] / (1024 ** 2)
        print(
            f"{summary['split']}: selected={summary['selected']}, "
            f"new={summary['extracted']}, existing={summary['skipped_existing']}, "
            f"failed={summary['failed']}, size={mib:.1f} MiB, "
            f"coverage={summary['checkpoint_coverage']}"
        )
    total = sum(item["selected"] for item in summaries)
    failed = sum(item["failed"] for item in summaries)
    print(f"Total selected: {total:,}; failures: {failed:,}; dry_run={args.dry_run}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
