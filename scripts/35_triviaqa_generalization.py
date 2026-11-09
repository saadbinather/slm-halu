"""Module 35: closed-book TriviaQA t=0 extraction and Path Signature evaluation.

Stages: select, extract, transitions, label, validate, transfer, train, summary.
Only select/validate/summary are lightweight. All other stages run solely when
the user explicitly calls them. All new files live under results/triviaqa/.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "triviaqa"
DATASET_ID = "mandarjoshi/trivia_qa"
CONFIG_NAME = "rc.nocontext"
DISPLAY_NAME = "TriviaQA"
TEST_ORIGIN = "held out from official validation; official test references are <unk>"
LEGACY_RESULT = ROOT / "results/triviaqa"
RESULT = LEGACY_RESULT / "answerable_validation_holdout"
PREPARED = ROOT / "triviaqa_data/splits"
SPLITS = ("train", "val", "test")
COUNTS = {"train": 1858, "val": 398, "test": 399}
SEEDS = (42, 43, 44, 45, 46)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 12
FEATURE_DIM = 258


@lru_cache(maxsize=None)
def previous(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def path_for(folder, split, sample_id, suffix):
    name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + suffix
    if split in ("train", "val") and folder in (
        "generations", "hidden_states", "transitions", "labels/exact", "labels/semantic"
    ):
        legacy = LEGACY_RESULT / folder / split / name
        if legacy.exists():
            return legacy  # read-only reuse of completed, unchanged examples
    return RESULT / folder / split / name


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_joblib(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def question_key(question):
    helper = previous("35_prepare_limited_qa.py", "module35_question_normalizer")
    return helper.question_key(question)


def read_prepared():
    info = read_json(PREPARED / "manifest.json")
    if info.get("dataset") != DATASET_ID or info.get("configuration") != CONFIG_NAME:
        raise ValueError(f"Expected previously prepared official {DISPLAY_NAME} {CONFIG_NAME}")
    source = {}
    for split in ("train", "val"):
        with (PREPARED / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "question", "gold_answer", "answers", "split", "official_split", "question_id"}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError(f"Prepared {split}.csv schema mismatch")
            source[split] = list(reader)
        if len(source[split]) != info["counts"][split] or len(source[split]) < COUNTS[split]:
            raise ValueError(f"Prepared {split}.csv count mismatch")
    return info, source


def usable_reference(row):
    gold = str(row["gold_answer"]).strip().casefold()
    return gold not in ("", "<unk>", "unknown", "none", "null")


def select():
    manifest_path = RESULT / "manifests/sample.json"
    if manifest_path.exists():
        return load_sample()
    info, source = read_prepared()
    rows, seen_ids, seen_questions = {}, set(), set()
    legacy_manifest_path = LEGACY_RESULT / "manifests/sample.json"
    legacy = read_json(legacy_manifest_path) if legacy_manifest_path.exists() else None
    for split_index, split in enumerate(SPLITS):
        source_split = "val" if split == "test" else split
        if legacy is not None and split in ("train", "val"):
            selected = legacy["rows"][split]
            if len(selected) != COUNTS[split]:
                raise ValueError(f"Existing {split} sample count changed")
            source_ids = {r["question_id"] for r in source[source_split]}
            for row in selected:
                if row["question_id"] not in source_ids or not usable_reference(row):
                    raise ValueError(f"Existing {split} sample is not usable prepared data")
                qkey = question_key(row["question"])
                if row["question_id"] in seen_ids or qkey in seen_questions:
                    raise ValueError("Duplicate ID/question in prior train/validation sample")
                seen_ids.add(row["question_id"])
                seen_questions.add(qkey)
            rows[split] = selected
            continue
        rng = np.random.default_rng(42 + split_index)
        selected = []
        for index in rng.permutation(len(source[source_split])):
            row = source[source_split][int(index)]
            if row["split"] != source_split or row["official_split"] != ("validation" if source_split == "val" else source_split):
                raise ValueError("Prepared source split was altered")
            if not usable_reference(row):
                continue
            qkey = question_key(row["question"])
            if row["question_id"] in seen_ids or qkey in seen_questions:
                continue
            aliases = json.loads(row["answers"])
            if not isinstance(aliases, list) or row["gold_answer"] not in aliases:
                raise ValueError("Missing official answer/alias")
            seen_ids.add(row["question_id"])
            seen_questions.add(qkey)
            selected.append({**{key: row[key] for key in ("sample_id", "question", "gold_answer", "answers",
                                                           "split", "official_split", "question_id")},
                             "split": split})
            if len(selected) == COUNTS[split]:
                break
        if len(selected) != COUNTS[split]:
            raise ValueError(f"Insufficient unique prepared {source_split} examples")
        rows[split] = selected
    manifest = {"dataset": DATASET_ID, "configuration": CONFIG_NAME,
                "sample_seed": 42,
                "sampling": ("reuse completed train/val IDs; seed-44 holdout from remaining prepared official validation"
                             if legacy is not None else "seed 42/43/44 disjoint train/validation/held-out-validation sampling"),
                "test_origin": TEST_ORIGIN,
                "source_manifest": str(PREPARED / "manifest.json"), "source_counts": info["counts"],
                "counts": COUNTS, "total": sum(COUNTS.values()), "rows": rows}
    atomic_json(manifest, manifest_path)
    print(f"Selected {manifest['total']} {DISPLAY_NAME} examples: {COUNTS}")
    return load_sample()


def load_sample():
    manifest = read_json(RESULT / "manifests/sample.json")
    if manifest.get("dataset") != DATASET_ID or manifest.get("configuration") != CONFIG_NAME:
        raise ValueError("Sample manifest dataset/configuration mismatch")
    if manifest.get("sample_seed") != 42 or manifest.get("counts") != COUNTS or manifest.get("total") != 2655:
        raise ValueError("Sample manifest does not specify the fixed 2,655-example selection")
    seen_ids, seen_questions = set(), set()
    for split in SPLITS:
        rows = manifest["rows"][split]
        if len(rows) != COUNTS[split]:
            raise ValueError(f"{split}: sample count mismatch")
        for row in rows:
            if row["split"] != split or row["official_split"] != ("validation" if split in ("val", "test") else "train"):
                raise ValueError("Official split contamination")
            if not usable_reference(row):
                raise ValueError("Unknown or missing gold answer in selected sample")
            aliases = json.loads(row["answers"])
            if not aliases or row["gold_answer"] not in aliases:
                raise ValueError("Missing gold answer or aliases")
            qkey = question_key(row["question"])
            if row["question_id"] in seen_ids or qkey in seen_questions:
                raise ValueError("Duplicate ID or normalized question across splits")
            seen_ids.add(row["question_id"])
            seen_questions.add(qkey)
    return manifest


def generation_metadata(record):
    return {key: record[key] for key in ("sample_id", "dataset", "split", "official_split", "question_id",
                                          "question", "gold_answer", "answers", "prompt_text", "prompt_ids",
                                          "generated_ids", "generated_answer", "num_prompt_tokens",
                                          "num_generated_tokens", "model")}


def validate_hidden(record, row):
    hidden = record.get("hidden_states")
    if record.get("sample_id") != row["sample_id"] or record.get("split") != row["split"]:
        raise ValueError("Hidden record/sample mismatch")
    if record.get("question") != row["question"] or record.get("gold_answer") != row["gold_answer"]:
        raise ValueError("Hidden record/manifest QA mismatch")
    if record.get("answers") != json.loads(row["answers"]) or record.get("model") != MODEL_ID:
        raise ValueError("Hidden record aliases/model mismatch")
    if not isinstance(hidden, torch.Tensor) or hidden.shape != (28, 1536) or not torch.isfinite(hidden).all():
        raise ValueError("Invalid t=0 [28,1536] hidden states")
    if record.get("time_positions") != [0] or record.get("num_prompt_tokens") != len(record.get("prompt_ids", [])):
        raise ValueError("Invalid t=0 or prompt-token metadata")
    if record.get("state_semantics") != "first_generate_prefill_last_prompt_token":
        raise ValueError("t=0 state provenance does not certify pre-generation extraction")
    if not 1 <= len(record.get("generated_ids", [])) <= MAX_NEW_TOKENS:
        raise ValueError("Invalid generated token count")
    if record.get("num_generated_tokens") != len(record["generated_ids"]):
        raise ValueError("Generated token count mismatch")


def extract(args):
    sample = load_sample()
    helper = previous("35_generate_t0_qa.py", "module35_existing_t0_inference")
    generator = previous("02_generate_answers.py", "module35_nq_prompt")
    config_path = RESULT / "manifests/extraction_config.json"
    config = {"model": MODEL_ID, "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
              "prompt": "Module 02 build_prompt", "tokenizer": "Qwen chat template; left padding only for batching",
              "t0": "first generate prefill, final prompt-token position", "hidden_layers": 28}
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing extraction used different configuration")
    if not config_path.exists():
        atomic_json(config, config_path)
    failed_path = RESULT / "manifests/extraction_failures.json"
    failures = read_json(failed_path) if failed_path.exists() else {}
    pending = []
    for split in SPLITS:
        for row in sample["rows"][split]:
            hidden_path = path_for("hidden_states", split, row["sample_id"], ".pt")
            if hidden_path.exists():
                record = torch.load(hidden_path, map_location="cpu", weights_only=False)
                validate_hidden(record, row)
                gen_path = path_for("generations", split, row["sample_id"], ".json")
                if not gen_path.exists():
                    atomic_json(generation_metadata(record), gen_path)
                failures.pop(row["sample_id"], None)
            elif not args.retry_failed_only or row["sample_id"] in failures:
                pending.append(row)
    atomic_json(failures, failed_path)
    print(f"Pending extraction: {len(pending)} of 2,655", flush=True)
    if not pending:
        return
    tokenizer, model, device = helper.load_generator(MODEL_ID)
    batch_size = args.batch_size
    index = 0
    with tqdm(total=len(pending), desc=f"{DISPLAY_NAME} generate + t0") as progress:
        while index < len(pending):
            window = pending[index:index + batch_size]
            current = [row for row in window if not path_for("hidden_states", row["split"], row["sample_id"], ".pt").exists()]
            if not current:
                index += len(window)
                progress.update(len(window))
                continue
            try:
                extracted = helper.infer_batch(current, tokenizer, model, device, DATASET_NAME,
                                               MAX_NEW_TOKENS, generator.build_prompt)
                for row, record in zip(current, extracted):
                    # The shared helper verifies adjacent differences, but
                    # Module 35 stores them once in transitions/ rather than
                    # duplicating them inside hidden_states/.
                    for key in ("transition_vectors", "label", "label_name", "label_method", "judge_result"):
                        record.pop(key, None)
                    record["state_semantics"] = "first_generate_prefill_last_prompt_token"
                    validate_hidden(record, row)
                    atomic_torch(record, path_for("hidden_states", row["split"], row["sample_id"], ".pt"))
                    atomic_json(generation_metadata(record), path_for("generations", row["split"], row["sample_id"], ".json"))
                    failures.pop(row["sample_id"], None)
                atomic_json(failures, failed_path)
                index += len(window)
                progress.update(len(window))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                oom = "out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError)
                if oom and device.type == "cuda":
                    torch.cuda.empty_cache()
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"Batch failed; retrying with batch size {batch_size}: {exc}", flush=True)
                    continue
                row = current[0]
                failures[row["sample_id"]] = {"error": str(exc), "split": row["split"]}
                atomic_json(failures, failed_path)
                index += 1
                progress.update(1)
            except Exception as exc:
                if batch_size > 1:
                    batch_size = 1
                    print(f"Batch failed; isolating examples: {exc}", flush=True)
                    continue
                row = current[0]
                failures[row["sample_id"]] = {"error": str(exc), "split": row["split"]}
                atomic_json(failures, failed_path)
                index += 1
                progress.update(1)
    print(f"Extraction pass complete; unresolved failures: {len(failures)}", flush=True)


def transitions():
    sample = load_sample()
    missing = []
    for split in SPLITS:
        for row in tqdm(sample["rows"][split], desc=f"Transitions {split}"):
            source_path = path_for("hidden_states", split, row["sample_id"], ".pt")
            if not source_path.exists():
                missing.append(row["sample_id"])
                continue
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            validate_hidden(source, row)
            expected = source["hidden_states"][1:] - source["hidden_states"][:-1]
            target_path = path_for("transitions", split, row["sample_id"], ".pt")
            if target_path.exists():
                saved = torch.load(target_path, map_location="cpu", weights_only=False)
                if saved["sample_id"] != row["sample_id"] or not torch.equal(saved["transition_vectors"], expected):
                    raise ValueError(f"{target_path}: incompatible saved transitions")
            else:
                atomic_torch({"sample_id": row["sample_id"], "split": split, "time_positions": [0],
                              "num_layer_transitions": 27, "hidden_dim": 1536,
                              "layer_transition_indices": torch.stack((torch.arange(27), torch.arange(1, 28)), 1),
                              "transition_vectors": expected}, target_path)
            generation_path = path_for("generations", split, row["sample_id"], ".json")
            if not generation_path.exists():
                atomic_json(generation_metadata(source), generation_path)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} hidden records; first IDs: {missing[:5]}")
    print("Validated/saved 2,655 t=0 transition tensors")


def exact_decision(answer, aliases, label_module):
    normalized = label_module.normalize_text(answer)
    match = next((alias for alias in aliases if normalized and normalized == label_module.normalize_text(alias)), None)
    return {"label": 0 if match else 1, "label_name": "correct" if match else "incorrect",
            "method": "normalized_exact" if match else "reference_mismatch",
            "matched_alias": match, "judge_output": None, "uncertain": False}


def label(args):
    sample = load_sample()
    label_module = previous("03_label_answers.py", "module35_nq_labeling")
    if args.variant == "semantic":
        # Exact labels must already exist; semantic judging only reviews
        # nonmatches and never re-generates an answer or hidden state.
        for split in SPLITS:
            for row in sample["rows"][split]:
                if not path_for("labels/exact", split, row["sample_id"], ".json").exists():
                    raise FileNotFoundError("Run exact labeling before semantic judging")
    pending = []
    for split in SPLITS:
        for row in sample["rows"][split]:
            gen_path = path_for("generations", split, row["sample_id"], ".json")
            if not gen_path.exists():
                raise FileNotFoundError(f"Missing generation {row['sample_id']}; run extraction/transitions first")
            generation = read_json(gen_path)
            if generation["sample_id"] != row["sample_id"] or generation["question"] != row["question"]:
                raise ValueError("Generation/manifest ID mismatch")
            output_path = path_for(f"labels/{args.variant}", split, row["sample_id"], ".json")
            if output_path.exists():
                old = read_json(output_path)
                if old.get("label") in (0, 1):
                    continue
            if args.variant == "semantic":
                exact = read_json(path_for("labels/exact", split, row["sample_id"], ".json"))
                if exact["label"] == 0:
                    atomic_json({**exact, "variant": "semantic", "method": "normalized_exact"}, output_path)
                    continue
            pending.append((row, generation, output_path))
    print(f"{args.variant}: {len(pending)} pending label decisions", flush=True)
    if args.variant == "exact":
        for row, generation, output_path in tqdm(pending, desc="Exact labels"):
            aliases = json.loads(row["answers"])
            decision = exact_decision(generation["generated_answer"], aliases, label_module)
            atomic_json({"sample_id": row["sample_id"], "split": row["split"], "variant": "exact",
                         **decision}, output_path)
    elif pending:
        judge_id = args.judge_model or JUDGE_ID
        helper = previous("35_generate_t0_qa.py", "module35_judge_loader")
        tokenizer, model, device = helper.load_generator(judge_id)
        failures = {}
        for row, generation, output_path in tqdm(pending, desc="Semantic labels"):
            aliases = json.loads(row["answers"])
            try:
                result = label_module.judge_with_qwen(model, tokenizer, device,
                                                      generation["generated_answer"], aliases)
                label_value = result["label"] if result["label"] in (0, 1) else None
                decision = {"sample_id": row["sample_id"], "split": row["split"],
                            "variant": "semantic", "label": label_value,
                            "label_name": ("correct" if label_value == 0 else "incorrect") if label_value is not None else "uncertain",
                            "method": "semantic_judge", "matched_alias": None,
                            "judge_output": result["judge_output"], "judge_model": judge_id,
                            "uncertain": label_value is None}
            except Exception as exc:
                decision = {"sample_id": row["sample_id"], "split": row["split"],
                            "variant": "semantic", "label": None, "label_name": "uncertain",
                            "method": "semantic_judge", "matched_alias": None,
                            "judge_output": str(exc), "judge_model": judge_id, "uncertain": True}
            atomic_json(decision, output_path)
            if decision["uncertain"]:
                failures[row["sample_id"]] = decision["judge_output"]
        atomic_json(failures, RESULT / "labels/semantic_failures.json")
    print(f"{args.variant}: pass complete; run validate --scope labels --variant {args.variant}")


def validate(args):
    sample = load_sample()
    scope = args.scope
    label_variant = args.variant
    if scope == "sample":
        print(f"Validated permanent sample: {sample['total']} examples, {sample['counts']}; unique IDs/questions and official splits")
        return {split: Counter({"selected": COUNTS[split]}) for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}
    missing = []
    for split in SPLITS:
        for row in tqdm(sample["rows"][split], desc=f"Validate {split}", disable=scope == "sample"):
            if scope == "sample":
                continue
            hpath = path_for("hidden_states", split, row["sample_id"], ".pt")
            gpath = path_for("generations", split, row["sample_id"], ".json")
            tpath = path_for("transitions", split, row["sample_id"], ".pt")
            if any(not path.exists() for path in (hpath, gpath, tpath)):
                missing.append(row["sample_id"])
                continue
            hidden = torch.load(hpath, map_location="cpu", weights_only=False)
            validate_hidden(hidden, row)
            generation = read_json(gpath)
            if generation != generation_metadata(hidden):
                raise ValueError("Generation JSON disagrees with hidden-state record")
            transition = torch.load(tpath, map_location="cpu", weights_only=False)
            expected = hidden["hidden_states"][1:] - hidden["hidden_states"][:-1]
            if transition["sample_id"] != row["sample_id"] or transition["split"] != split:
                raise ValueError("Transition/sample ID or split mismatch")
            if transition["transition_vectors"].shape != (27, 1536) or not torch.isfinite(transition["transition_vectors"]).all():
                raise ValueError("Invalid t=0 transition tensor")
            if not torch.equal(transition["transition_vectors"], expected):
                raise ValueError("Transition differs from exact h[l+1]-h[l]")
            counts[split]["extracted"] += 1
            if scope in ("labels", "all"):
                lpath = path_for(f"labels/{label_variant}", split, row["sample_id"], ".json")
                if not lpath.exists():
                    missing.append(row["sample_id"])
                    continue
                decision = read_json(lpath)
                if decision["sample_id"] != row["sample_id"] or decision["split"] != split or decision["variant"] != label_variant:
                    raise ValueError("Label/sample mismatch")
                if decision["label"] not in (0, 1):
                    missing.append(row["sample_id"])
                    continue
                label_module = previous("03_label_answers.py", "module35_nq_labeling")
                aliases = json.loads(row["answers"])
                exact = exact_decision(generation["generated_answer"], aliases, label_module)
                if label_variant == "exact":
                    if decision["label"] != exact["label"] or decision["method"] != exact["method"]:
                        raise ValueError(f"{row['sample_id']}: exact label disagrees with generation/reference")
                elif exact["label"] == 0:
                    if decision["label"] != 0 or decision["method"] != "normalized_exact":
                        raise ValueError(f"{row['sample_id']}: semantic variant changed an exact match")
                elif decision["method"] != "semantic_judge" or not decision.get("judge_output"):
                    raise ValueError(f"{row['sample_id']}: semantic nonmatch lacks a judge decision")
                counts[split]["correct" if decision["label"] == 0 else "incorrect"] += 1
    if missing:
        raise RuntimeError(f"Incomplete {scope}: {len(missing)} missing/uncertain IDs; first: {missing[:5]}")
    if scope in ("labels", "all") and any(counts[split]["correct"] == 0 or counts[split]["incorrect"] == 0 for split in SPLITS):
        raise ValueError("Both correctness classes are required in every split")
    print(f"Validated scope={scope}, variant={label_variant}, counts={ {k:dict(v) for k,v in counts.items()} }")
    return counts


def labeled_data(variant):
    args = argparse.Namespace(scope="labels", variant=variant)
    validate(args)
    sample = load_sample()
    data = {}
    for split in SPLITS:
        ids = [row["sample_id"] for row in sample["rows"][split]]
        labels = [read_json(path_for(f"labels/{variant}", split, sid, ".json"))["label"] for sid in ids]
        data[split] = {"ids": ids, "labels": np.asarray(labels, dtype=np.int64)}
    return sample, data


def require_nq_assets():
    paths = {"pca16": ROOT / "models/transition_pca/transition_pca_16d.joblib",
             "pca5": ROOT / "models/path_signature/train_projection_16_to_5.joblib",
             "node_scaler": ROOT / "models/path_signature/projected_node_scaler.joblib",
             "signature_scaler": ROOT / "models/path_signature/order3_train_scaler.joblib"}
    paths.update({f"seed{seed}": ROOT / f"models/path_signature/order3_seed{seed}.pt" for seed in SEEDS})
    paths.update({f"seed{seed}_result": ROOT / f"results/path_signature/order3_seed{seed}.json" for seed in SEEDS})
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing NQ transfer artifacts: " + ", ".join(missing) +
                                "; reproduce NQ artifacts with Modules 21, 22, 23, then 30 in that order")
    objects = {key: joblib.load(paths[key]) for key in ("pca16", "pca5", "node_scaler", "signature_scaler")}
    for key, obj in objects.items():
        if getattr(obj, "training_split_", None) != "train":
            raise ValueError(f"{paths[key]}: missing NQ train-only provenance")
    if objects["pca16"].n_features_in_ != 1536 or objects["pca16"].n_components != 16:
        raise ValueError("NQ first-stage PCA is not 1536->16")
    if objects["pca5"].n_features_in_ != 16 or objects["pca5"].n_components_ != 5:
        raise ValueError("NQ path PCA is not 16->5")
    if objects["node_scaler"].n_features_in_ != 5 or objects["signature_scaler"].n_features_in_ != FEATURE_DIM:
        raise ValueError("NQ scaler dimensions disagree with Module 30")
    return objects, paths


def load_within_preprocessing(sample):
    path = RESULT / "models/within_preprocessing.joblib"
    if path.exists():
        objects = joblib.load(path)
        if objects.get("sample_ids") != {split: [r["sample_id"] for r in sample["rows"][split]] for split in SPLITS}:
            raise ValueError(f"{DISPLAY_NAME} preprocessing belongs to another sample")
        if any(getattr(objects[key], "training_split_", None) != "train" for key in ("pca16", "pca5", "node_scaler", "signature_scaler")):
            raise ValueError(f"{DISPLAY_NAME} preprocessing lacks train-only provenance")
        return objects
    # Incremental PCA is fitted only on the 1,858 training examples. It reads
    # one [27,1536] tensor at a time, so full raw transitions never accumulate.
    train_rows = sample["rows"]["train"]
    pca16 = IncrementalPCA(n_components=16)
    buffer, buffered = [], 0
    for row in tqdm(train_rows, desc=f"Fit {DISPLAY_NAME} train PCA16"):
        item = torch.load(path_for("transitions", "train", row["sample_id"], ".pt"),
                          map_location="cpu", weights_only=False)
        values = item["transition_vectors"].numpy()
        if values.shape != (27, 1536) or not np.isfinite(values).all():
            raise ValueError("Invalid training transition vector")
        buffer.append(values)
        buffered += 27
        if buffered >= 8192:
            pca16.partial_fit(np.concatenate(buffer))
            buffer, buffered = [], 0
    if buffer:
        if buffered < 16:
            raise ValueError("Final incremental PCA batch too small")
        pca16.partial_fit(np.concatenate(buffer))
    pca16.training_split_ = "train"
    train16 = []
    for row in tqdm(train_rows, desc=f"Project {DISPLAY_NAME} train nodes"):
        item = torch.load(path_for("transitions", "train", row["sample_id"], ".pt"),
                          map_location="cpu", weights_only=False)
        # Module 22 persisted PCA-16 features as float32 before Module 30
        # fitted its 16->5 projection. Keep the same intermediate precision.
        train16.append(pca16.transform(item["transition_vectors"].numpy()).astype(np.float32))
    train16 = np.concatenate(train16)
    pca5 = PCA(n_components=5, svd_solver="full").fit(train16)
    pca5.training_split_ = "train"
    node_scaler = StandardScaler().fit(pca5.transform(train16))
    node_scaler.training_split_ = "train"
    objects = {"pca16": pca16, "pca5": pca5, "node_scaler": node_scaler,
               "sample_ids": {split: [r["sample_id"] for r in sample["rows"][split]] for split in SPLITS}}
    raw = signature_arrays(sample, objects, apply_signature_scaler=False)
    signature_scaler = StandardScaler().fit(raw["train"])
    signature_scaler.training_split_ = "train"
    objects["signature_scaler"] = signature_scaler
    atomic_joblib(objects, path)
    return objects


def signature_arrays(sample, objects, apply_signature_scaler=True):
    base = previous("30_train_path_signature.py", "module35_signature")
    signatures = {}
    for split in SPLITS:
        paths = []
        for row in tqdm(sample["rows"][split], desc=f"Project {split} paths"):
            item = torch.load(path_for("transitions", split, row["sample_id"], ".pt"),
                              map_location="cpu", weights_only=False)
            if item["sample_id"] != row["sample_id"] or item["split"] != split:
                raise ValueError("Transition/manifest alignment failed")
            raw = item["transition_vectors"].numpy()
            if raw.shape != (27, 1536) or not np.isfinite(raw).all():
                raise ValueError("Invalid transition shape or finite values")
            pca16 = objects["pca16"].transform(raw).astype(np.float32)
            projected = objects["node_scaler"].transform(objects["pca5"].transform(pca16))
            layer_coordinate = torch.linspace(0, 1, 27, dtype=torch.float32).numpy()[:, None]
            path = np.concatenate((projected, layer_coordinate), axis=1)
            paths.append(path)
        paths = np.stack(paths).astype(np.float64)
        blocks = [base.signature(paths[start:start + 128], 3).numpy()
                  for start in range(0, len(paths), 128)]
        value = np.concatenate(blocks)
        if value.shape != (COUNTS[split], FEATURE_DIM) or not np.isfinite(value).all():
            raise ValueError("Degree-3 Path Signature must be [N,258] and finite")
        if apply_signature_scaler:
            value = objects["signature_scaler"].transform(value).astype(np.float32)
        signatures[split] = value
    return signatures


def feature_cache(sample, experiment):
    path = RESULT / f"models/{experiment}_features.npz"
    if experiment == "transfer":
        objects, nq_paths = require_nq_assets()
        artifact_paths = [nq_paths[key] for key in ("pca16", "pca5", "node_scaler", "signature_scaler")]
    else:
        objects = load_within_preprocessing(sample)
        artifact_paths = [RESULT / "models/within_preprocessing.joblib"]
    digest = hashlib.sha256()
    for source_path in [RESULT / "manifests/sample.json", *artifact_paths]:
        with source_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    for split in SPLITS:
        for row in sample["rows"][split]:
            transition_path = path_for("transitions", split, row["sample_id"], ".pt")
            stat = transition_path.stat()
            digest.update(f"{split}/{transition_path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    fingerprint = digest.hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["fingerprint"].item()) != fingerprint:
                raise ValueError(f"{path}: cached features do not match sample/preprocessing")
            ids = {split: saved[f"{split}_ids"].tolist() for split in SPLITS}
            if ids != {split: [row["sample_id"] for row in sample["rows"][split]] for split in SPLITS}:
                raise ValueError(f"{path}: feature cache belongs to another sample")
            arrays = {split: saved[split] for split in SPLITS}
        if any(arrays[split].shape != (COUNTS[split], FEATURE_DIM) or not np.isfinite(arrays[split]).all() for split in SPLITS):
            raise ValueError(f"{path}: invalid feature cache")
        return arrays
    arrays = signature_arrays(sample, objects)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, fingerprint=np.asarray(fingerprint), **arrays,
                            **{f"{split}_ids": np.asarray([r["sample_id"] for r in sample["rows"][split]]) for split in SPLITS})
    os.replace(temporary, path)
    return arrays


def prediction_path(experiment, variant, seed, split="test"):
    return RESULT / "predictions" / f"{experiment}_{variant}_seed{seed}_{split}.csv"


def save_predictions(path, ids, labels, probabilities, threshold):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "label", "prob_incorrect", "prediction", "threshold"))
        for sid, label, probability in zip(ids, labels, probabilities):
            writer.writerow((sid, int(label), float(probability), int(probability >= threshold), threshold))
    os.replace(temporary, path)


def evaluation_path(experiment, variant, seed):
    return RESULT / "evaluations" / f"{experiment}_{variant}_seed{seed}.json"


def transfer(args):
    sample, data = labeled_data(args.variant)
    objects, paths = require_nq_assets()
    features = feature_cache(sample, "transfer")
    base = previous("30_train_path_signature.py", "module35_signature")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (RESULT / "predictions").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        destination = evaluation_path("transfer", args.variant, seed)
        prediction = prediction_path("transfer", args.variant, seed)
        if destination.exists() and prediction.exists():
            print(f"Skipping completed NQ->{DISPLAY_NAME} transfer seed {seed}")
            continue
        checkpoint = torch.load(paths[f"seed{seed}"], map_location="cpu", weights_only=False)
        nq_result = read_json(paths[f"seed{seed}_result"])
        if checkpoint.get("config") != "order3" or checkpoint.get("seed") != seed or checkpoint.get("feature_dim") != FEATURE_DIM:
            raise ValueError("NQ checkpoint is not the selected Module 30 degree-3 model")
        if nq_result.get("config") != "order3" or nq_result.get("seed") != seed:
            raise ValueError("NQ validation result does not match checkpoint")
        threshold = float(nq_result["validation"]["threshold"])
        model = base.SignatureMLP(FEATURE_DIM).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        scores = base.predict(model, features["test"], device)
        save_predictions(prediction, data["test"]["ids"], data["test"]["labels"], scores, threshold)
        atomic_json({"experiment": f"NQ_to_{DISPLAY_NAME}_transfer", "variant": args.variant, "seed": seed,
                     "trained_on": "Natural Questions only", "fine_tuned_on_target": False,
                     "nq_checkpoint": str(paths[f"seed{seed}"]),
                     "threshold_source": str(paths[f"seed{seed}_result"]),
                     "preprocessing": {key: str(paths[key]) for key in ("pca16", "pca5", "node_scaler", "signature_scaler")},
                     "test": base.metrics(data["test"]["labels"], scores, threshold),
                     "test_predictions": str(prediction), "counts": COUNTS}, destination)
        print(f"Transfer {args.variant} seed {seed}: AUROC={roc_auc_score(data['test']['labels'], scores):.4f}")


def train_one(data, features, variant, seed, device):
    base = previous("30_train_path_signature.py", "module35_signature")
    name = f"within_{variant}_seed{seed}"
    result_path = evaluation_path("within", variant, seed)
    checkpoint_path = RESULT / "models" / f"{name}.pt"
    resume_path = RESULT / "models" / f"{name}_train_state.pt"
    val_path = prediction_path("within", variant, seed, "val")
    test_path = prediction_path("within", variant, seed, "test")
    (RESULT / "predictions").mkdir(parents=True, exist_ok=True)
    output_paths = (result_path, checkpoint_path, val_path, test_path)
    if all(path.exists() for path in output_paths):
        saved = read_json(result_path)
        if saved.get("seed") != seed or saved.get("variant") != variant:
            raise ValueError(f"{result_path}: completed run metadata mismatch")
        print(f"Skipping completed {name}")
        return
    if result_path.exists():
        raise RuntimeError(f"{name}: result JSON exists but final artifacts are incomplete")
    if (val_path.exists() or test_path.exists()) and not resume_path.exists():
        raise RuntimeError(f"{name}: partial predictions exist without resumable training state")
    base.seed_all(seed)
    started = time.perf_counter()
    model = base.SignatureMLP(FEATURE_DIM).to(device)
    class_counts = np.bincount(data["train"]["labels"], minlength=2)
    weights = torch.as_tensor(len(data["train"]["labels"]) / (2 * class_counts), device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(features["train"]),
                                      torch.from_numpy(data["train"]["labels"])),
                        batch_size=64, shuffle=True, generator=generator)
    best, best_epoch, stale = -1.0, 0, 0
    history = []
    start_epoch = 1
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("variant") != variant or state.get("seed") != seed:
            raise ValueError(f"{resume_path}: wrong training state")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        # Optimizer tensors need to follow the active device after CPU reload.
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry[key] = value.to(device)
        best, best_epoch, stale = state["best"], state["best_epoch"], state["stale"]
        history = state["history"]
        start_epoch = state["epoch"] + 1
        generator.set_state(state["generator_state"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{checkpoint_path}: best checkpoint missing during resume")
        print(f"Resuming {name} from epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, 101) if stale < 15 else ():
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        val_scores = base.predict(model, features["val"], device)
        auc = float(roc_auc_score(data["val"]["labels"], val_scores))
        history.append({"epoch": epoch, "train_loss": loss_sum / COUNTS["train"], "val_auroc": auc})
        if auc > best:
            best, best_epoch, stale = auc, epoch, 0
            atomic_torch({"model_state_dict": model.state_dict(), "variant": variant,
                          "seed": seed, "feature_dim": FEATURE_DIM, "epoch": epoch,
                          "val_auroc": auc}, checkpoint_path)
        else:
            stale += 1
        atomic_torch({"variant": variant, "seed": seed, "epoch": epoch,
                      "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "best": best, "best_epoch": best_epoch, "stale": stale,
                      "history": history, "generator_state": generator.get_state(),
                      "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, resume_path)
        if stale >= 15:
            break
    if start_epoch > 100 or stale >= 15:
        epoch = history[-1]["epoch"]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state_dict"])
    val_scores = base.predict(model, features["val"], device)
    threshold = base.select_threshold(data["val"]["labels"], val_scores)
    test_scores = base.predict(model, features["test"], device)
    save_predictions(val_path, data["val"]["ids"], data["val"]["labels"], val_scores, threshold)
    save_predictions(test_path, data["test"]["ids"], data["test"]["labels"], test_scores, threshold)
    atomic_json({"experiment": f"{DISPLAY_NAME}_within_dataset", "variant": variant, "seed": seed,
                 "trained_on": f"{DISPLAY_NAME} train", "selection": f"{DISPLAY_NAME} validation AUROC",
                 "threshold_source": f"{DISPLAY_NAME} validation balanced accuracy",
                 "feature_dim": FEATURE_DIM, "parameter_count": sum(p.numel() for p in model.parameters()),
                 "best_epoch": best_epoch, "epochs_trained": epoch, "runtime_seconds": time.perf_counter() - started,
                 "validation": base.metrics(data["val"]["labels"], val_scores, threshold),
                 "test": base.metrics(data["test"]["labels"], test_scores, threshold),
                 "checkpoint": str(checkpoint_path), "test_predictions": str(test_path),
                 "counts": COUNTS, "history": history}, result_path)
    if resume_path.exists():
        resume_path.unlink()
    print(f"Within {variant} seed {seed}: AUROC={roc_auc_score(data['test']['labels'], test_scores):.4f}")


def train(args):
    sample, data = labeled_data(args.variant)
    features = feature_cache(sample, "within")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        train_one(data, features, args.variant, seed, device)


def read_predictions(experiment, variant, seed, expected_ids, expected_labels):
    path = prediction_path(experiment, variant, seed)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [r["sample_id"] for r in rows] != expected_ids or [int(r["label"]) for r in rows] != expected_labels.tolist():
        raise ValueError(f"{path}: predictions/labels do not align with test manifest")
    return np.asarray([float(r["prob_incorrect"]) for r in rows])


def bootstrap_auc(labels, experiments, replicates=2000):
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(3542)
    sampled = np.concatenate((rng.choice(positive, (replicates, len(positive)), replace=True),
                              rng.choice(negative, (replicates, len(negative)), replace=True)), axis=1)
    npos, nneg = len(positive), len(negative)
    output = {}
    for name, score_matrix in experiments.items():
        values = []
        for seed_scores in score_matrix:
            ranked = rankdata(seed_scores[sampled], axis=1)
            auc = (ranked[:, :npos].sum(axis=1) - npos * (npos + 1) / 2) / (npos * nneg)
            values.append(auc)
        output[name] = np.asarray(values).mean(axis=0)
    return output


def label_statistics(sample):
    counts = {}
    audit = []
    for split in SPLITS:
        exact, semantic = Counter(), Counter()
        for row in sample["rows"][split]:
            sid = row["sample_id"]
            e = read_json(path_for("labels/exact", split, sid, ".json"))
            s = read_json(path_for("labels/semantic", split, sid, ".json"))
            gen = read_json(path_for("generations", split, sid, ".json"))
            exact[e["label"]] += 1
            semantic[s["label"]] += 1
            if e["label"] == 1:
                semantic["judge_required"] += 1
                if s["label"] == 0:
                    semantic["judge_disagreed_with_exact"] += 1
            if s.get("uncertain"):
                semantic["uncertain"] += 1
            audit.append({"sample_id": sid, "split": split, "question": row["question"],
                          "gold_answer": row["gold_answer"], "aliases": row["answers"],
                          "generated_answer": gen["generated_answer"],
                          "exact_label": e["label"], "semantic_label": s["label"],
                          "judge_output": s.get("judge_output"), "uncertain": s.get("uncertain")})
        counts[split] = {"exact_correct": exact[0], "exact_incorrect": exact[1],
                         "semantic_correct": semantic[0], "semantic_incorrect": semantic[1],
                         "judge_required": semantic["judge_required"],
                         "judge_disagreed_with_exact": semantic["judge_disagreed_with_exact"],
                         "uncertain": semantic["uncertain"]}
    selected = random.Random(42).sample(audit, min(50, len(audit)))
    audit_path = RESULT / "labels/human_audit_50.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    os.replace(temporary, audit_path)
    return counts, audit_path


def summary():
    sample = load_sample()
    for variant in ("exact", "semantic"):
        validate(argparse.Namespace(scope="all", variant=variant))
    label_stats, audit_path = label_statistics(sample)
    report = {"module": 35, "dataset": f"{DISPLAY_NAME} {CONFIG_NAME}", "sample_seed": 42,
              "test_origin": sample["test_origin"],
              "counts": COUNTS, "total": 2655, "labels": label_stats,
              "exact_match_rate": sum(label_stats[k]["exact_correct"] for k in SPLITS) / 2655,
              "judge_required": sum(label_stats[k]["judge_required"] for k in SPLITS),
              "judge_disagreement": sum(label_stats[k]["judge_disagreed_with_exact"] for k in SPLITS),
              "judge_uncertain": sum(label_stats[k]["uncertain"] for k in SPLITS),
              "extraction_complete": True, "hidden_shape": [28, 1536],
              "transitions_shape": [27, 1536], "signature_dim": FEATURE_DIM,
              "historical_nq_auroc": {"mean": 0.811284988303622, "sd": 0.005122286379312862},
              "audit_file": str(audit_path), "results": {}}
    for variant in ("exact", "semantic"):
        _, data = labeled_data(variant)
        labels = data["test"]["labels"]
        scores = {}
        by_experiment = {}
        for experiment in ("transfer", "within"):
            runs = []
            score_matrix = []
            for seed in SEEDS:
                result = read_json(evaluation_path(experiment, variant, seed))
                if result["seed"] != seed or result["variant"] != variant:
                    raise ValueError("Evaluation/variant mismatch")
                scores_seed = read_predictions(experiment, variant, seed, data["test"]["ids"], labels)
                if abs(roc_auc_score(labels, scores_seed) - result["test"]["auroc"]) > 1e-8:
                    raise ValueError("Saved AUROC disagrees with test predictions")
                runs.append(result)
                score_matrix.append(scores_seed)
            score_matrix = np.asarray(score_matrix)
            scores[experiment] = score_matrix
            by_experiment[experiment] = {"per_seed_files": [str(evaluation_path(experiment, variant, seed)) for seed in SEEDS],
                                          "per_seed_auroc": [run["test"]["auroc"] for run in runs]}
            for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1"):
                values = np.asarray([run["test"][metric] for run in runs])
                by_experiment[experiment][metric] = {"mean": float(values.mean()),
                                                      "sd": float(values.std(ddof=1))}
        draws = bootstrap_auc(labels, scores)
        for experiment in ("transfer", "within"):
            by_experiment[experiment]["auroc_bootstrap_95_ci"] = np.percentile(draws[experiment], [2.5, 97.5]).tolist()
        difference = draws["within"] - draws["transfer"]
        by_experiment["within_minus_transfer"] = {"auroc_mean_difference":
            by_experiment["within"]["auroc"]["mean"] - by_experiment["transfer"]["auroc"]["mean"],
            "paired_bootstrap_95_ci": np.percentile(difference, [2.5, 97.5]).tolist()}
        report["results"][variant] = by_experiment
    atomic_json(report, RESULT / "summary.json")
    lines = [f"# {DISPLAY_NAME} Path Signature generalization", "",
             f"The 2,655 selected {CONFIG_NAME} questions use 1,858 official-training questions, "
             "398 official-validation questions for model selection, and a disjoint 399-question "
             f"holdout from official validation for testing. Test origin: {TEST_ORIGIN}.",
             "All generated answers and t=0 [28,1536] states/[27,1536] transitions are present and finite.",
             "Labels mean reference-correct (0) versus reference-incorrect (1), not verified factual hallucination.", "",
             "## Labels", "",
             "| Split | Exact correct/incorrect | Semantic correct/incorrect | Judge required | Judge disagreement | Uncertain |",
             "|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        item = label_stats[split]
        lines.append(f"| {split} | {item['exact_correct']}/{item['exact_incorrect']} | "
                     f"{item['semantic_correct']}/{item['semantic_incorrect']} | {item['judge_required']} | "
                     f"{item['judge_disagreed_with_exact']} | {item['uncertain']} |")
    lines.extend(["", f"Overall exact-match rate: {report['exact_match_rate']:.4%}. "
                  f"Manual audit: `{audit_path}`.", "", "## Held-out test results (399 questions)", "",
                  "| Label variant | Experiment | AUROC mean ± SD | AUROC 95% CI | AUPRC | Balanced accuracy | Macro-F1 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for variant in ("exact", "semantic"):
        for experiment in ("transfer", "within"):
            item = report["results"][variant][experiment]
            auc = item["auroc"]
            lines.append(f"| {variant} | {experiment} | {auc['mean']:.4f} ± {auc['sd']:.4f} | "
                         f"[{item['auroc_bootstrap_95_ci'][0]:.4f}, {item['auroc_bootstrap_95_ci'][1]:.4f}] | "
                         f"{item['auprc']['mean']:.4f} | {item['balanced_accuracy']['mean']:.4f} | "
                         f"{item['macro_f1']['mean']:.4f} |")
    lines.extend(["", "Historical Natural Questions order-3 Path Signature: **0.8113 ± 0.0051 AUROC**; "
                  f"its labels and test questions differ from {DISPLAY_NAME}.",
                  "Transfer loads NQ models and all NQ-fitted preprocessing unchanged, using each NQ validation threshold. "
                  f"Within-dataset training fits PCA and scalers on {DISPLAY_NAME} train only and selects epochs/thresholds on {DISPLAY_NAME} validation.",
                  "AUROC confidence intervals use 2,000 stratified resamples of the same 399 test questions, "
                  "averaged over the five seeds. Within-minus-transfer paired intervals are in `summary.json`.",
                  "Limitations: this is a held-out official-validation test, not the official test split; "
                  "reference correctness is a proxy; the semantic judge may err; "
                  "the 399-example test set and class balance limit precision; cross-dataset AUROCs are not directly comparable as absolute task difficulty.",
                  "", "Reproducibility: sample seed 42; model seeds 42–46; deterministic greedy Qwen generation; "
                  "Module 30 degree-3 zero-basepoint signature; 16→5 projected PCA path plus normalized layer coordinate; "
                  "validation-AUROC model selection; no test-fitted preprocessing."])
    (RESULT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {RESULT / 'summary.json'} and summary.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "extract", "transitions", "label", "validate", "transfer", "train", "summary"))
    parser.add_argument("--variant", choices=("exact", "semantic"), default="exact",
                        help="Label variant for label/validate/transfer/train")
    parser.add_argument("--scope", choices=("sample", "extraction", "labels", "all"), default="all",
                        help="Validation scope")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size; automatically halves on failure")
    parser.add_argument("--retry-failed-only", action="store_true", help="Extract only IDs recorded in extraction_failures.json")
    parser.add_argument("--judge-model", default=JUDGE_ID, help="Model for semantic-equivalence judging")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.stage == "select":
        select()
    elif args.stage == "extract":
        extract(args)
    elif args.stage == "transitions":
        transitions()
    elif args.stage == "label":
        label(args)
    elif args.stage == "validate":
        validate(args)
    elif args.stage == "transfer":
        transfer(args)
    elif args.stage == "train":
        train(args)
    else:
        summary()


if __name__ == "__main__":
    main()
