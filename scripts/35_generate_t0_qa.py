"""Resumable closed-book Qwen QA and t=0 trace extraction for limited samples.

`infer` does one deterministic generate() call per batch with hidden-state
output. Its first prefill step supplies all 28 t=0 block states; adjacent
differences yield 27 layer transitions. Per-sample .pt files are committed
atomically and skipped on rerun. `judge` optionally revises non-exact answers
using the existing semantic-equivalence prompt, without regenerating answers.
`validate` is offline and checks all expected examples, labels and tensors.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import unicodedata
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-3B-Instruct"
SPLITS = ("train", "val", "test")
CSV_FIELDS = ("sample_id", "dataset", "split", "official_split", "question_id", "question",
              "gold_answer", "answers", "prompt_text", "prompt_ids", "generated_ids",
              "generated_answer", "num_prompt_tokens", "num_generated_tokens",
              "label", "label_name", "label_method", "judge_result", "model")


def import_previous(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locations(dataset):
    root = ROOT / f"{dataset}_data" / "limited"
    return root, root / "splits", root / "records", root / "generations"


def rows_by_split(dataset):
    prep = import_previous("35_prepare_limited_qa.py", "limited_sample_preparer")
    manifest = prep.validate(dataset)
    _, split_dir, _, _ = locations(dataset)
    result = {}
    for split in SPLITS:
        with (split_dir / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            result[split] = list(csv.DictReader(handle))
    return manifest, result


def record_path(record_dir, row):
    name = hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest() + ".pt"
    return record_dir / row["split"] / name


def load_record(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def atomic_torch(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text or "")).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def initial_label(answer, aliases):
    candidate = normalize(answer)
    if not candidate:
        return 1, "reference_mismatch"
    for alias in aliases:
        gold = normalize(alias)
        if gold and candidate == gold:
            return 0, "normalized_exact"
    # This is reference correctness, not a factual hallucination assertion.
    return 1, "reference_mismatch"


def validate_record(record, row, dataset, max_new_tokens=None):
    if record["sample_id"] != row["sample_id"] or record["dataset"] != dataset or record["split"] != row["split"]:
        raise ValueError(f"{row['sample_id']}: record/sample mismatch")
    if record["question"] != row["question"] or record["gold_answer"] != row["gold_answer"]:
        raise ValueError(f"{row['sample_id']}: QA metadata mismatch")
    if record["answers"] != json.loads(row["answers"]) or record["official_split"] != row["official_split"]:
        raise ValueError(f"{row['sample_id']}: alias or official split mismatch")
    if record["label"] not in (0, 1) or record["label_name"] != ("correct" if record["label"] == 0 else "incorrect"):
        raise ValueError(f"{row['sample_id']}: invalid correctness label")
    hidden, transitions = record["hidden_states"], record["transition_vectors"]
    if hidden.shape != (28, 1536) or transitions.shape != (27, 1536):
        raise ValueError(f"{row['sample_id']}: wrong t=0 hidden/transition dimensions")
    if not torch.isfinite(hidden).all() or not torch.isfinite(transitions).all():
        raise ValueError(f"{row['sample_id']}: nonfinite states/transitions")
    if not torch.equal(transitions, hidden[1:] - hidden[:-1]):
        raise ValueError(f"{row['sample_id']}: transition does not equal adjacent difference")
    if record["time_positions"] != [0] or record["num_layers"] != 28 or record["hidden_dim"] != 1536:
        raise ValueError(f"{row['sample_id']}: invalid t=0 state metadata")
    if record["num_prompt_tokens"] != len(record["prompt_ids"]) or record["num_generated_tokens"] != len(record["generated_ids"]):
        raise ValueError(f"{row['sample_id']}: token count mismatch")
    if max_new_tokens is not None and not 1 <= len(record["generated_ids"]) <= max_new_tokens:
        raise ValueError(f"{row['sample_id']}: generated token count outside configured limit")
    if record["model"] != MODEL_ID:
        raise ValueError(f"{row['sample_id']}: unexpected generator model")


def csv_row(record):
    return {"sample_id": record["sample_id"], "dataset": record["dataset"],
            "split": record["split"], "official_split": record["official_split"],
            "question_id": record["question_id"], "question": record["question"],
            "gold_answer": record["gold_answer"],
            "answers": json.dumps(record["answers"], ensure_ascii=False),
            "prompt_text": record["prompt_text"], "prompt_ids": json.dumps(record["prompt_ids"]),
            "generated_ids": json.dumps(record["generated_ids"]),
            "generated_answer": record["generated_answer"],
            "num_prompt_tokens": record["num_prompt_tokens"],
            "num_generated_tokens": record["num_generated_tokens"],
            "label": record["label"], "label_name": record["label_name"],
            "label_method": record["label_method"], "judge_result": record.get("judge_result", ""),
            "model": record["model"]}


def write_csv(dataset, rows, record_dir, generation_dir, require_complete):
    generation_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        saved = []
        for row in rows[split]:
            path = record_path(record_dir, row)
            if not path.exists():
                if require_complete:
                    raise FileNotFoundError(f"Missing record: {path}")
                continue
            record = load_record(path)
            validate_record(record, row, dataset)
            saved.append(csv_row(record))
        path = generation_dir / f"{split}_generations_labeled.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(saved)
        os.replace(temporary, path)


def device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_generator(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device, dtype = device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = model.to(device).eval()
    return tokenizer, model, device


def config_for(root, max_new_tokens):
    path = root / "inference_config.json"
    proposed = {"model": MODEL_ID, "max_new_tokens": max_new_tokens, "do_sample": False,
                "prompt": "Module 02 Qwen chat template, question only; answer in 1 to 5 words",
                "t0": "first generate prefill hidden states at last prompt token",
                "labels": "reference correctness: 0 correct, 1 incorrect; no factual hallucination claim"}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != proposed:
            raise ValueError(f"{path}: inference settings changed across runs")
    else:
        atomic_json(proposed, path)
    return proposed


def infer_batch(batch_rows, tokenizer, model, device, dataset, max_new_tokens, build_prompt):
    prompts = [build_prompt(tokenizer, row["question"]) for row in batch_rows]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    padded_prompt_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                use_cache=True, return_dict_in_generate=True,
                                output_hidden_states=True)
    if not output.hidden_states or len(output.hidden_states[0]) != 29:
        raise RuntimeError("Generation did not return embedding + 28 t=0 layer states")
    first_step = output.hidden_states[0]
    if any(state.ndim != 3 or state.shape[0] != len(batch_rows) or state.shape[-1] != 1536 for state in first_step):
        raise RuntimeError("Unexpected t=0 generation hidden-state shape")
    result = []
    for index, row in enumerate(batch_rows):
        prompt_ids = inputs["input_ids"][index][inputs["attention_mask"][index].bool()].cpu().tolist()
        generated_ids = output.sequences[index, padded_prompt_width:].cpu().tolist()
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id) + 1]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        hidden = torch.stack([state[index, -1, :].detach().float().cpu() for state in first_step[1:]])
        if hidden.shape != (28, 1536):
            raise RuntimeError("Expected 28 hidden layers of width 1536")
        aliases = json.loads(row["answers"])
        label, method = initial_label(answer, aliases)
        record = {"sample_id": row["sample_id"], "dataset": dataset,
                  "split": row["split"], "official_split": row["official_split"],
                  "question_id": row["question_id"], "question": row["question"],
                  "gold_answer": row["gold_answer"], "answers": aliases,
                  "prompt_text": prompts[index], "prompt_ids": prompt_ids,
                  "generated_ids": generated_ids, "generated_answer": answer,
                  "num_prompt_tokens": len(prompt_ids), "num_generated_tokens": len(generated_ids),
                  "model": MODEL_ID, "label": label,
                  "label_name": "correct" if label == 0 else "incorrect",
                  "label_method": method, "judge_result": "", "time_positions": [0],
                  "num_layers": 28, "hidden_dim": 1536,
                  "hidden_states": hidden, "transition_vectors": hidden[1:] - hidden[:-1]}
        validate_record(record, row, dataset, max_new_tokens)
        result.append(record)
    return result


def infer(dataset, batch_size, max_new_tokens):
    manifest, rows = rows_by_split(dataset)
    root, _, record_dir, generation_dir = locations(dataset)
    config_for(root, max_new_tokens)
    pending = []
    for split in SPLITS:
        for row in rows[split]:
            path = record_path(record_dir, row)
            if path.exists():
                validate_record(load_record(path), row, dataset, max_new_tokens)
            else:
                pending.append(row)
    print(f"{dataset}: {len(pending)} pending of {manifest['total']} active samples", flush=True)
    if pending:
        module02 = import_previous("02_generate_answers.py", "limited_original_generation")
        tokenizer, model, device = load_generator(MODEL_ID)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            records = infer_batch(batch, tokenizer, model, device, dataset, max_new_tokens, module02.build_prompt)
            for row, record in zip(batch, records):
                atomic_torch(record, record_path(record_dir, row))
            print(f"{dataset}: saved {min(start + batch_size, len(pending))}/{len(pending)} pending", flush=True)
    write_csv(dataset, rows, record_dir, generation_dir, require_complete=True)
    validate(dataset)


def judge(dataset, batch_size):
    manifest, rows = rows_by_split(dataset)
    _, _, record_dir, generation_dir = locations(dataset)
    candidates = []
    for split in SPLITS:
        for row in rows[split]:
            path = record_path(record_dir, row)
            record = load_record(path)
            validate_record(record, row, dataset)
            if record["label"] == 1 and not record.get("judge_result"):
                candidates.append((row, record))
    print(f"{dataset}: {len(candidates)} reference mismatches pending semantic review", flush=True)
    if candidates:
        module03 = import_previous("03_label_answers.py", "limited_original_labeling")
        tokenizer, model, device = load_generator(JUDGE_ID)
        for start in range(0, len(candidates), batch_size):
            group = candidates[start:start + batch_size]
            prompts = [tokenizer.apply_chat_template(
                           [{"role": "user", "content": module03.build_judge_prompt(
                               record["generated_answer"], record["answers"])}],
                           tokenize=False, add_generation_prompt=True)
                       for _, record in group]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt",
                               add_special_tokens=False).to(device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
            for index, (row, record) in enumerate(group):
                verdict = tokenizer.decode(output[index, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                label, _ = module03.parse_judge_output(verdict)
                if label in (0, 1):
                    record["label"] = label
                    record["label_name"] = "correct" if label == 0 else "incorrect"
                    record["label_method"] = "semantic_equivalence_judge"
                    record["judge_result"] = verdict.strip()
                    record["judge_model"] = JUDGE_ID
                else:
                    record["judge_result"] = "judge_error: " + verdict.strip()
                atomic_torch(record, record_path(record_dir, row))
            print(f"{dataset}: judged {min(start + batch_size, len(candidates))}/{len(candidates)}", flush=True)
    write_csv(dataset, rows, record_dir, generation_dir, require_complete=True)
    validate(dataset)


def validate(dataset):
    manifest, rows = rows_by_split(dataset)
    root, _, record_dir, generation_dir = locations(dataset)
    config_path = root / "inference_config.json"
    if not config_path.exists():
        raise FileNotFoundError("Inference config missing; run infer first")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["model"] != MODEL_ID:
        raise ValueError("Unexpected generator model")
    counts = {}
    label_counts = {0: 0, 1: 0}
    for split in SPLITS:
        expected_paths = {record_path(record_dir, row) for row in rows[split]}
        actual_paths = set((record_dir / split).glob("*.pt"))
        # Expansion can leave completed records in place, but every record
        # must belong to the currently active sample, which only grows.
        if actual_paths != expected_paths:
            raise ValueError(f"{split}: missing or unexpected trace files ({len(actual_paths)} vs {len(expected_paths)})")
        csv_path = generation_dir / f"{split}_generations_labeled.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValueError(f"{csv_path}: output CSV schema mismatch")
            csv_rows = list(reader)
        if len(csv_rows) != len(rows[split]):
            raise ValueError(f"{split}: prediction CSV is incomplete")
        for row, output in zip(rows[split], csv_rows):
            record = load_record(record_path(record_dir, row))
            validate_record(record, row, dataset, config["max_new_tokens"])
            if output["sample_id"] != row["sample_id"] or int(output["label"]) != record["label"] or output["generated_answer"] != record["generated_answer"]:
                raise ValueError(f"{split}: prediction CSV/trace disagreement")
            label_counts[record["label"]] += 1
        counts[split] = len(rows[split])
    if counts != manifest["counts"] or sum(label_counts.values()) != manifest["total"]:
        raise ValueError("Sample or label counts mismatch")
    print(f"{dataset}: complete t=0 traces={manifest['total']} {counts}; labels correct={label_counts[0]}, incorrect={label_counts[1]}; 28 states/27 transitions each")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("infer", "judge", "validate"))
    parser.add_argument("--dataset", required=True, choices=("triviaqa", "hotpotqa"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("Batch size and max new tokens must be positive")
    if args.command == "infer":
        infer(args.dataset, args.batch_size, args.max_new_tokens)
    elif args.command == "judge":
        judge(args.dataset, args.batch_size)
    else:
        validate(args.dataset)


if __name__ == "__main__":
    main()
