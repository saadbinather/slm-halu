"""Prepare nested 300->2655 closed-book QA samples without model inference.

The initial call samples only the records needed for a fixed 2,655-example
roster and activates a configurable prefix (default 300). Later calls expand
that same roster without downloading or changing completed sample IDs.
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_TOTAL = 2655
FIELDS = ("sample_id", "question", "gold_answer", "answers", "split", "official_split", "question_id")
DATASETS = {
    "triviaqa": ("33_prepare_triviaqa.py", "mandarjoshi/trivia_qa", "rc.nocontext"),
    "hotpotqa": ("34_prepare_hotpotqa.py", "hotpotqa/hotpot_qa", "distractor"),
}


def import_preparer(dataset):
    filename = DATASETS[dataset][0]
    spec = importlib.util.spec_from_file_location(f"limited_{dataset}", ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.FIELDS) != FIELDS:
        raise ValueError(f"{filename}: unexpected output schema")
    return module


def roots(dataset):
    root = ROOT / f"{dataset}_data" / "limited"
    return root, root / "splits", root / "master_roster.json"


def counts(total):
    if not 1 <= total <= MAX_TOTAL:
        raise ValueError(f"--total must be 1..{MAX_TOTAL}")
    train = int(total * 0.70)
    val = int(total * 0.15)
    return {"train": train, "val": val, "test": total - train - val}


def question_key(question):
    normalized = unicodedata.normalize("NFKD", question).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", normalized).split())


def sample_official(source, split, required, seed, converter, seen_ids, seen_questions):
    rng = np.random.default_rng(seed)
    result = []
    # Deterministic random order. Only selected candidate rows are converted;
    # no contexts or supporting facts are copied into the roster.
    for index in rng.permutation(len(source)):
        row = converter(source[int(index)], split)
        qkey = question_key(row["question"])
        if row["question_id"] in seen_ids or qkey in seen_questions:
            continue
        seen_ids.add(row["question_id"])
        seen_questions.add(qkey)
        result.append(row)
        if len(result) == required:
            return result
    raise ValueError(f"Official {split} split has fewer than {required} unique usable questions")


def build_roster(dataset_name, seed):
    from datasets import load_dataset  # only the initial prepare command downloads
    preparer = import_preparer(dataset_name)
    _, dataset_id, config = DATASETS[dataset_name]
    source = load_dataset(dataset_id, config)
    full = counts(MAX_TOTAL)
    seen_ids, seen_questions = set(), set()
    if dataset_name == "triviaqa":
        needed = ("train", "validation", "test")
        if not all(split in source for split in needed):
            raise ValueError(f"TriviaQA official splits differ: {list(source)}")
        if any(not {"question", "question_id", "answer"} <= set(source[split].column_names) for split in needed):
            raise ValueError("TriviaQA official schema changed")
        roster = {
            "train": sample_official(source["train"], "train", full["train"], seed + 0,
                                     preparer.make_row, seen_ids, seen_questions),
            "val": sample_official(source["validation"], "validation", full["val"], seed + 1,
                                   preparer.make_row, seen_ids, seen_questions),
            "test": sample_official(source["test"], "test", full["test"], seed + 2,
                                    preparer.make_row, seen_ids, seen_questions),
        }
    else:
        if not {"train", "validation"} <= set(source) or "test" in source:
            raise ValueError(f"HotpotQA distractor official splits differ: {list(source)}")
        if any(not {"id", "question", "answer"} <= set(source[split].column_names) for split in ("train", "validation")):
            raise ValueError("HotpotQA official schema changed")
        train_source = sample_official(source["train"], "train", full["train"] + full["test"],
                                       seed, preparer.make_row, seen_ids, seen_questions)
        heldout = [dict(row, split="test") for row in train_source[full["train"]:]]
        roster = {"train": train_source[:full["train"]],
                  "val": sample_official(source["validation"], "validation", full["val"], seed + 1,
                                         preparer.make_row, seen_ids, seen_questions),
                  "test": heldout}
    return {"dataset": dataset_name, "hf_dataset": dataset_id, "configuration": config,
            "seed": seed, "max_total": MAX_TOTAL,
            "test_origin": "official test" if dataset_name == "triviaqa" else "held out from official train",
            "rows": roster}


def atomic_json(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(rows, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def prepare(dataset_name, total, seed):
    active = counts(total)
    root, split_dir, master_path = roots(dataset_name)
    if master_path.exists():
        master = json.loads(master_path.read_text(encoding="utf-8"))
        if master.get("dataset") != dataset_name or master.get("seed") != seed or master.get("max_total") != MAX_TOTAL:
            raise ValueError("Existing roster has different dataset, seed or size")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root}: existing limited artifacts without a master roster")
        master = build_roster(dataset_name, seed)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(master, master_path)
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("seed") != seed or previous.get("dataset") != dataset_name:
            raise ValueError("Existing active sample has a different seed or dataset")
        if total < previous["total"]:
            raise ValueError("Shrinking an active sample is disabled; preserve completed examples")
        for split in ("train", "val", "test"):
            with (split_dir / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
                old = list(csv.DictReader(handle))
            expected = master["rows"][split][:previous["counts"][split]]
            if old != expected:
                raise ValueError(f"Existing {split}.csv does not match master roster")
    for split in ("train", "val", "test"):
        atomic_csv(master["rows"][split][:active[split]], split_dir / f"{split}.csv")
    atomic_json({"dataset": dataset_name, "hf_dataset": master["hf_dataset"],
                 "configuration": master["configuration"], "seed": seed,
                 "total": total, "counts": active, "test_origin": master["test_origin"],
                 "labels": "unassigned until model answers are evaluated",
                 "prompt_policy": "closed-book question only; no evidence"}, manifest_path)
    validate(dataset_name)


def validate(dataset_name):
    _, split_dir, master_path = roots(dataset_name)
    master = json.loads(master_path.read_text(encoding="utf-8"))
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    if master["dataset"] != dataset_name or manifest["dataset"] != dataset_name or manifest["counts"] != counts(manifest["total"]):
        raise ValueError("Roster or manifest mismatch")
    ids, questions = set(), set()
    for split in ("train", "val", "test"):
        with (split_dir / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"{split}: CSV schema mismatch")
            rows = list(reader)
        if rows != master["rows"][split][:manifest["counts"][split]]:
            raise ValueError(f"{split}: active sample is not the master prefix")
        for row in rows:
            if row["split"] != split or not row["question"] or not row["gold_answer"]:
                raise ValueError(f"{split}: invalid QA row")
            if dataset_name == "hotpotqa" and row["official_split"] != ("validation" if split == "val" else "train"):
                raise ValueError(f"{split}: wrong HotpotQA official split origin")
            if dataset_name == "triviaqa" and row["official_split"] != ("validation" if split == "val" else split):
                raise ValueError(f"{split}: wrong TriviaQA official split origin")
            aliases = json.loads(row["answers"])
            if not isinstance(aliases, list) or row["gold_answer"] not in aliases:
                raise ValueError(f"{split}: invalid answer aliases")
            key = question_key(row["question"])
            if row["question_id"] in ids or key in questions:
                raise ValueError("Duplicate ID or question across active splits")
            ids.add(row["question_id"])
            questions.add(key)
    print(f"{dataset_name}: validated {manifest['total']} prepared samples {manifest['counts']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--total", type=int, default=300, help="Active sample size, 1..2655; default pilot is 300")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.total, args.seed)
    else:
        validate(args.dataset)


if __name__ == "__main__":
    main()
