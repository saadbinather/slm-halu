"""Prepare official TriviaQA rc.nocontext for future closed-book EpiGraph runs.

`prepare` uses Hugging Face datasets and may download data when run by the user.
`validate` checks only saved CSVs; it never accesses the network or a model.
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "triviaqa_data/splits"
DATASET_ID = "mandarjoshi/trivia_qa"
CONFIG = "rc.nocontext"
SPLITS = ("train", "validation", "test")
FIELDS = ("sample_id", "question", "gold_answer", "answers", "split", "official_split", "question_id")


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def make_row(record, official_split):
    if not isinstance(record, dict) or not {"question", "question_id", "answer"} <= set(record):
        raise ValueError("TriviaQA schema missing question, question_id, or answer")
    question = clean_text(record["question"])
    question_id = clean_text(record["question_id"])
    answer = record["answer"]
    if not isinstance(answer, dict) or not {"value", "aliases"} <= set(answer):
        raise ValueError("TriviaQA answer must contain value and aliases")
    gold = clean_text(answer["value"])
    aliases = answer["aliases"]
    if not isinstance(aliases, list):
        raise ValueError("TriviaQA aliases must be a list")
    accepted = list(dict.fromkeys([v for v in [gold] + [clean_text(a) for a in aliases] if v]))
    if not question or not question_id or not gold or not accepted:
        raise ValueError("TriviaQA question/ID/answer is empty")
    # NQ-compatible columns: sample_id, question, gold_answer, answers, split.
    # Preserve official split and source question_id. Never store evidence here.
    mapped = "val" if official_split == "validation" else official_split
    return {"sample_id": f"triviaqa_{question_id}", "question": question,
            "gold_answer": gold, "answers": json.dumps(accepted, ensure_ascii=False),
            "split": mapped, "official_split": official_split, "question_id": question_id}


def sample_rows(rows, limit, seed):
    if limit < 0:
        raise ValueError("Sample limit must be nonnegative")
    if limit == 0 or limit >= len(rows):
        return rows
    indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def prepare(output_dir, limits, seed):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the Hugging Face 'datasets' package before prepare") from exc
    if any(output_dir.glob("*.csv")) or (output_dir / "manifest.json").exists():
        raise FileExistsError(f"{output_dir}: existing prepared data; choose a fresh --output-dir")
    dataset = load_dataset(DATASET_ID, CONFIG)
    if not all(split in dataset for split in SPLITS):
        raise ValueError(f"Expected official splits {SPLITS}, found {list(dataset)}")
    required = {"question", "question_id", "answer"}
    for split in SPLITS:
        if not required <= set(dataset[split].column_names):
            raise ValueError(f"{split}: unexpected schema {dataset[split].column_names}")
    rows_by_split, seen_ids, seen_questions, excluded = {}, set(), set(), {}
    for split in SPLITS:
        rows, duplicate_count = [], 0
        for record in dataset[split]:
            row = make_row(record, split)
            qkey = row["question"].casefold()
            if row["question_id"] in seen_ids or qkey in seen_questions:
                duplicate_count += 1
                continue
            seen_ids.add(row["question_id"])
            seen_questions.add(qkey)
            rows.append(row)
        rows_by_split[split] = sample_rows(rows, limits[split], seed)
        excluded[split] = duplicate_count
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for official_split, rows in rows_by_split.items():
        mapped = "val" if official_split == "validation" else official_split
        path = output_dir / f"{mapped}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        counts[mapped] = len(rows)
    manifest = {"dataset": DATASET_ID, "configuration": CONFIG, "seed": seed,
                "official_to_output_split": {"train": "train", "validation": "val", "test": "test"},
                "counts": counts, "cross_split_or_within_split_duplicates_excluded": excluded,
                "sample_limits": limits, "labels": "unassigned; generate and evaluate model answers first",
                "prompt_policy": "closed-book; question only; no retrieved evidence"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate(output_dir)


def validate(output_dir):
    ids, questions, counts = set(), set(), {}
    for official_split in SPLITS:
        mapped = "val" if official_split == "validation" else official_split
        path = output_dir / f"{mapped}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or ()) != set(FIELDS):
                raise ValueError(f"{path}: expected columns {FIELDS}, found {reader.fieldnames}")
            count = 0
            for row in reader:
                if row["split"] != mapped or row["official_split"] != official_split:
                    raise ValueError(f"{path}: split mismatch")
                if not row["sample_id"].startswith("triviaqa_") or not row["question"] or not row["gold_answer"]:
                    raise ValueError(f"{path}: empty required field")
                aliases = json.loads(row["answers"])
                if not isinstance(aliases, list) or row["gold_answer"] not in aliases:
                    raise ValueError(f"{path}: invalid answer aliases")
                qkey = row["question"].casefold()
                if row["question_id"] in ids or qkey in questions:
                    raise ValueError(f"{path}: duplicate question across prepared splits")
                ids.add(row["question_id"])
                questions.add(qkey)
                count += 1
            counts[mapped] = count
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET_ID or manifest.get("configuration") != CONFIG or manifest.get("counts") != counts:
        raise ValueError("Manifest does not match prepared files")
    print(f"Validated TriviaQA {CONFIG} prepared data: {counts}; unique questions and IDs; no labels or evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means full official split")
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_dir, {"train": args.train_limit, "validation": args.val_limit,
                                  "test": args.test_limit}, args.seed)
    else:
        validate(args.output_dir)


if __name__ == "__main__":
    main()
