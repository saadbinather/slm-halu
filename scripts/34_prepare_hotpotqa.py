"""Prepare official HotpotQA distractor QA pairs without supporting evidence.

The official distractor configuration has train and validation only. This
script preserves both splits (validation -> val for NQ compatibility) and
does not fabricate a test split or hallucination labels. `validate` is offline.
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "hotpotqa_data/splits"
DATASET_ID = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLITS = ("train", "validation")
FIELDS = ("sample_id", "question", "gold_answer", "answers", "split", "official_split", "question_id")


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def make_row(record, official_split):
    if not isinstance(record, dict) or not {"id", "question", "answer"} <= set(record):
        raise ValueError("HotpotQA schema requires id, question, answer")
    question_id, question, answer = (clean(record[key]) for key in ("id", "question", "answer"))
    if not question_id or not question or not answer:
        raise ValueError("HotpotQA ID/question/answer is empty")
    mapped = "val" if official_split == "validation" else official_split
    return {"sample_id": f"hotpotqa_{question_id}", "question": question,
            "gold_answer": answer, "answers": json.dumps([answer], ensure_ascii=False),
            "split": mapped, "official_split": official_split, "question_id": question_id}


def sample_rows(rows, limit, seed):
    if limit < 0:
        raise ValueError("Limits must be nonnegative")
    if limit == 0 or limit >= len(rows):
        return rows
    return [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), limit))]


def prepare(output_dir, limits, seed):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install Hugging Face datasets to prepare HotpotQA") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: existing files; use a fresh --output-dir")
    dataset = load_dataset(DATASET_ID, CONFIG)
    if not all(split in dataset for split in SPLITS):
        raise ValueError(f"Expected official splits {SPLITS}, found {list(dataset)}")
    if "test" in dataset:
        raise ValueError("Unexpected distractor test split; inspect upstream dataset before proceeding")
    for split in SPLITS:
        if not {"id", "question", "answer"} <= set(dataset[split].column_names):
            raise ValueError(f"{split}: unexpected schema {dataset[split].column_names}")
    rows_by_split, seen_ids, seen_questions, excluded = {}, set(), set(), {}
    for split in SPLITS:
        rows, duplicate_count = [], 0
        for record in dataset[split]:
            row = make_row(record, split)
            key = row["question"].casefold()
            if row["question_id"] in seen_ids or key in seen_questions:
                duplicate_count += 1
                continue
            seen_ids.add(row["question_id"])
            seen_questions.add(key)
            rows.append(row)
        rows_by_split[split] = sample_rows(rows, limits[split], seed)
        excluded[split] = duplicate_count
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for official_split, rows in rows_by_split.items():
        mapped = "val" if official_split == "validation" else official_split
        with (output_dir / f"{mapped}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        counts[mapped] = len(rows)
    manifest = {"dataset": DATASET_ID, "configuration": CONFIG, "seed": seed,
                "official_to_output_split": {"train": "train", "validation": "val"},
                "official_test_available": False, "counts": counts,
                "duplicate_ids_or_questions_excluded": excluded, "sample_limits": limits,
                "labels": "unassigned; evaluate generated model answers against references first",
                "prompt_policy": "closed-book question only; no context or supporting facts"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validate(output_dir)


def validate(output_dir):
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET_ID or manifest.get("configuration") != CONFIG or manifest.get("official_test_available") is not False:
        raise ValueError("Unexpected HotpotQA manifest")
    if (output_dir / "test.csv").exists():
        raise ValueError("Official distractor configuration has no test split")
    ids, questions, counts = set(), set(), {}
    for official_split in SPLITS:
        mapped = "val" if official_split == "validation" else official_split
        path = output_dir / f"{mapped}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"{path}: schema mismatch")
            count = 0
            for row in reader:
                if row["split"] != mapped or row["official_split"] != official_split or row["sample_id"] != f"hotpotqa_{row['question_id']}":
                    raise ValueError(f"{path}: split/ID mismatch")
                if not row["question"] or not row["gold_answer"] or json.loads(row["answers"]) != [row["gold_answer"]]:
                    raise ValueError(f"{path}: missing answer or invalid aliases")
                key = row["question"].casefold()
                if row["question_id"] in ids or key in questions:
                    raise ValueError(f"{path}: repeated ID or question across splits")
                ids.add(row["question_id"])
                questions.add(key)
                count += 1
            counts[mapped] = count
    if manifest.get("counts") != counts:
        raise ValueError("Manifest counts disagree with CSV files")
    print(f"Validated HotpotQA distractor: {counts}; no official test, evidence, or hallucination labels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means full official train split")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means full official validation split")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_dir, {"train": args.train_limit, "validation": args.val_limit}, args.seed)
    else:
        validate(args.output_dir)


if __name__ == "__main__":
    main()
