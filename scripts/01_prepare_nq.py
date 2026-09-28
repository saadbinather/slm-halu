import json
import random
from pathlib import Path

import pandas as pd


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = Path("nq_data")
OUTPUT_DIR = DATA_DIR / "splits"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =========================================================
# FIND SOURCE DATA FILE
# =========================================================

def find_source_files():
    """
    Find JSON / JSONL files inside nq_data while ignoring generated splits.
    """

    files = []

    for path in DATA_DIR.rglob("*"):
        if not path.is_file():
            continue

        if "splits" in path.parts:
            continue

        if path.suffix.lower() in [".json", ".jsonl"]:
            files.append(path)

    if not files:
        raise FileNotFoundError(
            "No .json or .jsonl files found inside nq_data/"
        )

    print("\nFound dataset files:")

    for f in files:
        print(" -", f)

    return files


# =========================================================
# READ JSONL / JSON
# =========================================================

def read_nq_file(path):
    """
    Supports:
    1. JSONL: one JSON object per line
    2. JSON list: [{...}, {...}]
    """

    rows = []

    # First try JSONL
    try:
        with open(path, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)

                    if isinstance(obj, dict):
                        rows.append(obj)

                except json.JSONDecodeError:
                    # Probably not JSONL
                    rows = []
                    break

        if rows:
            print(
                f"Loaded {len(rows):,} rows from {path.name} "
                f"as JSONL"
            )
            return rows

    except UnicodeDecodeError as e:
        raise RuntimeError(
            f"Could not decode {path}: {e}"
        )

    # Fallback: normal JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        print(
            f"Loaded {len(data):,} rows from {path.name} "
            f"as JSON list"
        )
        return data

    raise ValueError(
        f"Unsupported JSON structure in {path}"
    )


# =========================================================
# CLEAN ONE SAMPLE
# =========================================================

def clean_sample(row):
    """
    Expected NQ format:

    {
        "question": "...",
        "answers": ["answer1", "answer2", ...],
        "ctxs": [...],
        "nq_annotated_gold": {...}
    }

    We only keep question + answers.
    """

    question = row.get("question")
    answers = row.get("answers")

    if question is None or answers is None:
        return None

    question = str(question).strip()

    if not question:
        return None

    # Make sure answers are always a list
    if isinstance(answers, str):
        answers = [answers]

    if not isinstance(answers, list):
        return None

    cleaned_answers = []

    for answer in answers:

        if answer is None:
            continue

        answer = str(answer).strip()

        if answer:
            cleaned_answers.append(answer)

    # Remove duplicate aliases while preserving order
    cleaned_answers = list(dict.fromkeys(cleaned_answers))

    if not cleaned_answers:
        return None

    # Main answer = first alias
    gold_answer = cleaned_answers[0]

    return {
        "question": question,
        "gold_answer": gold_answer,

        # JSON string so aliases remain recoverable later
        "answers": json.dumps(
            cleaned_answers,
            ensure_ascii=False
        )
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("EPIGRAPH — STEP 01: PREPARE NQ")
    print("=" * 70)

    random.seed(SEED)

    source_files = find_source_files()

    all_rows = []

    for path in source_files:
        rows = read_nq_file(path)
        all_rows.extend(rows)

    print(f"\nTotal raw examples: {len(all_rows):,}")

    # -----------------------------------------------------
    # CLEAN
    # -----------------------------------------------------

    cleaned = []

    for row in all_rows:

        sample = clean_sample(row)

        if sample is not None:
            cleaned.append(sample)

    df = pd.DataFrame(cleaned)

    print(f"Valid QA examples: {len(df):,}")

    # -----------------------------------------------------
    # REMOVE DUPLICATE QUESTIONS
    # -----------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["question"]
    ).reset_index(drop=True)

    removed = before - len(df)

    print(f"Duplicate questions removed: {removed:,}")

    # -----------------------------------------------------
    # OPTIONAL SHORT-ANSWER FILTER
    # -----------------------------------------------------

    # EpiGraph uses short factual QA.
    # Reject extremely long gold answers.

    df["answer_words"] = (
        df["gold_answer"]
        .str.split()
        .str.len()
    )

    before = len(df)

    df = df[
        df["answer_words"] <= 20
    ].copy()

    df.drop(
        columns=["answer_words"],
        inplace=True
    )

    print(
        "Long-answer examples removed:",
        before - len(df)
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    # -----------------------------------------------------
    # SHUFFLE
    # -----------------------------------------------------

    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # ADD SAMPLE IDS
    # -----------------------------------------------------

    df.insert(
        0,
        "sample_id",
        [
            f"nq_{i:06d}"
            for i in range(len(df))
        ]
    )

    # -----------------------------------------------------
    # SPLIT
    # -----------------------------------------------------

    total = len(df)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    val_df = df.iloc[
        train_end:val_end
    ].copy()

    test_df = df.iloc[
        val_end:
    ].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "val.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Total usable: {len(df):,}")
    print(f"Train:        {len(train_df):,}")
    print(f"Validation:   {len(val_df):,}")
    print(f"Test:         {len(test_df):,}")

    print("\nSaved:")
    print(" ", train_path)
    print(" ", val_path)
    print(" ", test_path)

    print("\nExample:")

    example = train_df.iloc[0]

    print("ID:      ", example["sample_id"])
    print("Question:", example["question"])
    print("Gold:    ", example["gold_answer"])
    print("Aliases: ", example["answers"])


if __name__ == "__main__":
    main()