import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# CONFIG
# =========================================================

GENERATIONS_DIR = Path("generations")

# Stronger local judge
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

MAX_JUDGE_TOKENS = 4


# =========================================================
# DEVICE
# =========================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16

    return torch.device("cpu"), torch.float32


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = text.lower()

    # punctuation -> spaces
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # remove articles
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    # collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# LOAD GOLD ALIASES
# =========================================================

def parse_aliases(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):

        aliases = value

    else:

        try:
            aliases = json.loads(value)

        except Exception:
            aliases = [value]

    if not isinstance(
        aliases,
        list
    ):
        aliases = [aliases]

    cleaned = []

    for alias in aliases:

        alias = str(alias).strip()

        if alias:
            cleaned.append(alias)

    return list(
        dict.fromkeys(cleaned)
    )


# =========================================================
# RULE-BASED ABSTENTION
# =========================================================

ABSTENTION_PATTERNS = [

    r"\bi don'?t know\b",
    r"\bi do not know\b",

    r"\bi'?m not sure\b",
    r"\bi am not sure\b",

    r"\bi cannot answer\b",
    r"\bi can'?t answer\b",

    r"\bi cannot determine\b",
    r"\bi can'?t determine\b",

    r"\bnot enough information\b",
    r"\binsufficient information\b",

    r"\bunable to answer\b",
    r"\bunable to determine\b",

    r"\bno idea\b",
]


def is_abstention(answer):

    if answer is None:
        return True

    text = str(answer).strip()

    if not text:
        return True

    for pattern in ABSTENTION_PATTERNS:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            return True

    return False


# =========================================================
# EXACT ALIAS MATCH
# =========================================================

def exact_alias_match(
    generated_answer,
    aliases
):

    generated_norm = normalize_text(
        generated_answer
    )

    if not generated_norm:

        return False, ""

    for alias in aliases:

        alias_norm = normalize_text(
            alias
        )

        if (
            alias_norm
            and
            generated_norm == alias_norm
        ):

            return True, alias

    return False, ""


# =========================================================
# SAFE ALIAS CONTAINMENT
# =========================================================

def safe_alias_containment(
    generated_answer,
    aliases
):

    """
    Example:

    Gold:
        turkey

    Candidate:
        turkey, stuffing, pumpkin pie

    => CORRECT


    But avoid:

    Gold:
        4

    Candidate:
        24

    => NOT automatically correct
    """

    generated_norm = normalize_text(
        generated_answer
    )

    if not generated_norm:

        return False, ""

    generated_tokens = (
        generated_norm.split()
    )

    for alias in aliases:

        alias_norm = normalize_text(
            alias
        )

        if not alias_norm:
            continue

        alias_tokens = (
            alias_norm.split()
        )

        # ---------------------------------------------
        # avoid risky tiny aliases
        # ---------------------------------------------

        if len(alias_tokens) == 1:

            token = alias_tokens[0]

            # numeric answers should be exact/judged
            if token.isdigit():
                continue

            # tiny strings are risky
            if len(token) < 3:
                continue

        # ---------------------------------------------
        # contiguous whole-token match
        # ---------------------------------------------

        n = len(alias_tokens)

        for i in range(
            len(generated_tokens) - n + 1
        ):

            candidate_window = (
                generated_tokens[
                    i:i+n
                ]
            )

            if (
                candidate_window
                == alias_tokens
            ):

                return True, alias

    return False, ""


# =========================================================
# LLM JUDGE PROMPT
# =========================================================

def build_judge_prompt(
    generated_answer,
    aliases
):

    aliases_text = "\n".join(
        f"- {alias}"
        for alias in aliases
    )

    return f"""
You are a strict answer-equivalence evaluator.

You are given:

1. GOLD ANSWERS
2. one CANDIDATE ANSWER

IMPORTANT:

Do NOT answer any underlying question.

Do NOT use outside knowledge.

Do NOT fact-check the gold answers.

Treat the GOLD ANSWERS as authoritative.

Your ONLY task is to decide whether the CANDIDATE ANSWER
is semantically equivalent to at least one GOLD ANSWER.


Return CORRECT only if:

- the candidate clearly refers to the same person/entity/value
- harmless wording differences exist
- spelling/capitalization/punctuation differs
- abbreviations clearly refer to the same entity
- the candidate expresses the same factual value


Return INCORRECT if:

- it gives a different person
- it gives a different place
- it gives a different date
- it gives a different number
- it gives a different entity
- it gives a different factual value
- it is merely related to the gold answer
- it is a plausible alternative but not equivalent
- equivalence is uncertain


Examples:

GOLD:
- In Time

CANDIDATE:
The Time Machine

LABEL:
INCORRECT


GOLD:
- the English

CANDIDATE:
Francis Calvert

LABEL:
INCORRECT


GOLD:
- by week 4 of development

CANDIDATE:
8 weeks after conception

LABEL:
INCORRECT


GOLD:
- epithelial surface

CANDIDATE:
into capillaries

LABEL:
INCORRECT


GOLD:
- Super Bowl LII

CANDIDATE:
2005 Super Bowl XXXIX

LABEL:
INCORRECT


GOLD:
- John F. Kennedy

CANDIDATE:
JFK

LABEL:
CORRECT


GOLD:
- non-ferrous

CANDIDATE:
non ferrous

LABEL:
CORRECT


Now evaluate:

GOLD ANSWERS:
{aliases_text}

CANDIDATE ANSWER:
{generated_answer}

Return exactly ONE word:

CORRECT

or

INCORRECT
""".strip()


# =========================================================
# PARSE JUDGE OUTPUT
# =========================================================

def parse_judge_output(text):

    text = str(
        text
    ).strip().upper()

    words = re.findall(
        r"\b[A-Z]+\b",
        text
    )

    if not words:

        return (
            -2,
            "judge_error"
        )

    verdict = words[0]

    if verdict == "CORRECT":

        return (
            0,
            "correct"
        )

    if verdict == "INCORRECT":

        return (
            1,
            "hallucinated"
        )

    return (
        -2,
        "judge_error"
    )


# =========================================================
# QWEN JUDGE
# =========================================================

def judge_with_qwen(
    model,
    tokenizer,
    device,
    generated_answer,
    aliases
):

    prompt = build_judge_prompt(
        generated_answer,
        aliases
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    input_length = (
        inputs["input_ids"]
        .shape[1]
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,

            max_new_tokens=MAX_JUDGE_TOKENS,

            do_sample=False,

            pad_token_id=(
                tokenizer.pad_token_id
            ),

            eos_token_id=(
                tokenizer.eos_token_id
            ),

            use_cache=True
        )

    new_tokens = output[
        0,
        input_length:
    ]

    raw_output = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    label, label_name = (
        parse_judge_output(
            raw_output
        )
    )

    return {
        "label": label,
        "label_name": label_name,
        "judge_output": raw_output
    }


# =========================================================
# CLASSIFY ONE SAMPLE
# =========================================================

def classify_sample(
    model,
    tokenizer,
    device,
    generated_answer,
    aliases
):

    # =====================================================
    # 1. EXPLICIT ABSTENTION
    # =====================================================

    if is_abstention(
        generated_answer
    ):

        return {
            "label": -1,
            "label_name": "abstain",
            "label_source": "rule_abstain",
            "judge_output": "RULE_ABSTAIN",
            "matched_alias": ""
        }

    # =====================================================
    # 2. EXACT MATCH
    # =====================================================

    matched, alias = (
        exact_alias_match(
            generated_answer,
            aliases
        )
    )

    if matched:

        return {
            "label": 0,
            "label_name": "correct",
            "label_source": "exact_match",
            "judge_output": "EXACT_MATCH",
            "matched_alias": alias
        }

    # =====================================================
    # 3. SAFE ALIAS CONTAINMENT
    # =====================================================

    matched, alias = (
        safe_alias_containment(
            generated_answer,
            aliases
        )
    )

    if matched:

        return {
            "label": 0,
            "label_name": "correct",
            "label_source": "alias_containment",
            "judge_output": "ALIAS_CONTAINMENT",
            "matched_alias": alias
        }

    # =====================================================
    # 4. EVERYTHING ELSE -> QWEN 3B
    # =====================================================

    result = judge_with_qwen(

        model=model,

        tokenizer=tokenizer,

        device=device,

        generated_answer=generated_answer,

        aliases=aliases
    )

    result[
        "label_source"
    ] = "qwen_equivalence_judge"

    result[
        "matched_alias"
    ] = ""

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=str
    )

    args = parser.parse_args()

    input_path = (
        GENERATIONS_DIR
        / args.input
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Missing file: {input_path}"
        )

    print("=" * 70)

    print(
        "EPIGRAPH — STEP 03: FINAL HYBRID LABELING"
    )

    print("=" * 70)

    df = pd.read_csv(
        input_path
    )

    print(
        f"\nLoaded {len(df):,} responses."
    )

    # =====================================================
    # DEVICE
    # =====================================================

    device, dtype = (
        get_device()
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Dtype:  {dtype}"
    )

    # =====================================================
    # LOAD JUDGE MODEL
    # =====================================================

    print(
        f"\nLoading judge: {MODEL_ID}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(

            MODEL_ID,

            dtype=dtype,

            low_cpu_mem_usage=True
        )
        .to(device)
    )

    model.eval()

    print(
        "Judge loaded.\n"
    )

    # =====================================================
    # LABEL
    # =====================================================

    labels = []

    label_names = []

    label_sources = []

    judge_outputs = []

    matched_aliases = []

    for _, row in tqdm(

        df.iterrows(),

        total=len(df),

        desc="Labeling"
    ):

        aliases = parse_aliases(
            row["answers"]
        )

        result = classify_sample(

            model=model,

            tokenizer=tokenizer,

            device=device,

            generated_answer=str(
                row["generated_answer"]
            ),

            aliases=aliases
        )

        labels.append(
            result["label"]
        )

        label_names.append(
            result["label_name"]
        )

        label_sources.append(
            result["label_source"]
        )

        judge_outputs.append(
            result["judge_output"]
        )

        matched_aliases.append(
            result["matched_alias"]
        )

    # =====================================================
    # ADD LABEL COLUMNS
    # =====================================================

    df[
        "label"
    ] = labels

    df[
        "label_name"
    ] = label_names

    df[
        "label_source"
    ] = label_sources

    df[
        "judge_output"
    ] = judge_outputs

    df[
        "matched_alias"
    ] = matched_aliases

    df[
        "judge_model"
    ] = MODEL_ID

    # =====================================================
    # SAVE
    # =====================================================

    output_path = (

        GENERATIONS_DIR
        /
        (
            input_path.stem
            + "_labeled.csv"
        )
    )

    df.to_csv(
        output_path,
        index=False
    )

    # =====================================================
    # COUNTS
    # =====================================================

    correct = int(
        (
            df["label"] == 0
        ).sum()
    )

    hallucinated = int(
        (
            df["label"] == 1
        ).sum()
    )

    abstain = int(
        (
            df["label"] == -1
        ).sum()
    )

    errors = int(
        (
            df["label"] == -2
        ).sum()
    )

    exact = int(
        (
            df["label_source"]
            == "exact_match"
        ).sum()
    )

    containment = int(
        (
            df["label_source"]
            == "alias_containment"
        ).sum()
    )

    judged = int(
        (
            df["label_source"]
            == "qwen_equivalence_judge"
        ).sum()
    )

    usable = (
        correct
        + hallucinated
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LABELING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Correct:          {correct}"
    )

    print(
        f"Hallucinated:     {hallucinated}"
    )

    print(
        f"Abstain:          {abstain}"
    )

    print(
        f"Judge errors:     {errors}"
    )

    print(
        f"Usable C/H:       {usable}"
    )

    print(
        f"\nExact matches:    {exact}"
    )

    print(
        f"Alias containment:{containment}"
    )

    print(
        f"LLM judged:       {judged}"
    )

    if usable:

        print(
            f"\nCorrect rate: "
            f"{correct / usable:.1%}"
        )

        print(
            f"Hallucination rate: "
            f"{hallucinated / usable:.1%}"
        )

    print(
        f"\nSaved:\n{output_path}"
    )

    # =====================================================
    # SAMPLE LLM-JUDGED CASES
    # =====================================================

    print(
        "\nSAMPLE LLM-JUDGED CASES:\n"
    )

    sample = df[
        df["label_source"]
        == "qwen_equivalence_judge"
    ].head(15)

    for _, row in sample.iterrows():

        print(
            "-" * 70
        )

        print(
            "Gold:",
            row["gold_answer"]
        )

        print(
            "Generated:",
            row["generated_answer"]
        )

        print(
            "Judge:",
            row["judge_output"]
        )

        print(
            "Final:",
            row["label_name"]
        )


if __name__ == "__main__":
    main()