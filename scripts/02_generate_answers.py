import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================================================
# CONFIG
# =========================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_DIR = Path("nq_data/splits")
OUTPUT_DIR = Path("generations")

MAX_NEW_TOKENS = 12


# =========================================================
# DEVICE SETUP
# =========================================================

def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")

        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16

    else:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


# =========================================================
# PROMPT
# =========================================================

def build_prompt(tokenizer, question):
    """
    Fixed prompt for every NQ question.

    Using Qwen chat template keeps formatting consistent with
    instruction tuning.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You answer factual questions concisely. "
                "Give only the answer, without explanation."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Answer in 1 to 5 words."
            )
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt_text


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of examples. Use 0 for the full split."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EPIGRAPH — STEP 02: GENERATE QWEN ANSWERS")
    print("=" * 70)

    # -----------------------------------------------------
    # DEVICE
    # -----------------------------------------------------

    device, dtype = get_device()

    print(f"\nDevice: {device}")
    print(f"Dtype:  {dtype}")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    input_path = DATA_DIR / f"{args.split}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing dataset split: {input_path}"
        )

    df = pd.read_csv(input_path)

    if args.limit > 0:
        df = df.head(args.limit).copy()

    print(f"\nSplit: {args.split}")
    print(f"Examples to process: {len(df):,}")

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.limit > 0:
        output_path = (
            OUTPUT_DIR /
            f"{args.split}_generations_{args.limit}.csv"
        )
    else:
        output_path = (
            OUTPUT_DIR /
            f"{args.split}_generations.csv"
        )

    # -----------------------------------------------------
    # RESUME SUPPORT
    # -----------------------------------------------------

    completed_ids = set()
    existing_rows = []

    if output_path.exists():

        existing_df = pd.read_csv(output_path)

        existing_rows = existing_df.to_dict(
            orient="records"
        )

        completed_ids = set(
            existing_df["sample_id"].astype(str)
        )

        print(
            f"Resuming existing file: "
            f"{len(completed_ids):,} already completed"
        )

    # -----------------------------------------------------
    # LOAD TOKENIZER
    # -----------------------------------------------------

    print(f"\nLoading tokenizer: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print(f"Loading model: {MODEL_ID}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    model = model.to(device)
    model.eval()

    print("\nModel loaded successfully.")

    # -----------------------------------------------------
    # GENERATION LOOP
    # -----------------------------------------------------

    results = existing_rows

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Generating"
    ):

        sample_id = str(row["sample_id"])

        if sample_id in completed_ids:
            continue

        question = str(row["question"])
        gold_answer = str(row["gold_answer"])
        aliases = str(row["answers"])

        # ---------------------------------------------
        # BUILD PROMPT
        # ---------------------------------------------

        prompt_text = build_prompt(
            tokenizer,
            question
        )

        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded[
            "attention_mask"
        ].to(device)

        prompt_length = input_ids.shape[1]

        # ---------------------------------------------
        # GENERATE
        # ---------------------------------------------

        with torch.inference_mode():

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,

                max_new_tokens=MAX_NEW_TOKENS,

                do_sample=False,

                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,

                use_cache=True
            )

        # ---------------------------------------------
        # EXTRACT EXACT GENERATED TOKEN IDS
        # ---------------------------------------------

        generated_ids = generated[
            0,
            prompt_length:
        ].detach().cpu().tolist()

        prompt_ids = input_ids[
            0
        ].detach().cpu().tolist()

        # ---------------------------------------------
        # DECODE ANSWER
        # ---------------------------------------------

        generated_answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        # ---------------------------------------------
        # SAVE ROW
        # ---------------------------------------------

        result = {
            "sample_id": sample_id,

            "question": question,

            "gold_answer": gold_answer,

            "answers": aliases,

            "prompt_text": prompt_text,

            # JSON strings so exact IDs are recoverable
            "prompt_ids": json.dumps(
                prompt_ids
            ),

            "generated_ids": json.dumps(
                generated_ids
            ),

            "generated_answer": generated_answer,

            "num_prompt_tokens": len(
                prompt_ids
            ),

            "num_generated_tokens": len(
                generated_ids
            ),

            "split": args.split,

            "model": MODEL_ID
        }

        results.append(result)

        # ---------------------------------------------
        # SAVE AFTER EVERY SAMPLE
        # ---------------------------------------------
        # Slower than saving once, but we don't lose
        # hours of generation if something crashes.
        # ---------------------------------------------

        pd.DataFrame(
            results
        ).to_csv(
            output_path,
            index=False
        )

        # ---------------------------------------------
        # CLEANUP
        # ---------------------------------------------

        del (
            input_ids,
            attention_mask,
            generated
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        gc.collect()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    final_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved {len(final_df):,} generations to:"
    )

    print(output_path)

    print("\nExamples:\n")

    for _, row in final_df.head(5).iterrows():

        print("-" * 70)

        print(
            "Question:",
            row["question"]
        )

        print(
            "Gold:",
            row["gold_answer"]
        )

        print(
            "Qwen:",
            row["generated_answer"]
        )

        print(
            "Generated tokens:",
            row["num_generated_tokens"]
        )


if __name__ == "__main__":
    main()