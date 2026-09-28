import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# CONFIG
# =========================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

GENERATIONS_DIR = Path("generations")
TRACE_DIR = Path("raw_traces")

# generation positions we want
TARGET_TIMES = [0, 1, 2, 4, 8]


# =========================================================
# DEVICE
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
# LOAD IDS FROM CSV
# =========================================================

def parse_ids(value):

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    return json.loads(value)


# =========================================================
# EXTRACT ONE SAMPLE
# =========================================================

def extract_sample(
    model,
    device,
    row
):

    prompt_ids = parse_ids(
        row["prompt_ids"]
    )

    generated_ids = parse_ids(
        row["generated_ids"]
    )

    if len(prompt_ids) == 0:
        raise ValueError(
            "Prompt IDs are empty."
        )

    if len(generated_ids) == 0:
        return None

    # -----------------------------------------------------
    # Reconstruct exact sequence
    # -----------------------------------------------------

    full_ids = (
        prompt_ids
        + generated_ids
    )

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.ones_like(
        input_ids,
        device=device
    )

    prompt_length = len(
        prompt_ids
    )

    # -----------------------------------------------------
    # NORMAL FORWARD PASS
    # -----------------------------------------------------

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,

            output_hidden_states=True,

            use_cache=False,

            return_dict=True
        )

    # -----------------------------------------------------
    # hidden_states structure:
    #
    # hidden_states[0] = embeddings
    # hidden_states[1] = layer 1 output
    # ...
    #
    # We ignore embedding output.
    # -----------------------------------------------------

    layer_hidden_states = (
        outputs.hidden_states[1:]
    )

    num_layers = len(
        layer_hidden_states
    )

    hidden_dim = (
        layer_hidden_states[0]
        .shape[-1]
    )

    # -----------------------------------------------------
    # Which t positions actually exist?
    #
    # t=0 -> last prompt token
    # t=1 -> first generated token
    # t=2 -> second generated token
    # ...
    # -----------------------------------------------------

    available_times = []

    extracted_states = []

    for t in TARGET_TIMES:

        if t == 0:

            token_position = (
                prompt_length - 1
            )

        else:

            # first generated token starts
            # at index prompt_length
            token_position = (
                prompt_length
                + t
                - 1
            )

            # sample may have fewer tokens
            if t > len(
                generated_ids
            ):
                continue

        # ---------------------------------------------
        # collect every layer at this time
        # ---------------------------------------------

        states_at_time = []

        for layer_state in layer_hidden_states:

            # shape:
            # [1, seq_len, hidden_dim]

            vector = (
                layer_state[
                    0,
                    token_position,
                    :
                ]
                .detach()
                .float()
                .cpu()
            )

            states_at_time.append(
                vector
            )

        # shape:
        # [num_layers, hidden_dim]

        states_at_time = torch.stack(
            states_at_time,
            dim=0
        )

        available_times.append(
            t
        )

        extracted_states.append(
            states_at_time
        )

    if not extracted_states:
        return None

    # -----------------------------------------------------
    # Final tensor:
    #
    # [num_available_times,
    #  num_layers,
    #  hidden_dim]
    # -----------------------------------------------------

    hidden_states = torch.stack(
        extracted_states,
        dim=0
    )

    # -----------------------------------------------------
    # Package
    # -----------------------------------------------------

    trace = {

        "sample_id": str(
            row["sample_id"]
        ),

        "split": str(
            row["split"]
        ),

        "label": int(
            row["label"]
        ),

        "label_name": str(
            row["label_name"]
        ),

        "question": str(
            row["question"]
        ),

        "gold_answer": str(
            row["gold_answer"]
        ),

        "generated_answer": str(
            row["generated_answer"]
        ),

        "prompt_length": int(
            prompt_length
        ),

        "num_generated_tokens": int(
            len(generated_ids)
        ),

        "time_positions": (
            available_times
        ),

        "num_layers": int(
            num_layers
        ),

        "hidden_dim": int(
            hidden_dim
        ),

        "hidden_states": (
            hidden_states
        )
    }

    del (
        outputs,
        input_ids,
        attention_mask,
        layer_hidden_states
    )

    return trace


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test"
        ],
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = full split"
    )

    args = parser.parse_args()

    print("=" * 70)

    print(
        "EPIGRAPH — STEP 04: EXTRACT HIDDEN STATES"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------

    input_path = (
        GENERATIONS_DIR
        /
        f"{args.split}_generations_labeled.csv"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Missing labeled file: {input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    # Only C/H examples.
    # Drop abstain and judge errors.
    df = df[
        df["label"].isin(
            [0, 1]
        )
    ].copy()

    if args.limit > 0:

        df = df.head(
            args.limit
        ).copy()

    print(
        f"\nSplit: {args.split}"
    )

    print(
        f"Usable samples: {len(df):,}"
    )

    # -----------------------------------------------------
    # OUTPUT FOLDER
    # -----------------------------------------------------

    output_dir = (
        TRACE_DIR
        /
        args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # DEVICE
    # -----------------------------------------------------

    device, dtype = (
        get_device()
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Dtype:  {dtype}"
    )

    # -----------------------------------------------------
    # LOAD GENERATOR MODEL
    # -----------------------------------------------------

    print(
        f"\nLoading model: {MODEL_ID}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
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
        "Model loaded."
    )

    print(
        f"\nModel layers: "
        f"{model.config.num_hidden_layers}"
    )

    print(
        f"Hidden size: "
        f"{model.config.hidden_size}"
    )

    # -----------------------------------------------------
    # EXTRACTION LOOP
    # -----------------------------------------------------

    extracted = 0
    skipped_existing = 0
    failed = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting"
    ):

        sample_id = str(
            row["sample_id"]
        )

        output_path = (
            output_dir
            /
            f"{sample_id}.pt"
        )

        # Resume support
        if output_path.exists():

            skipped_existing += 1
            continue

        try:

            trace = extract_sample(
                model=model,
                device=device,
                row=row
            )

            if trace is None:

                failed += 1
                continue

            torch.save(
                trace,
                output_path
            )

            extracted += 1

        except Exception as e:

            failed += 1

            print(
                f"\nFAILED {sample_id}: {e}"
            )

        # -------------------------------------------------
        # cleanup
        # -------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        gc.collect()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXTRACTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"New traces:       {extracted}"
    )

    print(
        f"Already existed:  {skipped_existing}"
    )

    print(
        f"Failed/skipped:    {failed}"
    )

    print(
        f"\nSaved in:\n{output_dir}"
    )


if __name__ == "__main__":
    main()