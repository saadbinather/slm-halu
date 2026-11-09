"""Optional future TriviaQA adapter for the unchanged NQ generation/label/trace scripts.

This adapter only redirects their input/output directories. Running any stage
performs its original model inference; Module 33 preparation never calls it.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "label", "extract"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT / "triviaqa_data")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    module_name = {"generate": "02_generate_answers.py", "label": "03_label_answers.py",
                   "extract": "04_extract_hidden_states.py"}[args.stage]
    spec = importlib.util.spec_from_file_location("triviaqa_reused_pipeline", ROOT / "scripts" / module_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generations = args.data_root / "generations"
    if args.stage == "generate":
        module.DATA_DIR = args.data_root / "splits"
        module.OUTPUT_DIR = generations
        original_args = [module_name, "--split", args.split, "--limit", str(args.limit)]
    elif args.stage == "label":
        module.GENERATIONS_DIR = generations
        if args.limit:
            name = f"{args.split}_generations_{args.limit}.csv"
        else:
            name = f"{args.split}_generations.csv"
        original_args = [module_name, "--input", name]
    else:
        if args.limit:
            parser.error("extract requires full labeled split; use --limit 0")
        module.GENERATIONS_DIR = generations
        module.TRACE_DIR = args.data_root / "raw_traces"
        original_args = [module_name, "--split", args.split, "--limit", "0"]
    old_argv = sys.argv
    try:
        sys.argv = original_args
        module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
