"""Module 36 HotpotQA adapter for Module 35's resumable QA/signature pipeline.

Uses prepared distractor CSVs only. Official train supplies train; disjoint
official validation subsets supply validation and a held-out test. Outputs are
isolated under results/hotpotqa/. No dataset download occurs in this script.
"""

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("epigraph_module35_for_hotpotqa", ROOT / "scripts/35_triviaqa_generalization.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.DATASET_NAME = "hotpotqa"
base.DATASET_ID = "hotpotqa/hotpot_qa"
base.CONFIG_NAME = "distractor"
base.DISPLAY_NAME = "HotpotQA"
base.TEST_ORIGIN = "disjoint holdout from official validation; distractor has no official test split"
base.RESULT = ROOT / "results/hotpotqa"
base.LEGACY_RESULT = base.RESULT
base.PREPARED = ROOT / "hotpotqa_data/splits"


def balance(variant):
    sample = base.load_sample()
    counts = {}
    for split in base.SPLITS:
        values = []
        for row in sample["rows"][split]:
            path = base.path_for(f"labels/{variant}", split, row["sample_id"], ".json")
            if not path.exists():
                raise FileNotFoundError(f"{path}: label missing; run label first")
            value = base.read_json(path)["label"]
            if value not in (0, 1):
                raise ValueError(f"{path}: uncertain label; resolve before evaluation")
            values.append(value)
        counts[split] = dict(Counter(values))
        minority = min(counts[split].get(0, 0), counts[split].get(1, 0))
        print(f"{split}: correct={counts[split].get(0, 0)}, incorrect={counts[split].get(1, 0)}, "
              f"minority={minority}/{len(values)} ({minority/len(values):.1%})")
        if minority < 20 or minority / len(values) < 0.10:
            print(f"WARNING: {split} is highly imbalanced; AUROC/threshold estimates may be unstable")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "extract", "transitions", "label", "validate", "balance", "transfer", "train", "summary"))
    parser.add_argument("--variant", choices=("exact", "semantic"), default="exact")
    parser.add_argument("--scope", choices=("sample", "extraction", "labels", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--retry-failed-only", action="store_true")
    parser.add_argument("--judge-model", default=base.JUDGE_ID)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.stage == "select":
        base.select()
    elif args.stage == "extract":
        base.extract(args)
    elif args.stage == "transitions":
        base.transitions()
    elif args.stage == "label":
        base.label(args)
    elif args.stage == "validate":
        base.validate(args)
    elif args.stage == "balance":
        balance(args.variant)
    elif args.stage == "transfer":
        balance(args.variant)
        base.transfer(args)
    elif args.stage == "train":
        balance(args.variant)
        base.train(args)
    else:
        exact_balance = balance("exact")
        semantic_balance = balance("semantic")
        base.summary()
        summary_path = base.RESULT / "summary.json"
        report = base.read_json(summary_path)
        report["label_balance"] = {"exact": exact_balance, "semantic": semantic_balance}
        report["balance_warning"] = "Small correctness classes can make AUROC and thresholds unstable; no relabeling or resampling was applied."
        base.atomic_json(report, summary_path)


if __name__ == "__main__":
    main()
