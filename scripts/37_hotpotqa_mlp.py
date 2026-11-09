"""Module 37: matched Module 26 transition+geometry MLP on saved HotpotQA.

Reuses Module 36's feature and training code with HotpotQA's existing Module 35
sample, labels, t=0 states, transitions, and train-fitted PCA. No generation,
hidden-state extraction, dataset loading, or PCA fitting occurs here.
"""

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mlp = import_script("36_triviaqa_mlp.py", "epigraph36_mlp_for_hotpotqa")
hot = import_script("36_hotpotqa_generalization.py", "epigraph36_hotpotqa")
mlp.base = hot.base
mlp.MODEL_ROOT = ROOT / "models/hotpotqa_mlp"
mlp.RESULT_ROOT = ROOT / "results/hotpotqa_mlp"
mlp.DATASET_DISPLAY = "HotpotQA"
mlp.MODULE_NUMBER = 37


def report():
    mlp.report()
    sample = hot.base.load_sample()
    counts = {}
    for variant in ("exact", "semantic"):
        counts[variant] = {}
        for split in hot.base.SPLITS:
            labels = [hot.base.read_json(hot.base.path_for(
                f"labels/{variant}", split, row["sample_id"], ".json"))["label"]
                      for row in sample["rows"][split]]
            counts[variant][split] = {"correct": labels.count(0), "incorrect": labels.count(1)}
    path = mlp.RESULT_ROOT / "summary.json"
    summary = hot.base.read_json(path)
    summary["label_counts"] = counts
    summary["test_origin"] = sample["test_origin"]
    summary["class_balance_note"] = (
        "The HotpotQA correctness classes are imbalanced. No labels or samples "
        "were altered. Interpret AUROC and thresholds with the test class counts."
    )
    hot.base.atomic_json(summary, path)
    with (mlp.RESULT_ROOT / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Test class balance\n\n")
        for variant in ("exact", "semantic"):
            item = counts[variant]["test"]
            handle.write(f"{variant}: {item['correct']} correct, {item['incorrect']} incorrect. ")
        handle.write("No labels or samples were altered.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "train", "evaluate", "report"))
    parser.add_argument("--variant", choices=("exact", "semantic"), default="exact")
    args = parser.parse_args()
    if args.stage == "validate":
        hot.base.validate(argparse.Namespace(scope="all", variant=args.variant))
        sample, features = mlp.load_scaled()
        print(f"HotpotQA sample: {sample['counts']}; MLP features: "
              f"{ {split: value.shape for split, value in features.items()} }")
    elif args.stage == "train":
        hot.balance(args.variant)
        _, data = hot.base.labeled_data(args.variant)
        _, features = mlp.load_scaled()
        mlp.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        mlp.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        device = mlp.torch.device("cuda" if mlp.torch.cuda.is_available() else "cpu")
        for seed in mlp.SEEDS:
            mlp.run_one(args.variant, seed, features, data, device)
    elif args.stage == "evaluate":
        mlp.evaluate(args.variant)
    else:
        report()


if __name__ == "__main__":
    main()
