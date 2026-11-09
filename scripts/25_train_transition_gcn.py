"""Train/evaluate GCNs on the PCA-16 Layer-Time Transition Graphs.

This is the GCN counterpart to Module 24. It uses the identical graph filters,
splits, hyperparameters, validation model selection, threshold selection, seeds,
metrics, and prediction format while keeping outputs under distinct GCN names.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE24_PATH = SCRIPT_DIR / "24_train_transition_graph.py"


def load_module24():
    spec = importlib.util.spec_from_file_location("epigraph_module24", MODULE24_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {MODULE24_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GCNClassifier(nn.Module):
    """Two-layer GCN matching Module 24's GraphSAGE dimensions/head."""

    def __init__(self, base):
        super().__init__()
        self.conv1 = GCNConv(base.INPUT_DIM, base.HIDDEN_DIM)
        self.norm1 = nn.BatchNorm1d(base.HIDDEN_DIM)
        self.conv2 = GCNConv(base.HIDDEN_DIM, base.HIDDEN_DIM)
        self.norm2 = nn.BatchNorm1d(base.HIDDEN_DIM)
        self.dropout = nn.Dropout(base.DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(base.HIDDEN_DIM, base.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(base.DROPOUT),
            nn.Linear(base.HIDDEN_DIM // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        x = self.dropout(torch.relu(self.norm1(self.conv1(x, edge_index))))
        x = self.dropout(torch.relu(self.norm2(self.conv2(x, edge_index))))
        return self.classifier(global_mean_pool(x, batch))


def load_graphsage_comparison(base):
    path = base.RESULT_DIR / "summary.json"
    if not path.exists():
        return {"available": False, "reason": f"missing {path}"}
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    comparisons = {}
    for experiment in base.EXPERIMENTS:
        try:
            comparisons[experiment] = report["results"][experiment]["graphsage"]["auroc"]
        except KeyError:
            comparisons[experiment] = None
    return {"available": True, "source": str(path), "auroc": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("At least one seed is required")

    base = load_module24()
    base.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Module 24's run loop calls this factory. Replacing it locally preserves
    # every other training/evaluation detail while selecting GCNConv.
    def gcn_factory(architecture):
        if architecture != "gcn":
            raise ValueError(f"Module 25 only supports GCN, got {architecture!r}")
        return GCNClassifier(base)

    base.model_for = gcn_factory

    runs = []
    dataset_report = {}
    for experiment, cutoff in base.EXPERIMENTS.items():
        datasets = {}
        coverage = {}
        for split in ("train", "val", "test"):
            datasets[split], coverage[split] = base.load_graphs(
                split, experiment, cutoff
            )
        base.validate_splits(datasets)
        dataset_report[experiment] = {
            "cutoff": cutoff,
            "counts": {split: len(graphs) for split, graphs in datasets.items()},
            "checkpoint_coverage": coverage,
            "no_split_overlap": True,
            "eos_nodes_present": False,
        }
        for seed in args.seeds:
            runs.append(
                base.run_one(
                    datasets, experiment, "gcn", seed, device, args.overwrite
                )
            )

    aggregate = base.aggregate_runs(runs)
    report = {
        "module": 25,
        "algorithm": "GCN",
        "graph_source": str(base.GRAPH_ROOT),
        "seeds": args.seeds,
        "dataset_validation": dataset_report,
        "hyperparameters": {
            "input_dim": base.INPUT_DIM,
            "hidden_dim": base.HIDDEN_DIM,
            "dropout": base.DROPOUT,
            "batch_size": base.BATCH_SIZE,
            "learning_rate": base.LEARNING_RATE,
            "weight_decay": base.WEIGHT_DECAY,
            "max_epochs": base.MAX_EPOCHS,
            "patience": base.PATIENCE,
            "selection": "best validation AUROC; threshold maximizes validation balanced accuracy",
        },
        "baselines": {
            "original_t0_five_seed_auroc": base.ORIGINAL_T0_FIVE_SEED_AUROC,
            "original_t0_cutoff_auroc_separate_experiment": base.ORIGINAL_T0_CUTOFF_AUROC,
            "module24_graphsage": load_graphsage_comparison(base),
        },
        "results": aggregate,
    }
    base.atomic_json_dump(report, base.RESULT_DIR / "gcn_summary.json")
    base.write_summary_csv(aggregate, base.RESULT_DIR / "gcn_summary.csv")

    print("\nGCN AGGREGATED RESULTS")
    for experiment, architectures in aggregate.items():
        values = architectures["gcn"]
        print(
            f"{experiment}: AUROC={values['auroc']['mean']:.4f}+-{values['auroc']['std']:.4f}; "
            f"AUPRC={values['auprc']['mean']:.4f}+-{values['auprc']['std']:.4f}; "
            f"BalAcc={values['balanced_accuracy']['mean']:.4f}+-{values['balanced_accuracy']['std']:.4f}; "
            f"MacroF1={values['macro_f1']['mean']:.4f}+-{values['macro_f1']['std']:.4f}"
        )


if __name__ == "__main__":
    main()
