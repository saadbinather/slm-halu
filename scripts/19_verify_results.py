from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


print("\n=== STEP 16: PAIRED STATISTICS ===")

path = RESULTS / "step16_paired_uncertainty.json"

if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2))
else:
    print("Missing:", path)


print("\n=== STEP 14: MULTI-SEED RESULTS ===")

path = RESULTS / "step14_t0_multiseed.csv"

if path.exists():
    df = pd.read_csv(path)
    print(df.to_string(index=False))

    print("\nMeans:")
    print(df[
        [
            "mlp_test_auroc",
            "real_epigraph_test_auroc",
            "shuffled_graphsage_test_auroc",
        ]
    ].mean())

    print("\nSample SD:")
    print(df[
        [
            "mlp_test_auroc",
            "real_epigraph_test_auroc",
            "shuffled_graphsage_test_auroc",
        ]
    ].std(ddof=1))
else:
    print("Missing:", path)


print("\n=== STEP 12: SINGLE-SEED T=0 RESULTS ===")

path = RESULTS / "step12_t0_all_baselines_seed42.csv"

if path.exists():
    print(pd.read_csv(path).to_string(index=False))
else:
    print("Missing:", path)