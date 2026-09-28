from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd


RESULT_DIR = Path("results")

INPUT_FILE = (
    RESULT_DIR
    / "step14_t0_multiseed.csv"
)

N_BOOTSTRAP = 50000
SEED = 123


# =========================================================
# PAIRED BOOTSTRAP OVER SEEDS
# =========================================================

def bootstrap_ci(
    differences,
    n_bootstrap=50000,
    seed=123
):

    differences = np.asarray(
        differences,
        dtype=np.float64
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(differences)

    boot_means = np.empty(
        n_bootstrap,
        dtype=np.float64
    )

    for i in range(
        n_bootstrap
    ):

        sampled = rng.choice(
            differences,
            size=n,
            replace=True
        )

        boot_means[i] = (
            sampled.mean()
        )

    lower = np.percentile(
        boot_means,
        2.5
    )

    upper = np.percentile(
        boot_means,
        97.5
    )

    return (
        float(lower),
        float(upper)
    )


# =========================================================
# EXACT PAIRED SIGN-FLIP TEST
#
# Null hypothesis:
# method labels are exchangeable within each seed.
#
# With 5 seeds there are only 2^5 = 32 possibilities.
# =========================================================

def exact_sign_flip_test(
    differences
):

    differences = np.asarray(
        differences,
        dtype=np.float64
    )

    observed = abs(
        differences.mean()
    )

    n = len(differences)

    null_means = []

    for signs in itertools.product(
        [-1, 1],
        repeat=n
    ):

        signs = np.asarray(
            signs,
            dtype=np.float64
        )

        value = abs(
            (
                differences
                * signs
            ).mean()
        )

        null_means.append(
            value
        )

    null_means = np.asarray(
        null_means
    )

    p = (
        np.sum(
            null_means
            >= observed
            - 1e-12
        )
        / len(null_means)
    )

    return float(p)


# =========================================================
# EFFECT SIZE:
# paired Cohen's dz
# =========================================================

def cohens_dz(
    differences
):

    differences = np.asarray(
        differences,
        dtype=np.float64
    )

    sd = differences.std(
        ddof=1
    )

    if sd == 0:
        return float("inf")

    return float(
        differences.mean()
        / sd
    )


# =========================================================
# ANALYZE ONE COMPARISON
# =========================================================

def analyze(
    name,
    method_a,
    method_b
):

    differences = (
        method_a
        - method_b
    )

    mean_diff = float(
        differences.mean()
    )

    std_diff = float(
        differences.std(
            ddof=1
        )
    )

    lower, upper = bootstrap_ci(
        differences,
        N_BOOTSTRAP,
        SEED
    )

    p_value = (
        exact_sign_flip_test(
            differences
        )
    )

    dz = cohens_dz(
        differences
    )

    print(
        "\n"
        + "=" * 70
    )

    print(name)

    print(
        "=" * 70
    )

    print(
        "Per-seed differences:"
    )

    for i, diff in enumerate(
        differences
    ):

        print(
            f"Seed {i+1}: "
            f"{diff:+.4f}"
        )

    print(
        f"\nMean AUROC difference: "
        f"{mean_diff:+.4f}"
    )

    print(
        f"SD of paired differences: "
        f"{std_diff:.4f}"
    )

    print(
        f"95% bootstrap CI: "
        f"[{lower:+.4f}, {upper:+.4f}]"
    )

    print(
        f"Exact sign-flip p: "
        f"{p_value:.4f}"
    )

    print(
        f"Paired Cohen dz: "
        f"{dz:.3f}"
    )

    return {
        "comparison":
            name,

        "mean_difference":
            mean_diff,

        "std_difference":
            std_diff,

        "bootstrap_95_ci": [
            lower,
            upper
        ],

        "exact_sign_flip_p":
            p_value,

        "cohens_dz":
            dz,

        "differences":
            differences.tolist()
    }


# =========================================================
# MAIN
# =========================================================

def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "seed",
        "mlp_test_auroc",
        "real_epigraph_test_auroc",
        "shuffled_graphsage_test_auroc"
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"Missing column: "
                f"{column}"
            )

    print("=" * 70)

    print(
        "PAIRED SEED UNCERTAINTY ANALYSIS"
    )

    print("=" * 70)

    print(
        f"\nSeeds: "
        f"{df['seed'].tolist()}"
    )

    real = df[
        "real_epigraph_test_auroc"
    ].to_numpy()

    mlp = df[
        "mlp_test_auroc"
    ].to_numpy()

    shuffled = df[
        "shuffled_graphsage_test_auroc"
    ].to_numpy()

    # =====================================================
    # RAW MEANS
    # =====================================================

    print(
        "\nMean test AUROC:"
    )

    print(
        f"EpiGraph: "
        f"{real.mean():.4f}"
    )

    print(
        f"MLP:      "
        f"{mlp.mean():.4f}"
    )

    print(
        f"Shuffled: "
        f"{shuffled.mean():.4f}"
    )

    # =====================================================
    # PAIRED COMPARISONS
    # =====================================================

    epigraph_vs_mlp = analyze(
        "EpiGraph - MLP",
        real,
        mlp
    )

    epigraph_vs_shuffled = analyze(
        "EpiGraph - Shuffled GraphSAGE",
        real,
        shuffled
    )

    # =====================================================
    # SAVE
    # =====================================================

    output = {
        "num_seeds":
            int(len(df)),

        "seeds":
            df["seed"].tolist(),

        "warning":
            (
                "Only 5 paired seeds are available. "
                "Seed-level significance testing has low resolution; "
                "interpret confidence intervals and effect sizes "
                "as exploratory robustness evidence."
            ),

        "epigraph_vs_mlp":
            epigraph_vs_mlp,

        "epigraph_vs_shuffled":
            epigraph_vs_shuffled
    }

    output_path = (
        RESULT_DIR
        / "step16_paired_uncertainty.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Saved:\n{output_path}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "With only 5 seeds, even a perfectly consistent "
        "difference cannot provide very fine p-value resolution."
    )


if __name__ == "__main__":
    main()