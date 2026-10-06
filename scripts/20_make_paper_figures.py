from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "paper_details"

OUT_DIR.mkdir(parents=True, exist_ok=True)


EARLY_FILE = RESULTS_DIR / "step10_early_detection_seed42.json"
MULTISEED_FILE = RESULTS_DIR / "step14_t0_multiseed.csv"
LAYERWISE_FILE = RESULTS_DIR / "step13_t0_layerwise_seed42.csv"
HEATMAP_FILE = RESULTS_DIR / "step15_layer_time_heatmap.csv"


# --------------------------------------------------
# Paper style
# --------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})


BLUE = "#356FA8"
LIGHT_BLUE = "#BFD7EA"
ORANGE = "#E69F45"
RED = "#C84C4C"
GRAY = "#9BAAB5"
DARK_GRAY = "#596773"


def save_figure(fig, name):
    fig.tight_layout()

    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"

    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# --------------------------------------------------
# 1. Early / pre-generation detection
# --------------------------------------------------

def plot_early_detection():
    if not EARLY_FILE.exists():
        print("Missing:", EARLY_FILE)
        return

    data = json.loads(EARLY_FILE.read_text(encoding="utf-8"))
    rows = data["results"]

    auroc = [r["test"]["auroc"] for r in rows]
    bacc = [r["test"]["balanced_accuracy"] for r in rows]

    labels = [
        "$t=0$",
        "$t\\leq1$",
        "$t\\leq2$",
        "$t\\leq4$",
        "$t\\leq8$",
    ]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    ax.plot(
        x,
        auroc,
        marker="o",
        markersize=7,
        linewidth=2.4,
        color=BLUE,
        label="Test AUROC",
    )

    ax.plot(
        x,
        bacc,
        marker="s",
        markersize=6,
        linewidth=2.1,
        color=ORANGE,
        label="Balanced accuracy",
    )

    best = int(np.argmax(auroc))

    ax.scatter(
        x[best],
        auroc[best],
        s=105,
        color=RED,
        zorder=5,
    )

    ax.annotate(
        f"Best: $t=0$ ({auroc[best]:.4f})",
        xy=(x[best], auroc[best]),
        xytext=(15, 12),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#cccccc",
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Available generation states")
    ax.set_ylabel("Score")
    ax.set_title("Pre-Generation Detection")

    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(frameon=False)

    save_figure(fig, "fig_early_detection")


# --------------------------------------------------
# 2. Five-seed comparison
# --------------------------------------------------

def plot_multiseed_comparison():
    if not MULTISEED_FILE.exists():
        print("Missing:", MULTISEED_FILE)
        return

    df = pd.read_csv(MULTISEED_FILE)

    methods = [
        "All-layer MLP",
        "EpiGraph",
        "Shuffled-depth\nGraphSAGE",
    ]

    cols = [
        "mlp_test_auroc",
        "real_epigraph_test_auroc",
        "shuffled_graphsage_test_auroc",
    ]

    means = np.array([df[c].mean() for c in cols])
    stds = np.array([df[c].std(ddof=1) for c in cols])

    colors = [
        LIGHT_BLUE,
        BLUE,
        "#CAD4DB",
    ]

    edges = [
        "#6B91B5",
        "#244F78",
        "#788995",
    ]

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    bars = ax.bar(
        x,
        means,
        width=0.68,
        color=colors,
        edgecolor=edges,
        linewidth=1.3,
        yerr=stds,
        capsize=5,
        error_kw={
            "elinewidth": 1.3,
            "capthick": 1.3,
        },
    )

    # Individual seed points
    for i, col in enumerate(cols):
        values = df[col].to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(values))

        ax.scatter(
            np.full(len(values), x[i]) + jitter,
            values,
            s=35,
            color=edges[i],
            zorder=4,
        )

    for bar, mean, sd in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + sd + 0.001,
            f"{mean:.4f}\n$\\pm$ {sd:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.set_ylabel("Test AUROC")
    ax.set_title(r"$t=0$ Five-Seed Comparison")

    ax.set_ylim(
        min(means - stds) - 0.006,
        max(means + stds) + 0.007,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.28)

    save_figure(fig, "fig_t0_multiseed_comparison")


# --------------------------------------------------
# 3. Layer-wise t=0 probes
# --------------------------------------------------

def plot_layerwise():
    if not LAYERWISE_FILE.exists():
        print("Missing:", LAYERWISE_FILE)
        return

    df = pd.read_csv(LAYERWISE_FILE)

    layer_col = None
    val_col = None
    test_col = None

    for col in df.columns:
        name = col.lower()

        if "layer" in name:
            layer_col = col

        if "val" in name and "auroc" in name:
            val_col = col

        if "test" in name and "auroc" in name:
            test_col = col

    if layer_col is None or test_col is None:
        print("Could not detect layer-wise columns.")
        print("Columns:", list(df.columns))
        return

    # CSV uses zero-based indexing.
    # Paper reports transformer layers as 1--28.
    df[layer_col] = df[layer_col].astype(int) + 1

    df = df.sort_values(layer_col)

    fig, ax = plt.subplots(figsize=(7.5, 4.7))

    ax.plot(
        df[layer_col],
        df[test_col],
        marker="o",
        markersize=4,
        linewidth=2.2,
        color=BLUE,
        label="Test AUROC",
    )

    if val_col is not None:
        ax.plot(
            df[layer_col],
            df[val_col],
            marker="s",
            markersize=3.5,
            linewidth=1.8,
            color=ORANGE,
            alpha=0.9,
            label="Validation AUROC",
        )

        best_idx = df[val_col].idxmax()
    else:
        best_idx = df[test_col].idxmax()

    best_layer = int(df.loc[best_idx, layer_col])
    best_test = float(df.loc[best_idx, test_col])

    ax.scatter(
        best_layer,
        best_test,
        s=90,
        color=RED,
        zorder=5,
    )

    ax.annotate(
        f"Selected: Layer {best_layer}",
        xy=(best_layer, best_test),
        xytext=(10, 14),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#cccccc",
        ),
    )

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("AUROC")
    ax.set_title(r"Layer-Wise Probes at $t=0$")

    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(frameon=False)

    save_figure(fig, "fig_layerwise_t0")


# --------------------------------------------------
# 4. Layer-time heatmap
# --------------------------------------------------

def plot_layer_time_heatmap():
    if not HEATMAP_FILE.exists():
        print("Missing:", HEATMAP_FILE)
        return

    df = pd.read_csv(HEATMAP_FILE)

    cols_lower = {c.lower(): c for c in df.columns}

    layer_col = next(
        (c for c in df.columns if "layer" in c.lower()),
        None,
    )

    time_col = next(
        (
            c for c in df.columns
            if "time" in c.lower() or "checkpoint" in c.lower()
        ),
        None,
    )

    test_col = next(
        (
            c for c in df.columns
            if "test" in c.lower() and "auroc" in c.lower()
        ),
        None,
    )

    # Long format
    if layer_col and time_col and test_col:
        temp = df.copy()
        temp[layer_col] = temp[layer_col].astype(int) + 1

        pivot = temp.pivot(
            index=layer_col,
            columns=time_col,
            values=test_col,
        )

    # Wide format fallback
    else:
        if layer_col is None:
            layer_col = df.columns[0]

        temp = df.copy()
        temp[layer_col] = temp[layer_col].astype(int) + 1
        temp = temp.set_index(layer_col)

        numeric_cols = []

        for c in temp.columns:
            converted = pd.to_numeric(temp[c], errors="coerce")
            if converted.notna().any():
                temp[c] = converted
                numeric_cols.append(c)

        pivot = temp[numeric_cols]

    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(6.5, 7.0))

    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )

    ax.set_xlabel("Generation checkpoint")
    ax.set_ylabel("Transformer layer")
    ax.set_title("Layer--Time Probe AUROC")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])

    # Avoid overcrowding y-axis
    step = 2
    yticks = np.arange(0, len(pivot.index), step)

    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [str(pivot.index[i]) for i in yticks]
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Test AUROC")



    save_figure(fig, "fig_layer_time_heatmap")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    print("Creating paper figures...")

    plot_early_detection()
    plot_multiseed_comparison()
    plot_layerwise()
    plot_layer_time_heatmap()

    print(f"\nDone. Figures saved in:\n{OUT_DIR}")