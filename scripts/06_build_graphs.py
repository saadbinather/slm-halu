import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch_geometric.data import Data


# =========================================================
# CONFIG
# =========================================================

FEATURE_DIR = Path("processed_features")
GRAPH_DIR = Path("graphs")

# Maximum generation position used in our experiment
MAX_TIME = 8


# =========================================================
# LOAD FEATURE FILES
# =========================================================

def get_feature_files(split):

    folder = FEATURE_DIR / split

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing folder: {folder}"
        )

    files = sorted(
        folder.glob("*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No processed feature files found in {folder}"
        )

    return files


# =========================================================
# NODE INDEX
# =========================================================

def node_id(
    time_index,
    layer_index,
    num_layers
):
    """
    Convert (time row, layer) into one graph node ID.

    Example with 28 layers:

    time row 0:
        nodes 0 ... 27

    time row 1:
        nodes 28 ... 55
    """

    return (
        time_index * num_layers
        + layer_index
    )


# =========================================================
# BUILD ONE GRAPH
# =========================================================

def build_graph(trace):

    # -----------------------------------------------------
    # PCA FEATURES
    #
    # Shape:
    # [num_times, num_layers, 16]
    # -----------------------------------------------------

    features = trace[
        "features"
    ].float()

    time_positions = trace[
        "time_positions"
    ]

    label = int(
        trace["label"]
    )

    sample_id = str(
        trace["sample_id"]
    )

    num_times = features.shape[0]
    num_layers = features.shape[1]
    pca_dim = features.shape[2]

    # -----------------------------------------------------
    # NODE FEATURES
    #
    # PCA(16)
    # +
    # normalized layer position
    # +
    # normalized time position
    #
    # total = 18 dimensions
    # -----------------------------------------------------

    node_features = []

    layer_ids = []
    time_ids = []

    for time_idx, time_pos in enumerate(
        time_positions
    ):

        # fixed normalization
        time_norm = float(
            time_pos
        ) / MAX_TIME

        for layer_idx in range(
            num_layers
        ):

            # 0 ... 1
            if num_layers > 1:

                layer_norm = (
                    layer_idx
                    /
                    (num_layers - 1)
                )

            else:

                layer_norm = 0.0

            pca_vector = features[
                time_idx,
                layer_idx
            ]

            position_features = (
                torch.tensor(
                    [
                        layer_norm,
                        time_norm
                    ],
                    dtype=torch.float32
                )
            )

            node_x = torch.cat(
                [
                    pca_vector,
                    position_features
                ],
                dim=0
            )

            node_features.append(
                node_x
            )

            layer_ids.append(
                layer_idx
            )

            time_ids.append(
                int(time_pos)
            )

    # ---------------------------------------------
    # Final x
    #
    # [num_nodes, 18]
    # ---------------------------------------------

    x = torch.stack(
        node_features,
        dim=0
    )

    # =====================================================
    # EDGES
    # =====================================================

    edge_source = []
    edge_target = []

    edge_types = []

    # -----------------------------------------------------
    # EDGE TYPE:
    #
    # 0 = depth edge
    # 1 = time edge
    # -----------------------------------------------------


    # =====================================================
    # A. DEPTH EDGES
    #
    # Same time:
    #
    # L0 <-> L1 <-> L2 ... <-> L27
    # =====================================================

    for time_idx in range(
        num_times
    ):

        for layer_idx in range(
            num_layers - 1
        ):

            a = node_id(
                time_idx,
                layer_idx,
                num_layers
            )

            b = node_id(
                time_idx,
                layer_idx + 1,
                num_layers
            )

            # Forward
            edge_source.append(a)
            edge_target.append(b)
            edge_types.append(0)

            # Reverse
            edge_source.append(b)
            edge_target.append(a)
            edge_types.append(0)


    # =====================================================
    # B. TIME EDGES
    #
    # Same layer across consecutive SAVED time points:
    #
    # t0 <-> t1 <-> t2 <-> t4 <-> t8
    #
    # if those positions exist
    # =====================================================

    for time_idx in range(
        num_times - 1
    ):

        next_time_idx = (
            time_idx + 1
        )

        for layer_idx in range(
            num_layers
        ):

            a = node_id(
                time_idx,
                layer_idx,
                num_layers
            )

            b = node_id(
                next_time_idx,
                layer_idx,
                num_layers
            )

            # Forward
            edge_source.append(a)
            edge_target.append(b)
            edge_types.append(1)

            # Reverse
            edge_source.append(b)
            edge_target.append(a)
            edge_types.append(1)


    # =====================================================
    # EDGE INDEX
    # =====================================================

    edge_index = torch.tensor(
        [
            edge_source,
            edge_target
        ],
        dtype=torch.long
    )

    edge_type = torch.tensor(
        edge_types,
        dtype=torch.long
    )


    # =====================================================
    # LABEL
    # =====================================================

    y = torch.tensor(
        [label],
        dtype=torch.long
    )


    # =====================================================
    # EXTRA NODE METADATA
    # =====================================================

    layer_index = torch.tensor(
        layer_ids,
        dtype=torch.long
    )

    time_position = torch.tensor(
        time_ids,
        dtype=torch.long
    )


    # =====================================================
    # CREATE PyG DATA OBJECT
    # =====================================================

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        layer_index=layer_index,
        time_position=time_position
    )

    # Safe metadata
    graph.sample_id = sample_id

    graph.num_time_steps = int(
        num_times
    )

    graph.num_transformer_layers = int(
        num_layers
    )

    graph.pca_dim = int(
        pca_dim
    )

    return graph


# =========================================================
# PROCESS SPLIT
# =========================================================

def process_split(
    split,
    limit=0
):

    files = get_feature_files(
        split
    )

    if limit > 0:
        files = files[:limit]

    output_dir = (
        GRAPH_DIR / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print(
        f"BUILDING EPIGRAPH GRAPHS — {split.upper()}"
    )
    print("=" * 70)

    print(
        f"\nInput samples: {len(files):,}"
    )

    created = 0
    skipped = 0
    failed = 0

    node_counts = []
    edge_counts = []

    for path in tqdm(
        files,
        desc=f"Graphs {split}"
    ):

        output_path = (
            output_dir
            / path.name
        )

        # resume support
        if output_path.exists():

            skipped += 1
            continue

        try:

            trace = torch.load(
                path,
                map_location="cpu",
                weights_only=False
            )

            # Only binary C/H samples
            if int(
                trace["label"]
            ) not in [0, 1]:

                failed += 1
                continue

            graph = build_graph(
                trace
            )

            torch.save(
                graph,
                output_path
            )

            node_counts.append(
                graph.num_nodes
            )

            edge_counts.append(
                graph.num_edges
            )

            created += 1

        except Exception as e:

            failed += 1

            print(
                f"\nFAILED {path.name}: {e}"
            )

    print("\nSummary:")

    print(
        f"New graphs:       {created}"
    )

    print(
        f"Already existed:  {skipped}"
    )

    print(
        f"Failed/skipped:    {failed}"
    )

    if node_counts:

        print(
            f"\nNodes per graph:"
            f" min={min(node_counts)},"
            f" max={max(node_counts)},"
            f" mean={sum(node_counts)/len(node_counts):.1f}"
        )

        print(
            f"Edges per graph:"
            f" min={min(edge_counts)},"
            f" max={max(edge_counts)},"
            f" mean={sum(edge_counts)/len(edge_counts):.1f}"
        )

    print(
        f"\nSaved in: {output_dir}"
    )


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
            "test",
            "all"
        ],
        default="all"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = use full split"
    )

    args = parser.parse_args()

    print("=" * 70)

    print(
        "EPIGRAPH — STEP 06: GRAPH CONSTRUCTION"
    )

    print("=" * 70)

    if args.split == "all":

        splits = [
            "train",
            "val",
            "test"
        ]

    else:

        splits = [
            args.split
        ]

    for split in splits:

        process_split(
            split=split,
            limit=args.limit
        )

    print("\n" + "=" * 70)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()