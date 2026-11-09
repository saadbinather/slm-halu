"""Build PCA-16/PCA-32 Layer-Time Transition Graphs for EpiGraph.

Geometry is standardized with training-only, non-EOS transition nodes. Graphs
exclude EOS checkpoints and preserve the original saved-checkpoint adjacency.
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm


INPUT_ROOTS = {
    16: Path("processed_features/transitions_pca16"),
    32: Path("processed_features/transitions_pca32"),
}
OUTPUT_ROOTS = {
    16: Path("processed_graphs/transitions_pca16"),
    32: Path("processed_graphs/transitions_pca32"),
}
MODEL_DIR = Path("models/transition_graph")
SCALER_PATH = MODEL_DIR / "geometry_standardizer.joblib"

COMPONENTS = (16, 32)
SPLITS = ("train", "val", "test")
GEOMETRY_NAMES = (
    "transition_magnitude",
    "relative_update_magnitude",
    "hidden_cosine_similarity",
    "update_alignment",
)
EXPECTED_TRANSITIONS = 27
EXPECTED_TIMES = (0, 1, 2, 4, 8)

DEPTH_EDGE = 0
TEMPORAL_EDGE = 1


def load_file(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def get_files(components, split):
    folder = INPUT_ROOTS[components] / split
    files = sorted(folder.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No PCA-{components} transition files found in {folder}")
    return files


def validate_source(item, path, components, expected_split):
    required = {
        "sample_id", "split", "label", "label_name", "time_positions",
        "expected_time_positions", "checkpoint_availability_mask",
        "checkpoint_token_ids", "checkpoint_is_eos", "checkpoint_token_types",
        "eos_token_id", "num_layer_transitions", "layer_transition_indices",
        "layer_index_normalized", "time_normalized", "transition_features",
        "transition_magnitude", "relative_update_magnitude",
        "hidden_cosine_similarity", "update_alignment", "geometric_features",
        "pca_dim", "pca_fit_split",
    }
    missing = required.difference(item)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")

    features = item["transition_features"]
    times = [int(t) for t in item["time_positions"]]
    is_eos = item["checkpoint_is_eos"]
    if tuple(features.shape) != (len(times), EXPECTED_TRANSITIONS, components):
        raise ValueError(f"{path}: unexpected PCA feature shape {tuple(features.shape)}")
    if is_eos.shape != (len(times),):
        raise ValueError(f"{path}: EOS metadata does not align with checkpoints")
    if (is_eos < 0).any():
        raise ValueError(f"{path}: EOS status is unknown; refusing to construct ambiguous nodes")
    if str(item["split"]) != expected_split:
        raise ValueError(f"{path}: split={item['split']!r}, expected {expected_split!r}")
    if int(item["pca_dim"]) != components or item["pca_fit_split"] != "train":
        raise ValueError(f"{path}: invalid PCA provenance")
    if times[0] != 0 or int(is_eos[0]) != 0:
        raise ValueError(f"{path}: t=0 must exist and must not be EOS")
    if any(t not in EXPECTED_TIMES for t in times) or times != sorted(times):
        raise ValueError(f"{path}: unexpected checkpoint ordering {times}")
    tensors = [features] + [item[name] for name in GEOMETRY_NAMES]
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError(f"{path}: source contains NaN/Inf")


def geometry_matrix(item):
    return torch.stack([item[name].float() for name in GEOMETRY_NAMES], dim=-1)


def non_eos_rows(item):
    return item["checkpoint_is_eos"].to(torch.int8) != 1


def atomic_joblib_dump(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_scaler(scaler, path):
    if int(scaler.n_features_in_) != len(GEOMETRY_NAMES):
        raise ValueError(f"{path}: expected {len(GEOMETRY_NAMES)} scaler features")
    if getattr(scaler, "training_split_", None) != "train":
        raise ValueError(f"{path}: missing train-only provenance")
    if tuple(getattr(scaler, "feature_names_", ())) != GEOMETRY_NAMES:
        raise ValueError(f"{path}: geometry feature order mismatch")
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError(f"{path}: scaler contains NaN/Inf")


def fit_geometry_scaler():
    """Fit only on training nodes that will actually appear in graphs."""
    files = get_files(16, "train")
    scaler = StandardScaler()
    rows = 0

    print(f"Fitting geometry scaler on TRAIN non-EOS nodes ({len(files):,} files)")
    for path in tqdm(files, desc="Fit geometry scaler"):
        item = load_file(path)
        validate_source(item, path, 16, "train")
        keep = non_eos_rows(item)
        values = geometry_matrix(item)[keep].reshape(-1, len(GEOMETRY_NAMES)).numpy()
        scaler.partial_fit(values)
        rows += values.shape[0]

    scaler.training_split_ = "train"
    scaler.training_file_count_ = len(files)
    scaler.training_node_count_ = rows
    scaler.eos_excluded_ = True
    scaler.feature_names_ = GEOMETRY_NAMES
    validate_scaler(scaler, SCALER_PATH)
    return scaler


def load_or_fit_scaler(overwrite):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if SCALER_PATH.exists() and not overwrite:
        scaler = joblib.load(SCALER_PATH)
        validate_scaler(scaler, SCALER_PATH)
        print(f"Loaded existing geometry scaler: {SCALER_PATH}")
        return scaler

    scaler = fit_geometry_scaler()
    atomic_joblib_dump(scaler, SCALER_PATH)
    reloaded = joblib.load(SCALER_PATH)
    validate_scaler(reloaded, SCALER_PATH)
    print(
        f"Saved geometry scaler: {SCALER_PATH} "
        f"({reloaded.training_node_count_:,} train nodes)"
    )
    return reloaded


def build_graph(item, scaler, components):
    times = [int(t) for t in item["time_positions"]]
    eos_flags = item["checkpoint_is_eos"].to(torch.int8)
    keep_rows = [index for index, flag in enumerate(eos_flags.tolist()) if flag != 1]
    kept_times = [times[index] for index in keep_rows]

    pca = item["transition_features"].float()[keep_rows]
    geometry = geometry_matrix(item)[keep_rows]
    flat_geometry = geometry.reshape(-1, len(GEOMETRY_NAMES)).numpy()
    standardized = torch.from_numpy(
        scaler.transform(flat_geometry).astype(np.float32, copy=False)
    ).reshape(len(keep_rows), EXPECTED_TRANSITIONS, len(GEOMETRY_NAMES))

    layer_norm = item["layer_index_normalized"].float()
    time_norm = item["time_normalized"].float()[keep_rows]
    layer_positions = layer_norm.view(1, -1, 1).expand(len(keep_rows), -1, -1)
    time_positions_norm = time_norm.view(-1, 1, 1).expand(-1, EXPECTED_TRANSITIONS, -1)
    x = torch.cat([pca, standardized, layer_positions, time_positions_norm], dim=-1)
    x = x.reshape(-1, components + 6).contiguous()

    def node_id(kept_time_index, transition_index):
        return kept_time_index * EXPECTED_TRANSITIONS + transition_index

    sources = []
    targets = []
    edge_types = []
    time_gaps = []
    time_deltas = []

    # Neighboring transition positions at the same retained checkpoint.
    for kept_index in range(len(keep_rows)):
        for transition_index in range(EXPECTED_TRANSITIONS - 1):
            left = node_id(kept_index, transition_index)
            right = node_id(kept_index, transition_index + 1)
            for source, target in ((left, right), (right, left)):
                sources.append(source)
                targets.append(target)
                edge_types.append(DEPTH_EDGE)
                time_gaps.append(0)
                time_deltas.append(0)

    # Connect only adjacent rows in the original saved sequence. If either row
    # is EOS, no temporal edge is created and the excluded row is not bridged.
    original_to_kept = {original: kept for kept, original in enumerate(keep_rows)}
    for original_index in range(len(times) - 1):
        next_original = original_index + 1
        if original_index not in original_to_kept or next_original not in original_to_kept:
            continue
        left_time = original_to_kept[original_index]
        right_time = original_to_kept[next_original]
        gap = times[next_original] - times[original_index]
        if gap <= 0:
            raise ValueError("temporal checkpoints must be strictly increasing")
        for transition_index in range(EXPECTED_TRANSITIONS):
            earlier = node_id(left_time, transition_index)
            later = node_id(right_time, transition_index)
            sources.extend([earlier, later])
            targets.extend([later, earlier])
            edge_types.extend([TEMPORAL_EDGE, TEMPORAL_EDGE])
            time_gaps.extend([gap, gap])
            time_deltas.extend([gap, -gap])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    transition_indices = torch.arange(EXPECTED_TRANSITIONS, dtype=torch.long)
    transition_index = transition_indices.repeat(len(keep_rows))
    layer_pairs = item["layer_transition_indices"].long().repeat(len(keep_rows), 1)
    node_times = torch.tensor(kept_times, dtype=torch.long).repeat_interleave(EXPECTED_TRANSITIONS)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        edge_time_gap=torch.tensor(time_gaps, dtype=torch.long),
        edge_time_delta=torch.tensor(time_deltas, dtype=torch.long),
        y=torch.tensor([int(item["label"])], dtype=torch.long),
        transition_index=transition_index,
        source_layer_index=layer_pairs[:, 0].contiguous(),
        target_layer_index=layer_pairs[:, 1].contiguous(),
        time_position=node_times,
    )
    graph.sample_id = str(item["sample_id"])
    graph.split = str(item["split"])
    graph.label_name = str(item["label_name"])
    graph.pca_dim = int(components)
    graph.node_feature_dim = int(components + 6)
    graph.num_layer_transitions = EXPECTED_TRANSITIONS
    graph.original_time_positions = torch.tensor(times, dtype=torch.long)
    graph.included_time_positions = torch.tensor(kept_times, dtype=torch.long)
    graph.checkpoint_availability_mask = item["checkpoint_availability_mask"].clone()
    graph.checkpoint_token_ids = item["checkpoint_token_ids"].clone()
    graph.checkpoint_is_eos = item["checkpoint_is_eos"].clone()
    graph.checkpoint_token_types = list(item["checkpoint_token_types"])
    graph.eos_token_id = item["eos_token_id"]
    graph.excluded_eos_count = int((eos_flags == 1).sum())
    graph.edge_type_names = {DEPTH_EDGE: "depth", TEMPORAL_EDGE: "temporal"}
    graph.geometry_feature_names = list(GEOMETRY_NAMES)
    graph.geometry_scaler = str(SCALER_PATH)
    graph.geometry_fit_split = "train"
    return graph


def validate_graph(graph, source, components, path):
    expected_dim = components + 6
    included_times = graph.included_time_positions.tolist()
    source_times = [int(t) for t in source["time_positions"]]
    expected_times = [
        time for time, flag in zip(source_times, source["checkpoint_is_eos"].tolist())
        if flag != 1
    ]
    if included_times != expected_times:
        raise ValueError(f"{path}: incorrect EOS filtering")
    if 0 not in included_times:
        raise ValueError(f"{path}: t=0 was lost")
    if int((graph.time_position == 0).sum()) != EXPECTED_TRANSITIONS:
        raise ValueError(f"{path}: t=0 must contain exactly 27 transition nodes")
    if graph.x.shape != (len(included_times) * EXPECTED_TRANSITIONS, expected_dim):
        raise ValueError(f"{path}: unexpected node feature shape {tuple(graph.x.shape)}")
    if len(included_times) == 1 and graph.num_nodes != EXPECTED_TRANSITIONS:
        raise ValueError(f"{path}: t=0-only graph must contain 27 nodes")
    if graph.num_nodes > len(EXPECTED_TIMES) * EXPECTED_TRANSITIONS:
        raise ValueError(f"{path}: graph exceeds 135 nodes")
    if not torch.isfinite(graph.x).all():
        raise ValueError(f"{path}: node features contain NaN/Inf")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(f"{path}: invalid edge_index shape")
    edge_count = graph.edge_index.shape[1]
    for attribute in (graph.edge_type, graph.edge_time_gap, graph.edge_time_delta):
        if attribute.shape != (edge_count,):
            raise ValueError(f"{path}: edge attribute length mismatch")
    if edge_count and (graph.edge_index.min() < 0 or graph.edge_index.max() >= graph.num_nodes):
        raise ValueError(f"{path}: edge points outside node range")
    if not set(graph.edge_type.tolist()).issubset({DEPTH_EDGE, TEMPORAL_EDGE}):
        raise ValueError(f"{path}: invalid edge type")

    expected_depth = len(included_times) * (EXPECTED_TRANSITIONS - 1) * 2
    expected_temporal_pairs = sum(
        1
        for index in range(len(source_times) - 1)
        if source["checkpoint_is_eos"][index] != 1
        and source["checkpoint_is_eos"][index + 1] != 1
    )
    expected_temporal = expected_temporal_pairs * EXPECTED_TRANSITIONS * 2
    if int((graph.edge_type == DEPTH_EDGE).sum()) != expected_depth:
        raise ValueError(f"{path}: incorrect depth-edge count")
    if int((graph.edge_type == TEMPORAL_EDGE).sum()) != expected_temporal:
        raise ValueError(f"{path}: incorrect temporal-edge count")
    if not torch.all(graph.edge_time_gap[graph.edge_type == DEPTH_EDGE] == 0):
        raise ValueError(f"{path}: depth edges have nonzero temporal gaps")
    if not torch.all(graph.edge_time_gap[graph.edge_type == TEMPORAL_EDGE] > 0):
        raise ValueError(f"{path}: temporal edges must have positive gaps")

    source_nodes = graph.edge_index[0]
    target_nodes = graph.edge_index[1]
    source_times = graph.time_position[source_nodes]
    target_times = graph.time_position[target_nodes]
    source_transitions = graph.transition_index[source_nodes]
    target_transitions = graph.transition_index[target_nodes]
    depth_mask = graph.edge_type == DEPTH_EDGE
    temporal_mask = graph.edge_type == TEMPORAL_EDGE
    if not torch.all(source_times[depth_mask] == target_times[depth_mask]):
        raise ValueError(f"{path}: depth edge crosses time")
    if not torch.all(torch.abs(source_transitions[depth_mask] - target_transitions[depth_mask]) == 1):
        raise ValueError(f"{path}: depth edge does not join neighboring transitions")
    if not torch.all(source_transitions[temporal_mask] == target_transitions[temporal_mask]):
        raise ValueError(f"{path}: temporal edge changes transition index")
    observed_deltas = target_times[temporal_mask] - source_times[temporal_mask]
    if not torch.equal(observed_deltas, graph.edge_time_delta[temporal_mask]):
        raise ValueError(f"{path}: temporal edge delta is incorrect")
    if not torch.equal(torch.abs(observed_deltas), graph.edge_time_gap[temporal_mask]):
        raise ValueError(f"{path}: temporal edge gap is incorrect")


def process_all(scaler, overwrite, limit):
    summaries = {}
    for components in COMPONENTS:
        component_summary = {}
        for split in SPLITS:
            files = get_files(components, split)
            if limit > 0:
                files = files[:limit]
            output_dir = OUTPUT_ROOTS[components] / split
            output_dir.mkdir(parents=True, exist_ok=True)
            new = existing_count = bytes_total = 0
            nodes = []
            edges = []
            coverage = Counter()

            for path in tqdm(files, desc=f"Graphs PCA-{components} {split}"):
                source = load_file(path)
                validate_source(source, path, components, split)
                output_path = output_dir / path.name
                if output_path.exists() and not overwrite:
                    graph = load_file(output_path)
                    validate_graph(graph, source, components, output_path)
                    existing_count += 1
                else:
                    graph = build_graph(source, scaler, components)
                    validate_graph(graph, source, components, output_path)
                    atomic_torch_save(graph, output_path)
                    reloaded = load_file(output_path)
                    validate_graph(reloaded, source, components, output_path)
                    graph = reloaded
                    new += 1

                nodes.append(graph.num_nodes)
                edges.append(graph.num_edges)
                coverage.update(graph.included_time_positions.tolist())
                bytes_total += output_path.stat().st_size

            component_summary[split] = {
                "selected": len(files), "new": new, "existing": existing_count,
                "bytes": bytes_total, "nodes": nodes, "edges": edges,
                "coverage": {time: coverage[time] for time in EXPECTED_TIMES},
            }
        summaries[components] = component_summary
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Per-split limit; 0 builds all graphs")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace scaler and graphs")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    scaler = load_or_fit_scaler(args.overwrite)
    summaries = process_all(scaler, args.overwrite, args.limit)

    print("\n" + "=" * 76)
    print("TRANSITION GRAPH SUMMARY")
    print("=" * 76)
    print(
        f"Geometry scaler: train files={scaler.training_file_count_:,}; "
        f"non-EOS nodes={scaler.training_node_count_:,}"
    )
    for components in COMPONENTS:
        for split in SPLITS:
            values = summaries[components][split]
            print(
                f"PCA-{components} {split}: selected={values['selected']} "
                f"new={values['new']} existing={values['existing']} "
                f"nodes={min(values['nodes'])}-{max(values['nodes'])} "
                f"edges={min(values['edges'])}-{max(values['edges'])} "
                f"size={values['bytes'] / (1024 ** 2):.1f} MiB "
                f"coverage={values['coverage']}"
            )


if __name__ == "__main__":
    main()
