#!/usr/bin/env python3
"""
Trajectory Classification and t-SNE Visualization Script based on Anchor Samples.

Classifies trajectory waypoints into groups by assigning each trajectory to its
nearest specified anchor point (exemplar sample). Then visualizes the resulting
classification on a 2D t-SNE embedding plot, coloring each sample by its group
and highlighting the anchor samples.

Usage:
  # Positional arguments:
  python scripts/classify_tsne.py <waypoints_dir> <anchor1> <anchor2> <anchor3>

  # Named arguments:
  python scripts/classify_tsne.py --waypoints_dir <waypoints_dir> --anchors <anchor1> <anchor2> <anchor3>

  # Example:
  python scripts/classify_tsne.py logs/maze2d-large-v1/plans/release_H384_T256_LimitsNormalizer_b1000_condFalse/waypoints 0 100 500
"""

import os
import sys
import re
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import pandas as pd
except ImportError:
    pd = None

# Curated palette for up to 10 groups
GROUP_PALETTES = [
    {
        "name": "Blue",
        "primary": "#1f77b4",
        "fill": "#edf4fc",
        "edge": "#1f77b4",
        "text": "#0f2e4d",
        "anchor_fill": "#0a4b78",
        "anchor_edge": "#ffd700",
    },
    {
        "name": "Amber",
        "primary": "#e67e22",
        "fill": "#fef5ec",
        "edge": "#d35400",
        "text": "#6e2c00",
        "anchor_fill": "#b94a00",
        "anchor_edge": "#ffd700",
    },
    {
        "name": "Emerald",
        "primary": "#27ae60",
        "fill": "#eafaf1",
        "edge": "#1e8449",
        "text": "#114b29",
        "anchor_fill": "#196f3d",
        "anchor_edge": "#ffd700",
    },
    {
        "name": "Purple",
        "primary": "#8e44ad",
        "fill": "#f4ecf7",
        "edge": "#6c3483",
        "text": "#4a235a",
        "anchor_fill": "#5b2c6f",
        "anchor_edge": "#ffd700",
    },
    {
        "name": "Crimson",
        "primary": "#c0392b",
        "fill": "#fdedec",
        "edge": "#922b21",
        "text": "#641e16",
        "anchor_fill": "#78281f",
        "anchor_edge": "#ffd700",
    },
    {
        "name": "Teal",
        "primary": "#16a085",
        "fill": "#e8f8f5",
        "edge": "#117864",
        "text": "#0b4f42",
        "anchor_fill": "#0e6251",
        "anchor_edge": "#ffd700",
    },
]


def normalize_path(path_str: str) -> str:
    """Normalize Windows UNC / WSL / relative paths to standard POSIX path."""
    if not path_str:
        return path_str
    path_str = path_str.replace("\\", "/")
    match = re.match(r"^//wsl(?:\.localhost)?/[^/]+(/.*)$", path_str, re.IGNORECASE)
    if match:
        return match.group(1)
    return os.path.abspath(os.path.expanduser(path_str))


def resample_trajectory_arc_length(traj: np.ndarray, n_points: int = 32) -> np.ndarray:
    """
    Reparameterizes a trajectory curve by normalized cumulative arc-length.
    traj: shape (horizon, dim) where first 2 dimensions are (y, x).
    Returns: resampled trajectory of shape (n_points, dim).
    """
    pos = traj[:, :2]
    diffs = np.diff(pos, axis=0)
    step_lens = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate(([0.0], np.cumsum(step_lens)))
    total_len = cum_dist[-1]

    if total_len < 1e-6:
        return np.repeat(traj[:1], n_points, axis=0)

    normalized_s = cum_dist / total_len
    target_s = np.linspace(0.0, 1.0, n_points)

    resampled = np.zeros((n_points, traj.shape[1]), dtype=traj.dtype)
    for d in range(traj.shape[1]):
        resampled[:, d] = np.interp(target_s, normalized_s, traj[:, d])

    return resampled


def extract_trajectory_features(
    waypoints: np.ndarray,
    rep: str = "arc_length",
    features: str = "position",
    n_resample: int = 32,
    pca_components: int = 15,
    apply_pca: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """
    Transforms raw waypoint arrays into feature vectors.
    waypoints: shape (N, horizon, dim)
    Returns: feature matrix of shape (N, n_features)
    """
    n_samples = waypoints.shape[0]

    # Select coordinates
    if features == "position":
        data = waypoints[:, :, :2]  # (N, horizon, 2)
    else:
        data = waypoints  # (N, horizon, dim)

    # Representation transformation
    if rep == "arc_length":
        resampled = np.array([
            resample_trajectory_arc_length(data[i], n_points=n_resample)
            for i in range(n_samples)
        ])
        feat = resampled.reshape(n_samples, -1)
    elif rep == "pca":
        feat = data.reshape(n_samples, -1)
    else:
        feat = data.reshape(n_samples, -1)

    # Pre-reduction PCA if applicable
    if apply_pca and rep in ["arc_length", "pca"] and feat.shape[1] > pca_components:
        n_comp = min(pca_components, n_samples - 1, feat.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=random_state)
            feat = pca.fit_transform(feat)

    return feat


def load_waypoints_from_dir(waypoints_dir: str):
    """
    Loads planned waypoints from a waypoints directory.
    Supports individual {id}_waypoints.npy and all_planned_waypoints.npy.
    """
    if not os.path.isdir(waypoints_dir):
        return None, None

    # Search for numbered individual run files: {id}_waypoints.npy
    files = glob.glob(os.path.join(waypoints_dir, "*_waypoints.npy"))
    numbered_pairs = []
    for f in files:
        basename = os.path.basename(f)
        m = re.match(r"^(\d+)_waypoints\.npy$", basename)
        if m:
            numbered_pairs.append((int(m.group(1)), f))

    if numbered_pairs:
        numbered_pairs.sort(key=lambda x: x[0])
        run_ids = [p[0] for p in numbered_pairs]
        trajs = [np.load(p[1]) for p in numbered_pairs]
        try:
            return np.array(trajs), run_ids
        except Exception:
            pass

    # Aggregated file
    all_file = os.path.join(waypoints_dir, "all_planned_waypoints.npy")
    if os.path.isfile(all_file):
        data = np.load(all_file)
        if data.ndim == 3:
            return data, list(range(len(data)))

    return None, None


def classify_samples_by_anchors(
    waypoints: np.ndarray,
    run_ids: list,
    anchor_ids: list,
    metric: str = "arc_length",
    n_resample: int = 32,
):
    """
    Classifies all trajectories by Euclidean distance to the nearest anchor trajectory.
    Returns:
      assignments: np.ndarray of shape (N,) with cluster indices (0, 1, 2, ...)
      distances: np.ndarray of shape (N, K) containing distance to each anchor
      anchor_indices: list of indices corresponding to the anchor_ids in run_ids
    """
    n_samples = len(waypoints)
    run_id_to_idx = {rid: idx for idx, rid in enumerate(run_ids)}

    # Validate anchor IDs
    anchor_indices = []
    for aid in anchor_ids:
        if aid not in run_id_to_idx:
            raise ValueError(f"Anchor sample ID {aid} not found in loaded samples! Available range: {min(run_ids)}..{max(run_ids)}")
        anchor_indices.append(run_id_to_idx[aid])

    if metric == "arc_length":
        # Extract arc-length resampled 2D coordinates (shape: N, n_resample * 2)
        feats = np.array([
            resample_trajectory_arc_length(waypoints[i, :, :2], n_points=n_resample).reshape(-1)
            for i in range(n_samples)
        ])
    else:
        # Raw point-to-point coordinates (shape: N, horizon * 2)
        feats = waypoints[:, :, :2].reshape(n_samples, -1)

    # Extract anchor vectors: shape (K, D)
    anchor_vectors = feats[anchor_indices]

    # Compute Euclidean distance to each anchor: shape (N, K)
    # dists[i, k] = ||feats[i] - anchor_vectors[k]||
    diffs = feats[:, None, :] - anchor_vectors[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)

    # Nearest anchor index for each sample
    assignments = np.argmin(distances, axis=1)

    return assignments, distances, anchor_indices


def plot_classified_tsne(
    exp_name: str,
    waypoints: np.ndarray,
    run_ids: list,
    anchor_ids: list,
    assignments: np.ndarray,
    distances: np.ndarray,
    save_path: str,
    rep: str = "arc_length",
    features: str = "position",
    user_perplexity: int = None,
    random_state: int = 42,
):
    """
    Generates a 2D t-SNE plot where each point is colored by its assigned group
    and the anchor points are highlighted.
    """
    n_samples = waypoints.shape[0]
    n_anchors = len(anchor_ids)

    # 1. Feature extraction & t-SNE embedding
    feat = extract_trajectory_features(
        waypoints,
        rep=rep,
        features=features,
        n_resample=32,
        pca_components=15,
        apply_pca=(n_samples > 3),
        random_state=random_state,
    )

    if user_perplexity is not None:
        perp = max(1, min(user_perplexity, n_samples - 1))
    else:
        perp = max(1, min(30, (n_samples - 1) // 2))

    print(f"Computing t-SNE embedding (N={n_samples}, perp={perp}, random_state={random_state})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca" if n_samples > 10 else "random",
        learning_rate="auto",
    )
    embeddings = tsne.fit_transform(feat)

    # 2. Setup figure
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    # Adaptive sizes
    if n_samples <= 50:
        marker_size = 280
        fontsize = 7.5
    elif n_samples <= 200:
        marker_size = 220
        fontsize = 6.5
    else:
        marker_size = 170
        fontsize = 5.5

    anchor_id_set = set(anchor_ids)

    # 3. Plot points per group
    for k in range(n_anchors):
        palette = GROUP_PALETTES[k % len(GROUP_PALETTES)]
        aid = anchor_ids[k]

        # Samples belonging to group k (excluding the anchor itself)
        group_mask = (assignments == k)
        group_indices = [
            i for i in range(n_samples)
            if group_mask[i] and run_ids[i] != aid
        ]

        count_total = int(np.sum(group_mask))
        pct = (count_total / n_samples) * 100.0

        if group_indices:
            ax.scatter(
                embeddings[group_indices, 0],
                embeddings[group_indices, 1],
                s=marker_size,
                facecolor=palette["fill"],
                edgecolor=palette["edge"],
                linewidths=1.2,
                alpha=0.92,
                zorder=3,
                label=f"Group {k} (Anchor {aid}): {count_total} samples ({pct:.1f}%)",
            )

            # Draw sample IDs centered on points
            for idx in group_indices:
                rid = run_ids[idx]
                ax.text(
                    embeddings[idx, 0],
                    embeddings[idx, 1],
                    str(rid),
                    fontsize=fontsize,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color=palette["text"],
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="#ffffff")],
                )

    # 4. Highlight the anchor points prominently
    run_id_to_idx = {rid: idx for idx, rid in enumerate(run_ids)}
    for k, aid in enumerate(anchor_ids):
        palette = GROUP_PALETTES[k % len(GROUP_PALETTES)]
        idx = run_id_to_idx[aid]
        x, y = embeddings[idx, 0], embeddings[idx, 1]

        # Draw glowing outer ring
        ax.scatter(
            [x], [y],
            s=marker_size * 3.0,
            facecolor="none",
            edgecolor="#ffd700",
            linewidths=3.0,
            zorder=6,
        )

        # Draw anchor badge
        ax.scatter(
            [x], [y],
            s=marker_size * 2.2,
            facecolor=palette["primary"],
            edgecolor="#222222",
            linewidths=2.0,
            marker="*",
            zorder=7,
            label=f"★ Anchor {aid} (Group {k})",
        )

        # Draw anchor label
        ax.text(
            x, y,
            f"Anchor {aid}",
            fontsize=9.0,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="#000000",
            zorder=8,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#ffffff")],
        )

    # Limits & padding
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    dx = (x_max - x_min) * 0.08 if x_max > x_min else 1.0
    dy = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)

    clean_name = exp_name.replace("H384_1e6steps_", "").replace("_10steps", "")
    ax.set_title(
        f"t-SNE Clustered by {n_anchors} Nearest Anchor Trajectories\n"
        f"{clean_name} (N={n_samples}, Anchors={anchor_ids}, rep={rep})",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10, labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.tick_params(colors="#444444", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.8)

    ax.legend(
        title="Nearest Anchor Classification",
        title_fontsize=10,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        fontsize=8.5,
        loc="best",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved classified t-SNE plot to: {save_path}")


def plot_classified_trajectories(
    waypoints: np.ndarray,
    run_ids: list,
    anchor_ids: list,
    assignments: np.ndarray,
    save_path: str,
):
    """
    Plots the spatial trajectories (y, x) colored by their assigned group,
    with anchor trajectories plotted prominently.
    """
    n_anchors = len(anchor_ids)
    run_id_to_idx = {rid: idx for idx, rid in enumerate(run_ids)}

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    # Plot trajectories for each group
    for k in range(n_anchors):
        palette = GROUP_PALETTES[k % len(GROUP_PALETTES)]
        aid = anchor_ids[k]
        group_mask = (assignments == k)
        group_indices = [i for i in range(len(waypoints)) if group_mask[i] and run_ids[i] != aid]

        # Plot sample trajectories faintly
        for idx in group_indices:
            traj = waypoints[idx]
            ax.plot(traj[:, 1], traj[:, 0], color=palette["primary"], alpha=0.15, linewidth=0.8, zorder=2)

        # Plot anchor trajectory with thick line and glowing border
        a_idx = run_id_to_idx[aid]
        a_traj = waypoints[a_idx]
        ax.plot(a_traj[:, 1], a_traj[:, 0], color="#000000", linewidth=3.2, zorder=5)
        ax.plot(
            a_traj[:, 1], a_traj[:, 0],
            color=palette["primary"],
            linewidth=2.0,
            linestyle="-",
            zorder=6,
            label=f"Group {k} Anchor {aid} ({np.sum(group_mask)} trajectories)",
        )
        # Mark anchor start & end
        ax.scatter([a_traj[0, 1]], [a_traj[0, 0]], color="#2ecc71", s=80, marker="o", edgecolors="black", zorder=7)
        ax.scatter([a_traj[-1, 1]], [a_traj[-1, 0]], color="#e74c3c", s=90, marker="X", edgecolors="black", zorder=7)

    ax.set_title(f"Planned Trajectories Grouped by Nearest Anchor (Anchors={anchor_ids})", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("X Position", fontsize=10)
    ax.set_ylabel("Y Position", fontsize=10)
    ax.set_ylim(8.0, 0.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved classified trajectory plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify trajectory waypoints into 3 groups based on nearest anchor samples and plot colored t-SNE."
    )
    parser.add_argument(
        "waypoints_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to waypoints directory containing .npy files.",
    )
    parser.add_argument(
        "anchors",
        type=int,
        nargs="*",
        default=None,
        help="Anchor sample numbers (e.g. 0 100 500).",
    )
    parser.add_argument(
        "--waypoints_dir",
        "--waypoints",
        dest="waypoints_dir_flag",
        type=str,
        default=None,
        help="Path to waypoints directory containing .npy files.",
    )
    parser.add_argument(
        "--anchors",
        dest="anchors_flag",
        type=int,
        nargs="+",
        default=None,
        help="Anchor sample numbers (e.g. 0 100 500).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["arc_length", "raw"],
        default="arc_length",
        help="Distance metric for nearest anchor classification: 'arc_length' (spatial curve distance, recommended) or 'raw' (Euclidean distance).",
    )
    parser.add_argument(
        "--rep",
        type=str,
        choices=["arc_length", "pca", "raw"],
        default="arc_length",
        help="Trajectory representation for t-SNE embedding: 'arc_length', 'pca', or 'raw'. Default: arc_length.",
    )
    parser.add_argument(
        "--features",
        type=str,
        choices=["position", "full"],
        default="position",
        help="Features for t-SNE embedding: 'position' or 'full'. Default: position.",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=None,
        help="Perplexity for t-SNE. Default: automatic.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Custom path to save the t-SNE plot. Defaults to <waypoints_dir>/tsne_clusters.png.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Custom path to save classification CSV. Defaults to <waypoints_dir>/classification_results.csv.",
    )
    parser.add_argument(
        "--plot_trajectories",
        action="store_true",
        default=True,
        help="Also generate a 2D spatial plot of trajectories colored by group. Enabled by default.",
    )

    args = parser.parse_args()

    # 1. Resolve arguments
    raw_dir = args.waypoints_dir_flag or args.waypoints_dir
    anchors = args.anchors_flag if args.anchors_flag is not None else args.anchors

    if not raw_dir:
        # Default fallback if present in workspace
        candidate = "logs/maze2d-large-v1/plans/release_H384_T256_LimitsNormalizer_b1000_condFalse/waypoints"
        if os.path.isdir(candidate):
            raw_dir = candidate
        else:
            parser.error("Please specify waypoints folder path (e.g. python scripts/classify_tsne.py <path> <anchor1> <anchor2> <anchor3>)")

    if not anchors or len(anchors) < 2:
        parser.error(f"Please provide at least 2 anchor sample numbers (e.g. 0 100 500). Got: {anchors}")

    waypoints_dir = normalize_path(raw_dir)

    # Check for waypoints subfolder
    if os.path.isdir(os.path.join(waypoints_dir, "waypoints")):
        sub_wp, _ = load_waypoints_from_dir(os.path.join(waypoints_dir, "waypoints"))
        if sub_wp is not None:
            waypoints_dir = os.path.join(waypoints_dir, "waypoints")

    print("=" * 70)
    print("Trajectory Classification & t-SNE Visualization")
    print("=" * 70)
    print(f"Waypoints Directory: {waypoints_dir}")
    print(f"Anchor Samples:      {anchors}")
    print(f"Distance Metric:     {args.metric}")
    print(f"t-SNE Representation:{args.rep}")
    print("=" * 70)

    # 2. Load waypoints
    waypoints, run_ids = load_waypoints_from_dir(waypoints_dir)
    if waypoints is None or len(waypoints) == 0:
        print(f"Error: No waypoints (.npy or .npz) found in {waypoints_dir}")
        sys.exit(1)

    n_samples = len(waypoints)
    print(f"\n[+] Successfully loaded {n_samples} trajectories with horizon {waypoints.shape[1]}.")

    # 3. Classify samples by nearest anchor
    assignments, distances, anchor_indices = classify_samples_by_anchors(
        waypoints,
        run_ids,
        anchor_ids=anchors,
        metric=args.metric,
    )

    # 4. Print Summary
    print("\nClassification Results:")
    print("-" * 70)
    for k, aid in enumerate(anchors):
        count = int(np.sum(assignments == k))
        pct = (count / n_samples) * 100.0
        mean_d = float(np.mean(distances[assignments == k, k])) if count > 0 else 0.0
        print(f"  Group {k} (Anchor {aid:4d}): {count:4d} samples ({pct:5.1f}%) | Mean Distance: {mean_d:.4f}")
    print("-" * 70)

    # 5. Save CSV Results
    csv_path = args.csv_path or os.path.join(waypoints_dir, "classification_results.csv")
    csv_rows = []
    for i, rid in enumerate(run_ids):
        assigned_k = int(assignments[i])
        assigned_anchor = anchors[assigned_k]
        min_dist = float(distances[i, assigned_k])
        row_dict = {
            "sample_id": rid,
            "group_id": assigned_k,
            "anchor_id": assigned_anchor,
            "distance_to_anchor": min_dist,
        }
        for k, aid in enumerate(anchors):
            row_dict[f"dist_to_anchor_{aid}"] = float(distances[i, k])
        csv_rows.append(row_dict)

    if pd is not None:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    else:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"\n[+] Saved detailed classification table to:\n    {csv_path}")

    # 6. Generate Classified t-SNE Plot
    parent_dir = os.path.dirname(waypoints_dir)
    exp_name = os.path.basename(parent_dir) if os.path.basename(waypoints_dir) == "waypoints" else os.path.basename(waypoints_dir)
    tsne_save_path = args.save_path or os.path.join(waypoints_dir, "tsne_clusters.png")

    plot_classified_tsne(
        exp_name=exp_name,
        waypoints=waypoints,
        run_ids=run_ids,
        anchor_ids=anchors,
        assignments=assignments,
        distances=distances,
        save_path=tsne_save_path,
        rep=args.rep,
        features=args.features,
        user_perplexity=args.perplexity,
        random_state=args.random_state,
    )

    # 7. Optional: Plot Trajectories by Group
    if args.plot_trajectories:
        traj_save_path = os.path.join(waypoints_dir, "trajectories_clustered.png")
        plot_classified_trajectories(
            waypoints=waypoints,
            run_ids=run_ids,
            anchor_ids=anchors,
            assignments=assignments,
            save_path=traj_save_path,
        )

    print("\nProcessing complete! All artifacts generated successfully.")


if __name__ == "__main__":
    main()
