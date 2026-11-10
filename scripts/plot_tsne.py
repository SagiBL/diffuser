#!/usr/bin/env python3
"""
t-SNE Visualization Script for Planned Trajectory Waypoints.

Accepts a waypoints folder path and loads the planned waypoint vectors (.npy):
  Generates an individual t-SNE plot saved inside the waypoints folder:
     <waypoints_folder>/tsne_waypoints.png

Uses spatial arc-length curve parameterization and PCA pre-reduction to eliminate
timing/velocity noise and produce structured, clustered manifold embeddings.
"""

import os
import sys
import glob
import re
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


def normalize_path(path_str: str) -> str:
    """Normalize Windows UNC / WSL paths to standard POSIX path."""
    if not path_str:
        return path_str
    # Convert backslashes
    path_str = path_str.replace("\\", "/")
    # Handle \\wsl.localhost\Ubuntu\... or \\wsl$\Ubuntu\...
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
        # Stationary or degenerate trajectory
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
    Transforms raw waypoint arrays into feature vectors optimized for t-SNE embedding.
    waypoints: shape (N, horizon, dim)
    Returns: feature matrix of shape (N, n_features)
    """
    n_samples = waypoints.shape[0]

    # 1. Select coordinates (position vs full)
    if features == "position":
        data = waypoints[:, :, :2]  # (N, horizon, 2)
    else:
        data = waypoints  # (N, horizon, dim)

    # 2. Representation transformation
    if rep == "arc_length":
        # Reparameterize each curve by cumulative distance to eliminate speed/timing phase shifts
        resampled = np.array([
            resample_trajectory_arc_length(data[i], n_points=n_resample)
            for i in range(n_samples)
        ])
        feat = resampled.reshape(n_samples, -1)
    elif rep == "pca":
        feat = data.reshape(n_samples, -1)
    else:
        # raw
        feat = data.reshape(n_samples, -1)

    # 3. Apply PCA pre-reduction (eliminates high-dimensional distance concentration)
    if apply_pca and rep in ["arc_length", "pca"] and feat.shape[1] > pca_components:
        n_comp = min(pca_components, n_samples - 1, feat.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=random_state)
            feat = pca.fit_transform(feat)

    return feat


def load_waypoints_from_dir(waypoints_dir: str):
    """
    Loads planned waypoints from a waypoints directory.
    Supports:
      - all_planned_waypoints.npy
      - individual {run_id}_waypoints.npy files
    Returns:
      (waypoints_array, run_ids) where waypoints_array has shape (N, horizon, dim)
    """
    if not os.path.isdir(waypoints_dir):
        return None, None

    # Search for numbered individual run files first: {id}_waypoints.npy
    pattern = os.path.join(waypoints_dir, "*_waypoints.npy")
    files = glob.glob(pattern)
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
            arr = np.array(trajs)
            return arr, run_ids
        except Exception:
            pass

    # Check for aggregated array
    all_file = os.path.join(waypoints_dir, "all_planned_waypoints.npy")
    if os.path.isfile(all_file):
        data = np.load(all_file)
        if data.ndim == 3:
            run_ids = list(range(len(data)))
            return data, run_ids

    # Also check for .npz files with planned_waypoints
    npz_files = glob.glob(os.path.join(waypoints_dir, "*.npz"))
    if npz_files:
        run_id_map = {}
        for f in npz_files:
            basename = os.path.basename(f)
            m = re.match(r"^(\d+)", basename)
            rid = int(m.group(1)) if m else len(run_id_map)
            try:
                with np.load(f) as d:
                    if "planned_waypoints" in d:
                        run_id_map[rid] = d["planned_waypoints"]
                    elif "planned" in d:
                        run_id_map[rid] = d["planned"]
            except Exception:
                pass
        if run_id_map:
            sorted_rids = sorted(run_id_map.keys())
            arr = np.array([run_id_map[r] for r in sorted_rids])
            return arr, sorted_rids

    return None, None


def load_metadata_csv(exp_dir: str):
    """Loads trajectories.csv if present to extract safety and score per run_id."""
    csv_path = os.path.join(exp_dir, "trajectories.csv")
    if not os.path.isfile(csv_path):
        return {}

    meta = {}
    try:
        if pd is not None:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rid = int(row["run_id"])
                meta[rid] = {
                    "safety": bool(row.get("safety", True)),
                    "score": float(row.get("score", 0.0)),
                    "total_violations": float(row.get("total_violations", 0.0)),
                }
        else:
            import csv
            with open(csv_path, mode="r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = int(row["run_id"])
                    safety_str = str(row.get("safety", "True")).strip().lower()
                    safety = safety_str in ["true", "1", "yes"]
                    meta[rid] = {
                        "safety": safety,
                        "score": float(row.get("score", 0.0)),
                        "total_violations": float(row.get("total_violations", 0.0)),
                    }
    except Exception:
        pass
    return meta


def plot_single_experiment_tsne(
    exp_name: str,
    waypoints: np.ndarray,
    run_ids: list,
    save_path: str,
    metadata: dict = None,
    rep: str = "arc_length",
    features: str = "position",
    highlight_ids: set = None,
    user_perplexity: int = None,
    random_state: int = 42,
):
    """
    Runs t-SNE and saves a scatter plot ONLY inside the waypoints subfolder.
    """
    n_samples = waypoints.shape[0]
    if n_samples == 0:
        return

    # Extract improved trajectory features
    feat = extract_trajectory_features(
        waypoints,
        rep=rep,
        features=features,
        n_resample=32,
        pca_components=15,
        apply_pca=(n_samples > 3),
        random_state=random_state,
    )

    # Adaptive perplexity and embedding
    if n_samples >= 2:
        if user_perplexity is not None:
            perp = max(1, min(user_perplexity, n_samples - 1))
        else:
            perp = max(1, min(30, (n_samples - 1) // 2))

        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=random_state,
            init="pca" if n_samples > 10 else "random",
            learning_rate="auto",
        )
        embeddings = tsne.fit_transform(feat)
    else:
        perp = 1
        embeddings = np.array([[0.0, 0.0]])

    # Setup plot with high resolution and clear margins
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    # Adaptive marker size and font size based on sample count
    if n_samples <= 30:
        marker_size = 280
        fontsize = 8
    elif n_samples <= 100:
        marker_size = 210
        fontsize = 7
    else:
        marker_size = 160
        fontsize = 6

    if highlight_ids:
        # Separate normal and highlighted indices
        normal_idx = [i for i, r in enumerate(run_ids) if r not in highlight_ids]
        highlight_idx = [i for i, r in enumerate(run_ids) if r in highlight_ids]

        # 1. Normal points (neutral soft grey/blue)
        if normal_idx:
            ax.scatter(
                embeddings[normal_idx, 0],
                embeddings[normal_idx, 1],
                s=140,
                facecolor="#f0f4f8",
                edgecolor="#b0c4de",
                linewidths=0.9,
                alpha=0.7,
                zorder=2,
                label=f"Other Samples ({len(normal_idx)})",
            )
            for i in normal_idx:
                ax.text(
                    embeddings[i, 0],
                    embeddings[i, 1],
                    str(run_ids[i]),
                    fontsize=6,
                    fontweight="normal",
                    ha="center",
                    va="center",
                    color="#708090",
                    zorder=3,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="#ffffff")],
                )

        # 2. Highlighted points in bold red
        if highlight_idx:
            ax.scatter(
                embeddings[highlight_idx, 0],
                embeddings[highlight_idx, 1],
                s=260,
                facecolor="#e63946",
                edgecolor="#8b0000",
                linewidths=1.8,
                alpha=0.95,
                zorder=5,
                label=f"Selected Samples ({len(highlight_idx)})",
            )
            for i in highlight_idx:
                ax.text(
                    embeddings[i, 0],
                    embeddings[i, 1],
                    str(run_ids[i]),
                    fontsize=8.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="#ffffff",
                    zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="#8b0000")],
                )
        ax.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=10, loc="best")
    else:
        # 1. Base badge circle for every point
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            s=marker_size,
            facecolor="#edf4fc",
            edgecolor="#1f77b4",
            linewidths=1.2,
            alpha=0.92,
            zorder=3,
        )

        # 2. Write the sample number centered on each point
        for i, rid in enumerate(run_ids):
            ax.text(
                embeddings[i, 0],
                embeddings[i, 1],
                str(rid),
                fontsize=fontsize,
                fontweight="bold",
                ha="center",
                va="center",
                color="#0f2942",
                zorder=4,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="#ffffff")],
            )

    # Expand limits slightly so outer numbers and badges aren't clipped
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    dx = (x_max - x_min) * 0.08 if x_max > x_min else 1.0
    dy = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)

    # Title & styling
    clean_name = exp_name.replace("H384_1e6steps_", "").replace("_10steps", "")
    ax.set_title(
        f"t-SNE of Planned Trajectory Waypoints\n{clean_name} (N={n_samples}, rep={rep}, perp={perp})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10, labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.tick_params(colors="#444444", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved t-SNE plot to waypoints subfolder: {save_path}")


def plot_combined_tsne(
    exp_data: dict,
    save_path: str,
    rep: str = "arc_length",
    features: str = "position",
    user_perplexity: int = None,
    random_state: int = 42,
):
    """
    Embeds trajectories from ALL experiment folders into a single joint t-SNE space.
    """
    all_feats = []
    exp_names = sorted(exp_data.keys())

    for exp_name in exp_names:
        w = exp_data[exp_name]
        feat = extract_trajectory_features(
            w,
            rep=rep,
            features=features,
            n_resample=32,
            apply_pca=False,
            random_state=random_state,
        )
        all_feats.append(feat)

    if not all_feats:
        return

    combined_feat = np.vstack(all_feats)
    total_samples = combined_feat.shape[0]

    if total_samples < 4:
        print(f"Total samples across all experiments is too small ({total_samples}). Skipping combined plot.")
        return

    # Apply joint PCA across all methods so they share the exact same principal subspace
    if rep in ["arc_length", "pca"] and combined_feat.shape[1] > 15:
        n_comp = min(15, total_samples - 1, combined_feat.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=random_state)
            combined_feat = pca.fit_transform(combined_feat)

    if user_perplexity is not None:
        perp = min(user_perplexity, total_samples - 1)
    else:
        perp = min(30, max(2, (total_samples - 1) // 3))

    print(f"\nComputing combined t-SNE for {total_samples} trajectories across {len(exp_names)} methods (rep={rep}, perp={perp})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca" if total_samples > 10 else "random",
        learning_rate="auto",
    )
    embeddings = tsne.fit_transform(combined_feat)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    fig.patch.set_facecolor("#fcfcfc")
    ax.set_facecolor("#ffffff")

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    start_idx = 0
    for i, exp_name in enumerate(exp_names):
        n = exp_data[exp_name].shape[0]
        sub_emb = embeddings[start_idx : start_idx + n]
        start_idx += n

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        clean_label = exp_name.replace("H384_1e6steps_", "").replace("_10steps", "")

        ax.scatter(
            sub_emb[:, 0],
            sub_emb[:, 1],
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.6,
            s=60,
            alpha=0.85,
            label=f"{clean_label} (N={n})",
            zorder=3,
        )

    ax.set_title(
        f"Joint t-SNE Embedding of Planned Waypoints Across Methods\nTotal Samples: {total_samples} (rep={rep}, perplexity={perp})",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.tick_params(colors="#444444", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.8)

    ax.legend(
        title="Guidance Method",
        title_fontsize=10,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        fontsize=9,
        loc="best",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSuccessfully generated combined t-SNE plot:\n  -> {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate t-SNE plot for planned trajectory waypoints from a waypoints directory."
    )
    parser.add_argument(
        "waypoints_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to the waypoints directory containing .npy files.",
    )
    parser.add_argument(
        "--waypoints_dir",
        "--waypoints_path",
        "--waypoints",
        dest="waypoints_dir_flag",
        type=str,
        default=None,
        help="Path to the waypoints directory containing .npy files.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional custom output path for the t-SNE plot. Defaults to <waypoints_dir>/tsne_waypoints.png.",
    )
    parser.add_argument(
        "--rep",
        type=str,
        choices=["arc_length", "pca", "raw"],
        default="arc_length",
        help="Trajectory representation: 'arc_length' (spatial curve parameterization + PCA, recommended), 'pca' (direct PCA), or 'raw'. Default: arc_length.",
    )
    parser.add_argument(
        "--features",
        type=str,
        choices=["position", "full"],
        default="position",
        help="Features to embed: 'position' (y, x coordinates) or 'full' (y, x, vy, vx). Default: position.",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=None,
        help="Perplexity for t-SNE. Default: automatically calculated from sample count.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--highlight",
        type=str,
        default=None,
        help="Comma-separated run IDs to highlight in red (e.g. '58,75,78,92,98,112,130,159,160,199').",
    )

    args = parser.parse_args()

    raw_path = args.waypoints_dir_flag or args.waypoints_dir
    if not raw_path:
        parser.error("Please specify the waypoints folder path as an argument (e.g. python scripts/plot_tsne.py <path_to_waypoints>)")

    waypoints_dir = normalize_path(raw_path)

    if not os.path.exists(waypoints_dir):
        print(f"Error: directory does not exist: {waypoints_dir}")
        sys.exit(1)

    if not os.path.isdir(waypoints_dir):
        print(f"Error: not a directory: {waypoints_dir}")
        sys.exit(1)

    # If the user passed the experiment folder containing a 'waypoints' subfolder:
    if os.path.isdir(os.path.join(waypoints_dir, "waypoints")):
        sub_wp, _ = load_waypoints_from_dir(os.path.join(waypoints_dir, "waypoints"))
        if sub_wp is not None:
            waypoints_dir = os.path.join(waypoints_dir, "waypoints")

    print(f"Loading waypoints from:\n  {waypoints_dir}\n")
    print(f"Configuration: rep={args.rep}, features={args.features}")

    waypoints, run_ids = load_waypoints_from_dir(waypoints_dir)

    if waypoints is None or len(waypoints) == 0:
        print(f"Error: No waypoints (.npy or .npz) found in {waypoints_dir}.")
        sys.exit(1)

    print(f"[+] Loaded {len(waypoints)} trajectories with shape {waypoints.shape[1:]}.")

    # Infer experiment name from directory structure
    parent_dir = os.path.dirname(waypoints_dir)
    if os.path.basename(waypoints_dir) == "waypoints":
        exp_name = os.path.basename(parent_dir)
        exp_path = parent_dir
    else:
        exp_name = os.path.basename(waypoints_dir)
        exp_path = waypoints_dir

    # Load metadata if available
    meta = load_metadata_csv(exp_path)

    save_path = args.save_path if args.save_path else os.path.join(waypoints_dir, "tsne_waypoints.png")

    plot_single_experiment_tsne(
        exp_name=exp_name,
        waypoints=waypoints,
        run_ids=run_ids,
        save_path=save_path,
        metadata=meta,
        rep=args.rep,
        features=args.features,
        highlight_ids=None,
        user_perplexity=args.perplexity,
        random_state=args.random_state,
    )

    # If --highlight was specified, also generate the highlighted plot
    highlight_set = None
    if args.highlight:
        try:
            highlight_set = {int(x.strip()) for x in args.highlight.split(",") if x.strip()}
        except ValueError:
            print(f"Warning: could not parse --highlight '{args.highlight}' as integers.")

    if highlight_set:
        save_path_highlighted = os.path.splitext(save_path)[0] + "_highlighted.png"
        plot_single_experiment_tsne(
            exp_name=exp_name,
            waypoints=waypoints,
            run_ids=run_ids,
            save_path=save_path_highlighted,
            metadata=meta,
            rep=args.rep,
            features=args.features,
            highlight_ids=highlight_set,
            user_perplexity=args.perplexity,
            random_state=args.random_state,
        )

    print("\nt-SNE processing complete!")



if __name__ == "__main__":
    main()
