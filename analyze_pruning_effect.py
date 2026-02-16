#!/usr/bin/env python3
"""
Analyze pruning effects across hybrid-pruned checkpoints.

This script computes and plots:
- Frequency score (low-frequency energy ratio per filter)
- Gradient saliency score (Taylor criterion per filter)
- Filter-pruning statistics per iteration and per layer
"""

import argparse
import gc
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from config import Config
from data.data_loader import DataLoader
from models.model_utils import ModelUtils
from pruning.frequency_rel_analyzer import FrequencyRelevanceAnalyzer


@dataclass
class CheckpointInfo:
    iteration: int
    label: str
    path: Path


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def resolve_models_dir(pruned_root: Path) -> Path:
    if not pruned_root.exists():
        raise FileNotFoundError(f"Pruned root does not exist: {pruned_root}")
    if pruned_root.is_file():
        raise ValueError(
            f"`pruned_root` must be a directory, got file: {pruned_root}"
        )

    direct_h5 = sorted(pruned_root.glob("*.h5"))
    if direct_h5:
        return pruned_root

    models_dir = pruned_root / "models"
    if models_dir.is_dir():
        model_h5 = sorted(models_dir.glob("*.h5"))
        if model_h5:
            return models_dir

    raise FileNotFoundError(
        f"Could not find .h5 files under {pruned_root} or {models_dir}"
    )


def collect_pruned_checkpoints(
    models_dir: Path,
    pattern: str = "*_hybrid_iter_*_best_model.h5",
) -> List[CheckpointInfo]:
    iter_re = re.compile(r"_iter_(\d+)_")
    checkpoints: List[CheckpointInfo] = []

    for path in sorted(models_dir.glob(pattern)):
        match = iter_re.search(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        checkpoints.append(
            CheckpointInfo(
                iteration=iteration,
                label=f"iter_{iteration}",
                path=path,
            )
        )

    checkpoints.sort(key=lambda item: item.iteration)
    return checkpoints


def load_dataset_for_analysis(config: Config, split: str) -> tf.data.Dataset:
    loader = DataLoader(config)
    if config.task == "cifar10":
        train_ds, val_ds, test_ds, train_eval_ds = loader.load_cifar10()
    elif config.task == "cifar100":
        train_ds, val_ds, test_ds, train_eval_ds = loader.load_cifar100()
    elif config.task == "tiny_imagenet":
        train_ds, val_ds, test_ds, train_eval_ds = loader.load_tiny_imagenet()
    else:
        raise ValueError(f"Unsupported task: {config.task}")

    split_map = {
        "train_eval": train_eval_ds,
        "val": val_ds,
        "test": test_ds,
        "train": train_ds,
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split: {split}")
    return split_map[split]


def compute_frequency_scores(
    model: tf.keras.Model,
    frn_analyzer: FrequencyRelevanceAnalyzer,
    kappa_ratio: float,
) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    kappa_ratio = float(np.clip(kappa_ratio, 0.0, 1.0))

    for layer in ModelUtils.get_conv_layers(model):
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        dct_kernel = frn_analyzer.dct2_ortho(
            tf.convert_to_tensor(kernel, dtype=tf.float32)
        ).numpy()
        h, w, _cin, _cout = dct_kernel.shape
        kappa = max(1, int(round(min(h, w) * kappa_ratio)))

        mask = np.zeros((h, w), dtype=np.float32)
        mask[:kappa, :kappa] = 1.0

        total = np.sum(np.square(dct_kernel), axis=(0, 1, 2)) + 1e-8
        low = np.sum(
            np.square(dct_kernel) * mask[:, :, None, None],
            axis=(0, 1, 2),
        )
        scores[layer.name] = (low / total).astype(np.float32)
    return scores


def is_sparse_target(y_true: tf.Tensor) -> bool:
    if y_true.dtype.is_integer:
        return True
    rank = y_true.shape.rank
    if rank == 1:
        return True
    if rank == 2 and y_true.shape[-1] == 1:
        return True
    return False


def infer_from_logits(model: tf.keras.Model) -> bool:
    last = model.layers[-1]
    if isinstance(last, tf.keras.layers.Softmax):
        return False
    activation = getattr(last, "activation", None)
    return activation != tf.keras.activations.softmax


def compute_accuracy(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> float:
    total = 0
    correct = 0

    for batch in dataset:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            continue
        xb, yb = batch[0], batch[1]
        preds = model(xb, training=False)
        pred_cls = tf.argmax(preds, axis=-1, output_type=tf.int32)

        if is_sparse_target(yb):
            true_cls = tf.reshape(tf.cast(yb, tf.int32), [-1])
        else:
            true_cls = tf.argmax(tf.cast(yb, tf.float32), axis=-1, output_type=tf.int32)

        matches = tf.equal(pred_cls, true_cls)
        correct += int(tf.reduce_sum(tf.cast(matches, tf.int32)).numpy())
        total += int(tf.size(true_cls).numpy())

    if total <= 0:
        return float("nan")
    return float(correct / total)


def compute_gradient_saliency(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: int = 8,
) -> Dict[str, np.ndarray]:
    conv_layers = ModelUtils.get_conv_layers(model)
    if not conv_layers:
        return {}

    kernels = [layer.kernel for layer in conv_layers]
    saliency = {
        layer.name: np.zeros((layer.filters,), dtype=np.float32)
        for layer in conv_layers
    }
    used_batches = 0
    from_logits = infer_from_logits(model)

    batches = dataset.take(max_batches) if max_batches > 0 else dataset
    for batch in batches:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            continue
        xb, yb = batch[0], batch[1]
        xb = tf.cast(xb, tf.float32)

        with tf.GradientTape() as tape:
            preds = model(xb, training=True)
            if is_sparse_target(yb):
                y_sparse = tf.reshape(tf.cast(yb, tf.int32), [-1])
                loss_vec = tf.keras.losses.sparse_categorical_crossentropy(
                    y_sparse, preds, from_logits=from_logits
                )
            else:
                y_dense = tf.cast(yb, tf.float32)
                loss_vec = tf.keras.losses.categorical_crossentropy(
                    y_dense, preds, from_logits=from_logits
                )
            loss = tf.reduce_mean(loss_vec)

        grads = tape.gradient(loss, kernels)
        for layer, grad in zip(conv_layers, grads):
            if grad is None:
                continue
            taylor = tf.abs(grad * layer.kernel)
            per_filter = tf.reduce_sum(taylor, axis=(0, 1, 2))
            saliency[layer.name] += per_filter.numpy().astype(np.float32)
        used_batches += 1

    if used_batches > 0:
        for name in saliency:
            saliency[name] /= float(used_batches)
    return saliency


def combine_scores(
    frequency_scores: Dict[str, np.ndarray],
    gradient_scores: Dict[str, np.ndarray],
    alpha: float,
) -> Dict[str, np.ndarray]:
    combined: Dict[str, np.ndarray] = {}
    eps = 1e-8
    for layer_name, grad in gradient_scores.items():
        freq = frequency_scores.get(layer_name)
        if freq is None:
            combined[layer_name] = grad.astype(np.float32)
            continue
        combined[layer_name] = (
            grad.astype(np.float32) * np.power(freq.astype(np.float32) + eps, alpha)
        )
    return combined


def prune_ratio_for_iteration(config: Config, iteration: int) -> float:
    prune_ratio = float(getattr(config, "hybrid_prune_fraction", 0.08))
    taper_start = int(getattr(config, "hybrid_taper_start", 15))
    late_ratio = float(getattr(config, "hybrid_late_prune_fraction", prune_ratio))
    if iteration >= taper_start:
        prune_ratio = min(prune_ratio, late_ratio)
    return float(np.clip(prune_ratio, 0.0, 1.0))


def select_pruning_masks(
    scores: Dict[str, np.ndarray],
    prune_ratio: float,
    min_filters: int,
) -> Dict[str, np.ndarray]:
    pools: List[Tuple[str, int, float]] = []
    for layer_name, layer_scores in scores.items():
        for idx, value in enumerate(layer_scores):
            pools.append((layer_name, idx, float(value)))

    masks: Dict[str, np.ndarray] = {
        layer_name: np.ones_like(layer_scores, dtype=bool)
        for layer_name, layer_scores in scores.items()
    }
    if not pools:
        return masks

    pools.sort(key=lambda item: item[2])
    target_prunes = int(len(pools) * float(prune_ratio))
    if target_prunes <= 0:
        return masks

    prune_indices = {(layer, idx) for layer, idx, _ in pools[:target_prunes]}

    for layer_name, layer_scores in scores.items():
        mask = masks[layer_name]
        for idx in range(len(layer_scores)):
            if (layer_name, idx) in prune_indices:
                mask[idx] = False

        keep = int(mask.sum())
        if keep < int(min_filters):
            # Keep highest-scoring channels to satisfy per-layer minimum.
            order = np.argsort(layer_scores)[::-1]
            mask[:] = False
            mask[order[: int(min_filters)]] = True
        masks[layer_name] = mask
    return masks


def build_filter_decision_rows(
    iteration: int,
    label: str,
    checkpoint: str,
    frequency_scores: Dict[str, np.ndarray],
    gradient_scores: Dict[str, np.ndarray],
    combined_scores: Dict[str, np.ndarray],
    decision_masks: Dict[str, np.ndarray],
    decision_iteration: int,
    prune_ratio: float,
) -> List[Dict]:
    rows: List[Dict] = []
    for layer_name, combined in combined_scores.items():
        freq = frequency_scores.get(layer_name)
        grad = gradient_scores.get(layer_name)
        mask = decision_masks.get(
            layer_name, np.ones((len(combined),), dtype=bool)
        )
        for idx in range(len(combined)):
            rows.append(
                {
                    "iteration": iteration,
                    "label": label,
                    "checkpoint": checkpoint,
                    "decision_iteration": int(decision_iteration),
                    "decision_prune_ratio": float(prune_ratio),
                    "layer_name": layer_name,
                    "filter_index": int(idx),
                    "frequency_score": float(freq[idx]) if freq is not None else float("nan"),
                    "gradient_score": float(grad[idx]) if grad is not None else float("nan"),
                    "combined_score": float(combined[idx]),
                    "decision": "kept" if bool(mask[idx]) else "pruned",
                }
            )
    return rows


def flatten_scores(scores: Dict[str, np.ndarray]) -> np.ndarray:
    parts = [arr for arr in scores.values() if arr is not None and arr.size > 0]
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def score_stats(arr: Optional[np.ndarray]) -> Dict[str, float]:
    if arr is None or arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def minmax_normalize(values: pd.Series) -> pd.Series:
    lo = values.min()
    hi = values.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    return (values - lo) / (hi - lo)


def save_global_plots(global_df: pd.DataFrame, output_dir: Path) -> None:
    x = global_df["iteration"].values

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(x, global_df["frequency_mean"], marker="o", color="#1f77b4")
    axes[0].set_ylabel("Frequency score (mean)")
    axes[0].set_title("Global Frequency Score Across Iterations")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    grad_safe = np.clip(global_df["gradient_mean"].values, 1e-12, None)
    axes[1].plot(x, grad_safe, marker="o", color="#d62728")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gradient saliency mean (log)")
    axes[1].set_title("Global Gradient Saliency Across Iterations")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(x, global_df["total_filters"], marker="o", color="#2ca02c")
    axes[2].set_ylabel("Total conv filters")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("Remaining Filters Across Iterations")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "global_score_trends.png", dpi=180)
    plt.close(fig)

    freq_norm = minmax_normalize(global_df["frequency_mean"])
    grad_norm = minmax_normalize(np.log10(np.clip(global_df["gradient_mean"], 1e-12, None)))
    prune_ratio_pct = global_df["cumulative_pruning_ratio"] * 100.0

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, freq_norm, marker="o", label="Frequency score (normalized)")
    ax.plot(x, grad_norm, marker="o", label="Gradient saliency (normalized)")
    ax.plot(x, minmax_normalize(prune_ratio_pct), marker="o", label="Pruning ratio (normalized)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized value [0, 1]")
    ax.set_title("Normalized Trend Comparison")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "normalized_trends.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(
        x,
        global_df["total_pruned_this_iter"],
        color="#ff7f0e",
        alpha=0.75,
        label="Pruned this iteration",
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Filters pruned this iteration")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        global_df["cumulative_pruning_ratio"] * 100.0,
        marker="o",
        color="#1f77b4",
        label="Cumulative pruning ratio",
    )
    ax2.set_ylabel("Cumulative pruning ratio (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "pruning_progress.png", dpi=180)
    plt.close(fig)


def save_layer_heatmaps(
    layer_df: pd.DataFrame,
    layer_order: List[str],
    output_dir: Path,
) -> None:
    freq_pivot = layer_df.pivot(
        index="layer_name", columns="iteration", values="frequency_mean"
    ).reindex(layer_order)
    grad_pivot = layer_df.pivot(
        index="layer_name", columns="iteration", values="gradient_mean"
    ).reindex(layer_order)
    prune_pivot = layer_df.pivot(
        index="layer_name", columns="iteration", values="pruned_this_iter"
    ).reindex(layer_order)
    remaining_pivot = layer_df.pivot(
        index="layer_name", columns="iteration", values="filter_count"
    ).reindex(layer_order)

    grad_log = np.log10(np.clip(grad_pivot, 1e-12, None))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.heatmap(freq_pivot, cmap="viridis", ax=axes[0, 0], cbar=True)
    axes[0, 0].set_title("Frequency score mean")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Layer")

    sns.heatmap(grad_log, cmap="magma", ax=axes[0, 1], cbar=True)
    axes[0, 1].set_title("Gradient saliency mean (log10)")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Layer")

    sns.heatmap(prune_pivot, cmap="Reds", ax=axes[1, 0], cbar=True)
    axes[1, 0].set_title("Filters pruned per iteration")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Layer")

    sns.heatmap(remaining_pivot, cmap="Blues", ax=axes[1, 1], cbar=True)
    axes[1, 1].set_title("Remaining filters")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Layer")

    fig.tight_layout()
    fig.savefig(output_dir / "layer_heatmaps.png", dpi=180)
    plt.close(fig)


def _plot_single_decision_scatter(
    ax: plt.Axes,
    subset: pd.DataFrame,
    title: str,
    use_log_x: bool = True,
    with_legend: bool = False,
) -> None:
    kept = subset[subset["decision"] == "kept"]
    pruned = subset[subset["decision"] == "pruned"]

    x_kept = np.clip(kept["gradient_score"].to_numpy(dtype=np.float64), 1e-12, None)
    y_kept = kept["frequency_score"].to_numpy(dtype=np.float64)
    x_pruned = np.clip(pruned["gradient_score"].to_numpy(dtype=np.float64), 1e-12, None)
    y_pruned = pruned["frequency_score"].to_numpy(dtype=np.float64)

    ax.scatter(x_kept, y_kept, s=8, c="#2ca02c", alpha=0.65, label="Kept", linewidths=0.0)
    ax.scatter(
        x_pruned, y_pruned, s=8, c="#d62728", alpha=0.8, label="Pruned", linewidths=0.0
    )

    if use_log_x:
        ax.set_xscale("log")
    ax.set_ylim(bottom=0.0)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.25)
    if with_legend:
        ax.legend(loc="best", fontsize=8)


def save_filter_decision_scatter_plots(
    filter_df: pd.DataFrame,
    checkpoints: List[CheckpointInfo],
    output_dir: Path,
    use_log_x: bool = True,
) -> None:
    if filter_df.empty:
        return

    n = len(checkpoints)
    max_iter = max(ckpt.iteration for ckpt in checkpoints)
    ncols = 5
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.1, nrows * 3.4))
    axes = np.array(axes).reshape(-1)

    for idx, ckpt in enumerate(checkpoints):
        ax = axes[idx]
        subset = filter_df[filter_df["iteration"] == ckpt.iteration]
        if subset.empty:
            ax.axis("off")
            continue
        decision_iter = int(subset["decision_iteration"].iloc[0])
        projected = " projected" if decision_iter > max_iter else ""
        title = f"{ckpt.label} (d={decision_iter}{projected})"
        _plot_single_decision_scatter(
            ax=ax,
            subset=subset,
            title=title,
            use_log_x=use_log_x,
            with_legend=(idx == 0),
        )
        if idx % ncols == 0:
            ax.set_ylabel("Frequency score")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Gradient saliency")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        "Filter Score Distribution and Pruning Decision (green=kept, red=pruned)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "filter_score_decision_grid.png", dpi=220)
    plt.close(fig)

    per_model_dir = output_dir / "filter_score_decision_plots"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in checkpoints:
        subset = filter_df[filter_df["iteration"] == ckpt.iteration]
        if subset.empty:
            continue
        decision_iter = int(subset["decision_iteration"].iloc[0])
        projected = " projected" if decision_iter > max_iter else ""
        fig, ax = plt.subplots(figsize=(7.4, 5.8))
        _plot_single_decision_scatter(
            ax=ax,
            subset=subset,
            title=f"{ckpt.label} (decision iteration={decision_iter}{projected})",
            use_log_x=use_log_x,
            with_legend=True,
        )
        ax.set_xlabel("Gradient saliency")
        ax.set_ylabel("Frequency score")
        fig.tight_layout()
        safe_label = ckpt.label.replace(" ", "_")
        fig.savefig(per_model_dir / f"filter_score_decision_{safe_label}.png", dpi=220)
        plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze frequency/saliency scores and pruning stats over iterations."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "tiny_imagenet"],
        help="Dataset/task used by checkpoints.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="EXPERIMENT/11022026_171013_cifar10_BASE/models/best_model_187_0.944.h5",
        help="Path to baseline (unpruned) model checkpoint.",
    )
    parser.add_argument(
        "--pruned_root",
        type=str,
        default="EXPERIMENT/15022026_015621_cifar10/",
        help="Directory containing pruned models or experiment root.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_hybrid_iter_*_best_model.h5",
        help="Glob pattern used to discover pruned checkpoints in models dir.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="val",
        choices=["train_eval", "val", "test", "train"],
        help="Dataset split for gradient saliency computation.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=8,
        help="Number of dataset batches per model for gradient saliency.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for loading dataset split.",
    )
    parser.add_argument(
        "--kappa_ratio",
        type=float,
        default=None,
        help="Low-frequency kappa ratio for frequency score. Default uses config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic analysis.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <pruned_root>/analysis or <pruned_root_parent>/analysis",
    )
    parser.add_argument(
        "--decision_alpha",
        type=float,
        default=None,
        help="Alpha used in combined score: grad * (freq + eps)^alpha. Default uses config.hybrid_alpha.",
    )
    parser.add_argument(
        "--linear_x",
        action="store_true",
        help="Use linear x-axis for gradient saliency in decision scatter plots (default: log scale).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    baseline_model = Path(args.baseline_model).expanduser().resolve()
    if not baseline_model.is_file():
        raise FileNotFoundError(f"Baseline model not found: {baseline_model}")

    pruned_root = Path(args.pruned_root).expanduser().resolve()
    models_dir = resolve_models_dir(pruned_root)
    pruned_ckpts = collect_pruned_checkpoints(models_dir, args.pattern)
    if not pruned_ckpts:
        raise FileNotFoundError(
            f"No pruned checkpoints found in {models_dir} with pattern `{args.pattern}`."
        )

    output_dir: Path
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        if models_dir.name == "models":
            output_dir = models_dir.parent / "analysis"
        else:
            output_dir = models_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(task=args.task)
    config.batch_size = int(args.batch_size)
    config.data_augmentation = False
    config.use_mixup = False

    analysis_ds = load_dataset_for_analysis(config, args.dataset_split)
    test_ds = analysis_ds if args.dataset_split == "test" else load_dataset_for_analysis(config, "test")
    frn_analyzer = FrequencyRelevanceAnalyzer(config)
    kappa_ratio = (
        float(args.kappa_ratio)
        if args.kappa_ratio is not None
        else float(config.hybrid_initial_kappa_ratio)
    )
    decision_alpha = (
        float(args.decision_alpha)
        if args.decision_alpha is not None
        else float(getattr(config, "hybrid_alpha", 0.5))
    )
    decision_min_filters = int(getattr(config, "hybrid_min_filters", 8))

    checkpoints: List[CheckpointInfo] = [
        CheckpointInfo(iteration=0, label="baseline", path=baseline_model)
    ] + pruned_ckpts
    checkpoints.sort(key=lambda item: item.iteration)

    global_rows: List[Dict] = []
    layer_rows: List[Dict] = []
    filter_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    baseline_counts: Optional[Dict[str, int]] = None
    prev_counts: Optional[Dict[str, int]] = None
    layer_order: List[str] = []

    for ckpt in checkpoints:
        print(f"[analyze] iteration={ckpt.iteration:>2} model={ckpt.path.name}")
        model = tf.keras.models.load_model(str(ckpt.path), compile=False)
        conv_layers = ModelUtils.get_conv_layers(model)
        counts = {layer.name: int(layer.filters) for layer in conv_layers}

        if baseline_counts is None:
            baseline_counts = counts.copy()
            prev_counts = counts.copy()
            layer_order = list(counts.keys())

        assert baseline_counts is not None
        assert prev_counts is not None

        frequency_scores = compute_frequency_scores(model, frn_analyzer, kappa_ratio)
        gradient_scores = compute_gradient_saliency(
            model=model, dataset=analysis_ds, max_batches=args.max_batches
        )
        combined_scores = combine_scores(
            frequency_scores=frequency_scores,
            gradient_scores=gradient_scores,
            alpha=decision_alpha,
        )
        decision_iteration = max(1, ckpt.iteration + 1)
        decision_prune_ratio = prune_ratio_for_iteration(config, decision_iteration)
        decision_masks = select_pruning_masks(
            scores=combined_scores,
            prune_ratio=decision_prune_ratio,
            min_filters=decision_min_filters,
        )
        filter_rows.extend(
            build_filter_decision_rows(
                iteration=ckpt.iteration,
                label=ckpt.label,
                checkpoint=str(ckpt.path),
                frequency_scores=frequency_scores,
                gradient_scores=gradient_scores,
                combined_scores=combined_scores,
                decision_masks=decision_masks,
                decision_iteration=decision_iteration,
                prune_ratio=decision_prune_ratio,
            )
        )

        freq_flat = flatten_scores(frequency_scores)
        grad_flat = flatten_scores(gradient_scores)
        freq_global = score_stats(freq_flat)
        grad_global = score_stats(grad_flat)

        base_total_filters = int(sum(baseline_counts.get(name, 0) for name in layer_order))
        total_filters = int(sum(counts.get(name, 0) for name in layer_order))
        total_pruned_this = int(
            sum(
                max(prev_counts.get(name, baseline_counts[name]) - counts.get(name, 0), 0)
                for name in layer_order
            )
        )
        total_pruned_cumulative = int(
            sum(max(baseline_counts[name] - counts.get(name, 0), 0) for name in layer_order)
        )
        cumulative_ratio = total_pruned_cumulative / max(base_total_filters, 1)
        model_params = int(model.count_params())
        model_flops = int(ModelUtils.compute_flops(model))
        test_accuracy = compute_accuracy(model, test_ds)
        model_size_mb = ckpt.path.stat().st_size / (1024.0 * 1024.0)

        global_rows.append(
            {
                "iteration": ckpt.iteration,
                "label": ckpt.label,
                "checkpoint": str(ckpt.path),
                "model_params": model_params,
                "model_size_mb": model_size_mb,
                "total_filters": total_filters,
                "total_pruned_this_iter": total_pruned_this,
                "total_pruned_cumulative": total_pruned_cumulative,
                "cumulative_pruning_ratio": cumulative_ratio,
                "test_accuracy": test_accuracy,
                "flops": model_flops,
                "frequency_mean": freq_global["mean"],
                "frequency_std": freq_global["std"],
                "frequency_median": freq_global["median"],
                "gradient_mean": grad_global["mean"],
                "gradient_std": grad_global["std"],
                "gradient_median": grad_global["median"],
            }
        )
        eval_rows.append(
            {
                "Iteration": int(ckpt.iteration),
                "Validation accuracy": float(test_accuracy),
                "Number of params": int(model_params),
                "FLOPs": int(model_flops),
            }
        )

        for layer_name in layer_order:
            base_count = int(baseline_counts.get(layer_name, 0))
            prev_count = int(prev_counts.get(layer_name, base_count))
            curr_count = int(counts.get(layer_name, 0))

            pruned_this = max(prev_count - curr_count, 0)
            pruned_cumulative = max(base_count - curr_count, 0)
            remaining_ratio = curr_count / max(base_count, 1)

            freq_stats = score_stats(frequency_scores.get(layer_name))
            grad_stats = score_stats(gradient_scores.get(layer_name))

            layer_rows.append(
                {
                    "iteration": ckpt.iteration,
                    "label": ckpt.label,
                    "checkpoint": str(ckpt.path),
                    "layer_name": layer_name,
                    "filter_count": curr_count,
                    "base_filter_count": base_count,
                    "pruned_this_iter": pruned_this,
                    "pruned_cumulative": pruned_cumulative,
                    "remaining_ratio": remaining_ratio,
                    "frequency_mean": freq_stats["mean"],
                    "frequency_std": freq_stats["std"],
                    "frequency_min": freq_stats["min"],
                    "frequency_max": freq_stats["max"],
                    "gradient_mean": grad_stats["mean"],
                    "gradient_std": grad_stats["std"],
                    "gradient_min": grad_stats["min"],
                    "gradient_max": grad_stats["max"],
                }
            )

        prev_counts = counts.copy()

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    global_df = pd.DataFrame(global_rows).sort_values("iteration").reset_index(drop=True)
    layer_df = (
        pd.DataFrame(layer_rows)
        .sort_values(["iteration", "layer_name"])
        .reset_index(drop=True)
    )
    eval_df = pd.DataFrame(eval_rows).sort_values("Iteration").reset_index(drop=True)
    if filter_rows:
        filter_df = (
            pd.DataFrame(filter_rows)
            .sort_values(["iteration", "layer_name", "filter_index"])
            .reset_index(drop=True)
        )
    else:
        filter_df = pd.DataFrame(
            columns=[
                "iteration",
                "label",
                "checkpoint",
                "decision_iteration",
                "decision_prune_ratio",
                "layer_name",
                "filter_index",
                "frequency_score",
                "gradient_score",
                "combined_score",
                "decision",
            ]
        )

    global_df.to_csv(output_dir / "global_summary.csv", index=False)
    layer_df.to_csv(output_dir / "layer_summary.csv", index=False)
    filter_df.to_csv(output_dir / "filter_scores_with_decision.csv", index=False)
    eval_df.to_csv(output_dir / "iteration_eval_summary.csv", index=False)

    save_global_plots(global_df, output_dir)
    save_layer_heatmaps(layer_df, layer_order, output_dir)
    save_filter_decision_scatter_plots(
        filter_df=filter_df,
        checkpoints=checkpoints,
        output_dir=output_dir,
        use_log_x=not args.linear_x,
    )

    final_row = global_df.iloc[-1].to_dict()
    meta = {
        "task": args.task,
        "baseline_model": str(baseline_model),
        "pruned_models_dir": str(models_dir),
        "num_checkpoints": len(checkpoints),
        "max_iteration": int(global_df["iteration"].max()),
        "dataset_split": args.dataset_split,
        "max_batches": int(args.max_batches),
        "kappa_ratio": float(kappa_ratio),
        "decision_alpha": float(decision_alpha),
        "decision_min_filters": int(decision_min_filters),
        "decision_axis_scale": "linear" if args.linear_x else "log",
        "final_summary": final_row,
    }
    with open(output_dir / "analysis_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    summary_lines = [
        f"Task: {args.task}",
        f"Baseline model: {baseline_model}",
        f"Pruned models dir: {models_dir}",
        f"Analyzed checkpoints: {len(checkpoints)}",
        f"Last iteration: {int(global_df['iteration'].max())}",
        f"Final cumulative pruning ratio: {final_row['cumulative_pruning_ratio'] * 100:.2f}%",
        f"Final total filters: {int(final_row['total_filters'])}",
        f"Final test accuracy: {final_row['test_accuracy']:.6f}",
        f"Final frequency mean: {final_row['frequency_mean']:.6f}",
        f"Final gradient mean: {final_row['gradient_mean']:.6f}",
        f"Decision alpha: {decision_alpha:.4f}",
        f"Decision x-axis scale: {'linear' if args.linear_x else 'log'}",
    ]
    with open(output_dir / "analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"[analyze] Analysis complete. Outputs saved to: {output_dir}")
    print("[analyze] Generated files:")
    for name in [
        "global_summary.csv",
        "layer_summary.csv",
        "filter_scores_with_decision.csv",
        "iteration_eval_summary.csv",
        "global_score_trends.png",
        "normalized_trends.png",
        "pruning_progress.png",
        "layer_heatmaps.png",
        "filter_score_decision_grid.png",
        str(output_dir / "filter_score_decision_plots"),
        "analysis_meta.json",
        "analysis_summary.txt",
    ]:
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    main()
