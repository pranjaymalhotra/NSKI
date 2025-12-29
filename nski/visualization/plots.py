"""
Publication-quality plotting functions for NSKI.

Creates figures suitable for academic papers with proper formatting,
confidence intervals, and statistical annotations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


logger = logging.getLogger(__name__)


# Publication-ready style settings
PUBLICATION_STYLE = {
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# Color palette
COLORS = {
    "baseline": "#7f7f7f",  # Gray
    "nski": "#2ca02c",      # Green
    "arditi": "#1f77b4",    # Blue
    "belitsky": "#ff7f0e",  # Orange
    "jbshield": "#9467bd",  # Purple
}


def setup_style() -> None:
    """Setup matplotlib style for publication figures."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping style setup")
        return
    
    plt.rcParams.update(PUBLICATION_STYLE)
    
    if SEABORN_AVAILABLE:
        sns.set_palette("colorblind")
        sns.set_style("whitegrid")


def plot_asr_comparison(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Attack Success Rate Comparison",
    figsize: Tuple[float, float] = (10, 6),
) -> Optional[Any]:
    """
    Plot ASR comparison across methods with confidence intervals.
    
    Args:
        results: List of result dicts with keys:
            - model: model name
            - method: method name
            - asr: attack success rate
            - asr_ci: [lower, upper] confidence interval (optional)
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
        
    Returns:
        matplotlib figure object or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None
    
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by model
    models = sorted(set(r["model"] for r in results))
    methods = sorted(set(r["method"] for r in results))
    
    x = np.arange(len(models))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method"] == method]
        
        asrs = []
        errors = []
        for model in models:
            model_result = next((r for r in method_results if r["model"] == model), None)
            if model_result:
                asrs.append(model_result["asr"])
                if "asr_ci" in model_result:
                    ci = model_result["asr_ci"]
                    errors.append([model_result["asr"] - ci[0], ci[1] - model_result["asr"]])
                else:
                    errors.append([0, 0])
            else:
                asrs.append(0)
                errors.append([0, 0])
        
        errors = np.array(errors).T
        
        color = COLORS.get(method, f"C{i}")
        offset = (i - len(methods) / 2 + 0.5) * width
        
        ax.bar(
            x + offset,
            asrs,
            width,
            label=method.upper(),
            color=color,
            yerr=errors,
            capsize=3,
            error_kw={"linewidth": 1},
        )
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    
    # Add baseline ASR annotation
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ASR comparison to: {output_path}")
    
    return fig


def plot_utility_comparison(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Utility Score Comparison",
    figsize: Tuple[float, float] = (10, 6),
) -> Optional[Any]:
    """
    Plot utility comparison across methods.
    
    Args:
        results: List of result dicts
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
        
    Returns:
        matplotlib figure object or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None
    
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = sorted(set(r["model"] for r in results))
    methods = sorted(set(r["method"] for r in results))
    
    x = np.arange(len(models))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        method_results = [r for r in results if r["method"] == method]
        
        utilities = []
        errors = []
        for model in models:
            model_result = next((r for r in method_results if r["model"] == model), None)
            if model_result:
                utilities.append(model_result["utility"])
                if "utility_ci" in model_result:
                    ci = model_result["utility_ci"]
                    errors.append([model_result["utility"] - ci[0], ci[1] - model_result["utility"]])
                else:
                    errors.append([0, 0])
            else:
                utilities.append(0)
                errors.append([0, 0])
        
        errors = np.array(errors).T
        
        color = COLORS.get(method, f"C{i}")
        offset = (i - len(methods) / 2 + 0.5) * width
        
        ax.bar(
            x + offset,
            utilities,
            width,
            label=method.upper(),
            color=color,
            yerr=errors,
            capsize=3,
        )
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Utility Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved utility comparison to: {output_path}")
    
    return fig


def plot_ablation_results(
    results: List[Dict[str, Any]],
    ablation_type: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Optional[Any]:
    """
    Plot ablation study results.
    
    Args:
        results: List of ablation result dicts
        ablation_type: Type of ablation ("layer", "strength", "component", "calibration")
        output_path: Path to save figure
        title: Figure title (auto-generated if None)
        figsize: Figure size
        
    Returns:
        matplotlib figure object or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None
    
    setup_style()
    
    # Filter by ablation type
    filtered = [r for r in results if r.get("type") == ablation_type]
    if not filtered:
        logger.warning(f"No results for ablation type: {ablation_type}")
        return None
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Sort by value
    if ablation_type in ["strength", "calibration"]:
        filtered = sorted(filtered, key=lambda x: float(x["value"]))
    
    values = [str(r["value"]) for r in filtered]
    asrs = [r["asr"] for r in filtered]
    utilities = [r["utility"] for r in filtered]
    
    x = np.arange(len(values))
    width = 0.35
    
    # ASR bars
    bars1 = ax1.bar(x - width/2, asrs, width, label="ASR", color=COLORS["nski"])
    ax1.set_xlabel(f"{ablation_type.title()} Value")
    ax1.set_ylabel("ASR", color=COLORS["nski"])
    ax1.tick_params(axis="y", labelcolor=COLORS["nski"])
    ax1.set_ylim(0, 1)
    
    # Utility on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, utilities, width, label="Utility", color=COLORS["arditi"])
    ax2.set_ylabel("Utility", color=COLORS["arditi"])
    ax2.tick_params(axis="y", labelcolor=COLORS["arditi"])
    ax2.set_ylim(0, 1.1)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(values, rotation=45 if len(values) > 6 else 0)
    
    if title is None:
        title = f"{ablation_type.title()} Ablation: ASR vs Utility"
    ax1.set_title(title)
    
    # Combined legend
    handles = [bars1, bars2]
    labels = ["ASR (↓ better)", "Utility (↑ better)"]
    ax1.legend(handles, labels, loc="upper right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ablation plot to: {output_path}")
    
    return fig


def plot_pareto_frontier(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "ASR vs Utility Trade-off",
    figsize: Tuple[float, float] = (8, 6),
) -> Optional[Any]:
    """
    Plot ASR vs Utility trade-off with Pareto frontier.
    
    Args:
        results: List of result dicts with asr and utility
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
        
    Returns:
        matplotlib figure object or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None
    
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each method
    for r in results:
        method = r.get("method", "unknown")
        color = COLORS.get(method, "#333333")
        
        ax.scatter(
            r["asr"],
            r["utility"],
            s=150,
            c=color,
            label=method.upper(),
            marker="o",
            edgecolors="white",
            linewidths=2,
        )
        
        # Add error bars if CI available
        if "asr_ci" in r and "utility_ci" in r:
            asr_ci = r["asr_ci"]
            util_ci = r["utility_ci"]
            ax.errorbar(
                r["asr"],
                r["utility"],
                xerr=[[r["asr"] - asr_ci[0]], [asr_ci[1] - r["asr"]]],
                yerr=[[r["utility"] - util_ci[0]], [util_ci[1] - r["utility"]]],
                fmt="none",
                c=color,
                alpha=0.5,
                capsize=3,
            )
    
    # Compute and plot Pareto frontier
    points = [(r["asr"], r["utility"]) for r in results]
    pareto_points = compute_pareto_frontier(points)
    if len(pareto_points) > 1:
        pareto_points = sorted(pareto_points, key=lambda x: x[0])
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, "k--", alpha=0.5, label="Pareto Frontier")
    
    # Ideal point
    ax.scatter(0, 1, s=200, marker="*", c="gold", edgecolors="black", label="Ideal", zorder=5)
    
    ax.set_xlabel("Attack Success Rate (↓ better)")
    ax.set_ylabel("Utility Score (↑ better)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.15)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower left")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Pareto plot to: {output_path}")
    
    return fig


def compute_pareto_frontier(
    points: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Compute Pareto frontier for ASR (minimize) vs Utility (maximize).
    
    Args:
        points: List of (asr, utility) tuples
        
    Returns:
        List of Pareto-optimal points
    """
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            # q dominates p if q has lower ASR and higher utility
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return pareto


def plot_layer_heatmap(
    layer_results: Dict[int, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Per-Layer Intervention Effect",
    figsize: Tuple[float, float] = (12, 4),
) -> Optional[Any]:
    """
    Plot heatmap of intervention effect per layer.
    
    Args:
        layer_results: Dict mapping layer index to metrics dict
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
        
    Returns:
        matplotlib figure object or None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available")
        return None
    
    setup_style()
    
    layers = sorted(layer_results.keys())
    asrs = [layer_results[l].get("asr", 0) for l in layers]
    utilities = [layer_results[l].get("utility", 0) for l in layers]
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # ASR heatmap
    asr_data = np.array(asrs).reshape(1, -1)
    im1 = axes[0].imshow(asr_data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[0].set_ylabel("ASR")
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0], orientation="vertical", pad=0.02)
    
    # Utility heatmap
    util_data = np.array(utilities).reshape(1, -1)
    im2 = axes[1].imshow(util_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_ylabel("Utility")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Layer Index")
    axes[1].set_xticks(range(len(layers)))
    axes[1].set_xticklabels(layers)
    plt.colorbar(im2, ax=axes[1], orientation="vertical", pad=0.02)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved layer heatmap to: {output_path}")
    
    return fig


def create_publication_figures(
    results_dir: str,
    output_dir: str,
    prefix: str = "fig",
) -> List[str]:
    """
    Create all publication figures from experiment results.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save figures
        prefix: Prefix for figure filenames
        
    Returns:
        List of paths to generated figures
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available, cannot create figures")
        return []
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    
    # Load main comparison results
    main_path = results_dir / "main_comparison" / "main_comparison_results.json"
    if main_path.exists():
        with open(main_path) as f:
            main_results = json.load(f)
        
        # Figure 1: ASR comparison
        fig_path = output_dir / f"{prefix}_1_asr_comparison.png"
        plot_asr_comparison(main_results, str(fig_path))
        generated.append(str(fig_path))
        
        # Figure 2: Utility comparison
        fig_path = output_dir / f"{prefix}_2_utility_comparison.png"
        plot_utility_comparison(main_results, str(fig_path))
        generated.append(str(fig_path))
        
        # Figure 3: Pareto frontier
        fig_path = output_dir / f"{prefix}_3_pareto_frontier.png"
        plot_pareto_frontier(main_results, str(fig_path))
        generated.append(str(fig_path))
    
    # Load ablation results
    ablation_path = results_dir / "ablation" / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_results = json.load(f)
        
        # Figure 4-7: Ablation studies
        for i, abl_type in enumerate(["layer", "strength", "component", "calibration"], 4):
            type_results = [r for r in ablation_results if r.get("ablation_type") == abl_type]
            if type_results:
                # Convert format
                converted = [
                    {
                        "type": r["ablation_type"],
                        "value": r["ablation_value"],
                        "asr": r["asr"],
                        "utility": r["utility"],
                    }
                    for r in type_results
                ]
                
                fig_path = output_dir / f"{prefix}_{i}_{abl_type}_ablation.png"
                plot_ablation_results(converted, abl_type, str(fig_path))
                generated.append(str(fig_path))
    
    logger.info(f"Generated {len(generated)} publication figures")
    return generated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate NSKI publication figures")
    parser.add_argument("--results", type=str, default="results", help="Results directory")
    parser.add_argument("--output", type=str, default="figures", help="Output directory")
    parser.add_argument("--prefix", type=str, default="fig", help="Figure filename prefix")
    args = parser.parse_args()
    
    figures = create_publication_figures(
        results_dir=args.results,
        output_dir=args.output,
        prefix=args.prefix,
    )
    
    print(f"\nGenerated {len(figures)} figures:")
    for f in figures:
        print(f"  - {f}")
