"""
Run all NSKI experiments.

This is the main entry point for running the complete experiment suite.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

import torch
import yaml

from .main_comparison import MainComparisonExperiment, ExperimentResult
from .ablation_study import AblationStudy, AblationResult
from ..core import setup_logging, set_seed


logger = logging.getLogger(__name__)


def run_all_experiments(
    config_path: Optional[str] = None,
    output_dir: str = "results",
    experiments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run all experiments.
    
    Args:
        config_path: Path to config file
        output_dir: Base output directory
        experiments: List of experiments to run. Options:
            - "main": Main comparison experiment
            - "ablation": Ablation study
            If None, runs all experiments.
            
    Returns:
        Dictionary containing all results
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(str(output_dir / "logs"))
    
    logger.info("=" * 70)
    logger.info("NSKI: Neural Surgical Key-Value Intervention")
    logger.info("Complete Experiment Suite")
    logger.info("=" * 70)
    
    # Determine which experiments to run
    if experiments is None:
        experiments = ["main", "ablation"]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiments": experiments,
        "main_comparison": None,
        "ablation": None,
    }
    
    # Run main comparison
    if "main" in experiments:
        logger.info("\n" + "=" * 70)
        logger.info("Running Main Comparison Experiment")
        logger.info("=" * 70)
        
        main_exp = MainComparisonExperiment(
            config_path=config_path,
            output_dir=str(output_dir / "main_comparison"),
        )
        main_results = main_exp.run()
        results["main_comparison"] = [
            {
                "model": r.model_name,
                "method": r.method,
                "asr": r.asr,
                "asr_ci": [r.asr_ci_lower, r.asr_ci_upper],
                "utility": r.utility,
                "utility_ci": [r.utility_ci_lower, r.utility_ci_upper],
                "perplexity": r.perplexity,
                "latency": r.latency_mean,
            }
            for r in main_results
        ]
    
    # Run ablation study
    if "ablation" in experiments:
        logger.info("\n" + "=" * 70)
        logger.info("Running Ablation Study")
        logger.info("=" * 70)
        
        ablation = AblationStudy(
            config_path=config_path,
            output_dir=str(output_dir / "ablation"),
        )
        ablation_results = ablation.run()
        results["ablation"] = [
            {
                "model": r.model_name,
                "type": r.ablation_type,
                "value": r.ablation_value,
                "asr": r.asr,
                "utility": r.utility,
                "perplexity": r.perplexity,
            }
            for r in ablation_results
        ]
    
    # Save combined results
    combined_path = output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 70)
    logger.info("All Experiments Complete!")
    logger.info(f"Results saved to: {combined_path}")
    logger.info("=" * 70)
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print experiment summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    if results.get("main_comparison"):
        print("\nMain Comparison Results:")
        print("-" * 50)
        print(f"{'Model':<15} {'Method':<12} {'ASR':>8} {'Utility':>8} {'PPL':>8}")
        print("-" * 50)
        for r in results["main_comparison"]:
            print(f"{r['model']:<15} {r['method']:<12} {r['asr']:>8.3f} {r['utility']:>8.3f} {r['perplexity']:>8.2f}")
    
    if results.get("ablation"):
        print("\nAblation Study Results:")
        print("-" * 50)
        
        # Group by type
        by_type = {}
        for r in results["ablation"]:
            t = r["type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r)
        
        for abl_type, items in by_type.items():
            print(f"\n{abl_type.upper()}:")
            for r in items:
                print(f"  {r['value']}: ASR={r['asr']:.3f}, Utility={r['utility']:.3f}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="NSKI Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python -m nski.experiments.run_all
  
  # Run only main comparison
  python -m nski.experiments.run_all --experiments main
  
  # Run with custom config
  python -m nski.experiments.run_all --config configs/custom.yaml
  
  # Specify output directory
  python -m nski.experiments.run_all --output results/run1
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["main", "ablation"],
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Run experiments
    results = run_all_experiments(
        config_path=args.config,
        output_dir=args.output,
        experiments=args.experiments,
    )
    
    return results


if __name__ == "__main__":
    main()
