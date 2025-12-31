#!/usr/bin/env python3
"""
NSKI Comprehensive Variant Comparison

Tests all NSKI variants and novel approaches on safety-trained models.

KEY INSIGHT: NSKI amplifies existing refusal behavior. Models without safety 
training (GPT-2, base models) have NO refusal to amplify, so results are identical.

This script:
1. Uses SAFETY-TRAINED models (Llama-2-chat, Mistral-Instruct)
2. Tests multiple NSKI variants
3. Tests novel approaches (RepE, ITI, ActAdd)
4. Provides comprehensive comparison

Author: Pranjay Malhotra
Date: December 2024
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from tqdm import tqdm

# Import NSKI components
from nski.models import ModelLoader, get_model_config
from nski.data import AdvBenchDataset, AlpacaDataset
from nski.evaluation import compute_asr
from nski.evaluation.judges import KeywordRefusalJudge
from nski.core.nski_variants import (
    NSKI_VARIANTS, 
    NSKIStandard, 
    NSKIEarlyLayers, 
    NSKIMiddleLayers, 
    NSKILateLayers,
    NSKIAdaptive,
    NSKIContrastive,
    RepresentationEngineering,
    InferenceTimeIntervention,
    ActivationAddition,
    NSKIVariantConfig,
)


@dataclass  
class ExperimentConfig:
    """Configuration for variant comparison."""
    # Models to test (MUST be safety-trained!)
    models: List[str] = None
    
    # Variants to test
    variants: List[str] = None
    
    # Strength values
    strengths: List[float] = None
    
    # Samples
    n_harmful: int = 100
    n_harmless: int = 100
    n_calibration: int = 50
    
    # Generation
    max_new_tokens: int = 150
    
    # Output
    output_dir: str = "results/variant_comparison"
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                # Safety-trained models that work with NSKI
                "mistralai/Mistral-7B-Instruct-v0.2",  # Good safety, no auth needed
                "microsoft/Phi-3-mini-4k-instruct",    # Small, safe, fast
            ]
        
        if self.variants is None:
            self.variants = [
                "baseline",          # No intervention
                "nski-standard",     # Original NSKI
                "nski-early",        # Early layers only
                "nski-middle",       # Middle layers only  
                "nski-late",         # Late layers only
                "nski-adaptive",     # Adaptive strength
                "nski-contrastive",  # Contrastive direction
                "actadd",            # Activation Addition
                # "repe",            # Requires sklearn
                # "iti",             # Inference-Time Intervention
            ]
        
        if self.strengths is None:
            self.strengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def install_dependencies():
    """Install required dependencies."""
    import subprocess
    
    packages = [
        "scikit-learn",  # For RepE
        "rich",          # For display
    ]
    
    for pkg in packages:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                check=True,
                capture_output=True
            )
        except:
            pass


def run_baseline(model, tokenizer, prompts, config):
    """Run baseline (no intervention)."""
    judge = KeywordRefusalJudge()
    
    result = compute_asr(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        judge=judge,
        max_new_tokens=config.max_new_tokens,
        temperature=0.0,
        do_sample=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        show_progress=True
    )
    
    return {
        "asr": result.asr,
        "refusal_rate": result.refusal_rate,
        "n_total": result.n_total,
    }


def run_variant(
    model, 
    tokenizer, 
    model_name: str,
    variant_name: str,
    prompts: List[str],
    calibration: tuple,
    strength: float,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Run a specific NSKI variant."""
    
    model_config = get_model_config(model_name)
    model_type = model_config.get("type", "llama")
    
    # Get variant class
    variant_class = NSKI_VARIANTS.get(variant_name.lower())
    if variant_class is None:
        return {"error": f"Unknown variant: {variant_name}"}
    
    # Create variant config
    variant_config = NSKIVariantConfig(
        strength=strength,
        normalize_direction=True,
    )
    
    # Initialize variant
    try:
        variant = variant_class(model, tokenizer, model_type=model_type, config=variant_config)
    except Exception as e:
        return {"error": f"Failed to initialize {variant_name}: {e}"}
    
    # Extract direction
    harmful_cal, harmless_cal = calibration
    try:
        variant.extract_direction(harmful_cal, harmless_cal)
    except Exception as e:
        return {"error": f"Failed to extract direction: {e}"}
    
    # Prepare intervention
    try:
        variant.prepare(strength=strength)
    except Exception as e:
        variant.cleanup()
        return {"error": f"Failed to prepare: {e}"}
    
    # Evaluate
    judge = KeywordRefusalJudge()
    try:
        result = compute_asr(
            model=model,
            tokenizer=tokenizer, 
            prompts=prompts,
            judge=judge,
            max_new_tokens=config.max_new_tokens,
            temperature=0.0,
            do_sample=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            show_progress=True
        )
    finally:
        variant.cleanup()
    
    return {
        "asr": result.asr,
        "refusal_rate": result.refusal_rate,
        "n_total": result.n_total,
    }


def print_results_table(results: List[Dict], title: str = "Results"):
    """Print results in a nice table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        
        console = Console()
        
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Variant", style="magenta")
        table.add_column("Strength", style="yellow")
        table.add_column("ASR", style="red")
        table.add_column("Refusal", style="green")
        table.add_column("Delta", style="blue")
        
        # Find baseline ASR for each model
        baseline_asr = {}
        for r in results:
            if r.get("variant") == "baseline":
                baseline_asr[r["model"]] = r.get("asr", 1.0)
        
        for r in results:
            model = r.get("model", "").split("/")[-1][:20]
            variant = r.get("variant", "")
            strength = r.get("strength", "-")
            asr = r.get("asr", "error")
            refusal = r.get("refusal_rate", "error")
            
            # Calculate delta from baseline
            if isinstance(asr, float) and r["model"] in baseline_asr:
                delta = asr - baseline_asr[r["model"]]
                delta_str = f"{delta:+.3f}"
                if delta < -0.05:
                    delta_str = f"[green]{delta_str}[/green]"  # Good!
                elif delta > 0.05:
                    delta_str = f"[red]{delta_str}[/red]"  # Bad
            else:
                delta_str = "-"
            
            if isinstance(asr, float):
                asr_str = f"{asr:.3f}"
            else:
                asr_str = str(asr)
            
            if isinstance(refusal, float):
                refusal_str = f"{refusal:.3f}"
            else:
                refusal_str = str(refusal)
            
            strength_str = f"{strength:.1f}" if isinstance(strength, float) else str(strength)
            
            table.add_row(model, variant, strength_str, asr_str, refusal_str, delta_str)
        
        console.print(table)
        
    except ImportError:
        # Fallback to simple print
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
        print(f"  {'Model':<20} {'Variant':<15} {'Strength':<8} {'ASR':<8} {'Refusal':<8}")
        print(f"  {'-'*70}")
        for r in results:
            model = r.get("model", "").split("/")[-1][:20]
            variant = r.get("variant", "")[:15]
            strength = r.get("strength", "-")
            asr = r.get("asr", "error")
            refusal = r.get("refusal_rate", "error")
            
            if isinstance(asr, float):
                asr = f"{asr:.3f}"
            if isinstance(refusal, float):
                refusal = f"{refusal:.3f}"
            if isinstance(strength, float):
                strength = f"{strength:.1f}"
            
            print(f"  {model:<20} {variant:<15} {strength:<8} {asr:<8} {refusal:<8}")


def main():
    parser = argparse.ArgumentParser(description="NSKI Variant Comparison")
    parser.add_argument("--models", nargs="+", help="Models to test")
    parser.add_argument("--variants", nargs="+", help="Variants to test")
    parser.add_argument("--samples", type=int, default=50, help="Number of harmful samples")
    parser.add_argument("--strengths", nargs="+", type=float, default=[1.0, 2.0], help="Strength values")
    parser.add_argument("--output", "-o", type=str, default="results/variant_comparison")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer samples")
    args = parser.parse_args()
    
    # Install dependencies
    print("Installing dependencies...")
    install_dependencies()
    
    # Create config
    config = ExperimentConfig(
        n_harmful=args.samples if not args.quick else 20,
        n_harmless=args.samples if not args.quick else 20,
        n_calibration=30 if not args.quick else 15,
        output_dir=args.output,
    )
    
    if args.models:
        config.models = args.models
    if args.variants:
        config.variants = args.variants
    if args.strengths:
        config.strengths = args.strengths
    
    # Print config
    print("\n" + "="*70)
    print("  NSKI VARIANT COMPARISON")
    print("="*70)
    print(f"  Models: {config.models}")
    print(f"  Variants: {config.variants}")
    print(f"  Strengths: {config.strengths}")
    print(f"  Samples: {config.n_harmful} harmful, {config.n_harmless} harmless")
    print("="*70 + "\n")
    
    # Load datasets
    print("Loading datasets...")
    advbench = AdvBenchDataset()
    alpaca = AlpacaDataset(n_samples=config.n_harmless)
    
    harmful_prompts = advbench.prompts[:config.n_harmful]
    harmless_prompts = alpaca.prompts[:config.n_harmless]
    calibration = (
        harmful_prompts[:config.n_calibration],
        harmless_prompts[:config.n_calibration]
    )
    
    print(f"  Harmful: {len(harmful_prompts)} prompts")
    print(f"  Harmless: {len(harmless_prompts)} prompts")
    print(f"  Calibration: {config.n_calibration} each")
    
    # Results storage
    all_results = []
    
    # Run experiments
    for model_name in config.models:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*70}")
        
        # Load model
        print(f"  Loading model...")
        try:
            loader = ModelLoader(model_name, quantization="4bit")
            model, tokenizer = loader.load()
            print(f"  Model loaded successfully")
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue
        
        # Run baseline first
        print(f"\n  [Baseline] Running...")
        baseline_result = run_baseline(model, tokenizer, harmful_prompts, config)
        baseline_result["model"] = model_name
        baseline_result["variant"] = "baseline"
        baseline_result["strength"] = "-"
        all_results.append(baseline_result)
        print(f"  [Baseline] ASR: {baseline_result['asr']:.3f}")
        
        # Run each variant at each strength
        for variant_name in config.variants:
            if variant_name == "baseline":
                continue
            
            for strength in config.strengths:
                print(f"\n  [{variant_name}] strength={strength}")
                
                result = run_variant(
                    model, tokenizer, model_name,
                    variant_name, harmful_prompts, calibration,
                    strength, config
                )
                
                result["model"] = model_name
                result["variant"] = variant_name
                result["strength"] = strength
                all_results.append(result)
                
                if "error" in result:
                    print(f"  [{variant_name}] ERROR: {result['error']}")
                else:
                    delta = result['asr'] - baseline_result['asr']
                    print(f"  [{variant_name}] ASR: {result['asr']:.3f} (delta: {delta:+.3f})")
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    # Print final results
    print("\n")
    print_results_table(all_results, "NSKI Variant Comparison Results")
    
    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"variant_comparison_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "config": asdict(config),
            "results": all_results,
            "timestamp": timestamp,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print key findings
    print("\n" + "="*70)
    print("  KEY FINDINGS")
    print("="*70)
    
    # Find best variant for each model
    for model_name in config.models:
        model_results = [r for r in all_results if r.get("model") == model_name and "asr" in r]
        if not model_results:
            continue
        
        baseline = next((r for r in model_results if r.get("variant") == "baseline"), None)
        if not baseline:
            continue
        
        best = min(model_results, key=lambda x: x.get("asr", 1.0))
        
        print(f"\n  {model_name.split('/')[-1]}:")
        print(f"    Baseline ASR: {baseline['asr']:.3f}")
        print(f"    Best variant: {best['variant']} (strength={best.get('strength', '-')})")
        print(f"    Best ASR: {best['asr']:.3f}")
        print(f"    Reduction: {baseline['asr'] - best['asr']:.3f} ({(baseline['asr'] - best['asr'])/baseline['asr']*100:.1f}%)")
    
    print("\n" + "="*70)
    print("  NOTE: NSKI requires safety-trained models to work properly!")
    print("  GPT-2 and base models have no refusal behavior to amplify.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
