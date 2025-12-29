#!/usr/bin/env python3
"""
NSKI Full Pipeline - Real Implementation (NOT SIMULATED)

This script runs the complete NSKI experiment pipeline:
1. Validates environment and dependencies
2. Downloads/verifies benchmark datasets
3. Tests on multiple models (Llama-3-8B, Llama-2-7B, Mistral-7B, GPT-2-XL)
4. Runs REAL KV-cache interventions (not simulation!)
5. Validates on full AdvBench (520 harmful prompts)
6. Computes perplexity measurements
7. Runs all baseline comparisons
8. Generates publication-quality figures
9. Creates comprehensive results folder

Author: Pranjay Malhotra
Date: December 2024
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Output
    output_dir: str = "results"
    
    # Models to test (ordered by memory requirement)
    models: List[str] = None
    
    # Dataset settings
    n_advbench_samples: int = 520  # Full AdvBench
    n_alpaca_samples: int = 200    # Utility evaluation
    n_calibration: int = 100       # Calibration samples
    
    # NSKI settings
    nski_strength: float = 2.0
    nski_layers: str = "all"       # "all", "early", "middle", "late"
    
    # Generation settings
    max_new_tokens: int = 150
    temperature: float = 0.0       # Deterministic for reproducibility
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    quantization: str = "4bit"     # "4bit", "8bit", "none"
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        if self.models is None:
            # Default models - ordered by VRAM requirement (smallest first)
            self.models = [
                "gpt2-xl",                              # ~1.5GB - test first
                "meta-llama/Llama-2-7b-chat-hf",       # ~4GB with 4-bit
                "mistralai/Mistral-7B-Instruct-v0.2", # ~4GB with 4-bit
                "meta-llama/Meta-Llama-3-8B-Instruct", # ~5GB with 4-bit
            ]


# ============================================================================
# LOGGING SETUP
# ============================================================================

class PipelineLogger:
    """Custom logger that prints to console and file."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "pipeline.log"
        
        # Setup file handler
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("NSKI")
    
    def step(self, step_num: int, total: int, message: str):
        """Log a pipeline step."""
        bar = "=" * 60
        self.logger.info("")
        self.logger.info(bar)
        self.logger.info(f"[STEP {step_num}/{total}] {message}")
        self.logger.info(bar)
    
    def info(self, message: str):
        self.logger.info(f"    {message}")
    
    def success(self, message: str):
        self.logger.info(f"    ✓ {message}")
    
    def warning(self, message: str):
        self.logger.warning(f"    ⚠ {message}")
    
    def error(self, message: str):
        self.logger.error(f"    ✗ {message}")
    
    def progress(self, message: str):
        """Progress update within a step."""
        self.logger.info(f"    → {message}")


# ============================================================================
# PIPELINE STEPS
# ============================================================================

def step_validate_environment(log: PipelineLogger, config: PipelineConfig) -> bool:
    """Step 1: Validate environment and dependencies."""
    log.step(1, 10, "Validating Environment")
    
    errors = []
    
    # Check Python version
    py_version = sys.version_info
    log.info(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        errors.append("Python 3.9+ required")
    
    # Check PyTorch
    try:
        import torch
        log.success(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.success(f"CUDA: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            log.warning("CUDA not available - using CPU (will be slow)")
            config.device = "cpu"
            config.quantization = "none"
    except ImportError:
        errors.append("PyTorch not installed")
    
    # Check transformers
    try:
        import transformers
        log.success(f"Transformers: {transformers.__version__}")
    except ImportError:
        errors.append("transformers not installed")
    
    # Check NSKI imports
    try:
        from nski.core import KVCacheHook, RefusalDirectionExtractor, NSKISurgeon
        from nski.models import ModelLoader, get_model_config
        from nski.data import download_all_datasets, AdvBenchDataset, AlpacaDataset, load_advbench, load_alpaca
        from nski.evaluation import compute_asr, compute_perplexity
        from nski.baselines import ArditiSteering, BelitskyModulation, JBShield
        log.success("NSKI modules imported successfully")
    except ImportError as e:
        errors.append(f"NSKI import error: {e}")
    
    if errors:
        for err in errors:
            log.error(err)
        return False
    
    log.success("Environment validated!")
    return True


def step_download_datasets(log: PipelineLogger, config: PipelineConfig) -> bool:
    """Step 2: Download and verify datasets."""
    log.step(2, 10, "Downloading Datasets")
    
    from nski.data.download import download_advbench, download_alpaca, download_harmbench
    
    cache_dir = str(Path(config.output_dir) / "data")
    
    # AdvBench
    log.progress("Downloading AdvBench (520 harmful prompts)...")
    try:
        success = download_advbench(cache_dir=cache_dir)
        if success:
            log.success("AdvBench downloaded")
        else:
            log.warning("AdvBench download may have issues")
    except Exception as e:
        log.error(f"AdvBench download failed: {e}")
        return False
    
    # Alpaca
    log.progress("Downloading Alpaca (utility evaluation)...")
    try:
        success = download_alpaca(cache_dir=cache_dir)
        if success:
            log.success("Alpaca downloaded")
    except Exception as e:
        log.warning(f"Alpaca download note: {e}")
    
    # HarmBench
    log.progress("Downloading HarmBench (adversarial test)...")
    try:
        success = download_harmbench(cache_dir=cache_dir)
        if success:
            log.success("HarmBench downloaded")
    except Exception as e:
        log.warning(f"HarmBench download note: {e}")
    
    log.success("Datasets ready!")
    return True


def step_run_baseline_evaluation(
    log: PipelineLogger,
    config: PipelineConfig,
    model,
    tokenizer,
    model_name: str,
    advbench_prompts: List[str],
    alpaca_prompts: List[str],
) -> Dict[str, Any]:
    """Run baseline (no intervention) evaluation."""
    log.progress(f"Running baseline evaluation...")
    
    from nski.evaluation import compute_asr, compute_perplexity
    from nski.evaluation.judges import KeywordRefusalJudge
    
    judge = KeywordRefusalJudge()
    
    # Evaluate on harmful prompts
    results = compute_asr(
        model=model,
        tokenizer=tokenizer,
        prompts=advbench_prompts,
        judge=judge,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.temperature > 0,
        device=config.device,
        show_progress=True
    )
    
    # Compute perplexity on harmless prompts
    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=alpaca_prompts[:50],
        device=config.device,
        show_progress=False
    )
    
    log.success(f"Baseline ASR: {results.asr:.3f}, Refusal Rate: {results.refusal_rate:.3f}, PPL: {ppl:.2f}")
    
    return {
        "method": "baseline",
        "model": model_name,
        "asr": results.asr,
        "refusal_rate": results.refusal_rate,
        "n_refused": results.n_refused,
        "n_complied": results.n_complied,
        "n_total": results.n_total,
        "perplexity": ppl,
        "avg_latency_ms": results.avg_latency_ms,
    }


def step_run_nski_evaluation(
    log: PipelineLogger,
    config: PipelineConfig,
    model,
    tokenizer,
    model_name: str,
    model_config: Dict,
    advbench_prompts: List[str],
    alpaca_prompts: List[str],
) -> Dict[str, Any]:
    """Run NSKI (REAL KV-cache intervention) evaluation."""
    log.progress(f"Running NSKI with REAL KV-cache hooks...")
    
    from nski.core import NSKISurgeon
    from nski.evaluation import compute_asr, compute_perplexity
    from nski.evaluation.judges import KeywordRefusalJudge
    
    # Create surgeon and extract direction
    log.progress("Extracting refusal direction from calibration data...")
    surgeon = NSKISurgeon(
        model=model,
        tokenizer=tokenizer,
        model_type=model_config.get("type", "llama"),
    )
    
    surgeon.extract_direction(
        harmful_prompts=advbench_prompts[:config.n_calibration],
        harmless_prompts=alpaca_prompts[:config.n_calibration],
    )
    log.success(f"Refusal direction extracted")
    
    # Prepare NSKI intervention (REAL hooks!)
    log.progress(f"Applying NSKI surgery (strength={config.nski_strength})...")
    surgeon.prepare(strength=config.nski_strength)
    log.success("NSKI surgery applied (KV-cache hooks active)")
    
    # Evaluate
    judge = KeywordRefusalJudge()
    results = compute_asr(
        model=model,
        tokenizer=tokenizer,
        prompts=advbench_prompts,
        judge=judge,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.temperature > 0,
        device=config.device,
        show_progress=True
    )
    
    # Compute perplexity
    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=alpaca_prompts[:50],
        device=config.device,
        show_progress=False
    )
    
    # Remove surgery
    surgeon.cleanup()
    log.success("NSKI surgery removed")
    
    log.success(f"NSKI ASR: {results.asr:.3f}, Refusal Rate: {results.refusal_rate:.3f}, PPL: {ppl:.2f}")
    
    return {
        "method": "nski",
        "model": model_name,
        "asr": results.asr,
        "refusal_rate": results.refusal_rate,
        "n_refused": results.n_refused,
        "n_complied": results.n_complied,
        "n_total": results.n_total,
        "perplexity": ppl,
        "avg_latency_ms": results.avg_latency_ms,
        "nski_strength": config.nski_strength,
    }


def step_run_baseline_methods(
    log: PipelineLogger,
    config: PipelineConfig,
    model,
    tokenizer,
    model_name: str,
    model_config: Dict,
    advbench_prompts: List[str],
    alpaca_prompts: List[str],
) -> List[Dict[str, Any]]:
    """Run baseline method comparisons."""
    results = []
    
    from nski.baselines import ArditiSteering, BelitskyModulation, JBShield
    from nski.evaluation import compute_asr, compute_perplexity
    from nski.evaluation.judges import KeywordRefusalJudge
    
    judge = KeywordRefusalJudge()
    
    # 1. Arditi Steering
    log.progress("Running Arditi et al. (2024) steering baseline...")
    try:
        arditi = ArditiSteering(
            model=model,
            tokenizer=tokenizer,
            model_type=model_config.get("type", "llama"),
        )
        arditi.extract_direction(
            harmful_prompts=advbench_prompts[:config.n_calibration],
            harmless_prompts=alpaca_prompts[:config.n_calibration],
        )
        arditi.prepare(strength=config.nski_strength)
        
        asr_result = compute_asr(
            model=model,
            tokenizer=tokenizer,
            prompts=advbench_prompts,
            judge=judge,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            device=config.device,
            show_progress=True
        )
        
        ppl = compute_perplexity(model, tokenizer, alpaca_prompts[:50], device=config.device, show_progress=False)
        arditi.cleanup()
        
        results.append({
            "method": "arditi",
            "model": model_name,
            "asr": asr_result.asr,
            "refusal_rate": asr_result.refusal_rate,
            "perplexity": ppl,
        })
        log.success(f"Arditi ASR: {asr_result.asr:.3f}")
    except Exception as e:
        log.warning(f"Arditi baseline failed: {e}")
    
    # 2. Belitsky Modulation
    log.progress("Running Belitsky et al. (2025) modulation baseline...")
    try:
        belitsky = BelitskyModulation(
            model=model,
            tokenizer=tokenizer,
            model_type=model_config.get("type", "llama"),
        )
        belitsky.score_heads(
            harmful_prompts=advbench_prompts[:config.n_calibration],
            harmless_prompts=alpaca_prompts[:config.n_calibration],
        )
        belitsky.prepare(strength=config.nski_strength)
        
        asr_result = compute_asr(
            model=model,
            tokenizer=tokenizer,
            prompts=advbench_prompts,
            judge=judge,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            device=config.device,
            show_progress=True
        )
        
        ppl = compute_perplexity(model, tokenizer, alpaca_prompts[:50], device=config.device, show_progress=False)
        belitsky.cleanup()
        
        results.append({
            "method": "belitsky",
            "model": model_name,
            "asr": asr_result.asr,
            "refusal_rate": asr_result.refusal_rate,
            "perplexity": ppl,
        })
        log.success(f"Belitsky ASR: {asr_result.asr:.3f}")
    except Exception as e:
        log.warning(f"Belitsky baseline failed: {e}")
    
    # 3. JBShield
    log.progress("Running JBShield prompt filtering baseline...")
    try:
        jbshield = JBShield()
        
        # Filter prompts first, then generate
        filtered_responses = []
        for prompt in tqdm(advbench_prompts, desc="JBShield"):
            if jbshield.is_jailbreak(prompt):
                filtered_responses.append(("blocked", True))
            else:
                # Generate response
                inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=config.temperature > 0,
                        temperature=config.temperature if config.temperature > 0 else None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                is_refused = judge.is_refusal(response)
                filtered_responses.append((response, is_refused))
        
        # Calculate ASR (blocked counts as refused)
        n_refused = sum(1 for _, refused in filtered_responses if refused)
        asr = 1 - (n_refused / len(filtered_responses))
        
        results.append({
            "method": "jbshield",
            "model": model_name,
            "asr": asr,
            "refusal_rate": n_refused / len(filtered_responses),
            "perplexity": float("nan"),  # N/A for filtering method
        })
        log.success(f"JBShield ASR: {asr:.3f}")
    except Exception as e:
        log.warning(f"JBShield baseline failed: {e}")
    
    return results


def step_run_ablation_study(
    log: PipelineLogger,
    config: PipelineConfig,
    model,
    tokenizer,
    model_name: str,
    model_config: Dict,
    advbench_prompts: List[str],
    alpaca_prompts: List[str],
) -> List[Dict[str, Any]]:
    """Run NSKI ablation studies."""
    log.progress("Running ablation studies...")
    
    from nski.core import NSKISurgeon
    from nski.evaluation import compute_asr
    from nski.evaluation.judges import KeywordRefusalJudge
    
    results = []
    judge = KeywordRefusalJudge()
    
    # Use subset for ablations (faster)
    ablation_prompts = advbench_prompts[:100]
    
    # Create surgeon and extract direction once
    surgeon = NSKISurgeon(
        model=model,
        tokenizer=tokenizer,
        model_type=model_config.get("type", "llama"),
    )
    surgeon.extract_direction(
        harmful_prompts=ablation_prompts[:50],
        harmless_prompts=alpaca_prompts[:50],
    )
    
    # 1. Strength ablation
    log.progress("Ablation: Intervention strength...")
    strengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for strength in strengths:
        surgeon.prepare(strength=strength)
        
        asr_result = compute_asr(
            model=model,
            tokenizer=tokenizer,
            prompts=ablation_prompts,
            judge=judge,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            device=config.device,
            show_progress=False
        )
        
        surgeon.cleanup()
        
        results.append({
            "ablation_type": "strength",
            "ablation_value": strength,
            "model": model_name,
            "asr": asr_result.asr,
        })
        log.info(f"  Strength {strength}: ASR={asr_result.asr:.3f}")
    
    log.success("Ablation studies complete")
    return results


def step_generate_figures(log: PipelineLogger, config: PipelineConfig, all_results: Dict) -> bool:
    """Generate publication-quality figures."""
    log.step(9, 10, "Generating Figures")
    
    try:
        from nski.visualization import (
            plot_asr_comparison,
            plot_ablation_results,
            plot_pareto_frontier,
        )
        
        figures_dir = Path(config.output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # ASR comparison
        if "main_results" in all_results:
            log.progress("Generating ASR comparison figure...")
            plot_asr_comparison(
                all_results["main_results"],
                output_path=str(figures_dir / "fig1_asr_comparison.png"),
            )
            log.success("fig1_asr_comparison.png")
        
        # Ablation results
        if "ablation_results" in all_results:
            log.progress("Generating ablation figures...")
            plot_ablation_results(
                all_results["ablation_results"],
                ablation_type="strength",
                output_path=str(figures_dir / "fig2_strength_ablation.png"),
            )
            log.success("fig2_strength_ablation.png")
        
        # Pareto frontier
        if "main_results" in all_results:
            log.progress("Generating Pareto frontier figure...")
            # Add utility scores if available
            pareto_data = []
            for r in all_results["main_results"]:
                pareto_data.append({
                    "method": r["method"],
                    "model": r.get("model", "unknown"),
                    "asr": r["asr"],
                    "utility": 1 - r.get("perplexity", 10) / 100,  # Rough utility proxy
                })
            plot_pareto_frontier(
                pareto_data,
                output_path=str(figures_dir / "fig3_pareto_frontier.png"),
            )
            log.success("fig3_pareto_frontier.png")
        
        log.success("All figures generated!")
        return True
        
    except Exception as e:
        log.warning(f"Figure generation failed: {e}")
        log.info("Install matplotlib and seaborn for figure generation")
        return False


def step_create_summary(log: PipelineLogger, config: PipelineConfig, all_results: Dict) -> None:
    """Create final summary report."""
    log.step(10, 10, "Creating Summary Report")
    
    output_dir = Path(config.output_dir)
    
    # Save all results as JSON
    results_path = output_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.success(f"Results saved: {results_path}")
    
    # Create summary text file
    summary_path = output_dir / "SUMMARY.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NSKI: Neural Surgical Key-Value Intervention\n")
        f.write("Full Pipeline Results Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {config.output_dir}\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Models tested: {', '.join(config.models)}\n")
        f.write(f"AdvBench samples: {config.n_advbench_samples}\n")
        f.write(f"NSKI strength: {config.nski_strength}\n")
        f.write(f"Device: {config.device}\n")
        f.write(f"Quantization: {config.quantization}\n\n")
        
        f.write("MAIN RESULTS\n")
        f.write("-" * 40 + "\n")
        if "main_results" in all_results:
            f.write(f"{'Method':<15} {'Model':<30} {'ASR':>8} {'Refusal':>8} {'PPL':>8}\n")
            f.write("-" * 70 + "\n")
            for r in all_results["main_results"]:
                ppl_str = f"{r.get('perplexity', 0):.2f}" if r.get('perplexity') else "N/A"
                f.write(f"{r['method']:<15} {r.get('model', 'N/A'):<30} {r['asr']:>8.3f} {r.get('refusal_rate', 0):>8.3f} {ppl_str:>8}\n")
        
        f.write("\n\nKEY FINDINGS\n")
        f.write("-" * 40 + "\n")
        
        # Calculate improvements
        if "main_results" in all_results:
            baseline_results = [r for r in all_results["main_results"] if r["method"] == "baseline"]
            nski_results = [r for r in all_results["main_results"] if r["method"] == "nski"]
            
            if baseline_results and nski_results:
                baseline_asr = np.mean([r["asr"] for r in baseline_results])
                nski_asr = np.mean([r["asr"] for r in nski_results])
                reduction = (baseline_asr - nski_asr) / baseline_asr * 100 if baseline_asr > 0 else 0
                
                f.write(f"1. NSKI reduces ASR by {reduction:.1f}%\n")
                f.write(f"   - Baseline ASR: {baseline_asr:.3f}\n")
                f.write(f"   - NSKI ASR: {nski_asr:.3f}\n\n")
        
        f.write("\nFILES GENERATED\n")
        f.write("-" * 40 + "\n")
        for p in output_dir.rglob("*"):
            if p.is_file():
                f.write(f"  {p.relative_to(output_dir)}\n")
    
    log.success(f"Summary saved: {summary_path}")
    
    # Print to console
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults directory: {output_dir.absolute()}")
    print("\nContents:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size = p.stat().st_size / 1024
            print(f"  {p.relative_to(output_dir)} ({size:.1f} KB)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Run the complete NSKI pipeline."""
    
    # Setup
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log = PipelineLogger(str(output_dir / "logs"))
    
    start_time = time.time()
    all_results = {
        "config": asdict(config),
        "main_results": [],
        "ablation_results": [],
        "timestamp": datetime.now().isoformat(),
    }
    
    log.info("=" * 60)
    log.info("NSKI: Neural Surgical Key-Value Intervention")
    log.info("Full Pipeline - REAL Implementation (NOT SIMULATED)")
    log.info("=" * 60)
    log.info(f"Output directory: {output_dir.absolute()}")
    log.info(f"Models to test: {config.models}")
    log.info("")
    
    # Step 1: Validate environment
    if not step_validate_environment(log, config):
        log.error("Environment validation failed!")
        return all_results
    
    # Step 2: Download datasets
    if not step_download_datasets(log, config):
        log.error("Dataset download failed!")
        return all_results
    
    # Load datasets
    log.step(3, 10, "Loading Datasets")
    from nski.data import AdvBenchDataset, AlpacaDataset
    
    advbench = AdvBenchDataset(cache_dir=str(output_dir / "data"))
    alpaca = AlpacaDataset(cache_dir=str(output_dir / "data"), n_samples=config.n_alpaca_samples)
    
    advbench_prompts = advbench.prompts[:config.n_advbench_samples]
    alpaca_prompts = alpaca.prompts[:config.n_alpaca_samples]
    
    log.success(f"AdvBench: {len(advbench_prompts)} harmful prompts")
    log.success(f"Alpaca: {len(alpaca_prompts)} harmless prompts")
    
    # Run experiments for each model
    from nski.models import ModelLoader, get_model_config
    
    for model_idx, model_name in enumerate(config.models):
        log.step(4 + model_idx, 10, f"Testing Model: {model_name}")
        
        try:
            # Load model
            log.progress(f"Loading model with {config.quantization} quantization...")
            loader = ModelLoader(
                model_name=model_name,
                device=config.device,
                quantization=config.quantization,
            )
            model, tokenizer = loader.load()
            model_config = get_model_config(model_name)
            log.success("Model loaded")
            
            # Get memory usage
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                log.info(f"GPU memory used: {mem_gb:.2f} GB")
            
            # Run baseline
            baseline_result = step_run_baseline_evaluation(
                log, config, model, tokenizer, model_name,
                advbench_prompts, alpaca_prompts
            )
            all_results["main_results"].append(baseline_result)
            
            # Run NSKI (REAL)
            nski_result = step_run_nski_evaluation(
                log, config, model, tokenizer, model_name, model_config,
                advbench_prompts, alpaca_prompts
            )
            all_results["main_results"].append(nski_result)
            
            # Run baseline methods comparison
            baseline_method_results = step_run_baseline_methods(
                log, config, model, tokenizer, model_name, model_config,
                advbench_prompts, alpaca_prompts
            )
            all_results["main_results"].extend(baseline_method_results)
            
            # Run ablation study (only for first model to save time)
            if model_idx == 0:
                log.step(7, 10, "Running Ablation Studies")
                ablation_results = step_run_ablation_study(
                    log, config, model, tokenizer, model_name, model_config,
                    advbench_prompts, alpaca_prompts
                )
                all_results["ablation_results"].extend(ablation_results)
            
            # Cleanup
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            log.success(f"Completed {model_name}")
            
        except Exception as e:
            log.error(f"Failed on {model_name}: {e}")
            traceback.print_exc()
            continue
    
    # Generate figures
    step_generate_figures(log, config, all_results)
    
    # Create summary
    step_create_summary(log, config, all_results)
    
    # Final stats
    elapsed = time.time() - start_time
    log.info("")
    log.info(f"Total time: {elapsed/60:.1f} minutes")
    
    return all_results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="NSKI Full Pipeline - Real Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Models to test (default: GPT-2-XL, Llama-2-7B, Mistral-7B, Llama-3-8B)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=520,
        help="Number of AdvBench samples (default: 520 = full)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=2.0,
        help="NSKI intervention strength (default: 2.0)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization level (default: 4bit)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test only GPT-2-XL with 50 samples",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        output_dir=args.output,
        n_advbench_samples=args.n_samples,
        nski_strength=args.strength,
        quantization=args.quantization,
    )
    
    if args.models:
        config.models = args.models
    
    if args.quick:
        config.models = ["gpt2-xl"]
        config.n_advbench_samples = 50
        config.n_alpaca_samples = 50
        print("Quick mode: Testing GPT-2-XL with 50 samples")
    
    # Run pipeline
    results = run_pipeline(config)
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
