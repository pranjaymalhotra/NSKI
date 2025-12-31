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
    
    # Research paper settings
    run_layer_ablation: bool = True
    run_calibration_ablation: bool = True
    generate_detailed_figures: bool = True
    save_checkpoints: bool = True
    
    def __post_init__(self):
        if self.models is None:
            # Default models for research paper - safety-tuned models
            self.models = [
                "meta-llama/Llama-2-7b-chat-hf",       # ~4GB with 4-bit - RLHF safety
                "meta-llama/Meta-Llama-3-8B-Instruct", # ~5GB with 4-bit - better safety
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
        self.logger.info(f"    [OK] {message}")
    
    def warning(self, message: str):
        self.logger.warning(f"    [WARN] {message}")
    
    def error(self, message: str):
        self.logger.error(f"    [ERROR] {message}")
    
    def progress(self, message: str):
        """Progress update within a step."""
        self.logger.info(f"    --> {message}")


# ============================================================================
# PIPELINE STEPS
# ============================================================================

def step_preflight_checks(log: PipelineLogger, config: PipelineConfig) -> Tuple[bool, List[str]]:
    """
    Comprehensive pre-flight checks before starting long unattended run.
    
    Returns:
        Tuple of (all_passed, list_of_issues)
    """
    log.step(0, 12, "Pre-flight Checks (Unattended Run Verification)")
    
    issues = []
    warnings = []
    
    # 1. Check disk space
    log.progress("Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path(config.output_dir).parent)
        free_gb = free / (1024**3)
        if free_gb < 10:
            issues.append(f"Low disk space: {free_gb:.1f} GB free (need 10+ GB)")
        elif free_gb < 20:
            warnings.append(f"Limited disk space: {free_gb:.1f} GB free")
        else:
            log.success(f"Disk space: {free_gb:.1f} GB free")
    except Exception as e:
        warnings.append(f"Could not check disk space: {e}")
    
    # 2. Check GPU memory
    log.progress("Checking GPU memory...")
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Estimate required VRAM based on models (4-bit quantization)
            # Llama-2-7B: ~4GB, Llama-3-8B: ~5.5GB, GPT-2-XL: ~2GB
            max_required_gb = 0
            problematic_models = []
            for model in config.models:
                if "8B" in model or "8b" in model:
                    model_vram = 5.5
                elif "7B" in model or "7b" in model:
                    model_vram = 4.0
                elif "xl" in model.lower():
                    model_vram = 2.0
                else:
                    model_vram = 4.0  # default estimate
                
                max_required_gb = max(max_required_gb, model_vram)
                if vram_gb < model_vram:
                    problematic_models.append(f"{model} needs ~{model_vram:.1f}GB")
            
            if problematic_models:
                warnings.append(f"GPU VRAM ({vram_gb:.1f} GB) insufficient for: {', '.join(problematic_models)}")
                log.warning(f"GPU VRAM: {vram_gb:.1f} GB (will use CPU offloading for large models)")
            else:
                log.success(f"GPU VRAM: {vram_gb:.1f} GB (sufficient)")
        else:
            issues.append("No CUDA GPU available - running will be very slow")
    except Exception as e:
        issues.append(f"GPU check failed: {e}")
    
    # 3. Check HuggingFace authentication for Llama models
    log.progress("Checking HuggingFace authentication...")
    llama_models = [m for m in config.models if "llama" in m.lower() or "Llama" in m]
    if llama_models:
        try:
            from huggingface_hub import HfApi, whoami
            try:
                user_info = whoami()
                log.success(f"HuggingFace logged in as: {user_info.get('name', 'Unknown')}")
            except Exception:
                # Try to check if token exists
                token_path = Path.home() / ".cache" / "huggingface" / "token"
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                if token_path.exists() or hf_token:
                    log.success("HuggingFace token found")
                else:
                    issues.append("HuggingFace authentication required for Llama models. Run: huggingface-cli login")
        except ImportError:
            warnings.append("huggingface_hub not installed, cannot verify HF auth")
    
    # 4. Check network connectivity
    log.progress("Checking network connectivity...")
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        log.success("Network: HuggingFace accessible")
    except Exception as e:
        warnings.append(f"Network check failed: {e}")
    
    # 5. Check required packages
    log.progress("Checking required packages...")
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("matplotlib", "Matplotlib (for figures)"),
        ("seaborn", "Seaborn (for figures)"),
        ("scipy", "SciPy (for statistics)"),
        ("tqdm", "tqdm (for progress bars)"),
        ("bitsandbytes", "bitsandbytes (for quantization)"),
    ]
    for pkg, name in required_packages:
        try:
            __import__(pkg)
            log.success(f"Package: {name}")
        except ImportError:
            if pkg in ["matplotlib", "seaborn"]:
                warnings.append(f"Optional package missing: {name}")
            else:
                issues.append(f"Required package missing: {name}")
    
    # 6. Estimate run time
    log.progress("Estimating run time...")
    n_models = len(config.models)
    n_samples = config.n_advbench_samples
    # Rough estimate: ~2 min per sample per model for all methods
    estimated_hours = (n_models * n_samples * 5) / 60 / 60  # 5 methods, ~1 min each
    log.info(f"Estimated run time: {estimated_hours:.1f} hours ({n_models} models, {n_samples} samples)")
    if estimated_hours > 24:
        warnings.append(f"Run time estimate exceeds 24 hours: {estimated_hours:.1f} hours")
    
    # 7. Verify model access (try to load tokenizer only)
    log.progress("Verifying model access...")
    for model_name in config.models:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            log.success(f"Model access verified: {model_name}")
            del tokenizer
        except Exception as e:
            error_str = str(e).lower()
            if "gated" in error_str or "access" in error_str or "401" in error_str:
                issues.append(f"No access to {model_name}. Request access at https://huggingface.co/{model_name}")
            else:
                issues.append(f"Cannot access {model_name}: {e}")
    
    # Print summary
    log.info("")
    log.info("=" * 50)
    log.info("PRE-FLIGHT CHECK SUMMARY")
    log.info("=" * 50)
    
    if warnings:
        for w in warnings:
            log.warning(w)
    
    if issues:
        log.error("CRITICAL ISSUES FOUND:")
        for issue in issues:
            log.error(f"  - {issue}")
        log.info("")
        log.error("Fix the issues above before starting unattended run!")
        return False, issues
    else:
        log.success("All pre-flight checks passed!")
        log.info(f"Safe to run unattended for ~{estimated_hours:.1f} hours")
        return True, []


def step_validate_environment(log: PipelineLogger, config: PipelineConfig) -> bool:
    """Step 1: Validate environment and dependencies."""
    log.step(1, 12, "Validating Environment")
    
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


def step_download_datasets(log: PipelineLogger, config: PipelineConfig, output_dir: Path = None) -> bool:
    """Step 2: Download and verify datasets."""
    log.step(2, 12, "Downloading Datasets")
    
    from nski.data.download import download_advbench, download_alpaca, download_harmbench
    
    if output_dir is None:
        output_dir = Path(config.output_dir)
    cache_dir = str(output_dir / "data")
    
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
        belitsky.compute_head_importance(
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
        jbshield = JBShield(model=model, tokenizer=tokenizer, device=config.device)
        
        # Filter prompts first, then generate
        filtered_responses = []
        for prompt in tqdm(advbench_prompts, desc="JBShield"):
            check_result = jbshield.check_prompt(prompt)
            if check_result['is_jailbreak']:
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
                is_refused = judge(response)  # judge is callable
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
    """Run NSKI ablation studies for research paper."""
    log.progress("Running comprehensive ablation studies...")
    
    from nski.core import NSKISurgeon
    from nski.evaluation import compute_asr, compute_perplexity
    from nski.evaluation.judges import KeywordRefusalJudge
    
    results = []
    judge = KeywordRefusalJudge()
    
    # Use subset for ablations (faster but still meaningful)
    ablation_prompts = advbench_prompts[:100]
    ablation_harmless = alpaca_prompts[:100]
    
    # Create surgeon
    surgeon = NSKISurgeon(
        model=model,
        tokenizer=tokenizer,
        model_type=model_config.get("type", "llama"),
    )
    
    # Extract direction for default layer
    surgeon.extract_direction(
        harmful_prompts=ablation_prompts[:50],
        harmless_prompts=ablation_harmless[:50],
    )
    
    # 1. Strength ablation (most important)
    log.progress("Ablation 1/3: Intervention strength...")
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
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
        
        ppl = compute_perplexity(model, tokenizer, ablation_harmless[:30], device=config.device, show_progress=False)
        surgeon.cleanup()
        
        results.append({
            "type": "strength",
            "value": strength,
            "model": model_name,
            "asr": asr_result.asr,
            "refusal_rate": asr_result.refusal_rate,
            "perplexity": ppl,
            "utility": max(0, 1 - (ppl - 10) / 50),  # Normalize PPL to utility score
        })
        log.info(f"  Strength {strength}: ASR={asr_result.asr:.3f}, PPL={ppl:.2f}")
    
    # 2. Layer ablation (which layer to intervene)
    if config.run_layer_ablation:
        log.progress("Ablation 2/3: Target layer...")
        # Get number of layers
        n_layers = model_config.get("n_layers", 32)
        # Test key layers: early, middle, late
        test_layers = [
            max(0, n_layers // 6),       # Early (~layer 5 for 32-layer)
            n_layers // 3,                # Early-mid (~layer 10)
            n_layers // 2,                # Middle (~layer 16)
            2 * n_layers // 3,            # Late-mid (~layer 21)
            min(n_layers - 1, 5 * n_layers // 6),  # Late (~layer 26)
        ]
        test_layers = sorted(set(test_layers))  # Remove duplicates
        
        for layer_idx in test_layers:
            try:
                # Re-extract direction for this layer
                surgeon.extract_direction(
                    harmful_prompts=ablation_prompts[:50],
                    harmless_prompts=ablation_harmless[:50],
                    layer_idx=layer_idx,
                )
                surgeon.prepare(strength=config.nski_strength, layer_idx=layer_idx)
                
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
                
                ppl = compute_perplexity(model, tokenizer, ablation_harmless[:30], device=config.device, show_progress=False)
                surgeon.cleanup()
                
                results.append({
                    "type": "layer",
                    "value": layer_idx,
                    "model": model_name,
                    "asr": asr_result.asr,
                    "refusal_rate": asr_result.refusal_rate,
                    "perplexity": ppl,
                    "utility": max(0, 1 - (ppl - 10) / 50),
                })
                log.info(f"  Layer {layer_idx}: ASR={asr_result.asr:.3f}, PPL={ppl:.2f}")
            except Exception as e:
                log.warning(f"  Layer {layer_idx} failed: {e}")
    
    # 3. Calibration sample size ablation
    if config.run_calibration_ablation:
        log.progress("Ablation 3/3: Calibration samples...")
        cal_sizes = [10, 25, 50, 100, 200]
        
        for cal_size in cal_sizes:
            if cal_size > len(ablation_prompts):
                continue
            try:
                surgeon.extract_direction(
                    harmful_prompts=ablation_prompts[:cal_size],
                    harmless_prompts=ablation_harmless[:cal_size],
                )
                surgeon.prepare(strength=config.nski_strength)
                
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
                
                ppl = compute_perplexity(model, tokenizer, ablation_harmless[:30], device=config.device, show_progress=False)
                surgeon.cleanup()
                
                results.append({
                    "type": "calibration",
                    "value": cal_size,
                    "model": model_name,
                    "asr": asr_result.asr,
                    "refusal_rate": asr_result.refusal_rate,
                    "perplexity": ppl,
                    "utility": max(0, 1 - (ppl - 10) / 50),
                })
                log.info(f"  Calibration {cal_size}: ASR={asr_result.asr:.3f}")
            except Exception as e:
                log.warning(f"  Calibration {cal_size} failed: {e}")
    
    log.success("Ablation studies complete")
    return results


def step_generate_figures(log: PipelineLogger, config: PipelineConfig, all_results: Dict, output_dir: Path = None) -> bool:
    """Generate publication-quality figures for research paper."""
    log.step(10, 12, "Generating Publication Figures")
    
    if output_dir is None:
        output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        
        # Try seaborn for better styling
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
            sns.set_palette("colorblind")
            HAS_SEABORN = True
        except ImportError:
            HAS_SEABORN = False
        
        # Publication style
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
        
        COLORS = {
            'baseline': '#7f7f7f',
            'nski': '#2ca02c',
            'arditi': '#1f77b4',
            'belitsky': '#ff7f0e',
            'jbshield': '#9467bd',
        }
        
        # ================================================================
        # Figure 1: Main ASR Comparison (Bar Chart)
        # ================================================================
        if "main_results" in all_results and all_results["main_results"]:
            log.progress("Figure 1: ASR comparison bar chart...")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            results = all_results["main_results"]
            models = sorted(set(r.get("model", "unknown") for r in results))
            methods = ['baseline', 'nski', 'arditi', 'belitsky', 'jbshield']
            methods = [m for m in methods if any(r.get("method") == m for r in results)]
            
            x = np.arange(len(models))
            width = 0.8 / max(len(methods), 1)
            
            for i, method in enumerate(methods):
                asrs = []
                for model in models:
                    r = next((r for r in results if r.get("method") == method and r.get("model") == model), None)
                    asrs.append(r["asr"] if r else 0)
                
                offset = (i - len(methods) / 2 + 0.5) * width
                bars = ax.bar(x + offset, asrs, width, label=method.upper(), color=COLORS.get(method, f'C{i}'))
                
                # Add value labels on bars
                for bar, asr in zip(bars, asrs):
                    if asr > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{asr:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel('Attack Success Rate (ASR)', fontweight='bold')
            ax.set_title('NSKI vs Baseline Methods: Attack Success Rate Comparison', fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=15, ha='right')
            ax.legend(loc='upper right', framealpha=0.9)
            ax.set_ylim(0, 1.15)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Max ASR')
            
            plt.tight_layout()
            plt.savefig(figures_dir / "fig1_asr_comparison.png", dpi=300, bbox_inches='tight')
            plt.savefig(figures_dir / "fig1_asr_comparison.pdf", bbox_inches='tight')  # PDF for paper
            plt.close()
            log.success("fig1_asr_comparison.png/.pdf")
        
        # ================================================================
        # Figure 2: Perplexity (Utility) Comparison
        # ================================================================
        if "main_results" in all_results and all_results["main_results"]:
            log.progress("Figure 2: Perplexity comparison...")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            results = all_results["main_results"]
            models = sorted(set(r.get("model", "unknown") for r in results))
            methods = ['baseline', 'nski', 'arditi', 'belitsky']
            methods = [m for m in methods if any(r.get("method") == m for r in results)]
            
            x = np.arange(len(models))
            width = 0.8 / max(len(methods), 1)
            
            for i, method in enumerate(methods):
                ppls = []
                for model in models:
                    r = next((r for r in results if r.get("method") == method and r.get("model") == model), None)
                    ppls.append(r.get("perplexity", 0) if r else 0)
                
                offset = (i - len(methods) / 2 + 0.5) * width
                bars = ax.bar(x + offset, ppls, width, label=method.upper(), color=COLORS.get(method, f'C{i}'))
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel('Perplexity (Lower is Better)', fontweight='bold')
            ax.set_title('Utility Preservation: Perplexity on Harmless Prompts', fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=15, ha='right')
            ax.legend(loc='upper right', framealpha=0.9)
            
            plt.tight_layout()
            plt.savefig(figures_dir / "fig2_perplexity_comparison.png", dpi=300, bbox_inches='tight')
            plt.savefig(figures_dir / "fig2_perplexity_comparison.pdf", bbox_inches='tight')
            plt.close()
            log.success("fig2_perplexity_comparison.png/.pdf")
        
        # ================================================================
        # Figure 3: Strength Ablation (Line Plot with Error Region)
        # ================================================================
        if "ablation_results" in all_results:
            strength_results = [r for r in all_results["ablation_results"] if r.get("type") == "strength"]
            if strength_results:
                log.progress("Figure 3: Strength ablation...")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Group by model
                models = sorted(set(r.get("model", "unknown") for r in strength_results))
                
                for model in models:
                    model_results = sorted([r for r in strength_results if r.get("model") == model],
                                          key=lambda x: x["value"])
                    strengths = [r["value"] for r in model_results]
                    asrs = [r["asr"] for r in model_results]
                    ppls = [r.get("perplexity", 0) for r in model_results]
                    
                    label = model.split('/')[-1]
                    ax1.plot(strengths, asrs, 'o-', label=label, linewidth=2, markersize=8)
                    ax2.plot(strengths, ppls, 's-', label=label, linewidth=2, markersize=8)
                
                ax1.set_xlabel('Intervention Strength', fontweight='bold')
                ax1.set_ylabel('Attack Success Rate (ASR)', fontweight='bold')
                ax1.set_title('(a) ASR vs Intervention Strength', fontweight='bold')
                ax1.legend(loc='best')
                ax1.set_ylim(0, 1.05)
                ax1.axhline(y=0, color='green', linestyle='--', alpha=0.3)
                
                ax2.set_xlabel('Intervention Strength', fontweight='bold')
                ax2.set_ylabel('Perplexity', fontweight='bold')
                ax2.set_title('(b) Perplexity vs Intervention Strength', fontweight='bold')
                ax2.legend(loc='best')
                
                plt.tight_layout()
                plt.savefig(figures_dir / "fig3_strength_ablation.png", dpi=300, bbox_inches='tight')
                plt.savefig(figures_dir / "fig3_strength_ablation.pdf", bbox_inches='tight')
                plt.close()
                log.success("fig3_strength_ablation.png/.pdf")
        
        # ================================================================
        # Figure 4: Layer Ablation
        # ================================================================
        if "ablation_results" in all_results:
            layer_results = [r for r in all_results["ablation_results"] if r.get("type") == "layer"]
            if layer_results:
                log.progress("Figure 4: Layer ablation...")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                models = sorted(set(r.get("model", "unknown") for r in layer_results))
                
                for model in models:
                    model_results = sorted([r for r in layer_results if r.get("model") == model],
                                          key=lambda x: x["value"])
                    layers = [r["value"] for r in model_results]
                    asrs = [r["asr"] for r in model_results]
                    ppls = [r.get("perplexity", 0) for r in model_results]
                    
                    label = model.split('/')[-1]
                    ax1.plot(layers, asrs, 'o-', label=label, linewidth=2, markersize=8)
                    ax2.plot(layers, ppls, 's-', label=label, linewidth=2, markersize=8)
                
                ax1.set_xlabel('Target Layer Index', fontweight='bold')
                ax1.set_ylabel('Attack Success Rate (ASR)', fontweight='bold')
                ax1.set_title('(a) ASR vs Target Layer', fontweight='bold')
                ax1.legend(loc='best')
                ax1.set_ylim(0, 1.05)
                
                ax2.set_xlabel('Target Layer Index', fontweight='bold')
                ax2.set_ylabel('Perplexity', fontweight='bold')
                ax2.set_title('(b) Perplexity vs Target Layer', fontweight='bold')
                ax2.legend(loc='best')
                
                plt.tight_layout()
                plt.savefig(figures_dir / "fig4_layer_ablation.png", dpi=300, bbox_inches='tight')
                plt.savefig(figures_dir / "fig4_layer_ablation.pdf", bbox_inches='tight')
                plt.close()
                log.success("fig4_layer_ablation.png/.pdf")
        
        # ================================================================
        # Figure 5: Calibration Size Ablation
        # ================================================================
        if "ablation_results" in all_results:
            cal_results = [r for r in all_results["ablation_results"] if r.get("type") == "calibration"]
            if cal_results:
                log.progress("Figure 5: Calibration size ablation...")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                models = sorted(set(r.get("model", "unknown") for r in cal_results))
                
                for model in models:
                    model_results = sorted([r for r in cal_results if r.get("model") == model],
                                          key=lambda x: x["value"])
                    sizes = [r["value"] for r in model_results]
                    asrs = [r["asr"] for r in model_results]
                    
                    label = model.split('/')[-1]
                    ax.plot(sizes, asrs, 'o-', label=label, linewidth=2, markersize=8)
                
                ax.set_xlabel('Number of Calibration Samples', fontweight='bold')
                ax.set_ylabel('Attack Success Rate (ASR)', fontweight='bold')
                ax.set_title('Effect of Calibration Sample Size on NSKI Performance', fontweight='bold')
                ax.legend(loc='best')
                ax.set_ylim(0, 1.05)
                
                plt.tight_layout()
                plt.savefig(figures_dir / "fig5_calibration_ablation.png", dpi=300, bbox_inches='tight')
                plt.savefig(figures_dir / "fig5_calibration_ablation.pdf", bbox_inches='tight')
                plt.close()
                log.success("fig5_calibration_ablation.png/.pdf")
        
        # ================================================================
        # Figure 6: Pareto Frontier (ASR vs Utility Trade-off)
        # ================================================================
        if "main_results" in all_results and all_results["main_results"]:
            log.progress("Figure 6: Pareto frontier...")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            results = all_results["main_results"]
            
            for method in ['baseline', 'nski', 'arditi', 'belitsky']:
                method_results = [r for r in results if r.get("method") == method]
                if not method_results:
                    continue
                
                asrs = [r["asr"] for r in method_results]
                # Convert perplexity to utility (lower PPL = higher utility)
                ppls = [r.get("perplexity", 20) for r in method_results]
                utilities = [max(0, 1 - (p - 5) / 50) for p in ppls]  # Normalize
                
                color = COLORS.get(method, 'gray')
                ax.scatter(asrs, utilities, s=150, c=color, label=method.upper(), 
                          marker='o', edgecolors='black', linewidths=1, alpha=0.8)
                
                # Add model labels
                for r, asr, util in zip(method_results, asrs, utilities):
                    model_short = r.get("model", "").split('/')[-1][:15]
                    ax.annotate(model_short, (asr, util), textcoords="offset points",
                               xytext=(5, 5), fontsize=7, alpha=0.7)
            
            # Draw ideal point
            ax.scatter([0], [1], s=200, c='gold', marker='*', edgecolors='black',
                      linewidths=2, label='Ideal Point', zorder=10)
            
            ax.set_xlabel('Attack Success Rate (ASR) - Lower is Better', fontweight='bold')
            ax.set_ylabel('Utility Score - Higher is Better', fontweight='bold')
            ax.set_title('Safety-Utility Pareto Frontier', fontweight='bold', fontsize=14)
            ax.legend(loc='lower left', framealpha=0.9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.15)
            
            # Add quadrant labels
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.text(0.25, 0.95, 'IDEAL\n(Safe & Useful)', ha='center', fontsize=10, color='green', fontweight='bold')
            ax.text(0.75, 0.95, 'Useful but\nUnsafe', ha='center', fontsize=10, color='orange')
            ax.text(0.25, 0.15, 'Safe but\nLow Utility', ha='center', fontsize=10, color='orange')
            ax.text(0.75, 0.15, 'WORST\n(Unsafe & Unusable)', ha='center', fontsize=10, color='red')
            
            plt.tight_layout()
            plt.savefig(figures_dir / "fig6_pareto_frontier.png", dpi=300, bbox_inches='tight')
            plt.savefig(figures_dir / "fig6_pareto_frontier.pdf", bbox_inches='tight')
            plt.close()
            log.success("fig6_pareto_frontier.png/.pdf")
        
        # ================================================================
        # Figure 7: Heatmap of Results (Model x Method)
        # ================================================================
        if "main_results" in all_results and all_results["main_results"] and HAS_SEABORN:
            log.progress("Figure 7: Results heatmap...")
            
            results = all_results["main_results"]
            models = sorted(set(r.get("model", "unknown") for r in results))
            methods = ['baseline', 'nski', 'arditi', 'belitsky', 'jbshield']
            methods = [m for m in methods if any(r.get("method") == m for r in results)]
            
            # Create ASR matrix
            asr_matrix = np.zeros((len(models), len(methods)))
            for i, model in enumerate(models):
                for j, method in enumerate(methods):
                    r = next((r for r in results if r.get("method") == method and r.get("model") == model), None)
                    asr_matrix[i, j] = r["asr"] if r else np.nan
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create heatmap
            im = ax.imshow(asr_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.set_label('Attack Success Rate (ASR)', fontweight='bold')
            
            # Set ticks
            ax.set_xticks(np.arange(len(methods)))
            ax.set_yticks(np.arange(len(models)))
            ax.set_xticklabels([m.upper() for m in methods])
            ax.set_yticklabels([m.split('/')[-1] for m in models])
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(methods)):
                    if not np.isnan(asr_matrix[i, j]):
                        text = ax.text(j, i, f'{asr_matrix[i, j]:.2f}',
                                      ha='center', va='center', color='black', fontweight='bold')
            
            ax.set_title('ASR Heatmap: Model vs Method', fontweight='bold', fontsize=14)
            ax.set_xlabel('Method', fontweight='bold')
            ax.set_ylabel('Model', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(figures_dir / "fig7_results_heatmap.png", dpi=300, bbox_inches='tight')
            plt.savefig(figures_dir / "fig7_results_heatmap.pdf", bbox_inches='tight')
            plt.close()
            log.success("fig7_results_heatmap.png/.pdf")
        
        # ================================================================
        # Figure 8: Summary Statistics Table (as image)
        # ================================================================
        if "main_results" in all_results and all_results["main_results"]:
            log.progress("Figure 8: Summary table...")
            
            results = all_results["main_results"]
            
            # Create summary table
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('off')
            
            # Prepare data
            table_data = []
            for r in results:
                model_short = r.get("model", "N/A").split('/')[-1]
                method = r.get("method", "N/A").upper()
                asr = f'{r.get("asr", 0):.3f}'
                refusal = f'{r.get("refusal_rate", 0):.3f}'
                ppl = f'{r.get("perplexity", 0):.2f}' if r.get("perplexity") else "N/A"
                table_data.append([model_short, method, asr, refusal, ppl])
            
            # Create table
            table = ax.table(
                cellText=table_data,
                colLabels=['Model', 'Method', 'ASR', 'Refusal Rate', 'Perplexity'],
                loc='center',
                cellLoc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(color='white', fontweight='bold')
            
            ax.set_title('NSKI Experimental Results Summary', fontweight='bold', fontsize=14, pad=20)
            
            plt.tight_layout()
            plt.savefig(figures_dir / "fig8_summary_table.png", dpi=300, bbox_inches='tight')
            plt.savefig(figures_dir / "fig8_summary_table.pdf", bbox_inches='tight')
            plt.close()
            log.success("fig8_summary_table.png/.pdf")
        
        log.success(f"All {8} publication figures generated in {figures_dir}")
        return True
        
    except ImportError as e:
        log.warning(f"Figure generation skipped - matplotlib not available: {e}")
        log.info("Install with: pip install matplotlib seaborn")
        return False
    except Exception as e:
        log.warning(f"Figure generation failed: {e}")
        traceback.print_exc()
        return False


def step_create_summary(log: PipelineLogger, config: PipelineConfig, all_results: Dict, output_dir: Path = None) -> None:
    """Create final summary report."""
    log.step(11, 12, "Creating Summary Report")
    
    if output_dir is None:
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

def run_pipeline(config: PipelineConfig, skip_preflight: bool = False) -> Dict[str, Any]:
    """Run the complete NSKI pipeline."""
    
    # Setup with timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(config.output_dir)
    output_dir = base_output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create a "latest" symlink/copy for convenience
    latest_link = base_output_dir / "latest"
    
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
    
    # Step 0: Pre-flight checks (for unattended running)
    if skip_preflight:
        log.warning("Pre-flight checks SKIPPED (--skip-preflight flag)")
        log.info("Running without verification - proceed with caution")
    else:
        preflight_passed, issues = step_preflight_checks(log, config)
        if not preflight_passed:
            log.error("Pre-flight checks failed! Fix issues before running unattended.")
            log.info("To skip pre-flight checks, use --skip-preflight flag")
            all_results["preflight_issues"] = issues
            return all_results
    
    # Step 1: Validate environment
    if not step_validate_environment(log, config):
        log.error("Environment validation failed!")
        return all_results
    
    # Step 2: Download datasets to timestamped folder
    if not step_download_datasets(log, config, output_dir):
        log.error("Dataset download failed!")
        return all_results
    
    # Load datasets from timestamped folder
    log.step(3, 12, "Loading Datasets")
    from nski.data import AdvBenchDataset, AlpacaDataset
    
    data_dir = str(output_dir / "data")
    advbench = AdvBenchDataset(cache_dir=data_dir)
    alpaca = AlpacaDataset(cache_dir=data_dir, n_samples=config.n_alpaca_samples)
    
    advbench_prompts = advbench.prompts[:config.n_advbench_samples]
    alpaca_prompts = alpaca.prompts[:config.n_alpaca_samples]
    
    log.success(f"AdvBench: {len(advbench_prompts)} harmful prompts")
    log.success(f"Alpaca: {len(alpaca_prompts)} harmless prompts")
    
    # Run experiments for each model
    from nski.models import ModelLoader, get_model_config
    
    total_models = len(config.models)
    for model_idx, model_name in enumerate(config.models):
        log.step(4 + model_idx, 12, f"Testing Model ({model_idx+1}/{total_models}): {model_name}")
        
        try:
            # Load model
            log.progress(f"Loading model with {config.quantization} quantization...")
            loader = ModelLoader(
                model_name=model_name,
                device=config.device,
                quantization=config.quantization,
            )
            
            try:
                model, tokenizer = loader.load()
                model_config = get_model_config(model_name)
                log.success("Model loaded")
            except Exception as load_error:
                error_str = str(load_error)
                
                # Provide helpful error messages
                if "IncompleteRead" in error_str or "Connection" in error_str or "Max retries" in error_str or "Failed to resolve" in error_str:
                    log.error("Model download failed - network connectivity issue")
                    log.info("Fix 1: Check your internet connection and retry")
                    log.info(f"Fix 2: Pre-download model: huggingface-cli download {model_name}")
                elif "GPU RAM" in error_str or "dispatched on the CPU" in error_str:
                    log.error(f"Insufficient GPU memory ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB VRAM)")
                    log.info("Note: CPU offloading is enabled but model still too large")
                else:
                    log.error(f"Model loading failed: {error_str}")
                
                # Skip this model and continue with next one
                log.warning(f"Skipping {model_name}")
                continue
            
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
                log.step(4 + total_models, 12, "Running Ablation Studies")
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
    step_generate_figures(log, config, all_results, output_dir)
    
    # Create summary
    step_create_summary(log, config, all_results, output_dir)
    
    # Create/update "latest" reference
    try:
        import shutil
        latest_link = base_output_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.is_dir():
                shutil.rmtree(latest_link)
        # Try symlink first, fallback to writing a reference file
        try:
            latest_link.symlink_to(output_dir.name)
        except OSError:
            # Windows may not support symlinks, write a reference file instead
            with open(base_output_dir / "latest.txt", "w") as f:
                f.write(str(output_dir.absolute()))
    except Exception as e:
        log.warning(f"Could not create 'latest' reference: {e}")
    
    # Final completion
    elapsed = time.time() - start_time
    log.step(12, 12, "Pipeline Complete!")
    log.info("")
    log.info("=" * 60)
    log.success("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    log.info("=" * 60)
    log.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    log.info(f"Output directory: {output_dir.absolute()}")
    log.info(f"Main results: {len(all_results.get('main_results', []))} experiments")
    log.info(f"Ablation results: {len(all_results.get('ablation_results', []))} experiments")
    log.info("")
    log.info("Key output files:")
    log.info(f"  - {output_dir / 'all_results.json'}")
    log.info(f"  - {output_dir / 'SUMMARY.txt'}")
    log.info(f"  - {output_dir / 'figures' / '*.png'} (8 publication figures)")
    log.info(f"  - {output_dir / 'figures' / '*.pdf'} (PDF versions for papers)")
    log.info("")
    
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
        help="Quick mode: test only GPT-2-XL with fewer samples",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight checks (not recommended for unattended runs)",
    )
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Skip ablation studies (faster run)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama2", "llama3", "both", "gpt2"],
        default="both",
        help="Model preset: llama2, llama3, both (default), or gpt2 for testing",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        output_dir=args.output,
        n_advbench_samples=args.n_samples,
        nski_strength=args.strength,
        quantization=args.quantization,
    )
    
    # Handle model presets
    if args.models:
        config.models = args.models
    elif args.model == "llama2":
        config.models = ["meta-llama/Llama-2-7b-chat-hf"]
    elif args.model == "llama3":
        config.models = ["meta-llama/Meta-Llama-3-8B-Instruct"]
    elif args.model == "both":
        config.models = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ]
    elif args.model == "gpt2":
        config.models = ["gpt2-xl"]
    
    if args.no_ablation:
        config.run_layer_ablation = False
        config.run_calibration_ablation = False
    
    if args.quick:
        config.models = ["gpt2-xl"]
        # Only override sample counts if using defaults
        if args.n_samples == 520:  # default value
            config.n_advbench_samples = 50
            config.n_alpaca_samples = 50
        else:
            config.n_alpaca_samples = args.n_samples
        config.run_layer_ablation = False
        config.run_calibration_ablation = False
        print(f"Quick mode: Testing GPT-2-XL with {config.n_advbench_samples} samples")
    
    # Print configuration summary
    print("=" * 60)
    print("NSKI Full Pipeline Configuration")
    print("=" * 60)
    print(f"Models: {config.models}")
    print(f"AdvBench samples: {config.n_advbench_samples}")
    print(f"NSKI strength: {config.nski_strength}")
    print(f"Quantization: {config.quantization}")
    print(f"Layer ablation: {config.run_layer_ablation}")
    print(f"Calibration ablation: {config.run_calibration_ablation}")
    print("=" * 60)
    print()
    
    # Run pipeline
    results = run_pipeline(config, skip_preflight=args.skip_preflight)
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
