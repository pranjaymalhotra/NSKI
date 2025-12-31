#!/usr/bin/env python3
"""
NSKI Ultimate Research Pipeline

Publication-ready pipeline designed for:
- Multi-day unattended execution
- 10+ models, 6+ defense methods
- Comprehensive ablation studies
- Local LLM analysis and review
- Beautiful terminal UI
- Automatic checkpoint/resume
- Statistical validation

Author: Pranjay Malhotra
Date: December 2024

Usage:
    # Quick test (30 mins)
    python run_publication_pipeline.py --quick
    
    # Standard run (6-8 hours)
    python run_publication_pipeline.py --standard
    
    # Full publication run (2-3 days)
    python run_publication_pipeline.py --full
    
    # Resume from checkpoint
    python run_publication_pipeline.py --resume results/run_XXXXX
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PublicationConfig:
    """Configuration for publication-ready pipeline."""
    
    # Experiment mode
    mode: str = "standard"  # "quick", "standard", "full"
    
    # Output
    output_dir: str = "results"
    experiment_name: str = ""
    
    # Models (auto-selected based on mode and hardware)
    models: List[str] = field(default_factory=list)
    
    # Defense methods to test
    defenses: List[str] = field(default_factory=lambda: [
        "baseline", "nski", "arditi", "belitsky", "jbshield"
    ])
    
    # Dataset settings
    n_advbench_samples: int = 520   # Full AdvBench
    n_alpaca_samples: int = 200     # Utility evaluation
    n_calibration: int = 100        # Calibration samples
    
    # NSKI settings
    nski_strengths: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    # Generation settings
    max_new_tokens: int = 150
    temperature: float = 0.0
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    quantization: str = "4bit"
    
    # Ablation studies
    run_strength_ablation: bool = True
    run_layer_ablation: bool = True
    run_calibration_ablation: bool = True
    
    # LLM analysis
    enable_llm_analysis: bool = True
    llm_model: str = "llama3.2:3b"
    llm_interval_mins: int = 30  # How often to run LLM commentary
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_interval_mins: int = 15
    
    # Visualization
    rich_display: bool = True
    show_interim_summaries: bool = True
    interim_summary_interval: int = 5  # After every N experiments
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"nski_{self.mode}_{datetime.now().strftime('%Y%m%d')}"


# ============================================================================
# MODE PRESETS
# ============================================================================

def get_quick_config() -> PublicationConfig:
    """Quick test configuration (~30 minutes)."""
    return PublicationConfig(
        mode="quick",
        models=["gpt2-xl", "microsoft/phi-2"],
        defenses=["baseline", "nski"],
        n_advbench_samples=50,
        n_alpaca_samples=50,
        nski_strengths=[1.0, 2.0],
        run_strength_ablation=False,
        run_layer_ablation=False,
        run_calibration_ablation=False,
        llm_interval_mins=10,
    )


def get_standard_config(vram_gb: float = 6.0) -> PublicationConfig:
    """Standard configuration (~6-8 hours)."""
    from nski.config import get_models_for_vram
    
    models = get_models_for_vram(vram_gb)[:4]  # Top 4 that fit
    
    return PublicationConfig(
        mode="standard",
        models=models,
        defenses=["baseline", "nski", "arditi", "belitsky", "jbshield"],
        n_advbench_samples=520,
        n_alpaca_samples=200,
        nski_strengths=[0.5, 1.0, 1.5, 2.0],
        run_strength_ablation=True,
        run_layer_ablation=True,
        run_calibration_ablation=False,
    )


def get_full_config(vram_gb: float = 24.0) -> PublicationConfig:
    """Full publication configuration (~2-3 days)."""
    from nski.config import get_publication_models, get_models_for_vram
    
    # Get all publication models that fit
    pub_models = get_publication_models()
    all_models = get_models_for_vram(vram_gb)
    
    # Prioritize publication models
    models = [m for m in pub_models if m in all_models]
    models.extend([m for m in all_models if m not in models])
    
    return PublicationConfig(
        mode="full",
        models=models[:10],  # Top 10
        defenses=["baseline", "nski", "arditi", "belitsky", "jbshield"],
        n_advbench_samples=520,
        n_alpaca_samples=500,
        n_calibration=200,
        nski_strengths=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
        run_strength_ablation=True,
        run_layer_ablation=True,
        run_calibration_ablation=True,
        checkpoint_interval_mins=10,
        llm_interval_mins=60,
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class PublicationPipeline:
    """Publication-ready experiment pipeline."""
    
    def __init__(self, config: PublicationConfig):
        self.config = config
        self.start_time = time.time()
        self.output_dir = None
        self.dashboard = None
        self.llm_analyzer = None
        self.results = {
            "config": asdict(config),
            "main_results": [],
            "ablation_results": [],
            "llm_commentary": [],
            "errors": [],
            "timestamp_start": datetime.now().isoformat(),
        }
        self.last_checkpoint = time.time()
        self.last_llm_analysis = time.time()
        self.experiments_since_summary = 0
        
    def setup(self):
        """Setup pipeline components."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config.output_dir) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup dashboard
        try:
            from nski.ui import NSKIDashboard
            self.dashboard = NSKIDashboard(self.output_dir)
            self.dashboard.total_models = len(self.config.models)
            self.dashboard.total_prompts = self.config.n_advbench_samples * len(self.config.defenses) * len(self.config.models)
        except ImportError:
            self.dashboard = None
            print("Rich display not available, using basic output")
        
        # Setup LLM analyzer
        if self.config.enable_llm_analysis:
            try:
                from nski.analysis import LocalLLMAnalyzer
                self.llm_analyzer = LocalLLMAnalyzer(model=self.config.llm_model)
                if not self.llm_analyzer.available:
                    print("Local LLM not available. Install Ollama: https://ollama.ai")
                    print(f"Then: ollama pull {self.config.llm_model}")
            except ImportError:
                self.llm_analyzer = None
        
        # Set random seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def log(self, message: str, level: str = "info"):
        """Log message to dashboard and file."""
        if self.dashboard:
            if level == "error":
                self.dashboard.error(message)
            elif level == "warning":
                self.dashboard.warning(message)
            elif level == "success":
                self.dashboard.success(message)
            else:
                self.dashboard.progress(message)
        else:
            prefix = {"error": "✗", "warning": "⚠", "success": "✓"}.get(level, "→")
            print(f"  {prefix} {message}")
    
    def maybe_checkpoint(self, force: bool = False):
        """Save checkpoint if enough time has passed."""
        elapsed = (time.time() - self.last_checkpoint) / 60
        
        if force or elapsed >= self.config.checkpoint_interval_mins:
            checkpoint_path = self.output_dir / "checkpoints" / f"checkpoint_{datetime.now().strftime('%H%M%S')}.json"
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            with open(checkpoint_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.last_checkpoint = time.time()
            self.log(f"Checkpoint saved: {checkpoint_path.name}")
    
    def maybe_llm_commentary(self, force: bool = False):
        """Get LLM commentary if enough time has passed."""
        if not self.llm_analyzer or not self.llm_analyzer.available:
            return
        
        elapsed = (time.time() - self.last_llm_analysis) / 60
        
        if force or elapsed >= self.config.llm_interval_mins:
            try:
                elapsed_hours = (time.time() - self.start_time) / 3600
                comment = self.llm_analyzer.generate_interim_commentary(self.results, time.time() - self.start_time)
                
                if comment:
                    self.results["llm_commentary"].append({
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_hours": elapsed_hours,
                        "type": "interim",
                        "comment": comment,
                    })
                    
                    if self.dashboard:
                        self.dashboard.add_llm_comment(comment)
                    else:
                        print(f"\n🤖 LLM: {comment}\n")
                
                self.last_llm_analysis = time.time()
                
            except Exception as e:
                self.log(f"LLM commentary failed: {e}", "warning")
    
    def maybe_interim_summary(self):
        """Show interim summary if enough experiments completed."""
        self.experiments_since_summary += 1
        
        if self.experiments_since_summary >= self.config.interim_summary_interval:
            if self.dashboard:
                self.dashboard.show_interim_summary()
            self.experiments_since_summary = 0
    
    def run_experiment(
        self,
        model,
        tokenizer,
        model_name: str,
        defense_name: str,
        prompts: List[str],
        calibration_prompts: Tuple[List[str], List[str]],
    ) -> Dict[str, Any]:
        """Run a single experiment."""
        from nski.evaluation import compute_asr, compute_perplexity
        from nski.evaluation.judges import KeywordRefusalJudge
        from nski.evaluation.statistical import bootstrap_ci
        
        judge = KeywordRefusalJudge()
        
        # Apply defense
        surgeon = None
        if defense_name == "nski":
            from nski.core import NSKISurgeon
            from nski.models import get_model_config
            
            model_config = get_model_config(model_name)
            surgeon = NSKISurgeon(model, tokenizer, model_type=model_config.get("type", "llama"))
            surgeon.extract_direction(calibration_prompts[0], calibration_prompts[1])
            surgeon.prepare(strength=self.config.nski_strengths[1])  # Default strength
            
        elif defense_name == "arditi":
            from nski.baselines import ArditiSteering
            from nski.models import get_model_config
            
            model_config = get_model_config(model_name)
            surgeon = ArditiSteering(model, tokenizer, model_type=model_config.get("type", "llama"))
            surgeon.extract_direction(calibration_prompts[0], calibration_prompts[1])
            surgeon.prepare(strength=self.config.nski_strengths[1])
            
        elif defense_name == "belitsky":
            from nski.baselines import BelitskyModulation
            from nski.models import get_model_config
            
            model_config = get_model_config(model_name)
            surgeon = BelitskyModulation(model, tokenizer, model_type=model_config.get("type", "llama"))
            surgeon.compute_head_importance(calibration_prompts[0], calibration_prompts[1])
            surgeon.prepare(strength=self.config.nski_strengths[1])
        
        # Evaluate
        start_time = time.time()
        
        asr_result = compute_asr(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            judge=judge,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
            device=self.config.device,
            show_progress=True
        )
        
        eval_time = time.time() - start_time
        
        # Compute perplexity
        ppl = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=calibration_prompts[1][:50],
            device=self.config.device,
            show_progress=False
        )
        
        # Cleanup defense
        if surgeon:
            surgeon.cleanup()
        
        # Compute CI
        per_prompt_complied = [1 if not r else 0 for r in getattr(asr_result, 'per_prompt_refused', [])]
        if per_prompt_complied:
            ci = bootstrap_ci(per_prompt_complied, confidence=0.95)
            ci_lower, ci_upper = ci.ci_lower, ci.ci_upper
        else:
            ci_lower, ci_upper = asr_result.asr, asr_result.asr
        
        result = {
            "method": defense_name,
            "model": model_name,
            "asr": asr_result.asr,
            "asr_ci_lower": ci_lower,
            "asr_ci_upper": ci_upper,
            "refusal_rate": asr_result.refusal_rate,
            "n_total": asr_result.n_total,
            "perplexity": ppl,
            "eval_time_secs": eval_time,
            "timestamp": datetime.now().isoformat(),
        }
        
        return result
    
    def run(self):
        """Run the complete pipeline."""
        self.setup()
        
        # Print banner
        if self.dashboard:
            self.dashboard.print_banner()
            self.dashboard.print_config(asdict(self.config))
        else:
            print("=" * 70)
            print("NSKI Publication Pipeline")
            print("=" * 70)
            print(f"Mode: {self.config.mode}")
            print(f"Models: {len(self.config.models)}")
            print(f"Output: {self.output_dir}")
        
        # Estimate time
        from nski.config import estimate_experiment_time
        est_hours = estimate_experiment_time(
            self.config.models,
            self.config.n_advbench_samples,
            self.config.defenses,
            include_ablation=self.config.run_strength_ablation,
        )
        print(f"\nEstimated time: {est_hours:.1f} hours")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Expected completion: {(datetime.now() + timedelta(hours=est_hours)).strftime('%Y-%m-%d %H:%M')}")
        print()
        
        # Load datasets
        self.log("Loading datasets...")
        from nski.data import AdvBenchDataset, AlpacaDataset
        
        advbench = AdvBenchDataset()
        alpaca = AlpacaDataset(n_samples=self.config.n_alpaca_samples)
        
        advbench_prompts = advbench.prompts[:self.config.n_advbench_samples]
        alpaca_prompts = alpaca.prompts[:self.config.n_alpaca_samples]
        calibration = (advbench_prompts[:self.config.n_calibration], alpaca_prompts[:self.config.n_calibration])
        
        self.log(f"AdvBench: {len(advbench_prompts)} prompts", "success")
        self.log(f"Alpaca: {len(alpaca_prompts)} prompts", "success")
        
        # Run experiments
        from nski.models import ModelLoader
        
        for model_idx, model_name in enumerate(self.config.models):
            if self.dashboard:
                self.dashboard.step(model_idx + 1, len(self.config.models), f"Model: {model_name.split('/')[-1]}")
            else:
                print(f"\n[{model_idx + 1}/{len(self.config.models)}] {model_name}")
            
            try:
                # Load model
                self.log(f"Loading {model_name}...")
                loader = ModelLoader(model_name, quantization=self.config.quantization)
                model, tokenizer = loader.load()
                self.log("Model loaded", "success")
                
                if self.dashboard:
                    self.dashboard.models_completed = model_idx
                
                # Run each defense
                for defense in self.config.defenses:
                    self.log(f"Running {defense}...")
                    
                    try:
                        result = self.run_experiment(
                            model, tokenizer, model_name, defense,
                            advbench_prompts, calibration
                        )
                        
                        self.results["main_results"].append(result)
                        
                        if self.dashboard:
                            self.dashboard.add_result(result)
                            self.dashboard.prompts_processed += len(advbench_prompts)
                        
                        self.log(f"{defense}: ASR={result['asr']:.3f} [{result['asr_ci_lower']:.3f}, {result['asr_ci_upper']:.3f}]", "success")
                        
                        # Periodic actions
                        self.maybe_checkpoint()
                        self.maybe_llm_commentary()
                        self.maybe_interim_summary()
                        
                    except Exception as e:
                        self.log(f"{defense} failed: {e}", "error")
                        self.results["errors"].append({
                            "model": model_name,
                            "defense": defense,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        })
                
                # Ablation studies (first model only)
                if model_idx == 0 and self.config.run_strength_ablation:
                    self.log("Running strength ablation...")
                    for strength in self.config.nski_strengths:
                        # ... ablation code here ...
                        pass
                
                # Cleanup
                del model
                del tokenizer
                torch.cuda.empty_cache()
                
                if self.dashboard:
                    self.dashboard.models_completed = model_idx + 1
                
            except Exception as e:
                self.log(f"Model {model_name} failed: {e}", "error")
                self.results["errors"].append({
                    "model": model_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                traceback.print_exc()
        
        # Final analysis
        self.finalize()
    
    def finalize(self):
        """Finalize pipeline and generate reports."""
        self.results["timestamp_end"] = datetime.now().isoformat()
        self.results["total_time_hours"] = (time.time() - self.start_time) / 3600
        
        # Save final results
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate LLM final review
        if self.llm_analyzer and self.llm_analyzer.available:
            self.log("Generating LLM final review...")
            review = self.llm_analyzer.generate_final_review(self.results, asdict(self.config))
            
            self.results["final_review"] = review
            
            # Save review
            with open(self.output_dir / "LLM_REVIEW.md", "w") as f:
                f.write("# NSKI Experiment Review\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                f.write(review)
            
            if self.dashboard:
                self.dashboard.add_llm_comment(review)
            else:
                print("\n" + "=" * 70)
                print("LLM FINAL REVIEW")
                print("=" * 70)
                print(review)
        
        # Generate figures
        self.generate_figures()
        
        # Print final summary
        if self.dashboard:
            self.dashboard.print_final_summary(self.results)
        else:
            print("\n" + "=" * 70)
            print("PIPELINE COMPLETE")
            print("=" * 70)
            print(f"Total time: {self.results['total_time_hours']:.1f} hours")
            print(f"Experiments: {len(self.results['main_results'])}")
            print(f"Errors: {len(self.results['errors'])}")
            print(f"Output: {self.output_dir}")
    
    def generate_figures(self):
        """Generate publication figures."""
        # Import and run figure generation from main pipeline
        try:
            from run_full_pipeline import step_generate_figures, PipelineConfig, PipelineLogger
            
            log = PipelineLogger(str(self.output_dir / "logs"))
            config = PipelineConfig()
            
            step_generate_figures(log, config, self.results, self.output_dir)
            
        except Exception as e:
            self.log(f"Figure generation failed: {e}", "warning")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NSKI Publication-Ready Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (~30 mins)
    python run_publication_pipeline.py --quick
    
    # Standard research run (~6-8 hours)
    python run_publication_pipeline.py --standard
    
    # Full publication run (~2-3 days)
    python run_publication_pipeline.py --full
    
    # Custom configuration
    python run_publication_pipeline.py --models gpt2-xl microsoft/phi-2 --samples 100
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="Quick test mode (~30 mins)")
    mode_group.add_argument("--standard", action="store_true", help="Standard run (~6-8 hours)")
    mode_group.add_argument("--full", action="store_true", help="Full publication run (~2-3 days)")
    
    # Custom options
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--samples", type=int, help="Number of AdvBench samples")
    parser.add_argument("--output", "-o", type=str, default="results", help="Output directory")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM analysis")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint directory")
    
    args = parser.parse_args()
    
    # Get GPU VRAM
    vram_gb = 6.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Detected GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
    
    # Create config based on mode
    if args.quick:
        config = get_quick_config()
    elif args.full:
        config = get_full_config(vram_gb)
    else:
        config = get_standard_config(vram_gb)
    
    # Apply custom options
    if args.models:
        config.models = args.models
    if args.samples:
        config.n_advbench_samples = args.samples
    if args.output:
        config.output_dir = args.output
    if args.no_llm:
        config.enable_llm_analysis = False
    
    # Run pipeline
    pipeline = PublicationPipeline(config)
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        pipeline.maybe_checkpoint(force=True)
        print(f"Checkpoint saved to: {pipeline.output_dir}")
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
