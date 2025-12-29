"""
Main comparison experiment: NSKI vs baselines on full AdvBench.

This script runs the complete evaluation pipeline:
1. Load model with quantization
2. Extract refusal direction
3. Apply NSKI surgery
4. Evaluate on AdvBench (520 prompts) for ASR
5. Evaluate on Alpaca subset for utility
6. Compare against baseline methods
7. Compute perplexity
8. Generate statistical analysis
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import yaml
from tqdm import tqdm

from ..core import (
    RefusalDirectionExtractor,
    NSKISurgeon,
    set_seed,
    get_memory_usage,
    setup_logging,
)
from ..data import (
    DatasetDownloader,
    AdvBenchDataset,
    AlpacaDataset,
)
from ..models import ModelLoader, get_model_config
from ..evaluation import (
    compute_asr,
    compute_utility,
    compute_perplexity,
    compute_latency,
    KeywordJudge,
    ClassifierJudge,
    StatisticalAnalyzer,
)
from ..baselines import ArditiSteering, BelitskyModulation, JBShield


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    model_name: str
    method: str
    asr: float
    asr_ci_lower: float
    asr_ci_upper: float
    utility: float
    utility_ci_lower: float
    utility_ci_upper: float
    perplexity: float
    latency_mean: float
    latency_std: float
    memory_mb: float
    n_harmful: int
    n_harmless: int
    timestamp: str


class MainComparisonExperiment:
    """
    Main comparison experiment runner.
    
    Compares NSKI against baseline methods on full datasets.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize experiment.
        
        Args:
            config_path: Path to YAML config file
            output_dir: Directory for results
        """
        # Load config
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Setup output directory
        self.output_dir = Path(output_dir or self.config.get("output", {}).get("results_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(
            log_dir=str(self.output_dir / "logs"),
            level=logging.INFO,
        )
        
        # Set seed
        set_seed(self.config.get("experiment", {}).get("seed", 42))
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.results: List[ExperimentResult] = []
        
        # Initialize judges
        self.keyword_judge = KeywordJudge()
        self.classifier_judge = None  # Lazy load
        
        # Statistical analyzer
        self.stats = StatisticalAnalyzer()
        
        logger.info(f"Initialized MainComparisonExperiment")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "experiment": {
                "name": "nski_main_comparison",
                "seed": 42,
                "models": ["llama3-8b"],
                "methods": ["baseline", "nski", "arditi", "belitsky", "jbshield"],
            },
            "model": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "dtype": "float16",
                "quantization": "4bit",
                "trust_remote_code": True,
            },
            "nski": {
                "n_calibration": 100,
                "strength": 2.0,
                "target_layers": "all",
                "target_components": ["attention", "mlp"],
            },
            "data": {
                "advbench_path": "data/advbench.json",
                "alpaca_path": "data/alpaca.json",
                "n_utility_samples": 200,
            },
            "evaluation": {
                "max_new_tokens": 256,
                "temperature": 0.0,
                "batch_size": 8,
                "judge_type": "keyword",
            },
            "output": {
                "results_dir": "results",
                "save_generations": True,
            },
        }
    
    def download_data(self) -> None:
        """Download required datasets."""
        logger.info("Downloading datasets...")
        downloader = DatasetDownloader(data_dir=str(self.output_dir.parent / "data"))
        
        # Download AdvBench
        advbench_path = downloader.download_advbench()
        logger.info(f"AdvBench downloaded: {advbench_path}")
        
        # Download Alpaca
        alpaca_path = downloader.download_alpaca()
        logger.info(f"Alpaca downloaded: {alpaca_path}")
    
    def load_model(self, model_name: str) -> None:
        """Load model with quantization."""
        logger.info(f"Loading model: {model_name}")
        
        loader = ModelLoader(
            model_name=model_name,
            device=self.config["model"]["device"],
            dtype=self.config["model"]["dtype"],
            quantization=self.config["model"]["quantization"],
            trust_remote_code=self.config["model"]["trust_remote_code"],
        )
        
        self.model, self.tokenizer = loader.load()
        logger.info(f"Model loaded. Memory: {get_memory_usage():.2f} MB")
    
    def load_datasets(self) -> Tuple[AdvBenchDataset, AlpacaDataset]:
        """Load evaluation datasets."""
        logger.info("Loading datasets...")
        
        # Load AdvBench
        advbench = AdvBenchDataset(
            data_path=self.config["data"]["advbench_path"],
        )
        logger.info(f"AdvBench: {len(advbench)} prompts")
        
        # Load Alpaca subset for utility
        alpaca = AlpacaDataset(
            data_path=self.config["data"]["alpaca_path"],
            n_samples=self.config["data"]["n_utility_samples"],
        )
        logger.info(f"Alpaca: {len(alpaca)} prompts")
        
        return advbench, alpaca
    
    def generate_responses(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> List[str]:
        """
        Generate responses for prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for prompt in tqdm(prompts, desc="Generating"):
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                if temperature == 0.0:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)
        
        return responses
    
    def evaluate_asr(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> Tuple[float, float, float]:
        """
        Evaluate attack success rate.
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            
        Returns:
            Tuple of (ASR, CI_lower, CI_upper)
        """
        # Classify responses
        refusals = [self.keyword_judge.is_refusal(r) for r in responses]
        successes = [not r for r in refusals]  # Success = not refused
        
        # Compute ASR with bootstrap CI
        asr, ci_lower, ci_upper = self.stats.bootstrap_ci(
            successes,
            statistic="mean",
        )
        
        return asr, ci_lower, ci_upper
    
    def evaluate_utility(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
    ) -> Tuple[float, float, float]:
        """
        Evaluate utility score.
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            references: Reference responses (optional)
            
        Returns:
            Tuple of (utility, CI_lower, CI_upper)
        """
        # Use response length as proxy if no references
        if references is None:
            # Utility = fraction of non-empty responses
            utilities = [len(r.strip()) > 20 for r in responses]
        else:
            # Compute BLEU-like scores
            utilities = []
            for resp, ref in zip(responses, references):
                if len(resp.strip()) < 10:
                    utilities.append(0.0)
                else:
                    # Simple word overlap
                    resp_words = set(resp.lower().split())
                    ref_words = set(ref.lower().split())
                    overlap = len(resp_words & ref_words) / max(len(ref_words), 1)
                    utilities.append(min(overlap * 2, 1.0))  # Scale up
        
        # Compute with bootstrap CI
        utility, ci_lower, ci_upper = self.stats.bootstrap_ci(
            utilities,
            statistic="mean",
        )
        
        return utility, ci_lower, ci_upper
    
    def run_baseline(
        self,
        model_name: str,
        advbench: AdvBenchDataset,
        alpaca: AlpacaDataset,
    ) -> ExperimentResult:
        """Run baseline (no intervention) evaluation."""
        logger.info("Running baseline (no intervention)...")
        
        # Generate on AdvBench
        start_time = time.time()
        harmful_responses = self.generate_responses(advbench.prompts)
        latency = (time.time() - start_time) / len(advbench.prompts)
        
        # Generate on Alpaca
        harmless_responses = self.generate_responses(alpaca.prompts)
        
        # Evaluate
        asr, asr_ci_l, asr_ci_u = self.evaluate_asr(advbench.prompts, harmful_responses)
        utility, util_ci_l, util_ci_u = self.evaluate_utility(alpaca.prompts, harmless_responses)
        
        # Perplexity
        ppl = compute_perplexity(
            self.model,
            self.tokenizer,
            alpaca.prompts[:50],  # Use subset for speed
        )
        
        return ExperimentResult(
            model_name=model_name,
            method="baseline",
            asr=asr,
            asr_ci_lower=asr_ci_l,
            asr_ci_upper=asr_ci_u,
            utility=utility,
            utility_ci_lower=util_ci_l,
            utility_ci_upper=util_ci_u,
            perplexity=ppl,
            latency_mean=latency,
            latency_std=0.0,
            memory_mb=get_memory_usage(),
            n_harmful=len(advbench),
            n_harmless=len(alpaca),
            timestamp=datetime.now().isoformat(),
        )
    
    def run_nski(
        self,
        model_name: str,
        advbench: AdvBenchDataset,
        alpaca: AlpacaDataset,
    ) -> ExperimentResult:
        """Run NSKI evaluation."""
        logger.info("Running NSKI...")
        
        # Get model config
        model_config = get_model_config(model_name)
        
        # Extract refusal direction
        logger.info("Extracting refusal direction...")
        extractor = RefusalDirectionExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            model_config=model_config,
        )
        
        # Use subset for calibration
        n_cal = self.config["nski"]["n_calibration"]
        harmful_cal = advbench.prompts[:n_cal]
        harmless_cal = alpaca.prompts[:n_cal]
        
        refusal_direction = extractor.extract(
            harmful_prompts=harmful_cal,
            harmless_prompts=harmless_cal,
        )
        logger.info(f"Refusal direction shape: {refusal_direction.shape}")
        
        # Apply NSKI surgery
        logger.info("Applying NSKI surgery...")
        surgeon = NSKISurgeon(
            model=self.model,
            model_config=model_config,
            refusal_direction=refusal_direction,
            strength=self.config["nski"]["strength"],
        )
        surgeon.apply()
        
        # Generate on AdvBench
        start_time = time.time()
        harmful_responses = self.generate_responses(advbench.prompts)
        latency = (time.time() - start_time) / len(advbench.prompts)
        
        # Generate on Alpaca
        harmless_responses = self.generate_responses(alpaca.prompts)
        
        # Evaluate
        asr, asr_ci_l, asr_ci_u = self.evaluate_asr(advbench.prompts, harmful_responses)
        utility, util_ci_l, util_ci_u = self.evaluate_utility(alpaca.prompts, harmless_responses)
        
        # Perplexity
        ppl = compute_perplexity(
            self.model,
            self.tokenizer,
            alpaca.prompts[:50],
        )
        
        # Remove surgery
        surgeon.remove()
        
        return ExperimentResult(
            model_name=model_name,
            method="nski",
            asr=asr,
            asr_ci_lower=asr_ci_l,
            asr_ci_upper=asr_ci_u,
            utility=utility,
            utility_ci_lower=util_ci_l,
            utility_ci_upper=util_ci_u,
            perplexity=ppl,
            latency_mean=latency,
            latency_std=0.0,
            memory_mb=get_memory_usage(),
            n_harmful=len(advbench),
            n_harmless=len(alpaca),
            timestamp=datetime.now().isoformat(),
        )
    
    def run_arditi(
        self,
        model_name: str,
        advbench: AdvBenchDataset,
        alpaca: AlpacaDataset,
    ) -> ExperimentResult:
        """Run Arditi baseline evaluation."""
        logger.info("Running Arditi steering baseline...")
        
        model_config = get_model_config(model_name)
        
        # Initialize and apply
        arditi = ArditiSteering(
            model=self.model,
            tokenizer=self.tokenizer,
            model_config=model_config,
        )
        
        n_cal = self.config["nski"]["n_calibration"]
        arditi.extract_steering_direction(
            harmful_prompts=advbench.prompts[:n_cal],
            harmless_prompts=alpaca.prompts[:n_cal],
        )
        arditi.apply_steering(strength=self.config["nski"]["strength"])
        
        # Generate
        start_time = time.time()
        harmful_responses = self.generate_responses(advbench.prompts)
        latency = (time.time() - start_time) / len(advbench.prompts)
        
        harmless_responses = self.generate_responses(alpaca.prompts)
        
        # Evaluate
        asr, asr_ci_l, asr_ci_u = self.evaluate_asr(advbench.prompts, harmful_responses)
        utility, util_ci_l, util_ci_u = self.evaluate_utility(alpaca.prompts, harmless_responses)
        ppl = compute_perplexity(self.model, self.tokenizer, alpaca.prompts[:50])
        
        arditi.remove_steering()
        
        return ExperimentResult(
            model_name=model_name,
            method="arditi",
            asr=asr,
            asr_ci_lower=asr_ci_l,
            asr_ci_upper=asr_ci_u,
            utility=utility,
            utility_ci_lower=util_ci_l,
            utility_ci_upper=util_ci_u,
            perplexity=ppl,
            latency_mean=latency,
            latency_std=0.0,
            memory_mb=get_memory_usage(),
            n_harmful=len(advbench),
            n_harmless=len(alpaca),
            timestamp=datetime.now().isoformat(),
        )
    
    def run_belitsky(
        self,
        model_name: str,
        advbench: AdvBenchDataset,
        alpaca: AlpacaDataset,
    ) -> ExperimentResult:
        """Run Belitsky baseline evaluation."""
        logger.info("Running Belitsky modulation baseline...")
        
        model_config = get_model_config(model_name)
        
        # Initialize and apply
        belitsky = BelitskyModulation(
            model=self.model,
            tokenizer=self.tokenizer,
            model_config=model_config,
        )
        
        n_cal = self.config["nski"]["n_calibration"]
        belitsky.identify_safety_heads(
            harmful_prompts=advbench.prompts[:n_cal],
            harmless_prompts=alpaca.prompts[:n_cal],
        )
        belitsky.apply_modulation(strength=self.config["nski"]["strength"])
        
        # Generate
        start_time = time.time()
        harmful_responses = self.generate_responses(advbench.prompts)
        latency = (time.time() - start_time) / len(advbench.prompts)
        
        harmless_responses = self.generate_responses(alpaca.prompts)
        
        # Evaluate
        asr, asr_ci_l, asr_ci_u = self.evaluate_asr(advbench.prompts, harmful_responses)
        utility, util_ci_l, util_ci_u = self.evaluate_utility(alpaca.prompts, harmless_responses)
        ppl = compute_perplexity(self.model, self.tokenizer, alpaca.prompts[:50])
        
        belitsky.remove_modulation()
        
        return ExperimentResult(
            model_name=model_name,
            method="belitsky",
            asr=asr,
            asr_ci_lower=asr_ci_l,
            asr_ci_upper=asr_ci_u,
            utility=utility,
            utility_ci_lower=util_ci_l,
            utility_ci_upper=util_ci_u,
            perplexity=ppl,
            latency_mean=latency,
            latency_std=0.0,
            memory_mb=get_memory_usage(),
            n_harmful=len(advbench),
            n_harmless=len(alpaca),
            timestamp=datetime.now().isoformat(),
        )
    
    def run_jbshield(
        self,
        model_name: str,
        advbench: AdvBenchDataset,
        alpaca: AlpacaDataset,
    ) -> ExperimentResult:
        """Run JBShield baseline evaluation."""
        logger.info("Running JBShield baseline...")
        
        # Initialize filter
        jbshield = JBShield()
        
        # Filter harmful prompts
        filtered_prompts = []
        for prompt in advbench.prompts:
            if jbshield.is_jailbreak(prompt):
                filtered_prompts.append("[BLOCKED]")
            else:
                filtered_prompts.append(prompt)
        
        # Generate for non-blocked prompts
        start_time = time.time()
        harmful_responses = []
        for prompt in tqdm(filtered_prompts, desc="JBShield"):
            if prompt == "[BLOCKED]":
                harmful_responses.append("I cannot assist with that request.")
            else:
                resp = self.generate_responses([prompt])[0]
                harmful_responses.append(resp)
        latency = (time.time() - start_time) / len(advbench.prompts)
        
        # Generate harmless normally
        harmless_responses = self.generate_responses(alpaca.prompts)
        
        # Evaluate
        asr, asr_ci_l, asr_ci_u = self.evaluate_asr(advbench.prompts, harmful_responses)
        utility, util_ci_l, util_ci_u = self.evaluate_utility(alpaca.prompts, harmless_responses)
        ppl = compute_perplexity(self.model, self.tokenizer, alpaca.prompts[:50])
        
        return ExperimentResult(
            model_name=model_name,
            method="jbshield",
            asr=asr,
            asr_ci_lower=asr_ci_l,
            asr_ci_upper=asr_ci_u,
            utility=utility,
            utility_ci_lower=util_ci_l,
            utility_ci_upper=util_ci_u,
            perplexity=ppl,
            latency_mean=latency,
            latency_std=0.0,
            memory_mb=get_memory_usage(),
            n_harmful=len(advbench),
            n_harmless=len(alpaca),
            timestamp=datetime.now().isoformat(),
        )
    
    def run(self) -> List[ExperimentResult]:
        """
        Run complete experiment.
        
        Returns:
            List of experiment results
        """
        logger.info("=" * 60)
        logger.info("NSKI Main Comparison Experiment")
        logger.info("=" * 60)
        
        # Download data
        self.download_data()
        
        # Load datasets
        advbench, alpaca = self.load_datasets()
        
        # Run for each model
        models = self.config["experiment"]["models"]
        methods = self.config["experiment"]["methods"]
        
        for model_name in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model_name}")
            logger.info(f"{'='*60}")
            
            # Load model
            self.load_model(model_name)
            
            # Run each method
            for method in methods:
                logger.info(f"\n--- Method: {method} ---")
                
                if method == "baseline":
                    result = self.run_baseline(model_name, advbench, alpaca)
                elif method == "nski":
                    result = self.run_nski(model_name, advbench, alpaca)
                elif method == "arditi":
                    result = self.run_arditi(model_name, advbench, alpaca)
                elif method == "belitsky":
                    result = self.run_belitsky(model_name, advbench, alpaca)
                elif method == "jbshield":
                    result = self.run_jbshield(model_name, advbench, alpaca)
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                self.results.append(result)
                
                # Log result
                logger.info(f"ASR: {result.asr:.3f} [{result.asr_ci_lower:.3f}, {result.asr_ci_upper:.3f}]")
                logger.info(f"Utility: {result.utility:.3f} [{result.utility_ci_lower:.3f}, {result.utility_ci_upper:.3f}]")
                logger.info(f"Perplexity: {result.perplexity:.2f}")
            
            # Free model memory
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self) -> None:
        """Save experiment results."""
        # Save as JSON
        results_json = [asdict(r) for r in self.results]
        json_path = self.output_dir / "main_comparison_results.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"Results saved to: {json_path}")
        
        # Save summary table
        summary_path = self.output_dir / "main_comparison_summary.txt"
        with open(summary_path, "w") as f:
            f.write("NSKI Main Comparison Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Model':<15} {'Method':<12} {'ASR':>10} {'Utility':>10} {'PPL':>10}\n")
            f.write("-" * 80 + "\n")
            for r in self.results:
                f.write(f"{r.model_name:<15} {r.method:<12} {r.asr:>10.3f} {r.utility:>10.3f} {r.perplexity:>10.2f}\n")
        logger.info(f"Summary saved to: {summary_path}")


def run_main_comparison(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> List[ExperimentResult]:
    """
    Run main comparison experiment.
    
    Args:
        config_path: Path to config file
        output_dir: Output directory
        
    Returns:
        List of experiment results
    """
    experiment = MainComparisonExperiment(
        config_path=config_path,
        output_dir=output_dir,
    )
    return experiment.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NSKI Main Comparison Experiment")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    results = run_main_comparison(
        config_path=args.config,
        output_dir=args.output,
    )
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    for r in results:
        print(f"{r.model_name} / {r.method}: ASR={r.asr:.3f}, Utility={r.utility:.3f}")
