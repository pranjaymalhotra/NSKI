"""
Ablation study for NSKI.

This script runs comprehensive ablation experiments:
1. Layer ablation: Which layers contribute most?
2. Strength ablation: Optimal intervention strength
3. Component ablation: Attention vs MLP
4. Calibration ablation: How many samples needed?
5. Strategy ablation: Different projection methods
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from itertools import product

import torch
import yaml
import numpy as np
from tqdm import tqdm

from ..core import (
    RefusalDirectionExtractor,
    NSKISurgeon,
    set_seed,
    get_memory_usage,
    setup_logging,
)
from ..data import AdvBenchDataset, AlpacaDataset
from ..models import ModelLoader, get_model_config
from ..evaluation import (
    compute_asr,
    compute_perplexity,
    KeywordJudge,
    StatisticalAnalyzer,
)


logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Container for ablation results."""
    model_name: str
    ablation_type: str
    ablation_value: Any
    asr: float
    asr_ci_lower: float
    asr_ci_upper: float
    utility: float
    perplexity: float
    timestamp: str


class AblationStudy:
    """
    Comprehensive ablation study runner.
    
    Tests effect of different NSKI parameters.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize ablation study."""
        # Load config
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Setup
        self.output_dir = Path(output_dir or "results/ablations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        setup_logging(str(self.output_dir / "logs"))
        set_seed(self.config.get("seed", 42))
        
        self.results: List[AblationResult] = []
        self.keyword_judge = KeywordJudge()
        self.stats = StatisticalAnalyzer()
        
        logger.info("Initialized AblationStudy")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default ablation config."""
        return {
            "seed": 42,
            "model_name": "llama3-8b",
            "n_harmful": 100,  # Subset for speed
            "n_harmless": 100,
            "n_calibration": 50,
            "ablations": {
                "layer": {
                    "enabled": True,
                    "values": ["early", "middle", "late", "all"],
                },
                "strength": {
                    "enabled": True,
                    "values": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                },
                "component": {
                    "enabled": True,
                    "values": ["attention", "mlp", "both"],
                },
                "calibration": {
                    "enabled": True,
                    "values": [10, 25, 50, 100, 200],
                },
            },
        }
    
    def load_model(self) -> Tuple[Any, Any]:
        """Load model."""
        logger.info(f"Loading model: {self.config['model_name']}")
        
        loader = ModelLoader(
            model_name=self.config["model_name"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            quantization="4bit",
        )
        
        return loader.load()
    
    def load_data(self) -> Tuple[List[str], List[str]]:
        """Load evaluation data."""
        advbench = AdvBenchDataset()
        alpaca = AlpacaDataset(n_samples=self.config["n_harmless"])
        
        harmful = advbench.prompts[:self.config["n_harmful"]]
        harmless = alpaca.prompts[:self.config["n_harmless"]]
        
        return harmful, harmless
    
    def generate_responses(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
    ) -> List[str]:
        """Generate responses."""
        responses = []
        
        for prompt in tqdm(prompts, desc="Generating", leave=False):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)
        
        return responses
    
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        harmful: List[str],
        harmless: List[str],
    ) -> Tuple[float, float, float, float]:
        """Evaluate ASR, utility, perplexity."""
        # Generate harmful responses
        harmful_responses = self.generate_responses(model, tokenizer, harmful)
        
        # Evaluate ASR
        refusals = [self.keyword_judge.is_refusal(r) for r in harmful_responses]
        successes = [not r for r in refusals]
        asr, ci_l, ci_u = self.stats.bootstrap_ci(successes, statistic="mean")
        
        # Generate harmless responses for utility
        harmless_responses = self.generate_responses(model, tokenizer, harmless[:50])
        utility = sum(len(r.strip()) > 20 for r in harmless_responses) / len(harmless_responses)
        
        # Perplexity
        ppl = compute_perplexity(model, tokenizer, harmless[:20])
        
        return asr, ci_l, ci_u, utility, ppl
    
    def run_layer_ablation(
        self,
        model: Any,
        tokenizer: Any,
        harmful: List[str],
        harmless: List[str],
        refusal_direction: torch.Tensor,
    ) -> List[AblationResult]:
        """Ablate over layer selection."""
        logger.info("Running layer ablation...")
        results = []
        
        model_config = get_model_config(self.config["model_name"])
        n_layers = model_config.get("n_layers", 32)
        
        layer_configs = {
            "early": list(range(n_layers // 3)),
            "middle": list(range(n_layers // 3, 2 * n_layers // 3)),
            "late": list(range(2 * n_layers // 3, n_layers)),
            "all": list(range(n_layers)),
        }
        
        for layer_type, layers in layer_configs.items():
            if layer_type not in self.config["ablations"]["layer"]["values"]:
                continue
            
            logger.info(f"  Layer type: {layer_type} ({len(layers)} layers)")
            
            # Apply surgery with specific layers
            surgeon = NSKISurgeon(
                model=model,
                model_config=model_config,
                refusal_direction=refusal_direction,
                strength=2.0,
                target_layers=layers,
            )
            surgeon.apply()
            
            # Evaluate
            asr, ci_l, ci_u, utility, ppl = self.evaluate(model, tokenizer, harmful, harmless)
            
            surgeon.remove()
            
            results.append(AblationResult(
                model_name=self.config["model_name"],
                ablation_type="layer",
                ablation_value=layer_type,
                asr=asr,
                asr_ci_lower=ci_l,
                asr_ci_upper=ci_u,
                utility=utility,
                perplexity=ppl,
                timestamp=datetime.now().isoformat(),
            ))
            
            logger.info(f"    ASR: {asr:.3f}, Utility: {utility:.3f}")
        
        return results
    
    def run_strength_ablation(
        self,
        model: Any,
        tokenizer: Any,
        harmful: List[str],
        harmless: List[str],
        refusal_direction: torch.Tensor,
    ) -> List[AblationResult]:
        """Ablate over intervention strength."""
        logger.info("Running strength ablation...")
        results = []
        
        model_config = get_model_config(self.config["model_name"])
        
        strengths = self.config["ablations"]["strength"]["values"]
        
        for strength in strengths:
            logger.info(f"  Strength: {strength}")
            
            surgeon = NSKISurgeon(
                model=model,
                model_config=model_config,
                refusal_direction=refusal_direction,
                strength=strength,
            )
            surgeon.apply()
            
            asr, ci_l, ci_u, utility, ppl = self.evaluate(model, tokenizer, harmful, harmless)
            
            surgeon.remove()
            
            results.append(AblationResult(
                model_name=self.config["model_name"],
                ablation_type="strength",
                ablation_value=strength,
                asr=asr,
                asr_ci_lower=ci_l,
                asr_ci_upper=ci_u,
                utility=utility,
                perplexity=ppl,
                timestamp=datetime.now().isoformat(),
            ))
            
            logger.info(f"    ASR: {asr:.3f}, Utility: {utility:.3f}")
        
        return results
    
    def run_component_ablation(
        self,
        model: Any,
        tokenizer: Any,
        harmful: List[str],
        harmless: List[str],
        refusal_direction: torch.Tensor,
    ) -> List[AblationResult]:
        """Ablate over component selection."""
        logger.info("Running component ablation...")
        results = []
        
        model_config = get_model_config(self.config["model_name"])
        
        component_configs = {
            "attention": ["attention"],
            "mlp": ["mlp"],
            "both": ["attention", "mlp"],
        }
        
        for comp_type, components in component_configs.items():
            if comp_type not in self.config["ablations"]["component"]["values"]:
                continue
            
            logger.info(f"  Component: {comp_type}")
            
            surgeon = NSKISurgeon(
                model=model,
                model_config=model_config,
                refusal_direction=refusal_direction,
                strength=2.0,
                target_components=components,
            )
            surgeon.apply()
            
            asr, ci_l, ci_u, utility, ppl = self.evaluate(model, tokenizer, harmful, harmless)
            
            surgeon.remove()
            
            results.append(AblationResult(
                model_name=self.config["model_name"],
                ablation_type="component",
                ablation_value=comp_type,
                asr=asr,
                asr_ci_lower=ci_l,
                asr_ci_upper=ci_u,
                utility=utility,
                perplexity=ppl,
                timestamp=datetime.now().isoformat(),
            ))
            
            logger.info(f"    ASR: {asr:.3f}, Utility: {utility:.3f}")
        
        return results
    
    def run_calibration_ablation(
        self,
        model: Any,
        tokenizer: Any,
        harmful: List[str],
        harmless: List[str],
    ) -> List[AblationResult]:
        """Ablate over calibration sample size."""
        logger.info("Running calibration ablation...")
        results = []
        
        model_config = get_model_config(self.config["model_name"])
        cal_sizes = self.config["ablations"]["calibration"]["values"]
        
        for n_cal in cal_sizes:
            logger.info(f"  Calibration samples: {n_cal}")
            
            # Re-extract refusal direction with different sample size
            extractor = RefusalDirectionExtractor(
                model=model,
                tokenizer=tokenizer,
                model_config=model_config,
            )
            
            refusal_dir = extractor.extract(
                harmful_prompts=harmful[:n_cal],
                harmless_prompts=harmless[:n_cal],
            )
            
            surgeon = NSKISurgeon(
                model=model,
                model_config=model_config,
                refusal_direction=refusal_dir,
                strength=2.0,
            )
            surgeon.apply()
            
            asr, ci_l, ci_u, utility, ppl = self.evaluate(model, tokenizer, harmful, harmless)
            
            surgeon.remove()
            
            results.append(AblationResult(
                model_name=self.config["model_name"],
                ablation_type="calibration",
                ablation_value=n_cal,
                asr=asr,
                asr_ci_lower=ci_l,
                asr_ci_upper=ci_u,
                utility=utility,
                perplexity=ppl,
                timestamp=datetime.now().isoformat(),
            ))
            
            logger.info(f"    ASR: {asr:.3f}, Utility: {utility:.3f}")
        
        return results
    
    def run(self) -> List[AblationResult]:
        """Run all ablation studies."""
        logger.info("=" * 60)
        logger.info("NSKI Ablation Study")
        logger.info("=" * 60)
        
        # Load model and data
        model, tokenizer = self.load_model()
        harmful, harmless = self.load_data()
        
        model_config = get_model_config(self.config["model_name"])
        
        # Extract baseline refusal direction
        logger.info("Extracting baseline refusal direction...")
        extractor = RefusalDirectionExtractor(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
        )
        
        n_cal = self.config["n_calibration"]
        refusal_direction = extractor.extract(
            harmful_prompts=harmful[:n_cal],
            harmless_prompts=harmless[:n_cal],
        )
        
        # Run ablations
        ablations = self.config["ablations"]
        
        if ablations["layer"]["enabled"]:
            self.results.extend(self.run_layer_ablation(
                model, tokenizer, harmful, harmless, refusal_direction
            ))
        
        if ablations["strength"]["enabled"]:
            self.results.extend(self.run_strength_ablation(
                model, tokenizer, harmful, harmless, refusal_direction
            ))
        
        if ablations["component"]["enabled"]:
            self.results.extend(self.run_component_ablation(
                model, tokenizer, harmful, harmless, refusal_direction
            ))
        
        if ablations["calibration"]["enabled"]:
            self.results.extend(self.run_calibration_ablation(
                model, tokenizer, harmful, harmless
            ))
        
        # Save results
        self.save_results()
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return self.results
    
    def save_results(self) -> None:
        """Save ablation results."""
        # Save JSON
        results_json = [asdict(r) for r in self.results]
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        
        # Save summary
        summary_path = self.output_dir / "ablation_summary.txt"
        with open(summary_path, "w") as f:
            f.write("NSKI Ablation Study Results\n")
            f.write("=" * 80 + "\n\n")
            
            # Group by ablation type
            for abl_type in ["layer", "strength", "component", "calibration"]:
                type_results = [r for r in self.results if r.ablation_type == abl_type]
                if type_results:
                    f.write(f"\n{abl_type.upper()} ABLATION\n")
                    f.write("-" * 40 + "\n")
                    for r in type_results:
                        f.write(f"  {r.ablation_value}: ASR={r.asr:.3f}, Utility={r.utility:.3f}, PPL={r.perplexity:.2f}\n")
        
        logger.info(f"Results saved to: {self.output_dir}")


def run_ablation_study(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> List[AblationResult]:
    """
    Run ablation study.
    
    Args:
        config_path: Path to config file
        output_dir: Output directory
        
    Returns:
        List of ablation results
    """
    study = AblationStudy(
        config_path=config_path,
        output_dir=output_dir,
    )
    return study.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NSKI Ablation Study")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, default="results/ablations", help="Output directory")
    args = parser.parse_args()
    
    results = run_ablation_study(
        config_path=args.config,
        output_dir=args.output,
    )
    
    print("\n" + "=" * 60)
    print("Ablation Study Complete!")
    print("=" * 60)
    for r in results:
        print(f"{r.ablation_type}/{r.ablation_value}: ASR={r.asr:.3f}, Utility={r.utility:.3f}")
