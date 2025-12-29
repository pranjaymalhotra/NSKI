"""
NSKI Command-Line Interface.

Provides command-line access to NSKI functionality:
- Run experiments
- Apply surgery to models
- Evaluate models
- Generate figures
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_experiment(args: argparse.Namespace) -> int:
    """Run experiments."""
    from .experiments import run_all_experiments
    
    results = run_all_experiments(
        config_path=args.config,
        output_dir=args.output,
        experiments=args.experiments,
    )
    
    return 0 if results else 1


def cmd_surgery(args: argparse.Namespace) -> int:
    """Apply NSKI surgery to a model and run inference."""
    from .models import ModelLoader, get_model_config
    from .core import RefusalDirectionExtractor, NSKISurgeon
    from .data import AdvBenchDataset, AlpacaDataset
    
    print(f"Loading model: {args.model}")
    loader = ModelLoader(
        model_name=args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        quantization=args.quantization,
    )
    model, tokenizer = loader.load()
    
    # Extract refusal direction
    print("Extracting refusal direction...")
    model_config = get_model_config(args.model)
    extractor = RefusalDirectionExtractor(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
    )
    
    # Use sample prompts for calibration
    advbench = AdvBenchDataset()
    alpaca = AlpacaDataset(n_samples=100)
    
    refusal_dir = extractor.extract(
        harmful_prompts=advbench.prompts[:50],
        harmless_prompts=alpaca.prompts[:50],
    )
    
    # Apply surgery
    print(f"Applying NSKI surgery (strength={args.strength})...")
    surgeon = NSKISurgeon(
        model=model,
        model_config=model_config,
        refusal_direction=refusal_dir,
        strength=args.strength,
    )
    surgeon.apply()
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("NSKI Surgery Applied! Enter prompts (Ctrl+C to exit):")
    print("=" * 50)
    
    try:
        while True:
            prompt = input("\n> ").strip()
            if not prompt:
                continue
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            print(f"\n{response}")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    surgeon.remove()
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a model on safety benchmarks."""
    from .models import ModelLoader, get_model_config
    from .core import RefusalDirectionExtractor, NSKISurgeon
    from .data import AdvBenchDataset, AlpacaDataset
    from .evaluation import compute_asr, KeywordJudge
    
    print(f"Loading model: {args.model}")
    loader = ModelLoader(
        model_name=args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        quantization=args.quantization,
    )
    model, tokenizer = loader.load()
    
    # Load data
    advbench = AdvBenchDataset()
    prompts = advbench.prompts[:args.n_samples]
    
    print(f"Evaluating on {len(prompts)} prompts...")
    
    judge = KeywordJudge()
    responses = []
    
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(prompts)}")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
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
    
    # Compute ASR
    refusals = [judge.is_refusal(r) for r in responses]
    asr = 1 - sum(refusals) / len(refusals)
    
    print(f"\n{'='*50}")
    print(f"Results for {args.model}:")
    print(f"  Attack Success Rate (ASR): {asr:.3f}")
    print(f"  Refusal Rate: {1-asr:.3f}")
    print(f"  Samples evaluated: {len(prompts)}")
    print(f"{'='*50}")
    
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    """Generate publication figures."""
    from .visualization import create_publication_figures
    
    figures = create_publication_figures(
        results_dir=args.results,
        output_dir=args.output,
        prefix=args.prefix,
    )
    
    print(f"Generated {len(figures)} figures")
    for f in figures:
        print(f"  - {f}")
    
    return 0 if figures else 1


def cmd_download(args: argparse.Namespace) -> int:
    """Download required datasets."""
    from .data import DatasetDownloader
    
    downloader = DatasetDownloader(data_dir=args.output)
    
    print("Downloading datasets...")
    
    if args.dataset in ["all", "advbench"]:
        path = downloader.download_advbench()
        print(f"  AdvBench: {path}")
    
    if args.dataset in ["all", "alpaca"]:
        path = downloader.download_alpaca()
        print(f"  Alpaca: {path}")
    
    if args.dataset in ["all", "harmbench"]:
        path = downloader.download_harmbench()
        print(f"  HarmBench: {path}")
    
    print("Done!")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nski",
        description="NSKI: Neural Surgical Key-Value Intervention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run experiments")
    exp_parser.add_argument("--config", type=str, help="Config file path")
    exp_parser.add_argument("--output", type=str, default="results", help="Output directory")
    exp_parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["main", "ablation"],
        help="Which experiments to run",
    )
    
    # Surgery command
    surgery_parser = subparsers.add_parser("surgery", help="Apply NSKI surgery interactively")
    surgery_parser.add_argument("--model", type=str, default="llama3-8b", help="Model name")
    surgery_parser.add_argument("--strength", type=float, default=2.0, help="Intervention strength")
    surgery_parser.add_argument("--quantization", type=str, default="4bit", help="Quantization (4bit, 8bit, none)")
    surgery_parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens")
    surgery_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on benchmarks")
    eval_parser.add_argument("--model", type=str, default="llama3-8b", help="Model name")
    eval_parser.add_argument("--quantization", type=str, default="4bit", help="Quantization")
    eval_parser.add_argument("--n-samples", type=int, default=100, help="Number of samples")
    
    # Figures command
    fig_parser = subparsers.add_parser("figures", help="Generate publication figures")
    fig_parser.add_argument("--results", type=str, default="results", help="Results directory")
    fig_parser.add_argument("--output", type=str, default="figures", help="Output directory")
    fig_parser.add_argument("--prefix", type=str, default="fig", help="Figure filename prefix")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download datasets")
    dl_parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "advbench", "alpaca", "harmbench"],
        help="Dataset to download",
    )
    dl_parser.add_argument("--output", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    setup_logging(args.verbose)
    
    # Dispatch to command handler
    handlers = {
        "experiment": cmd_experiment,
        "surgery": cmd_surgery,
        "evaluate": cmd_evaluate,
        "figures": cmd_figures,
        "download": cmd_download,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
