#!/usr/bin/env python3
"""
Extended Model and Method Configurations for NSKI

Includes:
- 10+ language models (various sizes and families)
- 6+ defense baselines
- Attack methods for evaluation
- Hardware-aware model selection

Author: Pranjay Malhotra  
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ModelFamily(Enum):
    """Model families for grouping."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    PHI = "phi"
    GEMMA = "gemma"
    GPT2 = "gpt2"
    VICUNA = "vicuna"
    FALCON = "falcon"


@dataclass
class ModelSpec:
    """Specification for a model."""
    name: str                    # HuggingFace model name
    family: ModelFamily          # Model family
    size_b: float               # Size in billions of parameters
    vram_4bit: float            # VRAM needed with 4-bit quantization (GB)
    vram_8bit: float            # VRAM needed with 8-bit quantization (GB)
    safety_trained: bool         # Whether model has safety training
    chat_template: bool          # Whether model uses chat template
    requires_auth: bool          # Whether HF authentication required
    context_length: int          # Max context length
    description: str             # Brief description
    priority: int = 5            # Priority for experiments (1=highest)


# ============================================================================
# MODEL REGISTRY - Publication-Ready Selection
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # -------------------------------------------------------------------------
    # TIER 1: Primary Models (Must test for publication)
    # -------------------------------------------------------------------------
    "meta-llama/Llama-2-7b-chat-hf": ModelSpec(
        name="meta-llama/Llama-2-7b-chat-hf",
        family=ModelFamily.LLAMA,
        size_b=7.0,
        vram_4bit=4.0,
        vram_8bit=7.5,
        safety_trained=True,
        chat_template=True,
        requires_auth=True,
        context_length=4096,
        description="Llama 2 7B Chat - RLHF safety-trained",
        priority=1,
    ),
    
    "meta-llama/Meta-Llama-3-8B-Instruct": ModelSpec(
        name="meta-llama/Meta-Llama-3-8B-Instruct",
        family=ModelFamily.LLAMA,
        size_b=8.0,
        vram_4bit=5.0,
        vram_8bit=9.0,
        safety_trained=True,
        chat_template=True,
        requires_auth=True,
        context_length=8192,
        description="Llama 3 8B Instruct - Latest safety alignment",
        priority=1,
    ),
    
    "mistralai/Mistral-7B-Instruct-v0.2": ModelSpec(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        family=ModelFamily.MISTRAL,
        size_b=7.0,
        vram_4bit=4.0,
        vram_8bit=7.5,
        safety_trained=True,
        chat_template=True,
        requires_auth=False,
        context_length=32768,
        description="Mistral 7B v0.2 - Different safety approach",
        priority=1,
    ),
    
    # -------------------------------------------------------------------------
    # TIER 2: Important Secondary Models
    # -------------------------------------------------------------------------
    "Qwen/Qwen2-7B-Instruct": ModelSpec(
        name="Qwen/Qwen2-7B-Instruct",
        family=ModelFamily.QWEN,
        size_b=7.0,
        vram_4bit=4.0,
        vram_8bit=7.5,
        safety_trained=True,
        chat_template=True,
        requires_auth=False,
        context_length=32768,
        description="Qwen2 7B - Non-Western safety perspective",
        priority=2,
    ),
    
    "google/gemma-2-9b-it": ModelSpec(
        name="google/gemma-2-9b-it",
        family=ModelFamily.GEMMA,
        size_b=9.0,
        vram_4bit=5.5,
        vram_8bit=10.0,
        safety_trained=True,
        chat_template=True,
        requires_auth=True,
        context_length=8192,
        description="Gemma 2 9B - Google's safety approach",
        priority=2,
    ),
    
    "microsoft/Phi-3-mini-4k-instruct": ModelSpec(
        name="microsoft/Phi-3-mini-4k-instruct",
        family=ModelFamily.PHI,
        size_b=3.8,
        vram_4bit=2.5,
        vram_8bit=4.5,
        safety_trained=True,
        chat_template=True,
        requires_auth=False,
        context_length=4096,
        description="Phi-3 Mini - Small but capable, good for ablations",
        priority=2,
    ),
    
    # -------------------------------------------------------------------------
    # TIER 3: Additional Models for Comprehensive Analysis
    # -------------------------------------------------------------------------
    "lmsys/vicuna-7b-v1.5": ModelSpec(
        name="lmsys/vicuna-7b-v1.5",
        family=ModelFamily.VICUNA,
        size_b=7.0,
        vram_4bit=4.0,
        vram_8bit=7.5,
        safety_trained=False,  # Less safety training than Llama
        chat_template=True,
        requires_auth=False,
        context_length=4096,
        description="Vicuna 7B - Weaker safety baseline",
        priority=3,
    ),
    
    "meta-llama/Llama-2-13b-chat-hf": ModelSpec(
        name="meta-llama/Llama-2-13b-chat-hf",
        family=ModelFamily.LLAMA,
        size_b=13.0,
        vram_4bit=7.5,
        vram_8bit=14.0,
        safety_trained=True,
        chat_template=True,
        requires_auth=True,
        context_length=4096,
        description="Llama 2 13B Chat - Scale comparison",
        priority=3,
    ),
    
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelSpec(
        name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        family=ModelFamily.MISTRAL,
        size_b=47.0,  # Total params (8 experts x 7B, but sparse)
        vram_4bit=24.0,
        vram_8bit=48.0,
        safety_trained=True,
        chat_template=True,
        requires_auth=False,
        context_length=32768,
        description="Mixtral 8x7B - MoE architecture",
        priority=4,
    ),
    
    # -------------------------------------------------------------------------
    # TIER 4: Testing/Development Models
    # -------------------------------------------------------------------------
    "gpt2-xl": ModelSpec(
        name="gpt2-xl",
        family=ModelFamily.GPT2,
        size_b=1.5,
        vram_4bit=1.5,
        vram_8bit=2.0,
        safety_trained=False,
        chat_template=False,
        requires_auth=False,
        context_length=1024,
        description="GPT-2 XL - Fast testing, no safety training",
        priority=5,
    ),
    
    "microsoft/phi-2": ModelSpec(
        name="microsoft/phi-2",
        family=ModelFamily.PHI,
        size_b=2.7,
        vram_4bit=2.0,
        vram_8bit=3.5,
        safety_trained=False,
        chat_template=False,
        requires_auth=False,
        context_length=2048,
        description="Phi-2 - Small, fast, limited safety",
        priority=5,
    ),
}


# ============================================================================
# DEFENSE METHOD REGISTRY
# ============================================================================

@dataclass
class DefenseSpec:
    """Specification for a defense method."""
    name: str                    # Method identifier
    full_name: str               # Full academic name
    paper_ref: str               # Paper reference
    description: str             # Brief description
    requires_calibration: bool   # Needs calibration data
    modifies_weights: bool       # Modifies model weights
    modifies_inference: bool     # Modifies inference process
    computational_cost: str      # "low", "medium", "high"
    implemented: bool            # Whether implemented in codebase


DEFENSE_REGISTRY: Dict[str, DefenseSpec] = {
    "baseline": DefenseSpec(
        name="baseline",
        full_name="No Defense (Baseline)",
        paper_ref="N/A",
        description="Raw model without any defense mechanism",
        requires_calibration=False,
        modifies_weights=False,
        modifies_inference=False,
        computational_cost="low",
        implemented=True,
    ),
    
    "nski": DefenseSpec(
        name="nski",
        full_name="Neural Surgical Key-Value Intervention",
        paper_ref="Malhotra et al., 2024",
        description="KV-cache intervention to suppress harmful patterns",
        requires_calibration=True,
        modifies_weights=False,
        modifies_inference=True,
        computational_cost="low",
        implemented=True,
    ),
    
    "arditi": DefenseSpec(
        name="arditi",
        full_name="Activation Steering",
        paper_ref="Arditi et al., 2024",
        description="Steering activation vectors away from harmful directions",
        requires_calibration=True,
        modifies_weights=False,
        modifies_inference=True,
        computational_cost="medium",
        implemented=True,
    ),
    
    "belitsky": DefenseSpec(
        name="belitsky",
        full_name="Safety Modulation",
        paper_ref="Belitsky et al., 2025",
        description="Attention head modulation for safety",
        requires_calibration=True,
        modifies_weights=False,
        modifies_inference=True,
        computational_cost="medium",
        implemented=True,
    ),
    
    "jbshield": DefenseSpec(
        name="jbshield",
        full_name="JBShield",
        paper_ref="JBShield, 2024",
        description="Input filtering to detect jailbreak attempts",
        requires_calibration=False,
        modifies_weights=False,
        modifies_inference=True,
        computational_cost="low",
        implemented=True,
    ),
    
    "circuit_breakers": DefenseSpec(
        name="circuit_breakers",
        full_name="Circuit Breakers",
        paper_ref="Zou et al., 2024",
        description="Representation rerouting for safety",
        requires_calibration=True,
        modifies_weights=True,
        modifies_inference=False,
        computational_cost="high",
        implemented=False,  # TODO: Implement
    ),
    
    "smoothllm": DefenseSpec(
        name="smoothllm",
        full_name="SmoothLLM",
        paper_ref="Robey et al., 2023",
        description="Input perturbation defense",
        requires_calibration=False,
        modifies_weights=False,
        modifies_inference=True,
        computational_cost="high",
        implemented=False,  # TODO: Implement
    ),
    
    "repeng": DefenseSpec(
        name="repeng",
        full_name="Representation Engineering",
        paper_ref="Zou et al., 2023",
        description="Concept erasure through representations",
        requires_calibration=True,
        modifies_weights=False,
        modifies_inference=True,
        computational_cost="medium",
        implemented=False,  # TODO: Implement
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_models_for_vram(vram_gb: float, quantization: str = "4bit") -> List[str]:
    """Get list of models that fit in given VRAM."""
    attr = "vram_4bit" if quantization == "4bit" else "vram_8bit"
    
    suitable = []
    for name, spec in MODEL_REGISTRY.items():
        if getattr(spec, attr) <= vram_gb * 0.9:  # 90% safety margin
            suitable.append((name, spec.priority))
    
    # Sort by priority
    suitable.sort(key=lambda x: x[1])
    return [name for name, _ in suitable]


def get_publication_models() -> List[str]:
    """Get models required for publication (priority 1-2)."""
    return [
        name for name, spec in MODEL_REGISTRY.items()
        if spec.priority <= 2
    ]


def get_implemented_defenses() -> List[str]:
    """Get list of implemented defense methods."""
    return [
        name for name, spec in DEFENSE_REGISTRY.items()
        if spec.implemented
    ]


def get_model_spec(model_name: str) -> Optional[ModelSpec]:
    """Get specification for a model."""
    return MODEL_REGISTRY.get(model_name)


def get_defense_spec(defense_name: str) -> Optional[DefenseSpec]:
    """Get specification for a defense."""
    return DEFENSE_REGISTRY.get(defense_name)


def estimate_experiment_time(
    models: List[str],
    n_prompts: int,
    defenses: List[str] = None,
    include_ablation: bool = True,
) -> float:
    """
    Estimate total experiment time in hours.
    
    Based on empirical measurements:
    - ~0.5s per prompt for inference
    - ~2 min for model loading
    - ~30 min for full ablation
    """
    if defenses is None:
        defenses = get_implemented_defenses()
    
    # Base time per model
    time_per_model_mins = 2  # Loading time
    
    # Time per prompt per defense
    time_per_prompt_secs = 0.5
    time_per_defense = (n_prompts * time_per_prompt_secs) / 60  # mins
    
    total_mins = 0
    for model in models:
        total_mins += time_per_model_mins
        total_mins += len(defenses) * time_per_defense
        
        if include_ablation:
            total_mins += 30  # Ablation study
    
    return total_mins / 60  # hours


def print_model_summary():
    """Print summary of all models."""
    print("\n" + "=" * 80)
    print("NSKI MODEL REGISTRY")
    print("=" * 80)
    
    for priority in [1, 2, 3, 4, 5]:
        models = [(n, s) for n, s in MODEL_REGISTRY.items() if s.priority == priority]
        if not models:
            continue
            
        print(f"\n[Priority {priority}]")
        print("-" * 60)
        
        for name, spec in models:
            auth = "🔐" if spec.requires_auth else "  "
            safety = "✓" if spec.safety_trained else "✗"
            print(f"{auth} {name:<45} {spec.size_b:>4.1f}B  4bit:{spec.vram_4bit:>4.1f}GB  safety:{safety}")


def print_defense_summary():
    """Print summary of all defenses."""
    print("\n" + "=" * 80)
    print("NSKI DEFENSE REGISTRY")
    print("=" * 80)
    
    for name, spec in DEFENSE_REGISTRY.items():
        status = "✓" if spec.implemented else "TODO"
        print(f"[{status}] {name:<15} - {spec.full_name}")
        print(f"      {spec.description}")
        print(f"      Cost: {spec.computational_cost}, Paper: {spec.paper_ref}")
        print()


if __name__ == "__main__":
    print_model_summary()
    print_defense_summary()
    
    print("\n" + "=" * 80)
    print("HARDWARE RECOMMENDATIONS")
    print("=" * 80)
    
    for vram in [6, 8, 12, 16, 24, 48]:
        models = get_models_for_vram(vram)
        print(f"\n{vram} GB VRAM: {len(models)} models")
        for m in models[:3]:
            print(f"  - {m}")
        if len(models) > 3:
            print(f"  ... and {len(models) - 3} more")
