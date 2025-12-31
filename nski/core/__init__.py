"""
Core NSKI modules for KV-cache intervention.
"""

from .kv_hooks import KVCacheHook, register_kv_hook, remove_kv_hooks
from .refusal_direction import RefusalDirectionExtractor
from .surgery import NSKISurgeon
from .utils import set_seed, get_device, memory_stats, get_memory_usage, setup_logging
from .nski_variants import (
    NSKI_VARIANTS,
    NSKIVariantBase,
    NSKIVariantConfig,
    NSKIStandard,
    NSKIEarlyLayers,
    NSKIMiddleLayers,
    NSKILateLayers,
    NSKIAdaptive,
    NSKIContrastive,
    RepresentationEngineering,
    InferenceTimeIntervention,
    ActivationAddition,
    get_variant,
    list_variants,
)

__all__ = [
    "KVCacheHook",
    "register_kv_hook", 
    "remove_kv_hooks",
    "RefusalDirectionExtractor",
    "NSKISurgeon",
    "set_seed",
    "get_device",
    "memory_stats",
    "get_memory_usage",
    "setup_logging",
    # Variants
    "NSKI_VARIANTS",
    "NSKIVariantBase",
    "NSKIVariantConfig",
    "NSKIStandard",
    "NSKIEarlyLayers",
    "NSKIMiddleLayers",
    "NSKILateLayers",
    "NSKIAdaptive",
    "NSKIContrastive",
    "RepresentationEngineering",
    "InferenceTimeIntervention",
    "ActivationAddition",
    "get_variant",
    "list_variants",
]