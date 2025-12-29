"""
NSKI: Neural Surgical Key-Value Intervention

A production-ready framework for inference-time safety interventions in LLMs
via precise manipulation of key-value cache representations.

GitHub: https://github.com/pranjaymalhotra/NSKI
"""

__version__ = "1.0.0"
__author__ = "Pranjay Malhotra"
__license__ = "MIT"

from .core import (
    KVCacheHook,
    register_kv_hook,
    remove_kv_hooks,
    RefusalDirectionExtractor,
    NSKISurgeon,
    set_seed,
    get_memory_usage,
    setup_logging,
)

from .models import (
    ModelLoader,
    get_model_config,
    SUPPORTED_MODELS,
)

from .data import (
    DatasetDownloader,
    AdvBenchDataset,
    AlpacaDataset,
    HarmBenchDataset,
)

from .evaluation import (
    compute_asr,
    compute_utility,
    compute_perplexity,
    compute_latency,
    KeywordJudge,
    ClassifierJudge,
    StatisticalAnalyzer,
)

from .baselines import (
    ArditiSteering,
    BelitskyModulation,
    JBShield,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    # Core
    "KVCacheHook",
    "register_kv_hook",
    "remove_kv_hooks",
    "RefusalDirectionExtractor",
    "NSKISurgeon",
    "set_seed",
    "get_memory_usage",
    "setup_logging",
    # Models
    "ModelLoader",
    "get_model_config",
    "SUPPORTED_MODELS",
    # Data
    "DatasetDownloader",
    "AdvBenchDataset",
    "AlpacaDataset",
    "HarmBenchDataset",
    # Evaluation
    "compute_asr",
    "compute_utility",
    "compute_perplexity",
    "compute_latency",
    "KeywordJudge",
    "ClassifierJudge",
    "StatisticalAnalyzer",
    # Baselines
    "ArditiSteering",
    "BelitskyModulation",
    "JBShield",
]
