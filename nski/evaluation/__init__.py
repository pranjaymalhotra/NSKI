"""
Evaluation modules for NSKI.
"""

from .metrics import (
    compute_asr,
    compute_utility,
    compute_perplexity,
    compute_latency,
    compute_memory_usage,
    EvaluationMetrics
)
from .judges import (
    is_refusal,
    RefusalJudge,
    KeywordRefusalJudge,
    ClassifierRefusalJudge
)
from .statistical import (
    bootstrap_ci,
    compute_cohens_h,
    mcnemar_test,
    StatisticalAnalysis
)

__all__ = [
    "compute_asr",
    "compute_utility",
    "compute_perplexity",
    "compute_latency",
    "compute_memory_usage",
    "EvaluationMetrics",
    "is_refusal",
    "RefusalJudge",
    "KeywordRefusalJudge",
    "ClassifierRefusalJudge",
    "bootstrap_ci",
    "compute_cohens_h",
    "mcnemar_test",
    "StatisticalAnalysis",
]
