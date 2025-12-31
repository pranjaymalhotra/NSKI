#!/usr/bin/env python3
"""
Statistical Analysis Functions for Publication-Ready Results

This module provides statistical validation including:
- Bootstrap confidence intervals
- McNemar's test for paired comparisons
- Cohen's h effect size
- Multiple testing corrections

Author: Pranjay Malhotra
Date: December 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_samples: int = 0
    significant: bool = False
    interpretation: str = ""


def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    statistic_func: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Array of values to compute CI for
        statistic_func: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    data = np.asarray(data)
    n = len(data)
    
    # Point estimate
    point_estimate = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = data[np.random.randint(0, n, size=n)]
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile method for CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return float(point_estimate), float(ci_lower), float(ci_upper)


def compute_asr_with_ci(
    refused: List[bool],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute Attack Success Rate with bootstrap CI.
    
    Args:
        refused: List of booleans (True = refused/safe, False = complied/unsafe)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        seed: Random seed
    
    Returns:
        Tuple of (asr, ci_lower, ci_upper)
    """
    # ASR = 1 - refusal_rate (compliance rate)
    complied = [not r for r in refused]
    
    return bootstrap_ci(
        complied,
        statistic_func=np.mean,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed
    )


def mcnemars_test(
    method1_refused: List[bool],
    method2_refused: List[bool],
) -> StatisticalResult:
    """
    McNemar's test for paired nominal data.
    
    Tests whether two methods have significantly different refusal rates
    on the same set of prompts.
    
    Args:
        method1_refused: List of booleans for method 1 (True = refused)
        method2_refused: List of booleans for method 2 (True = refused)
    
    Returns:
        StatisticalResult with test statistic and p-value
    """
    if len(method1_refused) != len(method2_refused):
        raise ValueError("Both methods must have same number of samples")
    
    # Build contingency table
    # b = method1 refused, method2 complied
    # c = method1 complied, method2 refused
    b = sum(1 for m1, m2 in zip(method1_refused, method2_refused) if m1 and not m2)
    c = sum(1 for m1, m2 in zip(method1_refused, method2_refused) if not m1 and m2)
    
    # McNemar's statistic (with continuity correction)
    if b + c == 0:
        return StatisticalResult(
            test_name="McNemar's Test",
            statistic=0.0,
            p_value=1.0,
            n_samples=len(method1_refused),
            significant=False,
            interpretation="No discordant pairs - methods are identical"
        )
    
    # Chi-squared statistic with Yates correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    
    # P-value from chi-squared distribution with 1 df
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
    except ImportError:
        # Approximate p-value without scipy
        # Using normal approximation for chi2 with 1 df
        z = np.sqrt(chi2)
        p_value = 2 * (1 - _normal_cdf(abs(z)))
    
    significant = p_value < 0.05
    
    if significant:
        if b > c:
            interpretation = "Method 2 significantly better (more refusals)"
        else:
            interpretation = "Method 1 significantly better (more refusals)"
    else:
        interpretation = "No significant difference between methods"
    
    return StatisticalResult(
        test_name="McNemar's Test",
        statistic=chi2,
        p_value=p_value,
        n_samples=len(method1_refused),
        significant=significant,
        interpretation=interpretation
    )


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for difference between two proportions.
    
    Args:
        p1: First proportion (e.g., ASR for method 1)
        p2: Second proportion (e.g., ASR for method 2)
    
    Returns:
        Cohen's h effect size
        - Small: |h| ≈ 0.2
        - Medium: |h| ≈ 0.5
        - Large: |h| ≈ 0.8
    """
    # Arcsine transformation
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    
    return phi1 - phi2


def interpret_cohens_h(h: float) -> str:
    """Interpret Cohen's h effect size."""
    h_abs = abs(h)
    if h_abs < 0.2:
        return "negligible"
    elif h_abs < 0.5:
        return "small"
    elif h_abs < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple testing.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level (default: 0.05)
    
    Returns:
        List of (adjusted_p_value, is_significant) tuples
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    results = []
    for p in p_values:
        adjusted_p = min(p * n_tests, 1.0)
        is_significant = p < adjusted_alpha
        results.append((adjusted_p, is_significant))
    
    return results


def compute_all_statistics(
    results: Dict[str, Dict],
    baseline_key: str = "baseline",
) -> Dict[str, StatisticalResult]:
    """
    Compute comprehensive statistics comparing methods.
    
    Args:
        results: Dict mapping method names to result dicts with 'refused' list and 'asr'
        baseline_key: Key for baseline method
    
    Returns:
        Dict of statistical results
    """
    stats = {}
    
    if baseline_key not in results:
        warnings.warn(f"Baseline '{baseline_key}' not found in results")
        return stats
    
    baseline = results[baseline_key]
    baseline_refused = baseline.get("per_prompt_refused", [])
    baseline_asr = baseline.get("asr", 0)
    
    for method_name, method_results in results.items():
        if method_name == baseline_key:
            continue
        
        method_refused = method_results.get("per_prompt_refused", [])
        method_asr = method_results.get("asr", 0)
        
        # Cohen's h effect size
        h = cohens_h(baseline_asr, method_asr)
        
        # McNemar's test (if per-prompt data available)
        if baseline_refused and method_refused and len(baseline_refused) == len(method_refused):
            mcnemar = mcnemars_test(baseline_refused, method_refused)
            mcnemar.effect_size = h
            stats[f"{method_name}_vs_baseline"] = mcnemar
        else:
            # Just report effect size
            stats[f"{method_name}_vs_baseline"] = StatisticalResult(
                test_name="Effect Size Only",
                statistic=0,
                p_value=float('nan'),
                effect_size=h,
                n_samples=0,
                interpretation=f"Cohen's h = {h:.3f} ({interpret_cohens_h(h)} effect)"
            )
    
    return stats


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF without scipy."""
    # Approximation using error function
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def format_statistics_table(stats: Dict[str, StatisticalResult]) -> str:
    """Format statistics as a readable table."""
    lines = [
        "=" * 80,
        "STATISTICAL ANALYSIS",
        "=" * 80,
        f"{'Comparison':<30} {'Test':<15} {'Stat':<10} {'p-value':<12} {'Effect':<10} {'Sig.':<5}",
        "-" * 80,
    ]
    
    for name, result in stats.items():
        sig_str = "***" if result.p_value < 0.001 else ("**" if result.p_value < 0.01 else ("*" if result.p_value < 0.05 else ""))
        effect_str = f"{result.effect_size:.3f}" if result.effect_size is not None else "N/A"
        p_str = f"{result.p_value:.4f}" if not np.isnan(result.p_value) else "N/A"
        
        lines.append(
            f"{name:<30} {result.test_name:<15} {result.statistic:<10.3f} {p_str:<12} {effect_str:<10} {sig_str:<5}"
        )
    
    lines.append("=" * 80)
    lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    lines.append("Effect size (Cohen's h): <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
    
    return "\n".join(lines)
