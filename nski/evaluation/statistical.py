"""
Statistical Analysis for NSKI

Implements statistical methods for rigorous evaluation:
- Bootstrap confidence intervals
- Effect size (Cohen's h)
- Significance testing (McNemar's test)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats
from loguru import logger


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int


@dataclass
class EffectSize:
    """Effect size result."""
    cohens_h: float
    interpretation: str  # "small", "medium", "large", "very large"


@dataclass
class SignificanceTest:
    """Significance test result."""
    statistic: float
    p_value: float
    significant: bool
    test_name: str


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    statistic: str = "mean"
) -> ConfidenceInterval:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default: 0.95 for 95% CI)
        statistic: Statistic to compute ("mean" or "proportion")
        
    Returns:
        ConfidenceInterval with mean and bounds
    """
    data = np.array(data)
    n = len(data)
    
    if n == 0:
        return ConfidenceInterval(
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            std=0.0,
            n_samples=0
        )
    
    # Bootstrap resampling
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        elif statistic == "proportion":
            bootstrap_stats.append(np.mean(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentiles
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return ConfidenceInterval(
        mean=np.mean(data),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std=np.std(bootstrap_stats),
        n_samples=n
    )


def compute_cohens_h(p1: float, p2: float) -> EffectSize:
    """
    Compute Cohen's h effect size for proportions.
    
    Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    
    Interpretation:
    - h < 0.2: small
    - 0.2 <= h < 0.5: medium
    - 0.5 <= h < 0.8: large
    - h >= 0.8: very large
    
    Args:
        p1: First proportion (e.g., baseline ASR)
        p2: Second proportion (e.g., NSKI ASR)
        
    Returns:
        EffectSize with Cohen's h and interpretation
    """
    # Clamp to valid range
    p1 = np.clip(p1, 0.001, 0.999)
    p2 = np.clip(p2, 0.001, 0.999)
    
    # Compute Cohen's h
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    h = abs(phi1 - phi2)
    
    # Interpret
    if h < 0.2:
        interpretation = "small"
    elif h < 0.5:
        interpretation = "medium"
    elif h < 0.8:
        interpretation = "large"
    else:
        interpretation = "very large"
    
    return EffectSize(cohens_h=h, interpretation=interpretation)


def mcnemar_test(
    baseline_results: List[bool],
    treatment_results: List[bool]
) -> SignificanceTest:
    """
    McNemar's test for paired nominal data.
    
    Tests whether the row and column marginal frequencies are equal,
    i.e., whether the treatment has a significant effect.
    
    Args:
        baseline_results: List of refusal outcomes for baseline (True=refusal)
        treatment_results: List of refusal outcomes for treatment
        
    Returns:
        SignificanceTest result
    """
    if len(baseline_results) != len(treatment_results):
        raise ValueError("Lists must have same length")
    
    # Build contingency table
    # b = cases where baseline refuses but treatment doesn't
    # c = cases where baseline doesn't refuse but treatment does
    b = sum(1 for bl, tr in zip(baseline_results, treatment_results) 
            if bl and not tr)
    c = sum(1 for bl, tr in zip(baseline_results, treatment_results)
            if not bl and tr)
    
    # McNemar's test
    if b + c == 0:
        return SignificanceTest(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            test_name="McNemar's test"
        )
    
    # Chi-squared statistic with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return SignificanceTest(
        statistic=chi2,
        p_value=p_value,
        significant=p_value < 0.05,
        test_name="McNemar's test"
    )


def proportion_z_test(
    p1: float,
    n1: int,
    p2: float,
    n2: int
) -> SignificanceTest:
    """
    Two-proportion z-test.
    
    Tests whether two proportions are significantly different.
    
    Args:
        p1: First proportion
        n1: First sample size
        p2: Second proportion
        n2: Second sample size
        
    Returns:
        SignificanceTest result
    """
    # Pooled proportion
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    if se == 0:
        return SignificanceTest(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            test_name="Two-proportion z-test"
        )
    
    # Z statistic
    z = (p1 - p2) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return SignificanceTest(
        statistic=z,
        p_value=p_value,
        significant=p_value < 0.05,
        test_name="Two-proportion z-test"
    )


class StatisticalAnalysis:
    """
    Complete statistical analysis for experiment results.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        alpha: float = 0.05
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.alpha = alpha
    
    def analyze(
        self,
        baseline_asr: float,
        baseline_n: int,
        treatment_asr: float,
        treatment_n: int,
        baseline_results: Optional[List[bool]] = None,
        treatment_results: Optional[List[bool]] = None
    ) -> Dict:
        """
        Perform complete statistical analysis.
        
        Args:
            baseline_asr: Baseline attack success rate
            baseline_n: Baseline sample size
            treatment_asr: Treatment (NSKI) attack success rate
            treatment_n: Treatment sample size
            baseline_results: Individual results for paired tests
            treatment_results: Individual results for paired tests
            
        Returns:
            Dictionary with all statistical results
        """
        results = {}
        
        # ASR reduction
        results['asr_reduction'] = (baseline_asr - treatment_asr) / baseline_asr if baseline_asr > 0 else 0
        results['asr_reduction_pct'] = results['asr_reduction'] * 100
        
        # Effect size
        effect = compute_cohens_h(baseline_asr, treatment_asr)
        results['effect_size'] = {
            'cohens_h': effect.cohens_h,
            'interpretation': effect.interpretation
        }
        
        # Z-test
        z_test = proportion_z_test(baseline_asr, baseline_n, treatment_asr, treatment_n)
        results['z_test'] = {
            'statistic': z_test.statistic,
            'p_value': z_test.p_value,
            'significant': z_test.significant
        }
        
        # McNemar's test if paired data available
        if baseline_results and treatment_results:
            mcnemar = mcnemar_test(baseline_results, treatment_results)
            results['mcnemar_test'] = {
                'statistic': mcnemar.statistic,
                'p_value': mcnemar.p_value,
                'significant': mcnemar.significant
            }
        
        # Bootstrap CI for ASR reduction
        if baseline_results and treatment_results:
            reductions = [
                1 - (1 - b + t) / 2  # Simplified paired difference
                for b, t in zip(baseline_results, treatment_results)
            ]
            ci = bootstrap_ci(reductions, self.n_bootstrap, self.confidence)
            results['bootstrap_ci'] = {
                'mean': ci.mean,
                'ci_lower': ci.ci_lower,
                'ci_upper': ci.ci_upper,
                'std': ci.std
            }
        
        # Summary
        results['summary'] = {
            'baseline_asr': baseline_asr,
            'treatment_asr': treatment_asr,
            'reduction': results['asr_reduction_pct'],
            'significant': z_test.significant,
            'p_value': z_test.p_value,
            'effect_size': effect.interpretation
        }
        
        return results
    
    def format_results(self, results: Dict) -> str:
        """Format results for display."""
        lines = [
            "Statistical Analysis Results",
            "=" * 40,
            f"Baseline ASR:    {results['summary']['baseline_asr']:.1%}",
            f"Treatment ASR:   {results['summary']['treatment_asr']:.1%}",
            f"ASR Reduction:   {results['asr_reduction_pct']:.1f}%",
            "",
            f"Effect Size:     h = {results['effect_size']['cohens_h']:.3f} ({results['effect_size']['interpretation']})",
            f"p-value:         {results['z_test']['p_value']:.4f}",
            f"Significant:     {'Yes' if results['summary']['significant'] else 'No'} (α = {self.alpha})",
        ]
        
        if 'bootstrap_ci' in results:
            ci = results['bootstrap_ci']
            lines.append(f"95% CI:          [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
        
        return "\n".join(lines)
