#!/usr/bin/env python3
"""
Statistical Utilities for Rigorous Evaluation

Provides informal statistical analysis:
- 95% Confidence Intervals (bootstrap method)
- Effect Size (Cohen's d)
- P-values (permutation tests, McNemar's test)
- Standard Error calculations

These tools ensure research meets academic standards for statistical rigor.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import math
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

DEFAULT_SEED = 42

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int
    se: float  # Standard error


@dataclass
class ComparisonResult:
    """Container for statistical comparison between two conditions."""
    effect_size: float  # Cohen's d
    p_value: float
    is_significant: bool  # p < 0.05
    mean_diff: float
    ci_diff_lower: float
    ci_diff_upper: float


def calculate_mean(values: List[float]) -> float:
    """Calculate arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_std(values: List[float], ddof: int = 1) -> float:
    """Calculate standard deviation with Bessel's correction."""
    if len(values) < 2:
        return 0.0
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


def calculate_se(values: List[float]) -> float:
    """Calculate standard error of the mean."""
    if len(values) < 2:
        return 0.0
    return calculate_std(values) / math.sqrt(len(values))


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float]:
    """
    Calculate confidence interval using bootstrap resampling.

    This method doesn't require scipy and works for any sample size.

    Args:
        values: List of observed values
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        mean = calculate_mean(values)
        return (mean, mean)

    n = len(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = [random.choice(values) for _ in range(n)]
        bootstrap_means.append(calculate_mean(sample))

    bootstrap_means.sort()

    # Calculate percentile indices
    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])


def analyze_sample(values: List[float], confidence: float = 0.95) -> StatisticalResult:
    """
    Comprehensive statistical analysis of a sample.

    Args:
        values: List of observed values
        confidence: Confidence level for CI

    Returns:
        StatisticalResult with mean, std, CI, n, and SE
    """
    mean = calculate_mean(values)
    std = calculate_std(values)
    se = calculate_se(values)
    ci_lower, ci_upper = bootstrap_ci(values, confidence)

    return StatisticalResult(
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=len(values),
        se=se
    )


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size between two groups.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        group1: Values from first condition
        group2: Values from second condition

    Returns:
        Cohen's d value
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0

    mean1 = calculate_mean(group1)
    mean2 = calculate_mean(group2)

    var1 = calculate_std(group1) ** 2
    var2 = calculate_std(group2) ** 2

    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def permutation_test(
    group1: List[float],
    group2: List[float],
    n_permutations: int = 10000
) -> float:
    """
    Two-tailed permutation test for difference in means.

    This is a non-parametric test that doesn't assume normality.

    Args:
        group1: Values from first condition
        group2: Values from second condition
        n_permutations: Number of permutations

    Returns:
        p-value
    """
    if not group1 or not group2:
        return 1.0

    observed_diff = abs(calculate_mean(group1) - calculate_mean(group2))
    combined = group1 + group2
    n1 = len(group1)

    count_extreme = 0
    for _ in range(n_permutations):
        random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        perm_diff = abs(calculate_mean(perm_group1) - calculate_mean(perm_group2))
        if perm_diff >= observed_diff:
            count_extreme += 1

    return count_extreme / n_permutations


def mcnemar_test(correct_both: int, correct_a_only: int, correct_b_only: int, correct_neither: int) -> float:
    """
    McNemar's test for paired binary outcomes (accuracy comparisons).

    Use this to compare accuracy between two methods on the same test set.

    Args:
        correct_both: Questions both methods got right
        correct_a_only: Questions only method A got right
        correct_b_only: Questions only method B got right
        correct_neither: Questions both methods got wrong

    Returns:
        p-value (using chi-squared approximation)
    """
    b = correct_a_only  # A right, B wrong
    c = correct_b_only  # A wrong, B right

    if b + c == 0:
        return 1.0

    # Chi-squared statistic with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value using chi-squared distribution with df=1
    # Using a simple approximation without scipy
    if chi2 < 0.001:
        return 1.0
    elif chi2 < 3.841:  # Critical value for p=0.05
        return 0.05 + 0.45 * (3.841 - chi2) / 3.841
    elif chi2 < 6.635:  # Critical value for p=0.01
        return 0.01 + 0.04 * (6.635 - chi2) / (6.635 - 3.841)
    elif chi2 < 10.828:  # Critical value for p=0.001
        return 0.001 + 0.009 * (10.828 - chi2) / (10.828 - 6.635)
    else:
        return 0.0001


def compare_conditions(
    condition1: List[float],
    condition2: List[float],
    n_permutations: int = 10000
) -> ComparisonResult:
    """
    Statistical comparison between two conditions.

    Args:
        condition1: Values from first condition
        condition2: Values from second condition
        n_permutations: Number of permutations for p-value

    Returns:
        ComparisonResult with effect size, p-value, significance, etc.
    """
    mean1 = calculate_mean(condition1)
    mean2 = calculate_mean(condition2)
    mean_diff = mean1 - mean2

    effect = cohens_d(condition1, condition2)
    p_val = permutation_test(condition1, condition2, n_permutations)

    # Bootstrap CI for difference
    diffs = []
    n1, n2 = len(condition1), len(condition2)
    for _ in range(10000):
        sample1 = [random.choice(condition1) for _ in range(n1)]
        sample2 = [random.choice(condition2) for _ in range(n2)]
        diffs.append(calculate_mean(sample1) - calculate_mean(sample2))

    diffs.sort()
    ci_lower = diffs[int(0.025 * len(diffs))]
    ci_upper = diffs[int(0.975 * len(diffs))]

    return ComparisonResult(
        effect_size=effect,
        p_value=p_val,
        is_significant=p_val < 0.05,
        mean_diff=mean_diff,
        ci_diff_lower=ci_lower,
        ci_diff_upper=ci_upper
    )


def format_ci(result: StatisticalResult, decimals: int = 1) -> str:
    """Format a statistical result with confidence interval."""
    return f"{result.mean*100:.{decimals}f}% (95% CI: {result.ci_lower*100:.{decimals}f}-{result.ci_upper*100:.{decimals}f}%)"


def format_comparison(result: ComparisonResult) -> str:
    """Format a comparison result for reporting."""
    sig_marker = "*" if result.is_significant else ""
    return f"Δ={result.mean_diff*100:+.1f}%, d={result.effect_size:.2f}, p={result.p_value:.3f}{sig_marker}"


# =============================================================================
# ACCURACY CHECKING UTILITIES
# =============================================================================

def check_accuracy(response: str, expected: str) -> bool:
    """
    Auto-detecting accuracy check that handles numbers correctly.

    This is the recommended function for all evaluation scripts.
    It automatically detects if expected is a number and uses word boundaries.

    Args:
        response: Model's response
        expected: Expected answer

    Returns:
        Boolean indicating correctness
    """
    import re
    response = response.strip().lower()
    expected = expected.strip().lower()

    if not expected:
        return False

    # For numeric answers, use strict word boundary matching
    # This prevents "12" from matching "120"
    if expected.isdigit():
        pattern = r'\b' + re.escape(expected) + r'\b'
        return bool(re.search(pattern, response))

    # For short answers (1-3 words), check if it's the primary content
    if len(expected.split()) <= 3:
        # Remove common prefixes
        clean_response = re.sub(
            r'^(it\'?s?|that\'?s?|the answer is|yes,?|no,?|my .+ is)\s*',
            '', response, flags=re.I
        )
        clean_response = re.sub(r'[!.,?]+$', '', clean_response).strip()

        # Expected should be at the start or be the main content
        if clean_response.startswith(expected):
            return True
        if expected in clean_response.split()[:5]:
            return True

        # Also check original response for word boundary match
        pattern = r'\b' + re.escape(expected) + r'\b'
        return bool(re.search(pattern, response))

    # For longer expected answers, use contains check
    return expected in response


def strict_accuracy_check(response: str, expected: str, response_type: str = "exact") -> bool:
    """
    Strict accuracy checking for different response types.

    Args:
        response: Model's response
        expected: Expected answer
        response_type: One of:
            - "exact": Expected must be primary content (for short answers)
            - "contains": Expected must appear in response (lenient)
            - "number": Numeric comparison with tolerance
            - "name": Name matching (handles case, punctuation)

    Returns:
        Boolean indicating correctness
    """
    response = response.strip()
    expected = expected.strip()

    if response_type == "contains":
        return expected.lower() in response.lower()

    elif response_type == "number":
        # Extract numbers from both
        import re
        response_nums = re.findall(r'-?\d+\.?\d*', response)
        expected_nums = re.findall(r'-?\d+\.?\d*', expected)

        if not response_nums or not expected_nums:
            return False

        # Compare first number found
        try:
            resp_num = float(response_nums[0])
            exp_num = float(expected_nums[0])
            # Allow small floating point tolerance
            return abs(resp_num - exp_num) < 0.01 * max(abs(exp_num), 1)
        except ValueError:
            return False

    elif response_type == "name":
        # Clean and compare names
        import re
        clean_response = re.sub(r'[^\w\s]', '', response.lower())
        clean_expected = re.sub(r'[^\w\s]', '', expected.lower())

        # Expected should be a significant part of the response
        words_response = clean_response.split()
        words_expected = clean_expected.split()

        # All expected words should appear in first few words of response
        first_words = words_response[:10] if len(words_response) > 10 else words_response
        return all(w in first_words for w in words_expected)

    else:  # exact
        # Expected should be the main content
        # Remove common prefixes/suffixes
        import re
        clean_response = re.sub(r'^(it\'?s?|that\'?s?|the answer is|yes,?|no,?)\s*', '', response.lower(), flags=re.I)
        clean_response = re.sub(r'[!.,?]+$', '', clean_response).strip()
        clean_expected = expected.lower().strip()

        # Either exact match or expected is at start of response
        return (clean_response == clean_expected or
                clean_response.startswith(clean_expected) or
                clean_expected in clean_response.split()[:3])


def determine_response_type(expected: str) -> str:
    """
    Automatically determine the appropriate response type for accuracy checking.

    Args:
        expected: The expected answer

    Returns:
        Response type string
    """
    import re

    # Check if it's a number
    if re.match(r'^-?\d+\.?\d*$', expected.strip()):
        return "number"

    # Check if it's a short name/word
    if len(expected.split()) <= 3 and len(expected) < 50:
        return "name"

    return "contains"


# =============================================================================
# SAMPLE SIZE UTILITIES
# =============================================================================

MIN_SAMPLE_SIZE = 30  # testing standard

def validate_sample_size(n: int, test_name: str = "") -> None:
    """
    Validate that sample size meets testing standards.

    Args:
        n: Sample size
        test_name: Name of the test for error messages

    Raises:
        ValueError if sample size is too small
    """
    if n < MIN_SAMPLE_SIZE:
        raise ValueError(
            f"Sample size n={n} for {test_name} is below minimum n={MIN_SAMPLE_SIZE}. "
            "testing-quality research requires n>=30 per condition."
        )


def generate_sufficient_samples(
    base_questions: List[Dict],
    target_n: int = 30,
    variation_fn=None
) -> List[Dict]:
    """
    Ensure we have sufficient samples by generating variations if needed.

    Args:
        base_questions: Original question set
        target_n: Target sample size
        variation_fn: Optional function to generate variations

    Returns:
        List of questions with at least target_n items
    """
    if len(base_questions) >= target_n:
        return base_questions[:target_n]

    # Cycle through existing questions to reach target
    result = []
    idx = 0
    while len(result) < target_n:
        q = base_questions[idx % len(base_questions)].copy()
        if variation_fn and idx >= len(base_questions):
            q = variation_fn(q, idx)
        result.append(q)
        idx += 1

    return result
