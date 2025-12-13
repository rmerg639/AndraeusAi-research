#!/usr/bin/env python3
"""
Unit Tests for Statistical Utilities

Tests for:
- Basic statistics (mean, std, se)
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d)
- Statistical comparisons
- Accuracy checking functions
- Sample size validation

Run with: pytest tests/test_stats_utils.py -v

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from stats_utils import (
    calculate_mean,
    calculate_std,
    calculate_se,
    bootstrap_ci,
    analyze_sample,
    cohens_d,
    permutation_test,
    compare_conditions,
    check_accuracy,
    strict_accuracy_check,
    determine_response_type,
    validate_sample_size,
    StatisticalResult,
    ComparisonResult,
    MIN_SAMPLE_SIZE,
)


# =============================================================================
# BASIC STATISTICS TESTS
# =============================================================================

class TestBasicStatistics:
    """Tests for basic statistical functions."""

    def test_mean_simple(self):
        """Test mean calculation."""
        assert calculate_mean([1, 2, 3, 4, 5]) == 3.0
        assert calculate_mean([10]) == 10.0

    def test_mean_empty(self):
        """Test mean with empty list."""
        assert calculate_mean([]) == 0.0

    def test_mean_floats(self):
        """Test mean with float values."""
        result = calculate_mean([0.9, 0.95, 0.85])
        assert abs(result - 0.9) < 0.0001

    def test_std_simple(self):
        """Test standard deviation calculation."""
        # Known values: [2, 4, 4, 4, 5, 5, 7, 9] has std ≈ 2.138
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        std = calculate_std(values)
        assert 2.0 < std < 2.3

    def test_std_single_value(self):
        """Test std with single value."""
        assert calculate_std([5]) == 0.0

    def test_std_empty(self):
        """Test std with empty list."""
        assert calculate_std([]) == 0.0

    def test_se_calculation(self):
        """Test standard error calculation."""
        values = [1, 2, 3, 4, 5]
        se = calculate_se(values)
        std = calculate_std(values)
        expected_se = std / (len(values) ** 0.5)
        assert abs(se - expected_se) < 0.0001


# =============================================================================
# BOOTSTRAP CI TESTS
# =============================================================================

class TestBootstrapCI:
    """Tests for bootstrap confidence interval calculation."""

    def test_ci_bounds_order(self):
        """Test that CI lower < upper."""
        values = [0.8, 0.85, 0.9, 0.92, 0.88, 0.87]
        lower, upper = bootstrap_ci(values)
        assert lower <= upper

    def test_ci_contains_mean(self):
        """Test that CI contains the sample mean."""
        values = [0.8, 0.85, 0.9, 0.92, 0.88, 0.87]
        lower, upper = bootstrap_ci(values)
        mean = calculate_mean(values)
        assert lower <= mean <= upper

    def test_ci_single_value(self):
        """Test CI with single value."""
        lower, upper = bootstrap_ci([0.9])
        assert lower == 0.9
        assert upper == 0.9

    def test_ci_confidence_level(self):
        """Test that higher confidence = wider interval."""
        values = list(range(100))
        ci_90 = bootstrap_ci(values, confidence=0.90)
        ci_99 = bootstrap_ci(values, confidence=0.99)
        width_90 = ci_90[1] - ci_90[0]
        width_99 = ci_99[1] - ci_99[0]
        assert width_99 >= width_90


# =============================================================================
# ANALYZE SAMPLE TESTS
# =============================================================================

class TestAnalyzeSample:
    """Tests for comprehensive sample analysis."""

    def test_returns_statistical_result(self):
        """Test that analyze_sample returns StatisticalResult."""
        values = [0.8, 0.85, 0.9]
        result = analyze_sample(values)
        assert isinstance(result, StatisticalResult)

    def test_result_fields(self):
        """Test that result has all required fields."""
        values = [0.8, 0.85, 0.9]
        result = analyze_sample(values)
        assert hasattr(result, 'mean')
        assert hasattr(result, 'std')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 'n')
        assert hasattr(result, 'se')

    def test_result_n_correct(self):
        """Test that n is correct."""
        values = [1, 2, 3, 4, 5]
        result = analyze_sample(values)
        assert result.n == 5


# =============================================================================
# COHEN'S D TESTS
# =============================================================================

class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_identical_groups(self):
        """Test d=0 for identical groups."""
        group = [1, 2, 3, 4, 5]
        d = cohens_d(group, group.copy())
        assert abs(d) < 0.1

    def test_large_effect(self):
        """Test large effect size detection."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [10, 11, 12, 13, 14]
        d = cohens_d(group1, group2)
        assert abs(d) > 0.8  # Large effect

    def test_small_effect(self):
        """Test small effect size detection."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [1.5, 2.5, 3.5, 4.5, 5.5]
        d = cohens_d(group1, group2)
        assert abs(d) < 0.5  # Small effect

    def test_sign_direction(self):
        """Test that sign indicates direction."""
        group1 = [10, 11, 12]
        group2 = [1, 2, 3]
        d = cohens_d(group1, group2)
        assert d > 0  # group1 > group2


# =============================================================================
# PERMUTATION TEST
# =============================================================================

class TestPermutationTest:
    """Tests for permutation test."""

    def test_identical_groups_not_significant(self):
        """Test p > 0.05 for identical groups."""
        group = [1, 2, 3, 4, 5]
        p = permutation_test(group, group.copy(), n_permutations=1000)
        assert p > 0.05

    def test_different_groups_significant(self):
        """Test p < 0.05 for clearly different groups."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [100, 101, 102, 103, 104]
        p = permutation_test(group1, group2, n_permutations=1000)
        assert p < 0.05

    def test_empty_group_returns_one(self):
        """Test that empty groups return p=1."""
        assert permutation_test([], [1, 2, 3]) == 1.0
        assert permutation_test([1, 2, 3], []) == 1.0


# =============================================================================
# COMPARE CONDITIONS TESTS
# =============================================================================

class TestCompareConditions:
    """Tests for comprehensive condition comparison."""

    def test_returns_comparison_result(self):
        """Test that compare_conditions returns ComparisonResult."""
        c1 = [0.8, 0.85, 0.9]
        c2 = [0.7, 0.75, 0.8]
        result = compare_conditions(c1, c2, n_permutations=100)
        assert isinstance(result, ComparisonResult)

    def test_result_fields(self):
        """Test that result has all required fields."""
        c1 = [0.8, 0.85, 0.9]
        c2 = [0.7, 0.75, 0.8]
        result = compare_conditions(c1, c2, n_permutations=100)
        assert hasattr(result, 'effect_size')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'is_significant')
        assert hasattr(result, 'mean_diff')


# =============================================================================
# ACCURACY CHECKING TESTS
# =============================================================================

class TestCheckAccuracy:
    """Tests for the auto-detecting accuracy check function."""

    def test_exact_match(self):
        """Test exact string matching."""
        assert check_accuracy("Alex", "Alex") == True
        assert check_accuracy("alex", "Alex") == True
        assert check_accuracy("ALEX", "alex") == True

    def test_number_boundary_strict(self):
        """Test that numbers respect word boundaries."""
        # This is the critical test - prevents "12" matching "120"
        assert check_accuracy("The answer is 120", "12") == False
        assert check_accuracy("The answer is 12", "12") == True
        assert check_accuracy("12 years old", "12") == True
        assert check_accuracy("He is 12.", "12") == True
        assert check_accuracy("You're 28 years old!", "28") == True

    def test_name_in_response(self):
        """Test name detection in response."""
        assert check_accuracy("That's Max!", "Max") == True
        assert check_accuracy("It's Max.", "Max") == True
        assert check_accuracy("Max is my pet", "Max") == True

    def test_false_positive_prevention(self):
        """Test that false positives are prevented."""
        # Note: "Max" in "Maximum" is a known edge case (prefix match)
        # Testing number boundaries which must be strict
        assert check_accuracy("1200 points", "12") == False  # Number at start
        assert check_accuracy("312 items", "12") == False  # Number at end

    def test_common_prefixes(self):
        """Test responses with common prefixes."""
        assert check_accuracy("It's Seattle!", "Seattle") == True
        assert check_accuracy("That's Seattle.", "Seattle") == True
        assert check_accuracy("The answer is Seattle", "Seattle") == True
        assert check_accuracy("Yes, it's Alex", "Alex") == True

    def test_empty_expected(self):
        """Test with empty expected value."""
        assert check_accuracy("Any response", "") == False

    def test_longer_expected(self):
        """Test with longer expected strings."""
        assert check_accuracy(
            "I work as a Software Engineer at Google",
            "Software Engineer"
        ) == True


class TestStrictAccuracyCheck:
    """Tests for strict accuracy checking with type hints."""

    def test_exact_type(self):
        """Test exact match type."""
        assert strict_accuracy_check("Seattle", "Seattle", "exact") == True
        assert strict_accuracy_check("It's Seattle!", "Seattle", "exact") == True

    def test_contains_type(self):
        """Test contains match type."""
        assert strict_accuracy_check("My name is Alex and I'm 28", "Alex", "contains") == True
        assert strict_accuracy_check("No Alex here", "Jordan", "contains") == False

    def test_number_type(self):
        """Test number extraction and comparison."""
        assert strict_accuracy_check("I am 28 years old", "28", "number") == True
        assert strict_accuracy_check("28", "28.0", "number") == True
        assert strict_accuracy_check("You're 28!", "28", "number") == True

    def test_name_type(self):
        """Test name matching."""
        assert strict_accuracy_check("Max is a great pet", "Max", "name") == True
        assert strict_accuracy_check("Your pet Max is cute", "Max", "name") == True


class TestDetermineResponseType:
    """Tests for automatic response type detection."""

    def test_detects_numbers(self):
        """Test that numbers are detected."""
        assert determine_response_type("28") == "number"
        assert determine_response_type("3.14") == "number"
        assert determine_response_type("-5") == "number"

    def test_detects_names(self):
        """Test that short names are detected."""
        assert determine_response_type("Alex") == "name"
        assert determine_response_type("Max") == "name"
        assert determine_response_type("Software Engineer") == "name"

    def test_detects_long_content(self):
        """Test that long content uses contains."""
        long_text = "This is a very long expected response that should use contains matching"
        assert determine_response_type(long_text) == "contains"


# =============================================================================
# SAMPLE SIZE VALIDATION TESTS
# =============================================================================

class TestSampleSizeValidation:
    """Tests for sample size validation."""

    def test_valid_sample_size(self):
        """Test that valid sample sizes pass."""
        # Should not raise
        validate_sample_size(30, "test")
        validate_sample_size(100, "test")

    def test_invalid_sample_size_raises(self):
        """Test that small sample sizes raise error."""
        with pytest.raises(ValueError, match="below minimum"):
            validate_sample_size(10, "test")

    def test_error_message_contains_info(self):
        """Test that error message is informative."""
        with pytest.raises(ValueError) as exc_info:
            validate_sample_size(15, "my_test")
        assert "n=15" in str(exc_info.value)
        assert "my_test" in str(exc_info.value)

    def test_minimum_constant(self):
        """Test that MIN_SAMPLE_SIZE is 30."""
        assert MIN_SAMPLE_SIZE == 30


# =============================================================================
# EDGE CASES AND ROBUSTNESS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_accuracy_unicode(self):
        """Test accuracy with unicode characters."""
        assert check_accuracy("Café", "Café") == True
        assert check_accuracy("你好", "你好") == True

    def test_accuracy_special_chars(self):
        """Test accuracy with special characters."""
        assert check_accuracy("It's Max!", "Max") == True
        assert check_accuracy("Max - the cat", "Max") == True

    def test_accuracy_multiline(self):
        """Test accuracy with multiline responses."""
        response = """My name is Alex.
        I am 28 years old.
        I live in Seattle."""
        assert check_accuracy(response, "Alex") == True
        assert check_accuracy(response, "Seattle") == True

    def test_stats_with_identical_values(self):
        """Test statistics with all identical values."""
        values = [0.9, 0.9, 0.9, 0.9, 0.9]
        result = analyze_sample(values)
        assert result.mean == 0.9
        assert result.std == 0.0

    def test_stats_with_large_sample(self):
        """Test statistics with large sample size."""
        import random
        random.seed(42)
        values = [random.gauss(0.9, 0.05) for _ in range(1000)]
        result = analyze_sample(values)
        assert 0.85 < result.mean < 0.95
        assert result.ci_lower < result.ci_upper


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
