"""Analytics module for feature analysis and model learning."""

from .feature_analyzer import (
    analyze_feature_importance,
    compute_optimal_weights,
    generate_insights_report,
    FeatureAnalyzer,
)

__all__ = [
    "analyze_feature_importance",
    "compute_optimal_weights",
    "generate_insights_report",
    "FeatureAnalyzer",
]
