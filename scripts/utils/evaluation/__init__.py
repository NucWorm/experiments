"""
Centroid evaluation utilities for nucleus detection methods.

This module provides comprehensive evaluation tools for methods that output
centroid coordinates, supporting both isotropic and anisotropic distance thresholds.
"""

from .evaluate_centroids import (
    evaluate_isotropic,
    evaluate_anisotropic,
    load_ground_truth_csv,
    load_predictions_npy,
    load_predictions_csv,
    find_matching_files,
    evaluate_single_volume,
    print_summary
)

__version__ = "1.0.0"
__author__ = "NucWorm Benchmark Team"

__all__ = [
    "evaluate_isotropic",
    "evaluate_anisotropic", 
    "load_ground_truth_csv",
    "load_predictions_npy",
    "load_predictions_csv",
    "find_matching_files",
    "evaluate_single_volume",
    "print_summary"
]
