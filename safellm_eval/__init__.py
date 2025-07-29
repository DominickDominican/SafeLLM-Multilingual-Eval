"""
SafeLLM Multilingual Evaluation Framework

A comprehensive framework for evaluating the safety, robustness, and alignment 
of large language models across multiple languages and high-stakes domains.
"""

__version__ = "0.1.0"
__author__ = "Dominick Dominican"
__email__ = "dominickdominican47@gmail.com"

try:
    from .evaluator import MultilingualEvaluator
    from .models import ModelClient
    from .scoring import SafetyScorer
    from .visualizer import ResultVisualizer
    from .config import ConfigManager
except ImportError:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn("Some components could not be imported. Please ensure all dependencies are installed.")

__all__ = [
    "MultilingualEvaluator",
    "ModelClient", 
    "SafetyScorer",
    "ResultVisualizer",
    "ConfigManager",
]