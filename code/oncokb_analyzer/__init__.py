"""
OncoKB Analyzer Package

A modular, extensible framework for analyzing OncoKB actionability in cancer genomics data.
Addresses methodological issues including data leakage, statistical rigor, and clinical utility.
"""

from .analyzer import OncoKBAnalyzer
from .config import Config
from .utils import OncoKBAnalyzerError, DataLoadError, ConfigurationError, ModelingError

__version__ = "2.0.0"
__author__ = "OncoKB Analysis Team"

__all__ = [
    "OncoKBAnalyzer",
    "Config",
    "OncoKBAnalyzerError",
    "DataLoadError",
    "ConfigurationError",
    "ModelingError"
]