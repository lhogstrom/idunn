"""Exception classes for OncoKB Analyzer."""


class OncoKBAnalyzerError(Exception):
    """Base exception for OncoKB Analyzer."""
    pass


class DataLoadError(OncoKBAnalyzerError):
    """Exception raised when data loading fails."""
    pass


class ConfigurationError(OncoKBAnalyzerError):
    """Exception raised for configuration validation errors."""
    pass


class ModelingError(OncoKBAnalyzerError):
    """Exception raised during model training or evaluation."""
    pass