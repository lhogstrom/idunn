"""
Unit tests for utils.py

Tests custom exception classes.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncokb_analyzer.utils import (
    OncoKBAnalyzerError,
    DataLoadError,
    ConfigurationError,
    ModelingError
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_base_exception_exists(self):
        """Test that base exception class exists."""
        error = OncoKBAnalyzerError("Test error")
        assert isinstance(error, Exception)

    def test_data_load_error_inherits_base(self):
        """Test DataLoadError inherits from OncoKBAnalyzerError."""
        error = DataLoadError("Data load failed")
        assert isinstance(error, OncoKBAnalyzerError)
        assert isinstance(error, Exception)

    def test_configuration_error_inherits_base(self):
        """Test ConfigurationError inherits from OncoKBAnalyzerError."""
        error = ConfigurationError("Config invalid")
        assert isinstance(error, OncoKBAnalyzerError)
        assert isinstance(error, Exception)

    def test_modeling_error_inherits_base(self):
        """Test ModelingError inherits from OncoKBAnalyzerError."""
        error = ModelingError("Model training failed")
        assert isinstance(error, OncoKBAnalyzerError)
        assert isinstance(error, Exception)


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_base_exception_message(self):
        """Test that base exception captures message."""
        message = "Custom error message"
        error = OncoKBAnalyzerError(message)
        assert str(error) == message

    def test_data_load_error_message(self):
        """Test DataLoadError captures message."""
        message = "Failed to load database"
        error = DataLoadError(message)
        assert str(error) == message

    def test_configuration_error_message(self):
        """Test ConfigurationError captures message."""
        message = "Invalid configuration parameter"
        error = ConfigurationError(message)
        assert str(error) == message

    def test_modeling_error_message(self):
        """Test ModelingError captures message."""
        message = "Insufficient training data"
        error = ModelingError(message)
        assert str(error) == message


class TestExceptionRaising:
    """Tests for raising and catching exceptions."""

    def test_raise_and_catch_base(self):
        """Test raising and catching base exception."""
        with pytest.raises(OncoKBAnalyzerError):
            raise OncoKBAnalyzerError("Test")

    def test_raise_and_catch_data_load(self):
        """Test raising and catching DataLoadError."""
        with pytest.raises(DataLoadError):
            raise DataLoadError("Database not found")

        # Should also be catchable as base exception
        with pytest.raises(OncoKBAnalyzerError):
            raise DataLoadError("Database not found")

    def test_raise_and_catch_configuration(self):
        """Test raising and catching ConfigurationError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid setting")

    def test_raise_and_catch_modeling(self):
        """Test raising and catching ModelingError."""
        with pytest.raises(ModelingError):
            raise ModelingError("Training failed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
