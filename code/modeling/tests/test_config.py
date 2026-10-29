"""Unit tests for config.py

Tests configuration initialization, validation, and YAML loading.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncokb_analyzer.config import Config


class TestConfigInitialization:
    """Tests for Config class initialization."""

    def test_default_config_creation(self):
        """Test that default config can be created."""
        config = Config.default()

        assert config is not None
        assert config.algorithm == "random_forest"
        assert config.cv_folds == 5
        assert config.random_state == 42
        assert len(config.cancer_types) > 0

    def test_config_with_custom_values(self):
        """Test config creation with custom values."""
        config = Config(
            db_path="/fake/path/db.sqlite",
            cancer_types=["LUAD", "BRCA", "MEL"],
            algorithm="logistic",
            cv_folds=10,
            random_state=123,
            feature_set="demographic"
        )

        assert config.db_path == "/fake/path/db.sqlite"
        assert config.cancer_types == ["LUAD", "BRCA", "MEL"]
        assert config.algorithm == "logistic"
        assert config.cv_folds == 10
        assert config.random_state == 123
        assert config.feature_set == "demographic"

    def test_config_default_values(self):
        """Test that default values are set correctly."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD"]
        )

        assert config.min_actionable_events == 250
        assert config.outcome_column == "isOncoKbLevel1"
        assert config.cv_strategy == "stratified_group"
        assert config.group_column == "SUBJECT_CENTER"
        assert config.n_jobs == -1


class TestConfigValidation:
    """Tests for Config.validate() method."""

    def test_valid_config_passes_validation(self):
        """Test that a valid config passes validation."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD", "BRCA"]
        )

        assert config.validate() is True

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm raises ValueError."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD"],
            algorithm="invalid_algorithm"
        )

        with pytest.raises(ValueError, match="Algorithm must be"):
            config.validate()

    def test_invalid_feature_set_raises_error(self):
        """Test that invalid feature_set raises ValueError."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD"],
            feature_set="invalid_set"
        )

        with pytest.raises(ValueError, match="feature_set must be"):
            config.validate()

    def test_invalid_cv_strategy_raises_error(self):
        """Test that invalid cv_strategy raises ValueError."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD"],
            cv_strategy="invalid_strategy"
        )

        with pytest.raises(ValueError, match="cv_strategy must be"):
            config.validate()

    def test_empty_cancer_types_raises_error(self):
        """Test that empty cancer_types raises ValueError."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=[]
        )

        with pytest.raises(ValueError, match="cancer_types cannot be empty"):
            config.validate()

    def test_invalid_cv_folds_raises_error(self):
        """Test that cv_folds < 2 raises ValueError."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD"],
            cv_folds=1
        )

        with pytest.raises(ValueError, match="cv_folds must be at least 2"):
            config.validate()

    def test_invalid_min_actionable_events_raises_error(self):
        """Test that min_actionable_events < 1 raises ValueError."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD"],
            min_actionable_events=0
        )

        with pytest.raises(ValueError, match="min_actionable_events must be positive"):
            config.validate()

    def test_group_column_required_for_group_cv(self):
        """Test that group_column is required for group CV strategies."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD"],
            cv_strategy="group",
            group_column=""  # Empty string
        )

        with pytest.raises(ValueError, match="group_column must be specified"):
            config.validate()


class TestConfigYamlLoading:
    """Tests for Config.from_yaml() method."""

    def test_load_from_valid_yaml(self):
        """Test loading config from valid YAML file."""
        yaml_content = {
            'db_path': '/fake/db.sqlite',
            'cancer_types': ['LUAD', 'BRCA'],
            'algorithm': 'logistic',
            'cv_folds': 3
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)

            assert config.db_path == '/fake/db.sqlite'
            assert config.cancer_types == ['LUAD', 'BRCA']
            assert config.algorithm == 'logistic'
            assert config.cv_folds == 3
        finally:
            Path(temp_path).unlink()

    def test_load_from_nonexistent_yaml_raises_error(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml('/nonexistent/path/config.yaml')

    def test_load_from_yaml_missing_required_fields(self):
        """Test that missing required fields raise ValueError."""
        yaml_content = {
            'algorithm': 'random_forest'
            # Missing db_path and cancer_types
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Required field"):
                Config.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()


class TestConfigToDict:
    """Tests for Config.to_dict() method."""

    def test_to_dict_returns_all_fields(self):
        """Test that to_dict returns all configuration fields."""
        config = Config(
            db_path="/fake/db.sqlite",
            cancer_types=["LUAD"]
        )

        config_dict = config.to_dict()

        assert 'db_path' in config_dict
        assert 'cancer_types' in config_dict
        assert 'algorithm' in config_dict
        assert 'cv_folds' in config_dict
        assert 'feature_set' in config_dict
        assert 'cv_strategy' in config_dict

    def test_to_dict_values_match(self):
        """Test that to_dict values match config attributes."""
        config = Config(
            db_path="/test/db.sqlite",
            cancer_types=["MEL"],
            algorithm="logistic",
            feature_set="technical"
        )

        config_dict = config.to_dict()

        assert config_dict['db_path'] == "/test/db.sqlite"
        assert config_dict['cancer_types'] == ["MEL"]
        assert config_dict['algorithm'] == "logistic"
        assert config_dict['feature_set'] == "technical"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
