"""
Feature Configuration Loader

Centralized feature definitions loaded from YAML configuration.
This module is the ONLY interface for accessing feature definitions.

Usage:
    from tools.feature_config import get_feature_list, get_numeric_features

    # Get demographic features
    features = get_feature_list('demographic')

    # Get numeric features for preprocessing
    numeric = get_numeric_features()

    # Get display name for visualization
    display = get_display_name('AGE_AT_SEQ_REPORT')  # 'Age at Sequencing'

    # Use custom config for a specific run
    with FeatureConfig('./my_features.yaml') as config:
        features = get_feature_list('demographic')

Environment Variable:
    BALDER_FEATURES_CONFIG - Override default config path

"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager

# Default config path (relative to this file)
_THIS_DIR = Path(__file__).parent
_DEFAULT_CONFIG_PATH = _THIS_DIR.parent / 'config' / 'features.yaml'

# Global config cache
_config_cache: Optional[Dict] = None
_config_path: Optional[Path] = None


def load_feature_config(path: Optional[str] = None, force_reload: bool = False) -> Dict:
    """
    Load feature configuration from YAML file.

    This is the main entry point for loading feature configuration.
    Call this at application startup or when switching to a custom config.

    Parameters
    ----------
    path : str, optional
        Path to custom config file. If None, uses default resolution:
        1. BALDER_FEATURES_CONFIG environment variable
        2. Default path: code/config/features.yaml
    force_reload : bool, default False
        Force reload even if cached.

    Returns
    -------
    dict
        Parsed YAML configuration
    """
    return _load_config(path, force_reload)


def _resolve_config_path(path: Optional[str] = None) -> Path:
    """
    Resolve config path with priority:
    1. Explicit path argument
    2. BALDER_FEATURES_CONFIG environment variable
    3. Default path: code/config/features.yaml
    """
    if path:
        return Path(path)

    env_path = os.getenv('BALDER_FEATURES_CONFIG')
    if env_path:
        return Path(env_path)

    return _DEFAULT_CONFIG_PATH


def _load_config(path: Optional[str] = None, force_reload: bool = False) -> Dict:
    """
    Load feature configuration from YAML file.

    Uses caching to avoid repeated file reads.

    Parameters
    ----------
    path : str, optional
        Path to custom config file. If None, uses cached config or default.
    force_reload : bool, default False
        Force reload even if cached.

    Returns
    -------
    dict
        Parsed YAML configuration

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If config file is invalid
    """
    global _config_cache, _config_path

    # If no path specified and we have a cached config, use it (unless force_reload)
    if path is None and _config_cache is not None and not force_reload:
        return _config_cache

    resolved_path = _resolve_config_path(path)

    # Return cached config if available and path matches
    if not force_reload and _config_cache is not None and _config_path == resolved_path:
        return _config_cache

    # Load config file
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Feature config file not found: {resolved_path}\n"
            f"Set BALDER_FEATURES_CONFIG environment variable or provide explicit path."
        )

    with open(resolved_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['feature_sets', 'feature_types']
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(
            f"Feature config missing required sections: {missing}. "
            f"File: {resolved_path}"
        )

    # Cache the config
    _config_cache = config
    _config_path = resolved_path

    return config


def get_feature_list(feature_set: str = 'demographic', config_path: Optional[str] = None) -> List[str]:
    """
    Get list of features for specified feature set.

    Parameters
    ----------
    feature_set : str, default='demographic'
        Which features to include:
        - 'demographic': Patient/sample characteristics (no data leakage)
        - 'technical': Sequencing/institutional features (may leak)
        - 'both': All features combined
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    list of str
        List of feature names

    Raises
    ------
    ValueError
        If feature_set not in ['demographic', 'technical', 'both']

    Examples
    --------
    >>> get_feature_list('demographic')
    ['SAMPLE_TYPE_DETAILED', 'SEX', 'ANCESTRY', 'ETHNICITY', 'AGE_AT_SEQ_REPORT', 'ANCESTRY_missing', 'ETHNICITY_missing']
    """
    valid_sets = ['demographic', 'technical', 'both']
    if feature_set not in valid_sets:
        raise ValueError(f"feature_set must be one of {valid_sets}, got: {feature_set}")

    config = _load_config(config_path)

    if feature_set == 'both':
        demographic = config['feature_sets']['demographic']['features']
        technical = config['feature_sets']['technical']['features']
        return demographic + technical
    else:
        return config['feature_sets'][feature_set]['features'].copy()


def get_numeric_features(config_path: Optional[str] = None) -> List[str]:
    """
    Get list of numeric feature names.

    These features are processed with median imputation in preprocessing.

    Parameters
    ----------
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    list of str
        List of numeric feature names
    """
    config = _load_config(config_path)
    return config['feature_types']['numeric']['features'].copy()


def get_categorical_features(config_path: Optional[str] = None) -> List[str]:
    """
    Get list of categorical feature names.

    These features are processed with most frequent imputation + one-hot encoding.

    Parameters
    ----------
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    list of str
        List of categorical feature names
    """
    config = _load_config(config_path)
    return config['feature_types']['categorical']['features'].copy()


def get_display_name(feature: str, config_path: Optional[str] = None) -> str:
    """
    Get human-readable display name for a feature.

    Used for SHAP visualizations and reports.

    Parameters
    ----------
    feature : str
        Original feature name (e.g., 'AGE_AT_SEQ_REPORT')
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    str
        Human-readable display name (e.g., 'Age at Sequencing')
    """
    config = _load_config(config_path)
    display_names = config.get('display_names', {})
    return display_names.get(feature, feature)


def get_value_display_name(value: str, config_path: Optional[str] = None) -> str:
    """
    Get human-readable display name for a categorical value.

    Parameters
    ----------
    value : str
        Categorical value (e.g., 'Black')
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    str
        Human-readable display name (e.g., 'Black or African American')
    """
    config = _load_config(config_path)
    value_names = config.get('value_display_names', {})
    return value_names.get(value, value)


def get_all_display_names(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get all display name mappings.

    Parameters
    ----------
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    dict
        Mapping from feature names to display names
    """
    config = _load_config(config_path)
    return config.get('display_names', {}).copy()


def get_all_value_display_names(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get all value display name mappings.

    Parameters
    ----------
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    dict
        Mapping from categorical values to display names
    """
    config = _load_config(config_path)
    return config.get('value_display_names', {}).copy()


def get_config_path() -> Optional[Path]:
    """
    Get the path to the currently loaded config file.

    Returns
    -------
    Path or None
        Path to loaded config file, or None if not yet loaded
    """
    return _config_path


def get_config_version(config_path: Optional[str] = None) -> str:
    """
    Get the version string from the config metadata.

    Parameters
    ----------
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    str
        Version string (e.g., '1.0')
    """
    config = _load_config(config_path)
    metadata = config.get('metadata', {})
    return metadata.get('version', 'unknown')


def validate_config(config_path: Optional[str] = None) -> bool:
    """
    Validate feature configuration for consistency.

    Checks:
    - All features in feature_sets have a type (numeric or categorical)
    - No duplicate features within or across sets
    - All numeric/categorical features exist in feature_sets

    Parameters
    ----------
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If validation fails with details
    """
    config = _load_config(config_path, force_reload=True)

    # Get all features from feature_sets
    all_features = set()
    for set_name, set_data in config['feature_sets'].items():
        features = set_data.get('features', [])

        # Check for duplicates within a set
        if len(features) != len(set(features)):
            duplicates = [f for f in features if features.count(f) > 1]
            raise ValueError(f"Duplicate features in {set_name}: {duplicates}")

        all_features.update(features)

    # Get all typed features
    numeric_features = set(config['feature_types']['numeric']['features'])
    categorical_features = set(config['feature_types']['categorical']['features'])
    typed_features = numeric_features | categorical_features

    # Check for overlap between numeric and categorical
    overlap = numeric_features & categorical_features
    if overlap:
        raise ValueError(f"Features appear in both numeric and categorical: {overlap}")

    # Check that all features have a type
    untyped = all_features - typed_features
    if untyped:
        raise ValueError(f"Features missing type classification: {untyped}")

    # Check that all typed features are in feature_sets
    orphaned = typed_features - all_features
    if orphaned:
        raise ValueError(f"Typed features not in any feature_set: {orphaned}")

    return True


def reload_config(config_path: Optional[str] = None) -> Dict:
    """
    Force reload of feature configuration.

    Use this after modifying the config file to pick up changes.

    Parameters
    ----------
    config_path : str, optional
        Path to custom config file

    Returns
    -------
    dict
        Reloaded configuration
    """
    return _load_config(config_path, force_reload=True)


class FeatureConfig:
    """
    Context manager for run-specific feature configurations.

    Temporarily switches to a custom config file, then restores the original.

    Example
    -------
    >>> # Use custom config for a specific experiment
    >>> with FeatureConfig('./experiment_features.yaml'):
    ...     features = get_feature_list('demographic')
    ...     # Uses custom config
    >>> # Original config restored automatically
    """

    def __init__(self, config_path: str):
        """
        Initialize with custom config path.

        Parameters
        ----------
        config_path : str
            Path to custom feature config YAML file
        """
        self.config_path = config_path
        self._previous_cache = None
        self._previous_path = None

    def __enter__(self):
        global _config_cache, _config_path

        # Save current state
        self._previous_cache = _config_cache
        self._previous_path = _config_path

        # Load new config
        _load_config(self.config_path, force_reload=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _config_cache, _config_path

        # Restore previous state
        _config_cache = self._previous_cache
        _config_path = self._previous_path

        return False


# Convenience function to copy config to output directory
def copy_config_to_run_dir(run_dir: str, config_path: Optional[str] = None) -> Path:
    """
    Copy the feature config to a run output directory for reproducibility.

    Parameters
    ----------
    run_dir : str
        Path to run output directory
    config_path : str, optional
        Path to custom config file (uses current loaded config if None)

    Returns
    -------
    Path
        Path to copied config file
    """
    import shutil

    source_path = _resolve_config_path(config_path)
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    dest_path = run_dir_path / 'features.yaml'
    shutil.copy2(source_path, dest_path)

    return dest_path


if __name__ == '__main__':
    # Self-test when run directly
    print("Feature Configuration Loader")
    print("=" * 60)

    # Load and validate config
    try:
        validate_config()
        print(f"Config version: {get_config_version()}")
        print(f"Config path: {get_config_path()}")
        print("\nValidation: PASSED")
    except ValueError as e:
        print(f"\nValidation: FAILED - {e}")
        exit(1)

    # Show feature counts
    print(f"\nDemographic features: {len(get_feature_list('demographic'))}")
    print(f"Technical features: {len(get_feature_list('technical'))}")
    print(f"All features: {len(get_feature_list('both'))}")

    print(f"\nNumeric features: {len(get_numeric_features())}")
    print(f"Categorical features: {len(get_categorical_features())}")

    # Show feature lists
    print("\n" + "-" * 60)
    print("Demographic Features:")
    for f in get_feature_list('demographic'):
        display = get_display_name(f)
        print(f"  - {f} -> {display}")

    print("\n" + "-" * 60)
    print("Technical Features:")
    for f in get_feature_list('technical'):
        display = get_display_name(f)
        print(f"  - {f} -> {display}")

    print("\n" + "-" * 60)
    print("Numeric Features:")
    for f in get_numeric_features():
        print(f"  - {f}")

    print("\n" + "-" * 60)
    print("Categorical Features:")
    for f in get_categorical_features():
        print(f"  - {f}")
