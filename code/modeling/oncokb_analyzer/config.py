"""
Minimal Configuration System for OncoKB Analyzer

Simple dataclass-based configuration with YAML support.
No advanced features - just basic functionality for end-to-end run.

Feature definitions are loaded from code/config/features.yaml via feature_config.
Use feature_config_path parameter to specify custom feature config per run.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
from feature_config import load_feature_config, get_config_path
from pathlib import Path


@dataclass
class Config:
    """Minimal configuration for basic functionality"""

    # Database
    db_path: str

    # Analysis settings
    cancer_types: List[str]
    min_actionable_events: int = 250
    outcome_column: str = "isOncoKbLevel1"

    # Model settings
    algorithm: str = "random_forest"  # or "logistic"
    cv_folds: int = 5
    random_state: int = 42
    feature_set: str = "both"  # "demographic", "technical", or "both"

    # Cross-validation strategy
    cv_strategy: str = "stratified_group"  # "standard", "group", or "stratified_group" (recommended)
    group_column: str = "SUBJECT_CENTER"  # Column for GroupKFold/StratifiedGroupKFold (e.g., "SUBJECT_CENTER", "SEQ_ASSAY_ID")

    # Train-test split strategy
    institution_aware_test_split: bool = False  # Use GroupShuffleSplit for test splits
    test_split_group_column: str = "SUBJECT_CENTER"  # Column for institution-aware test splits (can differ from cv group_column)

    # Multi-tier grouping (optional)
    grouping_mapping_file: str = None  # Path to patient-to-group mapping CSV for multi-tier analysis

    # Train/test split (optional)
    train_test_split_file: str = None  # Path to train/test split CSV file (PATIENT_ID, split columns)

    # Parallel processing
    n_jobs: int = -1  # Number of parallel jobs (-1 uses all cores, 1 for single-threaded to save memory)

    # Feature configuration
    feature_config_path: Optional[str] = None  # Path to custom feature config YAML (default: code/config/features.yaml)

    # Output
    output_dir: str = "output/"

    def __post_init__(self):
        """Load feature configuration after initialization."""
        if self.feature_config_path:
            load_feature_config(self.feature_config_path)
        # Otherwise, default config will be loaded on first access

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Validate required fields
        required_fields = ['db_path', 'cancer_types']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from configuration")

        return cls(**data)

    @classmethod
    def default(cls) -> 'Config':
        """Default configuration for quick testing"""
        return cls(
            db_path=get_default_db_path(),
            cancer_types=["LUAD", "BRCA"],  # Start with just 2 common types
            algorithm="random_forest"
        )

    def validate(self) -> bool:
        """Basic validation of configuration parameters"""
        # Check algorithm choice
        if self.algorithm not in ["random_forest", "logistic"]:
            raise ValueError(f"Algorithm must be 'random_forest' or 'logistic', got: {self.algorithm}")

        # Check minimum events threshold
        if self.min_actionable_events < 1:
            raise ValueError(f"min_actionable_events must be positive, got: {self.min_actionable_events}")

        # Check CV folds
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be at least 2, got: {self.cv_folds}")

        # Check that cancer_types is not empty
        if not self.cancer_types:
            raise ValueError("cancer_types cannot be empty")

        # Check feature_set choice
        valid_feature_sets = ["demographic", "technical", "both"]
        if self.feature_set not in valid_feature_sets:
            raise ValueError(f"feature_set must be one of {valid_feature_sets}, got: {self.feature_set}")

        # Check cv_strategy choice
        valid_cv_strategies = ["standard", "group", "stratified_group"]
        if self.cv_strategy not in valid_cv_strategies:
            raise ValueError(f"cv_strategy must be one of {valid_cv_strategies}, got: {self.cv_strategy}")

        # Check group_column is provided when using group or stratified_group strategy
        if self.cv_strategy in ["group", "stratified_group"] and not self.group_column:
            raise ValueError(f"group_column must be specified when cv_strategy is '{self.cv_strategy}'")

        # Check test_split_group_column is provided when using institution-aware test split
        if self.institution_aware_test_split and not self.test_split_group_column:
            raise ValueError("test_split_group_column must be specified when institution_aware_test_split is True")

        return True

    def to_dict(self) -> dict:
        """Convert configuration to dictionary (useful for logging/saving)"""
        return {
            'db_path': self.db_path,
            'cancer_types': self.cancer_types,
            'min_actionable_events': self.min_actionable_events,
            'outcome_column': self.outcome_column,
            'algorithm': self.algorithm,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'feature_set': self.feature_set,
            'cv_strategy': self.cv_strategy,
            'group_column': self.group_column,
            'institution_aware_test_split': self.institution_aware_test_split,
            'test_split_group_column': self.test_split_group_column,
            'grouping_mapping_file': self.grouping_mapping_file,
            'train_test_split_file': self.train_test_split_file,
            'n_jobs': self.n_jobs,
            'feature_config_path': self.feature_config_path,
            'output_dir': self.output_dir
        }