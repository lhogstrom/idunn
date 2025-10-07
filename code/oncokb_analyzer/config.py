"""
Minimal Configuration System for OncoKB Analyzer

Simple dataclass-based configuration with YAML support.
No advanced features - just basic functionality for end-to-end run.
"""

from dataclasses import dataclass
from typing import List
import yaml
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

    # Output
    output_dir: str = "output/"

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
            db_path="/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite",
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
            'output_dir': self.output_dir
        }