"""
Simple Data Loader for OncoKB Analyzer

Basic SQLite data loading functionality.
No caching, connection pooling, or advanced features - just load and filter.
"""

import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import logging


class DataLoader:
    """Simple data loader for SQLite databases"""

    def __init__(self, db_path: str):
        """
        Initialize data loader.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        # Create engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def load_patient_data(self) -> pd.DataFrame:
        """
        Load patient data from SQLite database.

        Returns:
            DataFrame containing patient-level data

        Raises:
            Exception: If data loading fails
        """
        try:
            self.logger.info(f"Loading patient data from {self.db_path}")

            # Load from the main patient table
            data = pd.read_sql_table('perPatientOncoKBFilteredForModeling', con=self.engine)

            self.logger.info(f"Loaded {len(data)} patient records with {data.shape[1]} columns")

            # Basic data validation
            if data.empty:
                raise ValueError("Loaded data is empty")

            # Check for required columns
            required_columns = ['ONCOTREE_CODE', 'isOncoKbLevel1']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            return data

        except Exception as e:
            self.logger.error(f"Failed to load patient data: {str(e)}")
            raise

    def filter_for_cancer_type(self, data: pd.DataFrame, cancer_type: str) -> pd.DataFrame:
        """
        Filter data for specific cancer type.

        Args:
            data: Full patient dataset
            cancer_type: OncoTree code for cancer type

        Returns:
            Filtered DataFrame for specified cancer type

        Raises:
            ValueError: If cancer type not found in data
        """
        if cancer_type not in data['ONCOTREE_CODE'].values:
            available_types = data['ONCOTREE_CODE'].value_counts().head(10)
            raise ValueError(
                f"Cancer type '{cancer_type}' not found in data. "
                f"Available types (top 10): {available_types.index.tolist()}"
            )

        filtered_data = data[data['ONCOTREE_CODE'] == cancer_type].copy()

        self.logger.info(
            f"Filtered to {len(filtered_data)} patients for cancer type {cancer_type}"
        )

        return filtered_data

    def get_available_cancer_types(self, data: pd.DataFrame, min_patients: int = 50) -> pd.Series:
        """
        Get available cancer types with patient counts.

        Args:
            data: Patient dataset
            min_patients: Minimum number of patients required

        Returns:
            Series with cancer types and patient counts (filtered by min_patients)
        """
        cancer_counts = data['ONCOTREE_CODE'].value_counts()
        return cancer_counts[cancer_counts >= min_patients]

    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Get basic summary statistics of the dataset.

        Args:
            data: Patient dataset

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_patients': len(data),
            'total_cancer_types': data['ONCOTREE_CODE'].nunique(),
            'actionable_patients': data['isOncoKbLevel1'].sum(),
            'actionable_rate': data['isOncoKbLevel1'].mean(),
            'available_columns': data.columns.tolist(),
            'missing_data_summary': data.isnull().sum().to_dict()
        }

        return summary