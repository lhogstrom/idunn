"""Load and filter patient data from the SQLite database."""

import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import logging

from .preprocessing import DEMOGRAPHIC_FEATURES, TECHNICAL_FEATURES


class DataLoader:
    """Simple data loader for SQLite databases."""

    def __init__(self, db_path: str):
        """Initialize data loader."""
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        # Create engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def load_patient_data(self) -> pd.DataFrame:
        """Load patient data from SQLite database."""
        try:
            self.logger.info(f"Loading patient data from {self.db_path}")

            # Load from the main patient table.
            data = pd.read_sql_table('perPatientOncoKBFilteredForModeling', con=self.engine)
            # Force plain str column names; sqlalchemy quoted_name breaks sklearn feature-name checks.
            data.columns = [str(c) for c in data.columns]
            if 'RACE' in data.columns:
                data = data.rename(columns={'RACE': 'ANCESTRY'})

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
        """Filter data for specific cancer type."""
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
        """Get available cancer types with patient counts."""
        cancer_counts = data['ONCOTREE_CODE'].value_counts()
        return cancer_counts[cancer_counts >= min_patients]

    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """Get basic summary statistics of the dataset."""
        summary = {
            'total_patients': len(data),
            'total_cancer_types': data['ONCOTREE_CODE'].nunique(),
            'actionable_patients': data['isOncoKbLevel1'].sum(),
            'actionable_rate': data['isOncoKbLevel1'].mean(),
            'available_columns': data.columns.tolist(),
            'missing_data_summary': data.isnull().sum().to_dict()
        }

        return summary

    def load_multiple_cancer_types(self, cancer_types: list, patient_ids_filter: list = None) -> dict:
        """Load data for multiple cancer types efficiently."""
        # Load full dataset once
        self.logger.info(f"Loading data for {len(cancer_types)} cancer types: {cancer_types}")
        full_data = self.load_patient_data()

        # Apply patient ID filter if provided (for train/test split)
        if patient_ids_filter is not None:
            original_count = len(full_data)
            full_data = full_data[full_data['PATIENT_ID'].isin(patient_ids_filter)].copy()
            self.logger.info(
                f"Applied patient ID filter: {original_count:,} -> {len(full_data):,} patients "
                f"({len(full_data)/original_count:.1%} retained)"
            )

        # Check which cancer types are available
        available_types = full_data['ONCOTREE_CODE'].unique()
        missing_types = [ct for ct in cancer_types if ct not in available_types]

        if missing_types:
            available_counts = full_data['ONCOTREE_CODE'].value_counts().head(10)
            raise ValueError(
                f"Cancer types not found in data: {missing_types}\n"
                f"Available types (top 10): {available_counts.index.tolist()}"
            )

        # Filter for each cancer type
        data_dict = {}
        for cancer_type in cancer_types:
            filtered_data = self.filter_for_cancer_type(full_data, cancer_type)
            data_dict[cancer_type] = filtered_data

        self.logger.info(
            f"Loaded {len(data_dict)} cancer types. "
            f"Total patients: {sum(len(df) for df in data_dict.values())}"
        )

        return data_dict

    def get_feature_matrix(self, data: pd.DataFrame, feature_set: str = "demographic",
                           outcome_col: str = "isOncoKbLevel1",
                           group_column: str = "SUBJECT_CENTER"):
        """Extract X (features), y (target), groups from DataFrame."""
        # Validate outcome column exists
        if outcome_col not in data.columns:
            raise ValueError(
                f"Outcome column '{outcome_col}' not found in data. "
                f"Available columns: {data.columns.tolist()}"
            )

        # Validate feature_set parameter
        valid_feature_sets = ["demographic", "technical", "both"]
        if feature_set not in valid_feature_sets:
            raise ValueError(
                f"feature_set must be one of {valid_feature_sets}, got: {feature_set}"
            )

        # Feature lists come from preprocessing.py.
        if feature_set == "demographic":
            feature_list = DEMOGRAPHIC_FEATURES.copy()
        elif feature_set == "technical":
            feature_list = TECHNICAL_FEATURES.copy()
        else:  # feature_set == "both"
            feature_list = DEMOGRAPHIC_FEATURES + TECHNICAL_FEATURES

        # Filter to columns that actually exist in the data
        available_cols = [col for col in feature_list if col in data.columns]

        if not available_cols:
            raise ValueError(
                f"No feature columns found in data for feature_set='{feature_set}'. "
                f"Available columns: {data.columns.tolist()}"
            )

        self.logger.info(
            f"Extracting {len(available_cols)} {feature_set} features from {len(data)} samples"
        )

        # Extract feature matrix (preserve DataFrame for column names)
        X = data[available_cols].copy()

        # Extract target
        y = data[outcome_col].copy()

        # Extract groups for GroupKFold
        if group_column not in data.columns:
            self.logger.warning(
                f"Group column '{group_column}' not found in data. "
                f"Groups will be None. Available columns: {data.columns.tolist()}"
            )
            groups = None
        else:
            groups = data[group_column].copy()
            self.logger.info(
                f"Extracted {groups.nunique()} unique groups from '{group_column}' column"
            )

        # Log basic statistics
        self.logger.info(
            f"Feature matrix shape: {X.shape}, "
            f"Target distribution: {y.value_counts().to_dict()}"
        )

        return X, y, groups

    def prepare_cancer_data_dict(self, cancer_types: list, feature_set: str = "demographic",
                                 outcome_col: str = "isOncoKbLevel1",
                                 group_column: str = "SUBJECT_CENTER",
                                 patient_ids_filter: list = None) -> dict:
        """Prepare data in format needed by NestedCVOptimizer."""
        filter_msg = ""
        if patient_ids_filter is not None:
            filter_msg = f" (filtered to {len(patient_ids_filter):,} patient IDs)"

        self.logger.info(
            f"Preparing cancer data dictionary for {len(cancer_types)} cancer types "
            f"with feature_set='{feature_set}'{filter_msg}"
        )

        # Load data for all cancer types
        data_dict = self.load_multiple_cancer_types(cancer_types, patient_ids_filter=patient_ids_filter)

        # Convert each DataFrame to (X, y, groups) format
        cancer_data_dict = {}
        for cancer_type, data in data_dict.items():
            X, y, groups = self.get_feature_matrix(
                data,
                feature_set=feature_set,
                outcome_col=outcome_col,
                group_column=group_column
            )
            cancer_data_dict[cancer_type] = (X, y, groups)

            self.logger.info(
                f"{cancer_type}: {X.shape[0]} samples, {X.shape[1]} features, "
                f"{y.sum()} actionable ({y.mean():.1%})"
            )

        self.logger.info(
            f"Cancer data dictionary prepared with {len(cancer_data_dict)} cancer types. "
            f"Total samples: {sum(X.shape[0] for X, y, groups in cancer_data_dict.values())}"
        )

        return cancer_data_dict

    def prepare_cancer_data_dict_multi_tier(
        self,
        grouping_mapping_file: str,
        feature_set: str = 'demographic',
        outcome_col: str = 'isOncoKbLevel1',
        group_column: str = 'SUBJECT_CENTER',
        patient_ids_filter: list = None
    ) -> dict:
        """Load data for all multi-tier cancer type groupings."""
        self.logger.info(
            f"Loading multi-tier groupings from {grouping_mapping_file}"
        )

        # Load patient-to-group mapping
        mapping_path = Path(grouping_mapping_file)
        if not mapping_path.exists():
            raise FileNotFoundError(f"Grouping mapping file not found: {mapping_path}")

        mapping_df = pd.read_csv(mapping_path)

        # Validate expected columns
        required_cols = ['PATIENT_ID', 'tier_prefixed_code']
        missing_cols = [col for col in required_cols if col not in mapping_df.columns]
        if missing_cols:
            raise ValueError(
                f"Grouping mapping file missing required columns: {missing_cols}. "
                f"Available columns: {mapping_df.columns.tolist()}"
            )

        # Get unique groups
        unique_groups = mapping_df['tier_prefixed_code'].unique()
        self.logger.info(f"Found {len(unique_groups)} unique multi-tier groups")

        # Load full patient data once
        full_data = self.load_patient_data()

        # Apply patient ID filter if provided (for train/test split)
        if patient_ids_filter is not None:
            original_count = len(full_data)
            full_data = full_data[full_data['PATIENT_ID'].isin(patient_ids_filter)].copy()
            self.logger.info(
                f"Applied patient ID filter: {len(full_data):,} / {original_count:,} patients (training set)"
            )

        # Prepare data for each group
        cancer_data_dict = {}

        for tier_prefixed_code in sorted(unique_groups):
            # Get patient IDs for this group
            patient_ids = mapping_df[
                mapping_df['tier_prefixed_code'] == tier_prefixed_code
            ]['PATIENT_ID'].values

            # Filter full data to patients in this group
            group_data = full_data[full_data['PATIENT_ID'].isin(patient_ids)].copy()

            if len(group_data) == 0:
                self.logger.warning(
                    f"No patients found for group {tier_prefixed_code} "
                    f"(mapping has {len(patient_ids)} patient IDs)"
                )
                continue

            # Extract feature matrix
            try:
                X, y, groups = self.get_feature_matrix(
                    group_data,
                    feature_set=feature_set,
                    outcome_col=outcome_col,
                    group_column=group_column
                )

                cancer_data_dict[tier_prefixed_code] = (X, y, groups)

                self.logger.info(
                    f"{tier_prefixed_code}: {X.shape[0]} samples, {X.shape[1]} features, "
                    f"{y.sum()} actionable ({y.mean():.1%})"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to extract features for {tier_prefixed_code}: {str(e)}"
                )
                continue

        total_samples = sum(X.shape[0] for X, y, groups in cancer_data_dict.values())
        unique_patients = mapping_df['PATIENT_ID'].nunique()

        self.logger.info(
            f"Multi-tier data dictionary prepared with {len(cancer_data_dict)} groups. "
            f"Total samples: {total_samples} (from {unique_patients} unique patients). "
            f"Overlap ratio: {total_samples / unique_patients:.2f}x"
        )

        return cancer_data_dict