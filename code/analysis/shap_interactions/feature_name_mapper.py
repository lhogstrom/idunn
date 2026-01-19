"""
Feature Name Mapper for SHAP Interaction Analysis

Converts sklearn's transformed feature names (e.g., 'num__x0', 'cat__x1_Male')
to human-readable names for better interpretability in visualizations and outputs.

Author: Balder Analysis Team
Date: December 2025
"""

import re
from typing import List, Dict, Optional


class FeatureNameMapper:
    """
    Maps transformed feature names back to human-readable versions.

    Handles sklearn's ColumnTransformer naming conventions:
    - Numeric features: 'num__x0', 'num__x1', ... → AGE_AT_SEQ_REPORT, etc.
    - Categorical features: 'cat__x0_value', 'cat__x1_Male', ... → SEX_Male, RACE_Asian, etc.

    Example:
        >>> mapper = FeatureNameMapper(original_features)
        >>> mapper.map_feature_name('num__x0')
        'Age at Sequencing'
        >>> mapper.map_feature_name('cat__x1_Male')
        'Sex: Male'
    """

    # Human-readable display names for demographic and technical features
    DISPLAY_NAMES = {
        # Demographic features
        'AGE_AT_SEQ_REPORT': 'Age at Sequencing',
        'SEX': 'Sex',
        'RACE': 'Race',
        'ETHNICITY': 'Ethnicity',
        'SAMPLE_TYPE': 'Sample Type',
        'SAMPLE_TYPE_DETAILED': 'Sample Type (Detailed)',
        # Technical features - categorical
        'SUBJECT_CENTER': 'Institution',
        'SEQ_ASSAY_ID': 'Sequencing Assay',
        'SEQ_YEAR': 'Sequencing Year',
        'SEQ_PIPELINE_ID': 'Sequencing Pipeline',
        'library_selection': 'Library Selection',
        'calling_strategy': 'Variant Calling Strategy',
        'target_capture_kit': 'Capture Kit',
        'platform': 'Sequencing Platform',
        'sample_preservation': 'Sample Preservation',
        'tumor_purity_bin': 'Tumor Purity',
        # Technical features - numeric
        'read_length': 'Read Length',
        'gene_padding': 'Gene Padding',
        'number_of_genes': 'Number of Genes',
    }

    # Value-specific display names
    VALUE_DISPLAY_NAMES = {
        'Male': 'Male',
        'Female': 'Female',
        'White': 'White',
        'Black': 'Black or African American',
        'Asian': 'Asian',
        'American Indian or Alaska Native': 'Native American',
        'Native Hawaiian or Other Pacific Islander': 'Pacific Islander',
        'Other': 'Other',
        'Unknown': 'Unknown',
        'Not Hispanic or Latino': 'Non-Hispanic',
        'Hispanic or Latino': 'Hispanic',
        'Metastatic': 'Metastatic',
        'Primary': 'Primary',
        'Local recurrence': 'Local Recurrence'
    }

    def __init__(self, original_feature_names: List[str],
                 numeric_features: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None):
        """
        Initialize the mapper with original feature names.

        Args:
            original_feature_names: List of original column names before transformation
            numeric_features: List of numeric feature names (in order)
            categorical_features: List of categorical feature names (in order)
        """
        self.original_features = original_feature_names

        # Auto-detect numeric vs categorical if not provided
        if numeric_features is None or categorical_features is None:
            self.numeric_features, self.categorical_features = self._auto_detect_feature_types()
        else:
            self.numeric_features = numeric_features
            self.categorical_features = categorical_features

        # Create mapping indices
        self._create_mapping_indices()

    # Known numeric features from preprocessing.py
    # IMPORTANT: These must match exactly what's in preprocessing.py NUMERIC_FEATURES
    KNOWN_NUMERIC_FEATURES = {
        'AGE_AT_SEQ_REPORT',
        'read_length',
        'gene_padding',
        'number_of_genes',
    }

    def _auto_detect_feature_types(self):
        """
        Auto-detect which features are numeric vs categorical based on known features.

        IMPORTANT: Order matters! Features must be in the same order as they appear
        in the original feature list to match sklearn's ColumnTransformer indexing.

        Uses exact matching against KNOWN_NUMERIC_FEATURES to avoid false positives
        (e.g., 'coverage' containing 'AGE' substring).
        """
        numeric = []
        categorical = []

        for feat in self.original_features:
            # Use exact matching against known numeric features only
            if feat in self.KNOWN_NUMERIC_FEATURES:
                numeric.append(feat)
            else:
                categorical.append(feat)

        return numeric, categorical

    def _create_mapping_indices(self):
        """Create index mappings for quick lookup.

        IMPORTANT: sklearn's ColumnTransformer uses ORIGINAL column indices,
        not sequential indices. So num__x5 means the 5th column in original
        feature list, not the 5th numeric feature.
        """
        # Map original index → feature name for numeric features
        # e.g., if AGE_AT_SEQ_REPORT is at position 5 in original_features,
        # then num__x5 → AGE_AT_SEQ_REPORT
        self.num_index_to_feature = {}
        self.cat_index_to_feature = {}

        for orig_idx, feat in enumerate(self.original_features):
            if feat in self.numeric_features:
                self.num_index_to_feature[orig_idx] = feat
            elif feat in self.categorical_features:
                self.cat_index_to_feature[orig_idx] = feat

    def map_feature_name(self, transformed_name: str,
                         include_value: bool = True,
                         short_form: bool = False) -> str:
        """
        Map transformed feature name to human-readable version.

        IMPORTANT: sklearn's ColumnTransformer does NOT preserve feature order in its indices!
        The cat__xN index does NOT correspond to the Nth categorical feature.
        Instead, we need to infer the original feature from the VALUE itself.

        Args:
            transformed_name: Sklearn transformed name (e.g., 'num__x0', 'cat__x1_Male')
            include_value: For categorical features, include the value (e.g., 'Sex: Male' vs 'Sex')
            short_form: Use abbreviated version (e.g., 'Age' vs 'Age at Sequencing')

        Returns:
            Human-readable feature name
        """
        # Handle numeric features: num__x0, num__x1, etc.
        num_match = re.match(r'num__x(\d+)', transformed_name)
        if num_match:
            index = int(num_match.group(1))
            if index in self.num_index_to_feature:
                original_name = self.num_index_to_feature[index]
                display_name = self.DISPLAY_NAMES.get(original_name, original_name)
                if short_form:
                    return display_name.split()[0]  # First word only
                return display_name

        # Handle categorical features: cat__x0_Male, cat__x1_Asian, etc.
        # Instead of using the index, infer the feature from the value
        cat_match = re.match(r'cat__x\d+_(.*)', transformed_name)
        if cat_match:
            value = cat_match.group(1)

            # Infer feature name from value
            original_name = self._infer_feature_from_value(value)

            feature_display = self.DISPLAY_NAMES.get(original_name, original_name)
            value_display = self.VALUE_DISPLAY_NAMES.get(value, value)

            if short_form:
                feature_display = feature_display.split()[0]  # First word only

            if include_value:
                return f"{feature_display}: {value_display}"
            else:
                return feature_display

        # Fallback: return original if no pattern matched
        return transformed_name

    def _infer_feature_from_value(self, value: str) -> str:
        """
        Infer the original feature name from the categorical value.

        This is necessary because sklearn's ColumnTransformer indices don't match feature order.
        """
        # SAMPLE_TYPE_DETAILED values (check first - most specific)
        if any(term in value for term in ['tumor', 'Metastasis site', 'recurrence', 'Lymph node',
                                           'hematologic malignancy', 'Not applicable or',
                                           'Primary tumor', 'Not otherwise specified']):
            return 'SAMPLE_TYPE_DETAILED'

        # SEX values (Male/Female are unique to SEX)
        if value in ['Male', 'Female']:
            return 'SEX'

        # RACE values (these are unique to RACE)
        if value in ['European', 'Asian', 'African', 'White', 'Black', 'Native American',
                     'American Indian or Alaska Native', 'Native Hawaiian or Other Pacific Islander']:
            return 'RACE'

        # ETHNICITY values (these are unique to ETHNICITY)
        if value in ['Non-Spanish/non-Hispanic', 'Spanish/Hispanic', 'Not Hispanic or Latino',
                     'Hispanic or Latino']:
            return 'ETHNICITY'

        # SAMPLE_TYPE values (simple, short values)
        if value in ['Primary', 'Metastasis', 'Not Applicable or Heme', 'Local recurrence',
                     'Not Collected', 'Unspecified']:
            return 'SAMPLE_TYPE'

        # ============ TECHNICAL FEATURES ============

        # SUBJECT_CENTER - Institution codes (check specific codes first)
        institution_codes = ['MSK', 'DFCI', 'JHU', 'VICC', 'UHN', 'GRCC', 'MDA', 'CHOP', 'NKI',
                            'YALE', 'UCSF', 'COLU', 'WAKE', 'PROV', 'SCI', 'GENIE', 'PHS', 'CRUK',
                            'DUKE', 'VHIO', 'UCHI']
        if value in institution_codes:
            return 'SUBJECT_CENTER'

        # SEQ_ASSAY_ID - Sequencing assay identifiers
        # These typically contain institution codes + panel info
        if any(pattern in value for pattern in ['IMPACT', '-GP', 'GENE', 'PANEL', 'TSO', 'F1',
                                                  'ONCOHEME', 'ONCOPANEL', 'CANCERPLEX',
                                                  'FOUNDATION', 'TEMPUS', 'GUARDANT']):
            return 'SEQ_ASSAY_ID'
        # SEQ_ASSAY_ID patterns: institution-number format (e.g., JHU-50GP, MSK-IMPACT468)
        if re.match(r'^[A-Z]+-\d+', value) or re.match(r'^[A-Z]+\d+', value):
            return 'SEQ_ASSAY_ID'

        # target_capture_kit - Kit names (contain "Panel", "Kit", "Genes", "Custom")
        if any(term in value for term in ['Panel', 'Kit', 'Genes', 'Custom', 'Capture',
                                           'Amplicon', 'Exome', 'WES', 'WGS']):
            return 'target_capture_kit'

        # library_selection - Library preparation methods
        if value in ['PCR', 'Hybrid Selection', 'Hybrid_Selection', 'Random', 'PolyA',
                     'cDNA', 'RANDOM', 'other', 'Affinity Enrichment', 'unspecified']:
            return 'library_selection'

        # calling_strategy - Variant calling strategies
        if value in ['tumor_only', 'tumor_normal', 'hotspot_regions', 'full_exome',
                     'targeted', 'wgs', 'wes', 'panel']:
            return 'calling_strategy'

        # SEQ_PIPELINE_ID - Pipeline identifiers (often contain version numbers or institution)
        if any(pattern in value.lower() for pattern in ['pipeline', 'v1', 'v2', 'v3', 'version']):
            return 'SEQ_PIPELINE_ID'

        # Sequencing platform
        if value in ['Ion Torrent', 'Illumina', 'PacBio', 'ONT', 'NovaSeq', 'HiSeq', 'MiSeq']:
            return 'platform'

        # Sample preservation - FFPE, fresh frozen, etc.
        if any(term in value for term in ['FFPE', 'frozen', 'fresh', 'Fixed', 'Paraffin']):
            return 'sample_preservation'

        # Tumor purity patterns (e.g., ">30%", "<20%")
        if re.match(r'^[<>=]?\d+%?$', value):
            return 'tumor_purity_bin'

        # Handle ambiguous "Unknown" and "Other" by context
        if value == 'Unknown':
            # "Unknown" could be SEX, RACE, ETHNICITY, or SAMPLE_TYPE
            # Without more context, default to SEX (most common)
            return 'SEX'

        if value == 'Other':
            # "Other" most commonly appears in RACE
            return 'RACE'

        # Default: return value itself if we can't determine
        return f'UNKNOWN_FEATURE_{value}'

    def map_all_feature_names(self, transformed_names: List[str],
                               include_value: bool = True,
                               short_form: bool = False) -> List[str]:
        """
        Map all transformed feature names to human-readable versions.

        Args:
            transformed_names: List of sklearn transformed names
            include_value: For categorical features, include the value
            short_form: Use abbreviated versions

        Returns:
            List of human-readable feature names
        """
        return [self.map_feature_name(name, include_value, short_form)
                for name in transformed_names]

    def create_interaction_label(self, feature_i: str, feature_j: str,
                                   separator: str = ' × ') -> str:
        """
        Create a human-readable label for an interaction pair.

        Args:
            feature_i: First feature (transformed name)
            feature_j: Second feature (transformed name)
            separator: String to separate feature names

        Returns:
            Human-readable interaction label (e.g., 'Age at Sequencing × Sex: Male')
        """
        name_i = self.map_feature_name(feature_i, include_value=True, short_form=False)
        name_j = self.map_feature_name(feature_j, include_value=True, short_form=False)

        return f"{name_i}{separator}{name_j}"

    def simplify_interaction_label(self, feature_i: str, feature_j: str,
                                     separator: str = ' × ') -> str:
        """
        Create a simplified interaction label (removes category values for cleaner display).

        Useful for heatmap axis labels where space is limited.

        Args:
            feature_i: First feature (transformed name)
            feature_j: Second feature (transformed name)
            separator: String to separate feature names

        Returns:
            Simplified interaction label (e.g., 'Age × Sex')
        """
        # Get base feature names without values
        name_i = self.map_feature_name(feature_i, include_value=False, short_form=True)
        name_j = self.map_feature_name(feature_j, include_value=False, short_form=True)

        return f"{name_i}{separator}{name_j}"

    def get_feature_group(self, transformed_name: str) -> str:
        """
        Get the feature group (base feature name) for a transformed feature.

        Useful for grouping one-hot encoded features.

        Args:
            transformed_name: Sklearn transformed name

        Returns:
            Base feature name (e.g., 'cat__x1_Male' → 'SEX')
        """
        # Numeric features
        num_match = re.match(r'num__x(\d+)', transformed_name)
        if num_match:
            index = int(num_match.group(1))
            return self.num_index_to_feature.get(index, transformed_name)

        # Categorical features
        cat_match = re.match(r'cat__x(\d+)_', transformed_name)
        if cat_match:
            index = int(cat_match.group(1))
            return self.cat_index_to_feature.get(index, transformed_name)

        return transformed_name

    def get_feature_type(self, feature_name: str) -> str:
        """
        Classify a feature as 'demographic' or 'technical'.

        Args:
            feature_name: Original feature name (e.g., 'SEX', 'SUBJECT_CENTER')
                         or transformed name (e.g., 'cat__x1_Male')

        Returns:
            'demographic' or 'technical'
        """
        # If it's a transformed name, get the base feature first
        if feature_name.startswith('num__') or feature_name.startswith('cat__'):
            # For categorical, infer from value
            cat_match = re.match(r'cat__x\d+_(.*)', feature_name)
            if cat_match:
                value = cat_match.group(1)
                feature_name = self._infer_feature_from_value(value)
            else:
                # For numeric, use index mapping
                num_match = re.match(r'num__x(\d+)', feature_name)
                if num_match:
                    index = int(num_match.group(1))
                    feature_name = self.num_index_to_feature.get(index, feature_name)

        return classify_feature_type(feature_name)


# Standalone function for feature type classification
DEMOGRAPHIC_FEATURES = {
    'AGE_AT_SEQ_REPORT', 'SEX', 'RACE', 'ETHNICITY',
    'SAMPLE_TYPE', 'SAMPLE_TYPE_DETAILED',
    'Age at Sequencing', 'Sex', 'Race', 'Ethnicity',
    'Sample Type', 'Sample Type (Detailed)'
}

TECHNICAL_FEATURES = {
    'SUBJECT_CENTER', 'SEQ_ASSAY_ID', 'SEQ_YEAR', 'SEQ_PIPELINE_ID',
    'library_selection', 'calling_strategy', 'target_capture_kit',
    'platform', 'sample_preservation', 'tumor_purity_bin',
    'Institution', 'Sequencing Assay', 'Sequencing Year', 'Sequencing Pipeline',
    'Library Selection', 'Variant Calling Strategy', 'Capture Kit',
    'Sequencing Platform', 'Sample Preservation', 'Tumor Purity',
    # Additional technical features that could match 'AGE' pattern falsely
    'coverage', 'Coverage', 'COVERAGE',
    'is_paired_end', 'instrument_model', 'gene_padding', 'read_length',
    'library_strategy', 'preservation_technique', 'specimen_tumor_cellularity',
    'alteration_types', 'number_of_genes', 'variant_classifications',
}


def classify_feature_type(feature_name: str) -> str:
    """
    Classify a feature as 'demographic' or 'technical'.

    Args:
        feature_name: Feature name (original or display name)

    Returns:
        'demographic' or 'technical'
    """
    # Check if it's a known demographic feature
    if feature_name in DEMOGRAPHIC_FEATURES:
        return 'demographic'

    # Check if it's a known technical feature
    if feature_name in TECHNICAL_FEATURES:
        return 'technical'

    # Check by pattern - demographic patterns
    demographic_patterns = ['SEX', 'RACE', 'ETHNICITY', 'AGE', 'SAMPLE_TYPE']
    if any(pattern in feature_name.upper() for pattern in demographic_patterns):
        return 'demographic'

    # Check by pattern - technical patterns
    technical_patterns = ['CENTER', 'ASSAY', 'PIPELINE', 'YEAR', 'LIBRARY', 'CALLING',
                          'CAPTURE', 'PLATFORM', 'PRESERVATION', 'PURITY', 'SEQ_',
                          'INSTITUTION', 'SEQUENCING', 'COVERAGE', 'INSTRUMENT',
                          'GENE_PADDING', 'READ_LENGTH', 'NUM__X', 'CAT__X',
                          'IS_PAIRED', 'ALTERATION']
    if any(pattern in feature_name.upper() for pattern in technical_patterns):
        return 'technical'

    # Default to technical for unknown features (safer assumption)
    return 'technical'


def create_mapper_from_data_loader(feature_cols: List[str]) -> FeatureNameMapper:
    """
    Convenience function to create a FeatureNameMapper from data loader feature columns.

    Args:
        feature_cols: List of original feature column names from DataLoader

    Returns:
        Configured FeatureNameMapper instance
    """
    # Separate numeric and categorical features
    numeric_features = []
    categorical_features = []

    for feat in feature_cols:
        if 'AGE' in feat or 'YEAR' in feat or 'PURITY' in feat:
            numeric_features.append(feat)
        else:
            categorical_features.append(feat)

    return FeatureNameMapper(
        original_feature_names=feature_cols,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
