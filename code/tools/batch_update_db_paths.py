#!/usr/bin/env python3
"""
Batch update database paths in all Python scripts to use BALDER_DB_PATH environment variable.

This script updates all hard-coded database paths to use the get_default_db_path() helper.
"""

import re
from pathlib import Path

# Find all Python files that need updating
files_to_update = [
    # Modeling config and test scripts
    "code/modeling/oncokb_analyzer/config.py",
    "code/modeling/test_multi_tier_loader.py",
    "code/modeling/test_hyperparameter_optimizer_stratified.py",
    "code/modeling/validate_stratified_groupkfold.py",
    "code/modeling/test_multi_tier_loading.py",
    "code/modeling/test_phase2a_quick.py",
    "code/modeling/tests/test_data_loader_multicancer.py",

    # Demographics analysis
    "code/analysis/demographics/analyze_demographics_by_cancer_type.py",
    "code/analysis/demographics/summarize_hispanic_age_stats.py",
    "code/analysis/demographics/plot_sex_race_composition.py",
    "code/analysis/demographics/plot_age_ethnicity_interaction.py",

    # Institutional analysis
    "code/analysis/institutional/visualize_institutional_fold_distribution.py",
    "code/analysis/institutional/analyze_all_cancer_types_institutional_stats.py",
    "code/analysis/institutional/investigate_gist_anomaly.py",
    "code/analysis/institutional/visualize_institutional_actionability_patterns.py",

    # Post-hoc stats
    "code/analysis/posthoc/analyze_shap_feature_associations.py",
    "code/analysis/posthoc/create_presentation_tables.py",
    "code/analysis/posthoc/analyze_age_by_actionability.py",
    "code/analysis/posthoc/visualize_raw_data.py",

    # Root-level analysis
    "code/analysis/define_cancer_type_groupings.py",
    "code/analysis/cancer_type_summary_stats.py",
    "code/analysis/feature_engineering_analysis.py",
    "code/analysis/plot_patient_volume_diversity.py",
    "code/analysis/visualize_shap_feature_rankings.py",

    # Variants and SHAP
    "code/analysis/variants/plot_variant_demographic_stratification.py",

    # Tools
    "code/tools/query_eligible_cancer_types.py",
]

def add_import_if_missing(content):
    """Add db_config import if not present."""
    if 'from db_config import get_default_db_path' in content:
        return content  # Already has import

    # Find where to add import - look for existing imports
    import_patterns = [
        r'(import\s+\w+\nimport\s+\w+\n)',  # After consecutive imports
        r'(from\s+\w+\s+import.*?\n)',  # After from imports
        r'(import\s+\w+.*?\n)',  # After simple imports
    ]

    # Try to find import section
    for pattern in import_patterns:
        match = re.search(pattern, content)
        if match:
            # Add import after last import line found
            insert_pos = match.end()
            before = content[:insert_pos]
            after = content[insert_pos:]

            # Determine relative path depth based on file location
            if '/analysis/' in str(files_to_update):
                import_line = "\n# Add tools to path\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))\nfrom db_config import get_default_db_path\n"
            else:
                import_line = "\n# Add tools to path\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))\nfrom db_config import get_default_db_path\n"

            return before + import_line + after

    # If no imports found, add at top after docstring
    if '"""' in content:
        parts = content.split('"""', 2)
        if len(parts) >= 3:
            return parts[0] + '"""' + parts[1] + '"""' + "\n\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))\nfrom db_config import get_default_db_path\n" + parts[2]

    return content

def update_db_paths(content):
    """Replace hard-coded database paths."""
    # Pattern 1: Absolute paths
    pattern1 = r'"/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522\.sqlite"'
    content = re.sub(pattern1, 'get_default_db_path()', content)

    # Pattern 2: Relative paths (commented out version)
    pattern2 = r'"\.\./(\.\./)*(output|data)/.*?balder-harmonized-biomarker-data.*?\.sqlite"'
    # Don't replace if already calling get_default_db_path
    if 'get_default_db_path()' not in content:
        content = re.sub(pattern2, 'get_default_db_path()', content)

    return content

def main():
    """Update all files."""
    repo_root = Path(__file__).parent.parent.parent
    updated = []
    skipped = []

    for file_path in files_to_update:
        full_path = repo_root / file_path

        if not full_path.exists():
            skipped.append(f"{file_path} (not found)")
            continue

        # Read content
        content = full_path.read_text()

        # Check if already updated
        if 'get_default_db_path()' in content and 'from db_config import' in content:
            skipped.append(f"{file_path} (already updated)")
            continue

        # Add import
        content = add_import_if_missing(content)

        # Update paths
        content = update_db_paths(content)

        # Write back
        full_path.write_text(content)
        updated.append(file_path)

    print(f"✓ Updated {len(updated)} files")
    print(f"⊘ Skipped {len(skipped)} files")

    if updated:
        print("\nUpdated files:")
        for f in updated:
            print(f"  - {f}")

if __name__ == '__main__':
    main()
