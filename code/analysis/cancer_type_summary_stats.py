"""
Cancer Type Summary Statistics Table

Generate comprehensive summary statistics for all 21 cancer types in the modeling dataset.
Creates a publication-ready table with demographics, actionability, and institutional metrics.

Usage:
    python cancer_type_summary_stats.py [--output-dir PATH]
    python cancer_type_summary_stats.py --format-table1 [--latex] [--markdown]
    python cancer_type_summary_stats.py --format-table1 --include-multi-tier --base-run-dir RUNS/run_YYYYMMDD
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import argparse
import shutil
import subprocess
from pathlib import Path

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from db_config import get_default_db_path

# Database configuration
DB_PATH = get_default_db_path()

# Default output directory (can be overridden by --output-dir)
DEFAULT_OUTPUT_DIR = Path("../../analysis_output/cancer_type_summary_stats")
OUTPUT_DIR = None  # Set by main()


def load_patient_data(db_path):
    """
    Load patient data from perPatientOncoKBFilteredForModeling table

    Args:
        db_path: Path to SQLite database

    Returns:
        pandas DataFrame with patient data
    """
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM perPatientOncoKBFilteredForModeling"

    print("Loading perPatientOncoKBFilteredForModeling table...")
    df = pd.read_sql_query(query, conn)

    conn.close()

    print(f"Loaded {len(df):,} patient records with {len(df.columns)} columns")

    return df


def get_cancer_types(df, min_actionable=250):
    """
    Get list of cancer types meeting minimum actionable event threshold

    Args:
        df: Patient data DataFrame
        min_actionable: Minimum number of actionable events required

    Returns:
        List of cancer type codes
    """
    # Count actionable events per cancer type
    actionable_counts = df[df['isOncoKbLevel1'] == 1].groupby('ONCOTREE_CODE').size()

    # Filter to those meeting threshold
    eligible_types = actionable_counts[actionable_counts >= min_actionable].index.tolist()

    print(f"\nFound {len(eligible_types)} cancer types with ≥{min_actionable} actionable events")

    return sorted(eligible_types)


def calculate_sex_stats(cancer_df):
    """Calculate sex distribution statistics"""
    sex_counts = cancer_df['SEX'].value_counts()

    # Handle Unknown as missing
    unknown_count = sex_counts.get('Unknown', 0)
    na_count = cancer_df['SEX'].isna().sum()
    missing_total = int(unknown_count + na_count)

    return {
        'Sex_Male': int(sex_counts.get('Male', 0)),
        'Sex_Female': int(sex_counts.get('Female', 0)),
        'Sex_Missing': missing_total
    }


def calculate_age_stats(cancer_df):
    """Calculate age statistics"""
    age_data = cancer_df['AGE_AT_SEQ_REPORT'].dropna()

    if len(age_data) > 0:
        return {
            'Mean_Age': float(age_data.mean()),
            'SD_Age': float(age_data.std()),
            'Median_Age': float(age_data.median()),
            'Q1_Age': float(age_data.quantile(0.25)),
            'Q3_Age': float(age_data.quantile(0.75)),
            'Age_Missing': int(cancer_df['AGE_AT_SEQ_REPORT'].isna().sum())
        }
    else:
        return {
            'Mean_Age': np.nan,
            'SD_Age': np.nan,
            'Median_Age': np.nan,
            'Q1_Age': np.nan,
            'Q3_Age': np.nan,
            'Age_Missing': len(cancer_df)
        }


def calculate_race_stats(cancer_df):
    """Calculate ancestry distribution statistics (DB column is RACE, values are ancestry labels)"""
    race_counts = cancer_df['RACE'].value_counts()

    # Database values are already ancestry labels (European, African, Asian)
    european_count = int(race_counts.get('European', 0))
    african_count = int(race_counts.get('African', 0))
    asian_count = int(race_counts.get('Asian', 0))

    # Combine other ancestry groups
    other_groups = ['Native American', 'Pacific Islander', 'Other']
    other_count = int(sum(race_counts.get(group, 0) for group in other_groups))

    # Unknown and NA treated as missing
    unknown_count = race_counts.get('Unknown', 0)
    na_count = cancer_df['RACE'].isna().sum()
    missing_count = int(unknown_count + na_count)

    return {
        'Ancestry_European': european_count,
        'Ancestry_African': african_count,
        'Ancestry_Asian': asian_count,
        'Ancestry_Other': other_count,
        'Ancestry_Missing': missing_count
    }


def calculate_ethnicity_stats(cancer_df):
    """Calculate ethnicity distribution statistics"""
    ethnicity_counts = cancer_df['ETHNICITY'].value_counts()

    # Map database values
    hispanic_count = int(ethnicity_counts.get('Spanish/Hispanic', 0))
    non_hispanic_count = int(ethnicity_counts.get('Non-Spanish/non-Hispanic', 0))

    # Unknown and NA treated as missing
    unknown_count = ethnicity_counts.get('Unknown', 0)
    na_count = cancer_df['ETHNICITY'].isna().sum()
    missing_count = int(unknown_count + na_count)

    return {
        'Ethnicity_Hispanic': hispanic_count,
        'Ethnicity_NonHispanic': non_hispanic_count,
        'Ethnicity_Missing': missing_count
    }


def calculate_institutional_stats(cancer_df):
    """Calculate institutional distribution statistics"""
    n_institutions = cancer_df['SUBJECT_CENTER'].nunique()

    return {
        'N_Institutions': int(n_institutions)
    }


def calculate_detailed_race_stats(cancer_df):
    """Calculate counts for every individual RACE category in the database."""
    race_counts = cancer_df['RACE'].value_counts()
    na_count = int(cancer_df['RACE'].isna().sum())
    unknown_count = int(race_counts.get('Unknown', 0))

    return {
        'Race_European': int(race_counts.get('European', 0)),
        'Race_African': int(race_counts.get('African', 0)),
        'Race_Asian': int(race_counts.get('Asian', 0)),
        'Race_NativeAmerican': int(race_counts.get('Native American', 0)),
        'Race_PacificIslander': int(race_counts.get('Pacific Islander', 0)),
        'Race_Other': int(race_counts.get('Other', 0)),
        'Race_Unknown': unknown_count + na_count,
    }


def calculate_sample_type_stats(cancer_df):
    """Calculate Primary vs Metastasis vs Other sample type counts, plus detailed breakdown."""
    st_counts = cancer_df['SAMPLE_TYPE'].value_counts()
    primary = int(st_counts.get('Primary', 0))
    metastasis = int(st_counts.get('Metastasis', 0))
    other = int(sum(st_counts.get(v, 0) for v in
                     ['Not Collected', 'Unspecified', 'Not Applicable or Heme']))

    # Detailed sample type breakdown
    std_counts = cancer_df['SAMPLE_TYPE_DETAILED'].value_counts()
    return {
        'SampleType_Primary': primary,
        'SampleType_Metastasis': metastasis,
        'SampleType_Other': other,
        'STD_PrimaryTumor': int(std_counts.get('Primary tumor', 0)),
        'STD_MetSiteUnspecified': int(std_counts.get('Metastasis site unspecified', 0)),
        'STD_DistantOrganMet': int(std_counts.get('Distant organ metastasis', 0)),
        'STD_LymphNodeMet': int(std_counts.get('Lymph node metastasis', 0)),
        'STD_LocalRecurrence': int(std_counts.get('Local recurrence', 0)),
        'STD_NotCollected': int(std_counts.get('Not Collected', 0)),
        'STD_NOS': int(std_counts.get('Not otherwise specified', 0)),
        'STD_NAorHeme': int(std_counts.get('Not applicable or hematologic malignancy', 0)),
    }


def generate_summary_table(df, cancer_types):
    """
    Generate comprehensive summary statistics table

    Args:
        df: Patient data DataFrame
        cancer_types: List of cancer type codes to analyze

    Returns:
        DataFrame with summary statistics
    """
    print("\n" + "="*80)
    print("GENERATING SUMMARY STATISTICS TABLE")
    print("="*80)

    summary_rows = []

    for cancer_type in cancer_types:
        print(f"\nProcessing {cancer_type}...")

        # Filter to this cancer type
        cancer_df = df[df['ONCOTREE_CODE'] == cancer_type].copy()

        # Basic counts
        n_patients = len(cancer_df)
        n_actionable = int((cancer_df['isOncoKbLevel1'] == 1).sum())
        actionable_rate = n_actionable / n_patients if n_patients > 0 else 0

        # Compile statistics
        row = {
            'Cancer_Type': cancer_type,
            'N_Patients': n_patients,
            'N_Actionable': n_actionable,
            'Actionable_Rate': actionable_rate
        }

        # Add demographic statistics
        row.update(calculate_sex_stats(cancer_df))
        row.update(calculate_age_stats(cancer_df))
        row.update(calculate_race_stats(cancer_df))
        row.update(calculate_ethnicity_stats(cancer_df))
        row.update(calculate_institutional_stats(cancer_df))
        row.update(calculate_detailed_race_stats(cancer_df))
        row.update(calculate_sample_type_stats(cancer_df))

        summary_rows.append(row)

        print(f"  {n_patients:,} patients, {n_actionable:,} actionable ({actionable_rate:.1%})")

    # Create DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Sort by patient count (descending)
    summary_df = summary_df.sort_values('N_Patients', ascending=False)

    print(f"\n✓ Summary table generated for {len(summary_df)} cancer types")

    return summary_df


def generate_detailed_json(df, cancer_types, summary_df):
    """
    Generate detailed supplementary information in JSON format

    Args:
        df: Patient data DataFrame
        cancer_types: List of cancer type codes
        summary_df: Summary statistics DataFrame

    Returns:
        Dictionary with detailed statistics
    """
    print("\nGenerating detailed supplementary data...")

    detailed_data = {}

    for cancer_type in cancer_types:
        cancer_df = df[df['ONCOTREE_CODE'] == cancer_type].copy()

        # Get age percentiles
        age_data = cancer_df['AGE_AT_SEQ_REPORT'].dropna()
        age_percentiles = {}
        if len(age_data) > 0:
            age_percentiles = {
                '25th': float(age_data.quantile(0.25)),
                '50th': float(age_data.quantile(0.50)),
                '75th': float(age_data.quantile(0.75)),
                'min': float(age_data.min()),
                'max': float(age_data.max())
            }

        # Full race breakdown
        race_breakdown = cancer_df['RACE'].value_counts().to_dict()
        race_breakdown = {str(k): int(v) for k, v in race_breakdown.items()}

        # Full ethnicity breakdown
        ethnicity_breakdown = cancer_df['ETHNICITY'].value_counts().to_dict()
        ethnicity_breakdown = {str(k): int(v) for k, v in ethnicity_breakdown.items()}

        # Actionability by sex
        actionability_by_sex = {}
        for sex in ['Male', 'Female']:
            sex_df = cancer_df[cancer_df['SEX'] == sex]
            if len(sex_df) > 0:
                actionability_by_sex[sex] = {
                    'n_patients': int(len(sex_df)),
                    'n_actionable': int((sex_df['isOncoKbLevel1'] == 1).sum()),
                    'rate': float((sex_df['isOncoKbLevel1'] == 1).sum() / len(sex_df))
                }

        # Institution distribution
        institution_dist = cancer_df['SUBJECT_CENTER'].value_counts().to_dict()
        institution_dist = {str(k): int(v) for k, v in institution_dist.items()}

        # Compile detailed info
        detailed_data[cancer_type] = {
            'age_percentiles': age_percentiles,
            'race_breakdown_full': race_breakdown,
            'ethnicity_breakdown_full': ethnicity_breakdown,
            'actionability_by_sex': actionability_by_sex,
            'institution_distribution': institution_dist
        }

    print(f"✓ Detailed data generated for {len(detailed_data)} cancer types")

    return detailed_data


def save_outputs(summary_df, detailed_data):
    """
    Save all output files

    Args:
        summary_df: Summary statistics DataFrame
        detailed_data: Detailed statistics dictionary
    """
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)

    # 1. Save CSV (machine-readable)
    csv_file = OUTPUT_DIR / "cancer_type_summary_statistics.csv"
    summary_df.to_csv(csv_file, index=False, float_format='%.3f')
    print(f"\n✓ CSV saved: {csv_file}")

    # 2. Save formatted text table (human-readable)
    txt_file = OUTPUT_DIR / "cancer_type_summary_statistics.txt"
    with open(txt_file, 'w') as f:
        f.write("Cancer Type Summary Statistics\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Generated from: {DB_PATH}\n")
        f.write(f"Number of cancer types: {len(summary_df)}\n")
        f.write(f"Total patients: {summary_df['N_Patients'].sum():,}\n")
        f.write(f"Total actionable: {summary_df['N_Actionable'].sum():,}\n\n")
        f.write("=" * 100 + "\n\n")

        # Format DataFrame for display
        display_df = summary_df.copy()
        display_df['Actionable_Rate'] = display_df['Actionable_Rate'].apply(lambda x: f"{x:.1%}")
        display_df['Mean_Age'] = display_df['Mean_Age'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NA")
        display_df['SD_Age'] = display_df['SD_Age'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "NA")

        f.write(display_df.to_string(index=False))
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("\nColumn Definitions:\n")
        f.write("-" * 50 + "\n")
        f.write("Cancer_Type: OncoTree code\n")
        f.write("N_Patients: Total patient count\n")
        f.write("N_Actionable: Patients with OncoKB Level 1 variants\n")
        f.write("Actionable_Rate: Proportion with actionable variants\n")
        f.write("Sex_*: Male/Female/Missing counts\n")
        f.write("Mean_Age: Mean age at sequencing (years)\n")
        f.write("SD_Age: Standard deviation of age\n")
        f.write("Age_Missing: Patients with missing age data\n")
        f.write("Race_*: White/Black/Asian/Other/Missing counts\n")
        f.write("Ethnicity_*: Hispanic/NonHispanic/Missing counts\n")
        f.write("N_Institutions: Number of unique sequencing centers\n")

    print(f"✓ TXT saved: {txt_file}")

    # 3. Save detailed JSON
    json_file = OUTPUT_DIR / "cancer_type_summary_detailed.json"
    with open(json_file, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    print(f"✓ JSON saved: {json_file}")

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")


def validate_results(summary_df):
    """
    Validate results against expected values

    Args:
        summary_df: Summary statistics DataFrame
    """
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)

    # Check 1: Expected number of cancer types
    expected_n_types = 21
    actual_n_types = len(summary_df)

    if actual_n_types == expected_n_types:
        print(f"✓ Cancer type count: {actual_n_types} (expected: {expected_n_types})")
    else:
        print(f"⚠ Cancer type count: {actual_n_types} (expected: {expected_n_types})")

    # Check 2: No missing required columns
    required_columns = [
        'Cancer_Type', 'N_Patients', 'N_Actionable', 'Actionable_Rate',
        'Sex_Male', 'Sex_Female', 'Sex_Missing',
        'Mean_Age', 'SD_Age', 'Median_Age', 'Q1_Age', 'Q3_Age', 'Age_Missing',
        'Ancestry_European', 'Ancestry_African', 'Ancestry_Asian', 'Ancestry_Other', 'Ancestry_Missing',
        'Ethnicity_Hispanic', 'Ethnicity_NonHispanic', 'Ethnicity_Missing',
        'N_Institutions'
    ]

    missing_cols = [col for col in required_columns if col not in summary_df.columns]
    if not missing_cols:
        print(f"✓ All required columns present ({len(required_columns)} columns)")
    else:
        print(f"⚠ Missing columns: {missing_cols}")

    # Check 3: Minimum actionable events
    min_actionable = summary_df['N_Actionable'].min()
    if min_actionable >= 250:
        print(f"✓ All cancer types have ≥250 actionable events (min: {min_actionable})")
    else:
        below_threshold = summary_df[summary_df['N_Actionable'] < 250]['Cancer_Type'].tolist()
        print(f"⚠ Cancer types with <250 actionable events: {below_threshold}")

    # Check 4: Actionable rate consistency
    summary_df['Calculated_Rate'] = summary_df['N_Actionable'] / summary_df['N_Patients']
    rate_diff = (summary_df['Actionable_Rate'] - summary_df['Calculated_Rate']).abs().max()
    if rate_diff < 0.001:
        print(f"✓ Actionable rates consistent (max diff: {rate_diff:.6f})")
    else:
        print(f"⚠ Actionable rate calculation issue (max diff: {rate_diff:.6f})")

    # Check 5: Patient count totals
    total_patients = summary_df['N_Patients'].sum()
    print(f"✓ Total patients across all cancer types: {total_patients:,}")

    # Check 6: Sex totals consistency
    for idx, row in summary_df.iterrows():
        sex_total = row['Sex_Male'] + row['Sex_Female'] + row['Sex_Missing']
        if sex_total != row['N_Patients']:
            print(f"⚠ Sex counts don't match N_Patients for {row['Cancer_Type']}: {sex_total} vs {row['N_Patients']}")

    print("\nValidation complete!")


def generate_overall_row(df):
    """Generate an aggregate 'Overall' row across all patients."""
    row = {
        'Cancer_Type': 'Overall',
        'N_Patients': len(df),
        'N_Actionable': int((df['isOncoKbLevel1'] == 1).sum()),
        'Actionable_Rate': float((df['isOncoKbLevel1'] == 1).sum() / len(df)) if len(df) > 0 else 0,
    }
    row.update(calculate_sex_stats(df))
    row.update(calculate_age_stats(df))
    row.update(calculate_race_stats(df))
    row.update(calculate_ethnicity_stats(df))
    row.update(calculate_institutional_stats(df))
    row.update(calculate_detailed_race_stats(df))
    row.update(calculate_sample_type_stats(df))
    return row


def _find_multi_tier_mapping_file(base_run_dir=None):
    """
    Locate the patient_to_group_mapping.csv file.

    Priority:
    1. If base_run_dir provided, look at {base_run_dir}/tier2_data_setup/multi_tier_groupings/
    2. Auto-search via multi_tier_utils.get_patient_to_group_mapping_file()

    Returns:
        Path to mapping file, or None if not found.
    """
    if base_run_dir:
        explicit_path = Path(base_run_dir) / "tier2_data_setup" / "multi_tier_groupings" / "patient_to_group_mapping.csv"
        if explicit_path.exists():
            return explicit_path
        print(f"  NOTE: No mapping at {explicit_path}; falling back to auto-search")

    sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
    from multi_tier_utils import get_patient_to_group_mapping_file
    return get_patient_to_group_mapping_file()


def add_multi_tier_rows(df, base_run_dir=None):
    """
    Add Tier 2 and Tier 3 cancer group rows using multi_tier_utils.

    The multi-tier mapping is loaded from:
    1. {base_run_dir}/tier2_data_setup/multi_tier_groupings/ (if base_run_dir provided)
    2. Auto-search (latest run with multi-tier groupings)

    Args:
        df: Full patient DataFrame
        base_run_dir: Optional path to a specific run directory

    Returns:
        List of summary row dicts for Tier 2/3 groups (empty list if mapping not found)
    """
    mapping_file = _find_multi_tier_mapping_file(base_run_dir)
    if mapping_file is None:
        print("  WARNING: Could not locate multi-tier mapping; skipping Tier 2/3 rows")
        print("  Run define_multi_tier_groupings_v3.py first to generate the mapping")
        return []

    print(f"  Loading multi-tier mapping from: {mapping_file}")
    mapping_df = pd.read_csv(mapping_file)

    # Get Tier 2 and Tier 3 groups
    tier23_mask = mapping_df['tier_prefixed_code'].str.startswith(('tier2_', 'tier3_'))
    tier23_groups = sorted(mapping_df[tier23_mask]['tier_prefixed_code'].unique())

    rows = []
    for group in tier23_groups:
        patient_ids = mapping_df[mapping_df['tier_prefixed_code'] == group]['PATIENT_ID'].unique()
        group_df = df[df['PATIENT_ID'].isin(patient_ids)].copy()

        if len(group_df) == 0:
            continue

        row = {
            'Cancer_Type': group,
            'N_Patients': len(group_df),
            'N_Actionable': int((group_df['isOncoKbLevel1'] == 1).sum()),
            'Actionable_Rate': float((group_df['isOncoKbLevel1'] == 1).sum() / len(group_df)),
        }
        row.update(calculate_sex_stats(group_df))
        row.update(calculate_age_stats(group_df))
        row.update(calculate_race_stats(group_df))
        row.update(calculate_ethnicity_stats(group_df))
        row.update(calculate_institutional_stats(group_df))
        row.update(calculate_detailed_race_stats(group_df))
        row.update(calculate_sample_type_stats(group_df))
        rows.append(row)

    n_tier2 = sum(1 for r in rows if r['Cancer_Type'].startswith('tier2_'))
    n_tier3 = sum(1 for r in rows if r['Cancer_Type'].startswith('tier3_'))
    print(f"  Added {len(rows)} multi-tier rows ({n_tier2} Tier 2, {n_tier3} Tier 3)")
    return rows


def format_n_pct(n, total):
    """Format as 'N (XX.X%)' for Table 1 presentation."""
    if total == 0:
        return f"{n} (0.0%)"
    return f"{n} ({100 * n / total:.1f}%)"


def format_table1(summary_df):
    """
    Convert raw summary stats to publication 'N (%)' formatted Table 1.

    Returns a new DataFrame with formatted string columns.
    """
    formatted_rows = []

    for _, row in summary_df.iterrows():
        n = row['N_Patients']
        fr = {
            'Cancer_Type': row['Cancer_Type'],
            'N': f"{n:,}",
            'Actionable_N_pct': format_n_pct(int(row['N_Actionable']), n),
            'Age_mean_SD': (
                f"{row['Mean_Age']:.1f} ({row['SD_Age']:.1f})"
                if pd.notna(row['Mean_Age']) else "NA"
            ),
            'Age_median_IQR': (
                f"{row['Median_Age']:.0f} [{row['Q1_Age']:.0f}\u2013{row['Q3_Age']:.0f}]"
                if pd.notna(row['Median_Age']) else "NA"
            ),
            'Female_N_pct': format_n_pct(int(row['Sex_Female']), n),
            'European_N_pct': format_n_pct(int(row['Ancestry_European']), n),
            'African_N_pct': format_n_pct(int(row['Ancestry_African']), n),
            'Asian_N_pct': format_n_pct(int(row['Ancestry_Asian']), n),
            'Hispanic_N_pct': format_n_pct(int(row['Ethnicity_Hispanic']), n),
            'N_Institutions': str(int(row['N_Institutions'])),
        }
        formatted_rows.append(fr)

    return pd.DataFrame(formatted_rows)


def export_latex(formatted_df, output_path):
    """Export Table 1 as a LaTeX tabular environment."""
    col_map = {
        'Cancer_Type': 'Cancer Type',
        'N': 'N',
        'Actionable_N_pct': 'Actionable, N (\\%)',
        'Age_mean_SD': 'Age, mean (SD)',
        'Age_median_IQR': 'Age, median [IQR]',
        'Female_N_pct': 'Female, N (\\%)',
        'European_N_pct': 'European, N (\\%)',
        'African_N_pct': 'African, N (\\%)',
        'Asian_N_pct': 'Asian, N (\\%)',
        'Hispanic_N_pct': 'Hispanic, N (\\%)',
        'N_Institutions': 'Institutions',
    }

    n_cols = len(col_map)
    col_spec = 'l' + 'r' * (n_cols - 1)

    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Patient Characteristics by Cancer Type}')
    lines.append('\\label{tab:table1}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\hline')

    # Header
    header = ' & '.join(col_map.values()) + ' \\\\'
    lines.append(header)
    lines.append('\\hline')

    # Data rows
    for _, row in formatted_df.iterrows():
        # Escape special LaTeX characters in data
        vals = []
        for col in col_map.keys():
            v = str(row[col])
            # Escape % signs in data values (not in header which we already handled)
            v = v.replace('%', '\\%')
            # En-dash is fine in LaTeX
            vals.append(v)
        lines.append(' & '.join(vals) + ' \\\\')

    lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    Path(output_path).write_text('\n'.join(lines))
    print(f"  LaTeX table: {output_path}")


def render_markdown_to_pdf(md_path, pdf_path, fontsize='9pt'):
    """
    Render a Markdown file to PDF using pandoc.

    Returns True on success, False if pandoc is unavailable or fails.
    """
    if shutil.which('pandoc') is None:
        print(f"  WARNING: pandoc not found on PATH; skipping PDF render")
        print(f"  Install: conda install -c conda-forge pandoc (or brew install pandoc)")
        return False

    # Use a landscape-wide geometry so wide tables fit
    cmd = [
        'pandoc',
        str(md_path),
        '-o', str(pdf_path),
        '-V', 'geometry:landscape',
        '-V', 'geometry:margin=0.5in',
        '-V', f'fontsize={fontsize}',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Rendered PDF: {pdf_path}")
            return True
        else:
            print(f"  WARNING: pandoc failed: {result.stderr.strip()}")
            # Retry without LaTeX geometry options (in case no LaTeX engine is installed)
            cmd_fallback = ['pandoc', str(md_path), '-o', str(pdf_path)]
            result2 = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if result2.returncode == 0:
                print(f"  Rendered PDF (fallback): {pdf_path}")
                return True
            print(f"  WARNING: pandoc fallback also failed: {result2.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  WARNING: pandoc invocation raised {e}")
        return False


def _load_oncotree_name_mapping(base_run_dir=None):
    """
    Build a {tier_prefixed_code: (group_code, group_name)} mapping from
    multi_tier_group_definitions.csv. Returns empty dict if file not found.
    """
    # Try explicit base-run-dir first
    defs_path = None
    if base_run_dir:
        candidate = Path(base_run_dir) / "tier2_data_setup" / "multi_tier_groupings" / "multi_tier_group_definitions.csv"
        if candidate.exists():
            defs_path = candidate

    if defs_path is None:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
        try:
            from multi_tier_utils import get_group_definitions_file
            defs_path = get_group_definitions_file()
        except Exception:
            defs_path = None

    if defs_path is None or not Path(defs_path).exists():
        return {}

    defs = pd.read_csv(defs_path)
    return {
        row['tier_prefixed_code']: (row['group_code'], row['group_name'])
        for _, row in defs.iterrows()
    }


def format_table1_narrow(summary_df, oncotree_name_map):
    """
    Narrow Table 1 variant:
      - Replaces European/African/Asian columns with single "Non-European, N (%)"
      - Keeps Hispanic column
      - Adds OncoTree_Code and OncoTree_Name columns
      - Keeps all other columns (N, Actionable, Age mean/SD,
        Female, Institutions)
    """
    formatted_rows = []

    for _, row in summary_df.iterrows():
        n = row['N_Patients']
        cancer_type = row['Cancer_Type']

        # OncoTree full name
        if cancer_type == 'Overall':
            oncotree_name = 'All cancer types'
        elif cancer_type in oncotree_name_map:
            _, oncotree_name = oncotree_name_map[cancer_type]
        else:
            oncotree_name = 'NA'

        # Non-European count = African + Asian + Other (excludes Missing)
        n_non_european = (
            int(row['Ancestry_African'])
            + int(row['Ancestry_Asian'])
            + int(row['Ancestry_Other'])
        )

        fr = {
            'Cancer_Type': cancer_type,
            'OncoTree_Name': oncotree_name,
            'N': f"{n:,}",
            'Actionable_N_pct': format_n_pct(int(row['N_Actionable']), n),
            'Age_mean_SD': (
                f"{row['Mean_Age']:.1f} ({row['SD_Age']:.1f})"
                if pd.notna(row['Mean_Age']) else "NA"
            ),
            'Female_N_pct': format_n_pct(int(row['Sex_Female']), n),
            'NonEuropean_N_pct': format_n_pct(n_non_european, n),
            'Hispanic_N_pct': format_n_pct(int(row['Ethnicity_Hispanic']), n),
            'N_Institutions': str(int(row['N_Institutions'])),
        }
        formatted_rows.append(fr)

    return pd.DataFrame(formatted_rows)


def export_latex_narrow(formatted_df, output_path):
    """Export narrow Table 1 as a LaTeX tabular environment."""
    col_map = {
        'Cancer_Type': 'Cancer Type',
        'OncoTree_Name': 'OncoTree Name',
        'N': 'N',
        'Actionable_N_pct': 'Actionable, N (\\%)',
        'Age_mean_SD': 'Age, mean (SD)',
        'Female_N_pct': 'Female, N (\\%)',
        'NonEuropean_N_pct': 'Non-European, N (\\%)',
        'Hispanic_N_pct': 'Hispanic, N (\\%)',
        'N_Institutions': 'Institutions',
    }

    n_cols = len(col_map)
    col_spec = 'll' + 'r' * (n_cols - 2)  # 2 text cols + right-aligned numerics

    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Patient Characteristics by Cancer Type (narrow variant)}')
    lines.append('\\label{tab:table1_narrow}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\hline')
    lines.append(' & '.join(col_map.values()) + ' \\\\')
    lines.append('\\hline')

    for _, row in formatted_df.iterrows():
        vals = []
        for col in col_map.keys():
            v = str(row[col]).replace('%', '\\%').replace('&', '\\&')
            vals.append(v)
        lines.append(' & '.join(vals) + ' \\\\')

    lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    Path(output_path).write_text('\n'.join(lines))
    print(f"  LaTeX table (narrow): {output_path}")


def export_markdown_narrow(formatted_df, output_path):
    """Export narrow Table 1 as a pipe-delimited markdown table."""
    col_map = {
        'Cancer_Type': 'Cancer Type',
        'OncoTree_Name': 'OncoTree Name',
        'N': 'N',
        'Actionable_N_pct': 'Actionable, N (%)',
        'Age_mean_SD': 'Age, mean (SD)',
        'Female_N_pct': 'Female, N (%)',
        'NonEuropean_N_pct': 'Non-European, N (%)',
        'Hispanic_N_pct': 'Hispanic, N (%)',
        'N_Institutions': 'Institutions',
    }

    # Pandoc sizes columns proportionally to separator dash width.
    # Give OncoTree_Name a wider separator so it renders wider in PDF.
    sep_widths = {
        'OncoTree_Name': '-' * 7,
    }
    default_sep = '----'

    lines = []
    lines.append('| ' + ' | '.join(col_map.values()) + ' |')
    lines.append('|' + '|'.join(sep_widths.get(col, default_sep) for col in col_map.keys()) + '|')
    for _, row in formatted_df.iterrows():
        vals = [str(row[col]) for col in col_map.keys()]
        lines.append('| ' + ' | '.join(vals) + ' |')

    Path(output_path).write_text('\n'.join(lines) + '\n')
    print(f"  Markdown table (narrow): {output_path}")


def _resolve_oncotree_name(cancer_type, oncotree_name_map):
    """Map cancer_type to OncoTree display name."""
    if cancer_type == 'Overall':
        return 'All cancer types'
    if cancer_type in oncotree_name_map:
        _, name = oncotree_name_map[cancer_type]
        return name
    return 'NA'


def _export_supplement_markdown(formatted_df, col_map, output_path, label='supplement'):
    """Shared helper to write a supplement markdown table with wider OncoTree Name."""
    sep_widths = {'OncoTree_Name': '-' * 7}
    default_sep = '----'

    lines = []
    lines.append('| ' + ' | '.join(col_map.values()) + ' |')
    lines.append('|' + '|'.join(sep_widths.get(col, default_sep) for col in col_map.keys()) + '|')
    for _, row in formatted_df.iterrows():
        vals = [str(row[col]) for col in col_map.keys()]
        lines.append('| ' + ' | '.join(vals) + ' |')

    Path(output_path).write_text('\n'.join(lines) + '\n')
    print(f"  Markdown table ({label}): {output_path}")


# --- Supplement Table S1: Ancestry ---

def format_supplement_ancestry(summary_df, oncotree_name_map):
    """Format supplement table with full ancestry/race breakdown."""
    rows = []
    for _, row in summary_df.iterrows():
        n = row['N_Patients']
        rows.append({
            'Cancer_Type': row['Cancer_Type'],
            'OncoTree_Name': _resolve_oncotree_name(row['Cancer_Type'], oncotree_name_map),
            'N': f"{n:,}",
            'European_N_pct': format_n_pct(int(row['Race_European']), n),
            'African_N_pct': format_n_pct(int(row['Race_African']), n),
            'Asian_N_pct': format_n_pct(int(row['Race_Asian']), n),
            'NativeAmerican_N_pct': format_n_pct(int(row['Race_NativeAmerican']), n),
            'PacificIslander_N_pct': format_n_pct(int(row['Race_PacificIslander']), n),
            'Other_N_pct': format_n_pct(int(row['Race_Other']), n),
            'Unknown_N_pct': format_n_pct(int(row['Race_Unknown']), n),
        })
    return pd.DataFrame(rows)


ANCESTRY_COL_MAP = {
    'Cancer_Type': 'Cancer Type',
    'OncoTree_Name': 'OncoTree Name',
    'N': 'N',
    'European_N_pct': 'European, N (%)',
    'African_N_pct': 'African, N (%)',
    'Asian_N_pct': 'Asian, N (%)',
    'NativeAmerican_N_pct': 'Native American, N (%)',
    'PacificIslander_N_pct': 'Pacific Islander, N (%)',
    'Other_N_pct': 'Ancestry Other, N (%)',
    'Unknown_N_pct': 'Ancestry Unknown, N (%)',
}


def export_markdown_ancestry(formatted_df, output_path):
    _export_supplement_markdown(formatted_df, ANCESTRY_COL_MAP, output_path, 'ancestry')


# --- Supplement Table S2: Ethnicity ---

def format_supplement_ethnicity(summary_df, oncotree_name_map):
    """Format supplement table with ethnicity breakdown."""
    rows = []
    for _, row in summary_df.iterrows():
        n = row['N_Patients']
        rows.append({
            'Cancer_Type': row['Cancer_Type'],
            'OncoTree_Name': _resolve_oncotree_name(row['Cancer_Type'], oncotree_name_map),
            'N': f"{n:,}",
            'Hispanic_N_pct': format_n_pct(int(row['Ethnicity_Hispanic']), n),
            'NonHispanic_N_pct': format_n_pct(int(row['Ethnicity_NonHispanic']), n),
            'Unknown_N_pct': format_n_pct(int(row['Ethnicity_Missing']), n),
        })
    return pd.DataFrame(rows)


ETHNICITY_COL_MAP = {
    'Cancer_Type': 'Cancer Type',
    'OncoTree_Name': 'OncoTree Name',
    'N': 'N',
    'Hispanic_N_pct': 'Hispanic, N (%)',
    'NonHispanic_N_pct': 'Non-Hispanic, N (%)',
    'Unknown_N_pct': 'Ethnicity Unknown, N (%)',
}


def export_markdown_ethnicity(formatted_df, output_path):
    _export_supplement_markdown(formatted_df, ETHNICITY_COL_MAP, output_path, 'ethnicity')


# --- Supplement Table S3: Sample Type ---

def format_supplement_sample_type(summary_df, oncotree_name_map):
    """Format supplement table with sample type summary breakdown."""
    rows = []
    for _, row in summary_df.iterrows():
        n = row['N_Patients']
        rows.append({
            'Cancer_Type': row['Cancer_Type'],
            'OncoTree_Name': _resolve_oncotree_name(row['Cancer_Type'], oncotree_name_map),
            'N': f"{n:,}",
            'Primary_N_pct': format_n_pct(int(row['SampleType_Primary']), n),
            'Metastasis_N_pct': format_n_pct(int(row['SampleType_Metastasis']), n),
            'Other_N_pct': format_n_pct(int(row['SampleType_Other']), n),
        })
    return pd.DataFrame(rows)


SAMPLE_TYPE_COL_MAP = {
    'Cancer_Type': 'Cancer Type',
    'OncoTree_Name': 'OncoTree Name',
    'N': 'N',
    'Primary_N_pct': 'Primary, N (%)',
    'Metastasis_N_pct': 'Metastasis, N (%)',
    'Other_N_pct': 'Other, N (%)',
}


def export_markdown_sample_type(formatted_df, output_path):
    _export_supplement_markdown(formatted_df, SAMPLE_TYPE_COL_MAP, output_path, 'sample type')


def export_markdown(formatted_df, output_path):
    """Export Table 1 as a pipe-delimited markdown table."""
    col_map = {
        'Cancer_Type': 'Cancer Type',
        'N': 'N',
        'Actionable_N_pct': 'Actionable, N (%)',
        'Age_mean_SD': 'Age, mean (SD)',
        'Age_median_IQR': 'Age, median [IQR]',
        'Female_N_pct': 'Female, N (%)',
        'European_N_pct': 'European, N (%)',
        'African_N_pct': 'African, N (%)',
        'Asian_N_pct': 'Asian, N (%)',
        'Hispanic_N_pct': 'Hispanic, N (%)',
        'N_Institutions': 'Institutions',
    }

    lines = []

    # Header
    lines.append('| ' + ' | '.join(col_map.values()) + ' |')
    lines.append('|' + '|'.join(['---'] * len(col_map)) + '|')

    # Data rows
    for _, row in formatted_df.iterrows():
        vals = [str(row[col]) for col in col_map.keys()]
        lines.append('| ' + ' | '.join(vals) + ' |')

    Path(output_path).write_text('\n'.join(lines) + '\n')
    print(f"  Markdown table: {output_path}")


def main():
    """Main execution function"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate summary statistics for all cancer types'
    )
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: analysis_output/cancer_type_summary_stats)')
    parser.add_argument('--format-table1', action='store_true',
                       help='Generate publication-formatted Table 1 with N (%%) columns')
    parser.add_argument('--include-multi-tier', action='store_true',
                       help='Include Tier 2 and Tier 3 cancer groups as additional rows '
                            '(auto-detects mapping file if --base-run-dir not provided)')
    parser.add_argument('--base-run-dir', type=str, default=None,
                       help='Run directory for multi-tier mapping (optional; falls back to auto-search)')
    parser.add_argument('--latex', action='store_true',
                       help='Export Table 1 as LaTeX tabular')
    parser.add_argument('--markdown', action='store_true',
                       help='Export Table 1 as pipe-delimited markdown')
    args = parser.parse_args()

    # Set output directory
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cancer Type Summary Statistics Table Generator")
    print("=" * 80)

    # Check if database exists
    if not Path(DB_PATH).exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Please update DB_PATH in the script.")
        return 1

    # Load patient data
    df = load_patient_data(DB_PATH)

    # Get cancer types meeting threshold
    cancer_types = get_cancer_types(df, min_actionable=250)

    # Generate summary table
    summary_df = generate_summary_table(df, cancer_types)

    # Generate detailed supplementary data
    detailed_data = generate_detailed_json(df, cancer_types, summary_df)

    # Validate results
    validate_results(summary_df)

    # Save standard outputs
    save_outputs(summary_df, detailed_data)

    # --- Table 1 formatted outputs ---
    if args.format_table1:
        print("\n" + "=" * 80)
        print("GENERATING PUBLICATION TABLE 1")
        print("=" * 80)

        # Add tier prefix for Tier 1 cancer types
        summary_df['Cancer_Type'] = summary_df['Cancer_Type'].apply(
            lambda x: f"tier1_{x}" if x != 'Overall' else x
        )

        # Add multi-tier rows if requested
        if args.include_multi_tier:
            print("\nAdding Tier 2/3 cancer group rows...")
            tier23_rows = add_multi_tier_rows(df, args.base_run_dir)
            if tier23_rows:
                tier23_df = pd.DataFrame(tier23_rows)
                summary_df = pd.concat([summary_df, tier23_df], ignore_index=True)

        # Add overall row
        overall_row = generate_overall_row(df)
        summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)

        # Sort: Tier 1 by N desc, then Tier 2 by N desc, then Tier 3 by N desc, Overall last
        def sort_key(cancer_type):
            if cancer_type == 'Overall':
                return (3, 0)
            elif cancer_type.startswith('tier1_'):
                return (0, 0)
            elif cancer_type.startswith('tier2_'):
                return (1, 0)
            elif cancer_type.startswith('tier3_'):
                return (2, 0)
            return (0, 0)

        summary_df['_sort_tier'] = summary_df['Cancer_Type'].apply(lambda x: sort_key(x)[0])
        summary_df = summary_df.sort_values(['_sort_tier', 'N_Patients'], ascending=[True, False])
        summary_df = summary_df.drop(columns=['_sort_tier'])

        # Save raw Table 1 CSV
        raw_path = OUTPUT_DIR / "table1_patient_characteristics.csv"
        summary_df.to_csv(raw_path, index=False, float_format='%.3f')
        print(f"  Raw CSV: {raw_path}")

        # Format Table 1 (full)
        formatted_df = format_table1(summary_df)

        # Save formatted TXT
        txt_path = OUTPUT_DIR / "table1_patient_characteristics_formatted.txt"
        txt_path.write_text(formatted_df.to_string(index=False))
        print(f"  Formatted TXT: {txt_path}")

        # Optional exports (full Table 1)
        if args.latex:
            export_latex(formatted_df, OUTPUT_DIR / "table1_patient_characteristics.tex")

        if args.markdown:
            md_path = OUTPUT_DIR / "table1_patient_characteristics.md"
            export_markdown(formatted_df, md_path)
            # Render markdown to PDF via pandoc
            pdf_path = OUTPUT_DIR / "table1_patient_characteristics.pdf"
            render_markdown_to_pdf(md_path, pdf_path)

        # --- Narrow Table 1 variant ---
        print("\nGenerating narrow Table 1 variant (Non-European + OncoTree columns)...")
        oncotree_name_map = _load_oncotree_name_mapping(args.base_run_dir)
        if not oncotree_name_map:
            print("  NOTE: OncoTree name mapping not found; OncoTree_Name column will show 'NA'")

        formatted_narrow_df = format_table1_narrow(summary_df, oncotree_name_map)

        narrow_txt = OUTPUT_DIR / "table1_patient_characteristics_narrow_formatted.txt"
        narrow_txt.write_text(formatted_narrow_df.to_string(index=False))
        print(f"  Narrow formatted TXT: {narrow_txt}")

        if args.latex:
            export_latex_narrow(formatted_narrow_df, OUTPUT_DIR / "table1_patient_characteristics_narrow.tex")

        if args.markdown:
            narrow_md = OUTPUT_DIR / "table1_patient_characteristics_narrow.md"
            export_markdown_narrow(formatted_narrow_df, narrow_md)
            narrow_pdf = OUTPUT_DIR / "table1_patient_characteristics_narrow.pdf"
            render_markdown_to_pdf(narrow_md, narrow_pdf)

        # --- Supplement Table S1: Ancestry breakdown ---
        print("\nGenerating supplement S1 (ancestry breakdown)...")
        anc_df = format_supplement_ancestry(summary_df, oncotree_name_map)
        anc_df.to_csv(OUTPUT_DIR / "supplement_ancestry.csv", index=False)
        if args.markdown:
            anc_md = OUTPUT_DIR / "supplement_ancestry.md"
            export_markdown_ancestry(anc_df, anc_md)
            render_markdown_to_pdf(anc_md, OUTPUT_DIR / "supplement_ancestry.pdf")

        # --- Supplement Table S2: Ethnicity breakdown ---
        print("\nGenerating supplement S2 (ethnicity breakdown)...")
        eth_df = format_supplement_ethnicity(summary_df, oncotree_name_map)
        eth_df.to_csv(OUTPUT_DIR / "supplement_ethnicity.csv", index=False)
        if args.markdown:
            eth_md = OUTPUT_DIR / "supplement_ethnicity.md"
            export_markdown_ethnicity(eth_df, eth_md)
            render_markdown_to_pdf(eth_md, OUTPUT_DIR / "supplement_ethnicity.pdf")

        # --- Supplement Table S3: Sample type breakdown ---
        print("\nGenerating supplement S3 (sample type breakdown)...")
        st_df = format_supplement_sample_type(summary_df, oncotree_name_map)
        st_df.to_csv(OUTPUT_DIR / "supplement_sample_type.csv", index=False)
        if args.markdown:
            st_md = OUTPUT_DIR / "supplement_sample_type.md"
            export_markdown_sample_type(st_df, st_md)
            render_markdown_to_pdf(st_md, OUTPUT_DIR / "supplement_sample_type.pdf")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR / 'cancer_type_summary_statistics.csv'}")
    print(f"  - {OUTPUT_DIR / 'cancer_type_summary_statistics.txt'}")
    print(f"  - {OUTPUT_DIR / 'cancer_type_summary_detailed.json'}")
    if args.format_table1:
        print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics.csv'}")
        print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics_formatted.txt'}")
        print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics_narrow_formatted.txt'}")
        print(f"  - {OUTPUT_DIR / 'supplement_ancestry.csv'}")
        print(f"  - {OUTPUT_DIR / 'supplement_ethnicity.csv'}")
        print(f"  - {OUTPUT_DIR / 'supplement_sample_type.csv'}")
        if args.latex:
            print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics.tex'}")
            print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics_narrow.tex'}")
        if args.markdown:
            print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics.md'}")
            print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics.pdf'} (if pandoc available)")
            print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics_narrow.md'}")
            print(f"  - {OUTPUT_DIR / 'table1_patient_characteristics_narrow.pdf'} (if pandoc available)")
            print(f"  - {OUTPUT_DIR / 'supplement_ancestry.{{md,pdf}}'}")
            print(f"  - {OUTPUT_DIR / 'supplement_ethnicity.{{md,pdf}}'}")
            print(f"  - {OUTPUT_DIR / 'supplement_sample_type.{{md,pdf}}'}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
