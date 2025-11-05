import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite"
OUTPUT_DIR = Path("experiments/demographic_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ACTIONABLE_EVENTS = 250

# Cancer type groupings by sex predominance
FEMALE_PREDOMINANT = ['BRCA', 'BRCANOS', 'BRCNOS', 'HGSOC', 'IDC', 'ILC', 'MDLC']
MALE_PREDOMINANT = ['PRAD']
BOTH_SEXES = ['AML', 'ASTR', 'BLCA', 'COAD', 'COADREAD', 'GIST', 'IHCH',
              'LUAD', 'MEL', 'NSCLC', 'ODG', 'SKCM', 'THPA']

# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_sex_race_composition_faceted(db_path, output_dir, min_events=250):
    """Create faceted heatmaps showing sex-race composition by sex predominance"""

    # Query data for all cancer types
    conn = sqlite3.connect(db_path)

    query_types = f"""
    SELECT DISTINCT ONCOTREE_CODE, ot_name
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN (
        SELECT ONCOTREE_CODE
        FROM perPatientOncoKBLevelSorted
        GROUP BY ONCOTREE_CODE
        HAVING COUNT(CASE WHEN isOncoKbLevel1 = 1 THEN 1 END) >= {min_events}
    )
    ORDER BY ONCOTREE_CODE
    """
    cancer_names = pd.read_sql_query(query_types, conn)
    cancer_types = cancer_names['ONCOTREE_CODE'].tolist()

    # Query demographic data
    query = f"""
    SELECT
        ONCOTREE_CODE,
        SEX,
        RACE,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ({','.join(['?']*len(cancer_types))})
    AND SEX IS NOT NULL
    AND RACE IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=cancer_types)
    conn.close()

    # Filter to major categories
    major_races = df['RACE'].value_counts().nlargest(4).index.tolist()
    df = df[df['RACE'].isin(major_races)]
    df = df[df['SEX'].isin(['Male', 'Female'])]

    # Calculate proportions for each cancer type
    composition_data = []
    for cancer_type in cancer_types:
        cancer_df = df[df['ONCOTREE_CODE'] == cancer_type]
        if len(cancer_df) == 0:
            continue

        # Get full name
        full_name = cancer_names[cancer_names['ONCOTREE_CODE'] == cancer_type]['ot_name'].values[0]

        for sex in ['Male', 'Female']:
            for race in major_races:
                count = len(cancer_df[(cancer_df['SEX'] == sex) & (cancer_df['RACE'] == race)])
                proportion = count / len(cancer_df) if len(cancer_df) > 0 else 0
                composition_data.append({
                    'Cancer Type': cancer_type,
                    'Cancer Name': full_name,
                    'Sex': sex,
                    'Race': race,
                    'Proportion': proportion,
                    'Count': count
                })

    comp_df = pd.DataFrame(composition_data)

    # Add grouping information
    def assign_group(cancer_type):
        if cancer_type in FEMALE_PREDOMINANT:
            return 'Female-Predominant'
        elif cancer_type in MALE_PREDOMINANT:
            return 'Male-Predominant'
        else:
            return 'Both Sexes'

    comp_df['Group'] = comp_df['Cancer Type'].apply(assign_group)

    # Create label combining code and name
    comp_df['Label'] = comp_df['Cancer Type'] + '\n' + comp_df['Cancer Name']

    # Create pivot table for heatmap
    pivot_df = comp_df.pivot_table(
        index='Label',
        columns=['Sex', 'Race'],
        values='Proportion',
        fill_value=0
    )

    # Sort by Female-White proportion (highest on top)
    # Find the Female-White column
    female_white_col = None
    for col in pivot_df.columns:
        if col[0] == 'Female' and 'White' in col[1]:
            female_white_col = col
            break

    if female_white_col is not None:
        pivot_df = pivot_df.sort_values(by=female_white_col, ascending=False)
    else:
        # Fallback: sort alphabetically if Female-White column not found
        pivot_df = pivot_df.sort_index()

    # Version 1: WITH annotations (values printed)
    fig, ax = plt.subplots(figsize=(16, 14))

    # Create heatmap WITHOUT seaborn's annotations
    sns.heatmap(pivot_df, annot=False, cmap='YlGnBu',
                cbar_kws={'label': 'Proportion of Patients'},
                linewidths=0.5, ax=ax, vmin=0, vmax=0.5)

    # Manually add text annotations to ensure ALL cells get values
    for i in range(len(pivot_df)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            text = ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                          ha='center', va='center',
                          fontsize=6, color='black')

    ax.set_title('Sex-Race Composition Across Cancer Types\n(Ordered by Female-White Proportion)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Sex × Race Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    # Keep y-axis labels horizontal for readability
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'sex_race_composition_heatmap_annotated.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sex_race_composition_heatmap_annotated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Version 2: WITHOUT annotations (no values printed) - standard font size
    fig, ax = plt.subplots(figsize=(16, 14))

    sns.heatmap(pivot_df, annot=False, cmap='YlGnBu',
                cbar_kws={'label': 'Proportion of Patients'},
                linewidths=0.5, ax=ax, vmin=0, vmax=0.5)

    ax.set_title('Sex-Race Composition Across Cancer Types\n(Ordered by Female-White Proportion)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Sex × Race Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    # Keep y-axis labels horizontal for readability
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'sex_race_composition_heatmap_clean.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sex_race_composition_heatmap_clean.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Version 3: WITHOUT annotations (no values printed) - LARGER font sizes
    fig, ax = plt.subplots(figsize=(16, 14))

    sns.heatmap(pivot_df, annot=False, cmap='YlGnBu',
                cbar_kws={'label': 'Proportion of Patients'},
                linewidths=0.5, ax=ax, vmin=0, vmax=0.5)

    ax.set_title('Sex-Race Composition Across Cancer Types\n(Ordered by Female-White Proportion)',
                 fontsize=18, fontweight='bold')
    ax.set_xlabel('Sex × Race Combination', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=16, fontweight='bold')

    # Rotate x-axis labels with larger font
    plt.xticks(rotation=45, ha='right', fontsize=12)
    # Keep y-axis labels horizontal for readability with larger font
    plt.yticks(rotation=0, fontsize=12)

    # Increase colorbar label and tick sizes
    cbar = ax.collections[0].colorbar
    cbar.set_label('Proportion of Patients', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'sex_race_composition_heatmap_clean_large_text.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sex_race_composition_heatmap_clean_large_text.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    plot_sex_race_composition_faceted(DB_PATH, OUTPUT_DIR, MIN_ACTIONABLE_EVENTS)
