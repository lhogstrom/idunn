# BALDER Pipeline Documentation

This document describes the key data filtering and aggregation stages of the BALDER Nextflow pipeline for analyzing AACR GENIE cancer mutation data.

## Pipeline Aggregation and Filtering Stages

### 1. Load OncoKB Results (`load_oncokb_run.r`)

**Process:** `load_oncokb_output`

**Purpose:** Loads and processes OncoKB annotation results, creating summary tables of clinically actionable variants.

**Inputs:**
- OncoKB annotator output file (`compiled_mutations_all_studies.oncokb.txt`)
- SQLite database (`balder-harmonized-biomarker-data-v20250522.sqlite`)

**Key Operations:**
- Reads OncoKB annotation results
- Creates boolean flags for sensitivity, resistance, diagnostic (DX), and prognostic (PX) entries
- Determines the highest clinical evidence level for each variant using a hierarchical ordering system (R1 > 1 > 2 > Dx1 > Px1 > 3A > 3B > 4 > R2 > Dx2 > Px2 > Dx3 > Px3)
- Generates summary statistics grouped by gene, protein change, and cancer type

**Outputs:**
- SQL table: `oncoKBAnnotatedVariantTable`
- Summary file: `oncokb_variant_summary_table.txt`

---

### 2. Clinical Annotation Variant Matching (`clinical_annotation_variant_matching.r`)

**Process:** `clinical_annotation_matching`

**Purpose:** Matches patient-observed variants with clinical annotations from MOA/CiVIC databases and integrates OncoKB results.

**Inputs:**
- MOA and CiVIC clinical annotation rules
- Patient observed variants
- OncoKB annotations
- OncoTree cancer type hierarchy

**Key Operations:**
- Loads clinical annotation rules from MOA/CiVIC databases
- Filters patient variants (excludes cell lines)
- Calculates Mutant Allele Frequency (MAF) from sequencing counts
- Joins variant data with sample clinical information
- Performs exhaustive annotation matching by genomic coordinate and amino acid change
- Selects representative variants per patient, prioritizing cancer type matches and high-confidence evidence
- Creates union of OncoKB-annotated variants and MOA/CiVIC-annotated variants

**Outputs:**
- SQL table: `patientObservedVariantTableWithSampleInfoOncokb` (variants with sample info and OncoKB annotations)
- SQL table: `exhaustiveClinicalAnnotatedPatientVariants` (all matching annotations)
- SQL table: `representativeClinicalAnnotatedPatientVariants` (one representative annotation per variant)
- SQL table: `representativeClinicalAnnotatedPatientVariantsWithOncoKB` (union of MOA/CiVIC and OncoKB)
- CSV file: `actionability_representative_variant_table.csv`

---

### 3. Targeted Panel Coordinate Lookup (`targeted_panel_coordinate_lookup.r`)

**Process:** `targeted_panel_coordinate_lookup`

**Purpose:** Analyzes genomic panel coverage to determine which actionable mutation positions are adequately covered across different sequencing assays.

**Inputs:**
- GENIE genomic information file (`genomic_information.txt`)
- Patient variant data with OncoKB annotations
- OncoKB Level 1 actionable variants

**Key Operations:**
- Calculates median genomic coverage per sample
- For each genomic position with Level 1 actionable mutations:
  - Determines which sequencing panels cover that position
  - Calculates the number of patients tested with coverage at that position
  - Computes the percentage of patients in each cancer type with panel coverage
- Identifies high-coverage positions (>90% of patients tested)
- Generates cumulative plots of actionable mutation counts by genomic position

**Outputs:**
- SQL table: `medianCoveragePerSample`
- SQL table: `genomicPositionLookup` (coverage statistics per position/cancer type)
- Text file: `panel_design_genomic_coverage_lookup.txt`
- CSV file: `genomic_position_lookup_table.csv`
- PDF plots: Cumulative mutation counts by genomic position with coverage labels

---

### 4. Generate Per-Patient OncoKB Level Status (`generate_perPatientOncoKBLevelSorted.r`)

**Process:** `generate_perPatientOncoKBLevelSorted`

**Purpose:** Creates a per-patient data product identifying the most clinically significant variant for each patient, prioritized by OncoKB evidence level and panel coverage.

**Inputs:**
- Patient variants with OncoKB annotations and sample info
- Genomic position coverage lookup table
- GENIE assay information

**Key Operations:**
- Creates boolean flags for OncoKB levels (Level 1, Level 2, Level 3, etc.)
- Identifies hematologic cancers (lymphoid/myeloid)
- Maps variants to genomic positions and flags those with high panel coverage (>90%)
- Per patient, selects one representative variant using hierarchical prioritization:
  1. High panel coverage positions (>90%)
  2. OncoKB Level 1 variants
  3. OncoKB Level 2 or Level 1 variants
  4. Any OncoKB level variants

**Outputs:**
- SQL table: `perPatientOncoKBLevelSorted`
- Text file: `patient_level1_OncoKB_status_data.txt`

---

### 5. Filter and Structure GENIE Data for Modeling (`filter_and_structure_genie_data_for_modeling.r`)

**Process:** `filter_and_structure_genie_data_for_modeling`

**Purpose:** Applies a series of quality filters to create a curated dataset suitable for machine learning modeling of actionable genomic events.

**Inputs:**
- Per-patient OncoKB level-sorted variants
- Genomic position lookup table with panel coverage

**Filtering Steps:**

**For Actionable (Level 1) Events:**
1. Start with all patient variants
2. Filter: GENIE data only (excludes PANCAN/Hartwig)
3. Filter: OncoKB Level 1 actionable events
4. Filter: Drug sensitivity events only
5. Filter: Sequencing coverage ≥ 20 reads
6. Filter: Panel coverage ≥ 90% (high-coverage positions only)
7. Filter: Cancer types with ≥ 250 patients

**For Non-Actionable Events (Controls):**
1. Start with all patient variants
2. Filter: GENIE data only
3. Filter: No Level 1 actionable events
4. Filter: Sequencing coverage ≥ 20 reads
5. Filter: Cancer types with ≥ 250 patients (same types as actionable set)

**Additional Transformations:**
- Recodes race categories: "White" → "European", "Black" → "African"
- Creates `positionCtypeID` identifier (ONCOTREE_CODE-Gene-Position)

**Outputs:**
- SQL table: `perPatientOncoKBFilteredForModeling` (combined actionable and non-actionable variants)
- Text file: `level1_actionable_event_filtering_results.txt` (audit trail for actionable filtering)
- Text file: `not_level1_actionable_event_filtering_results.txt` (audit trail for non-actionable filtering)
- Text file: `data_product_level1_actionable_events_after_filtering_results.txt` (final modeling dataset)

