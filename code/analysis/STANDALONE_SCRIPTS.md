# Standalone Figure / Analysis Scripts

These scripts are **not wired into `run_all_analyses.py`**. They produce one-off figures or sensitivity analyses tied to specific upstream runs, and should be re-run only when their inputs change. Listed in the order they were added.

The first two entries below depend on hand-curated literature-review CSVs in `code_review_and_feedback/lit_review/`. See `code_review_and_feedback/lit_review/LIT_REVIEW_WORKFLOW.md` for the end-to-end manual workflow (when to re-curate, which pipeline outputs feed it, which curated CSVs each script reads).

## Lit-review hit-list arrow grid

`code/analysis/figures/generate_hit_list_arrow_grid.py` — reads the curated lit review CSV `code_review_and_feedback/lit_review/gene_stratified_comparison_lit_review_hit_subset.csv` and renders the Tier 1 arrow-grid figure summarizing age/sex hits with arrow direction = older/female (↑) vs younger/male (↓), color = novel (`#1f4e79`) vs confirmed (`#e69f00`). Each gene-hit gets its own arrow with the gene symbol labeled. Two style variants:

- Basic: `tier1_hits_arrows.{pdf,png}`
- Polished: `tier1_hits_arrows_polished.{pdf,png}` — Helvetica font, real `FancyArrowPatch` arrows, alternating row banding, compact two-axis legend, subtitle with N counts.

Default output: `code_review_and_feedback/lit_review/figures/`. Override via `--csv-path` and `--output-dir`. Re-run only when the curated CSV changes — see `LIT_REVIEW_WORKFLOW.md` for the curation step that precedes this. Pulls OT-code → common-name display strings from the latest TIER 2a `cancer_type_group_definitions.csv`. The Tier 2+3 panel was retired alongside the OT-rollup strip on the publication branch; the curated CSV's Tier 2/3 rows (if present) are no longer rendered here.

## RF / LR cross-pathway hit intersection (race + sex)

`code/analysis/posthoc/visualize_rf_lr_hit_intersection.py` — takes one RF run and one LR run and inner-joins their gene-stratified significance tables on `(ONCOTREE_CODE, Hugo_Symbol)` for **both** RACE and SEX. Cramer's V (race), odds ratio + 95% Wald CI (sex), and raw counts are patient-level / model-agnostic and are asserted equal across pathways; only `p_fdr` legitimately differs (BH correction depends on the test set). Outputs land in `runs/rf_lr_hit_intersection/run_<timestamp>/` with `race/` and `sex/` subdirs:

- `race/per_hit/` — per-hit RACE contingency plot (reuses `code/tools/raw_data_plot_utils.py::plot_categorical_contingency`, existing TIER 7 style) plus a counts CSV per hit
- `race/fig1_lollipop_effect_sizes_{tier1,tier2_tier3}.{pdf,png}` — Cramer's V ranking with `-log10(p)` and N annotations
- `race/fig2_diverging_dpp_small_multiples_{tier1,tier2_tier3}.{pdf,png}` — Δpp (gene-actionable − non-actionable) per race per hit
- `sex/per_hit_counts/` — per-hit Female/Male counts CSV (no per-hit plot in v1)
- `sex/sex_or_forest_{all,tier1,tier2_tier3}.{pdf,png}` — Female/Male OR forest with 95% Wald CI on log(OR), color-coded by cancer system (lung / colorectal / myeloid / biliary)

Follows the `compare_shap_vs_lr_coefficients.py` precedent. Use `--features race sex` (default) or restrict to one. The current canonical intersected hit list (13 sex + 11 race) lives at `runs/rf_lr_hit_intersection/run_20260507_FIXED/`, regenerated post-audit from `runs/run_20260505_164207/tier7_posthoc_statistics_REGENERATED_20260507/` (RF) and `runs/run_20260313_111146/tier7_posthoc_statistics_REGENERATED_20260507/` (LR).

**SHAP-gate audit note (2026-05-07).** The pre-fix runs `run_20260212_190358/tier7_posthoc_statistics/`, `*_dedupfix/tier7_posthoc_statistics/`, `runs/rf_vs_lr_comparison_20260315/`, and `runs/rf_lr_hit_intersection/run_20260429_201733/` have been renamed to `*.DEPRECATED_20260507/` (see `code_review_and_feedback/shap_gate_provenance_audit_2026-05-07.md`). To re-run this script against the regenerated outputs, the script needs to be pointed at a directory where `tier7_posthoc_statistics/gene_stratified/{race,sex}_by_gene/*_gene_significant.csv` resolves — currently hard-coded at `visualize_rf_lr_hit_intersection.py:81,89`. Either symlink `tier7_posthoc_statistics → tier7_posthoc_statistics_REGENERATED_20260507` inside each RF/LR run, or add a `--tier7-subdir` flag (open ROADMAP follow-up).

## Sex-hit verification harness

`code/analysis/posthoc/verify_sex_hit_directionality.py` — re-derives each intersected sex hit's 2×2 contingency directly from `perPatientOncoKBFilteredForModeling`, independent of `analyze_gene_stratified_associations.py`. Five-phase output (Phase 1 faithful replication, Phase 1b symmetric PATIENT_ID dedup, Phase 2A within-actionable comparator, Phase 2B gene-exclusive numerator, Phase 2C per-cancer baseline F:M). Used April 2026 to discover and fix an asymmetric-dedup bug in `load_gene_patient_data` (the gene-actionable numerator was not deduped on `PATIENT_ID` while the non-actionable denominator was, inflating Ns by 15–25% but only marginally affecting OR). After the fix, 2/15 hits drop out (IDH2/AML and IDH2/MYELOID); remaining 13/13 stay female-enriched. Outputs at `runs/sex_hits_verification/run_<timestamp>/`. Companion writeup: `code_review_and_feedback/sex_hits_female_enrichment_review.md`. Re-run only when revisiting the sex-hit list or proposing further changes to the gene-stratified test logic.

Reference invocation:

```bash
python code/analysis/posthoc/verify_sex_hit_directionality.py \
  --intersection-run runs/rf_lr_hit_intersection/run_20260429_172151 \
  --mapping-csv runs/run_20260121_215129/tier2_data_setup/multi_tier_groupings/patient_to_group_mapping.csv
```

## SUBJECT_CENTER sensitivity harness

`code/analysis/posthoc/adjust_hits_for_subject_center.py` — sensitivity analysis answering "do the intersected hits still hold after controlling for sequencing institution?" For each intersected hit (sex / race / age, gene-stratified), fits `Logit(outcome) ~ demographic` and `Logit(outcome) ~ demographic + C(SUBJECT_CENTER)` and reports paired crude vs. adjusted ORs (sex/age) or omnibus LR-test p-values (race), plus a verdict per hit (`survives` / `attenuates` / `unstable`). Outcome semantics match the existing TIER 7 2-group setup (gene-actionable vs. non-actionable, deduped on `PATIENT_ID`).

Quasi-separation in `C(SUBJECT_CENTER)` is handled by pooling levels with constant outcome into `OTHER_SEPARATED` (not just dropping them) before retrying the fit; small centers (< 5 cohort patients) are pre-collapsed into `OTHER_SMALL`. Race is treated as multi-category with the largest level as reference and rare levels (< 5 cohort patients) collapsed into `OTHER_RACE_SMALL`.

May 2026 reference run: 9/13 sex hits stable (all survive; max Δlog(OR) = -0.023 for NSCLC/KRAS), 6/11 race hits stable (5 survive, **1 attenuates: tier3_BREAST/PTEN crude p=0.004 → adj p=0.100**), 18/24 age hits stable (all survive). Unstable hits (typically rare-cancer / heavy-institution-clustering combinations like IHCH/IDH1, GIST/KIT, COADREAD/BRAF) are reported with crude estimates and a `notes` column explaining the failure mode.

Outputs at `runs/subject_center_adjustment/run_<timestamp>/{sex,race,age}/` (CSVs + paired forest plots + `summary.md`). Companion writeup: `code_review_and_feedback/subject_center_sensitivity_review.md`. Follows the `verify_sex_hit_directionality.py` precedent.

Reference invocation:

```bash
# Post-SHAP-gate-audit (2026-05-07) — use the regenerated outputs:
python code/analysis/posthoc/adjust_hits_for_subject_center.py \
  --intersection-run runs/rf_lr_hit_intersection/run_20260507_FIXED \
  --rf-run runs/run_20260505_164207 \
  --lr-run runs/run_20260313_111146 \
  --mapping-csv runs/run_20260121_215129/tier2_data_setup/multi_tier_groupings/patient_to_group_mapping.csv
# Note: --rf-run / --lr-run must resolve `tier7_posthoc_statistics/`; symlink
# tier7_posthoc_statistics_REGENERATED_20260507/ to that name in each run
# until the script gains a --tier7-subdir flag (see ROADMAP).
```

**Caveat**: for race, this is a confounding-adjustment analysis but institution sits partly on the causal path (geography/insurance route patients to specific centers), so adjusted-OR underestimates a real signal — interpret race attenuations as "explained by institutional differences in patient mix and panel design" rather than "no true effect."

## SAMPLE_TYPE_DETAILED summary

`code/analysis/demographics/summarize_sample_type_detailed.py` — one-shot utility that joins `perPatientOncoKBFilteredForModeling` against the canonical multi-tier mapping and emits per-cancer-type counts and percentages of `SAMPLE_TYPE_DETAILED` (one row per patient). Default output: `runs/sample_type_detailed_summary/`. Re-run when the cohort or mapping changes.
