#!/usr/bin/env Rscript
#
# Collapse `perPatientOncoKBFilteredForModeling` from SAMPLE unit to
# PATIENT x CANCER-TYPE unit, in place, on a DB *copy*.
#
# ---------------------------------------------------------------------------
# WHAT THIS IS FOR
# ---------------------------------------------------------------------------
# The modeling table is one row per sequenced SAMPLE (85,419 rows), not per
# patient (76,796) -- see BACKLOG.md -> "Analytic unit is the sample, not the
# patient" and CLAUDE.md Gotcha #17. This script produces arm P of the Path C
# ladder by applying the recommended repair: a collapse on
# (PATIENT_ID, ONCOTREE_CODE) at the END of the filtering chain, i.e. after
# every quality filter has run.
#
# It is NOT wired into workflow/main.nf and must never be. It mutates a DB copy
# the same way code/tools/patch_age_redaction.py does, so that the OncoKB
# annotation and cohort membership of the source DB are provably untouched.
# Re-running main.nf to get this would re-fire run_oncokb against a newer
# OncoKB snapshot and confound every arm of the comparison.
#
# The body of `collapse_to_patient_unit()` is written to be transplantable
# verbatim to filter_and_structure_genie_data_for_modeling.r, immediately
# before its dbWriteTable() call, if/when the repair becomes permanent.
#
# ---------------------------------------------------------------------------
# THE COLLAPSE RULE  -- deterministic random pick  (CHANGED 2026-08-26, D-g)
# ---------------------------------------------------------------------------
# Keep one representative sample per (PATIENT_ID, ONCOTREE_CODE), chosen by a
# DETERMINISTIC RANDOM PICK keyed on the sample barcode:
#
#     sort key = md5(Tumor_Sample_Barcode), then Tumor_Sample_Barcode (asc)
#
# Tumor_Sample_Barcode is unique across the table, so the sort is a total order
# and the pick is reproducible regardless of input row ordering.
#
# WHY NOT set.seed() + sample(): a seeded shuffle depends on incoming row order,
# which depends on the rbind() in filter_and_structure_genie_data_for_modeling.r.
# That is exactly the F7 fragility this rule exists to avoid. A content-derived
# hash is stable across machines, row order, and library versions.
#
# WHY NOT the previous rule (highest OncoKB level, superseded 2026-08-15): it
# selected on the OUTCOME. Preferring the sample that carries the actionable
# variant inflates actionability among multi-sample patients by +3.8 pp at n=2
# and +7.2 pp at n=3 versus a random pick, and multi-sampleness varies ~2x
# across ancestry (Asian 10.49% vs Other 4.88%) -- so the inflation landed
# unevenly across the paper's own contrast. Decision record: ROADMAP_CHANGELOG.md
# 2026-08-15; BACKLOG.md -> "Patient-representative sample selection".
#
# CONSEQUENCE -- the outcome is NO LONGER the group max. Under the old rule a
# (patient, cancer) pair with contradictory isOncoKbLevel1 across its samples
# always resolved to ACTIONABLE. Under the random pick it resolves to whatever
# the drawn sample carries, so cohort actionable counts fall slightly. That is
# the intended estimand: "the actionability status of one sample drawn without
# reference to its outcome", not "this patient has >=1 actionable variant".
# 532 pairs are affected. The eligibility gate is therefore NO LONGER invariant
# by construction -- see check 6 below.
#
# Conclusions do not move: mean |MAX-RANDOM| per-cancer age shift is 0.080 y
# against effects of 4-17 y, with ODG the lone outlier at +1.06 y. This is an
# estimand-robustness change, not a numbers change.
#
# ---------------------------------------------------------------------------
# MEASURED PROPERTIES (ageUpper DB, 2026-08-13)
# ---------------------------------------------------------------------------
# !! Measured under the SUPERSEDED highest-OncoKB-level rule. Row counts and the
# !! lossless-covariate finding still hold (they do not depend on which sample is
# !! drawn), but the actionable counts and the age note below DO NOT. Re-measure
# !! after running under the random pick.
#
#   85,419 rows -> 78,086 rows (7,333 dropped; 5,657 pairs have >1 sample)
#   SEX / RACE / ETHNICITY / SUBJECT_CENTER never disagree within a pair (0/5,657),
#     so the collapse is lossless for those covariates.
#   AGE_AT_SEQ_REPORT: within-pair spread median 1.0y. Picking the actionable
#     row shifts age by +0.185y MORE for actionable than non-actionable pairs
#     (difference-in-differences) -- an upper bound on the age artifact this
#     rule can introduce, negligible against the 4-17y effects under study.
#   SAMPLE_TYPE_DETAILED (45.6%), number_of_genes (29.4%) and Hugo_Symbol
#     (47.1%) DO vary within pairs; the representative row decides them.
#
# Usage:
#   Rscript collapse_modeling_table_to_patient_unit.r \
#     --db /path/to/...-ageUpper-patientunit.sqlite \
#     [--report-dir DIR] [--force] [--dry-run]
#
# Exit codes: 0 ok, 1 usage/precondition failure, 2 verification failure.

suppressPackageStartupMessages({
  library(dplyr)
  library(DBI)
  library(RSQLite)
})

TABLE <- "perPatientOncoKBFilteredForModeling"
MIN_ACTIONABLE <- 250   # must match define_cancer_type_groupings.py
EXCLUDED_RESCUE_CODES <- c("THYROID", "ENCG", "BLADDER")  # ditto

# ---------------------------------------------------------------- arg parsing
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) == 0) return(default)
  if (i[1] == length(args)) stop(sprintf("%s requires a value", flag), call. = FALSE)
  args[i[1] + 1]
}
dbName     <- getarg("--db")
reportDir  <- getarg("--report-dir")
force      <- "--force"   %in% args
dryRun     <- "--dry-run" %in% args

if (is.null(dbName)) {
  cat("ERROR: --db is required\n\n")
  cat("Usage: Rscript collapse_modeling_table_to_patient_unit.r --db DB",
      "[--report-dir DIR] [--force] [--dry-run]\n")
  quit(status = 1)
}
if (!file.exists(dbName)) {
  cat(sprintf("ERROR: DB not found: %s\n", dbName)); quit(status = 1)
}

# Fail closed: this script overwrites a table. Refuse to touch anything that is
# not obviously a purpose-built copy unless the caller insists.
if (!grepl("patientunit", basename(dbName), fixed = TRUE) && !force) {
  cat(sprintf(paste0(
    "ERROR: refusing to modify '%s'.\n",
    "  This script OVERWRITES %s in place.\n",
    "  Expected a DB copy whose filename contains 'patientunit'.\n",
    "  Re-run with --force only if you are certain this is a disposable copy.\n"),
    basename(dbName), TABLE))
  quit(status = 1)
}

if (is.null(reportDir)) reportDir <- dirname(dbName)
dir.create(reportDir, showWarnings = FALSE, recursive = TRUE)

cat(strrep("=", 78), "\n")
cat("COLLAPSE MODELING TABLE TO PATIENT x CANCER-TYPE UNIT\n")
cat(strrep("=", 78), "\n")
cat(sprintf("DB          : %s\n", dbName))
cat(sprintf("Table       : %s\n", TABLE))
cat(sprintf("Report dir  : %s\n", reportDir))
cat(sprintf("Mode        : %s\n\n", if (dryRun) "DRY RUN (no write)" else "WRITE"))

# ------------------------------------------------------------------ the rule
# md5 of the barcode string, matching Python's
# hashlib.md5(s.encode()).hexdigest() byte for byte -- verified against
# code/analysis/posthoc/diagnose_sample_selection_rule.py, which carries the
# reference implementation of this rule.
md5_hex <- function(x) {
  if (requireNamespace("openssl", quietly = TRUE)) {
    as.character(openssl::md5(as.character(x)))
  } else if (requireNamespace("digest", quietly = TRUE)) {
    vapply(as.character(x),
           function(z) digest::digest(z, algo = "md5", serialize = FALSE),
           character(1), USE.NAMES = FALSE)
  } else {
    stop("the selection rule needs either the 'openssl' or the 'digest' package",
         call. = FALSE)
  }
}

# Transplantable body: operates on a data.frame, returns the collapsed frame.
#
# NOTE ON ORDERING: method = "radix" is used deliberately. R's radix sort orders
# character vectors in the C locale, guaranteed and version-independent, whereas
# the default sort is locale-sensitive. The sort key is a lowercase hex string,
# so in practice every locale agrees -- but the pick must be reproducible by
# construction, not by luck, which is the whole point of using a content hash.
collapse_to_patient_unit <- function(dataForModeling) {
  pickHash <- md5_hex(dataForModeling$Tumor_Sample_Barcode)
  ord <- order(dataForModeling$PATIENT_ID,
               dataForModeling$ONCOTREE_CODE,
               pickHash,
               dataForModeling$Tumor_Sample_Barcode,
               method = "radix")
  dataForModeling[ord, , drop = FALSE] %>%
    dplyr::group_by(PATIENT_ID, ONCOTREE_CODE) %>%
    dplyr::filter(dplyr::row_number() == 1) %>%
    dplyr::ungroup()
}

# --------------------------------------------------------------------- load
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)
on.exit(try(RSQLite::dbDisconnect(harmonizedDb), silent = TRUE), add = TRUE)

if (!DBI::dbExistsTable(harmonizedDb, TABLE)) {
  cat(sprintf("ERROR: table %s not found in %s\n", TABLE, dbName)); quit(status = 1)
}

cat("Reading table ...\n")
before <- RSQLite::dbGetQuery(harmonizedDb, sprintf("SELECT * FROM `%s`", TABLE))
cat(sprintf("  %s rows x %d columns\n", format(nrow(before), big.mark = ","), ncol(before)))

# ------------------------------------------------------------- preconditions
fail <- function(msg) { cat(sprintf("\nPRECONDITION FAILED: %s\n", msg)); quit(status = 1) }

needed <- c("PATIENT_ID", "ONCOTREE_CODE", "Tumor_Sample_Barcode", "isOncoKbLevel1",
            "isOncoKbLevel2or1", "isOncoKbLevelAny", "isHighCoverageByPanelDesign",
            "t_alt_plus_ref_count")
missing <- setdiff(needed, names(before))
if (length(missing)) fail(sprintf("missing columns: %s", paste(missing, collapse = ", ")))

nPairsBefore <- nrow(dplyr::distinct(before, PATIENT_ID, ONCOTREE_CODE))
if (nPairsBefore == nrow(before)) {
  cat("\nNOTHING TO DO: table is already one row per (PATIENT_ID, ONCOTREE_CODE).\n")
  cat("  This DB has already been collapsed. Refusing to re-write.\n")
  quit(status = 0)
}
if (anyDuplicated(before$Tumor_Sample_Barcode) != 0) {
  fail("Tumor_Sample_Barcode is not unique -- the sort would not be a total order")
}

# --------------------------------------------------------------- do the work
cat("\nCollapsing ...\n")
after <- collapse_to_patient_unit(before)
cat(sprintf("  %s rows -> %s rows (%s dropped)\n",
            format(nrow(before), big.mark = ","),
            format(nrow(after),  big.mark = ","),
            format(nrow(before) - nrow(after), big.mark = ",")))

# ------------------------------------------------------------- verification
checks <- list()
addcheck <- function(name, ok, detail) {
  checks[[length(checks) + 1]] <<- data.frame(check = name, status = if (ok) "PASS" else "FAIL",
                                              detail = detail, stringsAsFactors = FALSE)
}

# 1. exactly one row per pair
addcheck("one_row_per_pair", nrow(after) == nPairsBefore,
         sprintf("collapsed rows = %d, distinct pairs before = %d", nrow(after), nPairsBefore))

# 2. the SET of pairs is preserved (nothing invented, nothing lost)
pb <- dplyr::distinct(before, PATIENT_ID, ONCOTREE_CODE) %>% dplyr::arrange(PATIENT_ID, ONCOTREE_CODE)
pa <- dplyr::distinct(after,  PATIENT_ID, ONCOTREE_CODE) %>% dplyr::arrange(PATIENT_ID, ONCOTREE_CODE)
addcheck("pair_set_identical", isTRUE(all.equal(pb, pa, check.attributes = FALSE)),
         sprintf("%d pairs before, %d after", nrow(pb), nrow(pa)))

# 3. distinct patients preserved
addcheck("patient_set_identical",
         setequal(unique(before$PATIENT_ID), unique(after$PATIENT_ID)),
         sprintf("%d distinct patients before, %d after",
                 dplyr::n_distinct(before$PATIENT_ID), dplyr::n_distinct(after$PATIENT_ID)))

# 4. THE PICK IS ORDER-INVARIANT. Replaces the retired `outcome_is_group_max`
#    check, which asserted the rule superseded on 2026-08-15. Re-collapsing a
#    row-shuffled copy must retain exactly the same barcodes -- this is the
#    property a seeded shuffle would NOT have, and it is the direct test of the
#    F7 fragility class (a pick that silently depends on incoming row order).
set.seed(20260826)  # shuffles the INPUT only; the rule itself uses no RNG
shuffled <- before[sample.int(nrow(before)), , drop = FALSE]
reCollapsed <- collapse_to_patient_unit(shuffled)
addcheck("pick_is_order_invariant",
         setequal(after$Tumor_Sample_Barcode, reCollapsed$Tumor_Sample_Barcode) &&
           nrow(reCollapsed) == nrow(after),
         sprintf("%d of %d retained barcodes differ after a full row shuffle",
                 length(setdiff(after$Tumor_Sample_Barcode, reCollapsed$Tumor_Sample_Barcode)),
                 nrow(after)))

# 5. the representative belongs to ITS OWN group -- not merely to the table.
#    Stronger than the previous membership check, which only asked whether the
#    barcode existed somewhere upstream.
srcKey <- before %>% dplyr::select(Tumor_Sample_Barcode, srcPat = PATIENT_ID,
                                   srcCode = ONCOTREE_CODE)
memb <- after %>%
  dplyr::select(Tumor_Sample_Barcode, PATIENT_ID, ONCOTREE_CODE) %>%
  dplyr::left_join(srcKey, by = "Tumor_Sample_Barcode")
nBadMemb <- sum(is.na(memb$srcPat) | memb$srcPat != memb$PATIENT_ID |
                  memb$srcCode != memb$ONCOTREE_CODE)
addcheck("representative_is_group_member", nBadMemb == 0,
         sprintf("%d retained rows are not members of the pair they represent", nBadMemb))

# 6. ELIGIBILITY GATE -- the decisive one, and NO LONGER INVARIANT BY
#    CONSTRUCTION. Under the superseded max-OncoKB rule the pair outcome was
#    max(isOncoKbLevel1), so actionable counts were preserved exactly and this
#    check could demand count equality. The random pick does not preserve them:
#    per-code actionable counts fall slightly, and IHCH sits only ~8 above the
#    threshold. So the assertion is now on the SET of eligible codes, with the
#    counts reported rather than asserted. Counting matches
#    define_cancer_type_groupings.py exactly: COUNT(DISTINCT PATIENT_ID).
codeCounts <- function(df) {
  df %>%
    dplyr::filter(!ONCOTREE_CODE %in% EXCLUDED_RESCUE_CODES) %>%
    dplyr::group_by(ONCOTREE_CODE) %>%
    dplyr::summarise(
      total_patients = dplyr::n_distinct(PATIENT_ID),
      actionable_patients = dplyr::n_distinct(PATIENT_ID[isOncoKbLevel1 == 1]),
      .groups = "drop") %>%
    dplyr::arrange(dplyr::desc(actionable_patients))
}
elig <- function(df) dplyr::filter(df, actionable_patients >= MIN_ACTIONABLE)

cb <- codeCounts(before); ca <- codeCounts(after)
eb <- elig(cb);           ea <- elig(ca)

lost   <- setdiff(eb$ONCOTREE_CODE, ea$ONCOTREE_CODE)
gained <- setdiff(ea$ONCOTREE_CODE, eb$ONCOTREE_CODE)
addcheck("eligible_code_set_unchanged", length(lost) == 0 && length(gained) == 0,
         sprintf("%d codes before, %d after%s%s; min actionable %s -> %s",
                 nrow(eb), nrow(ea),
                 if (length(lost))   sprintf("; LOST: %s",   paste(lost, collapse = ",")) else "",
                 if (length(gained)) sprintf("; GAINED: %s", paste(gained, collapse = ",")) else "",
                 if (nrow(eb)) min(eb$actionable_patients) else NA,
                 if (nrow(ea)) min(ea$actionable_patients) else NA))
addcheck("eligible_code_count_is_21", nrow(ea) == 21,
         sprintf("%d codes clear >=%d actionable patients", nrow(ea), MIN_ACTIONABLE))

checkDf <- do.call(rbind, checks)
cat("\n"); cat(strrep("-", 78), "\n"); cat("VERIFICATION\n"); cat(strrep("-", 78), "\n")
for (i in seq_len(nrow(checkDf))) {
  cat(sprintf("  [%s] %-28s %s\n", checkDf$status[i], checkDf$check[i], checkDf$detail[i]))
}

# ---- per-cancer table, ALL codes (not only the eligible ones) -------------
# Sprint D item D5 (the >=250 threshold boundary check) needs the margin for
# every code, including the ones sitting just BELOW the line -- so this table
# is deliberately unfiltered.
eligCmp <- cb %>%
  dplyr::rename(total_before = total_patients, actionable_before = actionable_patients) %>%
  dplyr::full_join(ca %>% dplyr::rename(total_after = total_patients,
                                        actionable_after = actionable_patients),
                   by = "ONCOTREE_CODE") %>%
  dplyr::mutate(
    actionable_delta   = actionable_after - actionable_before,
    margin_after       = actionable_after - MIN_ACTIONABLE,
    eligible_before    = actionable_before >= MIN_ACTIONABLE,
    eligible_after     = actionable_after  >= MIN_ACTIONABLE) %>%
  dplyr::arrange(dplyr::desc(actionable_after))

# Write artifacts BEFORE printing, so a display bug cannot cost the record.
write.csv(checkDf, file.path(reportDir, "collapse_to_patient_unit_checks.csv"), row.names = FALSE)
write.csv(eligCmp, file.path(reportDir, "collapse_to_patient_unit_eligibility.csv"), row.names = FALSE)
write.csv(eligCmp %>% dplyr::select(ONCOTREE_CODE, total_after, actionable_after,
                                    margin_after, actionable_delta, eligible_after) %>%
            dplyr::arrange(margin_after),
          file.path(reportDir, "threshold_margin_distribution.csv"), row.names = FALSE)

cat("\nEligibility by cancer type (distinct patients):\n")
cat(sprintf("  %-14s %12s %12s %11s %11s %8s %7s\n", "ONCOTREE_CODE",
            "total_before", "total_after", "act_before", "act_after", "delta", "margin"))
for (i in seq_len(nrow(eligCmp))) {
  cat(sprintf("  %-14s %12s %12s %11s %11s %8s %7s%s\n", eligCmp$ONCOTREE_CODE[i],
              format(eligCmp$total_before[i], big.mark = ","),
              format(eligCmp$total_after[i],  big.mark = ","),
              format(eligCmp$actionable_before[i], big.mark = ","),
              format(eligCmp$actionable_after[i],  big.mark = ","),
              format(eligCmp$actionable_delta[i]),
              format(eligCmp$margin_after[i]),
              if (isTRUE(eligCmp$eligible_after[i])) "" else "   (below threshold)"))
}

# ---- threshold boundary summary (Sprint D item D5) ------------------------
elgA <- eligCmp %>% dplyr::filter(eligible_after)
belA <- eligCmp %>% dplyr::filter(!eligible_after) %>% dplyr::arrange(dplyr::desc(actionable_after))
cat("\n"); cat(strrep("-", 78), "\n")
cat(sprintf("THRESHOLD BOUNDARY (MIN_ACTIONABLE = %d)\n", MIN_ACTIONABLE))
cat(strrep("-", 78), "\n")
if (nrow(elgA)) {
  tight <- elgA %>% dplyr::arrange(margin_after) %>% utils::head(3)
  cat("  Narrowest margins ABOVE the line:\n")
  for (i in seq_len(nrow(tight))) {
    cat(sprintf("    %-14s actionable %6s  margin +%s\n", tight$ONCOTREE_CODE[i],
                format(tight$actionable_after[i], big.mark = ","), format(tight$margin_after[i])))
  }
}
if (nrow(belA)) {
  cat("  Nearest codes BELOW the line:\n")
  for (i in seq_len(min(3, nrow(belA)))) {
    cat(sprintf("    %-14s actionable %6s  margin %s\n", belA$ONCOTREE_CODE[i],
                format(belA$actionable_after[i], big.mark = ","), format(belA$margin_after[i])))
  }
}
cat("  -> threshold_margin_distribution.csv\n")

if (any(checkDf$status == "FAIL")) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("VERIFICATION FAILED -- table NOT modified.\n")
  cat("If `eligible_code_set_unchanged` or `eligible_code_count_is_21` failed, the\n")
  cat("cancer-type universe moved and arm P is no longer comparable to arms L/S.\n")
  cat("That is a SCOPE DECISION for the user, not something to work around here.\n")
  cat(strrep("=", 78), "\n")
  quit(status = 2)
}

# ---------------------------------------------------------------------- write
if (dryRun) {
  cat("\nDRY RUN -- all checks passed, table left unmodified.\n")
  quit(status = 0)
}

cat(sprintf("\nWriting %s rows to `%s` (overwrite) ...\n",
            format(nrow(after), big.mark = ","), TABLE))
RSQLite::dbWriteTable(harmonizedDb, TABLE, as.data.frame(after), overwrite = TRUE)

roundTrip <- RSQLite::dbGetQuery(
  harmonizedDb, sprintf("SELECT COUNT(*) n, COUNT(DISTINCT PATIENT_ID) p, SUM(isOncoKbLevel1) a FROM `%s`", TABLE))
cat(sprintf("Round-trip: %s rows, %s distinct patients, %s actionable\n",
            format(roundTrip$n, big.mark = ","), format(roundTrip$p, big.mark = ","),
            format(roundTrip$a, big.mark = ",")))
if (roundTrip$n != nrow(after)) {
  cat("ERROR: round-trip row count mismatch\n"); quit(status = 2)
}

cat("\nDONE. All checks passed.\n")
quit(status = 0)
