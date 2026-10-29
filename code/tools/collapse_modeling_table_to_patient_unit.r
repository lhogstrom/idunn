#!/usr/bin/env Rscript
# Collapses perPatientOncoKBFilteredForModeling from one row per sample to one row per
# (PATIENT_ID, ONCOTREE_CODE) pair, in place on a COPY of the database. Not part of workflow/main.nf.
# Rule: keep one sample per pair, chosen by md5(Tumor_Sample_Barcode). The pick is deterministic,
# independent of input row order, and not selected on the outcome.
# Fails closed: refuses a file not named *patientunit* unless --force; verification checks gate the write.
# Usage: Rscript collapse_modeling_table_to_patient_unit.r --db PATH [--report-dir DIR] [--force] [--dry-run]
# Exit codes: 0 ok, 1 usage/precondition failure, 2 verification failure.

suppressPackageStartupMessages({
  library(dplyr)
  library(DBI)
  library(RSQLite)
})

TABLE <- "perPatientOncoKBFilteredForModeling"
MIN_ACTIONABLE <- 250   # must match define_cancer_type_groupings.py
EXCLUDED_RESCUE_CODES <- c("THYROID", "ENCG", "BLADDER")  # must match define_cancer_type_groupings.py

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

# Fail closed: refuse anything that is not obviously a purpose-built copy unless forced.
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
# md5 of the barcode string, matching Python hashlib.md5(s.encode()).hexdigest().
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

# Operates on a data.frame and returns the collapsed frame.
# method = "radix" sorts in the C locale, so the pick is reproducible across locales.
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

# 4. the pick is order-invariant: collapsing a row-shuffled copy retains the same barcodes
set.seed(20260826)  # shuffles the input only; the rule itself uses no RNG
shuffled <- before[sample.int(nrow(before)), , drop = FALSE]
reCollapsed <- collapse_to_patient_unit(shuffled)
addcheck("pick_is_order_invariant",
         setequal(after$Tumor_Sample_Barcode, reCollapsed$Tumor_Sample_Barcode) &&
           nrow(reCollapsed) == nrow(after),
         sprintf("%d of %d retained barcodes differ after a full row shuffle",
                 length(setdiff(after$Tumor_Sample_Barcode, reCollapsed$Tumor_Sample_Barcode)),
                 nrow(after)))

# 5. each representative belongs to its own group
srcKey <- before %>% dplyr::select(Tumor_Sample_Barcode, srcPat = PATIENT_ID,
                                   srcCode = ONCOTREE_CODE)
memb <- after %>%
  dplyr::select(Tumor_Sample_Barcode, PATIENT_ID, ONCOTREE_CODE) %>%
  dplyr::left_join(srcKey, by = "Tumor_Sample_Barcode")
nBadMemb <- sum(is.na(memb$srcPat) | memb$srcPat != memb$PATIENT_ID |
                  memb$srcCode != memb$ONCOTREE_CODE)
addcheck("representative_is_group_member", nBadMemb == 0,
         sprintf("%d retained rows are not members of the pair they represent", nBadMemb))

# 6. eligibility gate: the SET of codes with >= MIN_ACTIONABLE distinct patients must be unchanged
#    (counts are reported, not asserted); counting matches define_cancer_type_groupings.py
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

# per-cancer table for ALL codes (not only eligible ones), to show the margin to the threshold
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

# Write artifacts before printing, so a display bug cannot cost the record.
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

# ---- threshold boundary summary ------------------------------------------
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
