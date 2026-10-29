# Filters the per-sample OncoKB table to the modeling cohort: GENIE only, OncoKB Level 1
# sensitive events (actionable) vs. no Level 1 event (non-actionable), median coverage above the
# bottom 1%, panel coverage >90% of patients (actionable arm), cancer types with >=250 actionable
# samples. Writes audit tables and the perPatientOncoKBFilteredForModeling table in the SQLite DB.
# Invoked by workflow/main.nf: Rscript <script> <baseDir> <dbName>

library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Two arguments must be supplied", call. = FALSE)
}

baseDir <- args[1] # should be the relative path to "balder/code" where this file is located
dbName <- args[2]

outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522")
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

perPatientOncoKBLevelSorted <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM perPatientOncoKBLevelSorted')
genomicPositionLookup <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM genomicPositionLookup')
medianCoveragePerSample <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM medianCoveragePerSample')

genomicPositionSubset <- genomicPositionLookup %>%
  filter(perc_of_patients_with_coverage > 90)


# Drop samples in the bottom 1% of median coverage
data_vec <- medianCoveragePerSample[medianCoveragePerSample$SourceStudy == "AACR-GENIE-data", "medianCoverage"]
print(summary(data_vec))
cutoffs <- quantile(data_vec, probs = c(0.01, 0.03, 0.05), na.rm = TRUE)
print(cutoffs)
bottom_1_perc <- cutoffs[1]

medianCoverageSubset <- medianCoveragePerSample %>%
  filter(!is.na(medianCoverage)) %>% 
  filter(medianCoverage >= bottom_1_perc) 

# Sequential filters, recording row counts for the audit table
df1 <- perPatientOncoKBLevelSorted %>%
  mutate(
    RACE = recode(
      RACE,
      "White" = "European",
      "Black" = "African"
    )
  )
df2 <- df1 %>% filter(SourceStudy == "AACR-GENIE-data")  # GENIE only
df3 <- df2 %>% filter(isOncoKbLevel1 == T)                
df4 <- df3 %>% filter(ONCOKB_HAS_SENSITIVE_ENTRY == T)                
df5 <- df4 %>% filter(Tumor_Sample_Barcode %in% medianCoverageSubset$Tumor_Sample_Barcode)  # drop bottom 1% of coverage
df6 <- df5 %>% mutate(positionCtypeID = paste0(ONCOTREE_CODE, "-", Hugo_Symbol, "-", Start_Position))
df7 <- df6 %>% filter(positionCtypeID %in% genomicPositionSubset$positionCtypeID)  # keep positions covered in >90% of patients

cTypeCntTble <- df7 %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::mutate(cTypeCntThresh = n >= 250) %>%
  dplyr::arrange(desc(n))
cTypeSelected <- cTypeCntTble[cTypeCntTble$cTypeCntThresh==T,]

df8 <- df7 %>%
  dplyr::filter(ONCOTREE_CODE %in% cTypeSelected$ONCOTREE_CODE)

# Audit table of row counts per filter step
audit_table_actionable <- tibble(
  step = c(
    "Start",
    "Filter: GENIE only",
    "Filter: Level 1 actionable events",
    "Filter: Drug sensitivity events",
    "Filter: medain coverage above 1 perc",
    "Mutate: add positionCtypeID",
    "Filter: panel coverage 90% or greater positions",
    "Filter: cancer type selection"
  ),
  rows = c(nrow(df1), nrow(df2), nrow(df3), nrow(df4), nrow(df5), nrow(df6), nrow(df7), nrow(df8))
) %>%
  mutate(
    delta = rows - dplyr::lag(rows),
    pct_change = if_else(is.na(dplyr::lag(rows)), NA_real_, 100 * (delta / dplyr::lag(rows)))
  )

outF <- paste0(outDir,"/level1_actionable_event_filtering_results.txt")
write.table(audit_table_actionable,outF,row.names=F,quote=F,sep="\t")

dfNotLvl1a <- perPatientOncoKBLevelSorted %>%
  mutate(
    RACE = recode(
      RACE,
      "White" = "European",
      "Black" = "African"
    )
  )
dfNotLvl1b <- dfNotLvl1a %>% filter(SourceStudy == "AACR-GENIE-data")  # GENIE only
dfNotLvl1c <- dfNotLvl1b %>% filter(isOncoKbLevel1 == F)               
dfNotLvl1d <- dfNotLvl1c %>% filter(Tumor_Sample_Barcode %in% medianCoverageSubset$Tumor_Sample_Barcode)
dfNotLvl1e <- dfNotLvl1d %>%
  dplyr::filter(ONCOTREE_CODE %in% cTypeSelected$ONCOTREE_CODE)

audit_table_not_actionable <- tibble(
  step = c(
    "Start",
    "Filter: GENIE only",
    "Filter: No Level 1 actionable events",
    "Filter: medain coverage above 1 perc",
    "Filter: cancer type selection"
  ),
  rows = c(nrow(dfNotLvl1a), nrow(dfNotLvl1b), nrow(dfNotLvl1c), nrow(dfNotLvl1d), nrow(dfNotLvl1e))
) %>%
  mutate(
    delta = rows - dplyr::lag(rows),
    pct_change = if_else(is.na(dplyr::lag(rows)), NA_real_, 100 * (delta / dplyr::lag(rows)))
  )

outF <- paste0(outDir,"/not_level1_actionable_event_filtering_results.txt")
write.table(audit_table_not_actionable,outF,row.names=F,quote=F,sep="\t")

dataForModeling <- rbind(df8,dfNotLvl1e)

outF <- paste0(outDir,"/data_product_level1_actionable_events_after_filtering_results.txt")
write.table(dataForModeling,outF,row.names=F,quote=F,sep="\t")

RSQLite::dbWriteTable(harmonizedDb, "perPatientOncoKBFilteredForModeling", dataForModeling, overwrite =T)

RSQLite::dbDisconnect(harmonizedDb)

