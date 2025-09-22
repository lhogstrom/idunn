library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)

args <- commandArgs(trailingOnly = TRUE)

# Check if the correct number of arguments are provided
if (length(args) != 2) {
  stop("Two arguments must be supplied", call. = FALSE)
}

# Assign Arguments to Variables
baseDir <- args[1] # should be the relative path to "balder/code" where this file is located
dbName <- args[2]

### connect to result DB and get variants
outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522/genomic_panel_coverage")
if (!dir.exists(geneOutDir)) {
  dir.create(geneOutDir, recursive = TRUE)
}

harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

perPatientOncoKBLevelSorted <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM perPatientOncoKBLevelSorted')
genomicPositionLookup <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM genomicPositionLookup')





#load positoin lookup table
#genomicPositionLookup

#Steps: 
#- filter variants with coverage <20
#- filter variants where panel coverage is less than 90%
#- filter PANCAN and Hartwig Data - GENIE only
#- rename race fields
#- Therapy selection events only (Level 1) only

#- add criteria of >250 actionable patients per cancer type

# # Identify those genomic positions with genomic panel coverage in most of the samples tested
# genomicPositionSubset <- genomicPositionLookup %>%
#   dplyr::filter(perc_of_patients_with_coverage > 90.0)
# 
# dataForModeling <- perPatientOncoKBLevelSorted %>%
#   dplyr::filter(SourceStudy=="AACR-GENIE-data") %>%
#   dplyr::filter(t_alt_plus_ref_count >= 20) %>%
#   dplyr::mutate(positionCtypeID=paste0(ONCOTREE_CODE,"-",Hugo_Symbol,"-",Start_Position)) %>%
#   dplyr::filter(positionCtypeID %in% genomicPositionSubset$positionCtypeID)
# 


#### Alterinative approach retain filtered data for counting: 

# 1) Precompute the genomic position subset (and the key you'll join on)
genomicPositionSubset <- genomicPositionLookup %>%
  filter(perc_of_patients_with_coverage > 90) #%>%

# 2) Start and step through filters, recording counts
df1 <- perPatientOncoKBLevelSorted %>%
  mutate(
    RACE = recode(
      RACE,
      "White" = "European",
      "Black" = "African"
    )
  )
df2 <- df1 %>% filter(SourceStudy == "AACR-GENIE-data")                # PANCAN/Hartwig removed (GENIE only)
df3 <- df2 %>% filter(isOncoKbLevel1 == T)                
df4 <- df3 %>% filter(ONCOKB_HAS_SENSITIVE_ENTRY == T)                
df5 <- df4 %>% filter(t_alt_plus_ref_count >= 20)                      # coverage >= 20
df6 <- df5 %>% mutate(positionCtypeID = paste0(ONCOTREE_CODE, "-", Hugo_Symbol, "-", Start_Position))
df7 <- df6 %>% filter(positionCtypeID %in% genomicPositionSubset$positionCtypeID)  # panel coverage >= 90% positions

cTypeCntTble <- df7 %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::mutate(cTypeCntThresh = n >= 250) %>%
  dplyr::arrange(desc(n))
cTypeSelected <- cTypeCntTble[cTypeCntTble$cTypeCntThresh==T,]

df8 <- df7 %>%
  dplyr::filter(ONCOTREE_CODE %in% cTypeSelected$ONCOTREE_CODE)

# 3) Build the audit table
audit_table_actionable <- tibble(
  step = c(
    "Start",
    "Filter: GENIE only",
    "Filter: Level 1 actionable events",
    "Filter: Drug sensitivity events",
    "Filter: coverage ≥ 20",
    "Mutate: add positionCtypeID",
    "Filter: panel coverage ≥ 90% positions",
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

### repeat for non-actionable
dfNotLvl1a <- perPatientOncoKBLevelSorted %>%
  mutate(
    RACE = recode(
      RACE,
      "White" = "European",
      "Black" = "African"
    )
  )
dfNotLvl1b <- dfNotLvl1a %>% filter(SourceStudy == "AACR-GENIE-data")                # PANCAN/Hartwig removed (GENIE only)
dfNotLvl1c <- dfNotLvl1b %>% filter(isOncoKbLevel1 == F)               
dfNotLvl1d <- dfNotLvl1c %>% filter(t_alt_plus_ref_count >= 20)                      # coverage >= 20
dfNotLvl1e <- dfNotLvl1d %>%
  dplyr::filter(ONCOTREE_CODE %in% cTypeSelected$ONCOTREE_CODE)

audit_table_not_actionable <- tibble(
  step = c(
    "Start",
    "Filter: GENIE only",
    "Filter: No Level 1 actionable events",
    "Filter: coverage ≥ 20",
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

RSQLite::dbWriteTable(harmonizedDb, "dataForModeling", perPatientOncoKBFilteredForModeling, overwrite =T)

### Disconnect from SQL db ###
RSQLite::dbDisconnect(harmonizedDb)

