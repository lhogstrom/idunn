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

outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522")

### connect to result DB and get variants
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

genomicPositionLookup <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM genomicPositionLookup')
svCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM patientObservedVariantTableWithSampleInfoOncokb') %>%
  dplyr::mutate(isHemeCancer=grepl("LYMPH",ot_code_level_1) | grepl("MYELOID",ot_code_level_1))
svCompiled$SEQ_ASSAY_ID_mod <- svCompiled$SEQ_ASSAY_ID
svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SEQ_ASSAY_ID_mod"] <- svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SourceStudy"]


#############################################################
###  per-patient data product of level1 and level2 status ###
#############################################################

# load genie info
assayInfo <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM assayInfoGENIE')

genomicPositionSubset <- genomicPositionLookup %>%
  filter(perc_of_patients_with_coverage > 90) # only those postions that are covered for most patients

# create representative variant per row per subject - choose level 1 variant if available
perPatientOncoKBLevelSorted <- svCompiled %>%
  dplyr::rename(SequenceCenter = Center) %>%
  dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID")) %>%
  dplyr::mutate(isOncoKbLevel1=ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_1" & !is.na(ONCOKB_HIGHEST_LEVEL_SUMMARY),
                isOncoKbLevel2=ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_2" & !is.na(ONCOKB_HIGHEST_LEVEL_SUMMARY),
                isOncoKbLevel3=ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_3" & !is.na(ONCOKB_HIGHEST_LEVEL_SUMMARY),
                isOncoKbLevelAny=ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_" & !is.na(ONCOKB_HIGHEST_LEVEL_SUMMARY),
                isOncoKbLevel2or1=(ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_1" | ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_2") & !is.na(ONCOKB_HIGHEST_LEVEL_SUMMARY)) %>%
  dplyr::mutate(isHemeCancer=grepl("LYMPH",ot_code_level_1) | grepl("MYELOID",ot_code_level_1)) %>%
  dplyr::mutate(positionCtypeID = paste0(ONCOTREE_CODE, "-", Hugo_Symbol, "-", Start_Position),
                isHighCoverageByPanelDesign=positionCtypeID %in% genomicPositionSubset$positionCtypeID) %>%
  dplyr::group_by(Tumor_Sample_Barcode) %>%
  dplyr::arrange(desc(isHighCoverageByPanelDesign)) %>%
  dplyr::arrange(desc(isOncoKbLevel1),desc(isOncoKbLevel2or1),desc(isOncoKbLevelAny)) %>%
  dplyr::filter(dplyr::row_number()==1) %>%
  dplyr::ungroup()
outF <- paste0(outDir,"/patient_level1_OncoKB_status_data.txt")
write.table(perPatientOncoKBLevelSorted,outF,row.names=F,quote=F,sep="\t")
RSQLite::dbWriteTable(harmonizedDb, "perPatientOncoKBLevelSorted", perPatientOncoKBLevelSorted, overwrite =T)

### Disconnect from SQL db ###
RSQLite::dbDisconnect(harmonizedDb)
