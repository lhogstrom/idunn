# Builds the harmonized SQLite database: sample/patient metadata and observed variants from
# Hartwig, PCAWG pan-cancer, AACR GENIE, and TCGA MC3, in harmonizedSampleInfo and
# patientObservedVariantTable (raw copies go to a second DB).
# Invoked by workflow/main.nf: Rscript <script> <baseDir> <harmonizedDbName> <rawDbName>

library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)

# GENIE redacts the tails of numeric clinical fields as inequality strings, e.g. age ">89"/"<18"
# and intervals ">32485"/"<6570" (days). A bare as.numeric() turns these into NA, so they are
# recoded to the nearest value consistent with the inequality (">89" -> 90, "<18" -> 17).
# Genuinely missing values ("Unknown", "Not Collected", ...) still become NA.
recode_genie_redaction <- function(x, upper_token, upper_value,
                                   lower_token, lower_value) {
  x <- trimws(as.character(x))
  x[!is.na(x) & x == upper_token] <- as.character(upper_value)
  x[!is.na(x) & x == lower_token] <- as.character(lower_value)
  suppressWarnings(as.numeric(x))
}

recode_genie_age <- function(x) {
  recode_genie_redaction(x, ">89", 90, "<18", 17)
}

# Day-scale analogue for the survival-interval fields: one day past the redaction bound.
recode_genie_interval_days <- function(x) {
  recode_genie_redaction(x, ">32485", 32486, "<6570", 6569)
}

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Three arguments must be supplied", call. = FALSE)
}

baseDir <- args[1] # should be the relative path to "balder/code" where this file is located
outDbName <- args[2]
outDbName2 <- args[3]

print("working dir:")
print(getwd())

print("base dir:")
print(baseDir)

outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522")

print("file for db connection:")
print(outDbName)


bDir <- paste0(baseDir,"/../../data/processed/balderResultsDb")
timestamp <- format(Sys.time(), "%Y%m%d") 

harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), outDbName)
rawDataDb <- DBI::dbConnect(RSQLite::SQLite(), outDbName2)


harmSampleCols <- c("SAMPLE_ID",
                    "PATIENT_ID",
                    "CANCER_TYPE",
                    "CANCER_TYPE_DETAILED",
                    "ONCOTREE_CODE",
                    "SAMPLE_TYPE",
                    "SAMPLE_TYPE_DETAILED",
                    "SEQ_ASSAY_ID",
                    "STAGE",
                    "TUMOR_PURITY",
                    "INT_Biopsy_To_Death",
                    "AGE_AT_SEQ_REPORT",
                    "DEAD",
                    "INT_SEQ_TO_DEATH",
                    "INT_SEQ_TO_CONTACT",
                    "SourceStudy",
                    "SEX",
                    "RACE",
                    "ETHNICITY",
                    "TUMOR_PLOIDY",
                    "CENTER")


hartwigDir <- paste0(baseDir,"/../../data/Hartwig/data")
inFile <- paste0(hartwigDir,"/samples.txt")
sampleList <- read.csv(inFile,sep=",", header = F)

inFile <- paste0(hartwigDir,"/metadata.tsv")
hMeta <- read.csv(inFile,sep="\t")

inFile <- paste0(hartwigDir,"/pre_biopsy_drugs.tsv")
hTreatment <- read.csv(inFile,sep="\t")

RSQLite::dbWriteTable(rawDataDb, "hartwigSampleInfo", sampleList, overwrite = T)
RSQLite::dbWriteTable(rawDataDb, "hartwigMetadata", hMeta, overwrite = T)
RSQLite::dbWriteTable(rawDataDb, "hartwigPreBiopsyDrugs", hTreatment, overwrite = T)

biopsyDateNum <- as.Date(hMeta$biopsyDate,"%Y-%m-%d")
deathDateNum <- as.Date(hMeta$deathDate,"%Y-%m-%d")
hMeta$INT_Biopsy_To_Death <- deathDateNum - biopsyDateNum

hSampleSubset <- hMeta[,c("sampleId","hmfPatientId","primaryTumorLocation","primaryTumorType","tumorPurity","INT_Biopsy_To_Death","gender","birthYear","biopsyDate")]
colnames(hSampleSubset) <- c("SAMPLE_ID","PATIENT_ID","CANCER_TYPE","CANCER_TYPE_DETAILED","TUMOR_PURITY","INT_Biopsy_To_Death","SEX","birthYear","biopsyDate")
hSampleSubset$SAMPLE_TYPE <- "Metastasis"
hSampleSubset$SourceStudy <- "Hartwig-data"
hSampleSubset$SourceStudyInfo <- "Hartwig_WGS_n4495"
hSampleSubset$ONCOTREE_CODE <- NA
hSampleSubset$SAMPLE_TYPE_DETAILED <- NA
hSampleSubset$SEQ_ASSAY_ID <- NA
hSampleSubset$DEAD <- "Unknown"
hSampleSubset$STAGE <- NA_character_
hSampleSubset$INT_SEQ_TO_DEATH <- hSampleSubset$INT_Biopsy_To_Death
hSampleSubset$INT_SEQ_TO_CONTACT <- NA

iBirthBiopYearNull <- hSampleSubset$biopsyDate=="null" | hSampleSubset$birthYear=="null"
hSampleSubset$biopsyDateFormat <- NA
tmpDate <- as.Date(hSampleSubset[!iBirthBiopYearNull,"biopsyDate"],"%Y")
hSampleSubset[!iBirthBiopYearNull,"biopsyDateFormat"] <- format(tmpDate, "%Y")
hSampleSubset$AGE_AT_SEQ_REPORT <- NA
hSampleSubset[!iBirthBiopYearNull,"AGE_AT_SEQ_REPORT"] <- as.numeric(hSampleSubset[!iBirthBiopYearNull,"biopsyDateFormat"]) - as.numeric(hSampleSubset[!iBirthBiopYearNull,"birthYear"])

hSampleSubset$RACE <- "Unknown"
hSampleSubset$ETHNICITY <- "Unknown"
hSampleSubset$TUMOR_PLOIDY <- NA
hSampleSubset$CENTER <- NA

iMale <- hSampleSubset$SEX=="male"
hSampleSubset[iMale,"SEX"] <- "Male"
iFemale <- hSampleSubset$SEX=="female"
hSampleSubset[iFemale,"SEX"] <- "Female"
ifmNull <- hSampleSubset$SEX=="null"
hSampleSubset[ifmNull,"SEX"] <- "Unknown"

RSQLite::dbWriteTable(harmonizedDb, "harmonizedSampleInfo", hSampleSubset[,harmSampleCols],overwrite=T)


inFile <- paste0(baseDir,"/../../data/ICGC_TCGA_WGS_2020/pancan_pcawg_2020/data_clinical_sample.txt")
pancanSampInfo <- read.csv(inFile,sep="\t",skip = 4)
RSQLite::dbWriteTable(rawDataDb, "pancanMetadata", pancanSampInfo, overwrite=T)

inFile <- paste0(baseDir,"/../../data/ICGC_TCGA_WGS_2020/pancan_pcawg_2020/data_clinical_patient.txt")
pancanPatientInfo <- read.csv(inFile,sep="\t",skip = 4)
RSQLite::dbWriteTable(rawDataDb, "PancanPatienteData", pancanPatientInfo, overwrite=T)

pancanSampInfo <- pancanSampInfo %>%
  dplyr::left_join(pancanPatientInfo,by="PATIENT_ID")

pancanSampleSubset <- pancanSampInfo[,c("SAMPLE_ID",
                                        "PATIENT_ID",
                                        "CANCER_TYPE",
                                        "CANCER_TYPE_DETAILED",
                                        "ONCOTREE_CODE",
                                        "SAMPLE_TYPE",
                                        "STAGE",
                                        "PURITY",
                                        "PLOIDY",
                                        "SEX",
                                        "OS_STATUS",
                                        "OS_MONTHS",
                                        "AGE")]
pancanSampleSubset$SourceStudy <- "PANCAN-WGS-data"
pancanSampleSubset$SourceStudyInfo <- "PANCAN_WGS_n2583"
pancanSampleSubset$TUMOR_PURITY <- pancanSampleSubset$PURITY
pancanSampleSubset$TUMOR_PLOIDY <- pancanSampleSubset$PLOIDY
pancanSampleSubset$SAMPLE_TYPE_DETAILED <- NA
pancanSampleSubset$SEQ_ASSAY_ID <- NA
pancanSampleSubset$AGE_AT_SEQ_REPORT <- pancanSampleSubset$AGE
pancanSampleSubset$INT_Biopsy_To_Death <- NA
pancanSampleSubset$INT_SEQ_TO_DEATH <- as.numeric(pancanSampleSubset$OS_MONTHS) * 31
pancanSampleSubset$INT_SEQ_TO_CONTACT <- NA

pancanSampleSubset$RACE <- "Unknown"
pancanSampleSubset$ETHNICITY <- "Unknown"
pancanSampleSubset$CENTER <- NA

pancanSampleSubset$DEAD <- "Unknown"
iNotDead <- pancanSampleSubset$OS_STATUS=="0:LIVING" & !is.na(pancanSampleSubset$OS_STATUS)
pancanSampleSubset[iNotDead,"DEAD"] <- "False"
iDead <- pancanSampleSubset$OS_STATUS=="1:DECEASED" & !is.na(pancanSampleSubset$OS_STATUS)
pancanSampleSubset[iDead,"DEAD"] <- "True"
  
RSQLite::dbWriteTable(harmonizedDb, "harmonizedSampleInfo", pancanSampleSubset[,harmSampleCols],append=T)

inFile <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_17p0_public/data_clinical_patient.txt")
patient.genie <- read.csv(inFile,sep="\t",skip=4)
RSQLite::dbWriteTable(rawDataDb, "GeniePatientData", patient.genie, overwrite=T)

inFile <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_17p0_public/data_clinical_sample.txt")
sample.genie <- read.csv(inFile,sep="\t",skip=4)
RSQLite::dbWriteTable(rawDataDb, "GenieClinicalSampleData", sample.genie, overwrite=T)

sample.genie$AGE_AT_SEQ_REPORT_NUMERIC <- recode_genie_age(sample.genie$AGE_AT_SEQ_REPORT)
sample.genie$INF_AGE_SEQ_REPORT_DAYS <- (sample.genie$AGE_AT_SEQ_REPORT_NUMERIC * 365) + 183
samplePatient <- sample.genie %>%
  dplyr::left_join(patient.genie,by="PATIENT_ID") %>%
  dplyr::mutate(
    INT_SEQ_TO_DEATH = dplyr::if_else(
      !is.na(recode_genie_interval_days(INT_DOD)) & !is.na(as.numeric(INF_AGE_SEQ_REPORT_DAYS)),
      recode_genie_interval_days(INT_DOD) - INF_AGE_SEQ_REPORT_DAYS,
      NA
    ),
    INT_SEQ_TO_CONTACT = dplyr::if_else(
      !is.na(recode_genie_interval_days(INT_CONTACT)) & !is.na(as.numeric(INF_AGE_SEQ_REPORT_DAYS)),
      recode_genie_interval_days(INT_CONTACT) - INF_AGE_SEQ_REPORT_DAYS,
      NA
    )
  )

genieSampleSubset <- samplePatient[,c("SAMPLE_ID",
                                      "PATIENT_ID",
                                      "CANCER_TYPE",
                                      "CANCER_TYPE_DETAILED",
                                      "ONCOTREE_CODE",
                                      "SAMPLE_TYPE",
                                      "SAMPLE_TYPE_DETAILED",
                                      "SEQ_ASSAY_ID",
                                      "AGE_AT_SEQ_REPORT",
                                      "DEAD",
                                      "INT_SEQ_TO_DEATH",
                                      "INT_SEQ_TO_CONTACT",
                                      "SEX",
                                      "PRIMARY_RACE",
                                      "ETHNICITY",
                                      "CENTER")]
genieSampleSubset$SourceStudy <- "AACR-GENIE-data"
genieSampleSubset$SourceStudyInfo <- "AACR_GENIE_TARGETED_n161797"
genieSampleSubset$AGE_AT_SEQ_REPORT <- recode_genie_age(genieSampleSubset$AGE_AT_SEQ_REPORT)
genieSampleSubset$STAGE <- NA_character_
genieSampleSubset$TUMOR_PURITY <- NA
genieSampleSubset$INT_Biopsy_To_Death <- NA

genieSampleSubset$AGE <- as.numeric(genieSampleSubset$AGE_AT_SEQ_REPORT)
genieSampleSubset$RACE <- genieSampleSubset$PRIMARY_RACE
genieSampleSubset$TUMOR_PLOIDY <- NA

iraceNa <- genieSampleSubset$RACE=="Not Applicable"
genieSampleSubset[iraceNa,"RACE"] <- "Unknown"
iraceNc <- genieSampleSubset$RACE=="Not Collected"
genieSampleSubset[iraceNc,"RACE"] <- "Unknown"

iEthnicityNc <- genieSampleSubset$ETHNICITY=="Not Collected"
genieSampleSubset[iEthnicityNc,"ETHNICITY"] <- "Unknown"

iDeadNc <- genieSampleSubset$DEAD=="Not Collected"
genieSampleSubset[iDeadNc,"DEAD"] <- "Unknown"
iDeadNr <- genieSampleSubset$DEAD=="Not Released"
genieSampleSubset[iDeadNr,"DEAD"] <- "Unknown"
iDeadNa <- genieSampleSubset$DEAD==""
genieSampleSubset[iDeadNa,"DEAD"] <- "Unknown"
iDeadT <- genieSampleSubset$DEAD=="TRUE"
genieSampleSubset[iDeadT,"DEAD"] <- "True"
iDeadF <- genieSampleSubset$DEAD=="FALSE"
genieSampleSubset[iDeadF,"DEAD"] <- "False"

RSQLite::dbWriteTable(harmonizedDb, "harmonizedSampleInfo", genieSampleSubset[,harmSampleCols],append=T)

### MC3 TCGA
inFile <- paste0(baseDir,"/../../data/curration/TCGA_tissue_source_site_codes.csv")
tss <- read.csv(inFile,sep=",")
RSQLite::dbWriteTable(harmonizedDb, "Mc3TCGASiteCodes", tss, overwrite=T)

inFile <- paste0(baseDir,"/../../data/mc3_tcga/clinical_data/combined_study_clinical_data.tsv")
tcgaPatient <- read.csv(inFile,sep="\t")

tcga_in_cols <- c("Patient.ID",
                  "Sample.ID",
                  "Cancer.Type",
                  "Cancer.Type.Detailed",
                  "American.Joint.Committee.on.Cancer.Metastasis.Stage.Code",
                  "Neoplasm.Disease.Stage.American.Joint.Committee.on.Cancer.Code",
                  "Oncotree.Code",
                  "Months.of.disease.specific.survival",
                  "Diagnosis.Age",
                  "Disease.specific.Survival.status",
                  "Sex",
                  "Race.Category",
                  "Ethnicity.Category",
                  "Tissue.Source.Site")
tcgaSubset <- tcgaPatient[,tcga_in_cols]
tcga_cols_rename <- c("SAMPLE_ID",
                      "PATIENT_ID",
                      "CANCER_TYPE",
                      "CANCER_TYPE_DETAILED",
                      "SAMPLE_TYPE",
                      "STAGE",
                      "ONCOTREE_CODE",
                      "Months.of.disease.specific.survival",
                      "AGE_AT_SEQ_REPORT",
                      "Disease.specific.Survival.status",
                      "SEX",
                      "RACE",
                      "ETHNICITY",
                      "CENTER")
colnames(tcgaSubset) <- tcga_cols_rename

tcgaSubset$SourceStudy <- "TCGA-MC3-data"
tcgaSubset$INT_SEQ_TO_DEATH <- as.numeric(tcgaSubset$Months.of.disease.specific.survival) * 31

iDeadF <- tcgaSubset$Disease.specific.Survival.status == "0:ALIVE OR DEAD TUMOR FREE" & !is.na(tcgaSubset$Disease.specific.Survival.status)
tcgaSubset[iDeadF,"DEAD"] <- "False"
iDeadT <- tcgaSubset$Disease.specific.Survival.status == "1:DEAD WITH TUMOR" & !is.na(tcgaSubset$Disease.specific.Survival.status)
tcgaSubset[iDeadF,"DEAD"] <- "True"

iBlack <- tcgaSubset$RACE =="Black or African American" & !is.na(tcgaSubset$RACE)
tcgaSubset[iBlack,"RACE"] <- "Black"
iNAm <- tcgaSubset$RACE =="American Indian or Alaska Native"  & !is.na(tcgaSubset$RACE)
tcgaSubset[iNAm,"RACE"] <- "Native American"
iPa <- tcgaSubset$RACE =="Native Hawaiian or Other Pacific Islander"  & !is.na(tcgaSubset$RACE)
tcgaSubset[iPa,"RACE"] <- "Pacific Islander"

tcgaSubset$SAMPLE_TYPE_DETAILED <- NA
tcgaSubset$SEQ_ASSAY_ID <- NA
tcgaSubset$TUMOR_PURITY <- NA
tcgaSubset$INT_Biopsy_To_Death <- NA
tcgaSubset$INT_SEQ_TO_CONTACT <- NA
tcgaSubset$TUMOR_PLOIDY <- NA
tcgaSubset$INT_Biopsy_To_Death <- NA

print(harmSampleCols %in% colnames(tcgaSubset))
print(harmSampleCols[!harmSampleCols %in% colnames(tcgaSubset)])

RSQLite::dbWriteTable(harmonizedDb, "harmonizedSampleInfo", tcgaSubset[,harmSampleCols], append=T)


sampleDataSources <- list(pancanSampInfo,
                       pancanPatientInfo,
                       patient.genie,
                       sample.genie,
                       hMeta,
                       hTreatment)

varNames <- c("pancan sample Info",
                          "pancan Patient Info",
                          "GENIE patient info",
                          "GENIE sample genie",
                          "Hartwig Sample Metadata",
                          "Hartwig Subject Treatment")

for (ix in seq(length(sampleDataSources))) {
  varName <- varNames[ix]
  sourceData <- sampleDataSources[[ix]]
  print(varName)
  tibble::glimpse(sourceData)
}


inFile <- paste0(baseDir,"/../../data/ICGC_TCGA_WGS_2020/pancan_pcawg_2020/data_mutations.txt")
svPancan <- read.csv(inFile,sep="\t",skip = 2)
RSQLite::dbWriteTable(rawDataDb, "PancanPatientVarients", svPancan, overwrite=T)


okbGeneList <- paste0(baseDir,"/../../data/oncokb/cancerGeneList.tsv") 
oncokbGenes <- read.csv(okbGeneList,sep="\t")

inFile <- paste0(baseDir,"/../../data/mc3_tcga/mc3.v0.2.8.PUBLIC.maf")
genie <- read.csv(inFile,sep="\t")
mc3 <- read.csv(inFile,sep="\t") %>%
  dplyr::rename(StrandPlusMinus=STRAND,
                Tumor_Sample_Barcode_Full=Tumor_Sample_Barcode) %>%
  dplyr::filter(Hugo_Symbol %in% oncokbGenes$Hugo.Symbol)
mc3$Tumor_Sample_Barcode <- substr(mc3$Tumor_Sample_Barcode_Full,1,15)


inFile <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_17p0_public/data_mutations_extended.txt")
genie <- read.csv(inFile,sep="\t")
RSQLite::dbWriteTable(rawDataDb, "GeniePatientVarients", genie, overwrite=T)

inFile <- paste0(baseDir,"/../../data/Hartwig/processed/hartwig_sample_pcgr_maf_oncokb_gene_subset.maf")
hartwig.pcgr <- read.csv(inFile,sep="\t")
RSQLite::dbWriteTable(rawDataDb, "HartwigPatientVarients", hartwig.pcgr, overwrite=T)


pancanCols <- colnames(svPancan)
genieCols <- colnames(genie)
mc3Cols <- colnames(mc3)
hartwigCols <- colnames(hartwig.pcgr)
uCols <- base::intersect(base::intersect(pancanCols,genieCols),base::intersect(hartwigCols,mc3Cols))

varTable <- svPancan[,uCols]
varTable$TVAF <- NA
varTable$Protein_position <- as.character(varTable$Protein_position)
varTable$SourceStudy <- "PANCAN-WGS-data"
RSQLite::dbWriteTable(harmonizedDb, "patientObservedVariantTable", varTable, overwrite=T)


varTableMC3 <- mc3[,uCols]
varTableMC3$TVAF <- NA
varTableMC3$SourceStudy <- "TCGA-MC3-data"
RSQLite::dbWriteTable(harmonizedDb, "patientObservedVariantTable", varTableMC3,append=T)

varTableGenie <- genie[,uCols]
varTableGenie$TVAF <- NA
varTableGenie$Protein_position <- as.character(varTableGenie$Protein_position)
varTableGenie$SourceStudy <- "AACR-GENIE-data"
RSQLite::dbWriteTable(harmonizedDb, "patientObservedVariantTable", varTableGenie,append=T)

varTableHartwig <- hartwig.pcgr[,c(uCols,"TVAF")]
varTableHartwig$SourceStudy <- "Hartwig-data"
RSQLite::dbWriteTable(harmonizedDb, "patientObservedVariantTable", varTableHartwig,append=T)


RSQLite::dbDisconnect(harmonizedDb)
RSQLite::dbDisconnect(rawDataDb)

outRFile <- paste0(outDir,"/genomicStudyiesDbCurration-",timestamp,".RData")
save.image(file = outRFile)
