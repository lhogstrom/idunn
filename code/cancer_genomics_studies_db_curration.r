library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(phenOncoX)

args <- commandArgs(trailingOnly = TRUE)

# Check if the correct number of arguments are provided
if (length(args) != 3) {
  stop("Three arguments must be supplied", call. = FALSE)
}

# Step 2: Assign Arguments to Variables
baseDir <- args[1] # should be the relative path to "balder/code" where this file is located
outDbName <- args[2]
outDbName2 <- args[3]
# outDbName <- paste0(baseDir,"/balder-harmonized-biomarker-data-v",timestamp,".sqlite")

print("working dir:")
print(getwd())

print("base dir:")
print(baseDir)

outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20240712")
#outDbName <- paste0(baseDir,"/../../data/processed/balderResultsDb/balder-harmonized-biomarker-data-v20240712.sqlite")

print("file for db connection:")
print(outDbName)

################################
### establish db connection ###
################################

bDir <- paste0(baseDir,"/../../data/processed/balderResultsDb")
timestamp <- format(Sys.time(), "%Y%m%d") 

# output db for harmonized data
#outDbName <- paste0(bDir,"/balder-harmonized-biomarker-data-v",timestamp,".sqlite")
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), outDbName)
# output db for compiled raw data
#outDbName2 <- paste0(bDir,"/balder-compiled-raw-data-v",timestamp,".sqlite")
rawDataDb <- DBI::dbConnect(RSQLite::SQLite(), outDbName2)

#############################
### get phenOncoX records ###
#############################

# download_dir <- tempdir()
# phenTreeDepth2 <- phenOncoX::get_tree(
#   cache_dir = download_dir, max_tree_depth = 2)
# 
# oncoterms <- phenOncoX::get_terms(
#   cache_dir = download_dir)
# 
# # select entries with DOI code
# doiRecords <- oncoterms$records %>%
#   dplyr::filter(!is.na(do_id)) 
# length(unique(doiRecords$do_id))
# otDoiRecords <- doiRecords %>%
#   dplyr::filter(!is.na(ot_code))
# length(unique(otDoiRecords$do_id))

# hMeta$doidsStr <- paste0("DOID:",hMeta$doids)
# table(hMeta$doidsStr %in% doiRecords$do_id)

######################################################
### patient and sample tables from various studies ###
######################################################

# define output columns for harmonized sample info
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

### load hartwig data

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

### calculate difference of age and sample type
biopsyDateNum <- as.Date(hMeta$biopsyDate,"%Y-%m-%d")
deathDateNum <- as.Date(hMeta$deathDate,"%Y-%m-%d")
hMeta$INT_Biopsy_To_Death <- deathDateNum - biopsyDateNum

# sample columns needed: cancer type, met/primary
hSampleSubset <- hMeta[,c("sampleId","hmfPatientId","primaryTumorLocation","primaryTumorType","tumorPurity","INT_Biopsy_To_Death","gender","birthYear","biopsyDate")]
colnames(hSampleSubset) <- c("SAMPLE_ID","PATIENT_ID","CANCER_TYPE","CANCER_TYPE_DETAILED","TUMOR_PURITY","INT_Biopsy_To_Death","SEX","birthYear","biopsyDate")
hSampleSubset$SAMPLE_TYPE <- "Metastasis" # assume all samples are met from Hartwig
hSampleSubset$SourceStudy <- "Hartwig-data"
hSampleSubset$SourceStudyInfo <- "Hartwig_WGS_n4495"
hSampleSubset$ONCOTREE_CODE <- NA # no code listed by authors
hSampleSubset$SAMPLE_TYPE_DETAILED <- NA
hSampleSubset$SEQ_ASSAY_ID <- NA
hSampleSubset$DEAD <- "Unknown"
hSampleSubset$STAGE <- NA_character_
hSampleSubset$INT_SEQ_TO_DEATH <- NA
hSampleSubset$INT_SEQ_TO_CONTACT <- NA

# get age based on birth year and biopsy date field
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

### PANCAN

# sample level info
inFile <- paste0(baseDir,"/../../data/ICGC_TCGA_WGS_2020/pancan_pcawg_2020/data_clinical_sample.txt")
pancanSampInfo <- read.csv(inFile,sep="\t",skip = 4)
RSQLite::dbWriteTable(rawDataDb, "pancanMetadata", pancanSampInfo, overwrite=T)

#  patient info
inFile <- paste0(baseDir,"/../../data/ICGC_TCGA_WGS_2020/pancan_pcawg_2020/data_clinical_patient.txt")
pancanPatientInfo <- read.csv(inFile,sep="\t",skip = 4)
RSQLite::dbWriteTable(rawDataDb, "PancanPatienteData", pancanPatientInfo, overwrite=T)

# subset selection
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

### AACR GENIE
inFile <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_14p1_public/data_clinical_patient.txt")
patient.genie <- read.csv(inFile,sep="\t",skip=4)
RSQLite::dbWriteTable(rawDataDb, "GeniePatientData", patient.genie, overwrite=T)

inFile <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_14p1_public/data_clinical_sample.txt")
sample.genie <- read.csv(inFile,sep="\t",skip=4)
RSQLite::dbWriteTable(rawDataDb, "GenieClinicalSampleData", sample.genie, overwrite=T)

### join sample data to patient data and create inferred variables
sample.genie$AGE_AT_SEQ_REPORT_NUMERIC <- as.numeric(sample.genie$AGE_AT_SEQ_REPORT)
sample.genie$INF_AGE_SEQ_REPORT_DAYS <- (sample.genie$AGE_AT_SEQ_REPORT_NUMERIC * 365) + 183
samplePatient <- sample.genie %>%
  dplyr::left_join(patient.genie,by="PATIENT_ID") %>%
  dplyr::mutate(
    INT_SEQ_TO_DEATH = dplyr::if_else(
      !is.na(as.numeric(INT_DOD)) & !is.na(as.numeric(INF_AGE_SEQ_REPORT_DAYS)),
      as.numeric(INT_DOD) - INF_AGE_SEQ_REPORT_DAYS,
      NA  # Assign NA of type double if conditions are not met
    ),
    INT_SEQ_TO_CONTACT = dplyr::if_else(
      !is.na(as.numeric(INT_CONTACT)) & !is.na(as.numeric(INF_AGE_SEQ_REPORT_DAYS)),
      as.numeric(INT_CONTACT) - INF_AGE_SEQ_REPORT_DAYS,
      NA  # Assign NA of type double if conditions are not met
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
genieSampleSubset$AGE_AT_SEQ_REPORT <- as.numeric(genieSampleSubset$AGE_AT_SEQ_REPORT)
genieSampleSubset$STAGE <- NA_character_
genieSampleSubset$TUMOR_PURITY <- NA
genieSampleSubset$INT_Biopsy_To_Death <- NA

genieSampleSubset$AGE <- as.numeric(genieSampleSubset$AGE_AT_SEQ_REPORT)
genieSampleSubset$RACE <- genieSampleSubset$PRIMARY_RACE
genieSampleSubset$TUMOR_PLOIDY <- NA

# harmonization steps
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
# TCGA site codes
inFile <- paste0(baseDir,"/../../data/curration/TCGA_tissue_source_site_codes.csv")
tss <- read.csv(inFile,sep=",")
RSQLite::dbWriteTable(harmonizedDb, "Mc3TCGASiteCodes", tss, overwrite=T)
# barcode naming convention: https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/
# tissue site codes: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes


inFile <- paste0(baseDir,"/../../data/mc3_tcga/clinical_data/combined_study_clinical_data.tsv")
tcgaPatient <- read.csv(inFile,sep="\t")


# tcga_in_cols <- c("Patient.ID,
#                   Sample.ID",
#                   "Cancer.Type",
#                   "Cancer.Type.Detailed",
#                   "",
#                   "",
#                   "",
#                   "",
#                   )
# tcgaSubset <- tcgaPatient
# # rename columns
# tcga_cols_rename <- c("SAMPLE_ID",
#                       "PATIENT_ID",
#                       "CANCER_TYPE",
#                       "CANCER_TYPE_DETAILED",
#                       "SAMPLE_TYPE",
#                       "",
#                       "",
#                       "",
#                       "",)
# 
# 
# 
# 
# 
# 
# 
# 
# SEQ_ASSAY_ID
# STAGE
# TUMOR_PURITY
# INT_Biopsy_To_Death
# AGE_AT_SEQ_REPORT
# DEAD
# INT_SEQ_TO_DEATH
# INT_SEQ_TO_CONTACT
# SourceStudy
# SEX
# RACE
# ETHNICITY
# TUMOR_PLOIDY
# CENTER
# 
# ONCOTREE_CODE
# SAMPLE_TYPE_DETAILED

## Review harmonized output
#sRes <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM harmonizedSampleInfo')
#print(dim(sRes))

# summarize input patient and sample information
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

################################################
### short variants - load SNV and indel data ###
################################################

### pancan
inFile <- paste0(baseDir,"/../../data/ICGC_TCGA_WGS_2020/pancan_pcawg_2020/data_mutations.txt")
svPancan <- read.csv(inFile,sep="\t",skip = 2)
RSQLite::dbWriteTable(rawDataDb, "PancanPatientVarients", svPancan, overwrite=T)
#inFile <- "../../data/ICGC_TCGA_WGS_2020/pancan_pcawg_2020/data_cna.txt"
#cna.full <- read.csv(inFile,sep="\t")
#
#inFile <- "../../data/ICGC_TCGA_WGS_2020/CNA_Genes.txt"
#cyto <- read.csv(inFile,sep="\t")
#

### MSK-IMPACT
# inFile <- paste0(baseDir,"/../../data/MSK_IMPACT/impact_2017_annotated_per_variant.tsv")
# msk <- read.csv(inFile,sep="\t")
# msk$MAF <- 100*(msk$t_alt_count / (msk$t_ref_count+msk$t_alt_count))
# RSQLite::dbWriteTable(rawDataDb, "Msk2017PatientVarients", msk, overwrite=T)

### MC3 TCGA
#inFile <- paste0(baseDir,"/../../data/mc3_tcga/scratch.sample.mc3.maf")
inFile <- "../../data/mc3_tcga/mc3.v0.2.8.PUBLIC.maf"
mc3 <- read.csv(inFile,sep="\t") %>%
  dplyr::rename(StrandPlusMinus=STRAND,
                Tumor_Sample_Barcode_Full=Tumor_Sample_Barcode)
mc3$Tumor_Sample_Barcode <- substr(mc3$Tumor_Sample_Barcode_Full,1,15)

RSQLite::dbWriteTable(rawDataDb, "Mc3PatientVarients", mc3, overwrite=T)

### add label for assay used - what does each panel capture?

### AACR GENIE
inFile <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_14p1_public/data_mutations_extended.txt")
genie <- read.csv(inFile,sep="\t")
RSQLite::dbWriteTable(rawDataDb, "GeniePatientVarients", genie, overwrite=T)

### Hartwig
#inFile <- paste0(baseDir,"/../../data/Hartwig/data/output_v4/hartwig_clinically_actionable_pcgr_entries.txt")
inFile <- paste0(baseDir,"/../../data/Hartwig/processed/hartwig_sample_pcgr_maf_oncokb_gene_subset.maf")
hartwig.pcgr <- read.csv(inFile,sep="\t")
RSQLite::dbWriteTable(rawDataDb, "HartwigPatientVarients", hartwig.pcgr, overwrite=T)

# harwigMod <- hartwig.pcgr %>%
#   dplyr::mutate(Hugo_Symbol = SYMBOL,
#                 #Entrez_Gene_Id = ENTREZ_ID,
#                 Center = NA,
#                 #NCBI_Build = NA,
#                 #Chromosome = CHROM,
#                 Start_Position = POS,
#                 End_Position = NA,
#                 Strand = NA,
#                 Variant_Classification = NA,
#                 Variant_Type = NA,
#                 Reference_Allele = REF,
#                 Tumor_Seq_Allele1 = NA,
#                 Tumor_Seq_Allele2 = ALT,
#                 dbSNP_RS = NA,
#                 dbSNP_Val_Status = NA,
#                 Tumor_Sample_Barcode = sample,
#                 Matched_Norm_Sample_Barcode = NA,
#                 Match_Norm_Seq_Allele1 = NA,
#                 Match_Norm_Seq_Allele2 = NA,
#                 Tumor_Validation_Allele1 = NA,
#                 Tumor_Validation_Allele2 = NA,
#                 Match_Norm_Validation_Allele1 = NA,
#                 Match_Norm_Validation_Allele2 = NA,
#                 Verification_Status = NA,
#                 Validation_Status = NA,
#                 Mutation_Status = NA,
#                 Sequencing_Phase = NA,
#                 Sequence_Source = NA,
#                 Validation_Method = NA,
#                 Score = NA,
#                 BAM_File = NA,
#                 Sequencer = NA,
#                 t_ref_count = NA,
#                 t_alt_count = NA,
#                 n_ref_count = NA,
#                 n_alt_count = NA,
#                 HGVSp_Short = HGVSp_short,
#                 Transcript_ID = NA)

###############################################################
### write common variant columns for each dataset to SQLite ###
###############################################################

# indentify intersection of column names across all variant sets
pancanCols <- colnames(svPancan)
genieCols <- colnames(genie)
mc3Cols <- colnames(mc3)
hartwigCols <- colnames(hartwig.pcgr)
#mskCols <- colnames(msk)
#uCols <- base::intersect(base::intersect(pancanCols,genieCols),base::intersect(mc3Cols,mskCols))
#uCols <- base::intersect(base::intersect(pancanCols,genieCols),)
uCols <- base::intersect(base::intersect(pancanCols,genieCols),base::intersect(hartwigCols,mc3Cols))

varTable <- svPancan[,uCols]
varTable$TVAF <- NA
varTable$Protein_position <- as.character(varTable$Protein_position)
varTable$SourceStudy <- "PANCAN-WGS-data"
RSQLite::dbWriteTable(harmonizedDb, "patientObservedVariantTable", varTable, overwrite=T)

# varTableMsk <- msk[,uCols]
# varTableMsk$TVAF <- NA
# varTableMsk$SourceStudy <- "MSK-IMPACT-2017-data"
# RSQLite::dbWriteTable(harmonizedDb, "patientObservedVariantTable", varTableMsk,append=T)

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

# vRes <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM patientObservedVariantTable')
# print(dim(vRes))

##############################
### Disconnect from SQL db ###
##############################

RSQLite::dbDisconnect(harmonizedDb)
RSQLite::dbDisconnect(rawDataDb)

########
outRFile <- paste0(outDir,"/genomicStudyiesDbCurration-",timestamp,".RData")
save.image(file = outRFile)
