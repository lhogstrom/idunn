# Matches observed variants to MOA/CIViC clinical annotations (by amino-acid change and genomic
# coordinate), picks one representative annotation per variant, and unions these with OncoKB-annotated
# variants. Writes the resulting tables to the SQLite DB.
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

utilFile <- paste0(baseDir,"/../code/R/snv_indel_annotation.R")
source(utilFile)

outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522")
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)


dbRules <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM MoaCiVICRuleEntries') %>%
  dplyr::rename(oncotree_code_annotation=oncotree_code,
                oncotree_term_annotation=oncotree_term,
                chromosome_annotation=chromosome)
dbRules$balderRuleID <- seq(1,dim(dbRules)[[1]])
dbRules$matchFlag <- "Y"
dbAlterations <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM actionableSNVsByAAChange')
dbGenome <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM actionableSNVsByGenomicCoordinate')
ot_code_full <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM OncoTreeCodeHierarchy')

poVariants<- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM patientObservedVariantTable') %>%
  dplyr::mutate(AAChangeObserved=stringr::str_replace(HGVSp_Short,"p.", ""),
                MAF=100*(t_alt_count / (t_ref_count+t_alt_count)),
                t_alt_plus_ref_count = t_ref_count+t_alt_count)
poVariants$balderVariantID <- seq(1,dim(poVariants)[[1]])
iHartwig <- poVariants$SourceStudy == "Hartwig-data"
poVariants[iHartwig,"MAF"] <- poVariants[iHartwig,"TVAF"]*100

sampleInfoCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM harmonizedSampleInfo') %>%
  dplyr::rename(SUBJECT_CENTER = CENTER)
sampleInfoCompiled[is.na(sampleInfoCompiled$SAMPLE_TYPE),"sampleInfoCompiled"] <- "Unspecified"

sampleInfoCompiled$INT_Seq_or_Biopsy_To_Death <- sampleInfoCompiled$INT_Biopsy_To_Death
iNonNullSeqDeath <- !is.na(sampleInfoCompiled[,"INT_SEQ_TO_DEATH"])
sampleInfoCompiled[iNonNullSeqDeath,"INT_Seq_or_Biopsy_To_Death"] <- sampleInfoCompiled[iNonNullSeqDeath,"INT_SEQ_TO_DEATH"]

svCompiledPrelim <- poVariants %>%
  dplyr::inner_join(sampleInfoCompiled[,!colnames(sampleInfoCompiled)=="SourceStudy"],
                    by=c("Tumor_Sample_Barcode"="SAMPLE_ID")) %>%
  dplyr::filter(!SAMPLE_TYPE == "Cell line") %>%
  dplyr::mutate(AAChangeObserved=stringr::str_replace(HGVSp_Short,"p.", ""),
                MAF=100*(t_alt_count / (t_ref_count+t_alt_count)),
                t_alt_plus_ref_count = t_ref_count+t_alt_count) 

svCompiledPrelim$balderVariantID <- seq(1,dim(svCompiledPrelim)[[1]])
iHartwig <- svCompiledPrelim$SourceStudy == "Hartwig-data"
svCompiledPrelim[iHartwig,"MAF"] <- svCompiledPrelim[iHartwig,"TVAF"]*100

oncokbRes <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM oncoKBAnnotatedVariantTable')
colnames(oncokbRes) <- paste0("ONCOKB_",colnames(oncokbRes))

svCompiled <- combine_patient_table_and_oncokb(svCompiledPrelim,
                                               sampleInfoCompiled,
                                               ot_code_full,
                                               oncokbRes) %>%
               dplyr::mutate(isHemeCancer=grepl("LYMPH",ot_code_level_1) | grepl("MYELOID",ot_code_level_1))
svCompiled$SEQ_ASSAY_ID_mod <- svCompiled$SEQ_ASSAY_ID
svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SEQ_ASSAY_ID_mod"] <- svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SourceStudy"]

RSQLite::dbWriteTable(harmonizedDb, "patientObservedVariantTableWithSampleInfoOncokb", svCompiled,overwrite=T)


aa.genome.exahustive <- annotation_matching_genomic_coordinate_and_AA_change(svCompiled,dbRules)
RSQLite::dbWriteTable(harmonizedDb, "exhaustiveClinicalAnnotatedPatientVariants", aa.genome.exahustive,overwrite=T)

# One representative annotation per variant, preferring restricted cancer-type matches
aa.genome.representative <- aa.genome.exahustive %>%
  dplyr::arrange(desc(cancerTypeMatch),
                 clinical.evidence.summary) %>%
  dplyr::group_by(Tumor_Sample_Barcode,balderVariantID) %>%
  dplyr::filter(dplyr::row_number()==1) %>%
  dplyr::mutate(clinical_annotation_status="clinically annotated")
write.csv(aa.genome.representative,
          paste0(outDir,"/actionability_representative_variant_table.csv"),
          row.names = F)
RSQLite::dbWriteTable(harmonizedDb, "representativeClinicalAnnotatedPatientVariants", aa.genome.representative,overwrite=T)


onckbAnnotated <- svCompiled %>%
  dplyr::filter(ONCOKB_IS_SENS_DX_PX_OR_RESISTANCE == T)

table(onckbAnnotated$balderVariantID %in% aa.genome.representative$balderVariantID)
table(aa.genome.representative$balderVariantID %in% onckbAnnotated$balderVariantID)

iOnckbOnlyGrp <- onckbAnnotated$balderVariantID %in% aa.genome.representative$balderVariantID
okBID <- onckbAnnotated[!iOnckbOnlyGrp,"balderVariantID"]
selectCols <- colnames(svCompiled)[colnames(svCompiled) %in% colnames(aa.genome.representative)]
oncokbOnly <- svCompiled[svCompiled$balderVariantID %in% okBID,selectCols]
iNewCols <- colnames(aa.genome.representative)[!colnames(aa.genome.representative) %in% colnames(oncokbOnly)]
oncokbOnly[,iNewCols] <- NA

aa.genome.representative.oncokb <- rbind(aa.genome.representative,
                                         oncokbOnly)
RSQLite::dbWriteTable(harmonizedDb, "representativeClinicalAnnotatedPatientVariantsWithOncoKB", aa.genome.representative.oncokb, overwrite=T)

RSQLite::dbDisconnect(harmonizedDb)
