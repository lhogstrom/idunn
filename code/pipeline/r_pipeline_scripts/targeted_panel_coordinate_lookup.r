# Builds the panel-design coverage lookup. For each cancer type x genomic position, computes the
# percent of patients whose sequencing panel covered the position, plus per-sample median coverage
# (written to the SQLite DB as medianCoveragePerSample and genomicPositionLookup).
# Invoked by workflow/main.nf: Rscript <script> <baseDir> <dbName>

library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(ggplot2)
library(reshape2)
library(RColorBrewer)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Two arguments must be supplied", call. = FALSE)
}

baseDir <- args[1] # should be the relative path to "balder/code" where this file is located
dbName <- args[2]

utilFile1 <- paste0(baseDir,"/../code/R/snv_indel_annotation.R")
source(utilFile1)
utilFile2 <- paste0(baseDir,"/../code/R/utils.R")
source(utilFile1)

outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522/genomic_panel_coverage")
geneOutDir <- paste0(outDir,"/gene_panel_coverage_plots")
if (!dir.exists(geneOutDir)) {
  dir.create(geneOutDir, recursive = TRUE)
}

harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

svCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM patientObservedVariantTableWithSampleInfoOncokb') %>%
  dplyr::mutate(isHemeCancer=grepl("LYMPH",ot_code_level_1) | grepl("MYELOID",ot_code_level_1))
svCompiled$SEQ_ASSAY_ID_mod <- svCompiled$SEQ_ASSAY_ID
svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SEQ_ASSAY_ID_mod"] <- svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SourceStudy"]

inF <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_15p0_public/genomic_information.txt")
genomicInfo <- read.csv(inF,sep="\t")


medianCoveragePerSample <- svCompiled %>%
  dplyr::group_by(Center,Tumor_Sample_Barcode) %>%
  dplyr::summarise(medianCoverage=median(t_alt_plus_ref_count,na.rm = TRUE))

outF <- paste0(outDir,"/median_coverage_per_sample.txt")
write.table(medianCoveragePerSample,outF,row.names=F,quote=F,sep="\t")
RSQLite::dbWriteTable(harmonizedDb, "medianCoveragePerSample", medianCoveragePerSample, overwrite =T)

print(summary(medianCoveragePerSample[medianCoveragePerSample$SourceStudy=="AACR-GENIE-data","medianCoverage"]))


panelGPCovTbl <- svCompiled %>%
  dplyr::filter(SourceStudy=="AACR-GENIE-data") %>%
  dplyr::filter(ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_1") %>%
  dplyr::group_by(ONCOTREE_CODE,Hugo_Symbol,Start_Position,SEQ_ASSAY_ID) %>%
  dplyr::summarise(n=dplyr::n_distinct(PATIENT_ID)) %>%
  dplyr::mutate(positionID=paste0(Hugo_Symbol,"-",Start_Position)) %>%
  dplyr::mutate(positionCtypeID=paste0(ONCOTREE_CODE,"-",Hugo_Symbol,"-",Start_Position)) %>%
  dplyr::arrange(desc(n))

perPatientPanelCount <- svCompiled %>%
  dplyr::group_by(ONCOTREE_CODE,SEQ_ASSAY_ID) %>%
  dplyr::summarise(n=dplyr::n_distinct(PATIENT_ID)) %>%
  dplyr::arrange(desc(n))

cTypeCnts <- svCompiled %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(nCtype=dplyr::n_distinct(PATIENT_ID)) %>%
  dplyr::arrange(desc(nCtype))

genomicPositionLookup <- data.frame()
for (pID in unique(panelGPCovTbl$positionCtypeID)) {
  pDf <- panelGPCovTbl[panelGPCovTbl$positionCtypeID==pID,]
  cType <- unique(pDf$ONCOTREE_CODE)
  gene <- unique(pDf$Hugo_Symbol)
  pos <- unique(pDf$Start_Position)
  uSeqIDs <- unique(pDf$SEQ_ASSAY_ID)
  
  geneSubset <- genomicInfo[genomicInfo$Hugo_Symbol==gene,]
  geneSubset$region_ID <- seq(1,dim(geneSubset)[1])
  geneSubset$assay_ID <- factor(geneSubset$SEQ_ASSAY_ID,labels=seq(1,length(unique(geneSubset$SEQ_ASSAY_ID))))
  geneSubset$assay_ID <- as.numeric(geneSubset$assay_ID)
  geneSubset$region_length <- geneSubset$End_Position - geneSubset$Start_Position
  geneSubset$coordinate <- pos

  iGrtr <- geneSubset$coordinate >= geneSubset$Start_Position
  iLst <- geneSubset$coordinate <= geneSubset$End_Position
  geneSubset$posInTarget <- iGrtr & iLst
  
  panelsWithCoordTested <- unique(geneSubset[geneSubset$posInTarget==T,"SEQ_ASSAY_ID"])
  iIDMatch <- perPatientPanelCount$SEQ_ASSAY_ID %in% panelsWithCoordTested
  iCtypeMatch <- perPatientPanelCount$ONCOTREE_CODE %in% cType
  panelCnts <- perPatientPanelCount[iIDMatch & iCtypeMatch,]
  genomicPositionLookup[pID,"patients_tested_in_panel_with_coverage"] <- sum(panelCnts$n)
  
  genomicPositionLookup[pID,"n_patients_mutated_at_position"] <- sum(pDf$n)
  genomicPositionLookup[pID,"positionCtypeID"] <- pID
  genomicPositionLookup[pID,"ONCOTREE_CODE"] <- cType
}
genomicPositionLookup <- genomicPositionLookup %>%
  dplyr::left_join(cTypeCnts,by="ONCOTREE_CODE") %>%
  dplyr::mutate(perc_of_patients_with_coverage=100*(patients_tested_in_panel_with_coverage/nCtype)) %>%
  dplyr::arrange(desc(patients_tested_in_panel_with_coverage))

outF <- paste0(outDir,"/panel_design_genomic_coverage_lookup.txt")
write.table(genomicPositionLookup,outF,row.names=F,quote=F,sep="\t")
RSQLite::dbWriteTable(harmonizedDb, "genomicPositionLookup", genomicPositionLookup, overwrite =T)
RSQLite::dbDisconnect(harmonizedDb)

topPostions <- genomicPositionLookup[genomicPositionLookup$perc_of_patients_with_coverage > 90.0,]
genomicPositionLookup$highCoveragePosition <- genomicPositionLookup$positionCtypeID %in% topPostions$positionCtypeID


iHighCoverage <- genomicPositionLookup$highCoveragePosition == T
print(sum(genomicPositionLookup[iHighCoverage,"n_patients_mutated_at_position"]))
print(sum(genomicPositionLookup[!iHighCoverage,"n_patients_mutated_at_position"]))


genomicPositionLookup$mutated_cumsum <- cumsum(genomicPositionLookup$patients_tested_in_panel_with_coverage)
genomicPositionLookup$perc_of_total <- 100* genomicPositionLookup$mutated_cumsum / sum(genomicPositionLookup$patients_tested_in_panel_with_coverage)

outF <- paste0(outDir,"/genomic_position_lookup_table.csv")
write.table(genomicPositionLookup,outF,row.names=F,quote=F,sep=",")

genomicPositionLookup$positionCtypeID <- factor(genomicPositionLookup$positionCtypeID, levels=genomicPositionLookup$positionCtypeID)
cMax <- max(genomicPositionLookup$mutated_cumsum)

outF <-  paste0(outDir,"/cumulative_mutation_counts_of_actionable_events_with_genomic_coverage_label.pdf")
ggplot(genomicPositionLookup[1:100,],aes(x=positionCtypeID,y=mutated_cumsum,color= highCoveragePosition))+
  geom_point() + 
  labs(title = "Count of patients with actionable mutations\n by genomic position",
       x = "cancer type & protein posiiton",
       y = "cumulative patients count") +
  ylim(0,cMax+2000) + 
  geom_hline(yintercept = cMax,linetype="dotted")+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6, width = 15)

outF <-  paste0(outDir,"/cumulative_mutation_counts_of_actionable_events_with_genomic_coverage_label_v2.pdf")
ggplot(genomicPositionLookup[1:250,],aes(x=positionCtypeID,y=mutated_cumsum,color= highCoveragePosition))+
  geom_point() + 
  labs(title = "Count of patients with actionable mutations\n by genomic position",
       x = "cancer type & protein posiiton",
       y = "cumulative patients count") +
  ylim(0,cMax+2000) + 
  geom_hline(yintercept = cMax,linetype="dotted")+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6, width = 15)

outF <-  paste0(outDir,"/cumulative_mutation_counts_of_actionable_events_with_genomic_coverage_label_v3.pdf")
ggplot(genomicPositionLookup,aes(x=positionCtypeID,y=mutated_cumsum,color= highCoveragePosition))+
  geom_point() + 
  labs(title = "Count of patients with actionable mutations\n by genomic position",
       x = "cancer type & protein posiiton",
       y = "cumulative patients count") +
  ylim(0,cMax+2000) + 
  geom_hline(yintercept = cMax,linetype="dotted")+
  theme_bw()+
  theme(axis.text.x = element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6, width = 15)

