library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(ggplot2)
library(reshape2)
library(RColorBrewer)

args <- commandArgs(trailingOnly = TRUE)

# Check if the correct number of arguments are provided
if (length(args) != 2) {
  stop("Two arguments must be supplied", call. = FALSE)
}

# Assign Arguments to Variables
baseDir <- args[1] # should be the relative path to "balder/code" where this file is located
dbName <- args[2]

utilFile1 <- paste0(baseDir,"/../code/R/snv_indel_annotation.R")
source(utilFile1)
utilFile2 <- paste0(baseDir,"/../code/R/utils.R")
source(utilFile1)

### connect to result DB and get variants
outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522/genomic_panel_coverage")
geneOutDir <- paste0(outDir,"/gene_panel_coverage_plots")
if (!dir.exists(geneOutDir)) {
  dir.create(geneOutDir, recursive = TRUE)
}

harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

#seqOtCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM seqOtCTypeSummary')
otCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM otCTypeSummary')
perPatientOncoKBLevelSorted <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM perPatientOncoKBLevelSorted')
assayInfo <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM assayInfoGENIE') 

svCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM patientObservedVariantTableWithSampleInfoOncokb') %>%
  dplyr::mutate(isHemeCancer=grepl("LYMPH",ot_code_level_1) | grepl("MYELOID",ot_code_level_1))
svCompiled$SEQ_ASSAY_ID_mod <- svCompiled$SEQ_ASSAY_ID
svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SEQ_ASSAY_ID_mod"] <- svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SourceStudy"]

##################
### assay info ###
##################

inF <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_15p0_public/genomic_information.txt")
genomicInfo <- read.csv(inF,sep="\t")

## SEQ_ASSAY_ID and SEQ_PIPELINE_ID
#seq_grp <- assayInfo %>%
#  dplyr::group_by(SEQ_ASSAY_ID,SEQ_PIPELINE_ID) %>%
#  dplyr::summarise(number_of_panels=n())
#outF <- paste0(outDir,"/institution_seq_platoform_IDs.csv")
#write.table(seq_grp,outF,row.names=F,quote=F,sep=",")
#
#table(svCompiled$SEQ_ASSAY_ID %in% genomicInfo$SEQ_ASSAY_ID,svCompiled$SourceStudy)
#
##
#length(unique(genomicInfo$Hugo_Symbol))
#
## rank genes according to which genes were measured most frequently
#geneRank <- genomicInfo %>%
#  dplyr::group_by(Hugo_Symbol) %>%
#  dplyr::summarise(n_assays=dplyr::n_distinct(SEQ_ASSAY_ID)) %>%
#  dplyr::arrange(desc(n_assays))
#
#geneRank$Hugo_Symbol <- factor(geneRank$Hugo_Symbol,levels=geneRank$Hugo_Symbol)
#outF <-  paste0(outDir,"/gene_counts_by_seq_assay_ID_top.pdf")
#ggplot(geneRank[1:60,], aes(x = Hugo_Symbol, y = n_assays)) +
#  geom_bar(stat = "identity") +
#  labs(title = "Count of assays measuring gene of interest (n=113 total)",
#       x = "gene",
#       y = "number of assays") +
#  theme_minimal()+
#  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#  theme(plot.title = element_text(hjust = 0.5))
#ggsave(outF,height = 7, width = 10)
#
#geneRank$Hugo_Symbol <- factor(geneRank$Hugo_Symbol,levels=geneRank$Hugo_Symbol)
#outF <-  paste0(outDir,"/gene_counts_by_seq_assay_ID_all.pdf")
#ggplot(geneRank, aes(x = Hugo_Symbol, y = n_assays)) +
#  geom_bar(stat = "identity") +
#  labs(title = "Count of assays measuring gene of interest (n=113 total)",
#       x = "gene",
#       y = "number of assays") +
#  theme_minimal()+
#  #theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#  theme(axis.text.x = element_blank())+
#  theme(plot.title = element_text(hjust = 0.5))
#ggsave(outF,height = 7, width = 10)
#
### what proportion actionable mutation appear in top ranking genes?
##iLevel1 <- grepl("1",oncokbOnly$ONCOKB_HIGHEST_LEVEL_SUMMARY)
##lvl1_compiled <- oncokbOnly[iLevel1,]
#
#iLevel1 <- svCompiled$ONCOKB_HIGHEST_LEVEL == "LEVEL_1"
#lvl1_compiled <- svCompiled[iLevel1,]
#
#lvl1_gene_cnts <- lvl1_compiled %>%
#  dplyr::group_by(Hugo_Symbol) %>%
#  dplyr::summarise(n_patients=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
#  dplyr::arrange(desc(n_patients))
#
### all assays together (heme+solid+other)
#rank_lvl1 <- geneRank %>%
#  dplyr::left_join(lvl1_gene_cnts,by="Hugo_Symbol")
#rank_lvl1$n_patients[is.na(rank_lvl1$n_patients)] <- 0
#rank_lvl1$n_patients_cumsum <- cumsum(rank_lvl1$n_patients)
#rank_lvl1$cumsum_perc_patients <- 100 * (rank_lvl1$n_patients_cumsum / length(unique(svCompiled$Tumor_Sample_Barcode)))
#rank_lvl1$gene_rank <- rownames(rank_lvl1)
#
#rank_lvl1$Hugo_Symbol <- factor(rank_lvl1$Hugo_Symbol,levels=geneRank$Hugo_Symbol)
#outF <-  paste0(outDir,"/cumulative_sum_patients_with_level1_gene_mutation.pdf")
#ggplot(rank_lvl1, aes(x = Hugo_Symbol, y = cumsum_perc_patients)) +
#  geom_point()+
#  #geom_bar(stat = "identity") +
#  labs(title = "Cumulative sum of patients with a level 1 mutaiton \n genes ranked by most commonly measured in panels",
#       x = "gene rank",
#       y = "percent of patients") +
#  theme_minimal()+
#  #theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#  theme(axis.text.x = element_blank())+
#  theme(plot.title = element_text(hjust = 0.5))
#ggsave(outF,height = 7, width = 10)
#
#rLvl1_long <- rank_lvl1[,c("Hugo_Symbol","n_assays","cumsum_perc_patients")] %>%
#  tidyr::pivot_longer(!Hugo_Symbol, names_to = "type", values_to = "value")
#
#rLvl1_long$Hugo_Symbol <- factor(rLvl1_long$Hugo_Symbol,levels=geneRank$Hugo_Symbol)
#outF <-  paste0(outDir,"/cumulative_sum_patients_with_level1_gene_mutation_and_prop.pdf")
#ggplot(rLvl1_long, aes(x = Hugo_Symbol, y = value)) +
#  geom_point()+
#  facet_grid(type ~ ., scales="free_y")+
#  labs(title = "Cumulative sum of patients with a level 1 mutaiton \n genes ranked by most commonly measured in panels",
#       x = "gene rank",
#       y = "percent of patients") +
#  theme_minimal()+
#  #theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#  theme(axis.text.x = element_blank())+
#  theme(plot.title = element_text(hjust = 0.5))
#ggsave(outF,height = 7, width = 10)
#
#outF <-  paste0(outDir,"/cumulative_sum_patients_with_level1_gene_mutation_and_prop_top.pdf")
#ggplot(rLvl1_long[1:180,], aes(x = Hugo_Symbol, y = value)) +
#  geom_point()+
#  facet_grid(type ~ ., scales="free_y")+
#  labs(title = "Cumulative sum of patients with a level 1 mutaiton \n genes ranked by most commonly measured in panels",
#       x = "gene rank",
#       y = "percent of patients") +
#  theme_minimal()+
#  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#  theme(plot.title = element_text(hjust = 0.5))
#ggsave(outF,height = 7, width = 12)
#
## what are examples of genes with high frequency, but low panel rank?
#iHighRank <- as.numeric(rank_lvl1$gene_rank) >= 30
#iHighPatient <- rank_lvl1$n_patients >= 300
#unexpected_hits <- rank_lvl1[iHighRank & iHighPatient,]
#
## print info on unexpected hits
#for (x in unexpected_hits$Hugo_Symbol) {
#  print(x)
#  iGeneMatch = lvl1_compiled$Hugo_Symbol==x
#  print("actionability type:")
#  print(table(lvl1_compiled[iGeneMatch,"ONCOKB_HIGHEST_LEVEL_SUMMARY"]))
#  print("disease level counts:")
#  print(table(lvl1_compiled[iGeneMatch,"level_1_disease"]))
#  print(table(lvl1_compiled[iGeneMatch,"level_3_disease"]))
#  print("study counts:")
#  print(table(lvl1_compiled[iGeneMatch,"SourceStudy"]))
#  print("panel types:")
#  print(table(lvl1_compiled[iGeneMatch,"SEQ_ASSAY_ID_mod"]))
#  print("--------")
#}
#
#### repeat plots above but split heme from solid tumor cancers
#geneRankHeme <- genomicInfo %>%
#  dplyr::left_join(assayInfo[,c("SEQ_ASSAY_ID","isHemePanel")]) %>%
#  dplyr::group_by(Hugo_Symbol,isHemePanel) %>%
#  dplyr::summarise(n_assays=dplyr::n_distinct(SEQ_ASSAY_ID)) %>%
#  dplyr::arrange(desc(n_assays))
#
#lvl1_gene_cnts_heme <- lvl1_compiled %>%
#  dplyr::group_by(Hugo_Symbol,isHemeCancer) %>%
#  dplyr::summarise(n_patients=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
#  dplyr::arrange(desc(n_patients))
#
#lvl1_hemRnak <-lvl1_gene_cnts_heme %>%
#  tidyr::pivot_wider(names_from = isHemeCancer,values_from=n_patients) %>%
#  dplyr::mutate(across(everything(), ~ if_else(is.na(.), 0, .))) %>%
#  dplyr::rename(notHeme=`FALSE`,
#                heme=`TRUE`) %>%
#  dplyr::mutate(hemeDiff = notHeme - heme) %>%
#  dplyr::arrange(hemeDiff)
#
#lvl1_gene_cnts_heme$Hugo_Symbol <- factor(lvl1_gene_cnts_heme$Hugo_Symbol, levels=lvl1_hemRnak$Hugo_Symbol)
#lvl1_gene_cnts_heme$heme_status <- "not heme cancer"
#lvl1_gene_cnts_heme[lvl1_gene_cnts_heme$isHemeCancer==T,"heme_status"] <- "heme cancer"
#
#### level 1 hits by gene
#outF <-  paste0(outDir,"/level1_hits_by_gene_and_heme_status.pdf")
#ggplot(lvl1_gene_cnts_heme, aes(x = Hugo_Symbol, y = n_patients)) +
#  geom_point()+
#  facet_grid(heme_status ~ .)+
#  #geom_bar(stat = "identity") +
#  labs(title = "Level 1 actionable mutations appearing most frequently in heme vs. other cancers",
#       x = "gene",
#       y = "number of patients") +
#  theme_minimal()+
#  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#  #theme(axis.text.x = element_blank())+
#  theme(plot.title = element_text(hjust = 0.5))
#ggsave(outF,height = 7, width = 10)
#
#
###
#rank_lvl1_heme <- geneRankHeme %>%
#  dplyr::left_join(lvl1_hemRnak,by="Hugo_Symbol") %>%
#  tidyr::pivot_longer(c("n_assays","notHeme","heme"), names_to = "variable", values_to = "count")
#rank_lvl1_heme$Hugo_Symbol <- factor(rank_lvl1_heme$Hugo_Symbol, levels=lvl1_hemRnak$Hugo_Symbol)
#
## outF <-  paste0(outDir,"/level1_hits_by_gene_and_heme_status_with_panel_info.pdf")
## ggplot(rank_lvl1_heme, aes(x = Hugo_Symbol, y = count)) +
##   geom_point()+
##   facet_grid(variable ~ isHemePanel,scale="free_y")+
##   #geom_bar(stat = "identity") +
##   labs(title = "Level 1 actionable mutations appearing most frequently in heme vs. other cancers",
##        x = "gene",
##        y = "number of patients") +
##   theme_minimal()+
##   theme(axis.text.x = element_text(angle = 90, hjust = 1))+
##   #theme(axis.text.x = element_blank())+
##   theme(plot.title = element_text(hjust = 0.5))
## ggsave(outF,height = 7, width = 10)
#
#
##rank_lvl1$n_patients_cumsum <- cumsum(rank_lvl1$n_patients)
##rank_lvl1$cumsum_perc_patients <- 100 * (rank_lvl1$n_patients_cumsum / length(unique(svCompiled$Tumor_Sample_Barcode)))
##rank_lvl1$gene_rank <- rownames(rank_lvl1)
#
#
#### pairwise intersection of genes among panels
#runPairwiseCalc <- 0
#if (runPairwiseCalc==1){
#  pairwiseIntersection <- data.frame()
#  panels <- unique(genomicInfo$SEQ_ASSAY_ID)
#  i <- 1
#  for (ix in seq(length(panels))) {
#    x <- panels[ix]
#    xg <- genomicInfo[genomicInfo$SEQ_ASSAY_ID==x,]
#    xgenes <- unique(xg$Hugo_Symbol)
#    for (iy in seq(length(panels))) {
#      y <- panels[iy]
#      yg <- genomicInfo[genomicInfo$SEQ_ASSAY_ID==y,]
#      ygenes <- unique(yg$Hugo_Symbol)
#      if (ix > iy) {
#        print("")
#      }
#      else {
#        print(ix)
#        print(iy)
#        print("----")
#        pairwiseIntersection[i,"panel_1"] <- x
#        pairwiseIntersection[i,"panel_2"] <- y
#        pairwiseIntersection[i,"panel_1_n_genes"] <- length(xgenes)
#        pairwiseIntersection[i,"panel_2_n_genes"] <- length(ygenes)
#        pairwiseIntersection[i,"panel_1_2_intersection"] <- length(intersect(xgenes,ygenes))
#      }
#      i <- i+1
#    }
#  }
#  pairwiseIntersection$intersection_as_prop_of_panel1 <- pairwiseIntersection$panel_1_2_intersection / pairwiseIntersection$panel_1_n_genes
#}

####################################
### genomic intersection plots ####
####################################
#
##geneList <- c("KRAS","TP53","ABL1","EGFR")
##geneList <- c("FBXW7","PPP2R1A","CCNE1")
#geneList <- geneRank$Hugo_Symbol[1:16]
#
#posPanCntFull <- data.frame()
#for (gene in geneList) {
#  print(gene)
#  # hit info
#  lvl1_comp_gene <- lvl1_compiled[lvl1_compiled$Hugo_Symbol==gene,]
#  nPosTotal <- dim(lvl1_comp_gene)[1]
#  lvl1PosCnts <- lvl1_comp_gene  %>%
#    dplyr::group_by(Start_Position) %>%
#    dplyr::summarise(n=n(),
#                     geneLevelFreq=n/nPosTotal) %>%
#    dplyr::arrange(desc(n)) %>%
#    dplyr::rename(coordinate=Start_Position,
#                  size=n) %>%
#    dplyr::mutate(pos="actionable_count",
#                  SEQ_ASSAY_ID="1_observed_events",
#                  type="observed")
#  
#  # genomic info
#  geneSubset <- genomicInfo[genomicInfo$Hugo_Symbol==gene,]
#  geneSubset$region_ID <- seq(1,dim(geneSubset)[1])
#  geneSubset$assay_ID <- factor(geneSubset$SEQ_ASSAY_ID,labels=seq(1,length(unique(geneSubset$SEQ_ASSAY_ID))))
#  geneSubset$assay_ID <- as.numeric(geneSubset$assay_ID)
#  geneSubset$region_length <- geneSubset$End_Position - geneSubset$Start_Position
#  geneSubset$type <- "panel"
#  geneSubset$size <- 1
#  
#  intervalPlotDf <- geneSubset[,c("Start_Position","End_Position","SEQ_ASSAY_ID","size","type")] %>%
#    tidyr::pivot_longer(c("Start_Position","End_Position"), names_to = "pos", values_to = "coordinate")
#  
#  densityPlotDf <- rbind(intervalPlotDf,lvl1PosCnts[,colnames(intervalPlotDf)])
#  
#  outF <-  paste0(geneOutDir,"/genomic_positions_of_panels_for_gene_",gene,".pdf")
#  ggplot(intervalPlotDf, aes(x = coordinate, y = SEQ_ASSAY_ID, shape = pos)) +
#    geom_point()+
#    #facet_grid(type ~ ., scales="free_y")+
#    labs(title = paste0(gene, "\nGenomic coordiantes tested by panel"),
#         x = "genomic coordinate",
#         y = "panel") +
#    theme_minimal()+
#    theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#    theme(plot.title = element_text(hjust = 0.5))
#  ggsave(outF,height = 18, width = 12)
#  
#  if (dim(lvl1PosCnts)[1] > 0) {
#    outF <-  paste0(geneOutDir,"/genomic_positions_of_panels_for_gene_",gene,"_with_counts.pdf")
#    ggplot(densityPlotDf, aes(x = coordinate, y = SEQ_ASSAY_ID, shape = pos, size=size,color=type)) +
#      geom_point(alpha=.2,size=1,color="black")+
#      geom_point(alpha=.6)+
#      #facet_grid(type ~ ., scales="free_y")+
#      labs(title = paste0(gene, "\nGenomic coordiantes tested by panel"),
#           x = "genomic coordinate",
#           y = "panel") +
#      theme_minimal()+
#      theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#      theme(plot.title = element_text(hjust = 0.5))
#    ggsave(outF,height = 18, width = 12)
#    
#    # create position-level evaluations
#    lvl1PosCnts$temp_key <- 1
#    geneSubset$temp_key <- 1
#    
#    # Use merge to create the Cartesian product
#    posTargetMerged <- merge(lvl1PosCnts[,!colnames(lvl1PosCnts) %in% c("SEQ_ASSAY_ID")],
#                             geneSubset[,!colnames(geneSubset) %in% c("type","size","")], by = "temp_key")
#    
#    iGrtr <- posTargetMerged$coordinate >= posTargetMerged$Start_Position
#    iLst <- posTargetMerged$coordinate <= posTargetMerged$End_Position
#    posTargetMerged$posInTarget <- iGrtr & iLst
#    
#    # for each position how many panels does it appear in?
#    nPanelsMeasuringGene <- length(unique(posTargetMerged$SEQ_ASSAY_ID))
#    nPanelsAll <- length(unique(genomicInfo$SEQ_ASSAY_ID))
#    posPanCnt <- posTargetMerged %>%
#      dplyr::filter(posInTarget) %>%
#      dplyr::group_by(coordinate) %>%
#      dplyr::summarise(n=dplyr::n_distinct(SEQ_ASSAY_ID),
#                       geneLevelFreq=mean(geneLevelFreq),
#                       percPanelsMeasuringGene=n/nPanelsMeasuringGene,
#                       percPanlesAll=n/nPanelsAll,
#                       Feature_Type=paste0(unique(Feature_Type),collapse=";")) %>%
#      dplyr::arrange(desc(n))
#    
#    posPanCnt$gene <- gene
#    posPanCntFull <- rbind(posPanCntFull,posPanCnt)
#    
#    outF <-  paste0(geneOutDir,"/genomic_positions_of_panels_for_gene_",gene,"_freq_by_panel_coverage.pdf")
#    ggplot(posPanCnt, aes(x = geneLevelFreq, y = percPanelsMeasuringGene, color=Feature_Type)) +
#      geom_point(alpha=.6)+
#      labs(title = paste0(gene, "\nActionable genomic positions tested by panel"),
#           x = "Frequency of actionable event at position",
#           y = "Fraction of panels measuring position") +
#      ylim(0,1)+
#      theme_minimal()+
#      theme(plot.title = element_text(hjust = 0.5))
#    ggsave(outF,height = 7, width = 7)
#    
#    outF <-  paste0(geneOutDir,"/genomic_positions_of_panels_for_gene_",gene,"_freq_by_panel_coverage_log.pdf")
#    ggplot(posPanCnt, aes(x = geneLevelFreq, y = percPanelsMeasuringGene, color=Feature_Type)) +
#      geom_point(alpha=.6)+
#      scale_x_log10(
#        breaks = scales::trans_breaks("log10", function(x) 10^x),
#        labels = scales::trans_format("log10", scales::math_format(10^.x))
#      )+
#      labs(title = paste0(gene, "\nActionable genomic positions tested by panel"),
#           x = "Frequency of actionable event at position",
#           y = "Fraction of panels measuring position") +
#      ylim(0,1)+
#      theme_minimal()+
#      theme(plot.title = element_text(hjust = 0.5))
#    ggsave(outF,height = 7, width = 7)
#    
#  }
#}
#
#outF <-  paste0(outDir,"/genomic_positions_of_panels_for_gene_1_freq_by_panel_coverage.pdf")
#ggplot(posPanCntFull, aes(x = geneLevelFreq, y = percPanelsMeasuringGene, color=gene)) +
#  geom_point(alpha=.6)+
#  labs(title = paste0("Actionable genomic positions tested by panel"),
#       x = "Frequency of actionable event at position",
#       y = "Fraction of panels measuring position") +
#  ylim(0,1)+
#  theme_minimal()+
#  theme(plot.title = element_text(hjust = 0.5))
#ggsave(outF,height = 7, width = 7)
#
#outF <-  paste0(outDir,"/genomic_positions_of_panels_for_gene_1_freq_by_panel_coverage_log.pdf")
#ggplot(posPanCntFull, aes(x = geneLevelFreq, y = percPanelsMeasuringGene, color=gene)) +
#  geom_point(alpha=.6)+
#  labs(title = paste0("Actionable genomic positions tested by panel"),
#       x = "Frequency of actionable event at position",
#       y = "Fraction of panels measuring position") +
#  scale_x_log10(
#    breaks = scales::trans_breaks("log10", function(x) 10^x),
#    labels = scales::trans_format("log10", scales::math_format(10^.x))
#  )+
#  ylim(0,1)+
#  theme_minimal()+
#  theme(plot.title = element_text(hjust = 0.5))
#ggsave(outF,height = 7, width = 7)
#

##############################################
### panel proportions by genomic position ####
##############################################

#panelGPCovTbl <- perPatientOncoKBLevelSorted %>%
#  dplyr::filter(isOncoKbLevel1==1) %>%
#  dplyr::group_by(ONCOTREE_CODE,Hugo_Symbol,Start_Position,SEQ_ASSAY_ID) %>%
#  dplyr::summarise(n=()) %>%
#  dplyr::mutate(positionID=paste0(Hugo_Symbol,"-",Start_Position)) %>%
#  dplyr::mutate(positionCtypeID=paste0(ONCOTREE_CODE,"-",Hugo_Symbol,"-",Start_Position)) %>%
#  dplyr::arrange(desc(n))

panelGPCovTbl <- svCompiled %>%
  dplyr::filter(ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_1") %>%
  dplyr::group_by(ONCOTREE_CODE,Hugo_Symbol,Start_Position,SEQ_ASSAY_ID) %>%
  dplyr::summarise(n=dplyr::n_distinct(PATIENT_ID)) %>%
  dplyr::mutate(positionID=paste0(Hugo_Symbol,"-",Start_Position)) %>%
  dplyr::mutate(positionCtypeID=paste0(ONCOTREE_CODE,"-",Hugo_Symbol,"-",Start_Position)) %>%
  dplyr::arrange(desc(n))

perPatientPanelCount <- perPatientOncoKBLevelSorted %>%
  dplyr::group_by(ONCOTREE_CODE,SEQ_ASSAY_ID) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::arrange(desc(n))

cTypeCnts <- perPatientOncoKBLevelSorted %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(nCtype=n()) %>%
  dplyr::arrange(desc(nCtype))

genomicPositionLookup <- data.frame()
for (pID in unique(panelGPCovTbl$positionCtypeID)) {
  pDf <- panelGPCovTbl[panelGPCovTbl$positionCtypeID==pID,]
  cType <- unique(pDf$ONCOTREE_CODE)
  gene <- unique(pDf$Hugo_Symbol)
  pos <- unique(pDf$Start_Position)
  uSeqIDs <- unique(pDf$SEQ_ASSAY_ID)
  
  ### get panel info ###
  geneSubset <- genomicInfo[genomicInfo$Hugo_Symbol==gene,]
  geneSubset$region_ID <- seq(1,dim(geneSubset)[1])
  geneSubset$assay_ID <- factor(geneSubset$SEQ_ASSAY_ID,labels=seq(1,length(unique(geneSubset$SEQ_ASSAY_ID))))
  geneSubset$assay_ID <- as.numeric(geneSubset$assay_ID)
  geneSubset$region_length <- geneSubset$End_Position - geneSubset$Start_Position
  geneSubset$coordinate <- pos

  iGrtr <- geneSubset$coordinate >= geneSubset$Start_Position
  iLst <- geneSubset$coordinate <= geneSubset$End_Position
  geneSubset$posInTarget <- iGrtr & iLst
  
  # which panels tested for the position of interest? 
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
  #dplyr::arrange(desc(nCtype)) 
  dplyr::arrange(desc(patients_tested_in_panel_with_coverage))

# reporting stats
topPostions <- genomicPositionLookup[genomicPositionLookup$perc_of_patients_with_coverage > 90.0,]
genomicPositionLookup$highCoveragePosition <- genomicPositionLookup$positionCtypeID %in% topPostions$positionCtypeID


iHighCoverage <- genomicPositionLookup$highCoveragePosition == T
print(sum(genomicPositionLookup[iHighCoverage,"n_patients_mutated_at_position"]))
print(sum(genomicPositionLookup[!iHighCoverage,"n_patients_mutated_at_position"]))

#################
### Plot data ###
#################

### cumulative sum of patients with coverage
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
  #theme(axis.text.x = element_blank())+
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
  #theme(axis.text.x = element_blank())+
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

