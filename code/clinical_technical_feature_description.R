library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(ggplot2)
library(reshape2)
library(RColorBrewer)
#library(ComplexHeatmap)

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
#outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20240924/oncokb_summary_and_modeling")
#outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250121/oncokb_summary_and_modeling")
outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522/oncokb_summary_and_modeling")
modelFeatureOutDir <- paste0(outDir,"/underlying_feature_model_plots_v2")
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

#seqOtCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM seqOtCTypeSummary') 
otCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM otCTypeSummary') 
perPatientLevel1 <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM perPatientLevel1') 

##########################
### examine features #####
##########################

# highest_c_counts <- unique(seqOtCTypeSummary$ONCOTREE_CODE)[1:15]
# iHighest <- seqOtCTypeSummary$ONCOTREE_CODE %in% as.character(highest_c_counts)
# iLevel1 <- grepl("LEVEL_1",seqOtCTypeSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY)
# 
# data <- seqOtCTypeSummary[iHighest & iLevel1,] %>%
#   #dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID" )) %>%
#   #dplyr::select(all_of(modelCols)) %>%
#   dplyr::group_by(ONCOTREE_CODE) %>%
#   dplyr::arrange(desc(perc)) %>% 
#   dplyr::mutate(actionable_perc_rank=dplyr::row_number())

highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:63] # select those OT codes with 500 or more samples
highest_c_counts <- highest_c_counts[!is.na(highest_c_counts)]
iHighest <- perPatientLevel1$ONCOTREE_CODE %in% as.character(highest_c_counts)

# clinical_cols <- c("median_age",
#                    "met_to_primary_ratio",
#                    "meanTumorPurity",
#                    "anyTumorPurityInfo",
#                    "perc_LocalRecurrence",
#                    "perc_NaHeme",
#                    "perc_dead",
#                    "perc_StageI",
#                    "perc_StageII",
#                    "perc_StageIII",
#                    "perc_StageIV",
#                    "median_IntSeqToDeath")
# technical_cols <- c("is_paired_end",
#                     "library_selection",
#                     "platform",
#                     "number_of_genes",
#                     "calling_strategy",
#                     "preservation_technique")
# 
# modelCols <- c("ONCOTREE_CODE",
#                "SEQ_ASSAY_ID_mod",
#                "ONCOKB_HIGHEST_LEVEL_SUMMARY",
#                clinical_cols,
#                technical_cols)

# modelCols <- c("SAMPLE_TYPE",
#              "RACE",
#              "ETHNICITY",
#              "is_paired_end",
#              "library_selection",
#              "platform",
#              "calling_strategy",
#              "preservation_technique")

baseCols <- c("isOncoKbLevel1",
              "ONCOTREE_CODE")
selectCols <- c("SAMPLE_TYPE",
               "AGE_AT_SEQ_REPORT",
               "RACE",
               "ETHNICITY",
               "is_paired_end",
               "library_selection",
               "platform",
               "number_of_genes",
               "calling_strategy",
               "preservation_technique")
modelCols <- c(baseCols,selectCols)

data <- perPatientLevel1[iHighest,] %>%
  dplyr::select(all_of(modelCols))
# exclude patients if any data entry is null
iNull <- rowSums(!is.na(data[,modelCols])) != length(modelCols) 
data <- data[!iNull,]


#library(magick)

### exploratory plots by relationship
ctCnts <- data %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(nSubjectOTcode=n()) %>%
  dplyr::arrange(desc(nSubjectOTcode))



for (ftr in selectCols) {
  fCnts <- data  %>%
    dplyr::group_by(.data[[ftr]],ONCOTREE_CODE,isOncoKbLevel1) %>%
    dplyr::summarise(n=n()) %>%
    dplyr::left_join(ctCnts, by="ONCOTREE_CODE") %>%
    dplyr::filter(nSubjectOTcode >= 200) %>%
    dplyr::mutate(perc=100*n/nSubjectOTcode)
  
  fCntsWide <- fCnts[,c(ftr,c("ONCOTREE_CODE","isOncoKbLevel1","n"))] %>%
    tidyr::pivot_wider(names_from = isOncoKbLevel1, values_from=n) 
  colnames(fCntsWide) <- c(ftr,c("ONCOTREE_CODE","isOncoKbLevel1_FALSE","isOncoKbLevel1_TRUE"))
  fCntsWide <- fCntsWide %>%
    dplyr::mutate(isOncoKbLevel1_TRUE = replace_na(isOncoKbLevel1_TRUE, 0),
                  isOncoKbLevel1_FALSE = replace_na(isOncoKbLevel1_FALSE, 0),
                  percent_actionable=100*isOncoKbLevel1_TRUE/(isOncoKbLevel1_TRUE+isOncoKbLevel1_FALSE))
  #fCntsWide$percent_actionable <- 100*fCntsWide$isOncoKbLevel1_TRUE / (fCntsWide$isOncoKbLevel1_TRU)
  # Scaling factor to align PercentDiff with RawCounts
  scale_factor <- max(fCnts$n) / max(fCnts$perc)
  
  ### run chi-squared to see if there are differences among clinical actionability by condition
  resChiSq <- data.frame()
  for (otCode in fCntsWide$ONCOTREE_CODE) {
    tmpF <- fCntsWide[fCntsWide$ONCOTREE_CODE==otCode,]
    cNames <- c("isOncoKbLevel1_TRUE","isOncoKbLevel1_FALSE")
    tmpFTest <- data.frame(tmpF[,cNames])
    rownames(tmpFTest) <- as.character(data.frame(tmpF)[,c(ftr)])
    
    if(all(colSums(tmpFTest) > 0)) {
      Xsq <- chisq.test(tmpFTest)
      Xsq$observed
      resChiSq[otCode,"pValue"] <- Xsq$p.value
      resChiSq[otCode,"xSquared"] <- Xsq$statistic
    }
  }
  
  resChiSq$ONCOTREE_CODE <- rownames(resChiSq)
  resChiSq$pFDR <- p.adjust(resChiSq$pValue, method = "fdr")
  resChiSq <- resChiSq %>%
    dplyr::arrange((pValue))
  
  fCntsWideJ <- fCntsWide %>%
    dplyr::left_join(resChiSq, by="ONCOTREE_CODE") %>%
    dplyr::mutate(strFull="",
                  str1=paste0("p=",round(pValue,4)),
                  str2=paste0("p=",round(pValue,4),
                              " p-adjusted=",round(pFDR,4))) %>%
    dplyr::arrange(pValue)
  iMatch <- fCntsWideJ$pValue <= 0.05 & ! is.na(fCntsWideJ$pValue)
  fCntsWideJ[iMatch,"strFull"] <- fCntsWideJ[iMatch,"str2"]
  fCntsWideJ$ONCOTREE_CODE <- factor(fCntsWideJ$ONCOTREE_CODE, levels=unique(fCntsWideJ$ONCOTREE_CODE))
  
  # plot percent actionable
  outF2 <-  paste0(modelFeatureOutDir,"/","sample_counts_with_Level1_action_",ftr,"_perc_actionable.pdf")
  ggplot(fCntsWideJ, aes(x = .data[[ftr]],y = percent_actionable)) +
    geom_bar(aes(), stat = "identity", alpha = 0.7) +
    facet_grid(ONCOTREE_CODE~.data[[ftr]],scale="free") +
    #scale_y_continuous(
    #  name = "Raw Counts",
    #  sec.axis = sec_axis(~ . / scale_factor, name = "Percent of patients") # Secondary y-axis
    #) +
    geom_text(
      aes(label = strFull),  # Add percent labels
      position = position_stack(vjust = 0.4)  # Position labels in the middle of slices
    ) +
    labs(
      title = " ",
      x = " "
    ) +
    #scale_fill_brewer(palette="Set2",drop=FALSE) + 
    theme_minimal() #+
  ggsave(outF2,height = 49, width = 12)
  
  
  fCnts$ONCOTREE_CODE <- factor(fCnts$ONCOTREE_CODE, levels=unique(fCntsWideJ$ONCOTREE_CODE))
  # Create the plot of patient counts
  outF1 <-  paste0(modelFeatureOutDir,"/","sample_counts_with_Level1_action_",ftr,".pdf")
  ggplot(fCnts, aes(x = isOncoKbLevel1)) +
    geom_bar(aes(y = n,fill=isOncoKbLevel1), stat = "identity", alpha = 0.7) +
    #facet_wrap(~ONCOTREE_CODE,scale="free_y") +
    #facet_grid(SAMPLE_TYPE~ONCOTREE_CODE,scale="free_y") +
    facet_grid(ONCOTREE_CODE~.data[[ftr]],scale="free_y") +
    scale_y_continuous(
      name = "Raw Counts",
      sec.axis = sec_axis(~ . / scale_factor, name = "Percent of patients") # Secondary y-axis
    ) +
    labs(
      title = " ",
      x = " "
    ) +
    #scale_fill_brewer(palette="Set2",drop=FALSE) +
    theme_minimal() #+
  ggsave(outF1,height = 49, width = 12)
  
# 
#   # # # Load the PDF images
#   pdf1 <- magick::image_read(outF1, density = 300)
#   pdf2 <- magick::image_read(outF2, density = 300)
# 
#   # Ensure both images have the same height by resizing
#   pdf1 <- image_scale(pdf1, "x1080")  # Adjust height as needed
#   pdf2 <- image_scale(pdf2, "x1080")
# 
#   # Combine the images side-by-side
#   combined_image <- image_append(c(pdf1, pdf2), stack = FALSE)
# 
#   #image_append(image_scale(c(image_append(banana[c(1,3)], stack = TRUE), wizard)))
# 
#   # Save the combined image as a PDF
#   outF3 <-  paste0(modelFeatureOutDir,"/","sample_counts_with_Level1_action_",ftr,"_perc_actionable_combined.pdf")
#   image_write(combined_image, path = outF3, format = "pdf")
  # 
}

##############################################################################
### plot genes that contribute most to actionability for each cancer type ###
##############################################################################

### perc of gene by cancer type
otCntsAction <- perPatientLevel1 %>%
  dplyr::filter(isOncoKbLevel1==T) %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(nOtSubjLevel1=n()) %>%
  dplyr::arrange(desc(nOtSubjLevel1))

otGeneCntsAction <- perPatientLevel1 %>%
  dplyr::filter(isOncoKbLevel1==T) %>%
  dplyr::group_by(ONCOTREE_CODE,Hugo_Symbol) %>%
  dplyr::summarise(nLevel1Actionable=n()) %>%
  dplyr::left_join(otCntsAction,by="ONCOTREE_CODE") %>%
  dplyr::mutate(perc=100*nLevel1Actionable/nOtSubjLevel1)

# create a table of 25 most common genes and 15 most common OT codes
topOtCodes <- otCntsAction$ONCOTREE_CODE[1:15] 
  
### perc of gene by cancer type
geneMutCnts <- perPatientLevel1 %>%
  dplyr::filter(isOncoKbLevel1==T,
                ONCOTREE_CODE %in% topOtCodes) %>%
  dplyr::group_by(Hugo_Symbol) %>%
  dplyr::summarise(nGeneMut=n()) %>%
  dplyr::arrange(desc(nGeneMut))

topGeneMuts <- geneMutCnts$Hugo_Symbol[1:25] 

otGeneCntsActionSubset <- otGeneCntsAction %>%
  dplyr::filter(Hugo_Symbol %in% topGeneMuts,
                ONCOTREE_CODE %in% topOtCodes)

outF1 <-  paste0(modelFeatureOutDir,"/","perc_of_OT_code_by_gene_level1_oncokb_events.pdf")
ggplot(otGeneCntsActionSubset, aes(x=ONCOTREE_CODE,size=perc,y = Hugo_Symbol)) +
  geom_point(alpha=.6) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF1,height = 8, width = 8)

# for a given OT code, look at the breakdown of commmon genes - where are there differences between centers? 
cntrOtCntsAction <- perPatientLevel1 %>%
  dplyr::filter(isOncoKbLevel1==T) %>%
  dplyr::group_by(ONCOTREE_CODE,CENTER) %>%
  dplyr::summarise(nOtSubjLevel1=n()) %>%
  dplyr::arrange(desc(nOtSubjLevel1))

otCntrCntsFull <- data.frame()
for (otCode in topOtCodes) {
  cntrOtCntsActionSubset <- cntrOtCntsAction[cntrOtCntsAction$ONCOTREE_CODE==otCode,]
  
  otCntrCntsAction <- perPatientLevel1 %>%
    dplyr::filter(isOncoKbLevel1==T,
                  ONCOTREE_CODE==otCode) %>%
    dplyr::group_by(CENTER,Hugo_Symbol) %>%
    dplyr::summarise(nLevel1Actionable=n()) %>%
    dplyr::left_join(cntrOtCntsActionSubset,by="CENTER") %>%
    dplyr::mutate(perc=100*nLevel1Actionable/nOtSubjLevel1)
  otCntrCntsAction$ONCOTREE_CODE <- otCode
  otCntrCntsFull <- rbind(otCntrCntsFull,otCntrCntsAction)
  
  outF1 <-  paste0(modelFeatureOutDir,"/gene_percent_breakdown_Level1_action_",otCode,".pdf")
  ggplot(otCntrCntsAction, aes(x=Hugo_Symbol,size=perc,y = CENTER)) +
    geom_point(alpha=.6) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))+
    theme(plot.title = element_text(hjust = 0.5))
  ggsave(outF1,height = 8, width = 8)
  
  # make matrix for heatma
  library(ComplexHeatmap)
  otCntWide <- otCntrCntsAction[,!colnames(otCntrCntsAction)=="nLevel1Actionable"] %>%
    tidyr::pivot_wider(names_from = Hugo_Symbol,values_from=perc) %>%
    mutate(across(everything(), ~ replace_na(., 0)))
    #dplyr::mutate(isOncoKbLevel1_TRUE = replace_na(isOncoKbLevel1_TRUE, 0),
    
  geneSubset <- unique(otCntrCntsAction$Hugo_Symbol)  
  dfMtrx <- data.frame(otCntWide[,geneSubset])
  rownames(dfMtrx) <- replace_na(otCntWide$CENTER,"Unknownn")
  mmtrx <- as.matrix(dfMtrx)
  
  outF <-  paste0(modelFeatureOutDir,"/gene_percent_breakdown_Level1_action_",otCode,"_heatmap.pdf")
  pdf(outF, width = 8, height = 8)
  cHeatmap <- ComplexHeatmap::Heatmap(dfMtrx, 
                                      name = "percent of Level1 events", 
                                      cluster_columns = TRUE,
                                      cluster_rows = TRUE)
  draw(cHeatmap)
  dev.off()
    
}


### make mtrx of full data
iCols <- !colnames(otCntrCntsAction) %in% c("nLevel1Actionable","nOtSubjLevel1")
otCntWideFull <- otCntrCntsFull[,iCols] %>%
  tidyr::pivot_wider(names_from = Hugo_Symbol,values_from=perc) %>%
  mutate(across(everything(), ~ replace_na(., 0)))

geneSubset <- unique(otCntrCntsFull$Hugo_Symbol)
dfMtrx <- data.frame(otCntWideFull[,geneSubset])
rownames(dfMtrx) <- paste0(otCntWideFull$ONCOTREE_CODE,
                           "-",
                           replace_na(otCntWideFull$CENTER,"Unknownn"))
mmtrx <- as.matrix(dfMtrx)

outF <-  paste0(modelFeatureOutDir,"/gene_percent_breakdown_Level1_action_all_cTypes_heatmap.pdf")
pdf(outF, width = 10, height = 20)
cHeatmap <- ComplexHeatmap::Heatmap(dfMtrx, 
                                    name = "percent of Level1 events", 
                                    cluster_columns = TRUE,
                                    cluster_rows = FALSE)
draw(cHeatmap)
dev.off()

# 
# ht_list = ht1 %v% ht2 %v% ht3
# draw(ht_list)
