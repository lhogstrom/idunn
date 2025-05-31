library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(ggplot2)
library(reshape2)
library(RColorBrewer)

outDir <- "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/demographic_technical_ml_modeling/"
fileShaps = paste0(outDir,"/shap_values_by_ot_code.csv")
shapDf <- read.csv(fileShaps)

demographic_list = c(
  'SAMPLE_TYPE',
  'SAMPLE_TYPE_DETAILED',
  'STAGE',
  'SEX',
  'RACE',
  'ETHNICITY',
  'AGE_AT_SEQ_REPORT',
  'TUMOR_PURITY',
  'TUMOR_PLOIDY'
)
###excluded columns:
#'PATIENT_ID',
#'CANCER_TYPE',
#'CANCER_TYPE_DETAILED',
#'ONCOTREE_CODE',
#'INT_Biopsy_To_Death',
#'DEAD',
#'INT_SEQ_TO_DEATH',
#'INT_SEQ_TO_CONTACT',    
#'INT_Seq_or_Biopsy_To_Death',

techinical_list = c(
  'SUBJECT_CENTER',
  'SEQ_ASSAY_ID',
  'is_paired_end',
  'library_selection',
  'library_strategy',
  'platform',
  'read_length',
  'target_capture_kit',
  'instrument_model',
  'gene_padding',
  'number_of_genes',
  'variant_classifications',
  'CENTER',
  'alteration_types',
  'preservation_technique',
  'specimen_tumor_cellularity',
  'calling_strategy',
  'coverage',
  'SEQ_PIPELINE_ID'
)

feature_set = c(demographic_list, techinical_list)


# for each cancer type find the mean shap values across each feature type
shapFtMean <- data.frame()
i <- 1
for (cType in unique(shapDf$ONCOTREE_CODE)) {
  print(cType)
  iCType <- shapDf$ONCOTREE_CODE == cType
  ctypeDf <- shapDf[iCType,]
  for (f in feature_set) {
    print(f)
    iFMatch <- grepl(f,colnames(ctypeDf))
    fMatchCols <- colnames(ctypeDf)[iFMatch]
    print(fMatchCols)
    
    # subset shap values and get mean absolute value
    subsetDf <- data.frame(ctypeDf[,fMatchCols])
    absSubsetDf <- abs(subsetDf)
    if (dim(subsetDf)[2] > 1) {
      shapRowSum <- data.frame(base::rowSums(absSubsetDf,na.rm=TRUE))
    } else {
      shapRowSum <- data.frame(absSubsetDf)
    }
    ftMeanShapVal <- mean(shapRowSum[,1])
    shapFtMean[i,"ONCOTREE_CODE"] <- cType
    shapFtMean[i,"feature"] <- f
    shapFtMean[i,"meanShapVal"] <- ftMeanShapVal
    i <- i+1
  }
}

#### create rank by shap value
shapFtRank <- shapFtMean %>% 
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::arrange(desc(meanShapVal)) %>%
  dplyr::mutate(rank=row_number(),
                isTechnicalFeature=feature %in% techinical_list,
                isDemographicFeature=feature %in% demographic_list)
shapFtRank$feature_type <- "technical feature"
shapFtRank[shapFtRank$isDemographicFeature==T,"feature_type"] <- "demographic feature"
outF <- paste0(outDir,"/mean_shap_values_by_feature_rank.txt")
write.table(shapFtRank,outF,row.names=F,quote=F,sep="\t")  

### make rank plot
topSelect <- 8
selectRank <- shapFtRank[shapFtRank$rank <= topSelect,]
topFts <- selectRank %>%
  dplyr::group_by(feature) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::arrange(desc(n))
nTopFeatures <- 6
shapRankTopFt <- shapFtRank[shapFtRank$feature %in% topFts$feature[1:nTopFeatures],] %>%
  dplyr::filter(rank <= 10)

outF <-  paste0(outDir,"/shap_feature_rank_plot.pdf")
ggplot(shapRankTopFt, aes(x = ONCOTREE_CODE, y = rank,color=feature,shape=feature)) +
  geom_line(aes(group = feature), linewidth = 1, alpha=.4) +
  geom_point(size=3) +
  scale_y_reverse() +
  labs(title = "SHAP feature rank",
       x = "OT Code",
       y = "feature rank") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  #theme(axis.text.x = element_blank())+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8, width = 8)

## which technical factors have the biggest impact? 
technicalShap <- shapFtRank %>%
  dplyr::filter(feature %in% techinical_list)

technicalRank <- technicalShap %>%
  dplyr::group_by(feature) %>%
  dplyr::summarise(meanRank=mean(rank)) %>%
  dplyr::arrange(meanRank)

technicalShap$feature <- factor(technicalShap$feature,levels=technicalRank$feature)

outF <-  paste0(outDir,"/technical_feature_shap_rank.pdf")
ggplot(technicalShap, aes(x = feature, y = rank)) +
  geom_boxplot() +
  geom_jitter(width = .2,alpha=.5) +
  scale_y_reverse() +
  labs(title = "Technical features: SHAP rank across cancer types",
       x = "feature",
       y = "feature rank") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  #theme(axis.text.x = element_blank())+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8, width = 8)

### plot technical and demographic ranks - facted
orderedFtRank <- shapFtRank %>%
  dplyr::group_by(feature) %>%
  dplyr::summarise(meanRank=mean(rank)) %>%
  dplyr::arrange(meanRank)

shapFtRank$feature <- factor(shapFtRank$feature,levels=orderedFtRank$feature)

outF <-  paste0(outDir,"/technical_demographic_feature_shap_rank.pdf")
ggplot(shapFtRank, aes(x = feature, y = rank, fill=feature_type)) +
  geom_boxplot() +
  geom_jitter(width = .2,alpha=.5) +
  scale_y_reverse() +
  facet_grid(.~feature_type,scale="free",space = "free") +
  labs(title = "Technical features: SHAP rank across cancer types",
       x = "feature",
       y = "feature rank") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  #theme(axis.text.x = element_blank())+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8, width = 10)