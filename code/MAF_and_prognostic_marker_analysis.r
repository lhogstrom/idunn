library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(ggplot2)
library(rlang)
library(ggalluvial)
library(ggrepel)

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
outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522/MAF_prognostic_analysis")
ProgOutDir <- paste0(outDir,"/prognostic_analysis_restricted_cancer_types")
MafOutDir <- paste0(outDir,"/MAF_analysis_restricted_cancer_types")
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

##########################
### load data products ###
##########################

ot_code_full <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM OncoTreeCodeHierarchy')
sampleInfoCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM harmonizedSampleInfo') 

svCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM patientObservedVariantTableWithSampleInfoOncokb') 
aa.genome.representative <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM representativeClinicalAnnotatedPatientVariants') 
aa.genome.exahustive <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM exhaustiveClinicalAnnotatedPatientVariants') 
aa.genome.representative.oncokb <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM representativeClinicalAnnotatedPatientVariantsWithOncoKB') 

non.annotated.variants <- svCompiled %>%
  dplyr::filter(!balderVariantID %in% aa.genome.representative$balderVariantID) %>%
  dplyr::mutate(clinical_annotation_status="not annotated")
iNewCols <- colnames(aa.genome.representative)[!colnames(aa.genome.representative) %in% colnames(non.annotated.variants)]
non.annotated.variants[,iNewCols] <- NA
annotated.and.non.annotated <- rbind(aa.genome.representative,non.annotated.variants) # same as svCompiled, but with annotation columns

##############################################
### mutant allele frequency (MAF) analysis ###
##############################################

iMafOver100 <- (annotated.and.non.annotated$MAF > 100) | is.na(annotated.and.non.annotated$MAF)
outF <-  paste0(outDir,"/MAF_distribution_clinically_actionable_vs_non_actionable.pdf")
ggplot(annotated.and.non.annotated[!iMafOver100,],aes(x=MAF,fill=clinical_annotation_status))+
  #geom_histogram(alpha=.4)+
  geom_density(alpha=.4)+
  theme_bw()+
  facet_grid(SourceStudy~.,scale="free_y",)+
  xlab("MAF (%)")+
  ggtitle(paste0("MAF of unrestricted annotated and non-annotated variants \n in source studies"))+
  scale_fill_brewer(palette="Set1",drop=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 7)

# repeat plot by cancer type
rownames(ot_code_full) <- ot_code_full$ot_code

cType.sample.counts.primary.met <- svCompiled %>%
  dplyr::group_by(SAMPLE_TYPE,ONCOTREE_CODE) %>%
  dplyr::summarise(n.patients=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n.patients)) %>%
  data.frame()

iPrimary <- cType.sample.counts.primary.met$SAMPLE_TYPE=="Primary"
iHighCount <- cType.sample.counts.primary.met$n.patients > 500
TopPrimTissue <- cType.sample.counts.primary.met[iPrimary & iHighCount,"ONCOTREE_CODE"]

for (cType in TopPrimTissue) {
  cTypeName <- ot_code_full[cType,"ot_name"]
  print(cTypeName)
  iCtypeMatch <- (annotated.and.non.annotated$ONCOTREE_CODE == cType) & !is.na(annotated.and.non.annotated$ONCOTREE_CODE)
  outF <-  paste0(outDir,"/cancer_type_MAF_distribution_clinically_actionable_vs_non_actionable_",cType,".pdf")
  ggplot(annotated.and.non.annotated[!iMafOver100 & iCtypeMatch,],aes(x=MAF,fill=clinical_annotation_status))+
    #geom_histogram(alpha=.4)+
    geom_density(alpha=.4)+
    theme_bw()+
    facet_grid(SourceStudy~.,scale="free_y",)+
    xlab("MAF (%)")+
    ggtitle(paste0(cTypeName,"\n MAF of unrestricted annotated and non-annotated variants \n in source studies"))+
    scale_fill_brewer(palette="Set1",drop=FALSE)+
    theme(plot.title = element_text(hjust = 0.5))
  ggsave(outF,height = 8,width = 7)
}

### Examine MAF distributions ###

### select cancer type unrestricted data for evaluation
#data_for_evaluation <- aa.genome.representative

### select cancer type restricted data for evaluation
data_for_evaluation <- aa.genome.representative %>%
  dplyr::filter(cancerTypeMatch==T)

outF <-  paste0(MafOutDir,"/MAF_distribution_clinically_actionable.pdf")
ggplot(data_for_evaluation,aes(x=MAF,fill=actionability.summary))+
  geom_histogram(alpha=.4)+
  theme_bw()+
  facet_grid(SourceStudy~.,scale="free_y",)+
  xlab("MAF (%)")+
  ggtitle(paste0("MAF of clinically actioanable variants \n in source studies"))+
  scale_fill_brewer(palette="Set1",drop=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 10,width = 10)

outF <-  paste0(MafOutDir,"/MAF_distribution_clinically_actionable_type.pdf")
ggplot(data_for_evaluation,aes(x=MAF,fill=actionability.summary))+
  geom_density(alpha=.4)+
  theme_bw()+
  facet_grid(SourceStudy~.,scale="free_y",)+
  xlab("MAF (%)")+
  ggtitle(paste0("MAF of clinically actioanable variants \n by type"))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 10,width = 10)


isMetPrim <- data_for_evaluation$SAMPLE_TYPE %in% c("Metastasis","Primary")
isMC3 <- data_for_evaluation$SourceStudy == "TCGA-MC3-data"
met.or.primary <- data_for_evaluation[isMetPrim & !isMC3,]

### density plots of actionability type,met/primary, and data source
outF <-  paste0(MafOutDir,"/MAF_distribution_clinically_actionable_type_primary_met.pdf")
ggplot(met.or.primary,aes(x=MAF,fill=actionability.summary))+
  geom_density(alpha=.4)+
  theme_bw()+
  #facet_grid(SourceStudy~SAMPLE_TYPE,scale="free_y",)+
  facet_grid(SAMPLE_TYPE~SourceStudy,scale="free_y",)+
  xlab("MAF (%)")+
  ggtitle(paste0("MAF of clinically actioanable variants \n by type"))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 10,width = 10)

outF <-  paste0(MafOutDir,"/MAF_distribution_met_vs_primary.pdf")
ggplot(met.or.primary,aes(x=MAF,fill=SAMPLE_TYPE))+
  geom_density(alpha=.4)+
  theme_bw()+
  #facet_grid(SourceStudy~SAMPLE_TYPE,scale="free_y",)+
  facet_grid(.~SourceStudy,scale="free_y",)+
  xlab("MAF (%)")+
  ggtitle(paste0("MAF of clinically actioanable variants \n by tissue type"))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6,width = 10)

### create comparisons of distribution types
testResDf <- data.frame()

srcStudies <- unique(data_for_evaluation$SourceStudy)
#sampleTypes <- unique(data_for_evaluation$SAMPLE_TYPE)
sampleTypes <- c("Metastasis","Primary")
actionSummTypes <- unique(data_for_evaluation$actionability.summary)
for (testSource in srcStudies) {
  print(testSource)
  ixSrc <- data_for_evaluation$SourceStudy == testSource
  for (xMetPrim in sampleTypes) {
    ixMetPrim <- data_for_evaluation$SAMPLE_TYPE == xMetPrim
    for (i in 1:(length(actionSummTypes) - 1)) {
      for (j in (i + 1):length(actionSummTypes)) { 
        
        xAS <- actionSummTypes[i]
        yAS <- actionSummTypes[j]
        
        # conditions of test
        ixActionSumm <- data_for_evaluation$actionability.summary == xAS
        iyActionSumm <- data_for_evaluation$actionability.summary == yAS
        
        xVarSubset <- data_for_evaluation[ixSrc & ixMetPrim & ixActionSumm,]
        yVarSubset <- data_for_evaluation[ixSrc & ixMetPrim & iyActionSumm,]
        print(dim(xVarSubset))
        print(dim(yVarSubset))
        
        xSize <- dim(xVarSubset)[[1]]
        ySize <- dim(yVarSubset)[[1]]
        
        # skip statsitical test if there are fewer than N exemplars in a given group
        if ((xSize < 100) | (ySize < 100)) {
          next
        }
        
        # create vectors of MAF values
        xMAFVecTmp <- xVarSubset$MAF
        yMAFVecTmp <- yVarSubset$MAF
        
        ixNa <- is.na(xMAFVecTmp)
        iyNa <- is.na(yMAFVecTmp)
        ixInf <- is.infinite(xMAFVecTmp)
        iyInf <- is.infinite(yMAFVecTmp)
        
        # remove nulls and Inf from vectors
        xMAFVec <- xMAFVecTmp[!ixNa & !ixInf]
        yMAFVec <- yMAFVecTmp[!iyNa & !iyInf]
        
        # Organize data for density plot
        mafDf <- data.frame(value=c(xMAFVec,yMAFVec),
                            condition=xAS)
        iC2 <- seq((length(xMAFVec)+1),dim(mafDf)[[1]])
        mafDf[iC2,"condition"] <- yAS
        #mafDf$condition <- factor(mafDf$condition)
        
        ecdf1_data <- data.frame(x = unique(xMAFVec), y = ecdf(xMAFVec)(unique(xMAFVec)))
        ecdf2_data <- data.frame(x = unique(yMAFVecTmp), y = ecdf(yMAFVecTmp)(unique(yMAFVecTmp)))
        
        # Label the data
        ecdf1_data$Condition <- xAS
        ecdf2_data$Condition <- yAS
        
        # Combine the data frames
        ecdf_data <- rbind(ecdf1_data, ecdf2_data)
        
        # Calculate the KS statistic and the point of maximum difference
        ks_test <- ks.test(xMAFVec, yMAFVec)
        ks_stat <- ks_test$statistic
        xx <- seq(0,100,.1)
        diffs <- abs(ecdf(xMAFVec)(xx) - ecdf(yMAFVec)(xx))
        ks_value <- xx[which.max(diffs)]
        
        # Plot the empirical CDFs using ggplot2
        outF <-  paste0(MafOutDir,"/MAF_distributions_actionable_",xAS,"_",yAS,"_",xMetPrim,"_",testSource,"_ecdf.pdf")
        ggplot(ecdf_data, aes(x = x, y = y, color = Condition)) + 
          geom_line() +
          geom_vline(xintercept = ks_value, linetype = "dotted", color = "black", size = 1) +
          annotate("text", x = ks_value, y = 0.5, label = paste("KS Statistic = ", round(ks_stat, digits=4)), hjust = 1.2) +
          #scale_color_manual(values = c(xAS = 'blue', yAS = 'red')) +
          labs(title = "Empirical CDFs with KS Statistic", x = "Value", y = "ECDF") +
          theme_minimal()
        ggsave(outF,height = 6,width = 6)
        
        ## density plot
        outF <-  paste0(MafOutDir,"/MAF_distributions_actionable_",xAS,"_",yAS,"_",xMetPrim,"_",testSource,".pdf")
        ggplot(mafDf,aes(x=value,fill=condition))+
          geom_density(alpha=.4)+
          theme_bw()+
          geom_vline(xintercept = ks_value, linetype = "dotted", color = "black", size = 1)+ 
          xlab("MAF (%)")+
          ggtitle(paste0("MAF values in ",xMetPrim," tissue \n",testSource))+
          theme(plot.title = element_text(hjust = 0.5))
        ggsave(outF,height = 6,width = 6)
        
        
        # perform wilcox test
        wrs.twosided <- wilcox.test(xMAFVec,
                                    yMAFVec,
                                    alternative = "two.sided",
                                    exact=FALSE,
                                    conf.int=TRUE,
                                    paired=FALSE)
        
        ## create outputs
        c1_temp_summary <- summary(xMAFVec)
        c2_temp_summary <- summary(yMAFVec)
        
        # Create a row to append to the dataframe
        new_row <- data.frame(
          # condition info
          source_study = testSource,
          SampleType = xMetPrim,
          c1_actionability_summary = xAS,
          c2_actionability_summary = yAS,
          c1_variant_count = xSize,
          c2_variant_count = ySize,
          # summary stats condition 1
          c1_min = c1_temp_summary[[1]],
          c1_first_quartile = c1_temp_summary[[2]],
          c1_median = c1_temp_summary[[3]],
          c1_mean = c1_temp_summary[[4]],
          c1_third_quartile = c1_temp_summary[[5]],
          c1_max = c1_temp_summary[[6]],
          # summary stats condition 2
          c2_min = c2_temp_summary[[1]],
          c2_first_quartile = c2_temp_summary[[2]],
          c2_median = c2_temp_summary[[3]],
          c2_mean = c2_temp_summary[[4]],
          c2_third_quartile = c2_temp_summary[[5]],
          c2_max = c2_temp_summary[[6]],
          # wilcox results
          wilcox_p_value = wrs.twosided$p.value,
          wilcox_statistic = wrs.twosided$statistic[[1]],
          wilcox_estimate = wrs.twosided$estimate[[1]]
        )
        
        # Append the new row to the summary dataframe
        testResDf <- rbind(testResDf, new_row)
        
      }
    }
  }
}

# Apply FDR correction
testResDf$p_adjusted_wilcox <- p.adjust(testResDf$wilcox_p_value, method = "fdr")

testResDf <- testResDf %>%
  dplyr::arrange(wilcox_p_value) %>%
  dplyr::mutate(significant = p_adjusted_wilcox < 0.05,
                c1_c2_median_MAF_difference = c1_median - c2_median)
outF <- paste0(MafOutDir,"/actionability_type_MAF_wilcox_test.csv")
write.csv(testResDf,outF)

### plot directionality of relationships and significance
outF <-  paste0(MafOutDir,"/MAF_distribution_wilcox_test_results.pdf")
ggplot(testResDf,aes(x=source_study,y=wilcox_estimate,shape=significant,color=SampleType))+
  geom_point(alpha=.8)+
  theme_bw()+
  facet_grid(c1_actionability_summary~c2_actionability_summary,scale="free_y",)+
  ylim(-8,8)+
  geom_hline(yintercept = 0,linetype="dotted")+
  ggtitle(paste0("Wilcox tests restults of MAF distributions \n by actionability type"))+
  #scale_color_brewer(palette="Set1",drop=FALSE)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 7)

### plot wilcox estimate and median MAF difference 
outF <-  paste0(MafOutDir,"/MAF_wilcox_estimate_and_median_diff.pdf")
ggplot(testResDf,aes(x=wilcox_estimate,y=c1_c2_median_MAF_difference,shape=significant,color=SampleType))+
  geom_point(alpha=.8)+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 5,width = 5)

### top variant summary
atGroupVariantSummary <- data_for_evaluation %>%
  dplyr::group_by(actionability.summary,Hugo_Symbol,HGVSp_Short) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::arrange(desc(n))

atGroupVariantSummaryTop <- atGroupVariantSummary %>%
  dplyr::group_by(actionability.summary) %>%
  dplyr::arrange(desc(n)) %>%
  dplyr::filter(dplyr::row_number() < 16) %>%
  dplyr::mutate(variantName=paste0(Hugo_Symbol," ",HGVSp_Short))

outF <-  paste0(MafOutDir,"/top_variants_by_actionability_type_unristricted.pdf")
ggplot(atGroupVariantSummaryTop,aes(x=actionability.summary,y=n))+
  geom_point(alpha=.8)+
  ggtitle(paste0("Top 15 variants by actionability type \n unrestricted cancer type"))+
  #geom_text(aes(label = variantName), nudge_x = 0.0, nudge_y = 0.2)+
  geom_text_repel(aes(label = variantName))+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 9,width = 9)

outF <-  paste0(MafOutDir,"/top_variants_by_actionability_type_unristricted_log.pdf")
ggplot(atGroupVariantSummaryTop[atGroupVariantSummaryTop$n>1,],aes(x=actionability.summary,y=n))+
  geom_point(alpha=.8)+
  #geom_jitter(alpha=.8,width = .1)+
  ggtitle(paste0("Top 15 variants by actionability type"))+
  #geom_text(aes(label = variantName), nudge_x = 0.0, nudge_y = 0.2)+
  geom_text_repel(aes(label = variantName))+
  scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 9,width = 12)

table(data_for_evaluation$actionability.summary,data_for_evaluation$cancerTypeMatch)

#####################################
### sequencing or report to death ###
#####################################

# compare GENIE to Hartwig 
outF <-  paste0(outDir,"/hartwig_vs_aacr_time_to_death.pdf")
ggplot(sampleInfoCompiled,aes(x=INT_SEQ_TO_DEATH,fill=SourceStudy))+
  geom_density(alpha=.4)+
  theme_bw()+
  xlim(-500,2000)+
  ggtitle(paste0("Infered days between biopsy or sequencing \nreport and death"))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6,width = 6)

outF <-  paste0(outDir,"/hartwig_vs_aacr_time_to_death_sample_type.pdf")
iMetPrim = (sampleInfoCompiled$SAMPLE_TYPE=="Primary" | sampleInfoCompiled$SAMPLE_TYPE=="Metastasis") & !is.na(sampleInfoCompiled$SAMPLE_TYPE)
ggplot(sampleInfoCompiled[iMetPrim,],aes(x=INT_SEQ_TO_DEATH,fill=SourceStudy))+
  geom_density(alpha=.4)+
  facet_grid(SAMPLE_TYPE~.)+
  theme_bw()+
  xlim(-500,2000)+
  ggtitle(paste0("Infered days between biopsy or sequencing \nreport and death"))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6,width = 6)

####################################
### aggregate prognostic markers ###
####################################

# which samples have a prognostic marker: high conf, low, conf
prognosticRepCnts <- aa.genome.representative %>% 
  dplyr::filter(cancerTypeMatch==T) %>%
  dplyr::group_by(Tumor_Sample_Barcode) %>%
  dplyr::summarise(nVariants=n(),
                   nPrognosticVars=sum(actionability.summary == "Prognostic"),
                   nMarkerBetterOutcome=sum(clinical_significance == "Better Outcome"),
                   nMarkerPoorOutcome=sum(clinical_significance == "Poor Outcome"),
                   nPrognosticHighConfVars=sum((actionability.summary == "Prognostic") & (clinical.evidence.summary=="High Confidence")),
                   nPrognosticLowConfVarssum=sum((actionability.summary == "Prognostic") & (clinical.evidence.summary=="Lower Confidence")))


# some samples have both poor outcome and better outcome prognostic markers
table(prognosticRepCnts$nMarkerBetterOutcome,prognosticRepCnts$nMarkerPoorOutcome)

# make prognostic marker assignments by sample
iProgMarkers <- prognosticRepCnts$nPrognosticVars > 0
sampleInfoCompiled$hasPrognosticMarker <- sampleInfoCompiled$SAMPLE_ID %in% prognosticRepCnts[iProgMarkers,"Tumor_Sample_Barcode"][[1]]

sampleInfoCompiled$prognosticMarkerType <- "Prognostic Marker Unspecified"
iBetterProgMarkers <- prognosticRepCnts$nMarkerBetterOutcome > 0
iBetter <- sampleInfoCompiled$SAMPLE_ID %in% prognosticRepCnts[iBetterProgMarkers,"Tumor_Sample_Barcode"][[1]]
sampleInfoCompiled[iBetter,"prognosticMarkerType"] <- "Better Outcome Prognostic Marker"
iPoorProgMarkers <- prognosticRepCnts$nMarkerPoorOutcome > 0
iPoor <- sampleInfoCompiled$SAMPLE_ID %in% prognosticRepCnts[iPoorProgMarkers,"Tumor_Sample_Barcode"][[1]]
sampleInfoCompiled[iPoor,"prognosticMarkerType"] <- "Poor Outcome Prognostic Marker"
sampleInfoCompiled[!sampleInfoCompiled$hasPrognosticMarker,"prognosticMarkerType"] <- "No Prognostic Marker"

outF <-  paste0(outDir,"/hartwig_vs_aacr_time_to_death_sample_type.pdf")
iMetPrim = (sampleInfoCompiled$SAMPLE_TYPE=="Primary" | sampleInfoCompiled$SAMPLE_TYPE=="Metastasis") & !is.na(sampleInfoCompiled$SAMPLE_TYPE) & !is.na(sampleInfoCompiled$INT_SEQ_TO_DEATH)
ggplot(sampleInfoCompiled[iMetPrim,],aes(x=INT_SEQ_TO_DEATH,fill=prognosticMarkerType))+
  geom_density(alpha=.4)+
  facet_grid(SourceStudy~.)+
  theme_bw()+
  xlim(-500,2000)+
  ggtitle(paste0("Infered days between biopsy or sequencing \nreport and death"))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6,width = 8)

# plot only AACR data without "unspecified prog markers
outF <-  paste0(outDir,"/hartwig_vs_aacr_time_to_death_sample_type_v2.pdf")
iProgSelect = (sampleInfoCompiled$SAMPLE_TYPE=="Primary" | sampleInfoCompiled$SAMPLE_TYPE=="Metastasis") & 
  !is.na(sampleInfoCompiled$SAMPLE_TYPE) & 
  !is.na(sampleInfoCompiled$INT_SEQ_TO_DEATH) &
  !sampleInfoCompiled$SourceStudy == "Hartwig-data" &
  !sampleInfoCompiled$prognosticMarkerType == "Prognostic Marker Unspecified"
ggplot(sampleInfoCompiled[iProgSelect,],aes(x=INT_SEQ_TO_DEATH,fill=prognosticMarkerType))+
  geom_density(alpha=.4)+
  facet_grid(SourceStudy~.)+
  theme_bw()+
  xlim(-500,2000)+
  ggtitle(paste0("Infered days between biopsy or sequencing \nreport and death"))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6,width = 8)

####################################
### specific prognostic markers ###
####################################

prognosticTypeMutCnts <- aa.genome.representative %>%
  dplyr::filter(cancerTypeMatch==T,
                actionability.summary == "Prognostic") %>%
  dplyr::group_by(ONCOTREE_CODE,CANCER_TYPE,gene,AAChangeObserved,clinical_significance) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n))
outF <- paste0(ProgOutDir,"/prognostic_variant_frequency_rank.csv")
write.csv(prognosticTypeMutCnts,outF)

prognosticTypeCnts <- aa.genome.representative %>%
  dplyr::filter(cancerTypeMatch==T,
                actionability.summary == "Prognostic") %>%
  dplyr::group_by(CANCER_TYPE,gene,clinical_significance) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n))

# cTypeCnts <- svCompiled %>% dplyr::group_by(ot_name) %>% summarize(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>% arrange(desc(n))

####################################
### specific prognostic markers ###
####################################

prognosticTypeMutCnts <- aa.genome.representative %>%
  dplyr::filter(cancerTypeMatch==T,
                actionability.summary == "Prognostic") %>%
  dplyr::group_by(ONCOTREE_CODE,CANCER_TYPE,gene,AAChangeObserved,clinical_significance) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n))
outF <- paste0(ProgOutDir,"/prognostic_variant_frequency_rank.csv")
write.csv(prognosticTypeMutCnts,outF)

prognosticTypeCnts <- aa.genome.representative %>%
  dplyr::filter(cancerTypeMatch==T,
                actionability.summary == "Prognostic") %>%
  dplyr::group_by(CANCER_TYPE,gene,clinical_significance) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n))

### create comparisons of distribution types
testResProgDf <- data.frame()

#otCodes <- unique(prognosticTypeCnts$ONCOTREE_CODE)
ctTypes <- unique(prognosticTypeCnts$CANCER_TYPE)
progGenes <- unique(prognosticTypeCnts$gene)
progSig <- unique(prognosticTypeCnts$clinical_significance)

for (iX in rownames(prognosticTypeCnts)) {
  print(iX)
  ctType <- prognosticTypeCnts[iX,"CANCER_TYPE"][[1]]
  progGene <- prognosticTypeCnts[iX,"gene"][[1]]
  progSigDir <- prognosticTypeCnts[iX,"clinical_significance"][[1]]
  print(paste(ctType,"-",progGene,"-",progSigDir))
  
  # select variants with the prognostic markers
  xVarSubset <- aa.genome.representative %>%
    dplyr::filter(cancerTypeMatch==T,
                  actionability.summary == "Prognostic",
                  CANCER_TYPE == ctType,
                  gene == progGene,
                  clinical_significance == progSigDir)
  # all other variants for the cancer type
  yVarSubset <- aa.genome.representative %>%
    dplyr::filter(CANCER_TYPE == ctType,
                  !Tumor_Sample_Barcode %in% xVarSubset$Tumor_Sample_Barcode)
  
  # select the samples with the variants (and the metric)
  xSubSetSamples <- xVarSubset %>%
    dplyr::group_by(Tumor_Sample_Barcode) %>%
    dplyr::filter(dplyr::row_number()==1)
  
  ySubSetSamples <- yVarSubset %>%
    dplyr::group_by(Tumor_Sample_Barcode) %>%
    dplyr::filter(dplyr::row_number()==1)
  
  xSize <- dim(xSubSetSamples)[[1]]
  ySize <- dim(ySubSetSamples)[[1]]
  
  # skip statsitical test if there are fewer than N exemplars in a given group
  if ((xSize < 30) | (ySize < 30)) {
    next
  }
  
  # create vectors of MAF values
  xSeqToDeathTmp <- xSubSetSamples$INT_SEQ_TO_DEATH
  ySeqToDeathTmp <- ySubSetSamples$INT_SEQ_TO_DEATH
  
  ixNa <- is.na(xSeqToDeathTmp)
  iyNa <- is.na(ySeqToDeathTmp)
  ixInf <- is.infinite(xSeqToDeathTmp)
  iyInf <- is.infinite(ySeqToDeathTmp)
  
  # remove nulls and Inf from vectors
  xSeqToDeathVec <- xSeqToDeathTmp[!ixNa & !ixInf]
  ySeqToDeathVec <- ySeqToDeathTmp[!iyNa & !iyInf]
  
  # Organize data for density plot
  mafDf <- data.frame(value=c(xSeqToDeathVec,ySeqToDeathVec),
                      condition="sample has prognostic marker")
  iC2 <- seq((length(xSeqToDeathVec)+1),dim(mafDf)[[1]])
  mafDf[iC2,"condition"] <- "no prognostic marker in sample"
  
  # Calculate the KS statistic and the point of maximum difference
  ks_test <- ks.test(xSeqToDeathVec, ySeqToDeathVec)
  ks_stat <- ks_test$statistic
  xx <- seq(0,100,.1)
  diffs <- abs(ecdf(xSeqToDeathVec)(xx) - ecdf(ySeqToDeathVec)(xx))
  ks_value <- xx[which.max(diffs)]
  
  outF <-  paste0(ProgOutDir,"/days_from_seq_to_death_",ctType,"_",progGene,"_",progSigDir,".pdf")
  ggplot(mafDf,aes(x=value,fill=condition))+
    geom_density(alpha=.4)+
    theme_bw()+
    geom_vline(xintercept = ks_value, linetype = "dotted", color = "black", size = 1)+
    xlab("Days between sequencing and death")+
    ggtitle(paste0("Death outcome for ",ctType,"\n with prognostic ",progGene," mutation \n expected direction: ",progSigDir))+
    theme(plot.title = element_text(hjust = 0.5))
  ggsave(outF,height = 6,width = 6)
  
  # perform wilcox test
  wrs.twosided <- wilcox.test(xSeqToDeathVec,
                              ySeqToDeathVec,
                              alternative = "two.sided",
                              exact=FALSE,
                              conf.int=TRUE,
                              paired=FALSE)
  
  ## create outputs
  c1_temp_summary <- summary(xSeqToDeathVec)
  c2_temp_summary <- summary(ySeqToDeathVec)
  
  # Create a row to append to the dataframe
  new_row <- data.frame(
    # condition info
    cancer_type = ctType,
    prognostic_marker_gene=progGene,
    expected_significance=progSigDir,
    # summary stats condition 1
    c1_min = c1_temp_summary[[1]],
    c1_first_quartile = c1_temp_summary[[2]],
    c1_median = c1_temp_summary[[3]],
    c1_mean = c1_temp_summary[[4]],
    c1_third_quartile = c1_temp_summary[[5]],
    c1_max = c1_temp_summary[[6]],
    # summary stats condition 2
    c2_min = c2_temp_summary[[1]],
    c2_first_quartile = c2_temp_summary[[2]],
    c2_median = c2_temp_summary[[3]],
    c2_mean = c2_temp_summary[[4]],
    c2_third_quartile = c2_temp_summary[[5]],
    c2_max = c2_temp_summary[[6]],
    # wilcox results
    wilcox_p_value = wrs.twosided$p.value,
    wilcox_statistic = wrs.twosided$statistic[[1]],
    wilcox_estimate = wrs.twosided$estimate[[1]]
  )
  
  # Append the new row to the summary dataframe
  testResProgDf <- rbind(testResProgDf, new_row)
}

# Apply FDR correction
testResProgDf$p_adjusted_wilcox <- p.adjust(testResProgDf$wilcox_p_value, method = "fdr")

testResProgDf <- testResProgDf %>%
  dplyr::arrange(wilcox_p_value) %>%
  dplyr::mutate(significant = p_adjusted_wilcox < 0.05,
                c1_c2_median_MAF_difference = c1_median - c2_median)
outF <- paste0(ProgOutDir,"/prognostic_markers_days_from_seq_to_death_wilcox_test.csv")
write.csv(testResProgDf,outF)

outF <-  paste0(outDir,"/prognostic_markers_int_to_death_qq_plot_raw_p_values.pdf")
pdf(outF)
qqplot(-log10(ppoints(length(testResProgDf$wilcox_p_value))), 
       -log10(sort(testResProgDf$wilcox_p_value)), 
       xlab = "Expected -log10(p-values)", 
       ylab = "Observed -log10(p-values)", 
       main = "QQ-plot of p-values")
abline(0, 1, col = "red")  # Line y = x for reference
ggsave(outF,height = 8,width = 12)
dev.off()

# QQ-plot with FDR-corrected p-values
outF <-  paste0(outDir,"/prognostic_markers_int_to_death_qq_plot_fdr_corrected_p_values.pdf")
pdf(outF)
qqplot(-log10(ppoints(length(testResProgDf$p_adjusted_wilcox))), 
       -log10(sort(testResProgDf$p_adjusted_wilcox)), 
       xlab = "Expected -log10(FDR-corrected p-values)", 
       ylab = "Observed -log10(FDR-corrected p-values)", 
       main = "QQ-plot of FDR-corrected p-values")
abline(0, 1, col = "blue")  # Line y = x for reference
dev.off()

#######################################################
### Sankey diagram of biomarker to drug assignments ###
#######################################################

### Sankey data product - drung
aa.genome.representative$AAChangeStr <- paste0(aa.genome.representative$gene,"-",aa.genome.representative$AAChangeObserved)
SankeyDrugCnts <- aa.genome.representative %>%
  dplyr::filter(cancerTypeMatch==T,
                actionability.summary == "Predictive therapy actionability") %>%
  dplyr::group_by(CANCER_TYPE,gene,AAChangeStr,Drugs) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n))
outF <- paste0(outDir,"/sankey_plot_drug_table.csv")
write.csv(SankeyDrugCnts,outF)


outF <-  paste0(outDir,"/sankey_plot_drugs_diagram.pdf")
ggplot(data = SankeyDrugCnts[seq(1:20),],
       aes(axis1 = CANCER_TYPE, axis2 = AAChangeStr, axis3 = Drugs, y = n)) +
  geom_alluvium(aes(fill = CANCER_TYPE)) + 
  geom_stratum() +  
  geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 3) +  # Add text labels
  theme_minimal() +  
  ggtitle("Top restricted drug matches")+  
  scale_fill_brewer(palette="Set2",drop=FALSE)+
  #theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 12)


### Sankey data product - prognosis by AA Change
SankeyPrognosticCnts <- aa.genome.representative %>%
  dplyr::filter(cancerTypeMatch==T,
                actionability.summary == "Prognostic") %>%
  dplyr::group_by(CANCER_TYPE,gene,AAChangeStr,clinical_significance) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n))
outF <- paste0(outDir,"/sankey_plot_prognosis_table.csv")
write.csv(SankeyPrognosticCnts,outF)

outF <-  paste0(outDir,"/sankey_plot_prognosis_diagram.pdf")
ggplot(data = SankeyPrognosticCnts[seq(1:20),],
       aes(axis1 = CANCER_TYPE, axis2 = AAChangeStr, axis3 = clinical_significance, y = n)) +
  geom_alluvium(aes(fill = CANCER_TYPE)) + 
  geom_stratum() +  
  geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 3) +  # Add text labels
  theme_minimal() +  
  ggtitle("Top restricted drug matches")+  
  #scale_fill_brewer(palette="Set2",drop=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 12)

### Sankey data product - prognosis by Gene Change
SankeyPrognosticCntsGene <- aa.genome.representative %>%
  dplyr::filter(cancerTypeMatch==T,
                actionability.summary == "Prognostic") %>%
  dplyr::group_by(CANCER_TYPE,gene,clinical_significance) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n))
outF <- paste0(outDir,"/sankey_plot_prognosis_by_gene_table.csv")
write.csv(SankeyPrognosticCntsGene,outF)


outF <-  paste0(outDir,"/sankey_plot_prognosis_by_gene_diagram.pdf")
ggplot(data = SankeyPrognosticCntsGene[seq(1:15),],
       aes(axis1 = CANCER_TYPE, axis2 = gene, axis3 = clinical_significance, y = n)) +
  geom_alluvium(aes(fill = CANCER_TYPE)) + 
  geom_stratum() +  
  geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 3) +  # Add text labels
  theme_minimal() +  
  ggtitle("Top restricted drug matches")+  
  #scale_fill_brewer(palette="Set2",drop=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 10,width = 10)



# tumor purity
outF <-  paste0(outDir,"/tumor_purity_by_study.pdf")
ggplot(svCompiled,aes(x=TUMOR_PURITY,fill=SourceStudy))+
  geom_density(alpha=.4)+
  theme_minimal() +
  ggtitle("Tumor purity values listed by study")+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6,width = 6)
# 
# # VAF and tumor purity - all points
iNonNullPurity <- !is.na(svCompiled$TUMOR_PURITY)
# outF <- paste0(outDir,"/tumor_purity_and_MAF_by_study.pdf")
# ggplot(svCompiled[iNonNullPurity,],aes(x=TUMOR_PURITY, y=MAF, color=SourceStudy))+
#   geom_point(alpha=.02)+
#   geom_density_2d()+
#   #stat_density_2d(aes(fill = ..level..), color = "black")+ #geom = "polygon",
#   ylim(c(0,99))+
#   ggtitle("Tumor purity and MAF values listed by study")+  
#   scale_color_brewer(palette="Set2",drop=FALSE)+
#   theme(plot.title = element_text(hjust = 0.5))
# ggsave(outF,height = 8,width = 8)


outF <- paste0(outDir,"/tumor_purity_and_MAF_by_study_v2.pdf")
ggplot(svCompiled[iNonNullPurity,],aes(x=TUMOR_PURITY, y=MAF, color=SourceStudy))+
  #geom_point(alpha=.02)+
  geom_density_2d_filled(alpha = 0.5) +
  geom_density_2d(linewidth = 0.25)+
  ylim(c(0,99))+
  ggtitle("Tumor purity and MAF values listed by study")+  
  scale_color_brewer(palette="Set2",drop=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 8)

iNonNullPurity <- !is.na(aa.genome.representative$TUMOR_PURITY)
outF <- paste0(outDir,"/tumor_purity_and_MAF_by_study_actionable.pdf")
ggplot(aa.genome.representative[iNonNullPurity,],aes(x=TUMOR_PURITY, y=MAF, color=SourceStudy))+
  geom_point(alpha=.02)+
  geom_density_2d()+
  #stat_density_2d(aes(fill = ..level..), color = "black")+ #geom = "polygon",
  ylim(c(0,99))+
  ggtitle("Tumor purity and MAF values listed by study")+  
  scale_color_brewer(palette="Set2",drop=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 8)


# calculate MAF Purity products
aa.genome.representative$MAF_x_Purity <- NA
aa.genome.representative[iNonNullPurity,"MAF_x_Purity"] <- aa.genome.representative[iNonNullPurity,"MAF"] * aa.genome.representative[iNonNullPurity,"TUMOR_PURITY"]

aa.genome.representative$MAF_divided_by_Purity <- NA
aa.genome.representative[iNonNullPurity,"MAF_divided_by_Purity"] <- aa.genome.representative[iNonNullPurity,"MAF"] / aa.genome.representative[iNonNullPurity,"TUMOR_PURITY"]

outF <-  paste0(outDir,"/MAF_x_Purity_tumor_purity_by_study.pdf")
ggplot(aa.genome.representative[iNonNullPurity,],aes(x=MAF_x_Purity,fill=SourceStudy))+
  geom_density(alpha=.4)+
  theme_minimal() +
  ggtitle("Tumor purity values listed by study")+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6,width = 6)

outF <-  paste0(outDir,"/MAF_divided_by_Purity_tumor_purity_by_study.pdf")
ggplot(aa.genome.representative[iNonNullPurity,],aes(x=MAF_divided_by_Purity,fill=SourceStudy))+
  geom_density(alpha=.4)+
  theme_minimal() +
  ggtitle("Tumor purity values listed by study")+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6,width = 6)

# model VAF inputs: alt count and (alt+ref cnt)
aa.genome.representative <- aa.genome.representative %>%
  dplyr::mutate(t_alt_plus_ref_count = t_ref_count+t_alt_count)
iOutlier <- aa.genome.representative$t_alt_plus_ref_count > 3000
isNaReadCnt <- is.na(aa.genome.representative$t_alt_plus_ref_count) | is.na(aa.genome.representative$t_alt_count)
outF <- paste0(outDir,"/alt_cnt_ref_cnt_MAF_by_study_actionable.pdf")
dfForPlot <- aa.genome.representative[!iOutlier & !isNaReadCnt,]
dfForPlot$fivePercThresh <- dfForPlot$t_alt_plus_ref_count*0.05
dfForPlot$onePercThresh <- dfForPlot$t_alt_plus_ref_count*0.01
ggplot(dfForPlot,aes(x=t_alt_plus_ref_count, y=t_alt_count, color=SourceStudy))+ #
  geom_point(alpha=.02)+
  geom_density_2d()+
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  #stat_density_2d(aes(fill = ..level..), color = "black")+ #geom = "polygon",
  #ylim(c(0,99))+
  ggtitle("Ref and Alt counts for actionable mutations")+  
  scale_color_brewer(palette="Set2",drop=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 8)

# with 5% threshold point
outF <- paste0(outDir,"/alt_cnt_ref_cnt_MAF_with_5Perc_thresh.pdf")
ggplot(dfForPlot,aes(x=t_alt_plus_ref_count, y=t_alt_count))+ #, color=SourceStudy
  geom_point(alpha=.02)+
  geom_density_2d()+
  geom_line(data=data.frame(t_alt_plus_ref_count=dfForPlot$t_alt_plus_ref_count,t_alt_count=dfForPlot$fivePercThresh),color="green",alpha=.3)+
  geom_line(data=data.frame(t_alt_plus_ref_count=dfForPlot$t_alt_plus_ref_count,t_alt_count=dfForPlot$onePercThresh),color="red",alpha=.3)+
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  ggtitle("Ref and Alt counts for actionable mutations")+  
  scale_color_brewer(palette="Set2",drop=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 8)

# model raw MAF
iNonNullPurity <- !is.na(aa.genome.representative$TUMOR_PURITY)
iCancerMatched <- aa.genome.representative$cancerTypeMatch==T
linModData <- aa.genome.representative[iNonNullPurity & iCancerMatched,]

# Convert categorical variable into one-hot encoding
linModDataOHE <- linModData %>%
  mutate(row_id = row_number()) %>%  # Add a unique row identifier
  pivot_wider(names_from = actionability.summary, values_from = actionability.summary, values_fill = 0, 
              values_fn = list(actionability.summary = length)) %>%
  select(-row_id) %>%  # Remove the row identifier
  mutate(across(.cols = everything(), .fns = ~ as.integer(. > 0)))  # Convert counts to binary


##model <- stats::lm(actionability.summary ~ MAF + TUMOR_PURITY + SourceStudy, data = aa.genome.representative[iNonNullPurity,])
#model <- stats::lm(MAF ~ TUMOR_PURITY + actionability.summary + SourceStudy, data = linModData)
#model <- stats::lm(MAF ~ TUMOR_PURITY + actionability.summary + ot_code_level_1 + SourceStudy, data = linModData)
#
## model MAF divided by purity
#model <- stats::lm(MAF_divided_by_Purity ~ actionability.summary + SourceStudy, data = linModData)
##model <- stats::lm(MAF_divided_by_Purity ~ Predictive.therapy.actionability + Therapy.resistance +  Prognostic + SourceStudy, data = linModDataOHE)
#
#
## Summarize the model
#summary(model)

# plot model output and residuals
# library(car)
# # Creating a partial residual plot for 'age' in the model
# crPlots(model)
# avPlots(model)
# par(mfrow=c(2,2))
# plot(model)
