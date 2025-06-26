library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(ggplot2)
library(reshape2)

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
outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250522/oncokb_summary_and_modeling")
geneOutDir <- paste0(outDir,"/gene_panel_coverage_plots")
modelFeatureOutDir <- paste0(outDir,"/underlying_feature_model_plots")
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

##########################
### load data products ###
##########################

ot_code_full <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM OncoTreeCodeHierarchy')
sampleInfoCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM harmonizedSampleInfo') 

svCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM patientObservedVariantTableWithSampleInfoOncokb') %>%
  dplyr::mutate(isHemeCancer=grepl("LYMPH",ot_code_level_1) | grepl("MYELOID",ot_code_level_1))
svCompiled$SEQ_ASSAY_ID_mod <- svCompiled$SEQ_ASSAY_ID
svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SEQ_ASSAY_ID_mod"] <- svCompiled[is.na(svCompiled$SEQ_ASSAY_ID),"SourceStudy"]

aa.genome.representative <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM representativeClinicalAnnotatedPatientVariants') 
aa.genome.representative.oncokb <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM representativeClinicalAnnotatedPatientVariantsWithOncoKB') 

#################################
### summary of oncokb results ###
#################################

### find oncokb matched variants
onckbAnnotated <- svCompiled %>%
  dplyr::filter(ONCOKB_IS_SENS_DX_PX_OR_RESISTANCE == T)

table(onckbAnnotated$balderVariantID %in% aa.genome.representative$balderVariantID)
table(aa.genome.representative$balderVariantID %in% onckbAnnotated$balderVariantID)

## which variants are onckb annotated but not annotated via MIA/CiVIC? 
iOnckbOnlyGrp <- onckbAnnotated$balderVariantID %in% aa.genome.representative$balderVariantID
okBID <- onckbAnnotated[!iOnckbOnlyGrp,"balderVariantID"]
oncokbOnly <-svCompiled[svCompiled$balderVariantID %in% okBID,]  

###########################
### oncokb level counts ###
###########################

inF <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_15p0_public/assay_information.txt")
assayInfo <- read.csv(inF,sep="\t") %>%
  dplyr::mutate(isHemePanel=grepl("HEME",SEQ_ASSAY_ID) | grepl("MYELOID",SEQ_ASSAY_ID))
RSQLite::dbWriteTable(harmonizedDb, "assayInfoGENIE", assayInfo, overwrite =T)

# SEQ_ASSAY_ID and SEQ_PIPELINE_ID
seq_grp <- assayInfo %>%
  dplyr::group_by(SEQ_ASSAY_ID,SEQ_PIPELINE_ID) %>%
  dplyr::summarise(number_of_panels=n())
outF <- paste0(outDir,"/institution_seq_platoform_IDs.csv")
write.table(seq_grp,outF,row.names=F,quote=F,sep=",")

# variant counts per subject
svAssaySummary <- svCompiled %>%
  filter(!is.na(SEQ_ASSAY_ID)) %>%
  dplyr::group_by(SEQ_ASSAY_ID,Tumor_Sample_Barcode) %>%
  dplyr::summarize(n.variants=dplyr::n_distinct(balderVariantID),
                   n.patients=dplyr::n_distinct(Tumor_Sample_Barcode),
                   n.oncokb.level1=sum(ONCOKB_HIGHEST_LEVEL_SUMMARY == "LEVEL_1"),
                   has.oncokb.level1=any(ONCOKB_HIGHEST_LEVEL_SUMMARY == "LEVEL_1")) %>%
  dplyr::left_join(assayInfo,by="SEQ_ASSAY_ID") %>%
  dplyr::arrange(desc(n.variants))
svAssaySummary[is.na(svAssaySummary$has.oncokb.level1),"has.oncokb.level1"] <- F



# nullSeq <- is.na(svAssaySummary$SEQ_ASSAY_ID)
# outF <-  paste0(outDir,"/variant_count_by_assay_type.pdf")
# ggplot(svAssaySummary[!nullSeq,])+
#   #geom_point(alpha=.2)+
#   geom_density_2d(aes(x=number_of_genes,y=n.variants,color=library_selection))+
#   theme_bw()+
#   #facet_grid(SAMPLE_TYPE~.,scale="free_y",)+
#   xlab("Number of genes measured")+
#   ylab("number of variants")+
#   ggtitle(paste0("Per patient variant count by assay type"))+
#   scale_fill_brewer(palette="Set1",drop=FALSE)+
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#   theme(plot.title = element_text(hjust = 0.5))
# ggsave(outF,height = 8,width = 7)

###########################
### oncokb level counts ###
###########################

### create an incidence table by cancer type
cType.sample.counts <- svCompiled %>%
  #dplyr::group_by(ONCOTREE_CODE) %>%
  group_by(!!sym("ONCOTREE_CODE")) %>%
  #group_by(as.symbol("ONCOTREE_CODE")) %>%
  dplyr::summarise(n_patients_total=dplyr::n_distinct(Tumor_Sample_Barcode),
                   CANCER_TYPE=paste0(unique(CANCER_TYPE),collapse=";")) %>%
  dplyr::arrange(desc(n_patients_total))

otCodeOncokBSummary <- oncokbOnly %>%
  dplyr::group_by(ONCOTREE_CODE,ONCOKB_HIGHEST_LEVEL_SUMMARY) %>%
  dplyr::summarize(n.variants.matched=dplyr::n_distinct(balderVariantID),
                   n.patients.matched=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n.patients.matched))

otCTypeSummary <- cType.sample.counts %>%
  dplyr::left_join(otCodeOncokBSummary,by="ONCOTREE_CODE") %>%
  dplyr::mutate(perc=100*(n.patients.matched/n_patients_total))
otCTypeSummary$ONCOTREE_CODE <- factor(otCTypeSummary$ONCOTREE_CODE, levels=unique(otCTypeSummary$ONCOTREE_CODE))
outF <- paste0(outDir,"/oncokb_match_level_summary_by_ot_code.csv")
write.table(otCTypeSummary,outF,row.names=F,quote=F,sep=",")
RSQLite::dbWriteTable(harmonizedDb, "otCTypeSummary", otCTypeSummary, overwrite =T)

highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:10]
iHighest <- otCTypeSummary$ONCOTREE_CODE %in% as.character(highest_c_counts)

outF <-  paste0(outDir,"/cancer_type_oncokb_evidence_level.pdf")
ggplot(otCTypeSummary[iHighest,],aes(x=ONCOTREE_CODE,y=perc,fill=ONCOKB_HIGHEST_LEVEL_SUMMARY))+
  geom_bar(stat = "identity", position = "stack",alpha=.6) +
  theme_bw()+
  #facet_grid(SAMPLE_TYPE~.,scale="free_y",)+
  xlab("Oncotree Code")+
  ylab("perc of patients")+
  ggtitle(paste0("Oncokb evidence level by cancer type"))+
  scale_fill_brewer(palette="Set1",drop=FALSE)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 7)

otTissueOncokBSummary <- oncokbOnly %>%
  dplyr::group_by(ot_code_level_1,ONCOKB_HIGHEST_LEVEL_SUMMARY) %>%
  dplyr::summarize(n.variants.restricted.matched=dplyr::n_distinct(balderVariantID),
                   n.patients.restricted.matched=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n.patients.restricted.matched))

#############################################
### oncokb level counts  - primary vs met ###
#############################################

iPrimOrMet <- (oncokbOnly$SAMPLE_TYPE=="Primary") | (oncokbOnly$SAMPLE_TYPE=="Metastasis")
primMetOncokBSummary <- oncokbOnly[iPrimOrMet,] %>%
  dplyr::group_by(ONCOTREE_CODE,ONCOKB_HIGHEST_LEVEL_SUMMARY,SAMPLE_TYPE) %>%
  dplyr::summarize(n.variants.matched=dplyr::n_distinct(balderVariantID),
                   n.patients.matched=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n.patients.matched))
primMetOncokBSummary$ONCOTREE_CODE <- factor(primMetOncokBSummary$ONCOTREE_CODE, levels=unique(otCTypeSummary$ONCOTREE_CODE))

highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:20]
iHighest <- primMetOncokBSummary$ONCOTREE_CODE %in% as.character(highest_c_counts)

iLevel <- grepl("1",primMetOncokBSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY)

outF <-  paste0(outDir,"/cancer_type_prim_met_oncokb_evidence_level.pdf")
ggplot(primMetOncokBSummary[iHighest & iLevel,],aes(x=ONCOTREE_CODE,y=n.patients.matched,fill=ONCOKB_HIGHEST_LEVEL_SUMMARY))+
  geom_bar(stat = "identity", position = "stack",alpha=.6) +
  theme_bw()+
  facet_grid(SAMPLE_TYPE~.,scale="free_y",)+
  xlab("Oncotree Code")+
  ylab("perc of patients")+
  ggtitle(paste0("Oncokb evidence level by cancer type"))+
  #scale_fill_brewer(palette="Set1",drop=FALSE)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8,width = 7)

#######################################
### oncokb level counts  - assay ID ###
#######################################

# Which data sources see the largest differences in actionability rates? Is this because of cohort differences or technology differences? Which specific mutations explain the key differences?

assayCntRank <- svCompiled %>%
  dplyr::group_by(SEQ_ASSAY_ID_mod) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>% 
  dplyr::arrange(desc(n)) %>%
  dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID")) 
outF <- paste0(outDir,"/institution_seq_platoform.txt")
write.table(assayCntRank,outF,row.names=F,quote=F,sep="\t")

assayCTypeCntRank <- svCompiled %>%
  dplyr::mutate(isHemeCancer=grepl("LYMPH",ot_code_level_1) | grepl("MYELOID",ot_code_level_1)) %>%
  dplyr::group_by(SEQ_ASSAY_ID_mod,isHemeCancer) %>%
  dplyr::summarise(n=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n)) %>%
  dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID"))
outF <- paste0(outDir,"/institution_seq_platoform_heme_status.txt")
write.table(assayCTypeCntRank,outF,row.names=F,quote=F,sep="\t")
assayCTypeCntRank$sampleHemeStatus <- "not heme cancer"
assayCTypeCntRank[assayCTypeCntRank$isHemeCancer,"sampleHemeStatus"] <- "heme cancer"


outF <-  paste0(outDir,"/institution_seq_platoform_heme_status.pdf")
ggplot(assayCTypeCntRank,aes(x=SEQ_ASSAY_ID_mod,y=n,fill=isHemePanel))+
  #geom_point()+
  geom_bar(stat = "identity", position = "stack",alpha=.6) +
  facet_grid(sampleHemeStatus ~ .)+
  theme_bw()+
  xlab("panel")+
  ylab("number of subjects")+
  #ylim(0,60)+
  ggtitle(paste0("sample counts by panel type and heme status"))+
  scale_fill_brewer(palette="Set1",drop=FALSE)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 10, width = 15)

# select assays with >N patients
assaySelect <- assayCntRank[assayCntRank$n>5000,"SEQ_ASSAY_ID_mod"][[1]]

seqTypeCntAssay <- svCompiled[svCompiled$SEQ_ASSAY_ID_mod %in% assaySelect,] %>%
  dplyr::group_by(ONCOTREE_CODE,SEQ_ASSAY_ID_mod) %>% # SAMPLE_TYPE
  dplyr::summarise(n_patients_total=dplyr::n_distinct(Tumor_Sample_Barcode),
                   CANCER_TYPE=paste0(unique(CANCER_TYPE),collapse=";")) %>%
  dplyr::arrange(desc(n_patients_total))

# compile clinical info by sequencing type
seqTypeClinicalInfo <- svCompiled[svCompiled$SEQ_ASSAY_ID_mod %in% assaySelect,] %>%
  dplyr::group_by(Tumor_Sample_Barcode) %>% 
  dplyr::filter(dplyr::row_number() == 1) %>% # select a single representative entry per subject
  dplyr::ungroup() %>%
  dplyr::group_by(ONCOTREE_CODE,SEQ_ASSAY_ID_mod) %>% # SAMPLE_TYPE
  dplyr::summarise(n_patients_total=dplyr::n_distinct(Tumor_Sample_Barcode),
                   CANCER_TYPE=paste0(unique(CANCER_TYPE),collapse=";"),
                   n_Metastatic=sum(SAMPLE_TYPE == "Metastasis"),
                   n_Primary=sum(SAMPLE_TYPE == "Primary"),
                   met_to_primary_ratio=n_Metastatic/n_Primary,
                   n_LocalRecurrence=sum(SAMPLE_TYPE == "LocalRecurance"),
                   perc_LocalRecurrence=n_LocalRecurrence/n_patients_total,
                   meanTumorPurity=mean(TUMOR_PURITY,na.rm=T),
                   anyTumorPurityInfo=any(!is.na(TUMOR_PURITY)),
                   n_NaHeme = sum(SAMPLE_TYPE == "NA or Heme"),
                   perc_NaHeme=n_NaHeme/n_patients_total,
                   median_age=median(AGE_AT_SEQ_REPORT,na.rm=T),
                   n_dead=sum(DEAD == "TRUE" | DEAD == "True",na.rm=T),
                   perc_dead=n_dead/n_patients_total,
                   median_IntSeqToDeath=median(INT_Seq_or_Biopsy_To_Death, na.rm=T),
                   n_StageI=sum(STAGE=="I" | STAGE=="1" | STAGE=="1a" | STAGE=="Ia" | STAGE=="IA" |  STAGE=="Ib" | STAGE=="IB" | STAGE=="1b", na.rm=T),
                   perc_StageI=n_StageI/n_patients_total,
                   n_StageII=sum(STAGE=="II" | STAGE=="2" | STAGE=="2a" | STAGE=="IIa" | STAGE=="IIA" | STAGE=="IIb" | STAGE=="IIB" | STAGE=="2b", na.rm=T),
                   perc_StageII=n_StageII/n_patients_total,
                   n_StageIII=sum(STAGE=="III" | STAGE=="3" | STAGE=="3a" | STAGE=="IIIA" | STAGE=="IIIB" | STAGE=="3B", na.rm=T),
                   perc_StageIII=n_StageIII/n_patients_total,
                   n_StageIV=sum(STAGE=="IV" | STAGE=="M1" | STAGE=="M2" | STAGE=="M3", na.rm=T),
                   perc_StageIV=n_StageIV/n_patients_total) %>%
  dplyr::arrange(desc(n_patients_total))
outF <- paste0(outDir,"/cancer_type_assay_clinical_info.csv")
write.table(seqTypeClinicalInfo,outF,row.names=F,quote=F,sep=",")
RSQLite::dbWriteTable(harmonizedDb, "seqTypeClinicalInfo", seqTypeClinicalInfo, overwrite =T)

iPrimOrMet <- (oncokbOnly$SAMPLE_TYPE=="Primary") | (oncokbOnly$SAMPLE_TYPE=="Metastasis")
seqOncokBSummary <- oncokbOnly[iPrimOrMet,] %>%
  dplyr::group_by(ONCOTREE_CODE,SEQ_ASSAY_ID_mod,ONCOKB_HIGHEST_LEVEL_SUMMARY) %>%
  dplyr::summarize(n.variants.matched=dplyr::n_distinct(balderVariantID),
                   n.patients.matched=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n.patients.matched))
primMetOncokBSummary$ONCOTREE_CODE <- factor(primMetOncokBSummary$ONCOTREE_CODE, levels=unique(otCTypeSummary$ONCOTREE_CODE))

seqOtCTypeSummary <- seqTypeCntAssay %>%
  dplyr::left_join(seqOncokBSummary,by=c("ONCOTREE_CODE","SEQ_ASSAY_ID_mod")) %>%
  dplyr::filter(n_patients_total>50) %>%
  dplyr::mutate(perc=100*(n.patients.matched/n_patients_total)) %>%
  dplyr::left_join(seqTypeClinicalInfo[,!colnames(seqTypeClinicalInfo) %in% c("n_patients_total","CANCER_TYPE")],
                   by=c("ONCOTREE_CODE","SEQ_ASSAY_ID_mod")) %>%
  dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID" ))
seqOtCTypeSummary$ONCOTREE_CODE <- factor(seqOtCTypeSummary$ONCOTREE_CODE, levels=unique(otCTypeSummary$ONCOTREE_CODE))
outF <- paste0(outDir,"/oncokb_match_level_summary_by_seq_type.csv")
write.table(seqOtCTypeSummary,outF,row.names=F,quote=F,sep=",")
RSQLite::dbWriteTable(harmonizedDb, "seqOtCTypeSummary", seqOtCTypeSummary, overwrite =T)

highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:5]
iHighest <- seqOtCTypeSummary$ONCOTREE_CODE %in% as.character(highest_c_counts)
iLevel1 <- grepl("1",seqOtCTypeSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY)
iLevel2 <- grepl("2",seqOtCTypeSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY)
seqOtCTypeSummary$level_1 <- iLevel1

outF <-  paste0(outDir,"/cancer_type_Seq_Type_oncokb_evidence_level.pdf")
ggplot(seqOtCTypeSummary[iHighest & (iLevel1),],aes(x=ONCOTREE_CODE,y=perc,fill=ONCOKB_HIGHEST_LEVEL_SUMMARY))+
  geom_bar(stat = "identity", position = "stack",alpha=.6) +
  theme_bw()+
  facet_grid(SEQ_ASSAY_ID_mod~.,scale="free_y",)+
  xlab("Oncotree Code")+
  ylab("perc of patients")+
  ylim(0,60)+
  ggtitle(paste0("Oncokb evidence level by cancer type"))+
  #scale_fill_brewer(palette="Set1",drop=FALSE)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 14, width = 8)

highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:40]
iHighest <- seqOtCTypeSummary$ONCOTREE_CODE %in% as.character(highest_c_counts)
outF <-  paste0(outDir,"/cancer_type_Seq_Type_oncokb_level1_jitter.pdf")
ggplot(seqOtCTypeSummary[iHighest & (iLevel1),],aes(x=ONCOTREE_CODE,y=perc,color=ONCOKB_HIGHEST_LEVEL_SUMMARY))+
  #geom_bar(stat = "identity", position = "stack",alpha=.6) +
  geom_jitter(width=.1,alpha=.8)+
  theme_bw()+
  #facet_grid(level_1~.,scale="free_y",)+
  xlab("Oncotree Code")+
  ylab("perc of patients")+
  ylim(0,85)+
  ggtitle(paste0("Oncokb evidence level by cancer type"))+
  scale_color_brewer(palette="Set2",drop=FALSE)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 6, width = 8)

### repeat plot ordered by most variability in level 1 incidence 
#iLevel1 <- grepl("1",seqOtCTypeSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY)
iLevel1 <- seqOtCTypeSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY == "LEVEL_1"
selectDf <- seqOtCTypeSummary[iHighest & (iLevel1),]

#variability ordering
var_order <- selectDf %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(
    group_variability = sd(perc, na.rm = TRUE)
  ) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(desc(group_variability))

selectDf$ONCOTREE_CODE <- factor(selectDf$ONCOTREE_CODE, levels=var_order$ONCOTREE_CODE)
outF <-  paste0(outDir,"/cancer_type_Seq_Type_oncokb_level1_ordered_by_variability.pdf")
ggplot(selectDf,aes(x=ONCOTREE_CODE,y=perc))+
  geom_boxplot(alpha=.2)+
  geom_jitter(width=.1,alpha=.6,color="dark green")+
  theme_bw()+
  #facet_grid(level_1~.,scale="free_y",)+
  xlab("Oncotree Code")+
  ylab("perc of patients")+
  ylim(0,85)+
  ggtitle(paste0("Institutional variability in clinically actionable mutation rates"))+
  scale_color_brewer(palette="Set2",drop=FALSE)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 4.5, width = 7)

outF <-  paste0(outDir,"/cancer_type_Seq_Type_oncokb_jitter_by_level.pdf")
ggplot(seqOtCTypeSummary[iHighest & (iLevel1 | iLevel2),],aes(x=ONCOTREE_CODE,y=perc,color=ONCOKB_HIGHEST_LEVEL_SUMMARY))+
  #geom_bar(stat = "identity", position = "stack",alpha=.6) +
  geom_jitter(width=.1,alpha=.8)+
  theme_bw()+
  facet_grid(level_1~.,scale="free_y",)+
  xlab("Oncotree Code")+
  ylab("perc of patients")+
  ylim(0,85)+
  ggtitle(paste0("Oncokb evidence level by cancer type"))+
  scale_color_brewer(palette="Set2",drop=FALSE)+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 8, width = 18)

#############################################
### level1 and level 2 hits by assay type ###
#############################################

iLevel1 <- grepl("1",seqOtCTypeSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY)
iLevel2 <- grepl("2",seqOtCTypeSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY)

level12_hits_per_subj <- oncokbOnly[iLevel1 | iLevel2,] %>% 
  group_by(Tumor_Sample_Barcode,SEQ_ASSAY_ID_mod) %>% 
  dplyr::summarise(n=n()) %>%
  dplyr::group_by(n,SEQ_ASSAY_ID_mod) %>%
  dplyr::summarise(nPerSub=n()) %>%
  dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID" )) 

##################################
### proportion of null entries ###
##################################

# what proportion of each column is null
perSubjVar <- svCompiled[svCompiled$SEQ_ASSAY_ID_mod %in% assaySelect,] %>%
  dplyr::group_by(Tumor_Sample_Barcode) %>% 
  dplyr::filter(dplyr::row_number() == 1) # select a single representative entry per subject
#  dplyr::ungroup() %>%

nullCnts <- data.frame()  
i<-1
colList <- colnames(perSubjVar) 
for (x in colList[-length(colList)]) {
  for (y in unique(perSubjVar$SEQ_ASSAY_ID_mod)) {
    iSeq <- perSubjVar$SEQ_ASSAY_ID_mod == y
    selectCol <- perSubjVar[iSeq,x]
    iNull <- is.na(selectCol) | (selectCol=="")
    
    nullCnts[i,"column"] <- x
    nullCnts[i,"SEQ_ASSAY_ID_mod"] <- y
    nullCnts[i,"proportion_null"] <- mean(iNull)
    i<-i+1
  }
}

nullCntsWide <- nullCnts %>%
  pivot_wider(names_from = column,values_from=proportion_null)
outF <- paste0(outDir,"/null_entry_counts_v1.csv")
write.table(nullCntsWide,outF,row.names=F,quote=F,sep=",")

na_proportions_raw <- perSubjVar %>%
  dplyr::group_by(SEQ_ASSAY_ID_mod) %>% 
  summarise(across(tidyselect::everything(), ~ mean(is.na(.)))) #%>%
  #summarise(dplyr::across(clinical_cols, ~ mean(is.na(.)))) %>%
  #pivot_longer(cols = everything(), names_to = "column", values_to = "proportion")
outF <- paste0(outDir,"/null_entry_counts_v2.csv")
write.table(na_proportions_raw,outF,row.names=F,quote=F,sep=",")



# na_proportions_summary <- seqOtCTypeSummary %>%
#   dplyr::group_by(SEQ_ASSAY_ID_mod) %>% 
#   #summarise(across(tidyselect::everything(), ~ mean(is.na(.)))) %>%
#   summarise(dplyr::across(clinical_cols, ~ mean(is.na(.)))) %>%
#   pivot_longer(cols = everything(), names_to = "column", values_to = "proportion")
# 


nullClinInfo <-  perSubjVar %>%
  dplyr::group_by(SEQ_ASSAY_ID_mod) %>% # SAMPLE_TYPE
  dplyr::summarise(n_patients_total=dplyr::n_distinct(Tumor_Sample_Barcode),
                   CANCER_TYPE=paste0(unique(CANCER_TYPE),collapse=";"),
                   n_Metastatic=sum(SAMPLE_TYPE == "Metastasis"),
                   n_Primary=sum(SAMPLE_TYPE == "Primary"),
                   met_to_primary_ratio=n_Metastatic/n_Primary,
                   n_LocalRecurrence=sum(SAMPLE_TYPE == "LocalRecurance"),
                   perc_LocalRecurrence=n_LocalRecurrence/n_patients_total,
                   meanTumorPurity=mean(TUMOR_PURITY,na.rm=T),
                   anyTumorPurityInfo=any(!is.na(TUMOR_PURITY)),
                   n_NaHeme = sum(SAMPLE_TYPE == "NA or Heme"),
                   perc_NaHeme=n_NaHeme/n_patients_total,
                   median_age=median(AGE_AT_SEQ_REPORT,na.rm=T),
                   n_dead=sum(DEAD == "TRUE" | DEAD == "True",na.rm=T),
                   perc_dead=n_dead/n_patients_total,
                   median_IntSeqToDeath=median(INT_Seq_or_Biopsy_To_Death, na.rm=T),
                   n_StageI=sum(STAGE=="I" | STAGE=="1" | STAGE=="1a" | STAGE=="Ia" | STAGE=="IA" |  STAGE=="Ib" | STAGE=="IB" | STAGE=="1b", na.rm=T),
                   perc_StageI=n_StageI/n_patients_total,
                   n_StageII=sum(STAGE=="II" | STAGE=="2" | STAGE=="2a" | STAGE=="IIa" | STAGE=="IIA" | STAGE=="IIb" | STAGE=="IIB" | STAGE=="2b", na.rm=T),
                   perc_StageII=n_StageII/n_patients_total,
                   n_StageIII=sum(STAGE=="III" | STAGE=="3" | STAGE=="3a" | STAGE=="IIIA" | STAGE=="IIIB" | STAGE=="3B", na.rm=T),
                   perc_StageIII=n_StageIII/n_patients_total,
                   n_StageIV=sum(STAGE=="IV" | STAGE=="M1" | STAGE=="M2" | STAGE=="M3", na.rm=T),
                   perc_StageIV=n_StageIV/n_patients_total) %>%
  dplyr::arrange(desc(n_patients_total))
outF <- paste0(outDir,"/cancer_type_assay_clinical_info.csv")
write.table(seqTypeClinicalInfo,outF,row.names=F,quote=F,sep=",")
