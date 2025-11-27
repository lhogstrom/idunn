library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(ggplot2)
library(reshape2)
library(RColorBrewer)


outDir <- "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/demographic_technical_ml_modeling"
inSFile <- paste0(outDir,"/mean_shap_values_by_feature_rank.txt")
shapFtRank <- read.csv(inSFile,sep="\t")

dbName <- "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite"
baseDir <- "/Users/larsonhogstrom/Documents/balder_analysis/balder/code"
outDir <- "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/underlying_feature_model_plots_v2" 
#modelFeatureOutDir <- paste0(outDir,"/underlying_feature_model_plots")
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

#seqOtCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM seqOtCTypeSummary') 
otCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM otCTypeSummary') 
perPatientOncoKBLevelSorted <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM perPatientOncoKBLevelSorted') 


## counts
cTypeLvl1Cnts <- perPatientOncoKBLevelSorted %>%
  dplyr::filter(isOncoKbLevel1 == 1) %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(n_Level1_patients=n()) %>%
  dplyr::arrange(desc(n_Level1_patients))
lvl1Thresh <- 250
cTypeThresh <- cTypeLvl1Cnts[cTypeLvl1Cnts$n_Level1_patients >= lvl1Thresh,]

cTypeRep <- perPatientOncoKBLevelSorted %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::filter(dplyr::row_number()==1)

cTypeCnts <- perPatientOncoKBLevelSorted %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(n_patients=n(),
  ) %>%
  dplyr::arrange(desc(n_patients))


cTypeCntTbl <- cTypeThresh %>%
  dplyr::left_join(cTypeRep[,c("ONCOTREE_CODE","CANCER_TYPE","CANCER_TYPE_DETAILED")],by=c("ONCOTREE_CODE")) %>%
  dplyr::left_join(cTypeCnts,by=c("ONCOTREE_CODE")) %>%
  dplyr::mutate(percActionable = 100*n_Level1_patients/n_patients,
                actionableEvents = paste0(n_Level1_patients, " (", round(percActionable, 1), "%)"))

####

print("subjects without a calling stategy value:")
print(table(is.na(perPatientOncoKBLevelSorted$calling_strategy)))

print(table(perPatientOncoKBLevelSorted$calling_strategy))

iMatchCtype <- perPatientOncoKBLevelSorted$ONCOTREE_CODE %in% cTypeThresh$ONCOTREE_CODE
iUnknown <- perPatientOncoKBLevelSorted$SEX == "Unknown" | perPatientOncoKBLevelSorted$SEX == "Other"
lvlEvents <- perPatientOncoKBLevelSorted[iMatchCtype & !iUnknown,]
lvlEvents$actionabilityStr <- "no Level 1 event"
lvlEvents[lvlEvents$isOncoKbLevel1==1,"actionabilityStr"] <- "Level 1 event"
lvlEvents$actionabilityStr2 <- "Level1Event"
lvlEvents[!lvlEvents$isOncoKbLevel1==1,"actionabilityStr2"] <- "NoLevel1Event"

# ### look at the impact of technical factors
tmp <- shapFtRank[shapFtRank$feature=="calling_strategy",]
tmp$ONCOTREE_CODE[1:6]

lvlTechnical <- lvlEvents %>%
  dplyr::group_by(ONCOTREE_CODE,actionabilityStr2,calling_strategy) %>%
  dplyr::summarise(n=n()) 

lvlTechnical$ONCOTREE_CODE <- factor(lvlTechnical$ONCOTREE_CODE, levels = tmp$ONCOTREE_CODE)
outF <-  paste0(outDir,"/calling_stategy_by_actionability.pdf")
ggplot(lvlTechnical,aes(x=actionabilityStr2,y=n,fill=calling_strategy))+
  geom_bar(stat = "identity") + 
  coord_flip() +
  facet_grid(ONCOTREE_CODE~.,scale="free") 
ggsave(outF,height = 15, width = 9)  

lvlProportion <- lvlTechnical %>%
  dplyr::group_by(ONCOTREE_CODE,actionabilityStr2) %>%
  dplyr::mutate(proportion=n/sum(n))

outF <-  paste0(outDir,"/calling_stategy_proportion_by_actionability.pdf")
ggplot(lvlProportion,aes(x=actionabilityStr2,y=proportion,fill=calling_strategy))+
  geom_bar(stat = "identity") + 
  coord_flip() +
  facet_grid(ONCOTREE_CODE~.,scale="free") 
ggsave(outF,height = 15, width = 6)  

#### proportional center plots - librrary selection 
topEntries <- shapFtRank[shapFtRank$feature=="library_selection",]
lvlTechCntr <- lvlEvents %>%
  dplyr::filter(ONCOTREE_CODE %in% topEntries$ONCOTREE_CODE[1:3]) %>%
  dplyr::group_by(CENTER,ONCOTREE_CODE,library_selection) %>%
  dplyr::summarise(n=n(),
                   propLevel1=sum(isOncoKbLevel1)/n) 

outF <-  paste0(outDir,"/library_selection_proportion_by_actionability.pdf")
ggplot(lvlTechCntr,aes(x=ONCOTREE_CODE,y=propLevel1,color=library_selection,size=n))+
  geom_violin(scale="count") +
  geom_jitter(height=0,width=.2,alpha=.8) +
  scale_color_brewer(palette="Set2",drop=FALSE)+
  theme_bw()
ggsave(outF,height = 4, width = 6)  

# reported stat of center & seq ID:
print("Specifically, the sequencing center or assay ID were found to be one of the top 4 features in N of the 21 models constructed.")
iTop4 <- shapFtRank$rank <= 4
iSid <- shapFtRank$feature == "SEQ_ASSAY_ID"
iCenter <- shapFtRank$feature == "CENTER"
sIDMatch <- shapFtRank[iTop4 & (iSid | iCenter),]
print(length(unique(sIDMatch$ONCOTREE_CODE)))

# reported stat on ancestry:
print("The effect of ancestry was especially prominent in X and Y cancer types:")
iTop4 <- shapFtRank$rank <= 2
iRace <- shapFtRank$feature == "RACE"
iEthnicity <- shapFtRank$feature == "ETHNICITY"
sRaceMatch <- shapFtRank[iTop4 & (iRace | iEthnicity),]
print(length(unique(sIDMatch$ONCOTREE_CODE)))


#### proportional center plots
topEntries <- shapFtRank[shapFtRank$feature=="calling_strategy",]
lvlTechCntr <- lvlEvents %>%
  dplyr::filter(ONCOTREE_CODE %in% topEntries$ONCOTREE_CODE[1:3]) %>%
  dplyr::group_by(CENTER,ONCOTREE_CODE,calling_strategy) %>%
  dplyr::summarise(n=n(),
                   propLevel1=sum(isOncoKbLevel1)/n) 


outF <-  paste0(outDir,"/calling_stategy_proportion_by_actionability.pdf")
ggplot(lvlTechCntr,aes(x=ONCOTREE_CODE,y=propLevel1,color=calling_strategy,size=n))+
  geom_violin(scale="count") +
  geom_jitter(height=0,width=.2,alpha=.8) +
  scale_color_brewer(palette="Set1",drop=FALSE)+
  theme_bw()
ggsave(outF,height = 4, width = 6)  
 
#### cancer type and assay ID counts
cntrCnts <- lvlEvents %>%
  dplyr::group_by(CENTER) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::arrange(desc(n))

print("proportion patients accounted by top 3 instittuitons:")
print(sum(cntrCnts$n[1:3])/sum(cntrCnts$n))

assayCnts <- lvlEvents %>%
  dplyr::group_by(CENTER,SEQ_ASSAY_ID) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::arrange(desc(n))

top3CntrAssay <- assayCnts[assayCnts$CENTER %in% cntrCnts$CENTER[1:3], ]
print("the three institutions collectively utilized nsequencing assays with unique assay IDs: ")
print(dim(top3CntrAssay)[1])

print("there were a total of Y unique assay IDs included in the study:")
print(dim(assayCnts)[1])

print("n assays with 5 or more subjects:")
table(assayCnts$n > 5)
