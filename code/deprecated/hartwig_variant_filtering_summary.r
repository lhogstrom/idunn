library(dplyr)
library(tidyr)
library(ggplot2)
library(phenOncoX)
source("R/utils.R")

baseDir <- "."
figDir <- "../../output/hartwig_variant_filtering_20240808"

inFile <- paste0(baseDir,"/../../data/Hartwig/processed/hartwig_sample_pcgr_maf_oncokb_gene_subset.maf")
varDf <- read.csv(inFile,sep="\t")

inFile <- paste0(baseDir,"/../../data/Hartwig/processed/hartwig_sample_pcgr_maf_oncokb_gene_subset_cnt_tbl.csv")
cntDf <- read.csv(inFile,sep="\t")

inFile <- paste0(baseDir,"/../../output/cancer_type_matching_20231220/phenoOncoX_records.txt")
pOncX <- read.csv(inFile,sep="\t")


####################################
### plot variant count summaries ###
####################################

outF <- paste0(figDir,"/hartwig_raw_variant_counts.png")
ggplot(cntDf,aes(x=rawVariantCnt,y=filteredVariantCnt))+
  geom_point(alpha=.1)+
  theme_bw()+
  ggtitle(paste0("Hartwig sample variant count"))+
  scale_color_brewer(palette="Set2")+
  theme(plot.title = element_text(hjust = 0.5))+
  ggsave(outF,height = 6,width = 6)

outF <- paste0(figDir,"/hartwig_raw_variant_counts_log.png")
ggplot(cntDf,aes(x=rawVariantCnt,y=filteredVariantCnt))+
  geom_point(alpha=.1)+
  theme_bw()+
  #facet_wrap(~Hugo_Symbol,scale="free_x")+
  #ylab("")+
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  ggtitle(paste0("Hartwig sample variant count"))+
  scale_color_brewer(palette="Set2")+
  theme(plot.title = element_text(hjust = 0.5))+
  ggsave(outF,height = 6,width = 6)

outF <- paste0(figDir,"/hartwig_raw_variant_counts_filtered.png")
ggplot(cntDf,aes(x=filteredVariantCnt))+
  geom_bar()+
  theme_bw()+
  xlim(0,250)+
  ggtitle(paste0("Hartwig filtered sample variant count"))+
  scale_color_brewer(palette="Set2")+
  theme(plot.title = element_text(hjust = 0.5))+
  ggsave(outF,height = 6,width = 6)

#######################
### plot properties ###
#######################

varClassCnt <- varDf %>%
  dplyr::group_by(Variant_Classification) %>%
  dplyr::summarise(n=n())

outF <- paste0(figDir,"/hartwig_raw_variant_classification.png")
ggplot(varDf,aes(y=Variant_Classification))+
  geom_bar()+
  ggtitle(paste0("Hartwig filtered samples - Variant Classification"))+
  ggsave(outF,height = 6,width = 6)  

# variants without a listed protein change
table(varDf$HGVSp == "")

iNoHgvsp <- varDf$HGVSp == ""
outF <- paste0(figDir,"/hartwig_filtred_vars_without_hgvsp.png")
ggplot(varDf[iNoHgvsp,],aes(y=Variant_Classification))+
  geom_bar()+
  ggtitle(paste0("Hartwig filtered samples - without HGVSp effect"))+
  ggsave(outF,height = 6,width = 6)  

#######################
### check cancer type properties ###
#######################

### load hartwig data
hartwigDir <- paste0(baseDir,"/../../data/Hartwig/data")
inFile <- paste0(hartwigDir,"/samples.txt")
sampleList <- read.csv(inFile,sep=",", header = F)

inFile <- paste0(hartwigDir,"/metadata.tsv")
hMeta <- read.csv(inFile,sep="\t")

# which samples have at least one valid vcf and pcgr entry
hMetaSelect <- hMeta[hMeta$sampleId %in% varDf$sample,]

sampleCntsPerSubj <- hMetaSelect %>%
  dplyr::group_by(hmfPatientId) %>%
  dplyr::summarise(n=n())
table(sampleCntsPerSubj$n)

# select an arbitrary sample per subject
hMetaSampleSelect <- hMetaSelect %>%
  dplyr::group_by(hmfPatientId) %>%
  dplyr::filter(dplyr::row_number()==1)
  
#### create table to prep for oncoTree assignments
cTypeTble <- hMetaSampleSelect %>%
  dplyr::group_by(primaryTumorLocation,primaryTumorSubLocation,primaryTumorType,primaryTumorSubType) %>%
  dplyr::summarise(numberOfSubjects=n()) %>%
  dplyr::arrange(desc(numberOfSubjects))
outF <- paste0(figDir,"/hartwig_cancer_type_info_table.txt")
write.table(cTypeTble,outF,row.names=F,quote=F,sep="\t")

cTypeTbleFull <- hMetaSampleSelect %>%
  dplyr::group_by(primaryTumorLocation,primaryTumorSubLocation,primaryTumorType,primaryTumorSubType,primaryTumorExtraDetails) %>%
  dplyr::summarise(numberOfSubjects=n()) %>%
  dplyr::arrange(desc(numberOfSubjects))
outF <- paste0(figDir,"/hartwig_cancer_type_info_table_full.txt")
write.table(cTypeTbleFull,outF,row.names=F,quote=F,sep="\t")


####################
## phenOncoX load ##
####################

### load phenOncoX
download_dir <- tempdir()
oncoterms <- phenOncoX::get_terms(
  cache_dir = download_dir)
#outFile <- paste0(figDir,"/phenoOncoX_records.txt")
#write.table(oncoterms$records,outFile,sep="\t",row.names = F)

sourceTermList <- unique(cTypeTbleFull$primaryTumorLocation)

pOncEntries <- get_representative_phenOncoX_record(oncoterms, sourceTermList)
pOncEntriesSelect <- pOncEntries[pOncEntries$stringTypeApproach=="modifiedSourceString" & 
                                   pOncEntries$recordMatchApproach=="substringMatch" &
                                   pOncEntries$searchMatchType=="MaxHit",]

### join on phenOnX records
cTypeTblePOX <- cTypeTbleFull %>%
  dplyr::left_join(pOncEntriesSelect,
                   by=c("primaryTumorLocation"="queryTerm"))#,
                   #relationship="one-to-one")
  
outF <- paste0(figDir,"/hartwig_cancer_type_info_table_with_phenoOncoX_entries.txt")
write.table(cTypeTblePOX,outF,row.names=F,quote=F,sep="\t")

