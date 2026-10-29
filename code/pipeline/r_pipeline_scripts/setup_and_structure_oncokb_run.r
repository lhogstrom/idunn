# Joins observed variants to sample metadata (dropping cell lines) and writes the variant table
# that is submitted to the OncoKB annotator.
# Invoked by workflow/main.nf: Rscript <script> <baseDir> <dbName>

library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Three arguments must be supplied", call. = FALSE)
}

baseDir <- args[1] # should be the relative path to "balder/code" where this file is located
dbName <- args[2]
outVarTable <- args[3]

harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

poVariants<- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM patientObservedVariantTable') %>%
  dplyr::mutate(AAChangeObserved=stringr::str_replace(HGVSp_Short,"p.", ""),
                MAF=100*(t_alt_count / (t_ref_count+t_alt_count)),
                t_alt_plus_ref_count = t_ref_count+t_alt_count)
poVariants$balderVariantID <- seq(1,dim(poVariants)[[1]])
iHartwig <- poVariants$SourceStudy == "Hartwig-data"
poVariants[iHartwig,"MAF"] <- poVariants[iHartwig,"TVAF"]*100

sampleInfoCompiled <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM harmonizedSampleInfo')
sampleInfoCompiled[is.na(sampleInfoCompiled$SAMPLE_TYPE),"sampleInfoCompiled"] <- "Unspecified"

svCompiledPrelim <- poVariants %>%
  dplyr::inner_join(sampleInfoCompiled[,!colnames(sampleInfoCompiled)=="SourceStudy"],
                    by=c("Tumor_Sample_Barcode"="SAMPLE_ID")) %>%
  dplyr::filter(!SAMPLE_TYPE == "Cell line")
print(dim(svCompiledPrelim))


oncoKBCols <- c("Hugo_Symbol",
                "Tumor_Sample_Barcode",
                "HGVSp",
                "HGVSp_Short",
                "ONCOTREE_CODE")
write.table(svCompiledPrelim[,oncoKBCols],outVarTable,row.names=F,quote=F,sep="\t")

