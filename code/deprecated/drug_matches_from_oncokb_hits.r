library(dplyr)
library(tidyr)
library(ggplot2)
library(RColorBrewer)
library(DBI)
library(RSQLite)
library(openFDA)

dbName <- "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite"
baseDir <- "/Users/larsonhogstrom/Documents/balder_analysis/balder/code"
outDir <- "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/underlying_feature_model_plots_v2" 
harmonizedDb <- DBI::dbConnect(RSQLite::SQLite(), dbName)

otCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM otCTypeSummary') 
perPatientOncoKBLevelSorted <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM perPatientOncoKBLevelSorted') 

## what drugs were associated with Level 1 hits
drugCnts <- perPatientOncoKBLevelSorted %>%
  dplyr::group_by(ONCOKB_LEVEL_1,ONCOTREE_CODE,ot_name) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::filter(!ONCOKB_LEVEL_1=="")

# get info openFDA
#apiKey <- "" # key from: https://open.fda.gov/apis/authentication/
openFDA::set_api_key(apiKey)
drugForSearch <- tolower("Afatinib")
drugForSearch <- tolower("Adagrasib")
searchStringFDA <- paste0("openfda.generic_name:",drugForSearch)
res <- openFDA(search = searchStringFDA, 
               endpoint = "drug-drugsfda", 
               limit = 20)
results  <- httr2::resp_body_json(res)$results
flat_df <- map_dfr(results, ~ flatten(data.frame(.x, stringsAsFactors = FALSE)))
flat_df$submissions.application_docs.date

