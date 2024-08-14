
#' Given a data frame, provide the counts of groupings acording to
#' two columns. Also provide the count percentage relative to all entries.
#'
#' @param df Input data frame that contains to the two columns of interest
#' @param group_col1 string of first column name 
#' @param group_col2 string of second column name
#'
#' @return
#' @export
#'
#' @examples
two_group_count_percentage <- function(df, group_col1, group_col2) {
  n <- dim(df)[[1]]
  df %>%
    group_by(!!sym(group_col1), !!sym(group_col2)) %>%
    summarise(count = n()) %>%
    mutate(percentage = (count / n) * 100, 
           counts_and_percent = paste0(count, " (", round(percentage, 1), "%)")) %>%
    ungroup()
}


#' Get representative phenOncoX ID for each source term
#'
#' @param oncoterms phenOncoX term records
#' @param sourceTermList list of input query terms
#'
#' @return
#' @export
#'
#' @examples
get_representative_phenOncoX_record <- function(oncoterms, sourceTermList) {
  # define name columns
  nameCols <- c("primary_site",
                "ot_main_type",
                "ot_name",
                "ot_code",
                "cui_name",
                "efo_name",
                "do_name")
  
  stringTypeList <- c("origSourceString","modifiedSourceString")
  recordMatchList <- c("exactStringMatch","substringMatch")
  
  #stringTypeList <- c("modifiedSourceString")
  #recordMatchList <- c("substringMatch")
  
  dfRepresenativeSets <- data.frame()
  for (stringType in stringTypeList) {
    for (recordMatch in recordMatchList) {
      for (sourceTermOrig in sourceTermList) {
        print(sourceTermOrig)
  
        # define and modify source term
        sourceTerm <- tolower(sourceTermOrig)
        sourceTerm <- gsub(" cancer", "", sourceTerm)
        sourceTerm <- gsub(" disease", "", sourceTerm) # other terms to consider: “recurrent” or “malignant” or “adult”
  
        ### Create matrix of boolean matches
        oncotermNames <- oncoterms$records[,nameCols]
        queryMatch <- oncotermNames
        # for each column convert to lower
        for (xCol in nameCols) {
          resRecordsOrig <- oncotermNames[,xCol]
          resRecords <- tolower(resRecordsOrig)
          resRecords <- gsub("_", " ", resRecords)
          # remove phrases
          oncotermNames[,xCol] <- resRecords
  
          if (stringType == "origSourceString" & recordMatch == "substringMatch") {
            queryMatch[,xCol] <- grepl(sourceTermOrig,resRecordsOrig) # is it a substring
          }
  
          if (stringType == "modifiedSourceString" & recordMatch == "substringMatch") {
            queryMatch[,xCol] <- grepl(sourceTerm,resRecords) # is it a substring
          }
  
          if (stringType == "origSourceString" & recordMatch == "exactStringMatch") {
            queryMatch[,xCol] <- sourceTermOrig == resRecordsOrig # require exact match
          }
  
          if (stringType == "modifiedSourceString" & recordMatch == "exactStringMatch") {
            queryMatch[,xCol] <- sourceTerm == resRecords # require exact match
          }
  
          # NOTE: what is the highest resoultion you can have with each database? List average/distirbuion OT level for each DB.
        }
  
        queryCols <- paste0("query_match_",nameCols)
        colnames(queryMatch) <- queryCols
  
        # provide index
        oncoterms$records$recordID <- seq(1,dim(oncoterms$records)[[1]])
        queryMatch$recordID <- seq(1,dim(queryMatch)[[1]])
  
        # join terms with query results
        oncoQueryRes <- oncoterms$records %>%
          dplyr::left_join(queryMatch,by="recordID")
        oncoQueryRes$queryHits <- rowSums(oncoQueryRes[,queryCols])
        oncoQueryRes$queryTerm <- sourceTermOrig
  
  
        oncoQueryRes$queryHitRepresentative <- F
        repEntryMaxHit <- oncoQueryRes %>%
          dplyr::filter(queryHits > 0) %>%
          dplyr::filter(queryHits == max(queryHits)) %>%
          dplyr::filter(dplyr::row_number()==1) %>%
          mutate(queryHitRepresentative=T,
                 searchMatchType="MaxHit",
                 stringTypeApproach=stringType,
                 recordMatchApproach=recordMatch)
        dfRepresenativeSets <- rbind(dfRepresenativeSets,repEntryMaxHit)
  
        repEntryMaxLevel <- oncoQueryRes %>%
          dplyr::filter(queryHits > 0) %>%
          dplyr::arrange(desc(ot_level)) %>%
          dplyr::filter(dplyr::row_number()==1) %>%
          mutate(queryHitRepresentative=T,
                 searchMatchType="MaxLevel",
                 stringTypeApproach=stringType,
                 recordMatchApproach=recordMatch)
        dfRepresenativeSets <- rbind(dfRepresenativeSets,repEntryMaxLevel)
      }
    }
  }
  return(dfRepresenativeSets)
}



