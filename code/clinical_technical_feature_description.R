library(magick)

### exploratory plots by relationship
ctCnts <- train_data %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(nSubjectOTcode=n()) %>%
  dplyr::arrange(desc(nSubjectOTcode))

ftrList <- c("SAMPLE_TYPE",
  "RACE",
  "ETHNICITY",
  "is_paired_end",
  "library_selection",
  "platform",
  "calling_strategy",
  "preservation_technique")

for (ftr in ftrList) {
  fCnts <- train_data %>%
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
                              " p-adjusted=",round(pFDR,4))) 
  iMatch <- fCntsWideJ$pValue <= 0.05 & ! is.na(fCntsWideJ$pValue)
  fCntsWideJ[iMatch,"strFull"] <- fCntsWideJ[iMatch,"str2"]
  
  
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
    scale_fill_brewer(palette="Set2",drop=FALSE)
    theme_minimal() #+
  ggsave(outF1,height = 49, width = 12)
  
  # plot of ratios
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
    scale_fill_brewer(palette="Set2",drop=FALSE)
  theme_minimal() #+
  ggsave(outF2,height = 49, width = 12)
  
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
  
