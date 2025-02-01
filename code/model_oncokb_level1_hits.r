library(dplyr)
library(tidyr)
library(DBI)
library(RSQLite)
library(ggplot2)
library(corrplot)
library(reshape2)
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)

library(circlize)
#library(devtools)
#install_github("jokergoo/ComplexHeatmap")
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
outDir <- paste0(baseDir,"/../../output/actionability_db_curration_20250121/oncokb_summary_and_modeling")
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

otCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM otCTypeSummary') 
seqTypeClinicalInfo <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM seqTypeClinicalInfo') 
seqOtCTypeSummary <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM seqOtCTypeSummary') 
assayInfo <- RSQLite::dbGetQuery(harmonizedDb, 'SELECT * FROM assayInfoGENIE') 

#################################
### summary of oncokb results ###
#################################

### find oncokb matched variants
onckbAnnotated <- svCompiled %>%
  dplyr::filter(ONCOKB_IS_SENS_DX_PX_OR_RESISTANCE == T)

## which variants are onckb annotated but not annotated via MIA/CiVIC? 
iOnckbOnlyGrp <- onckbAnnotated$balderVariantID %in% aa.genome.representative$balderVariantID
okBID <- onckbAnnotated[!iOnckbOnlyGrp,"balderVariantID"]
oncokbOnly <-svCompiled[svCompiled$balderVariantID %in% okBID,]  

###########################
### oncokb level counts ###
###########################

highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:10]
iHighest <- otCTypeSummary$ONCOTREE_CODE %in% as.character(highest_c_counts)

############################
### exploratory plotting ###
############################

###########################
### feature engineering ###
###########################

# # Handle missing values
# preProcess_missingdata <- preProcess(data, method = 'knnImpute')
# data_imputed <- predict(preProcess_missingdata, newdata = data)
# 
# # Scale and center numeric variables
# preProcess_scale <- preProcess(data_imputed[, sapply(data_imputed, is.numeric)], method = c("center", "scale"))
# data_imputed[, sapply(data_imputed, is.numeric)] <- predict(preProcess_scale, newdata = data_imputed[, sapply(data_imputed, is.numeric)])
# 
# # Convert categorical variables to dummy variables
# dummies_model <- dummyVars(ResponseVar ~ ., data = data_imputed)
# data_dummies <- predict(dummies_model, newdata = data_imputed)
# 
# # Combine the response variable back with the transformed predictors
# data_final <- cbind(data_dummies, ResponseVar = data_imputed$ResponseVar)
# 
# # Convert to a data frame for further processing
# data_final <- as.data.frame(data_final)
# 
# # Display the final processed dataset
# print(data_final)

###############################################
### modeling level1 hits by sequencing site ###
###############################################

# model which sequence facility factors have the biggest impact on proportion of level1 variants per cancer type

#library(mlflow)

highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:15]
iHighest <- seqOtCTypeSummary$ONCOTREE_CODE %in% as.character(highest_c_counts)
iLevel1 <- grepl("LEVEL_1",seqOtCTypeSummary$ONCOKB_HIGHEST_LEVEL_SUMMARY)

clinical_cols <- c("median_age",
                   "met_to_primary_ratio",
                   "meanTumorPurity",
                   "anyTumorPurityInfo",
                   "perc_LocalRecurrence",
                   "perc_NaHeme",
                   "perc_dead",
                   "perc_StageI",
                   "perc_StageII",
                   "perc_StageIII",
                   "perc_StageIV",
                   "median_IntSeqToDeath")
technical_cols <- c("is_paired_end",
                    "library_selection",
                    "platform",
                    "number_of_genes",
                    "calling_strategy",
                    "preservation_technique")

modelCols <- c("perc",
               "ONCOTREE_CODE",
               "SEQ_ASSAY_ID_mod",
               "ONCOKB_HIGHEST_LEVEL_SUMMARY",
               clinical_cols,
               technical_cols)

data <- seqOtCTypeSummary[iHighest & iLevel1,] %>%
  #dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID" )) %>%
  dplyr::select(all_of(modelCols)) %>%
  dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::arrange(desc(perc)) %>% 
  dplyr::mutate(actionable_perc_rank=dplyr::row_number())

# Convert categorical variables to factors (if not already factors)
data$ONCOTREE_CODE <- factor(data$ONCOTREE_CODE, levels=unique(data$ONCOTREE_CODE))
data$is_paired_end <- as.factor(data$is_paired_end)
data$library_selection <- as.factor(data$library_selection)
data$platform <- as.factor(data$platform)
data$calling_strategy <- as.factor(data$calling_strategy)
data$preservation_technique <- as.factor(data$preservation_technique)

# Split the data into training and testing sets
set.seed(42)
train_index <- caret::createDataPartition(data$perc, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# recursive binary splitting 
# splitPoints <- data.frame() # will contain the split info: 1) variable and 2) split point
# stop <- F
# x <- 10
# max_region_member_count <- 5
# while (stop==F) {
#   x <- x-1
#   print(x)
#   # propose a new split
#   tmpRssScores <- data.frame()
#   for (testCol in colsToTest) {
#     splitPoints <- unique(train_data[,testCol])
#     for (tSplit in splitPoints) {
#       iStr <- paste0(testCol,"-",tSplit)
#       tmpRssScores[iStr,"test col"] <- testCol
#       tmpRssScores[iStr,"split point"] <- tSplit
#       # check the current RSS of a proposed split
#       tmpRssScores[iStr,"RSS"] <- testCol
#     }
#   }
#   # identify top RSS score
#   
#   # remove from as possible split point in next iteration
#   
#   # check the max membership count of current regions
#   if (x <= max_region_member_count) {
#     stop <- T
#   }
# }


# Define the formula for the model
#formula <- perc ~ ONCOTREE_CODE + is_paired_end + library_selection + preservation_technique
formula <- actionable_perc_rank ~ is_paired_end + library_selection + preservation_technique

# Train the Decision Tree Regressor
tree_model <- rpart(formula, data = train_data, method = "anova")

# Print the model summary
print(summary(tree_model))

# Visualize the tree
rpart.plot(tree_model)

# Extract the rules (cut points) from the tree
tree_rules <- as.data.frame(tree_model$frame)
tree_rules_no_leaves <- tree_rules[tree_rules$var != "<leaf>", ]  # Exclude leaf nodes

# Print the rules
print(tree_rules_no_leaves)

# plot model
plot(tree_model)
text(tree_model, pretty = 0)

# Predict on the test data
predictions <- predict(tree_model, newdata = test_data)

# Evaluate the model performance
mse <- mean((predictions - test_data$perc)^2)
cat("Mean Squared Error (MSE):", mse, "\n")

###########################################
### chi-sq test for categorical columns ###
###########################################

# feature analysis 

seqOtFull <- seqOtCTypeSummary[iHighest & iLevel1,] %>%
  dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID" )) 


# 1. Correlation matrix and heatmap for numeric variables
# numeric_vars <- df[, sapply(df, is.numeric)]
# cor_matrix <- cor(numeric_vars, use = "complete.obs")
# corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# 2. Chi-squared test of independence for categorical variables
aInfo <- data.frame(lapply(assayInfo, as.factor))
cSelect <- colnames(aInfo)
cSelect <- cSelect[!cSelect %in% c("SEQ_ASSAY_ID")]
categorical_vars <- aInfo[, cSelect]
chi_sq_matrix <- matrix(NA, ncol = ncol(categorical_vars), nrow = ncol(categorical_vars))
colnames(chi_sq_matrix) <- colnames(categorical_vars)
rownames(chi_sq_matrix) <- colnames(categorical_vars)

for (i in 1:ncol(categorical_vars)) {
  for (j in 1:ncol(categorical_vars)) {
    if (i != j) {
      chi_sq_matrix[i, j] <- chisq.test(categorical_vars[, i], categorical_vars[, j])$p.value
    } else {
      chi_sq_matrix[i, j] <- NA
    }
  }
}

chi_sq_melt <- melt(chi_sq_matrix, na.rm = TRUE)
colnames(chi_sq_melt) <- c("Var1", "Var2", "value")
chi_sq_melt$log_value <- -log10(chi_sq_melt$value)

# Plot chi-squared test results as heatmap
outF <-  paste0(outDir,"/sequencing_platoform_categorical_variables_chi_sq.pdf")
ggplot(data = chi_sq_melt, aes(x = Var1, y = Var2, fill = log_value)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "red", na.value = "grey50", name = "-log10 p-value") +
  #scale_fill_gradient(low = "red", high = "white", na.value = "grey50", name = "p-value") + # , trans = "reverse"
  theme_minimal() +
  labs(title = "Chi-squared Test of Independence for Categorical Variables", x = "Variables", y = "Variables") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(outF,height = 10, width = 10)

############################################
### corr structure of clinical variables ###
############################################

clin_train <- train_data[,clinical_cols]

# Compute Pearson correlation matrix
pearson_corr <- cor(clin_train, 
                    use = "pairwise.complete.obs",
                    method = "pearson")

# Visualize the Pearson correlation matrix
outF <-  paste0(outDir,"/clinical_factor_pairwise_pearson_corr.pdf")
pdf(outF, width = 10, height = 10)
corrplot(pearson_corr, method = "color", type = "upper", 
         addCoef.col = "black", tl.col = "black", tl.srt = 45, 
         title = "Pearson Correlation Matrix", mar = c(0,0,1,0))
dev.off()

# Compute spearman correlation matrix
spearman_corr <- cor(clin_train, 
                     #use = "pairwise.complete.obs",
                     method = "spearman")

# Visualize the Pearson correlation matrix
outF <-  paste0(outDir,"/clinical_factor_pairwise_spearman_corr.pdf")
pdf(outF, width = 10, height = 10)
corrplot(spearman_corr, method = "color", type = "upper", 
         addCoef.col = "black", tl.col = "black", tl.srt = 45, 
         title = "Spearman Correlation Matrix", mar = c(0,0,1,0))
dev.off()

### proportion of null values
# na_proportions <- train_data %>%
#   summarise_all(~ mean(is.na(.))) %>%
#   gather(key = "column", value = "proportion")

na_proportions <- train_data[,clinical_cols] %>%
  #summarise(across(tidyselect::everything(), ~ mean(is.na(.)))) %>%
  summarise(dplyr::across(clinical_cols, ~ mean(is.na(.)))) %>%
  pivot_longer(cols = everything(), names_to = "column", values_to = "proportion")

##################################################### 
### perform per-patient modeling of level1 status ### 
##################################################### 

# create representative variant per row per subject - choose level 1 variant if avaible 
perPatientLevel1 <- svCompiled %>%
  dplyr::rename(SequenceCenter = Center) %>%
  dplyr::left_join(assayInfo,by=c("SEQ_ASSAY_ID_mod"="SEQ_ASSAY_ID")) %>%
  dplyr::mutate(isOncoKbLevel1=ONCOKB_HIGHEST_LEVEL_SUMMARY=="LEVEL_1" & !is.na(ONCOKB_HIGHEST_LEVEL_SUMMARY)) %>%
  dplyr::group_by(Tumor_Sample_Barcode) %>%
  dplyr::arrange(desc(isOncoKbLevel1)) %>%
  dplyr::filter(dplyr::row_number()==1) %>%
  dplyr::ungroup()
outF <- paste0(outDir,"/patient_level_OncoKB_status_for_modeling_head.txt")
write.table(head(perPatientLevel1,1000),outF,row.names=F,quote=F,sep="\t")  
RSQLite::dbWriteTable(harmonizedDb, "perPatientLevel1", perPatientLevel1, overwrite =T)

# make data subset selections
#highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:5]
highest_c_counts <- unique(otCTypeSummary$ONCOTREE_CODE)[1:63] # select those OT codes with 500 or more samples
highest_c_counts <- highest_c_counts[!is.na(highest_c_counts)]
iHighest <- perPatientLevel1$ONCOTREE_CODE %in% as.character(highest_c_counts)

# cancer type and subtype name summary
cTypeSubtypeCnts <- perPatientLevel1 %>% #[iHighest,]
  dplyr::group_by(ONCOTREE_CODE,CANCER_TYPE,CANCER_TYPE_DETAILED) %>%
  dplyr::summarise(nPatients=n()) %>%
  dplyr::arrange(ONCOTREE_CODE,desc(nPatients))
outF <- paste0(outDir,"/cancer_type_and_subtype_counts.txt")
write.table(cTypeSubtypeCnts,outF,row.names=F,quote=F,sep="\t")  

modelCols <- c("isOncoKbLevel1",
               "ONCOTREE_CODE",
               "SAMPLE_TYPE",
               "AGE_AT_SEQ_REPORT",
               "RACE",
               "ETHNICITY",
               "is_paired_end",
               "library_selection",
               "platform",
               "number_of_genes",
               "calling_strategy",
               "preservation_technique")

# select top cancers and subset columns
data <- perPatientLevel1[iHighest,] %>%
  dplyr::select(all_of(modelCols))
# exclude patients if any data entry is null
iNull <- rowSums(!is.na(data[,modelCols])) != length(modelCols) 
data <- data[!iNull,]

# Convert categorical variables to factors (if not already factors)
data$isOncoKbLevel1 <- factor(data$isOncoKbLevel1)
data$ONCOTREE_CODE <- factor(data$ONCOTREE_CODE, levels=unique(data$ONCOTREE_CODE))
data$is_paired_end <- as.factor(data$is_paired_end)
data$library_selection <- as.factor(data$library_selection)
data$platform <- as.factor(data$platform)
data$calling_strategy <- as.factor(data$calling_strategy)
data$preservation_technique <- as.factor(data$preservation_technique)

# Split the data into training and testing sets
set.seed(42)
train_index <- caret::createDataPartition(data$isOncoKbLevel1, p = 0.3, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Define the formula for the model
formula <- isOncoKbLevel1 ~ ONCOTREE_CODE + SAMPLE_TYPE + AGE_AT_SEQ_REPORT + RACE + ETHNICITY + is_paired_end + library_selection + preservation_technique

# Train the Decision Tree Regressor
tree_model <- rpart(formula, data = train_data, method = "class")

# Print the model summary
print(summary(tree_model))

# Visualize the tree
rpart.plot(tree_model)

# Extract the rules (cut points) from the tree
tree_rules <- as.data.frame(tree_model$frame)
tree_rules_no_leaves <- tree_rules[tree_rules$var != "<leaf>", ]  # Exclude leaf nodes

# Print the rules
print(tree_rules_no_leaves)

# plot model
#plot(tree_model)
#text(tree_model, pretty = 0)

# Predict on the test data
predictions <- predict(tree_model, newdata = test_data)

# Evaluate the model performance
mse <- mean((predictions - test_data$perc)^2)
cat("Mean Squared Error (MSE):", mse, "\n")

### repeat with logistic regression
# library(Amelia)
# missmap(train_data, main = "Missing values vs observed")

model <- glm(isOncoKbLevel1 ~.,
             family=binomial(link='logit'),
             data=train_data)
summary(model)

# library(broom)
# # Tidy and plot
# tidy_model <- tidy(model, conf.int = TRUE)
# ggplot(tidy_model, aes(x = term, y = estimate)) +
#   geom_point() +
#   geom_errorbar(aes(ymin = conf.low, ymax = conf.high)) +
#   coord_flip()

# make roc curve


# Fit model and predict probabilities
prob <- predict(model, type = "response")

# Plot ROC
original_par <- par(no.readonly = TRUE)
roc_curve <- roc(train_data$isOncoKbLevel1, prob)
plot(roc_curve)
par(original_par)

#pred <- ifelse(prob > 0.5, 1, 0)
pred <- ifelse(prob > 0.4, 1, 0)
table(pred,train_data$isOncoKbLevel1)
binLevel1 <- as.integer(train_data$isOncoKbLevel1) - 1 # convert boolean to 0/1
cm <- confusionMatrix(factor(pred), factor(binLevel1))

# Print the confusion matrix
print(cm)


# Assess the relationship between each predictor and the probability of the outcome using pdp package.
# library(pdp)
# # Partial dependence plot for predictor1
# pdp_model <- pdp::partial(model, pred.var = "is_paired_end")
# plotPartial(pdp_model)

### try building model for individual cancer types. Control for subtypes
modelResCtype <- data.frame()
for (cType in unique(train_data$ONCOTREE_CODE)) {
  print(cType)
  cTypeModelCols <- modelCols[!modelCols %in% c("ONCOTREE_CODE")]
  cTypeTrain <- train_data[train_data$ONCOTREE_CODE==cType,cTypeModelCols]
  print(dim(cTypeTrain))
  
  model <- glm(isOncoKbLevel1 ~.,
               family=binomial(link='logit'),
               data=cTypeTrain)
  print(summary(model))
  
  # # Tidy and plot
  # tidy_model <- tidy(model, conf.int = TRUE)
  # tidy_model$ONCOTREE_CODE <- cType
  # modelResCtype <- rbind(modelResCtype,tidy_model)
  # 
  # # plot
  # outF <-  paste0(outDir,"/","logit_model_by_cancer_type_LR_coef_",cType,".pdf")
  # ggplot(tidy_model, aes(x = term, y = estimate)) +
  #   geom_point() +
  #   geom_errorbar(aes(ymin = conf.low, ymax = conf.high)) +
  #   coord_flip()
  # ggsave(outF,height = 7, width = 10)
  
  # make roc curve
  
  # Fit model and predict probabilities
  prob <- predict(model, type = "response")
  
  # Plot ROC
  roc_curve <- roc(cTypeTrain$isOncoKbLevel1, prob)
  
  outF <-  paste0(outDir,"/","logit_model_by_cancer_type_ROC_curve_",cType,".pdf")
  pdf(outF)
  plot(roc_curve)
  dev.off() 
  
  # plot underlying relationships
  #
  cTypeTrain$isOncoKbLevel1_string <- "Patient has Level1 mutation"
  cTypeTrain[cTypeTrain$isOncoKbLevel1==F,"isOncoKbLevel1_string"] <- "No Level1 mutation"
  for (f in colnames(cTypeTrain)) {
    
    outF <-  paste0(modelFeatureOutDir,"/",cType,"_tain_data_",f,"_by_actionable_status.pdf")
    ggplot(cTypeTrain, aes(x = .data[[f]])) +
      geom_bar()+
      facet_grid(isOncoKbLevel1_string ~ .,scale="free_y")+
      labs(title = paste0("training data\n",cType)) +
      theme_minimal()+
      theme(axis.text.x = element_text(angle = 90, hjust = 1))+
      theme(plot.title = element_text(hjust = 0.5))
    ggsave(outF,height = 7, width = 7)
    
    # put bars side-by-side
    outF <-  paste0(modelFeatureOutDir,"/",cType,"_tain_data_",f,"_by_actionable_status_side_by_side.pdf")
    ggplot(cTypeTrain, aes(x = .data[[f]],fill=isOncoKbLevel1_string)) +
      geom_bar(position = "dodge",alpha=.6)+
      labs(title = paste0("training data\n",cType)) +
      theme_minimal()+
      theme(axis.text.x = element_text(angle = 90, hjust = 1))+
      theme(plot.title = element_text(hjust = 0.5))
    ggsave(outF,height = 7, width = 7)
    
  }
  
}

modelResCtype$pLessThanP01 <- modelResCtype$p.value <= 0.01
outF <-  paste0(outDir,"/logit_model_by_cancer_type_LR_scores.pdf")
#ggplot(modelResCtype[!modelResCtype$ONCOTREE_CODE=="PAAD",], aes(x = estimate, y = term, shape=pSignificant,colour = ONCOTREE_CODE)) +
ggplot(modelResCtype[,], aes(x = estimate, y = term, shape=pLessThanP01,colour = ONCOTREE_CODE)) +
  #geom_point() +
  geom_jitter(height=.2)
labs(title = "logistic regression modeling of Level 1 actionability",
     x = "LR score",
     y = "covariate") +
  #theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 10, width = 15)

outF <-  paste0(outDir,"/logit_model_by_cancer_type_LR_scores_trimmed_LR.pdf")
ggplot(modelResCtype[!modelResCtype$ONCOTREE_CODE=="PAAD",], aes(x = estimate, y = term, shape=pLessThanP01,colour = ONCOTREE_CODE)) +
  geom_point() +
  labs(title = "logistic regression modeling of Level 1 actionability \n trimmed LRs ",
       x = "LR score",
       y = "covariate") +
  xlim(-3,3)+
  #theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 10, width = 15)


outF <-  paste0(outDir,"/logit_model_by_cancer_type_LR_scores_trimmed_LR_significant_only.pdf")
ggplot(modelResCtype[!modelResCtype$ONCOTREE_CODE=="PAAD" & modelResCtype$pLessThanP01==T,], aes(x = estimate, y = term, shape=pLessThanP01,colour = ONCOTREE_CODE)) +
  geom_point() +
  labs(title = "logistic regression modeling of Level 1 actionability \n trimmed LRs ",
       x = "LR score",
       y = "covariate") +
  xlim(-3,3)+
  #theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 10)


### Apply GLM model

# Fit a GLM model
# glm_model <- glm(isOncoKbLevel1 ~ ONCOTREE_CODE + is_paired_end + library_selection + preservation_technique, data = train_data, family = gaussian)
# 
# # Display the summary of the model
# summary(glm_model)
# 
# # Extract coefficients and convert them to a data frame
# coefficients_df <- broom::tidy(glm_model)
# 
# # Plot the coefficients
# outF <-  paste0(outDir,"/glm_coefficients.pdf")
# ggplot(coefficients_df, aes(x = term, y = estimate)) +
#   geom_bar(stat = "identity") +
#   geom_errorbar(aes(ymin = estimate - std.error, ymax = estimate + std.error), width = 0.2) +
#   labs(title = "GLM Coefficients",
#        x = "Predictors",
#        y = "Coefficient Estimate") +
#   theme_minimal()+
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#   theme(plot.title = element_text(hjust = 0.5))
# ggsave(outF,height = 7, width = 7)

### try z-scoring proportion data by cancer type (or rank-based)

##################
### assay info ###
##################

inF <- paste0(baseDir,"/../../data/AACR_Project_GENIE/Release_15p0_public/genomic_information.txt")
genomicInfo <- read.csv(inF,sep="\t")

# SEQ_ASSAY_ID and SEQ_PIPELINE_ID
seq_grp <- assayInfo %>%
  dplyr::group_by(SEQ_ASSAY_ID,SEQ_PIPELINE_ID) %>%
  dplyr::summarise(number_of_panels=n())
outF <- paste0(outDir,"/institution_seq_platoform_IDs.csv")
write.table(seq_grp,outF,row.names=F,quote=F,sep=",")

table(svCompiled$SEQ_ASSAY_ID %in% genomicInfo$SEQ_ASSAY_ID,svCompiled$SourceStudy)

#
length(unique(genomicInfo$Hugo_Symbol))

# rank genes according to which genes were measured most frequently
geneRank <- genomicInfo %>%
  dplyr::group_by(Hugo_Symbol) %>%
  dplyr::summarise(n_assays=dplyr::n_distinct(SEQ_ASSAY_ID)) %>%
  dplyr::arrange(desc(n_assays))

geneRank$Hugo_Symbol <- factor(geneRank$Hugo_Symbol,levels=geneRank$Hugo_Symbol)
outF <-  paste0(outDir,"/gene_counts_by_seq_assay_ID_top.pdf")
ggplot(geneRank[1:60,], aes(x = Hugo_Symbol, y = n_assays)) +
  geom_bar(stat = "identity") +
  labs(title = "Count of assays measuring gene of interest (n=113 total)",
       x = "gene",
       y = "number of assays") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 10)

geneRank$Hugo_Symbol <- factor(geneRank$Hugo_Symbol,levels=geneRank$Hugo_Symbol)
outF <-  paste0(outDir,"/gene_counts_by_seq_assay_ID_all.pdf")
ggplot(geneRank, aes(x = Hugo_Symbol, y = n_assays)) +
  geom_bar(stat = "identity") +
  labs(title = "Count of assays measuring gene of interest (n=113 total)",
       x = "gene",
       y = "number of assays") +
  theme_minimal()+
  #theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(axis.text.x = element_blank())+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 10)

## what proportion actionable mutation appear in top ranking genes?
iLevel1 <- grepl("1",oncokbOnly$ONCOKB_HIGHEST_LEVEL_SUMMARY)
lvl1_compiled <- oncokbOnly[iLevel1,]

lvl1_gene_cnts <- lvl1_compiled %>%
  dplyr::group_by(Hugo_Symbol) %>%
  dplyr::summarise(n_patients=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n_patients))

## all assays together (heme+solid+other)
rank_lvl1 <- geneRank %>%
  dplyr::left_join(lvl1_gene_cnts,by="Hugo_Symbol") 
rank_lvl1$n_patients[is.na(rank_lvl1$n_patients)] <- 0  
rank_lvl1$n_patients_cumsum <- cumsum(rank_lvl1$n_patients)
rank_lvl1$cumsum_perc_patients <- 100 * (rank_lvl1$n_patients_cumsum / length(unique(svCompiled$Tumor_Sample_Barcode)))
rank_lvl1$gene_rank <- rownames(rank_lvl1)

rank_lvl1$Hugo_Symbol <- factor(rank_lvl1$Hugo_Symbol,levels=geneRank$Hugo_Symbol)
outF <-  paste0(outDir,"/cumulative_sum_patients_with_level1_gene_mutation.pdf")
ggplot(rank_lvl1, aes(x = Hugo_Symbol, y = cumsum_perc_patients)) +
  geom_point()+
  #geom_bar(stat = "identity") +
  labs(title = "Cumulative sum of patients with a level 1 mutaiton \n genes ranked by most commonly measured in panels",
       x = "gene rank",
       y = "percent of patients") +
  theme_minimal()+
  #theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(axis.text.x = element_blank())+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 10)

rLvl1_long <- rank_lvl1[,c("Hugo_Symbol","n_assays","cumsum_perc_patients")] %>%
  tidyr::pivot_longer(!Hugo_Symbol, names_to = "type", values_to = "value") 

rLvl1_long$Hugo_Symbol <- factor(rLvl1_long$Hugo_Symbol,levels=geneRank$Hugo_Symbol)
outF <-  paste0(outDir,"/cumulative_sum_patients_with_level1_gene_mutation_and_prop.pdf")
ggplot(rLvl1_long, aes(x = Hugo_Symbol, y = value)) +
  geom_point()+
  facet_grid(type ~ ., scales="free_y")+
  labs(title = "Cumulative sum of patients with a level 1 mutaiton \n genes ranked by most commonly measured in panels",
       x = "gene rank",
       y = "percent of patients") +
  theme_minimal()+
  #theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(axis.text.x = element_blank())+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 10)

outF <-  paste0(outDir,"/cumulative_sum_patients_with_level1_gene_mutation_and_prop_top.pdf")
ggplot(rLvl1_long[1:180,], aes(x = Hugo_Symbol, y = value)) +
  geom_point()+
  facet_grid(type ~ ., scales="free_y")+
  labs(title = "Cumulative sum of patients with a level 1 mutaiton \n genes ranked by most commonly measured in panels",
       x = "gene rank",
       y = "percent of patients") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 12)

# what are examples of genes with high frequency, but low panel rank? 
iHighRank <- as.numeric(rank_lvl1$gene_rank) >= 30
iHighPatient <- rank_lvl1$n_patients >= 300
unexpected_hits <- rank_lvl1[iHighRank & iHighPatient,]

# print info on unexpected hits
for (x in unexpected_hits$Hugo_Symbol) {
  print(x)
  iGeneMatch = lvl1_compiled$Hugo_Symbol==x
  print("actionability type:")
  print(table(lvl1_compiled[iGeneMatch,"ONCOKB_HIGHEST_LEVEL_SUMMARY"]))
  print("disease level counts:")
  print(table(lvl1_compiled[iGeneMatch,"level_1_disease"]))
  print(table(lvl1_compiled[iGeneMatch,"level_3_disease"]))
  print("study counts:")
  print(table(lvl1_compiled[iGeneMatch,"SourceStudy"]))
  print("panel types:")
  print(table(lvl1_compiled[iGeneMatch,"SEQ_ASSAY_ID_mod"]))
  print("--------")
}

### repeat plots above but split heme from solid tumor cancers
geneRankHeme <- genomicInfo %>%
  dplyr::left_join(assayInfo[,c("SEQ_ASSAY_ID","isHemePanel")]) %>%
  dplyr::group_by(Hugo_Symbol,isHemePanel) %>%
  dplyr::summarise(n_assays=dplyr::n_distinct(SEQ_ASSAY_ID)) %>%
  dplyr::arrange(desc(n_assays))

lvl1_gene_cnts_heme <- lvl1_compiled %>%
  dplyr::group_by(Hugo_Symbol,isHemeCancer) %>%
  dplyr::summarise(n_patients=dplyr::n_distinct(Tumor_Sample_Barcode)) %>%
  dplyr::arrange(desc(n_patients)) 

lvl1_hemRnak <-lvl1_gene_cnts_heme %>%
  tidyr::pivot_wider(names_from = isHemeCancer,values_from=n_patients) %>%
  dplyr::mutate(across(everything(), ~ if_else(is.na(.), 0, .))) %>%
  dplyr::rename(notHeme=`FALSE`,
                heme=`TRUE`) %>%
  dplyr::mutate(hemeDiff = notHeme - heme) %>%
  dplyr::arrange(hemeDiff)

lvl1_gene_cnts_heme$Hugo_Symbol <- factor(lvl1_gene_cnts_heme$Hugo_Symbol, levels=lvl1_hemRnak$Hugo_Symbol)
lvl1_gene_cnts_heme$heme_status <- "not heme cancer"
lvl1_gene_cnts_heme[lvl1_gene_cnts_heme$isHemeCancer==T,"heme_status"] <- "heme cancer"

### level 1 hits by gene
outF <-  paste0(outDir,"/level1_hits_by_gene_and_heme_status.pdf")
ggplot(lvl1_gene_cnts_heme, aes(x = Hugo_Symbol, y = n_patients)) +
  geom_point()+
  facet_grid(heme_status ~ .)+
  #geom_bar(stat = "identity") +
  labs(title = "Level 1 actionable mutations appearing most frequently in heme vs. other cancers",
       x = "gene",
       y = "number of patients") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  #theme(axis.text.x = element_blank())+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 10)


## 
rank_lvl1_heme <- geneRankHeme %>%
  dplyr::left_join(lvl1_hemRnak,by="Hugo_Symbol") %>%
  tidyr::pivot_longer(c("n_assays","notHeme","heme"), names_to = "variable", values_to = "count")
rank_lvl1_heme$Hugo_Symbol <- factor(rank_lvl1_heme$Hugo_Symbol, levels=lvl1_hemRnak$Hugo_Symbol)

# outF <-  paste0(outDir,"/level1_hits_by_gene_and_heme_status_with_panel_info.pdf")
# ggplot(rank_lvl1_heme, aes(x = Hugo_Symbol, y = count)) +
#   geom_point()+
#   facet_grid(variable ~ isHemePanel,scale="free_y")+
#   #geom_bar(stat = "identity") +
#   labs(title = "Level 1 actionable mutations appearing most frequently in heme vs. other cancers",
#        x = "gene",
#        y = "number of patients") +
#   theme_minimal()+
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))+
#   #theme(axis.text.x = element_blank())+
#   theme(plot.title = element_text(hjust = 0.5))
# ggsave(outF,height = 7, width = 10)


#rank_lvl1$n_patients_cumsum <- cumsum(rank_lvl1$n_patients)
#rank_lvl1$cumsum_perc_patients <- 100 * (rank_lvl1$n_patients_cumsum / length(unique(svCompiled$Tumor_Sample_Barcode)))
#rank_lvl1$gene_rank <- rownames(rank_lvl1)


### pairwise intersection of genes among panels
runPairwiseCalc <- 0
if (runPairwiseCalc==1){
  pairwiseIntersection <- data.frame()
  panels <- unique(genomicInfo$SEQ_ASSAY_ID)
  i <- 1
  for (ix in seq(length(panels))) {
    x <- panels[ix]
    xg <- genomicInfo[genomicInfo$SEQ_ASSAY_ID==x,]
    xgenes <- unique(xg$Hugo_Symbol)
    for (iy in seq(length(panels))) {
      y <- panels[iy]
      yg <- genomicInfo[genomicInfo$SEQ_ASSAY_ID==y,]
      ygenes <- unique(yg$Hugo_Symbol)
      if (ix > iy) {
        print("")
      }
      else {
        print(ix) 
        print(iy) 
        print("----")
        pairwiseIntersection[i,"panel_1"] <- x
        pairwiseIntersection[i,"panel_2"] <- y
        pairwiseIntersection[i,"panel_1_n_genes"] <- length(xgenes)
        pairwiseIntersection[i,"panel_2_n_genes"] <- length(ygenes)
        pairwiseIntersection[i,"panel_1_2_intersection"] <- length(intersect(xgenes,ygenes))
      }
      i <- i+1
    }
  }
  pairwiseIntersection$intersection_as_prop_of_panel1 <- pairwiseIntersection$panel_1_2_intersection / pairwiseIntersection$panel_1_n_genes
}

### genomic intersection plots
#geneList <- c("KRAS","TP53","ABL1","EGFR")
#geneList <- c("FBXW7","PPP2R1A","CCNE1")
geneList <- geneRank$Hugo_Symbol[1:16]

posPanCntFull <- data.frame()
for (gene in geneList) {
  print(gene)
  # hit info
  lvl1_comp_gene <- lvl1_compiled[lvl1_compiled$Hugo_Symbol==gene,]
  nPosTotal <- dim(lvl1_comp_gene)[1]
  lvl1PosCnts <- lvl1_comp_gene  %>%
    dplyr::group_by(Start_Position) %>%
    dplyr::summarise(n=n(),
                     geneLevelFreq=n/nPosTotal) %>%
    dplyr::arrange(desc(n)) %>%
    dplyr::rename(coordinate=Start_Position,
                  size=n) %>%
    dplyr::mutate(pos="actionable_count",
                  SEQ_ASSAY_ID="1_observed_events",
                  type="observed")
  
  # genomic info
  geneSubset <- genomicInfo[genomicInfo$Hugo_Symbol==gene,]
  geneSubset$region_ID <- seq(1,dim(geneSubset)[1])
  geneSubset$assay_ID <- factor(geneSubset$SEQ_ASSAY_ID,labels=seq(1,length(unique(geneSubset$SEQ_ASSAY_ID))))
  geneSubset$assay_ID <- as.numeric(geneSubset$assay_ID)
  geneSubset$region_length <- geneSubset$End_Position - geneSubset$Start_Position
  geneSubset$type <- "panel"
  geneSubset$size <- 1
  
  intervalPlotDf <- geneSubset[,c("Start_Position","End_Position","SEQ_ASSAY_ID","size","type")] %>%
    tidyr::pivot_longer(c("Start_Position","End_Position"), names_to = "pos", values_to = "coordinate")
  
  densityPlotDf <- rbind(intervalPlotDf,lvl1PosCnts[,colnames(intervalPlotDf)])
  
  outF <-  paste0(geneOutDir,"/genomic_positions_of_panels_for_gene_",gene,".pdf")
  ggplot(intervalPlotDf, aes(x = coordinate, y = SEQ_ASSAY_ID, shape = pos)) +
    geom_point()+
    #facet_grid(type ~ ., scales="free_y")+
    labs(title = paste0(gene, "\nGenomic coordiantes tested by panel"),
         x = "genomic coordinate",
         y = "panel") +
    theme_minimal()+
    theme(axis.text.x = element_text(angle = 90, hjust = 1))+
    theme(plot.title = element_text(hjust = 0.5))
  ggsave(outF,height = 18, width = 12)
  
  if (dim(lvl1PosCnts)[1] > 0) {
    outF <-  paste0(geneOutDir,"/genomic_positions_of_panels_for_gene_",gene,"_with_counts.pdf")
    ggplot(densityPlotDf, aes(x = coordinate, y = SEQ_ASSAY_ID, shape = pos, size=size,color=type)) +
      geom_point(alpha=.2,size=1,color="black")+
      geom_point(alpha=.6)+
      #facet_grid(type ~ ., scales="free_y")+
      labs(title = paste0(gene, "\nGenomic coordiantes tested by panel"),
           x = "genomic coordinate",
           y = "panel") +
      theme_minimal()+
      theme(axis.text.x = element_text(angle = 90, hjust = 1))+
      theme(plot.title = element_text(hjust = 0.5))
    ggsave(outF,height = 18, width = 12)
    
    # create position-level evaluations
    lvl1PosCnts$temp_key <- 1
    geneSubset$temp_key <- 1
    
    # Use merge to create the Cartesian product
    posTargetMerged <- merge(lvl1PosCnts[,!colnames(lvl1PosCnts) %in% c("SEQ_ASSAY_ID")], 
                             geneSubset[,!colnames(geneSubset) %in% c("type","size","")], by = "temp_key")
    
    iGrtr <- posTargetMerged$coordinate >= posTargetMerged$Start_Position
    iLst <- posTargetMerged$coordinate <= posTargetMerged$End_Position
    posTargetMerged$posInTarget <- iGrtr & iLst
    
    # for each position how many panels does it appear in? 
    nPanelsMeasuringGene <- length(unique(posTargetMerged$SEQ_ASSAY_ID))
    nPanelsAll <- length(unique(genomicInfo$SEQ_ASSAY_ID))
    posPanCnt <- posTargetMerged %>%
      dplyr::filter(posInTarget) %>%
      dplyr::group_by(coordinate) %>%
      dplyr::summarise(n=dplyr::n_distinct(SEQ_ASSAY_ID),
                       geneLevelFreq=mean(geneLevelFreq),
                       percPanelsMeasuringGene=n/nPanelsMeasuringGene,
                       percPanlesAll=n/nPanelsAll,
                       Feature_Type=paste0(unique(Feature_Type),collapse=";")) %>%
      dplyr::arrange(desc(n))
    
    posPanCnt$gene <- gene
    posPanCntFull <- rbind(posPanCntFull,posPanCnt)
    
    outF <-  paste0(geneOutDir,"/genomic_positions_of_panels_for_gene_",gene,"_freq_by_panel_coverage.pdf")
    ggplot(posPanCnt, aes(x = geneLevelFreq, y = percPanelsMeasuringGene, color=Feature_Type)) +
      geom_point(alpha=.6)+
      labs(title = paste0(gene, "\nActionable genomic positions tested by panel"),
           x = "Frequency of actionable event at position",
           y = "Fraction of panels measuring position") +
      ylim(0,1)+
      theme_minimal()+
      theme(plot.title = element_text(hjust = 0.5))
    ggsave(outF,height = 7, width = 7)
    
    outF <-  paste0(geneOutDir,"/genomic_positions_of_panels_for_gene_",gene,"_freq_by_panel_coverage_log.pdf")
    ggplot(posPanCnt, aes(x = geneLevelFreq, y = percPanelsMeasuringGene, color=Feature_Type)) +
      geom_point(alpha=.6)+
      scale_x_log10(
        breaks = scales::trans_breaks("log10", function(x) 10^x),
        labels = scales::trans_format("log10", scales::math_format(10^.x))
      )+
      labs(title = paste0(gene, "\nActionable genomic positions tested by panel"),
           x = "Frequency of actionable event at position",
           y = "Fraction of panels measuring position") +
      ylim(0,1)+
      theme_minimal()+
      theme(plot.title = element_text(hjust = 0.5))
    ggsave(outF,height = 7, width = 7)
    
  }
}

outF <-  paste0(outDir,"/genomic_positions_of_panels_for_gene_1_freq_by_panel_coverage.pdf")
ggplot(posPanCntFull, aes(x = geneLevelFreq, y = percPanelsMeasuringGene, color=gene)) +
  geom_point(alpha=.6)+
  labs(title = paste0("Actionable genomic positions tested by panel"),
       x = "Frequency of actionable event at position",
       y = "Fraction of panels measuring position") +
  ylim(0,1)+
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 7)

outF <-  paste0(outDir,"/genomic_positions_of_panels_for_gene_1_freq_by_panel_coverage_log.pdf")
ggplot(posPanCntFull, aes(x = geneLevelFreq, y = percPanelsMeasuringGene, color=gene)) +
  geom_point(alpha=.6)+
  labs(title = paste0("Actionable genomic positions tested by panel"),
       x = "Frequency of actionable event at position",
       y = "Fraction of panels measuring position") +
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  )+
  ylim(0,1)+
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 7, width = 7)


################################
### detailed tumor type info ###
################################

c_type_count <- perSubjVar %>%
  dplyr::group_by(CANCER_TYPE) %>%
  dplyr::summarise(number_of_subjects=n()) %>%
  dplyr::arrange(desc(number_of_subjects))

c_type_detailed <- perSubjVar %>%
  dplyr::group_by(CANCER_TYPE,CANCER_TYPE_DETAILED) %>%
  dplyr::summarise(number_of_subjects=n()) %>%
  dplyr::arrange(desc(number_of_subjects))
outF <- paste0(outDir,"/cancer_type_detailed_counts.txt")
write.table(c_type_detailed,outF,row.names=F,quote=F,sep="\t")

iTop <- c_type_detailed$CANCER_TYPE %in% c_type_count$CANCER_TYPE[1:10]
c_type_subset <- c_type_detailed[iTop,]

c_type_subset$CANCER_TYPE_DETAILED <- factor(c_type_subset$CANCER_TYPE_DETAILED,levels=c_type_subset$CANCER_TYPE_DETAILED)
outF <-  paste0(outDir,"/cancer_type_detailed_counts.pdf")
ggplot(c_type_detailed[1:20,], aes(x = CANCER_TYPE_DETAILED, y = number_of_subjects)) +
  geom_bar(stat = "identity") +
  labs(title = "Sample counts for each detailed cancer type categorizaiton (most common cancers)",
       x = "gene",
       y = "number of assays") +
  facet_wrap(CANCER_TYPE ~ ., scales="free_x") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  theme(plot.title = element_text(hjust = 0.5))
ggsave(outF,height = 14, width = 10)

#####################################
### summary table by assays used ####
#####################################
svCompiledAssay <- svCompiled %>%
  dplyr::left_join(assayInfo,by="SEQ_ASSAY_ID")

# study, institution, assay counts
institution_summary <- svCompiledAssay %>%
  dplyr::mutate(assay_type=paste0(library_selection,"-",platform,"-",preservation_technique)) %>%
  dplyr::group_by(SourceStudy,Center) %>%
  dplyr::summarise(n_subjects=dplyr::n_distinct(Tumor_Sample_Barcode),
                   n_assays_used=dplyr::n_distinct(SEQ_ASSAY_ID_mod),
                   assay_types=paste0(unique(assay_type),collapse=";")) %>%
  dplyr::arrange(desc(n_subjects)) %>%
  dplyr::mutate(cntrString = paste0(SourceStudy," ", Center))
outF <- paste0(outDir,"/institution_and_assay_sample_counts.txt")
write.table(institution_summary,outF,row.names=F,quote=F,sep="\t")

arbitrary_variant_per_subject <- svCompiledAssay %>%
  dplyr::group_by(Tumor_Sample_Barcode) %>%
  dplyr::filter(dplyr::row_number()==1) # pick a representative sample

# FFPE and PCR/hybrid capture proportions
#  plyr::summarise(n_FFPE=sum(preservation_technique=="FFPE"))

# most common cancer types, race, age, demographic, etc

# cancer type counts by center
cancer_types_sample_cnts <- arbitrary_variant_per_subject %>% 
  dplyr::group_by(level_1_disease) %>%
  #dplyr::group_by(ONCOTREE_CODE) %>%
  dplyr::summarise(n=n()) %>% 
  dplyr::arrange(desc(n))

nTopCancer <- 12
top_n_cancer_types_cnts <- arbitrary_variant_per_subject %>% 
  dplyr::filter(level_1_disease %in% c(cancer_types_sample_cnts$level_1_disease)[1:nTopCancer]) %>%
  dplyr::group_by(SourceStudy,Center,level_1_disease) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::left_join(institution_summary,by=c("SourceStudy","Center")) %>%
  dplyr::mutate(percOfCenter=100*n/n_subjects,
                cntrString = paste0(SourceStudy," ", Center))

# make heatmap of cancer type percentages for top N cancers
# library(circlize)
# #library(devtools)
# #install_github("jokergoo/ComplexHeatmap")
# library(ComplexHeatmap)

#### scratch circlize example code
# set.seed(123)
# mat1 = rbind(cbind(matrix(rnorm(50*5, mean = 1), nr = 50), 
#                    matrix(rnorm(50*5, mean = -1), nr = 50)),
#              cbind(matrix(rnorm(50*5, mean = -1), nr = 50), 
#                    matrix(rnorm(50*5, mean = 1), nr = 50))
# )
# rownames(mat1) = paste0("R", 1:100)
# colnames(mat1) = paste0("C", 1:10)
# mat1 = mat1[sample(100, 100), ] # randomly permute rows
# split = sample(letters[1:5], 100, replace = TRUE)
# spilt = factor(split, levels = letters[1:5])
# col_fun1 = colorRamp2(c(-2, 0, 2), c("blue", "white", "red"))
# circos.heatmap(mat1, split = split, col = col_fun1)
# circos.clear()
# 
# circos.par(start.degree = 90, gap.degree = 10)
# circos.heatmap(mat1, split = split, col = col_fun1, track.height = 0.4, 
#                bg.lwd = 2, bg.lty = 2, show.sector.labels = TRUE,
#                rownames.side = "outside")
# circos.clear()

# add column names
# circos.par(gap.after = c(2, 2, 2, 2, 10))
# circos.heatmap(mat1, split = split, col = col_fun1, track.height = 0.4)
# circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
#   if(CELL_META$sector.numeric.index == 5) { # the last sector
#     cn = colnames(mat1)
#     n = length(cn)
#     circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"), 
#                 1:n - 0.5, cn, 
#                 cex = 0.5, adj = c(0, 0.5), facing = "inside")
#   }
# }, bg.border = NA)

### Make circular plot of cancer type counts by study
topNwide <- top_n_cancer_types_cnts[,c("cntrString","level_1_disease","percOfCenter")] %>%
  tidyr::pivot_wider(names_from = cntrString,values_from=percOfCenter) %>%
  dplyr::mutate(across(everything(), ~ replace_na(., 0))) %>%
  data.frame() 

# topNwideMtrx <- as.matrix(topNwide[,!colnames(topNwide) %in% c("level_1_disease")])
# orderStr <- gsub("[- ]", ".",  institution_summary$cntrString)
# topNwideMtrx <- topNwideMtrx[,head(orderStr,-1)]
# 
# rownames(topNwideMtrx) <- topNwide$level_1_disease

# outF <-  paste0(outDir,"/cancer_type_by_institution_center.pdf")
# pdf(outF, width = 8, height = 6)
# ComplexHeatmap::Heatmap(topNwideMtrx) #, row_split = split)
# dev.off()
# 
# 
# outF <-  paste0(outDir,"/cancer_type_by_institution_center_circular.pdf")
# pdf(outF, width = 8, height = 8)
# col_fun1 = colorRamp2(c(0, 30, 60), c("snow2", "grey", "blue"))
# #circos.par(gap.after = c(2, 2, 2, 2, 10))
# #circos.heatmap(topNwideMtrx, col = col_fun1)
# #circos.par(gap.after = c(2, 2, 2, 2, 10))
# circos.par(gap.after = c(90))
# circos.heatmap(t(topNwideMtrx), col = col_fun1, track.height = 0.4,
#                bg.lwd = 2, bg.lty = 2, show.sector.labels = TRUE,
#                cluster = F,
#                rownames.side = "outside")
# #circos.clear()
# 
# #circos.heatmap(topNwideMtrx,col = col_fun1, track.height = 0.4)
# circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
#   #if(CELL_META$sector.numeric.index == 5) { # the last sector
#     cn = rownames(topNwideMtrx)
#     n = length(cn)
#     circos.text(rep(CELL_META$cell.xlim[2], n) + convert_x(1, "mm"),
#                 1:n - 0.5,
#                 cn,
#                 cex = 0.5,
#                 adj = c(0, 0.5),
#                 facing = "inside")
#   #}
# }, bg.border = NA)
# circos.clear()
# dev.off()

# bar chart for institution counts
data <- institution_summary[,c("Center","n_subjects")]

## make circular bar chart
empty_bar <- 6

# Add lines to the initial dataset
to_add <- matrix(NA, empty_bar, ncol(data))
colnames(to_add) <- colnames(data)
data <- rbind(data, to_add)
data$id <- seq(1, nrow(data))

outF <-  paste0(outDir,"/cancer_type_by_institution_center_bar_counts.pdf")
ggplot(data, aes(x=as.factor(id), y=n_subjects)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
  
  # This add the bars with a blue color
  geom_bar(stat="identity", fill=alpha("wheat2", 0.4)) +
  
  # Limits of the plot = very important. The negative value controls the size of the inner circle, the positive one is useful to add size over each bar
  ylim(-max(institution_summary$n_subjects),max(institution_summary$n_subjects)) +
  
  # Custom the theme: no axis title and no cartesian grid
  theme_minimal() +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-2,4), "cm")     # This remove unnecessary margin around plot
  ) +
  
  # This makes the coordinate polar instead of cartesian.
  coord_polar(start = 0,clip = "on",
              direction=1)
ggsave(outF,height = 8,width = 8)

outRFile <- paste0(outDir,"/work_space_20250125.RData")
save.image(file = outRFile)
