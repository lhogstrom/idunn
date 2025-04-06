import pandas as pd
from sqlalchemy import create_engine
import os
import shap
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc

db_path = os.path.expanduser("/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20240924/balder-harmonized-biomarker-data-v20240924.sqlite")

outDir = '/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20240924/ml_modeling_python'

# Create the engine for SQLite
engine = create_engine(f"sqlite:///{db_path}")

# Name of the table you want to load
table_name = 'perPatientLevel1'   
df = pd.read_sql_table(table_name, con=engine)



demographic_list = [
#'PATIENT_ID',
#'CANCER_TYPE',
#'CANCER_TYPE_DETAILED',
#'ONCOTREE_CODE',
'SAMPLE_TYPE',
'SAMPLE_TYPE_DETAILED',
'STAGE',
#'INT_Biopsy_To_Death',
#'DEAD',
#'INT_SEQ_TO_DEATH',
#'INT_SEQ_TO_CONTACT',
'SEX',
'RACE',
'ETHNICITY',
#'INT_Seq_or_Biopsy_To_Death',
'AGE_AT_SEQ_REPORT',
'TUMOR_PURITY',
'TUMOR_PLOIDY'
]

techinical_list = [
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
]

feature_list = demographic_list+techinical_list
outcome='isOncoKbLevel1'
column_subset = feature_list + [outcome]

all_present = set(column_subset).issubset(df.columns)
print("All columns present?", all_present)

df['ONCOTREE_CODE'].value_counts()
#cType = 'LUAD'
#cType = 'COAD'
cType = 'IDC'


cType_df = df[df['ONCOTREE_CODE'] == cType]
subsetDf = cType_df[column_subset]


#### model

subsetDf.columns = [str(col) for col in subsetDf.columns]

# Define features and target
X = subsetDf.drop('isOncoKbLevel1', axis=1)
y = subsetDf['isOncoKbLevel1']


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
print("training set size:" + str(X_train.shape))

# Ensure that the train and test DataFrames have plain string column names
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Identify categorical and numeric columns
numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

# Create pipelines for numeric and categorical data including imputation
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


# set to dense array
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine the transformations using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Create a pipeline that includes preprocessing and the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Perform 5-fold cross-validation on the training data
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print("CV scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# Fit the model on full training data
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))

# output transformed data for inspection
X_train_transformed = preprocessor.transform(X_train)
feature_names = preprocessor.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
outF = os.path.join(outDir, cType+"_head_train_data_transformed.csv")
X_train_df.head(20).to_csv(outF, index=False)
print("post-process feature set size:" + str(X_train_df.shape))

#### make ROC curve #### 

# Compute the false positive rate, true positive rate, and thresholds
y_proba = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)
print("auc value:" + str(roc_auc))

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
outF = os.path.join(outDir, cType+"_roc_curve.pdf")
plt.savefig(outF, format="pdf", bbox_inches='tight')
plt.close()



#### plot shap values

# Now extract the fitted preprocessor and transform data
fitted_preprocessor = pipeline.named_steps['preprocessor']
X_train_trans = fitted_preprocessor.transform(X_train)
X_test_trans = fitted_preprocessor.transform(X_test)

# Create SHAP TreeExplainer using the fitted classifier and background data
model = pipeline.named_steps['classifier']
explainer = shap.TreeExplainer(model, data=X_train_trans)

# Compute SHAP values for the test set (for a binary classifier, this returns a list)
shap_values = explainer.shap_values(X_test_trans, check_additivity=False)

# Optionally, get feature names if available
try:
    feature_names = fitted_preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = None

# Plot SHAP summary for class 1 (adjust index as needed)
#shap.summary_plot(shap_values[0], X_test_trans, feature_names=feature_names)
shap_values_class1 = shap_values[:, :, 1]

plt.figure(figsize=(30, 30))
shap.summary_plot(shap_values_class1, 
                  X_test_trans, 
                  feature_names=feature_names, 
                  max_display=12,
                  show=False)
outF = os.path.join(outDir, cType+"_shap_values_plot.pdf")
plt.savefig(outF, format="pdf", bbox_inches='tight')
plt.close()


shap.summary_plot(shap_values_class1, 
                  X_test_trans, 
                  feature_names=feature_names, 
                  max_display=12,
                  show=False)
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
outF = os.path.join(outDir, cType + "_shap_values_plot.pdf")
plt.savefig(outF, format="pdf", bbox_inches='tight')
plt.close()

# waterfall plot for a single point
explainer = shap.Explainer(model.predict, X_train_trans)
X100 = shap.utils.sample(X_train_trans, 100)  # 100 instances for use as the background distribution
X100 = pd.DataFrame(X100,columns=feature_names)
shap_values_explained = explainer(X100)
sample_ind = 7

shap.plots.waterfall(shap_values_explained[sample_ind],
                     max_display=14,
                     show=False)
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
outF = os.path.join(outDir, cType + "_shap_example_subject_" + str(sample_ind) +".pdf")
plt.savefig(outF, format="pdf", bbox_inches='tight')
plt.close()




# ### pairwise corr structure 
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import scipy.stats as ss

# def cramers_v(confusion_matrix):
#     """Calculate Cramér's V statistic for categorical-categorical correlation."""
#     if confusion_matrix.empty:
#         return np.nan
#     chi2 = ss.chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     if n == 0:
#         return np.nan
#     phi2 = chi2 / n
#     r, k = confusion_matrix.shape
#     return np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else np.nan

# def mixed_corr(df):
#     cols = df.columns
#     corr = pd.DataFrame(index=cols, columns=cols)

#     for col1 in cols:
#         for col2 in cols:
#             series1 = df[col1].dropna()
#             series2 = df[col2].dropna()
            
#             if series1.empty or series2.empty:
#                 corr.loc[col1, col2] = np.nan
#                 continue

#             if df[col1].dtype == 'object' and df[col2].dtype == 'object':
#                 confusion = pd.crosstab(df[col1], df[col2])
#                 corr.loc[col1, col2] = cramers_v(confusion)
#             elif df[col1].dtype == 'object' or df[col2].dtype == 'object':
#                 # Use point biserial for binary categorical vs numeric
#                 cat_col, num_col = (col1, col2) if df[col1].dtype == 'object' else (col2, col1)
#                 try:
#                     if df[cat_col].nunique() == 2:
#                         labels = pd.Categorical(df[cat_col]).codes
#                         corr.loc[col1, col2] = ss.pointbiserialr(df[num_col], labels)[0]
#                     else:
#                         corr.loc[col1, col2] = np.nan
#                 except:
#                     corr.loc[col1, col2] = np.nan
#             else:
#                 corr.loc[col1, col2] = df[[col1, col2]].corr().iloc[0, 1]

#     return corr.astype(float)


# corr_matrix = mixed_corr(subsetDf)

# # Plot
# plt.figure(figsize=(20, 20))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
# plt.title("Pairwise Correlation Structure (Mixed Types)")
# plt.tight_layout()
# outF = os.path.join(outDir, "correlation_matrix.pdf")
# plt.savefig(outF, format="pdf")
# #plt.show()




