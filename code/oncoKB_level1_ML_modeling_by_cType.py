import pandas as pd
from sqlalchemy import create_engine
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc



def get_patient_level_data(db_path):
    # Create the engine for SQLite
    engine = create_engine(f"sqlite:///{db_path}")
    # Name of the table you want to load
    table_name = 'perPatientLevel1'   
    df = pd.read_sql_table(table_name, con=engine)
    return df

def subset_patient_data(df,cType):
    # demographic columns
    demographic_list = [
    'SAMPLE_TYPE',
    'SAMPLE_TYPE_DETAILED',
    'STAGE',
    'SEX',
    'RACE',
    'ETHNICITY',
    'AGE_AT_SEQ_REPORT',
    'TUMOR_PURITY',
    'TUMOR_PLOIDY'
    ]
    ###excluded columns:
    #'PATIENT_ID',
    #'CANCER_TYPE',
    #'CANCER_TYPE_DETAILED',
    #'ONCOTREE_CODE',
    #'INT_Biopsy_To_Death',
    #'DEAD',
    #'INT_SEQ_TO_DEATH',
    #'INT_SEQ_TO_CONTACT',    
    #'INT_Seq_or_Biopsy_To_Death',
        
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

    cType_df = df[df['ONCOTREE_CODE'] == cType]
    subsetDf = cType_df[column_subset]
    return subsetDf

def build_ml_pipeline(X_train):
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
    return pipeline, preprocessor

#def make_shap_explainer(pipeline,X_train,X_test):
    
#    return explainer

def make_age_dist_plot(df,cType,outF): 
    # Split data
    iLevel1 = df['isOncoKbLevel1'].astype(bool)
    ages_mutated = df[iLevel1]['AGE_AT_SEQ_REPORT'].dropna()
    ages_not_mutated = df[~iLevel1]['AGE_AT_SEQ_REPORT'].dropna()
    
    # Define bins
    bins = range(10, 105, 5)
    x_grid = np.linspace(10, 105, 300)
    
    # KDEs
    kde_mutated = gaussian_kde(ages_mutated)
    kde_not_mutated = gaussian_kde(ages_not_mutated)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
    
    # Not mutated
    ax1.hist(ages_not_mutated, bins=bins, color='skyblue', edgecolor='black', alpha=0.6, density=True)
    ax1.plot(x_grid, kde_not_mutated(x_grid), color='blue', lw=2, label='Smoothed KDE')
    ax1.set_title('No Level 1 actionable event')
    ax1.set_ylabel('Density')
    ax1.grid(True)
    ax1.legend()
    
    # Mutated
    ax2.hist(ages_mutated, bins=bins, color='salmon', edgecolor='black', alpha=0.6, density=True)
    ax2.plot(x_grid, kde_mutated(x_grid), color='red', lw=2, label='Smoothed KDE')
    ax2.set_title('Level 1 actionable event')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Density')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(outF, format="pdf", bbox_inches='tight')
    plt.close()

def make_age_gender_dist_plot(df,cType,outF,excludeUnknown=True): 
    
    # Ensure correct data types and labels
    df['isOncoKbLevel1'] = df['isOncoKbLevel1'].astype(bool)
    
    mutation_labels = [False, True]  # False = No event (top), True = Actionable (bottom)
    if excludeUnknown:
        sex_labels = ['Female', 'Male']  # Columns
    else:
        sex_labels = ['Female', 'Male', 'Unknown']  # Columns
    
    # Binning and KDE setup
    bins = range(10, 105, 5)
    x_grid = np.linspace(10, 105, 300)
    
    # Set up 2x3 subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharex=True, sharey=True)
    
    # Loop through mutation status and sex
    for i, mutated in enumerate(mutation_labels):
        for j, sex in enumerate(sex_labels):
            ax = axes[i, j]
            label = 'Level 1 actionable event' if mutated else 'No Level 1 actionable event'
    
            # Subset and drop NA
            subset = df[(df['isOncoKbLevel1'] == mutated) & (df['SEX'] == sex)]
            ages = subset['AGE_AT_SEQ_REPORT'].dropna()
    
            # Plot histogram
            ax.hist(ages, bins=bins, density=True,
                    color='skyblue' if not mutated else 'salmon',
                    edgecolor='black', alpha=0.6)
    
            # Plot KDE if enough points
            if len(ages) > 1:
                kde = gaussian_kde(ages)
                ax.plot(x_grid, kde(x_grid),
                        color='blue' if not mutated else 'red',
                        lw=2, label='Smoothed KDE')
    
            # Titles and labels
            ax.set_title(f"{sex}")
            if j == 0:
                ax.set_ylabel(f"{label}\n\nDensity")
            else:
                ax.set_ylabel("")
    
            if i == 1:
                ax.set_xlabel("Age at Report")
            else:
                ax.set_xlabel("")
    
            ax.grid(True)
            ax.legend()    
    # Overall plot formatting
    fig.suptitle("Age Distribution by OncoKB Level 1 Status and Gender", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for title
    plt.savefig(outF, format="pdf", bbox_inches='tight')
    plt.close()    

def gender_grouped_bar(df,cType,outF):     
    # Create a summary table
    plot_df = df.groupby(['isOncoKbLevel1', 'SEX']).size().reset_index(name='Count')
    plot_df['isOncoKbLevel1'] = plot_df['isOncoKbLevel1'].map({False: 'No Level 1 Event', True: 'Level 1 Event'})
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=plot_df, x='isOncoKbLevel1', y='Count', hue='SEX')
    plt.title('Counts by Gender and Mutation Status')
    plt.xlabel('Mutation Status')
    plt.ylabel('Number of Subjects')
    plt.legend(title='Sex')
    plt.tight_layout()
    plt.savefig(outF, format="pdf", bbox_inches='tight')
    plt.close()  

def main():
    outDir = '/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20240924/ml_modeling_python'
    db_path = os.path.expanduser("/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20240924/balder-harmonized-biomarker-data-v20240924.sqlite")
    df = get_patient_level_data(db_path)
    
    otCodeNames = df.groupby('ONCOTREE_CODE')['CANCER_TYPE_DETAILED'].first().reset_index()
    countCtypes = pd.DataFrame(df['ONCOTREE_CODE'].value_counts())
    countCtypesResetF = countCtypes.reset_index()
    countCtypesReset = pd.merge(countCtypesResetF, otCodeNames, on='ONCOTREE_CODE', how='left')
    
    actionCnt = df.groupby(['ONCOTREE_CODE','isOncoKbLevel1']).size().reset_index(name='actionable_events')
    actionCntPos = actionCnt[actionCnt['isOncoKbLevel1'] == 1]
    
    ctype_action_cnts = pd.merge(countCtypesReset, actionCntPos, on='ONCOTREE_CODE', how='left')
    ctype_action_cnts['proportion_actionable'] = ctype_action_cnts['actionable_events'] / ctype_action_cnts['count'] 
    
    cTypes_to_test = ctype_action_cnts[ctype_action_cnts['actionable_events'] >= 250]
    dfThresh = df[df['ONCOTREE_CODE'].isin(cTypes_to_test.ONCOTREE_CODE)]
    print("patients in OT code with required Level 1 events: " + str(dfThresh.shape[0]))
    
    # make age plots
    for cType in cTypes_to_test['ONCOTREE_CODE'][5:17]: 
        print(cType)
        subsetDf = subset_patient_data(df,cType)
        outF = os.path.join(outDir, cType+"_age_distribution.pdf")
        make_age_dist_plot(subsetDf,cType,outF)
        outF = os.path.join(outDir, cType+"_age_gender_distribution.pdf")
        make_age_gender_dist_plot(subsetDf,cType,outF,excludeUnknown=True)
        outF = os.path.join(outDir, cType+"_gender_grouped_bar.pdf")
        gender_grouped_bar(subsetDf,cType,outF)
    
    model_metrics = []
    all_shaps = []
    for cType in cTypes_to_test['ONCOTREE_CODE'][5:17]: 
        #cType = 'LUAD'
        print(cType)
        subsetDf = subset_patient_data(df,cType)

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
        
        pipeline, preprocessor = build_ml_pipeline(X_train)
            
        # Perform 5-fold cross-validation on the training data
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        print("CV scores:", cv_scores)
        print("Mean CV accuracy:", cv_scores.mean())

        # Fit the model on full training data
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy_val = accuracy_score(y_test, y_pred)
        print("Test accuracy:", accuracy_val)

        # output transformed data for inspection
        X_train_transformed = preprocessor.transform(X_train)
        feature_names = preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        #outF = os.path.join(outDir, cType+"_head_train_data_transformed.csv")
        #X_train_df.head(20).to_csv(outF, index=False)
        print("post-process feature set size:" + str(X_train_df.shape))

        #### make ROC curve #### 

        # Compute the false positive rate, true positive rate, and thresholds
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        preds = pipeline.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        print("auc value:" + str(roc_auc))

        # Compute the area under the ROC curve
        auc_value = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print("auc value 1:" + str(auc_value))
        
        model_metrics.append({
            'ONCOTREE_CODE': cType,
            'AUC': auc_value,
            'Accuracy': accuracy_val,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

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
        
        # calculate shap values
        #explainer = make_shap_explainer(pipeline,X_train,X_test)
        
        # extract the fitted preprocessor and transform data
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
        tmpShapDf = pd.DataFrame(shap_values_class1)
        tmpShapDf.columns = feature_names
        tmpShapDf["ONCOTREE_CODE"] = cType

        all_shaps.append(tmpShapDf)

        plt.figure(figsize=(30, 30))
        shap.summary_plot(shap_values_class1, 
                          X_test_trans, 
                          feature_names=feature_names, 
                          max_display=12,
                          show=False)
        outF = os.path.join(outDir, cType+"_shap_values_plot.pdf")
        plt.savefig(outF, format="pdf", bbox_inches='tight')
        plt.close()
        
  
    metrics_df = pd.DataFrame(model_metrics)
    metrics_extended = pd.merge(cTypes_to_test, metrics_df, on='ONCOTREE_CODE', how='left')
    outF = os.path.join(outDir, "ml_rf_metrics.csv")
    metrics_extended.to_csv(outF, index=False)
    print(metrics_df)
    
    #shapsDf = pd.DataFrame(all_shaps)
    shapsDf = pd.concat(all_shaps, ignore_index=True)
    outF = os.path.join(outDir, "shap_values_by_ot_code.csv")
    shapsDf.to_csv(outF, index=False)
    
    feature_ranks = []
    for cType in shapsDf.ONCOTREE_CODE.unique():
        print(cType)
        subsetShapDf = shapsDf[shapsDf['ONCOTREE_CODE']==cType]
        subsetShapDf = subsetShapDf.drop(columns='ONCOTREE_CODE')
        # 1. Compute mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(subsetShapDf).mean(axis=0)
        #mean_raw_shap = subsetShapDf.mean(axis=0)
        
        # 2. Create a DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            'ONCOTREE_CODE': cType,
            'feature': mean_abs_shap.index,
            'mean_abs_shap': mean_abs_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
        importance_df['feature_rank'] = range(1, len(importance_df) + 1)
        feature_ranks.append(importance_df)
        
        # Optional: view top N
        print(importance_df.head(10))

    featureRanksDf = pd.concat(feature_ranks, ignore_index=True)
    outF = os.path.join(outDir, "shap_feature_ranks.csv")
    featureRanksDf.to_csv(outF, index=False)
    
    # order features for bump plot
    topFeatures = featureRanksDf[featureRanksDf['feature_rank']<8]
    ftTopCnts = topFeatures['feature'].value_counts()
    topHits = ftTopCnts.index[:5]
    
    rankBump = topFeatures[topFeatures['feature'].isin(topHits)]
    
    # bump rank plot
    plt.figure(figsize=(8, 6))

    # Plot each item's rank trajectory
    for ft in rankBump['feature'].unique():
        item_data = rankBump[rankBump['feature'] == ft]
        plt.plot(item_data['ONCOTREE_CODE'], item_data['feature_rank'], marker='o', label=ft)
    
    plt.gca().invert_yaxis()  # Lower ranks (1) at top
    plt.title('SHAP Feature ranking to predict Level 1 mutation status')
    plt.xlabel('Cancer type')
    plt.ylabel('freature rank')
    plt.legend(title='Item')
    plt.grid(True)
    plt.tight_layout()
    outF = os.path.join(outDir,"shap_feature_rank_bump_plot.pdf")
    plt.savefig(outF, format="pdf", bbox_inches='tight')
    plt.close()
        
if __name__ == '__main__':
    main()
    
# %%