// main.nf

params.input_data_dir = '../../data/'
params.base_dir = './'
params.results_dir = '../../output/actionability_db_curration_20250522' // consider timestamp variable
params.oncokbToken = file('../../data/ONCOKB_token.txt')
params.dbDir = '../../data/processed/balderResultsDb'
params.dbName = "balder-harmonized-biomarker-data-v20250522.sqlite"
params.dbName2 = "balder-compiled-raw-data-v20250522.sqlite"
params.tmpVarTbl = "compiled_mutations_all_studies.txt"
params.oncokbOut = "compiled_mutations_all_studies.oncokb.txt"
params.scripts_dir = '../code'
// scripts for primary data products
params.scripts1 = file('../code/clinical_annotation_db_curration.r')
params.scripts2 = file('../code/cancer_genomics_studies_db_curration.r')
params.scripts3 = file('../code/setup_and_structure_oncokb_run.r')
params.scripts4 = file('../code/run_oncokb_annotator.sh')
params.scripts5 = file('../code/load_oncokb_run.r')
params.scripts6 = file('../code/clinical_annotation_variant_matching.r')
params.scripts7 = file('../code/targeted_panel_coordinate_lookup.r')
params.scripts8 = file('../code/generate_perPatientOncoKBLevelSorted.r')

// scripts for reporting and analysis
params.scripts9 = file('../code/incidence_of_clinically_annotated_mutations_reporting.r')
params.scripts10 = file('../code/MAF_and_prognostic_marker_analysis.r')
params.scripts11 = file('../code/summary_oncokb_by_cancer_type.r')
params.scripts12 = file('../code/clinical_technical_feature_description.R')

params.scripts13 = file('../code/filter_and_structure_genie_data_for_modeling.r')
params.scripts14 = file('../code/oncoKB_level1_ML_modeling_by_cType.py')

log.info """\
    B A L D E R - N F   P I P E L I N E
    ===================================
    data dir     : ${params.input_data_dir}
    base dir     : ${params.base_dir}
    database dir : ${params.dbDir}
    outdir       : ${params.results_dir}
    reports dir  : ${params.reports_dir}
    """
    .stripIndent()


// Define the preprocessing
process preprocess_data1 {

    input:
    path base_dir

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts1 $base_dir $params.dbName
    """
}

process preprocess_data2 {
    publishDir params.dbDir, mode:'copy'

    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts2 $base_dir $params.dbName $params.dbName2
    """
}

process oncokb_file_setup {
    publishDir params.results_dir, mode:'copy'

    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.tmpVarTbl"

    script:
    """
    Rscript $params.scripts3 $base_dir $params.dbName $params.tmpVarTbl
    """
}

process run_oncokb {
    publishDir params.results_dir, mode:'copy'

    input:
    path "$params.tmpVarTbl"
    path "$params.oncokbToken"

    output:
    path "$params.oncokbOut"

    script:
    """
    bash $params.scripts4 $params.tmpVarTbl $params.oncokbToken $params.oncokbOut
    """
}

process load_oncokb_output {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts5 $base_dir $params.dbName
    """
}

process clinical_annotation_matching {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts6 $base_dir $params.dbName
    """
}

process targeted_panel_coordinate_lookup {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts7 $base_dir $params.dbName
    """
}

process generate_perPatientOncoKBLevelSorted {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts8 $base_dir $params.dbName
    """
}

process incidence_of_clinically_annotated_mutations_reporting {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts9 $base_dir $params.dbName
    """
}

process MAF_and_prognostic_marker_analysis {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts10 $base_dir $params.dbName
    """
}

process summary_oncokb_by_cancer_type {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts11 $base_dir $params.dbName
    """
}

process clinical_technical_feature_description {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts12 $base_dir $params.dbName
    """
}


process filter_and_structure_genie_data_for_modeling {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    Rscript $params.scripts13 $base_dir $params.dbName
    """
}

process oncoKB_level1_ML_modeling_by_cType {
    publishDir params.results_dir, mode:'copy'
    
    input:
    path base_dir
    path "$params.dbName"

    output:
    path "$params.dbName"

    script:
    """
    python $params.scripts14
    """
}


workflow {
    //sqldb = preprocess_data1(file("${params.base_dir}"))
    //sqldb2 = preprocess_data2(file("${params.base_dir}"),sqldb)
    //tmpVarTbl = oncokb_file_setup(file("${params.base_dir}"),sqldb2)
    ////oncokbOut = run_oncokb(tmpVarTbl,file("${params.oncokbToken}"))
    //sqldb3 = load_oncokb_output(file("${params.base_dir}"),sqldb2)
    //sqldb4 = clinical_annotation_matching(file("${params.base_dir}"),sqldb3)
    
    // start with cashed db
    sqldb4 = clinical_annotation_matching(file("${params.base_dir}"),file("/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite"))
    sqldb5 = targeted_panel_coordinate_lookup(file("${params.base_dir}"),sqldb4)
    sqldb6 = generate_perPatientOncoKBLevelSorted(file("${params.base_dir}"),sqldb5)

    //sqldbNull = incidence_of_clinically_annotated_mutations_reporting(file("${params.base_dir}"),sqldb6)
    //sqldbNull = MAF_and_prognostic_marker_analysis(file("${params.base_dir}"),sqldb6)
    //sqldbNull = summary_oncokb_by_cancer_type(file("${params.base_dir}"),sqldb6) // has some sql output
    //sqldbNull = clinical_technical_feature_description(file("${params.base_dir}"),sqldb6)

    sqldb7 = filter_and_structure_genie_data_for_modeling(file("${params.base_dir}"),sqldb6)
    //sqldb8 = oncoKB_level1_ML_modeling_by_cType(file("${params.base_dir}"),sqldb7)
    
}

