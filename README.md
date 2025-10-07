# Description
TBD

# History
Idunn is a fork of the Balder repo with code focused on ML modeling of clinically actionable cancer biomarkers

## workflow environment and execution
A conda environment specifying all required dependencies can be utilized as follows:
```
conda env create -f code/balder_conda.yml
conda activate balder_conda_env
```

After the environment is activated, the data processing and analysis pipeline is executed with the following command:
```
nextflow run workflow/main.nf
```

