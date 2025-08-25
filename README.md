# VS Code Setup

Create virtual environment  
`python -m venv venv`

Activate virtual environment  
`.\venv\Scripts\Activate`

Make sure that pip installation points to virtual environment  
`pip --version`

Install dependencies  
`pip install -r requirements.txt`

Install recommended extensions specified in  
`extensions.json`

# Project structure

## :file_folder: notebooks/

This folder contains jupyter notebooks for multi-task learning algorithms and transformer models testing on different datasets. Check each notebook for more info.

## :file_folder: scripts/

This folder contains all model implementations, utilities and other useful scripts.

## :pencil: Accuracies so far

### Multi-task learning

| Model                                      | Dataset    | Evaluation    | Accuracy (%) |
| ------------------------------------------ | ---------- | ------------- | ------------ |
| MultiTaskLinearClassifier                  | BCI 2008 B | Cross-session | 77           |
| MultiTaskLinearClassifier                  | BCI 2008 B | Cross-subject | 74           |
| MultiTaskLinearClassifierWithDataSelection | BCI 2008 B | Cross-session | 79           |
| MultiTaskLinearClassifierWithDataSelection | BCI 2008 B | Cross-subject | 77           |
| MultiTaskLinearClassifier                  | BCI IV 2a  | Cross-session | 69           |
| MultiTaskLinearClassifierWithDataSelection | BCI IV 2a  | Cross-session | 74           |

### Transformers

| Model                  | Dataset   | Finetuning                    | Run 1 (%) | Run 2 (%) | Run 3 (%) | Run 4 (%) | Run 5 (%) | Mean (%)      |
| ---------------------- | --------- | ----------------------------- | --------- | --------- | --------- | --------- | --------- | ------------- |
| SpatialTransformer     | Physionet | :negative_squared_cross_mark: | 56.55     | 58.69     | 57.62     | 57.38     | 56.19     | 57.29 +- 0.98 |
| SpatialTransformer     | Physionet | :heavy_check_mark:            | 63.93     | 62.74     | 62.50     | 64.40     | 62.98     | 63.31 +- 0.82 |
| TemporalTransformer    | Physionet | :negative_squared_cross_mark: | 66.43     | 63.93     | 64.88     | 64.52     | 66.67     | 65.29 +- 1.20 |
| TemporalTransformer    | Physionet | :heavy_check_mark:            | 68.45     | 65.71     | 66.67     | 67.02     | 67.14     | 67.00 +- 0.99 |
| SpatialCNNTransformer  | Physionet | :negative_squared_cross_mark: | 61.79     | 62.26     | 61.79     | 62.62     | 63.57     | 62.40 +- 0.74 |
| SpatialCNNTransformer  | Physionet | :heavy_check_mark:            | 66.43     | 66.19     | 68.10     | 65.71     | 68.69     | 67.02 +- 1.29 |
| TemporalCNNTransformer | Physionet | :negative_squared_cross_mark: | 57.14     | 61.67     | 55.95     | 59.52     | 58.81     | 58.62 +- 2.20 |
| TemporalCNNTransformer | Physionet | :heavy_check_mark:            | 64.29     | 64.88     | 66.79     | 64.76     | 66.07     | 65.36 +- 1.03 |
| FusionCnnTransformer   | Physionet | :negative_squared_cross_mark: | 61.31     | 61.90     | 61.19     | 60.95     | 62.26     | 61.52 +- 0.54 |
| FusionCnnTransformer   | Physionet | :heavy_check_mark:            | 64.52     | 66.07     | 67.38     | 65.95     | 67.26     | 66.24 +- 1.16 |

## :warning: Deprecated scripts

### **download.py**

This script downloads raw datasets related to motor imagery. It accepts dataset name as a starting param.  
If none is provided, it will download all datasets. Supported dataset names are:  
`bci3a` - for BCI Competition III 3a  
`bci2a` - for BCI Competition IV 2a  
`bci2b` - for BCI Competition IV 2b  
`physionet` - for Physionet

### **epochs.py**

This script extracts expochs for selected dataset and saves them in predefined directory. It accepts dataset name as a starting param. If none is provided, it will not do anything. Supported dataset names are:  
`bci3a` - for BCI Competition III 3a  
`bci2a` - for BCI Competition IV 2a  
`bci2b` - for BCI Competition IV 2b  
`physionet` - for Physionet

### **train.py**

This scrips performs 5 fold cross validation for selected transformer model. The final model accuracy is mean from all 5 fold accuracies. Model name is accepted as a starting param. If none is provided, it will not do anything. Supported model names are:  
`spatial` - for SpatialTransformer  
`temporal` - for TemporalTransformer  
`spatialcnn` - for SpatialCNNTransformer  
`temporalcnn` - for TemporalCNNTransformer  
`fusion` - for FusionCNNTransformer
