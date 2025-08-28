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

| Model                  | Dataset   | Finetuning               | Run 1 (%) | Run 2 (%) | Run 3 (%) | Run 4 (%) | Run 5 (%) | Mean (%)    |
| ---------------------- | --------- | ------------------------ | --------- | --------- | --------- | --------- | --------- | ----------- |
| SpatialTransformer     | Physionet | :heavy_multiplication_x: | 54.17     | 55.60     | 57.02     | 55.48     | 57.26     | 55.90+-1.13 |
| SpatialTransformer     | Physionet | :heavy_check_mark:       | 63.93     | 61.79     | 62.26     | 61.31     | 61.79     | 62.21+-0.91 |
| TemporalTransformer    | Physionet | :heavy_multiplication_x: | 65.48     | 64.64     | 65.95     | 63.57     | 64.17     | 64.76+-0.86 |
| TemporalTransformer    | Physionet | :heavy_check_mark:       | 65.83     | 68.10     | 66.43     | 66.19     | 67.26     | 66.76+-0.82 |
| SpatialCNNTransformer  | Physionet | :heavy_multiplication_x: | 60.71     | 61.31     | 60.48     | 62.14     | 60.83     | 61.10+-0.59 |
| SpatialCNNTransformer  | Physionet | :heavy_check_mark:       | 66.55     | 68.21     | 67.14     | 67.38     | 69.52     | 67.76+-1.03 |
| TemporalCNNTransformer | Physionet | :heavy_multiplication_x: | 58.81     | 59.88     | 57.74     | 59.88     | 58.81     | 59.02+-0.80 |
| TemporalCNNTransformer | Physionet | :heavy_check_mark:       | 65.60     | 66.90     | 65.95     | 66.79     | 67.98     | 66.64+-0.83 |
| FusionCnnTransformer   | Physionet | :heavy_multiplication_x: | 61.07     | 61.43     | 61.43     | 64.17     | 63.33     | 62.29+-1.23 |
| FusionCnnTransformer   | Physionet | :heavy_check_mark:       | 67.38     | 64.88     | 65.36     | 65.48     | 66.67     | 65.95+-0.93 |

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
