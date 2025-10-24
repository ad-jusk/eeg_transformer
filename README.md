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

## :pencil: Best accuracies so far

### Multi-task learning algorithms

| Model                                      | Dataset    | Evaluation    | Accuracy (%) |
| ------------------------------------------ | ---------- | ------------- | ------------ |
| MultiTaskLinearClassifier                  | BCI 2008 B | Cross-session | 79.38        |
| MultiTaskLinearClassifier                  | BCI 2008 B | Cross-subject | 75.28        |
| MultiTaskLinearClassifierWithDataSelection | BCI 2008 B | Cross-session | 78.19        |
| MultiTaskLinearClassifierWithDataSelection | BCI 2008 B | Cross-subject | 75.55        |
| MultiTaskLinearClassifier                  | BCI IV 2a  | Cross-session | 70.68        |
| MultiTaskLinearClassifierWithDataSelection | BCI IV 2a  | Cross-session | 70.68        |

### Transformer models

#### 6s epochs

| Model                  | Dataset   | Finetuning               | Run 1 (%) | Run 2 (%) | Run 3 (%) | Run 4 (%) | Run 5 (%) | Mean (%) |
| ---------------------- | --------- | ------------------------ | --------- | --------- | --------- | --------- | --------- | -------- |
| SpatialTransformer     | Physionet | :heavy_multiplication_x: | 57.14     | 58.89     | 57.46     | 57.94     | 57.78     | 57.84    |
| SpatialTransformer     | Physionet | :heavy_check_mark:       | 68.41     | 67.78     | 65.24     | 67.14     | 65.56     | 66.83    |
| TemporalTransformer    | Physionet | :heavy_multiplication_x: | 70.16     | 67.46     | 70.48     | 68.25     | 69.52     | 69.17    |
| TemporalTransformer    | Physionet | :heavy_check_mark:       | 72.38     | 73.97     | 70.48     | 72.70     | 70.16     | 71.94    |
| SpatialCNNTransformer  | Physionet | :heavy_multiplication_x: | 66.35     | 65.08     | 66.19     | 63.17     | 65.08     | 65.17    |
| SpatialCNNTransformer  | Physionet | :heavy_check_mark:       | 71.11     | 68.10     | 67.46     | 65.40     | 66.67     | 67.75    |
| TemporalCNNTransformer | Physionet | :heavy_multiplication_x: | 65.56     | 65.71     | 64.44     | 66.51     | 65.71     | 65.59    |
| TemporalCNNTransformer | Physionet | :heavy_check_mark:       | 70.32     | 72.22     | 71.75     | 73.17     | 72.86     | 72.06    |
| FusionCnnTransformer   | Physionet | :heavy_multiplication_x: | 66.67     | 63.02     | 63.97     | 67.14     | 62.70     | 64.70    |
| FusionCnnTransformer   | Physionet | :heavy_check_mark:       | 71.59     | 71.59     | 68.10     | 69.05     | 70.95     | 70.25    |

#### 3s epochs

| Model                  | Dataset   | Finetuning               | Run 1 (%) | Run 2 (%) | Run 3 (%) | Run 4 (%) | Run 5 (%) | Mean (%) |
| ---------------------- | --------- | ------------------------ | --------- | --------- | --------- | --------- | --------- | -------- |
| SpatialTransformer     | Physionet | :heavy_multiplication_x: | 54.92     | 52.70     | 53.49     | 53.17     | 56.98     | 54.25    |
| SpatialTransformer     | Physionet | :heavy_check_mark:       | 62.06     | 60.48     | 61.59     | 64.29     | 64.60     | 62.60    |
| TemporalTransformer    | Physionet | :heavy_multiplication_x: | 62.06     | 62.70     | 63.65     | 62.54     | 62.70     | 62.73    |
| TemporalTransformer    | Physionet | :heavy_check_mark:       | 66.51     | 67.14     | 68.89     | 65.56     | 64.13     | 66.44    |
| SpatialCNNTransformer  | Physionet | :heavy_multiplication_x: | 61.27     | 60.48     | 62.38     | 63.65     | 60.79     | 61.71    |
| SpatialCNNTransformer  | Physionet | :heavy_check_mark:       | 65.40     | 64.92     | 69.21     | 62.06     | 67.78     | 65.87    |
| TemporalCNNTransformer | Physionet | :heavy_multiplication_x: | 56.19     | 51.43     | 57.46     | 54.13     | 57.78     | 55.40    |
| TemporalCNNTransformer | Physionet | :heavy_check_mark:       | 63.81     | 66.51     | 62.22     | 63.97     | 65.40     | 64.38    |
| FusionCnnTransformer   | Physionet | :heavy_multiplication_x: | 61.90     | 60.63     | 65.08     | 60.48     | 63.17     | 62.25    |
| FusionCnnTransformer   | Physionet | :heavy_check_mark:       | 66.03     | 66.03     | 65.56     | 64.44     | 64.98     | 65.81    |

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
