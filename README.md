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
