# Bass Connections 2023-2024
We are Bass Connections Team working on tracking climate change with satellites and artificial intelligence at Duke University. This repository provides an image classification model using PyTorch. It utilizes a ResNet-50 architecture and is trained on the EuroSAT dataset released by Helber et al. in 2019. Our evaluation metrics include a confusion matrix and accuracy rate on the test data.

## Dataset
*mlrsnet:


## Requirements
*mlrsnet:
We mainly used the aitlas package to deal with the mlrsnet. We create our version of aitlas and please visit via:
https://github.com/Evan-xma/aitlas/tree/master
Follow the instruction to install the aitlas.


## Usage
*mlrsnet:
1. To do EDA, use 'eda.ipynb' in the 'notebooks' folder. This would give you a good sense of the mlrsnet data.
2. After doing EDA, use python files in 'src' folder to train and evaluate a model.
   * m_data.py: This file is to read the dataset from the given specified directory and return dataloaders along with dataset sizes.
   * m_model.py: This file is to prepare the model
   * m_train_baseline.py: This file is to train the model as a baseline.
   * m_test_different_size.py: This file is to train the model with data size from 500 to 65000 to test the influence of data size on the deep learning model performance.
