# Bass Connections 2023-2024
We are Bass Connections Team working on tracking climate change with satellites and artificial intelligence at Duke University. This repository provides an image classification model using PyTorch. It utilizes a ResNet-50 architecture and is trained on the 3 different dataset: MlrsNet, FloodNet, Brazilian Coffee Scenes.

## Dataset
**mlrsnet: https://github.com/cugbrs/MLRSNet**

MLRSNet provides different perspectives of the world captured from satellites. That is, it is composed of high spatial resolution optical satellite images. MLRSNet contains 109,161 remote sensing images that are annotated into 46 categories, and the number of sample images in a category varies from 1,500 to 3,000. The images have a fixed size of 256×256 pixels with various pixel resolutions (~10m to 0.1m). Moreover, each image in the dataset is tagged with several of 60 predefined class labels, and the number of labels associated with each image varies from 1 to 13.
In the data folder, we provided our train, validation, test split CSV file

## Requirements
**mlrsnet:**

We mainly used the aitlas package to deal with the mlrsnet. We create our version of aitlas and please visit via:
https://github.com/Evan-xma/aitlas/tree/master

Follow the instruction to install the aitlas.


## Usage
**mlrsnet:**
1. To download the data, run the Makefile: mrlsnet_Download.
2. To do EDA, use 'mlrsnet_eda.ipynb' in the 'notebooks' folder. This would give you a good sense of the mlrsnet data.
3. After doing EDA, use python files in 'src' folder to train and evaluate a model.
   * m_data.py: This file is to read the dataset from the given specified directory and return dataloaders along with dataset sizes.
   * m_model.py: This file is to prepare the model
   * m_train_baseline.py: This file is to train the model as a baseline.
   * m_test_different_size.py: This file is to train the model with data size from 500 to 65000 to test the influence of data size on the deep learning model performance.
4. Makefile: mrlsnet_Train_baseline: train and test the baseline.
5. Makefile: mrlsnet_Train_with_different_data_size: train the model by different data sizes.
