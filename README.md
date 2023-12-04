# Bass Connections 2023-2024
We are the Bass Connections Team working on tracking climate change with satellites and artificial intelligence at Duke University. This repository provides an image classification model using PyTorch. It utilizes a ResNet-50 architecture and is trained on the 3 different datasets: MlrsNet, FloodNet, and Brazilian Coffee Scenes.

## Dataset
**MLRSNet: https://github.com/cugbrs/MLRSNet**

MLRSNet provides different perspectives of the world captured from satellites. That is, it is composed of high spatial resolution optical satellite images. MLRSNet contains 109,161 remote sensing images that are annotated into 46 categories, and the number of sample images in a category varies from 1,500 to 3,000. The images have a fixed size of 256×256 pixels with various pixel resolutions (~10m to 0.1m). Moreover, each image in the dataset is tagged with several of 60 predefined class labels, and the number of labels associated with each image varies from 1 to 13.
In the data folder, we provided our train, validation, test split CSV file

**Brazilian Coffee Scenes: http://patreo.dcc.ufmg.br/2017/11/12/brazilian-coffee-scenes-dataset/**

The Brazilian Coffee Scenes dataset is a set of satellite images taken from four counties in the State of Minas Gerais, Brazil - Arceburgo, Guaranesia, Guaxupé and Monte Santo - in 2005. There are two classes of images: Coffee (an image with at least 85% of coffee pixels) and Non-Coffee (an image with less than 10% of coffee pixels). There are 2,876 tiles of 64x64 pixels. The dataset is originally formatted as containing 5 folds with equal splits of Coffee and Non-Coffee images, but our work combines these folds into one set of images that we split into test and training sets. In the data folder, we provide the zip file of images and .txt files with the labels for each image.

## Requirements
**MLRSNet:**

We mainly used the aitlas package to deal with the MLRSNet. We create our version of aitlas and please visit via:
https://github.com/Evan-xma/aitlas/tree/master

Follow the instructions in the aitlas link to install the environment.

To unrar the data file, install the unrar:
```
sudo apt-get install unrar
```


## Usage
**MLRSNet:**
1. To download the data, run the make mrlsnet_Download.
2. To do EDA, use 'mlrsnet_eda.ipynb' in the 'notebooks' folder. This would give you a good sense of the mlrsnet data.
3. After doing EDA, use Python files in 'src' folder to train and evaluate a model.
   * m_data.py: This file reads the dataset from the given specified directory and returns dataloaders.
   * m_model.py: This file is to prepare the model.
   * m_train_baseline.py: This file is to train the model as a baseline.
   * m_test_different_size.py: This file is to train the model with data size from 500 to 65000 to test the influence of data size on the deep learning model performance.
4. make mrlsnet_Train_baseline: train and test the baseline.
5. make mrlsnet_Train_with_different_data_size: train the model by different data sizes.

**Brazilian Coffee Scenes:**
1. To perform EDA, use 'BrazilianCoffeeEDA.ipynb' in the 'notebooks' folder. This shows the distribution of the data.
2. Once EDA is complete, use our analysis notebook.
  * 'CoffeeScenesAnalysis.ipynb' is located in the 'notebooks' folder. This is set up to perform our baseline model and analysis, but can be modified by uncommenting code lines in the code related to modifying bands of the image. We are working on making this more reproducible by splitting it into .py files.
