from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
from m_data import prepare_data
from m_model import prepare_model

import contextlib
import yaml

# This is for the code to train the model as a baseline.

# load config file:
with open('config.yaml') as p:
        config = yaml.safe_load(p)
train_dataset, test_dataset = prepare_data(config)

# Prepare the model
model = prepare_model(config)

# Train and evaluate model
model.train_and_evaluate_model(
    train_dataset=train_dataset,
    epochs=config["m_epochs"],
    model_directory=config["m_model_directory"], 
    val_dataset=test_dataset,
)
