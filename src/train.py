from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
from data import prepare_data
import contextlib
import yaml

# This is for the code to train the model as a baseline.

# load config file:
with open('../config.yaml') as p:
        config = yaml.safe_load(p)
train_dataset, test_dataset = prepare_data(config)

# Prepare the model
model = ResNet50MultiLabel({
    "num_classes": 60,  # Assuming 60 classes, change if different
    "learning_rate": config["m_learning_rate"],
    "pretrained": config["m_pretrained"],
    "threshold": config["m_threshold"],
    "metrics": config["m_metrics"]
})
model.prepare()

# Train and evaluate model
model.train_and_evaluate_model(
    train_dataset=train_dataset,
    epochs=config["m_epochs"],
    model_directory=config["model_directory"],  # Define or load this variable
    val_dataset=test_dataset,
)
