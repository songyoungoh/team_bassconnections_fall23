from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
import contextlib
import yaml

# this is for the code to train the model as a base line.

# load config file:
with open('../config.yaml') as p:
        config = yaml.safe_load(p)

# preprocessing the data

train_dataset = MLRSNetMultiLabelDataset({
    "batch_size": config["batch_size"],
    "shuffle": config["shuffle"],
    "num_workers": config["num_workers"],
    "data_dir": config["data_dir"],
    "csv_file": config["train_csv_file"]
})
train_dataset.transform = ResizeCenterCropFlipHVToTensor()

# Use parameters from config for test dataset
# Note: Assuming similar parameters for the test dataset, adjust if needed
test_dataset = MLRSNetMultiLabelDataset({
    "batch_size": config["batch_size"],
    "shuffle": config["shuffle"],
    "num_workers": config["num_workers"],
    "data_dir": config["data_dir"],
    "csv_file": config["test_csv_file"]
})
test_dataset.transform = ResizeCenterCropToTensor()

# Print dataset sizes
print("Data size - Training:", len(train_dataset), ", Testing:", len(test_dataset))

# Prepare the model
model = ResNet50MultiLabel({
    "num_classes": 60,  # Assuming 60 classes, change if different
    "learning_rate": config["learning_rate"],
    "pretrained": config["pretrained"],
    "threshold": config["threshold"],
    "metrics": config["metrics"]
})
model.prepare()

# Train and evaluate model
epochs = 20
model.train_and_evaluate_model(
    train_dataset=train_dataset,
    epochs=epochs,
    model_directory=model_directory,  # Define or load this variable
    val_dataset=test_dataset,
)
