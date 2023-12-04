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
    "batch_size": config["m_batch_size"],
    "shuffle": config["m_shuffle"],
    "num_workers": config["m_num_workers"],
    "data_dir": config["m_data_dir"],
    "csv_file": config["m_train_csv_file"]
})
train_dataset.transform = ResizeCenterCropFlipHVToTensor()

# Use parameters from config for test dataset
# Note: Assuming similar parameters for the test dataset, adjust if needed
test_dataset = MLRSNetMultiLabelDataset({
    "batch_size": config["m_batch_size"],
    "shuffle": config["m_shuffle"],
    "num_workers": config["m_num_workers"],
    "data_dir": config["m_data_dir"],
    "csv_file": config["m_test_csv_file"]
})
test_dataset.transform = ResizeCenterCropToTensor()

# Print dataset sizes
print("Data size - Training:", len(train_dataset), ", Testing:", len(test_dataset))

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
    model_directory=model_directory,  # Define or load this variable
    val_dataset=test_dataset,
)
