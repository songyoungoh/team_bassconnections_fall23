from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
from m_data import prepare_data
from m_model import prepare_model
import contextlib
import yaml

# load config file:
with open('config.yaml') as p:
        config = yaml.safe_load(p)

# prepare the data
train_dataset, test_dataset = prepare_data(config)

# prepare the model
model = prepare_model(config)

# train the model for different data size
train_len = 500
while train_len < 65000:
    print("---------------------------------------------------Training Length:",train_len)
    model.train_and_evaluate_model_subset(
        train_subset_size = train_len,
        run_id= f"size{train_len}",
        train_dataset=train_dataset,
        epochs=config["m_epochs"],
        model_directory=config["m_model_directory"], 
        val_dataset=test_dataset,
    )
    # When data size is big enough, the small increase in data size will not influence the performance
    if train_len < 2000:
        train_len += 1000
    elif train_len < 6000: 
        train_len +=2000
    else:
        train_len += 4000
