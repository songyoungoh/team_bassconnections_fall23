from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
from data import prepare_data
import contextlib
import yaml

# load config file:
with open('../config.yaml') as p:
        config = yaml.safe_load(p)

# prepare the data
train_dataset, test_dataset = prepare_data(config)

# prepare the model
model_directory = "/data/scratch/public/mlrsnet/model"
model = ResNet50MultiLabel({
    "num_classes": 60, # mlrsnet has 60 different classes
    "learning_rate": config["m_learning_rate"],
    "pretrained": config["m_pretrained"],
    "threshold": config["m_threshold"],
    "metrics": config["m_metrics"]
})
model.prepare()

train_len = 500
while train_len < 65000:
    print("---------------------------------------------------Training Length:",train_len)
    model.train_and_evaluate_model_subset(
        train_subset_size = train_len,
        run_id= f"size{train_len}",
        train_dataset=train_dataset,
        epochs=config["m_epochs"],
        model_directory=config["model_directory"], 
        val_dataset=test_dataset,
    )
    if train_len < 2000:
        train_len += 1000
    elif train_len < 6000: 
        train_len +=2000
    else:
        train_len += 4000
