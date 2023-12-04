from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
import contextlib
from datetime import datetime

train_dataset_config = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": 4,
    "data_dir": "/data/scratch/public/mlrsnet/data/Images",
    "csv_file": "/data/scratch/public/mlrsnet/data/mlrsnet_train.csv"
}

train_dataset = MLRSNetMultiLabelDataset(train_dataset_config)
train_dataset.transform = ResizeCenterCropFlipHVToTensor() 

test_dataset_config = {
    "batch_size": 64,
    "shuffle": False,
    "num_workers": 4,
    "data_dir": "/data/scratch/public/mlrsnet/data/Images",
    "csv_file": "/data/scratch/public/mlrsnet/data/mlrsnet_test.csv",
    "transforms": ["aitlas.transforms.ResizeCenterCropToTensor"]
}

test_dataset = MLRSNetMultiLabelDataset(test_dataset_config)
print("data size:", len(train_dataset), len(test_dataset))

epochs = 20
model_config = {
    "num_classes": 60, 
    "learning_rate": 0.001,
    "pretrained": False, 
    "threshold": 0.5, 
    "metrics": ["accuracy", "precision", "recall", "f1_score"]
}
model = ResNet50MultiLabel(model_config)
model.prepare()
model.train_and_evaluate_model(
    train_dataset=train_dataset,
    epochs=epochs,
    model_directory=model_directory,
    val_dataset=test_dataset,
    train_subset_size = train_len,
    run_id= f"size{train_len}",
)
