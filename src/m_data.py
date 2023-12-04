from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
import contextlib
import yaml

# this is for the code to train the model as a base line.
def prepare_data(config):
    # preprocessing the data
    # for the train data
    train_dataset = MLRSNetMultiLabelDataset({
        "batch_size": config["m_batch_size"],
        "shuffle": config["m_shuffle"],
        "num_workers": config["m_num_workers"],
        "data_dir": config["m_data_dir"],
        "csv_file": config["m_train_csv_file"]
    })
    train_dataset.transform = ResizeCenterCropFlipHVToTensor()
    # for the test data
    test_dataset = MLRSNetMultiLabelDataset({
        "batch_size": config["m_batch_size"],
        "shuffle": config["m_shuffle"],
        "num_workers": config["m_num_workers"],
        "data_dir": config["m_data_dir"],
        "csv_file": config["m_test_csv_file"]
    })
    test_dataset.transform = ResizeCenterCropToTensor()
    return train_dataset, test_dataset

if __name__ == "__main__":
  with open('../config.yaml') as p:
        config = yaml.safe_load(p)
  train_dataset, test_dataset = prepare_data(config)
  print("Data size - Training:", len(train_dataset), ", Testing:", len(test_dataset))
