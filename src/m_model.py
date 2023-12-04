from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
from data import prepare_data
import contextlib
import yaml

def prepare_model(config):
  # load the model with the parameter in config
  # Prepare the model
  # use the code in the aitlas
  model = ResNet50MultiLabel({
      "num_classes": 60, 
      "learning_rate": config["m_learning_rate"],
      "pretrained": config["m_pretrained"],
      "threshold": config["m_threshold"],
      "metrics": config["m_metrics"]
  })
  model.prepare()
  return model

if __name__ == "__main__":
  with open('../config.yaml') as p:
        config = yaml.safe_load(p)
  resmodel = prepare_model(config)
