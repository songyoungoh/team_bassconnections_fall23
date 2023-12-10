import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from torch.utils.data import random_split
from collections import Counter
import pandas as pd
from data import data_create
from model import our_ResNet
from evaluate import evaluate_model
import yaml
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(data_dir):
    # Load datasets
    dataloaders, dataset_sizes = data_create(data_dir)

    # Initialize and train the model
    model1 = our_ResNet(num_classes=2)
    model1.train_model(dataloaders, dataset_sizes, num_epochs=10)
    model1.plot_losses()

    # Evaluate the trained model
    accuracy, f1, cm = evaluate_model(model1, data_dir)
    
    print("Test Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    
    class_names = dataloaders['Test'].dataset.classes

    return accuracy, f1, cm


if __name__ == "__main__":
    # Load the configuration file
    with open('/scratch/public/floodnet/team_bassconnections/config.yaml') as p:
        config = yaml.safe_load(p)

    data_dir = config['data_dir']

    # Run the main function
    run(data_dir)

