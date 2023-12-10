import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, data_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Test'), data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return accuracy, f1, cm


