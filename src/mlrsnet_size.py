from aitlas.datasets import MLRSNetMultiLabelDataset
from aitlas.models import ResNet50MultiLabel
from aitlas.transforms import ResizeCenterCropFlipHVToTensor, ResizeCenterCropToTensor
from aitlas.utils import image_loader
import sys
import contextlib
from datetime import datetime

# current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# sys.stdout = open(f'/data/scratch/public/mlrsnet/output/output_{current_datetime}.txt', "w")
# logger = logging.getLogger(__name__)
# FileOutputHandler = logging.FileHandler(f'/data/scratch/public/mlrsnet/output/output_{current_datetime}.txt')
# logger.addHandler(FileOutputHandler)
# logging.basicConfig(filename='std.log', filemode='w')


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
model_directory = "/data/scratch/public/mlrsnet/model"
model_config = {
    "num_classes": 60, 
    "learning_rate": 0.001,
    "pretrained": False, 
    "threshold": 0.5, 
    "metrics": ["accuracy", "precision", "recall", "f1_score"]
}
model = ResNet50MultiLabel(model_config)
model.prepare()


train_len = 500
while train_len < 65000:
    print("---------------------------------------------------Training Length:",train_len)
    model.train_and_evaluate_model_subset(
        train_dataset=train_dataset,
        epochs=epochs,
        model_directory=model_directory,
        val_dataset=test_dataset,
        train_subset_size = train_len,
        run_id= f"size{train_len}",
    )
    if train_len < 2000:
        train_len += 1000
    elif train_len < 6000: 
        train_len +=2000
    else:
        train_len += 4000
