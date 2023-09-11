import numpy as np
from modules.BirdImageDataset import BirdImageDataset
# import torch transforms
from torchvision.transforms import Resize, ToTensor, Compose

TRAIN_SIZE = 84636 # find splits/train -type f | wc -l
TEST_SIZE = 2626

def get_train_test_split(path_to_metadata_csv: str, path_to_splits, resize = (224, 224), train_size: int = TRAIN_SIZE, test_size: int = TEST_SIZE):
    transform = Compose([Resize(size=resize)])
    
    x_train = np.empty((train_size, *resize, 3), dtype=float)
    y_train = np.empty((train_size), dtype=int)

    x_test = np.empty((test_size, *resize, 3), dtype=float)
    y_test = np.empty((test_size), dtype=int)
    train_dataset = BirdImageDataset(split="train", path_to_metadata_csv=path_to_metadata_csv, path_to_splits=path_to_splits, transform=transform)
    test_dataset = BirdImageDataset(split="test", path_to_metadata_csv=path_to_metadata_csv, path_to_splits=path_to_splits, transform=transform)

    count = 0
    for i, (image, label) in enumerate(train_dataset):
        x_train[i] = image
        y_train[i] = label
        count += 1
        if count > 100:
            break
        
    count = 0
    for i, (image, label) in enumerate(test_dataset):
        x_test[i] = image
        y_test[i] = label
        count += 1
        if count > 100:
            break
        
    return x_train, y_train, x_test, y_test