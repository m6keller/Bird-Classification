import cv2
import os
import pandas as pd
from torch.utils.data import Dataset

class BirdImageDataset(Dataset):
    def __init__(self, split: ["train", "test", "val"], path_to_metadata_csv: str, path_to_splits: str, transform = None):
        self._metadata = pd.read_csv(path_to_metadata_csv)
        self._path_to_splits = path_to_splits
        self.split = split    
        self._transform = transform
    
    def __getitem__(self, index):
        rel_file_path = self._metadata.iloc[index]["filepaths"]
        file_path = os.path.join(self._path_to_splits, rel_file_path)
        label = self._metadata.iloc[index]["labels"]
        image = cv2.imread(file_path)
        assert image is not None, f"Could not read image at {file_path}"
        if self._transform:
            image = self._transform(image)

        return image, label
    
    def __len__(self):
        return self._metadata.shape[0]
    
    
if __name__ == "__main__":
    example_dataset = BirdImageDataset("test", "./metadata/birds.csv", "./splits")
    for image, label in example_dataset:
        print(image.shape, label)
        break