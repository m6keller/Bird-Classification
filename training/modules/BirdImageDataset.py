import cv2
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BirdImageDataset(Dataset):
    def __init__(self, split: ["train", "test", "valid"], path_to_metadata_csv: str, path_to_splits: str, transform = None):
        metadata = pd.read_csv(path_to_metadata_csv)
        filtered_metadata = metadata[metadata["data set"] == split]
        self._metadata = filtered_metadata
        self._path_to_splits = path_to_splits
        self.split = split    
        self._transform = transform
    
    def __getitem__(self, index):
        rel_file_path = self._metadata.iloc[index]["filepaths"]
        file_path = os.path.join(self._path_to_splits, rel_file_path)
        label = int(self._metadata.iloc[index]["class id"])

        image = Image.open(file_path)
        # image = cv2.imread(file_path)
        assert image is not None, f"Could not read image at {file_path}"
        # convert image to rgb
        image = image.convert("RGB")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._transform:
            # image = self._transform(torch.from_numpy(image))
            image = self._transform(image)
        # might have to encode labels as an integer with LabelEncoder
        return image, label
    
    # def __getitem__(self, index):
    #     rel_file_path = self._metadata.iloc[index]["filepaths"]
    #     file_path = os.path.join(self._path_to_splits, rel_file_path)
    #     label = int(self._metadata.iloc[index]["class id"])
    #     # with open(file_path, 'rb') as f:
    #     #     img = cv2.open(f)
    #     #     img = img.convert('RGB')
    #     image
    #     if self._transform is not None:
    #         img = self._transform(img)
    #     label = torch.tensor(label, dtype=float)
    #     return img, label


    
    def __len__(self):
        return self._metadata.shape[0]
    
    
if __name__ == "__main__":
    example_dataset = BirdImageDataset("test", "./metadata/birds.csv", "./splits")
    for image, label in example_dataset:
        print(image.shape, label)
        break