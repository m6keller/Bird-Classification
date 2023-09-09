import cv2
import pandas as pd
from torch.utils.data import Dataset

class BirdImageDataset(Dataset):
    def __init__(self, split: ["train", "test", "val"], path_to_metadata_csv: str):
        self._metadata = pd.read_csv(path_to_metadata_csv)
        self._split = split    
    
    def __getitem__(self, index):
        file_path = self._metadata.iloc[index, 1]
        label = self._metadata.iloc[index, 0]
        image = cv2.open(file_path)
        if self._transform:
            image = self._transform(image)

        return image, label
    
    def __len__(self):
        return self._metadata.shape[0]