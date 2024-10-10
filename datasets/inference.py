import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import torch
from datasets.transforms import preprocess, transforms
from glob import glob
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


class TorchDataset(Dataset):
    def __init__(
            self,
            config,
            mode='train'
    ):
        self.img_dir = "/home/arseniy.zemerov/Research/ilya.naletov/Data"
        self.mode = mode
        self.config = config
        data_dir = "/home/arseniy.zemerov/Research/arseniy.zemerov/NaiveUniform/Uniform/{}/*.csv"
        self.__init_transforms(**config['transforms'])

        self.data = pd.read_csv(config["data_path"])
        self.total_len = len(self.data)

    def __init_transforms(
            self, aug_prob, side_size
    ):
        self.preprocess = preprocess()
        self.transforms = transforms(aug_prob) if self.mode == "train" else None

    def __len__(self):
        return self.total_len

    def load_sample(self, idx):
        image_path, person, binary = self.data.iloc[idx]
        image_path = image_path.replace("jpg", "npy")
        if not os.path.exists(os.path.join(self.img_dir, image_path)):
            raise ValueError(f"{os.path.join(self.img_dir, image_path)} doesn't exist")
        image = np.load(os.path.join(self.img_dir, image_path))
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        return image, person, binary, image_path

    def __getitem__(self, idx):
        image, person, binary, image_path = self.load_sample(idx)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        image = self.preprocess(image=image)['image']
        return image, torch.as_tensor(binary), torch.as_tensor(person), torch.as_tensor(idx)

    def get_weights(self):
        person = self.data["person"]
        binary = self.data["binary"]
        binary = binary[binary.notnull()].values
        person = person[person.notnull()].values
        binary_weight = compute_class_weight(
            class_weight = "balanced",
            classes = np.array([0, 1]),
            y = binary
        )
        person_weight = compute_class_weight(
            class_weight = "balanced",
            classes = np.array([0, 1, 2]),
            y = person
        )
        return binary_weight, person_weight


def collate_fn(batch):
    items = list(zip(*batch))
    return [torch.stack(item) for item in items]

def create_loader_dataset(
        config,
        mode='train'
):
    
    dataset = TorchDataset(
        config, mode=mode
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=False,
        **config['dataloader']
    )
    weights = dataset.get_weights()
    return dataloader, dataset, weights
