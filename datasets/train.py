import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import torch
import skimage.io as io
from datasets.transforms import preprocess, transforms
from glob import glob
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
# from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration


class TorchDataset(Dataset):
    def __init__(
            self,
            config,
            mode='train'
    ):
        self.mode = mode
        self.config = config
        self.__init_transforms(**config['transforms'])

        if self.mode == "train":
            self.data = pd.read_csv('data_lung/train_1.csv')
        else:
            self.data = pd.read_csv('data_lung/test_1.csv')

    def __init_transforms(
            self, aug_prob
    ):
        self.preprocess = preprocess()
        self.transforms = transforms(aug_prob) if self.mode == "train" else None

    def __len__(self):
        return len(self.data)

    def load_sample(self, idx):
        image_name, mask_name = self.data.loc[idx]
        image = io.imread(os.path.join(
            image_name
        ))
        mask = Image.open(os.path.join(
            mask_name
        ))
        mask = np.array(mask)
        if mask.shape[-1] == 3:
            mask = mask[:, :, 0].copy()
        # print(mask_name, mask.shape)
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.load_sample(idx)
        if self.transforms is not None:
            # if np.random.random_integers(0, 3) == 0:
            #     image = rgb_perturb_stain_concentration(image)
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        transformed = self.preprocess(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].to(dtype=torch.long) - 1
        # mask = transformed['mask'].to(dtype=torch.long)
        # mask = torch.where(mask > 1, mask - 2, -1)
        return image, mask

    def get_weights(self):
        if self.mode == "train":
            return np.load('/home/user1/Research/user1/segmentation_train/data/weights_not_grouped.npy').tolist()
        else:
            return None


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
    weights = dataset.get_weights()
    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset, weights
