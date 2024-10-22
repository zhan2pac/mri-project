from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torchio as tio
from utils import ROOT_PATH
import SimpleITK as sitk
import torch.nn.functional as F

sitk.ProcessObject_SetGlobalWarningDisplay(False)


class iSegDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.mode = mode
        self.config = config

        self.data_path = ROOT_PATH / "data/iSeg-2019/"
        if mode == "train" or mode == "val":
            self.data_path = self.data_path / "iSeg-2019-Training"
        elif mode == "test":
            self.data_path = self.data_path / "iSeg-2019-Validation"

        self.transforms = None
        self.resize = tio.transforms.Resize((192, 192, 256))

        self.images = []
        self.labels = []
        self.load_data()

    def normalize(self, image):
        return (image - image.min()) / (image.max() - image.min())

    def load_data(self):
        data = defaultdict(dict)

        for subject_path in self.data_path.iterdir():
            if subject_path.suffix == ".img":
                _, idx, type = subject_path.stem.split("-")
                data[int(idx)][type] = tio.ScalarImage(subject_path)

        for idx, subject in data.items():
            subject = data[idx]
            t1_image = subject["T1"].tensor
            t2_image = subject["T2"].tensor

            image = torch.cat([t1_image, t2_image], dim=0)
            image = self.resize(image)
            image = self.normalize(image.unsqueeze(0))  # [1, channels=2, depth, height, width]

            if self.mode == "test":
                self.images.append(image)
                continue

            label_image = subject["label"].tensor
            label_image = self.resize(label_image).long()  # [1, depth, height, width]

            if self.mode == "train" and (idx <= 6):
                self.images.append(image)
                self.labels.append(label_image)

            elif self.mode == "val" and (idx > 6):
                self.images.append(image)
                self.labels.append(label_image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        There are 2 channels: T1 and T2
        Returns:
            - image (Tensor): image of size [1, channels, depth, height, width]
            - label (Tensor): labeled pixels of size [1, depth, height, width]
        """
        image = self.images[idx]
        if self.transforms is not None:
            image = self.transforms(image)

        if self.mode == "test":
            return image

        label = self.labels[idx]
        return image, label


def collate_fn(batch: list[tuple]):
    batched_images = []
    batched_labels = []

    for image, label in batch:
        batched_images.append(image)
        batched_labels.append(label)

    batched_images = torch.cat(batched_images, dim=0)
    batched_labels = torch.cat(batched_labels, dim=0)
    return batched_images, batched_labels


def get_dataloader(config, mode="train"):
    dataset = iSegDataset(config, mode=mode)
    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config["dataloader"],
    )
    return dataloader
