from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset
import torch
import torchio as tio
from utils import ROOT_PATH
import SimpleITK as sitk
import torch.nn.functional as F

from .transforms import transforms

sitk.ProcessObject_SetGlobalWarningDisplay(False)


class iSegDataset(Dataset):
    def __init__(self, config, transforms, mode="train"):
        self.mode = mode
        self.config = config

        self.data_path = ROOT_PATH / "data/iSeg-2019/"
        if mode == "train" or mode == "val":
            self.data_path = self.data_path / "iSeg-2019-Training"
        elif mode == "test":
            self.data_path = self.data_path / "iSeg-2019-Validation"

        self.transforms = transforms
        self.subjects = []
        self.load_data()
        if mode == "train":
            self.subjects = self.subjects[:6]
        elif mode == "val":
            self.subjects = self.subjects[6:]

    def load_data(self):
        data = defaultdict(dict)

        for subject_path in self.data_path.iterdir():
            if subject_path.suffix == ".img":
                _, idx, type = subject_path.stem.split("-")
                data[int(idx)][type] = subject_path

        for idx, subject_dict in data.items():
            t1_image = tio.ScalarImage(subject_dict["T1"])
            t2_image = tio.ScalarImage(subject_dict["T2"])
            joined_image = torch.cat([t1_image.tensor, t2_image.tensor], dim=0)

            if self.mode == "test":
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=joined_image),  # [channels=2, depth, height, width]
                )
            else:
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=joined_image),  # [channels=2, depth, height, width]
                    label=tio.ScalarImage(subject_dict["label"]),  # [1, depth, height, width]
                )

            self.subjects.append(subject)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        """
        There are 2 channels: T1 and T2
        Returns dict of:
            - image (Tensor): image of size [channels, depth, height, width]
            - label (Tensor): labeled pixels of size [depth, height, width]
        """
        subject = self.subjects[idx]
        if self.transforms is not None:
            subject = self.transforms(subject)

        image = subject.image.tensor
        label = subject.label.tensor.squeeze(0) if self.mode != "test" else None

        return {"image": image, "label": label}


def collate_fn(batch: list[dict]):
    """
    Returns:
        - images (Tensor): image of size [batch_size, channels, depth, height, width]
        - labels (Tensor): labeled pixels of size [batch_size, depth, height, width]
    """
    images = []
    labels = []

    for item in batch:
        images.append(item["image"])
        if item["label"] is not None:
            labels.append(item["label"])

    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0) if len(labels) else None
    return images, labels


def get_dataloader(config, mode="train"):
    dataset = iSegDataset(
        config,
        transforms=transforms(),
        mode=mode,
    )

    print(f"Dataset partition {mode} of size: {len(dataset)}")
    dataloader = tio.SubjectsLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config["dataloader"],
    )
    return dataloader
