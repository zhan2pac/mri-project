from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torchio as tio
import SimpleITK as sitk

sitk.ProcessObject_SetGlobalWarningDisplay(False)

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent


class iSegDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.mode = mode
        self.config = config

        self.data_path = ROOT_PATH / "data/iSeg-2019/"
        if mode == "train":
            self.data_path = self.data_path / "iSeg-2019-Training"
        else:
            self.data_path = self.data_path / "iSeg-2019-Validation"

        self.transforms = None

        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        data = defaultdict(dict)

        for subject_path in self.data_path.iterdir():
            if subject_path.suffix == ".img":
                _, idx, type = subject_path.stem.split("-")
                data[idx][type] = tio.ScalarImage(subject_path)

        for idx in data.keys():
            t1_image = data[idx]["T1"].tensor
            t2_image = data[idx]["T2"].tensor
            label_image = data[idx]["label"].tensor  # [1, depth, height, width]

            image = torch.cat([t1_image, t2_image], dim=0)
            image = image.unsqueeze(0)  # [1, channels=2, depth, height, width]

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
        label = self.labels[idx]

        if self.transforms is not None:
            image = self.transforms(image)

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


def create_loader_dataset(config, mode="train"):
    dataset = TorchDataset(config, mode=mode)
    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config["dataloader"],
    )
    return dataloader, dataset
