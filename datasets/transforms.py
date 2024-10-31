import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torchio as tio


def transforms():
    return tio.transforms.Compose(
        [
            tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            tio.transforms.Resize((160, 192, 256)),  # original size was (144, 192, 256)
            tio.transforms.RescaleIntensity(out_min_max=(0, 1), exclude=["label"]),
        ]
    )


def transforms2d(probs=0.25):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            A.OneOf(
                [
                    A.OpticalDistortion(distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
                ],
                p=0.1,
            ),
            A.ColorJitter(),
            A.Blur(blur_limit=2),
        ]
    )


def preprocess2d():
    return A.Compose([A.Normalize(mean=[0.776, 0.631, 0.737], std=[0.131, 0.166, 0.127]), ToTensorV2()])
