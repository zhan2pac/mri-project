import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def transforms(probs=0.25):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        A.OneOf([
            A.OpticalDistortion(
                distort_limit=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ], p=0.1),
        A.ColorJitter(),
        A.Blur(blur_limit=2),
    ])


def preprocess():
    return A.Compose([
        A.Normalize(mean=[0.776, 0.631, 0.737], std=[0.131, 0.166, 0.127]),
        ToTensorV2()
    ])
