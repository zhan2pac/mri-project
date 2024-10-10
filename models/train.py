# import json
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import cv2
# import os
# import torch
# from datasets.transforms import preprocess, transforms
# from glob import glob
# import pandas as pd
# from sklearn.utils.class_weight import compute_class_weight
# from PIL import Image


# class TorchDataset(Dataset):
#     def __init__(
#             self,
#             config,
#             mode='train'
#     ):
#         self.mode = mode
#         self.config = config
#         self.data_dir = config["data_dir"]
#         self.__init_transforms(**config['transforms'])

#         if self.mode == "train":
#             self.data = ['00.png']
#             # self.data = pd.read_csv(config['train_data'])
#         else:
#             self.data = ['01.png']
#             # self.data = pd.read_csv(config['train_data'])

#     def __init_transforms(
#             self, aug_prob, side_size
#     ):
#         self.preprocess = preprocess(side_size)
#         self.transforms = transforms(aug_prob) if self.mode == "train" else None

#     def __len__(self):
#         return len(self.data)

#     def load_sample(self, idx):
#         image_name = self.data[idx]
#         # if not os.path.exists(os.path.join(self.data_dir, 'images', image_name)):
#         #     raise ValueError(f"{os.path.join(self.data_dir, 'images', image_name)} doesn't exist")
#         image = cv2.imread(os.path.join(self.data_dir, 'images', image_name))
#         mask = Image.open(os.path.join(self.data_dir, 'masks', image_name))
#         mask = np.array(mask)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         return image, mask

#     def __getitem__(self, idx):
#         image, mask = self.load_sample(idx)
#         if self.transforms is not None:
#             transformed = self.transforms(image=image, mask=mask)
#             image = transformed['image']
#             mask = transformed['mask']
#         transformed = self.preprocess(image=image, mask=mask)
#         return transformed['image'], transformed['mask'].to(torch.int64)

#     # def get_weights(self):
#     #     person = self.data["person"]
#     #     binary = self.data["binary"]
#     #     binary = binary[binary.notnull()].values
#     #     person = person[person.notnull()].values
#     #     binary_weight = compute_class_weight(
#     #         class_weight = "balanced",
#     #         classes = np.array([0, 1]),
#     #         y = binary
#     #     )
#     #     person_weight = compute_class_weight(
#     #         class_weight = "balanced",
#     #         classes = np.array([0, 1, 2]),
#     #         y = person
#     #     )
#     #     return binary_weight, person_weight


# def collate_fn(batch):
#     items = list(zip(*batch))
#     return [torch.stack(item) for item in items]

# def create_loader_dataset(
#         config,
#         mode='train'
# ):
    
#     dataset = TorchDataset(
#         config, mode=mode
#     )

#     dataloader = DataLoader(
#         dataset,
#         shuffle=(mode == "train"),
#         collate_fn=collate_fn,
#         **config['dataloader']
#     )
#     return dataloader, dataset
