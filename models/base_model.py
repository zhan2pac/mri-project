import torch.nn as nn
import torch
import timm
from copy import deepcopy
from timm import create_model
import torchvision
import segmentation_models_pytorch as smp
# from .models.axialnet import MedT


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()

        self.model = smp.create_model(
            config['model_type'],
            encoder_name=config['encoder_name'],      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",               # use `imagenet` pre-trained weights for encoder initialization
            in_channels=config['in_channels'],        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=config['num_classes'],            # model output channels (number of classes in your dataset)
        )
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        # print(self.model)
        # if config['model_type'] == "unet":
        #     self.model = smp.Unet(
        #         encoder_name=config['encoder_name'],      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #         encoder_weights="imagenet",               # use `imagenet` pre-trained weights for encoder initialization
        #         in_channels=config['in_channels'],        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #         classes=config['num_classes'],            # model output channels (number of classes in your dataset)
        #     )
        # elif config['model_type'] == "unet++":
        #     self.model = smp.UnetPlusPlus(
        #         encoder_name=config['encoder_name'],      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #         encoder_weights="imagenet",               # use `imagenet` pre-trained weights for encoder initialization
        #         in_channels=config['in_channels'],        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #         classes=config['num_classes'],            # model output channels (number of classes in your dataset)
        #         encoder_depth=config['encoder_depth']
        #     )
        # elif config['model_type'] == "deeplabv3":
        #     self.model = smp.DeepLabV3(
        #         encoder_name=config['encoder_name'],      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #         encoder_weights="imagenet",               # use `imagenet` pre-trained weights for encoder initialization
        #         in_channels=config['in_channels'],        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #         classes=config['num_classes'],            # model output channels (number of classes in your dataset)
        #         encoder_depth=config['encoder_depth']
        #     )
        # elif config['model_type'] == "deeplabv3+":
        #     self.model = smp.DeepLabV3Plus(
        #         encoder_name=config['encoder_name'],      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #         encoder_weights="imagenet",               # use `imagenet` pre-trained weights for encoder initialization
        #         in_channels=config['in_channels'],        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #         classes=config['num_classes'],            # model output channels (number of classes in your dataset)
        #         encoder_depth=config['encoder_depth']
        #     )
        # elif config['model_type'] == "MedT":
        #     self.model = MedT(pretrained=True, img_size=512, num_classes=config['num_classes'], imgchan=config['in_channels'])
    def forward(self, x):
        return self.model(x)