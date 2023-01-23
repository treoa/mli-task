import os
import sys
import string
import random
import torch
import warnings

import numpy as np
import pandas as pd
import torch.nn as nn
import torchio as tio
import seaborn as sns
import albumentations as A
import scipy.stats as stats
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from tqdm import tqdm
from torch import optim
from ignite.metrics import IoU
from PIL import Image, ImageOps, ImageFilter
from typing import Callable, List, Tuple, Dict, Optional
from albumentations import transforms
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import BaseCompose, Compose
from albumentations.core.transforms_interface import ImageOnlyTransform
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, LightningDataModule, LightningModule

warnings.filterwarnings("ignore")


def get_classes():
    file = open("Pascal-part/classes.txt", "r")
    classes = file.read()
    classes = classes.split("\n")
    file.close()
    classes = [x.split(":") for x in classes]
    # convert list to dictionary
    classes = dict(classes)
    return classes


# function to get the list of train image ids
def get_train_ids():
    file = open("Pascal-part/train_id.txt", "r")
    train_ids = file.read()
    train_ids = train_ids.split("\n")
    file.close()
    return train_ids


# function to get the list of test image ids
def get_test_ids():
    file = open("Pascal-part/val_id.txt", "r")
    val_ids = file.read()
    val_ids = val_ids.split("\n")
    file.close()
    return val_ids


# function to get the mask
def get_mask(id):
    mask = np.load(f"Pascal-part/gt_masks/{id}.npy", mmap_mode="r")
    return mask


# function to get the image
def get_image(id):
    # read image
    image = Image.open(f"Pascal-part/JPEGImages/{id}.jpg")
    return image


# create the dataset class for the segmentation task using LightningDataModule and torchio for Dataset creation and albumentations for data augmentation
class PascalSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 16, transforms: Optional[Callable] = None, train_ratio: float = 0.85):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms if transforms else self.get_transforms()
        self.train_ratio = train_ratio

    def get_transforms(self):
        if self.stage:
            return A.Compose(
                [A.Resize(height=512, width=512, p=1),
                 A.OneOf(
                    [A.HueSaturationValue(
                        hue_shift_limit=0.1,
                        sat_shift_limit=0.1,
                        val_shift_limit=0.1,
                        p=0.56),
                     A.RandomBrightnessContrast(
                        brightness_limit=0.10,
                        contrast_limit=0.10,
                        p=0.56), ],
                    p=0.7,),
                 A.Cutout(num_holes=8, max_h_size=24,
                          max_w_size=24, fill_value=0, p=0.16),
                 A.MotionBlur(blur_limit=(3, 5), p=0),
                 A.HorizontalFlip(p=0.5),
                 A.RandomRotate90(p=0.5),
                 ToTensorV2(p=1.0), ]
            )
        else:
            return A.Compose([A.Resize(height=512, width=512, p=1), ToTensorV2(p=1.0)])

    def setup(self, stage=None):
        self.stage = stage
        # get the train and test ids
        train_ids = get_train_ids()
        # split the train ids into train and validation into random ratios
        train_ids, val_ids = random_split(
            train_ids, [int(len(train_ids) * self.train_ratio), len(train_ids) - int(len(train_ids) * self.train_ratio)])
        test_ids = get_test_ids()
        # create the dataset
        self.train_dataset = tio.SubjectsDataset(
            [
                tio.Subject(
                    image=tio.ScalarImage(
                        path=f"Pascal-part/JPEGImages/{id}.jpg"
                    ),
                    mask=tio.LabelMap(
                        path=f"Pascal-part/gt_masks/{id}.npy", affine=np.eye(4)
                    ),
                )
                for id in train_ids
            ],
            transform=self.transforms,
        )
        self.val_dataset = tio.SubjectsDataset(
            [
                tio.Subject(
                    image=tio.ScalarImage(
                        path=f"Pascal-part/JPEGImages/{id}.jpg"
                    ),
                    mask=tio.LabelMap(
                        path=f"Pascal-part/gt_masks/{id}.npy", affine=np.eye(4)
                    ),
                )
                for id in val_ids
            ],
            transform=self.transforms,
        )
        self.test_dataset = tio.SubjectsDataset(
            [
                tio.Subject(
                    image=tio.ScalarImage(
                        path=f"Pascal-part/JPEGImages/{id}.jpg"
                    ),
                    mask=tio.LabelMap(
                        path=f"Pascal-part/gt_masks/{id}.npy", affine=np.eye(4)
                    ),
                )
                for id in test_ids
            ],
            transform=self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


"""
    Using the LightningModule to create the segmentation model using UNet with ResNet-18 as the backbone 
    and AdamW as the optimizer with Jacard loss (IoU) as the loss function and sigmoid activation function, 
    where number of classes is given from the length of output of get_classes() function.
"""


class PascalSegmentationModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, num_classes: int = 20):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
        )
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


model = PascalSegmentationModel()
# create the trainer using all gpus, ddp method, saving checkpoints at each epoch and by best loss at folder "ckpts" and 16-bit precision.
trainer = pl.Trainer(gpus=-1, accelerator="ddp", checkpoint_callback=ModelCheckpoint(
    dirpath="ckpts", save_top_k=-1, monitor="val_loss", mode="min"), precision=16)
# train the model
trainer.fit(model, PascalSegmentationDataModule())
