from pathlib import Path
import numpy as np
import cv2
import json

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning
import albumentations as A


class GSDTestDataset(Dataset):
    def __init__(self, root_dir, target_size=512):
        super(GSDTestDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir.joinpath("image")
        self.masks_dir = self.root_dir.joinpath("mask")

        self.image_fns = sorted(list(self.images_dir.glob("*.jpg")), key=lambda x: str(x.stem))
        self.mask_fns = sorted(list(self.masks_dir.glob("*.png")), key=lambda x: str(x.stem))

        self.target_size = target_size
        self.transforms_img = A.Compose([
            A.Resize(self.target_size, self.target_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        assert len(self.image_fns) == len(self.mask_fns)

    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, idx):
        image_fn = self.image_fns[idx]
        mask_fn = self.mask_fns[idx]

        if image_fn.stem != mask_fn.stem:
            print("image_fn: {}, \nmask_fn: {}".format(image_fn, mask_fn))
            raise ValueError("image_fn and mask_fn do not match")

        image = cv2.imread(str(image_fn))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_fn))
        mask = np.sum(mask.astype(float), axis=2) > 0
        mask = mask.astype(np.uint8)

        # check shape mismatch
        if image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
            image = np.rot90(image, k=1, axes=(0, 1))
    
        image = self.transforms_img(image=image)["image"]
        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).float()

        return image, mask, image_fn.stem

class GSDDataset(Dataset):
    def __init__(self, root_dir, train=True, target_size=518):
        super(GSDDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.images_dir = self.root_dir.joinpath("image")
        self.masks_dir = self.root_dir.joinpath("mask")
        self.centerness_dir = self.root_dir.joinpath("centerness")

        self.image_fns = sorted(list(self.images_dir.glob("*.jpg")), key=lambda x: str(x.stem))
        self.mask_fns = sorted(list(self.masks_dir.glob("*.png")), key=lambda x: str(x.stem))
        self.centerness_fns = sorted(list(self.centerness_dir.glob("*.png")), key=lambda x: str(x.stem))

        if train:
            self.transforms_all = A.Compose([
                A.RandomRotate90(p=0.75),
                A.Resize(self.target_size, self.target_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)
            ], additional_targets={
                "mask": "image",
                "centerness": "image",
            })
            self.transforms_img = A.Compose([
                A.RandomBrightnessContrast(p=0.25),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Blur(p=0.25)
            ])
        else:
            self.transforms_all = A.Compose([
                A.Resize(self.target_size, self.target_size)
            ], additional_targets={
                "mask": "image",
                "centerness": "image",
            })

            self.transforms_img = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])


        assert len(self.image_fns) == len(self.mask_fns)

    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, idx):
        image_fn = self.image_fns[idx]
        mask_fn = self.mask_fns[idx]

        if image_fn.stem != mask_fn.stem:
            print("image_fn: {}, \nmask_fn: {}".format(image_fn, mask_fn))
            raise ValueError("image_fn and mask_fn do not match")

        image = cv2.imread(str(image_fn))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_fn), cv2.IMREAD_GRAYSCALE)
        centerness = cv2.imread(str(self.centerness_fns[idx]), cv2.IMREAD_GRAYSCALE)

        # check shape mismatch
        if image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
            image = np.rot90(image, k=1, axes=(0, 1))

        transformed = self.transforms_all(image=image, mask=mask, centerness=centerness)
        image = transformed["image"]
        mask = transformed["mask"]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(mask)
        boundary = cv2.drawContours(boundary, contours, -1, 255, self.target_size // 128)
        boundary = boundary.astype(float)

        # boundary = transformed["boundary"].astype(float) / 255.0
        boundary = boundary / 255.0
        centerness = transformed["centerness"].astype(float) / 255.0
    
        image = self.transforms_img(image=image)["image"]

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask > 0).float().unsqueeze(0)
        boundary = torch.tensor(boundary).float().unsqueeze(0)
        centerness = torch.tensor(centerness).float().unsqueeze(0)

        return image, centerness, mask
    

class GSDDataModule(lightning.LightningDataModule):
    def __init__(self, root_dir, batch_size=4, target_size=518):
        super(GSDDataModule, self).__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size

        self.train_dir = self.root_dir.joinpath("train")
        self.val_dir = self.root_dir.joinpath("test")

        self.train_dataset = GSDDataset(root_dir=self.train_dir, target_size=target_size)
        self.val_dataset = GSDDataset(root_dir=self.val_dir, target_size=target_size, train=False)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    root_dir = Path("/home/zhangkai/data/glass_seg/GSD")
    datamodule = GSDDataModule(root_dir=root_dir, batch_size=1, target_size=518)
    dataloader = datamodule.train_dataloader()

    img, centerness, mask = datamodule.train_dataset[0]
    centerness = (centerness.squeeze().numpy() * 255).astype(np.uint8)
    cv2.imwrite("centerness.png", centerness)
    cv2.imwrite("mask.png", mask.squeeze().numpy().astype(np.uint8) * 255)