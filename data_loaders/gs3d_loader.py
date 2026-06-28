from pathlib import Path
import numpy as np
import cv2
import json

import os
os.environ["NO_ALBUMENTATIONS_UPDATE "] = "1"

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import lightning
import albumentations as A

import random

def vector_to_angles(x, y, z):
    """
    Convert 3D vector [x, y, z] to two normalized angles in range [0, 1]:
    - a1: angle between vector and its projection on xz-plane
    - a2: angle between x-axis and projection in xz-plane
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    r_xz = np.sqrt(x**2 + z**2)

    # Angle 1: between full vector and xz-plane
    cos_theta1 = r_xz / r
    cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)
    theta1 = np.arccos(cos_theta1)  # in [0, pi/2]
    if y < 0:
        theta1 = - theta1  # flip if pointing downwards
    # a1 = theta1 / (np.pi / 2)      # normalize to [0, 1]
    a1 = theta1

    # Angle 2: angle in xz-plane from x-axis
    if r_xz == 0:
        a2 = 0  # default, straight down on y-axis
    else:
        cos_theta2 = z / r_xz
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2 = np.arccos(cos_theta2)  # in [0, pi]
        # if z < 0:
        theta2 = np.pi - theta2
        if x < 0:
            theta2 = - theta2
        # a2 = theta2 / (np.pi)  # normalize to [0, 1]
        a2 = theta2

    return a1, a2


class GS3DDataset(Dataset):
    def __init__(self, root_dir, train=True):
        super(GS3DDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.bag_dirs = sorted(list(self.root_dir.iterdir()))

        self.train = train

        self.image_fns = []
        self.mask_fns = []
        self.centerness_fns = []
        self.pc_depth_map_fns = []
        self.depth_gt_fns = []
        self.plane_dict_fns = []
        self.cam_cfg_fns = []

        for bag_dir in self.bag_dirs:
            data_dirs = sorted(list(bag_dir.iterdir()))
            for data_dir in data_dirs:
                data_idx = data_dir.name
                image_fp = data_dir.joinpath("{}_image.png".format(data_idx))
                mask_fp = data_dir.joinpath("{}_mask.png".format(data_idx))
                centerness_fp = data_dir.joinpath("{}_centerness.txt".format(data_idx))
                pc_depth_map_fp = data_dir.joinpath("{}_pc_depth_map.txt".format(data_idx))
                depth_gt_fp = data_dir.joinpath("{}_depth.txt".format(data_idx))
                plane_dict_fp = data_dir.joinpath("{}_plane.json".format(data_idx))
                cam_cfg_fp = data_dir.joinpath("{}_camera_config.json".format(data_idx))

                self.image_fns.append(str(image_fp))
                self.mask_fns.append(str(mask_fp))
                self.centerness_fns.append(str(centerness_fp))
                self.pc_depth_map_fns.append(str(pc_depth_map_fp))
                self.depth_gt_fns.append(str(depth_gt_fp))
                self.plane_dict_fns.append(str(plane_dict_fp))
                self.cam_cfg_fns.append(str(cam_cfg_fp))

        assert len(self.image_fns) == len(self.mask_fns)
        assert len(self.image_fns) == len(self.centerness_fns)
        assert len(self.image_fns) == len(self.pc_depth_map_fns)
        assert len(self.image_fns) == len(self.depth_gt_fns)
        assert len(self.image_fns) == len(self.plane_dict_fns)
        assert len(self.image_fns) == len(self.cam_cfg_fns)

        print("Total number of samples: {}".format(len(self.image_fns)))

        self.transforms_img = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, idx):
        image_fn = self.image_fns[idx]
        mask_fn = self.mask_fns[idx]

        image = cv2.imread(str(image_fn))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_plane_map = cv2.imread(str(mask_fn), cv2.IMREAD_GRAYSCALE)
        mask_plane_map = mask_plane_map.astype(int) - 100
        mask = mask_plane_map > 0
        mask = mask.astype(np.uint8)

        centerness = np.loadtxt(self.centerness_fns[idx])
        # take square root on non zero
        centerness[centerness > 0] = np.sqrt(centerness[centerness > 0])

        depth_map_sparse = np.loadtxt(self.pc_depth_map_fns[idx])
        depth_map = np.zeros_like(mask)
        depth_map[depth_map_sparse[:, 1].astype(int), depth_map_sparse[:, 0].astype(int)] = depth_map_sparse[:, 2]

        depth_gt = np.loadtxt(self.depth_gt_fns[idx])

        with open(self.plane_dict_fns[idx], "r") as f:
            plane_dict = json.load(f)
        with open(self.cam_cfg_fns[idx], "r") as f:
            cam_cfg = json.load(f)

        # normalize plane vector
        image_plane = torch.zeros(image.shape[0], image.shape[1], 3)
        for plane_idx in plane_dict.keys():
            plane_params = plane_dict[plane_idx]
            plane_vec = np.array(plane_params[:3])
            plane_vec_norm = np.linalg.norm(plane_vec)
            plane_vec = plane_vec / plane_vec_norm
            plane_dist = plane_params[3] / plane_vec_norm

            if plane_vec[2] > 0:
                plane_vec = -plane_vec
                plane_dist = -plane_dist

            ang1, ang2 = vector_to_angles(plane_vec[0], plane_vec[1], plane_vec[2])

            plane_dict[plane_idx] = torch.tensor([ang1, ang2, plane_dist]).float()
            image_plane[mask_plane_map == int(plane_idx)] = torch.tensor(plane_dict[plane_idx]).float()

        image = image.astype(float)
        image = self.transforms_img(image=image)["image"]

        image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
        depth_map = torch.tensor(depth_map).float().unsqueeze(0).unsqueeze(0)
        mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
        mask_plane_map = torch.tensor(mask_plane_map).float().unsqueeze(0).unsqueeze(0)
        centerness = torch.tensor(centerness).float().unsqueeze(0).unsqueeze(0)
        depth_gt = torch.tensor(depth_gt).float().unsqueeze(0).unsqueeze(0)
        image_plane = torch.tensor(image_plane).permute(2, 0, 1).float().unsqueeze(0)

        if random.random() < 0.5 and self.train:
            # random flip h
            image = torch.flip(image, dims=[-1])
            depth_map = torch.flip(depth_map, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
            mask_plane_map = torch.flip(mask_plane_map, dims=[-1])
            centerness = torch.flip(centerness, dims=[-1])
            depth_gt = torch.flip(depth_gt, dims=[-1])
            image_plane = torch.flip(image_plane, dims=[-1])
            image_plane[:, 1,:,:] = -image_plane[:, 1,:,:]  # flip angle2

        if random.random() < 0.5 and self.train:
            # random flip v
            image = torch.flip(image, dims=[-2])
            depth_map = torch.flip(depth_map, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
            mask_plane_map = torch.flip(mask_plane_map, dims=[-2])
            centerness = torch.flip(centerness, dims=[-2])
            depth_gt = torch.flip(depth_gt, dims=[-2])
            image_plane = torch.flip(image_plane, dims=[-2])
            image_plane[:, 0,:,:] = -image_plane[:, 0,:,:]

        B, C, H, W = image.shape
        target_shape = (504, 630)

        # resize to half size using nearest neighbor interpolation
        image = F.interpolate(image, size=target_shape, mode="nearest-exact").squeeze(0)
        depth_map = F.interpolate(depth_map, size=target_shape, mode="nearest-exact").squeeze(0)
        mask = F.interpolate(mask, size=target_shape, mode="nearest-exact").squeeze(0)
        mask_plane_map = F.interpolate(mask_plane_map, size=target_shape, mode="nearest-exact").squeeze(0)
        centerness = F.interpolate(centerness, size=target_shape, mode="nearest-exact").squeeze(0)
        depth_gt = F.interpolate(depth_gt, size=target_shape, mode="nearest-exact").squeeze(0)
        image_plane = F.interpolate(image_plane, size=target_shape, mode="nearest-exact").squeeze(0)

        # return everything in a dict
        return {
            "image": image,
            "depth_map": depth_map,
            "mask": mask,
            "mask_plane_map": mask_plane_map,
            "centerness": centerness,
            "depth_gt": depth_gt,
            "plane_dict": plane_dict,
            "cam_cfg": cam_cfg,
            "image_plane": image_plane,
            "fn": image_fn
        }
    

class GS3DDataModule(lightning.LightningDataModule):
    def __init__(self, root_dir, batch_size=4):
        super(GS3DDataModule, self).__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size

        self.train_dir = self.root_dir.joinpath("train")
        self.val_dir = self.root_dir.joinpath("test")

        self.train_dataset = GS3DDataset(root_dir=self.train_dir)
        self.val_dataset = GS3DDataset(root_dir=self.val_dir)

    def gs3d_collate_fn(self, batch):
        image = torch.stack([item["image"] for item in batch])
        depth_map = torch.stack([item["depth_map"] for item in batch])
        mask = torch.stack([item["mask"] for item in batch])
        mask_plane_map = torch.stack([item["mask_plane_map"] for item in batch])
        centerness = torch.stack([item["centerness"] for item in batch])
        depth_gt = torch.stack([item["depth_gt"] for item in batch])
        image_plane = torch.stack([item["image_plane"] for item in batch])
        plane_dict = [item["plane_dict"] for item in batch]
        cam_cfg = [item["cam_cfg"] for item in batch]

        return {
            "image": image,
            "depth_map": depth_map,
            "mask": mask,
            "mask_plane_map": mask_plane_map,
            "centerness": centerness,
            "depth_gt": depth_gt,
            "plane_dict": plane_dict,
            "cam_cfg": cam_cfg,
            "image_plane": image_plane
        }
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.gs3d_collate_fn, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.gs3d_collate_fn)



