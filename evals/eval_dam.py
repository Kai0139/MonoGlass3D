from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import albumentations as Alb
import time

from model.tps_params import small_tps_params
from model.depth_anything_v2.dpt import DepthAnythingV2


if __name__ == "__main__":
    tps_param = small_tps_params()
    
    dam_path = Path(__file__).resolve().parent.parent.joinpath("weights", "resp_1", "dam_vits_nometric_wgs_ep50.pth")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = torch.load(str(dam_path), map_location="cpu", weights_only=False)
    model.cuda().eval()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    root_dir = Path("/mnt/slurmfs-3090node2/user_data/kzhang740/gs3d_resplit_251117")
    bag_dirs = list(root_dir.joinpath("val").iterdir())

    total_abs_rel = 0
    total_mae = 0
    total_rmse = 0
    total_iou = 0
    total_sigma_1 = 0
    total_sigma_2 = 0
    total_sigma_3 = 0
    n_sample = 0

    for bag_dir in bag_dirs:
        bag_name = bag_dir.name
        data_dirs = list(bag_dir.iterdir())

        for data_dir in data_dirs:
            data_idx = data_dir.name
            
            image_fp = bag_dir.joinpath( data_idx, "{}_image.png".format(data_idx))
            depth_fp = bag_dir.joinpath(data_idx, "{}_depth.txt".format(data_idx))

            depth_gt = np.loadtxt(depth_fp)
            depth_gt = torch.tensor(depth_gt).float().unsqueeze(0).unsqueeze(0)

            transforms_img = Alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            image = cv2.imread(str(image_fp))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(float)
            image = transforms_img(image=image)["image"]
            image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)

            _, _, H, W = image.shape
            image = F.interpolate(image, size=(504, 630), mode="nearest-exact")
            depth_gt = F.interpolate(depth_gt, size=(504, 630), mode="nearest-exact")

            with torch.no_grad():
                t_start = time.time()
                res = model(image.cuda().float())
                t_end = time.time()
            t_cost = t_end - t_start

            mask_gt = depth_gt > 0.001
            eval_mask = mask_gt.squeeze()

            res_image = res.squeeze().cpu().numpy()
            
            res_image = cv2.normalize(res_image, None, 0, 255, cv2.NORM_MINMAX)
            res_image = res_image.astype(np.uint8)
            
            res = res.squeeze()[eval_mask].detach().cpu()
            depth_gt = depth_gt.squeeze()[eval_mask]

            depth_diff = torch.abs(res - depth_gt)
            mae = torch.mean(depth_diff)
            rmse = torch.sqrt(torch.mean(torch.pow(depth_diff, 2)))
            abs_rel = torch.mean(depth_diff / depth_gt)

            thresh = torch.max((depth_gt / res), (res / depth_gt))

            d1 = torch.sum(thresh < 1.25).float() / len(thresh)
            d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
            d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

            total_mae += mae
            total_rmse += rmse
            total_abs_rel += abs_rel
            total_sigma_1 += d1
            total_sigma_2 += d2
            total_sigma_3 += d3
            n_sample += 1
            print("bag: {}, data: {}, mae: {}, rmse: {}, abs_rel: {}, sigma_1: {}, sigma_2: {}, sigma_3: {}".format(
                bag_name, data_idx, mae, rmse, abs_rel, d1, d2, d3))

    print("mae: {}, rmse: {}, abs_rel: {}, sigma_1: {}, sigma_2: {}, sigma_3: {}".format(
        total_mae / n_sample, total_rmse / n_sample, total_abs_rel / n_sample,
        total_sigma_1 / n_sample, total_sigma_2 / n_sample, total_sigma_3 / n_sample))
