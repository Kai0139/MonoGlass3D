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

from model.tps_params import reduced_tps_params
from model.mono_g3d import MonoG3D
import open3d as o3d
import json

from data_loaders.gs3d_loader import GS3DDataset

from model.loss_cluster import angles_to_vector_tensor, miou_loss_fn, find_intersection_depth


if __name__ == "__main__":
    
    weight_path = Path(__file__).resolve().parent.parent.joinpath("weights", 
                                                                  "mono_g3d", 
                                                                  "mono_g3d_nn.pth")

    tps_param = reduced_tps_params()
    if weight_path.suffix == ".ckpt":
        model = MonoG3D.load_from_checkpoint(str(weight_path), params=tps_param)
    elif weight_path.suffix == ".pth":
        model = MonoG3D(params=tps_param)
        model.load_state_dict(torch.load(str(weight_path)))
    model.eval()
    model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    root_dir = Path("/mnt/slurmfs-3090node2/user_data/kzhang740/gs3d_resplit_251213")
    bag_dirs = list(root_dir.joinpath("test").iterdir())

    total_abs_rel = 0
    total_mae = 0
    total_rmse = 0
    total_iou = 0
    total_t_cost = 0
    n_sample = 0

    threshold_1 = 1.25
    threshold_2 = threshold_1 ** 2
    threshold_3 = threshold_1 ** 3

    total_sigma_1 = 0
    total_sigma_2 = 0
    total_sigma_3 = 0

    for bag_dir in bag_dirs:
        bag_name = bag_dir.name
        data_dirs = list(bag_dir.iterdir())

        # if "day" not in bag_name or "out" not in bag_name:
        #     continue

        for data_dir in data_dirs:
            data_idx = data_dir.name
            
            image_fp = bag_dir.joinpath(data_idx, "{}_image.png".format(data_idx))
            depth_fp = bag_dir.joinpath(data_idx, "{}_depth.txt".format(data_idx))
            cam_cfg_fp = bag_dir.joinpath(data_idx, "{}_camera_config.json".format(data_idx))

            cam_cfg_file = open(str(cam_cfg_fp), "r")
            cam_cfg = json.load(cam_cfg_file)
            cam_cfg_file.close()

            depth_gt = np.loadtxt(depth_fp)
            depth_gt = torch.tensor(depth_gt).float().unsqueeze(0).unsqueeze(0)

            transforms_img = Alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            image = cv2.imread(str(image_fp))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(float)
            image = transforms_img(image=image)["image"]
            image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)

            _, _, H, W = image.shape
            target_size = (504, 630)
            image = F.interpolate(image, size=target_size, mode="nearest-exact").cuda()
            depth_gt = F.interpolate(depth_gt, size=target_size, mode="nearest-exact")

            with torch.no_grad():
                t_start = time.time()
                res = model(image)
                t_end = time.time()
                t_cost = t_end - t_start
                used_mem_mb = torch.cuda.memory_allocated(device=image.device) / (1024 ** 2)
                print(f"Used memory: {used_mem_mb:.2f} MB")

            plane_pred = res["final_plane_pred"]
            if plane_pred.shape[2:] != target_size:
                plane_pred = F.interpolate(plane_pred, size=target_size, mode="nearest-exact")
            _, _, predH, predW = plane_pred.shape

            image_scale = predH / 1024.0
            cam_intrinsic = np.array([
                [cam_cfg["camera_internal"]["fx"], 0, cam_cfg["camera_internal"]["cx"]],
                [0, cam_cfg["camera_internal"]["fy"], cam_cfg["camera_internal"]["cy"]],
                [0, 0, 1.0 / image_scale]
            ]) * image_scale

            plane_pred = plane_pred.detach().squeeze().cpu()
            plane_pred = angles_to_vector_tensor(plane_pred)
            plane_depth = find_intersection_depth(plane_pred.permute(1, 2, 0), cam_intrinsic)

            final_seg = res["final_seg"].detach()
            final_seg = F.interpolate(final_seg, size=target_size, mode="nearest")
            final_seg = final_seg.cpu().squeeze() > 0.5
            
            mask_gt = depth_gt > 0
            eval_mask = torch.logical_and(mask_gt.cpu(), final_seg.cpu())

            iou = 1 - miou_loss_fn(final_seg.squeeze(), mask_gt.squeeze())
            
            plane_depth = plane_depth[mask_gt.squeeze()]
            depth_gt = depth_gt.squeeze()[mask_gt.squeeze()]
            depth_diff = torch.abs(plane_depth - depth_gt)
            mae = torch.mean(depth_diff)
            rmse = torch.sqrt(torch.mean(torch.pow(depth_diff, 2)))
            abs_rel = torch.mean(depth_diff / depth_gt)

            thresh = torch.max((depth_gt / plane_depth), (plane_depth / depth_gt))

            d1 = torch.sum(thresh < 1.25).float() / len(thresh)
            d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
            d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

            total_mae += mae
            total_rmse += rmse
            total_iou += iou
            total_abs_rel += abs_rel

            total_sigma_1 += d1
            total_sigma_2 += d2
            total_sigma_3 += d3

            total_t_cost += t_cost
            n_sample += 1
            print(f"bag: {bag_name}, data: {data_idx}, iou: {iou:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, abs_rel: {abs_rel:.4f}, sigma_1: {d1:.4f}, sigma_2: {d2:.4f}, sigma_3: {d3:.4f}, t_cost: {t_cost:.4f}")

    print(f"\n=== Average Evaluation Results ===")
    print(f"Total samples: {n_sample}")
    print(f"iou: {total_iou / n_sample:.4f}")
    print(f"mae: {total_mae / n_sample:.4f}")
    print(f"rmse: {total_rmse / n_sample:.4f}")
    print(f"abs_rel: {total_abs_rel / n_sample:.4f}")
    print(f"sigma_1: {total_sigma_1 / n_sample:.4f}")
    print(f"sigma_2: {total_sigma_2 / n_sample:.4f}")
    print(f"sigma_3: {total_sigma_3 / n_sample:.4f}")    
    print(f"t_cost: {total_t_cost / n_sample:.4f}")
