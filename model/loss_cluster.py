import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def angles_to_vector_tensor(planes):
    """
    planes: torch.tensor(3, H, W)
    # theta1 = a1 * (np.pi / 2)
    # theta2 = a2 * (np.pi)
    # r_xz = abs(r * np.cos(theta1))
    # x = r_xz * np.sin(theta2)
    # y = r * np.sin(theta1)
    # z = r_xz * np.cos(np.pi - theta2)
    """
    theta1 = planes[0,:,:]
    theta2 = planes[1,:,:]
    r = torch.ones_like(theta1)
    # theta1 = theta1 * (torch.pi / 2)
    # theta2 = theta2 * torch.pi

    r_xz = abs(r * torch.cos(theta1))
    x = r_xz * torch.sin(theta2)
    y = r * torch.sin(theta1)
    z = r_xz * torch.cos(torch.pi - theta2)

    return torch.stack([x, y, z, planes[2,:,:]], dim=0)

def dice_loss_fn(pred, target, smooth=1e-5, threshold=0.5):
    # Calculate Dice Coefficient
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = torch.sum(pred * target)
    dice_coeff = (2. * intersection) / (torch.sum(pred) + torch.sum(target) + smooth)
    
    return 1 - dice_coeff

def miou_loss_fn(pred, target, smooth=1e-5, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    miou = (intersection) / (union + smooth)
    
    return 1 - miou

def find_intersection_depth(planes, cam_intrinsics):
    """
    find intersection of planes in image space
    planes: H * W * 4
    cam_intrinsics: 3 * 3
    """
    H, W, _ = planes.shape
    # find intersection of planes in image space
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = torch.ones_like(x)
    points = torch.stack([x, y, z], axis=1).float().to(planes.device)

    # find intersection of planes in camera space
    cam_mat = torch.tensor(cam_intrinsics).float().to(planes.device)
    
    points = torch.linalg.inv(cam_mat) @ points.T
    points = points.T
    points = points.reshape(H, W, 3)
    # find intersection of planes in image space
    abc = planes[:, :, :3] # H * W * 3
    d = planes[:, :, 3] # H * W
    denominators = torch.abs(torch.sum(abc * points, dim=2)) # H * W
    t = d / (denominators + 1e-9) # H * W
    return t

def plane_dist_error(plane_pred, plane_gt, cam_intrinsics):
    """
    find intersection of planes in image space
    planes: H, W, 4
    cam_intrinsics: 3, 3
    """
    H, W, _ = plane_pred.shape
    # find intersection of planes in image space
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = torch.ones_like(x)
    points = torch.stack([x, y, z], axis=1).float().to(plane_pred.device)

    # find intersection of planes in camera space
    cam_mat = torch.tensor(cam_intrinsics).float().to(plane_pred.device)
    
    points = torch.linalg.inv(cam_mat) @ points.T
    points = points.T
    points = points.reshape(H, W, 3)
    # find intersection of planes in image space
    abc = plane_pred[:, :, :3] # H * W * 3
    d = plane_pred[:, :, 3] # H * W
    denominators = torch.abs(torch.sum(abc * points, dim=2)) # H * W
    t = d / (denominators + 1e-9) # H * W

    points_3d = points * t.unsqueeze(-1) # H, W, 3

    v1 = torch.stack([-plane_pred[:, :, 1], plane_pred[:, :, 0], torch.zeros(H, W).to(plane_pred.device)], dim=2) # [-b, a, 0]
    v1 = v1 / torch.norm(v1, dim=2, keepdim=True) # normalize
    v2 = torch.cross(abc, v1, dim=2) # cross product
    v2 = v2 / torch.norm(v2, dim=2, keepdim=True) # normalize

    pt_rad = 1.0
    points_3d_1 = points_3d + v1 * pt_rad # H, W, 3
    points_3d_2 = points_3d + v2 * pt_rad # H, W, 3
    points_3d_3 = points_3d - v1 * pt_rad # H, W, 3
    points_3d_4 = points_3d - v2 * pt_rad # H, W, 3

    # find distance to gt plane
    abc_gt = plane_gt[:, :, :3] # H * W * 3
    d_gt = plane_gt[:, :, 3] # H * W
    dist_denomenators = torch.sqrt(torch.square(abc_gt).sum(dim=-1)) # H * W

    dist  = torch.abs(torch.sum(abc_gt * points_3d_1, dim=2) + d_gt) / (dist_denomenators + 1e-9) # H * W
    dist += torch.abs(torch.sum(abc_gt * points_3d_2, dim=2) + d_gt) / (dist_denomenators + 1e-9) # H * W
    dist += torch.abs(torch.sum(abc_gt * points_3d_3, dim=2) + d_gt) / (dist_denomenators + 1e-9) # H * W
    dist += torch.abs(torch.sum(abc_gt * points_3d_4, dim=2) + d_gt) / (dist_denomenators + 1e-9) # H * W

    return dist / 4.0

def miou_w_zero_mask(pred, target, smooth=1e-5, threshold=0.5):

    pred = (pred > threshold).float()

    # if torch.sum(target) == 0:
    #     return torch.sum(pred) / torch.numel(target)
    
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    miou = (intersection) / (union + smooth)

    return 1 - miou


class CenterLoss(torch.nn.Module):
    def __init__(self, n_cascades: int, cascade_weights: list):
        super(CenterLoss, self).__init__()
        self.n_cascades = n_cascades
        self.cascade_weights = cascade_weights
    
    def forward(self, pred_cascades, final_pred, gt):
        n_cascades = len(pred_cascades)
        losses = []
        for i in range(n_cascades):
            gt_resize = F.interpolate(gt, size=pred_cascades[i].shape[-2:], mode="bilinear", align_corners=False)
            losses.append(F.binary_cross_entropy(pred_cascades[i], gt_resize) * self.cascade_weights[i])
        
        final_pred_weight = 1 - sum(self.cascade_weights)
        
        if final_pred.shape[-1] != gt.shape[-1] or final_pred.shape[-2]!= gt.shape[-2]:
            gt_resize = F.interpolate(gt, size=final_pred.shape[-2:], mode="bilinear", align_corners=False)
        else:
            gt_resize = gt

        final_loss = F.binary_cross_entropy(final_pred, gt_resize) * final_pred_weight

        losses.append(final_loss)

        return losses
    
class SegmentationLoss(torch.nn.Module):
    def __init__(self, n_cascades: int, cascade_weights: list = []):
        super(SegmentationLoss, self).__init__()
        self.n_cascades = n_cascades
        self.cascade_weights = cascade_weights
    
    def forward(self, pred_cascades, final_pred, gt):
        n_cascades = len(pred_cascades)
        ce_losses = []
        miou_losses = []
        dice_losses = []
        for i in range(n_cascades):
            gt_resize = F.interpolate(gt, size=pred_cascades[i].shape[-2:], mode="bilinear", align_corners=False)
            ce_losses.append(F.binary_cross_entropy(pred_cascades[i], gt_resize) * self.cascade_weights[i])
            miou_losses.append(miou_w_zero_mask(pred_cascades[i], gt_resize) * self.cascade_weights[i])
            dice_losses.append(dice_loss_fn(pred_cascades[i], gt_resize) * self.cascade_weights[i])
        
        final_pred_weight = 1 - sum(self.cascade_weights)

        if final_pred.shape[-1] != gt.shape[-1] or final_pred.shape[-2]!= gt.shape[-2]:
            gt_resize = F.interpolate(gt, size=final_pred.shape[-2:], mode="bilinear", align_corners=False)
        else:
            gt_resize = gt

        final_ce_loss = F.binary_cross_entropy(final_pred, gt_resize) * final_pred_weight
        final_miou_loss = miou_w_zero_mask(final_pred, gt_resize) * final_pred_weight
        final_dice_loss = dice_loss_fn(final_pred, gt_resize) * final_pred_weight

        ce_losses.append(final_ce_loss)
        miou_losses.append(final_miou_loss)
        dice_losses.append(final_dice_loss)

        return ce_losses, miou_losses, dice_losses
    
class DepthLoss(torch.nn.Module):
    def __init__(self, cascade_weights, seg_threshold=0.5):
        super(DepthLoss, self).__init__()
        self.cascade_weights = cascade_weights
        self.seg_threshold = seg_threshold

    def forward(self, depth_cascade, final_depth_pred, depth_gt, mask_gt):

        total_loss = 0
        for i in range(len(depth_cascade)):
            gt_resize = F.interpolate(depth_gt, size=depth_cascade[i].shape[-2:], mode="nearest-exact")
            seg_resize = F.interpolate(mask_gt, size=depth_cascade[i].shape[-2:], mode="nearest-exact")
            seg_resize = seg_resize > self.seg_threshold
            depth_seg = depth_cascade[i]
            loss = F.smooth_l1_loss(depth_seg[seg_resize], gt_resize[seg_resize], beta=0.1) * self.cascade_weights[i]
            total_loss += loss

        gt_resize = F.interpolate(depth_gt, size=final_depth_pred.shape[-2:], mode="nearest-exact")
        seg_resize = F.interpolate(mask_gt, size=final_depth_pred.shape[-2:], mode="nearest-exact")
        seg_resize = seg_resize > self.seg_threshold
        final_loss = F.smooth_l1_loss(final_depth_pred[seg_resize], gt_resize[seg_resize], beta=0.1)
        total_loss += final_loss * (1 - sum(self.cascade_weights))

        return total_loss, final_loss
    

class PlaneLossSM(torch.nn.Module):
    def __init__(self, cascade_weights: list, image_scale=0.5):
        super(PlaneLossSM, self).__init__()
        self.image_scale = image_scale
        self.cascade_weights = cascade_weights
        self.l2_loss = torch.nn.MSELoss()

    def plane_pred_loss(self, plane_pred, image_plane_gt, mask_gt, plane_map, plane_dict):

        naive_plane_diff = 0
        seg_mask = mask_gt > 0
        seg_mask = torch.cat([seg_mask, seg_mask, seg_mask], dim=1)

        plane_ang_diff = F.smooth_l1_loss(plane_pred[:,0:2,:,:], image_plane_gt[:,0:2,:,:], beta=0.02)
        plane_dist_diff = F.smooth_l1_loss(plane_pred[:,2,:,:], image_plane_gt[:,2,:,:], beta=0.02)

        plane_ang_diff_masked = F.smooth_l1_loss(plane_pred[:,0:2,:,:][seg_mask[:,0:2,:,:]], image_plane_gt[:,0:2,:,:][seg_mask[:,0:2,:,:]], beta=0.02)
        plane_dist_diff_masked = F.smooth_l1_loss(plane_pred[:,2,:,:][seg_mask[:,2,:,:]], image_plane_gt[:,2,:,:][seg_mask[:,2,:,:]], beta=0.02)

        naive_plane_diff = plane_ang_diff * 3 + plane_dist_diff
        naive_plane_diff_masked = plane_ang_diff_masked * 3 + plane_dist_diff_masked

        return naive_plane_diff, naive_plane_diff_masked
    
    def plane_pred_mapped_loss(self, plane_pred, image_plane_gt, plane_map, plane_dict):
        batch_size = plane_pred.shape[0]

        plane_diff_masked = 0

        n_planes = 0

        for b in range(batch_size):
            plane_pred_b = plane_pred[b]
            image_plane_gt_b = image_plane_gt[b]
            plane_map_b = plane_map[b]
            plane_dict_b = plane_dict[b]
            
            for plane_id in plane_dict_b.keys():
                plane_map_mask = plane_map_b.squeeze() == int(plane_id)
                if torch.sum(plane_map_mask) >= 1:
                    n_planes += 1
                else:
                    continue

                plane_diff_masked += F.smooth_l1_loss(plane_pred_b[:, plane_map_mask], image_plane_gt_b[:, plane_map_mask], beta=0.01)
                
        return plane_diff_masked / (n_planes + 1e-8)
    
    def plane_emb_variance_loss(self, plane_emb, plane_map, plane_dict):
        batch_size = plane_emb.shape[0]
        plane_emb_var_loss = 0
        plane_emb_diff_loss = 0

        plane_emb_diff_batch = 0

        for b in range(batch_size):
            plane_emb_b = plane_emb[b] # C, H, W
            plane_map_b = F.interpolate(plane_map[b].unsqueeze(0), size=plane_emb_b.shape[-2:], mode="nearest-exact").squeeze()
            plane_dict_b = plane_dict[b]

            plane_emb_var_loss_b = 0
            n_planes = 0.0
            plane_means = []
            for plane_id in plane_dict_b.keys():
                plane_map_mask = plane_map_b == int(plane_id) # H, W
                if torch.sum(plane_map_mask) > 1:
                    n_planes += 1
                else:
                    continue
                plane_emb_b_masked = plane_emb_b[:, plane_map_mask] # C, N
                plane_emb_var_loss_b += torch.std(plane_emb_b_masked, dim=-1).mean()
                plane_means.append(plane_emb_b_masked.mean(dim=-1))

            plane_emb_var_loss_b /= n_planes

            # push means apart
            if n_planes < 2:
                continue
            else:
                plane_emb_diff_batch += 1

            plane_emb_diff_loss_b = 0
            n_diffs = 0.0
            for i in range(len(plane_means)):
                for j in range(i+1, len(plane_means)):
                    plane_emb_diff_loss_b += torch.abs(plane_means[i] - plane_means[j]).mean()
                    n_diffs += 1
            
            plane_emb_diff_loss_b /= n_diffs

            plane_emb_var_loss += plane_emb_var_loss_b
            plane_emb_diff_loss += plane_emb_diff_loss_b

        return plane_emb_var_loss / batch_size, 1 - plane_emb_diff_loss / (plane_emb_diff_batch + 1e-8)
    
    def plane_cascade_loss(self, res, data):
        mask_gt = data["mask"]
        image_plane_gt = data["image_plane"]
        mask_plane_map = data["mask_plane_map"]
        plane_dict = data["plane_dict"]

        plane_cascades = res["cas_plane"]
        final_plane_pred = res["final_plane_pred"]

        total_naive_masked_loss = 0

        for i in range(len(plane_cascades)):
            plane_pred = plane_cascades[i]
            plane_gt_resize = F.interpolate(image_plane_gt, size=plane_pred.shape[-2:], mode="nearest-exact")
            mask_plane_map_resize = F.interpolate(mask_plane_map, size=plane_pred.shape[-2:], mode="nearest-exact")

            naive_masked_loss = self.plane_pred_mapped_loss(plane_pred, plane_gt_resize, mask_plane_map_resize, plane_dict)
            total_naive_masked_loss += naive_masked_loss * self.cascade_weights[i]

        if image_plane_gt.shape[-2:] != final_plane_pred.shape[-2:]:
            plane_gt_resize = F.interpolate(image_plane_gt, size=final_plane_pred.shape[-2:], mode="nearest-exact")
            mask_plane_map_resize = F.interpolate(mask_plane_map, size=final_plane_pred.shape[-2:], mode="nearest-exact")
        else:
            plane_gt_resize = image_plane_gt
            mask_plane_map_resize = mask_plane_map
            
        final_naive_masked_loss = self.plane_pred_mapped_loss(final_plane_pred, plane_gt_resize, mask_plane_map_resize, plane_dict)
        total_naive_masked_loss += final_naive_masked_loss * (1 - sum(self.cascade_weights))

        return total_naive_masked_loss, final_naive_masked_loss
    
    def plane_depth_loss(self, res, data):

        mask_plane_map = data["mask_plane_map"]
        plane_dict = data["plane_dict"]
        depth_gt = data["depth_gt"]
        cam_cfg = data["cam_cfg"]
        plane_pred = res["final_plane_pred"]

        B, _, H, W = plane_pred.shape
        depth_gt = F.interpolate(depth_gt, size=(H, W), mode="nearest-exact")
        mask_plane_map = F.interpolate(mask_plane_map, size=(H, W), mode="nearest-exact")

        total_loss_l1 = 0
        n_planes = 0

        
        for b in range(B):
            fx = cam_cfg[b]["camera_internal"]["fx"]
            fy = cam_cfg[b]["camera_internal"]["fy"]
            cx = cam_cfg[b]["camera_internal"]["cx"]
            cy = cam_cfg[b]["camera_internal"]["cy"]
            image_scale = H / 1024.0
            cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0/image_scale]]) * image_scale

            plane_pred_b = plane_pred[b]

            plane_dist_pred = plane_pred_b[2,:,:]
            plane_pred_b = torch.cat([plane_pred_b[0:2,:,:], plane_dist_pred.unsqueeze(0)], dim=0)
            plane_pred_b = angles_to_vector_tensor(plane_pred_b)

            # find depth intersection of planes in image space
            plane_depth_pred = find_intersection_depth(plane_pred_b.permute(1,2,0), cam_intr)
            plane_depth_pred.to(depth_gt.device)

            # compute loss
            plane_map_b = mask_plane_map[b]
            plane_dict_b = plane_dict[b]
            for plane_id in plane_dict_b.keys():
                plane_map_mask = plane_map_b.squeeze() == int(plane_id)
                if torch.sum(plane_map_mask) >= 1:
                    plane_depth_pred_b = plane_depth_pred[plane_map_mask]
                    depth_gt_b = depth_gt[b,0][plane_map_mask]

                    loss_l1 = F.l1_loss(plane_depth_pred_b, depth_gt_b)
                    total_loss_l1 += loss_l1
                    n_planes += 1
            
        return total_loss_l1 / (n_planes + 1e-8)
    
    def plane_dist_loss(self, plane_pred, data):
        mask_plane_map = data["mask_plane_map"]
        plane_dict = data["plane_dict"]
        cam_cfg = data["cam_cfg"]
        image_plane_gt = data["image_plane"]

        B, _, H, W = plane_pred.shape

        total_loss_l1 = 0
        n_planes = 0

        for b in range(B):
            fx = cam_cfg[b]["camera_internal"]["fx"]
            fy = cam_cfg[b]["camera_internal"]["fy"]
            cx = cam_cfg[b]["camera_internal"]["cx"]
            cy = cam_cfg[b]["camera_internal"]["cy"]
            image_scale = H / 1024.0
            cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0/image_scale]]) * image_scale

            plane_pred_b = plane_pred[b]

            plane_pred_b = angles_to_vector_tensor(plane_pred_b)
            plane_gt_b = angles_to_vector_tensor(image_plane_gt[b])
            # if the shape of pred and gt does not match, interpolate gt to pred shape
            if plane_pred_b.shape != plane_gt_b.shape:
                # plane_pred_b = F.interpolate(plane_pred_b.unsqueeze(0), size=plane_gt_b.shape[-2:], mode="nearest-exact").squeeze(0)
                plane_gt_b = F.interpolate(plane_gt_b.unsqueeze(0), size=plane_pred_b.shape[-2:], mode="nearest-exact").squeeze(0)

            # find depth intersection of planes in image space
            plane_dist_loss = plane_dist_error(plane_pred_b.permute(1,2,0), plane_gt_b.permute(1,2,0), cam_intr)

            # compute loss
            plane_map_b = mask_plane_map[b]
            plane_dict_b = plane_dict[b]
            for plane_id in plane_dict_b.keys():
                plane_map_mask = plane_map_b.squeeze() == int(plane_id)
                if plane_map_mask.shape != plane_pred_b.shape[-2:]:
                    plane_map_mask = F.interpolate(plane_map_mask.float().unsqueeze(0).unsqueeze(0), size=plane_pred_b.shape[-2:], mode="nearest-exact").squeeze(0).squeeze(0)
                    plane_map_mask = plane_map_mask.bool()
                if torch.sum(plane_map_mask) >= 1:
                    plane_dist_loss_p = plane_dist_loss[plane_map_mask]
                    total_loss_l1 += plane_dist_loss_p.mean()
                    # total_loss_l1 += torch.square(plane_dist_loss_p).mean()
                    n_planes += 1
            
        return total_loss_l1 / (n_planes + 1e-8)

    def plane_dist_cascade_loss(self, res, data):

        plane_cascades = res["cas_plane"]
        final_plane_pred = res["final_plane_pred"]

        total_loss_l1 = 0

        for i in range(len(plane_cascades)):
            plane_pred = plane_cascades[i]
            plane_dist_loss = self.plane_dist_loss(plane_pred, data)
            total_loss_l1 += plane_dist_loss * self.cascade_weights[i]

        final_plane_dist_loss = self.plane_dist_loss(final_plane_pred, data)
        total_loss_l1 += final_plane_dist_loss * (1 - sum(self.cascade_weights))

        return total_loss_l1, final_plane_dist_loss
    
    def forward(self, res, data):

        l_plane_masked, final_naive_masked_loss = self.plane_cascade_loss(res, data)
        l_plane_depth_l1_masked = self.plane_depth_loss(res, data)
        l_plane_embvar, l_plane_embdiff = self.plane_emb_variance_loss(res["plane_emb"], data["mask_plane_map"], data["plane_dict"])
        l_plane_dist = self.plane_dist_loss(res["final_plane_pred"], data)

        loss = {
            "l_plane_masked": l_plane_masked,
            "final_naive_masked_loss": final_naive_masked_loss,
            "l_plane_depth_l1_masked": l_plane_depth_l1_masked,
            "l_plane_embvar": l_plane_embvar,
            "l_plane_embdiff": l_plane_embdiff,
            "l_plane_dist": l_plane_dist
        }
        
        return loss

if __name__ == "__main__":
    
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
            theta1 = np.pi - theta1  # flip if pointing downwards
        a1 = theta1 / (np.pi)      # normalize to [0, 1]

        # Angle 2: angle in xz-plane from x-axis
        if r_xz == 0:
            a2 = 0  # default, straight down on y-axis
        else:
            cos_theta2 = x / r_xz
            cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
            theta2 = np.arccos(cos_theta2)  # in [0, pi]
            if z > 0:
                theta2 = 2 * np.pi - theta2
            a2 = theta2 / (np.pi)  # normalize to [0, 1]

        return a1, a2
    
    plane = [0.07257716652199878, -0.0031808988016651663, -0.18923231502880644, 1.0]
    plane = np.array(plane)
    plane_vec_norm = np.linalg.norm(plane[0:3])
    plane[0:3] = plane[0:3] / plane_vec_norm
    plane[3] = plane[3] / plane_vec_norm
    print("plane: ", plane)
    a1, a2 = vector_to_angles(plane[0], plane[1], plane[2])
    print("a1: ", a1, "a2: ", a2)
    




