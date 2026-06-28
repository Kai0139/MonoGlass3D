from pathlib import Path
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.loss_cluster import CenterLoss, SegmentationLoss, PlaneLossSM

from model.tps_params import TPSParams
from model.mono_g3d import MonoG3D

class MonoG3DTrainer(lightning.LightningModule):
    def __init__(self, params: TPSParams):
        super(MonoG3DTrainer, self).__init__()

        self.params = params
        self.model = MonoG3D(params)

        self.centerness_loss = CenterLoss(n_cascades=2, cascade_weights=self.params.cascade_weights[0:2]) # last cascade is final
        self.seg_loss = SegmentationLoss(n_cascades=3, cascade_weights=self.params.seg_cascade_weights)
        self.plane_loss = PlaneLossSM(cascade_weights=self.params.plane_cascade_weights, image_scale = 504.0 / 1024.0)

    def load_image_backbone(self):
        print(f"loading weights from {self.params.encoder_weights_path}")
        self.load_state_dict({k: v for k, v in torch.load(str(self.params.encoder_weights_path), map_location='cpu').items() if 'pretrained' in k}, strict=False)

    def configure_optimizers(self):

        image_backbone_params = self.model.pretrained.parameters()
        decoder_params = self.model.decoder.parameters()

        optimizer = torch.optim.AdamW([{"params": image_backbone_params, "lr": 5e-6},
                                       {"params": decoder_params, "lr": 5e-5}],
                                       weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=4, threshold=1e-4, min_lr=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/important_sum",
                "name": "lr",
            }
        }

    def forward(self, image):

        image_features = self.model.pretrained.get_intermediate_layers(image, n=self.model.n_cascades, return_class_token=False, reshape=True)
        res = self.model.decoder(image_features)

        return res

    def step(self, batch):

        data = batch
        image = data["image"]
        mask = data["mask"]
        centerness = data["centerness"]

        res = self.forward(image)

        cas_center = res["cas_center"]
        cas_seg = res["cas_seg"]
        final_seg = res["final_seg"]

        # centerness loss
        l_centerness = self.centerness_loss(cas_center[0:-1], cas_center[-1], centerness)
        l_centerness = sum(l_centerness)
        # segmentation loss
        l_seg_ce, l_seg_miou, l_seg_dice = self.seg_loss(cas_seg, final_seg, mask)
        l_seg_ce = sum(l_seg_ce)
        l_seg_miou = sum(l_seg_miou)
        l_seg_dice = sum(l_seg_dice)
        # plane loss
        loss = self.plane_loss(res, data)
        
        l_plane_masked = loss["l_plane_masked"]
        final_naive_masked_loss = loss["final_naive_masked_loss"]
        l_plane_depth_l1_masked = loss["l_plane_depth_l1_masked"]
        l_plane_embvar = loss["l_plane_embvar"]
        l_plane_embdiff = loss["l_plane_embdiff"]
        l_plane_dist = loss["l_plane_dist"]

        # loss without running norm
        loss_sum = self.params.w_l_centerness *    l_centerness        + \
                   self.params.w_l_seg_ce *        l_seg_ce            + \
                   self.params.w_l_seg_miou *      l_seg_miou          + \
                   self.params.w_l_plane_masked *  l_plane_masked      + \
                   self.params.w_l_plane_embvar *  l_plane_embvar      + \
                   self.params.w_l_plane_dist *    l_plane_dist
        
        important_loss_sum = l_plane_dist + l_seg_miou

        loss_dict = {
            "total_loss": loss_sum,
            "centerness": l_centerness,
            "seg_ce": l_seg_ce,
            "seg_miou": l_seg_miou,
            "seg_dice": l_seg_dice,
            "plane_depth_l1_masked": l_plane_depth_l1_masked,
            "plane_masked": l_plane_masked,
            "final_masked": final_naive_masked_loss,
            "plane_embvar": l_plane_embvar,
            "plane_embdiff": l_plane_embdiff,
            "important_sum": important_loss_sum,
            "plane_dist": l_plane_dist,
        }

        return loss_dict

    def log_metrics(self, losses, prefix, sync_dist=True):
        for loss_name, loss_value in losses.items():
            self.log(f"{prefix}/{loss_name}", loss_value, sync_dist=sync_dist)
    
    def training_step(self, batch, batch_idx):
        
        loss_dict = self.step(batch)
        
        self.log_metrics(loss_dict, "train")

        return loss_dict["total_loss"]
    
    def validation_step(self, batch, batch_idx):
        
        loss_dict = self.step(batch)

        self.log_metrics(loss_dict, "val")

        return loss_dict["total_loss"]

if __name__ == "__main__":
    from tps_params import down_scaled_tps_params
    params = down_scaled_tps_params()
    model = MonoG3DTrainer(params=params)
    model.load_image_backbone()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    # image_size = 512
    image = torch.randn(2, 3, 504, 630)
    res = model(image)
    for key in res.keys():
        print(key)
        if isinstance(res[key], list):
            for i in range(len(res[key])):
                print(res[key][i].shape)
        else:
            print(res[key].shape)