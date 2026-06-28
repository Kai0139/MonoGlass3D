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
from model.depth_anything_v2.dinov2 import DINOv2 

from model.modules import MultiBranchFusion, CascadeHeads2D, RefinementBlock2D
from model.tps_params import TPSParams



class CascadeGlass3DDecoder(nn.Module):
    def __init__(self, params: TPSParams, n_cascades: int, encoder_dims=[128,256,512,1024]):
        super(CascadeGlass3DDecoder, self).__init__()

        self.params = params
        self.encoder_dims = encoder_dims

        self.last_input_proj = nn.Conv2d(encoder_dims[-1], self.params.base_hidden_dim, kernel_size=1)

        self.n_cascades = n_cascades

        self.enc_fusion = nn.ModuleList(
            [
                MultiBranchFusion(self.encoder_dims[i], self.encoder_dims[i], self.params.base_hidden_dim) for i in range(self.n_cascades)
            ]
        )

        self.cas_refine = nn.ModuleList(
            [
                RefinementBlock2D(self.params.base_hidden_dim, self.params.base_hidden_dim) if i > 0 else nn.Identity() for i in range(self.n_cascades)
            ]
        )

        self.cascade_heads = nn.ModuleList(
            [
                CascadeHeads2D(self.params.base_hidden_dim) for _ in range(self.n_cascades)
            ]
        )
        
        self.final_seg_head = nn.Sequential(
            nn.Conv2d(self.params.base_hidden_dim*2, self.params.base_hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.params.base_hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(self.params.base_hidden_dim, self.params.base_hidden_dim // 4, 4, stride=4, bias=False),
            nn.BatchNorm2d(self.params.base_hidden_dim // 4),
            nn.Conv2d(self.params.base_hidden_dim // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, features):

        cas_center, cas_seg = [], []
        
        for i in range(self.n_cascades):
            x = features[i]
            # Fusion
            x_center, x_fused, x_plane  = self.enc_fusion[i](x, x)
            # refine
            if i > 0:
                x_center, x_fused = self.cas_refine[i](
                    torch.cat([x_center, cas_center[-1]], dim=1),
                    torch.cat([x_fused, cas_seg[-1]], dim=1)
                )

            # cascade pred
            center, seg = self.cascade_heads[i](x_center, x_fused, x_plane)
            
            cas_center.append(center)
            cas_seg.append(seg)

        x_fused = x_fused + x_fused * (1 - cas_center[-1])
        x_plane = x_plane + x_plane * (1 - cas_center[-1])
        final_seg = self.final_seg_head(torch.cat([x_fused, x_plane], dim=1))
        
        # put all results in a dict
        res = {
            "cas_center": cas_center,
            "cas_seg": cas_seg,
            "final_seg": final_seg,
        }

        return res
    
class TPS2D(lightning.LightningModule):
    def __init__(self, params: TPSParams):
        super(TPS2D, self).__init__()

        self.params = params

        self.pretrained = DINOv2(model_name=self.params.image_encoder_name)

        self.n_cascades = 3

        self.image_encoder_dims = [self.pretrained.embed_dim for _ in range(self.n_cascades)]

        self.decoder = CascadeGlass3DDecoder(self.params, n_cascades=self.n_cascades,encoder_dims=self.image_encoder_dims)

        self.centerness_loss = CenterLoss(n_cascades=2, cascade_weights=self.params.cascade_weights[0:2]) # last cascade is final
        self.seg_loss = SegmentationLoss(n_cascades=3, cascade_weights=self.params.seg_cascade_weights)

    def load_image_backbone(self):
        self.load_state_dict({k: v for k, v in torch.load(str(self.params.encoder_weights_path), map_location='cpu', weights_only=False).items() if 'pretrained' in k}, strict=False)

    def configure_optimizers(self):

        image_backbone_params = self.pretrained.parameters()
        decoder_params = self.decoder.parameters()

        optimizer = torch.optim.AdamW([{"params": image_backbone_params, "lr": 5e-6},
                                       {"params": decoder_params, "lr": 5e-5}],
                                       weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=8, threshold=1e-4, min_lr=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/seg_miou",
                "name": "lr",
            }
        }
    
    def on_train_epoch_end(self):
        # self.params.w_l_plane_depth = np.clip((self.current_epoch // 20) * 0.2, 0, 1)
        # print("w_l_plane_depth: ", self.params.w_l_plane_depth)
        return

    def forward(self, image):

        image_features = self.pretrained.get_intermediate_layers(image, n=self.n_cascades, return_class_token=False, reshape=True)
        res = self.decoder(image_features)

        return res
    
    def training_step(self, batch, batch_idx):
        image, centerness, mask = batch

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

        # loss without running norm
        loss_sum = self.params.w_l_centerness *    l_centerness        + \
                   self.params.w_l_seg_ce *        l_seg_ce            + \
                   self.params.w_l_seg_miou *      l_seg_miou
        
        batch_size = image.shape[0]
        self.log("train/total_loss", loss_sum, sync_dist=True, batch_size=batch_size)
        self.log("train/centerness", l_centerness, sync_dist=True, batch_size=batch_size)
        self.log("train/seg_ce", l_seg_ce, sync_dist=True, batch_size=batch_size)
        self.log("train/seg_miou", l_seg_miou, sync_dist=True, batch_size=batch_size)
        self.log("train/seg_dice", l_seg_dice, sync_dist=True, batch_size=batch_size)

        return loss_sum
    
    def validation_step(self, batch, batch_idx):
        image, centerness, mask = batch

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

        # loss without running norm
        loss_sum = self.params.w_l_centerness *    l_centerness        + \
                   self.params.w_l_seg_ce *        l_seg_ce            + \
                   self.params.w_l_seg_miou *      l_seg_miou
        
        batch_size = image.shape[0]
        self.log("val/total_loss", loss_sum, sync_dist=True, batch_size=batch_size)
        self.log("val/centerness", l_centerness, sync_dist=True, batch_size=batch_size)
        self.log("val/seg_ce", l_seg_ce, sync_dist=True, batch_size=batch_size)
        self.log("val/seg_miou", l_seg_miou, sync_dist=True, batch_size=batch_size)
        self.log("val/seg_dice", l_seg_dice, sync_dist=True, batch_size=batch_size)

        return loss_sum

if __name__ == "__main__":
    from tps_params import small_tps_params
    params = small_tps_params()
    model = TPS2D(params=params)
    model.load_image_backbone()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    # image_size = 512
    image = torch.randn(2, 3, 518, 518)
    res = model(image)
    for key in res.keys():
        print(key)
        if isinstance(res[key], list):
            for i in range(len(res[key])):
                print(res[key][i].shape)
        else:
            print(res[key].shape)