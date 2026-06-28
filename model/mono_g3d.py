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

from model.transformer import build_transformer, build_transformer_encoder

from model.modules import MultiBranchFusion, CascadeHeads, RefinementBlock, UpSampleBlock
from model.tps_params import TPSParams

class AttentionModules(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_heads=4, num_layers=6):
        super(AttentionModules, self).__init__()

        self.input_down_sample = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, 1, stride=1),
        )
        self.decode_down_sample = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, 1, stride=1),
        )

        self.decode_self_attention = build_transformer_encoder(
            src_vocab_size=input_dim,
            tgt_vocab_size=output_dim,
            d_model=hidden_dim,
            N=num_layers,
            h=num_heads,
            dropout=0.1,
            d_ff=1024
        )

        self.transformer = build_transformer(
            src_vocab_size=input_dim,
            tgt_vocab_size=output_dim,
            d_model=hidden_dim,
            N=num_layers,
            h=num_heads,
            dropout=0.1,
            d_ff=1024
        )

    def forward(self, input_feat, decode_feat):
        """
        输入: feat - 特征图 [B, C, H, W]
        输出: weights - 平面分配权重 [B, K, H, W]
        """
        # print("input_feat shape: ", input_feat.shape)
        input_feat = self.input_down_sample(input_feat)
        decode_feat = self.decode_down_sample(decode_feat)
        B, C, H, W = input_feat.shape
        
        # 特征展平
        input_flat = input_feat.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        decode_flat = decode_feat.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        decode_flat = self.decode_self_attention(decode_flat, None)

        out = self.transformer(
            input_flat,  # 输入特征
            None,
            decode_flat,  # 查询向量
            None
        )
        return out, decode_flat
  

class CascadeGlass3DDecoder(nn.Module):
    def __init__(self, params: TPSParams, n_cascades: int, encoder_dims=[128,256,512,1024]):
        super(CascadeGlass3DDecoder, self).__init__()

        self.params = params
        self.encoder_dims = encoder_dims

        self.refine_proj_channels = 32

        self.n_cascades = n_cascades

        self.enc_fusion = nn.ModuleList(
            [
                MultiBranchFusion(self.encoder_dims[i], self.encoder_dims[i], self.params.base_hidden_dim) for i in range(self.n_cascades)
            ]
        )

        self.cas_refine = nn.ModuleList(
            [
                RefinementBlock(self.params.base_hidden_dim, self.params.base_hidden_dim, self.refine_proj_channels) if i > 0 else nn.Identity() for i in range(self.n_cascades)
            ]
        )

        self.cascade_scales = [1, 4, 8]
        self.cascade_heads = nn.ModuleList(
            [
                CascadeHeads(self.params.base_hidden_dim, self.cascade_scales[i]) for i in range(self.n_cascades)
            ]
        )

        self.final_plane_proj = nn.Conv2d(3, self.refine_proj_channels, 3, padding=1)
        self.final_plane_refine = nn.Conv2d(self.params.base_hidden_dim + self.refine_proj_channels, self.params.base_hidden_dim, 1)
        
        self.final_seg_head = nn.Sequential(
            nn.Conv2d(self.params.base_hidden_dim*2, self.params.base_hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=self.params.base_hidden_dim),
            nn.GELU(),
            UpSampleBlock(self.params.base_hidden_dim, self.params.base_hidden_dim, scale=4),
            nn.Conv2d(self.params.base_hidden_dim, self.params.base_hidden_dim // 4, 3, padding=1, bias=False),
            nn.Conv2d(self.params.base_hidden_dim // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.plane_clustering = AttentionModules(input_dim=self.params.base_hidden_dim,
                                                 output_dim=self.params.base_hidden_dim,
                                                 hidden_dim=self.params.transformer_hidden_dim,
                                                 num_heads=self.params.n_transformer_heads,
                                                 num_layers=self.params.n_transformer_layers)

        self.final_plane_emb = nn.Sequential(
            UpSampleBlock(self.params.base_hidden_dim, self.params.base_hidden_dim, scale=4),
            nn.GELU(),
            nn.Conv2d(self.params.base_hidden_dim, self.params.base_hidden_dim // 2, 3, padding=1, bias=False),
            UpSampleBlock(self.params.base_hidden_dim // 2, self.params.base_hidden_dim // 2, scale=3.5),
            nn.Conv2d(self.params.base_hidden_dim // 2, self.params.base_hidden_dim // 4, 1, padding=0, bias=False)
        )

        self.final_plane_head = nn.Conv2d(self.params.base_hidden_dim // 4, 3, 1, bias=True)

    def forward(self, features):

        cas_center, cas_seg, cas_plane = [], [], []
        
        for i in range(self.n_cascades):
            x = features[i]
            # Fusion
            x_center, x_fused, x_plane  = self.enc_fusion[i](x, x)
            # refine
            if i > 0:
                H, W = x_fused.shape[2:]
                x_center, x_fused, x_plane = self.cas_refine[i](
                    x_center, F.interpolate(cas_center[-1], size=(H, W), mode='bilinear'),
                    x_fused, F.interpolate(cas_seg[-1], size=(H, W), mode='bilinear'),
                    x_plane, F.interpolate(cas_plane[-1], size=(H, W), mode='nearest-exact')
                )

            # cascade pred
            center, seg, plane = self.cascade_heads[i](x_center, x_fused, x_plane)

            plane_ang1 = F.tanh(plane[:,0,:,:]) * self.params.ang1_upscale
            plane_ang2 = F.tanh(plane[:,1,:,:]) * self.params.ang2_upscale
            plane_dist = plane[:,2,:,:] * self.params.plane_dist_upscale
            plane = torch.stack([plane_ang1, plane_ang2, plane_dist], dim=1)
            
            cas_center.append(center)
            cas_seg.append(seg)
            cas_plane.append(plane)

        x_fused = x_fused + x_fused * (1 - F.interpolate(cas_center[-1], size=x_fused.shape[2:], mode='bilinear'))
        x_plane = x_plane + x_plane * (1 - F.interpolate(cas_center[-1], size=x_plane.shape[2:], mode='bilinear'))
        
        x_plane_proj = self.final_plane_proj(cas_plane[-1])
        x_plane = self.final_plane_refine(torch.cat([x_plane, F.interpolate(x_plane_proj, size=x_plane.shape[2:], mode='bilinear')], dim=1))
        B, _, H, W = x_plane.shape

        x_plane, x_fused = self.plane_clustering(x_plane, x_fused) # B, H*W, C
        _, N, C = x_plane.shape
        x_plane = x_plane.permute(0, 2, 1).view(B, C, H, W).contiguous() # B, C, H, W
        _, N, C = x_fused.shape
        x_fused = x_fused.permute(0, 2, 1).view(B, C, H, W).contiguous() # B, C, H, W

        final_seg = self.final_seg_head(torch.cat([x_fused, x_plane], dim=1))

        x_plane = self.final_plane_emb(x_plane) # B, C, H, W
        final_plane_pred = self.final_plane_head(x_plane) # B, 3, H, W
        final_ang1 = F.tanh(final_plane_pred[:,0,:,:]) * self.params.ang1_upscale
        final_ang2 = F.tanh(final_plane_pred[:,1,:,:]) * self.params.ang2_upscale
        final_dist = final_plane_pred[:,2,:,:] * self.params.plane_dist_upscale
        final_plane_pred = torch.stack([final_ang1, final_ang2, final_dist], dim=1)
        
        # put all results in a dict
        res = {
            "cas_center": cas_center,
            "cas_seg": cas_seg,
            "cas_plane": cas_plane,
            "final_seg": final_seg,
            "plane_emb": x_plane,
            "final_plane_pred": final_plane_pred
        }

        return res
    
class MonoG3D(nn.Module):
    def __init__(self, params: TPSParams):
        super(MonoG3D, self).__init__()

        self.params = params

        self.pretrained = DINOv2(model_name=self.params.image_encoder_name)

        self.n_cascades = 3

        self.image_encoder_dims = [self.pretrained.embed_dim for _ in range(self.n_cascades)]

        self.decoder = CascadeGlass3DDecoder(self.params, n_cascades=self.n_cascades,encoder_dims=self.image_encoder_dims)

    def forward(self, image):

        image_features = self.pretrained.get_intermediate_layers(image, n=self.n_cascades, return_class_token=False, reshape=True)
        res = self.decoder(image_features)

        return res
