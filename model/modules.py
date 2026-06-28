import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="bilinear", scale=4):
        super().__init__()
        align_corners = True
        if mode == "nearest":
            align_corners = None
        self.block = nn.Sequential(
            # 1. Resize (Artifact-free upsampling)
            nn.Upsample(scale_factor=scale, mode=mode, align_corners=align_corners),
            
            # 2. Convolution (Learn to sharpen and refine)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)

class MultiBranchFusion(nn.Module):
    def __init__(self, center_ch, plane_ch, out_ch, norm_groups=32):
        super().__init__()
        self.center_conv = nn.Sequential(
            nn.Conv2d(center_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_ch),
            nn.GELU()
        )
        self.plane_conv = nn.Sequential(
            nn.Conv2d(plane_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_ch),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch, 1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_ch),
            nn.GELU()
        )

    def forward(self, center_feat, depth_feat):
        center = self.center_conv(center_feat)
        plane = self.plane_conv(depth_feat)
        fused = self.fusion(torch.cat([plane, center], dim=1))

        return center, fused, plane
    
class CascadeHeads(nn.Module):
    def __init__(self, in_channels, upsample_scale=1, norm_groups=32):
        super().__init__()
        self.upsample_scale = upsample_scale

        self.center_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.GELU(),
            UpSampleBlock(in_channels // 2, in_channels // 2, scale=self.upsample_scale),
            nn.Conv2d(in_channels // 2, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.seg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.GELU(),
            UpSampleBlock(in_channels // 2, in_channels // 2, scale=self.upsample_scale),
            nn.Conv2d(in_channels // 2, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.plane_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            UpSampleBlock(in_channels // 2, in_channels // 2, scale=self.upsample_scale),
            nn.Conv2d(in_channels // 2, 3, 1, bias=True)
        )
        
    def forward(self, x_center, x_seg, x_plane):
        center = self.center_conv(x_center) 
        seg = self.seg_conv(x_seg + x_seg * (1 - F.interpolate(center, size=x_seg.shape[2:], mode='bilinear', align_corners=True)))
        plane = self.plane_conv(x_plane + x_plane * (1 - F.interpolate(center, size=x_plane.shape[2:], mode='bilinear', align_corners=True)))
        
        return center, seg, plane
    
class RefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels, proj_channels=64, norm_groups=32):
        super(RefinementBlock, self).__init__()

        self.center_proj = nn.Conv2d(1, proj_channels, 3, padding=1)
        self.center_refine = nn.Sequential(
            nn.Conv2d(in_channels + proj_channels, out_channels, 1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
            nn.GELU()
        )

        self.seg_proj = nn.Conv2d(1, proj_channels, 3, padding=1)
        self.seg_refine = nn.Sequential(
            nn.Conv2d(in_channels + proj_channels, out_channels, 1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),    
            nn.GELU()
        )
        
        self.plane_proj = nn.Conv2d(3, proj_channels, 3, padding=1)
        self.plane_refine = nn.Sequential(
            nn.Conv2d(in_channels + proj_channels, out_channels, 1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
            nn.GELU()
        )

    def forward(self, x_center, center, x_seg, seg, x_plane, plane):
        x_center = self.center_refine(torch.cat([x_center, self.center_proj(center)], dim=1))
        x_seg = self.seg_refine(torch.cat([x_seg, self.seg_proj(seg)], dim=1))
        x_plane = self.plane_refine(torch.cat([x_plane, self.plane_proj(plane)], dim=1))
        return x_center, x_seg, x_plane

# For 2D seg
class CascadeHeads2D(nn.Module):
    def __init__(self, in_channels):
        super(CascadeHeads2D, self).__init__()
        self.center_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 2, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.seg_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 1, 1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x_center, x_seg, x_plane):
        center = self.center_conv(x_center)
        seg = self.seg_conv(torch.cat([
            x_seg + x_seg * (1 - center),
            x_plane + x_plane * (1 - center)
        ], dim=1))
        return center, seg

class RefinementBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefinementBlock2D, self).__init__()

        self.center_refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, out_channels, 1)
        )
        self.seg_refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, out_channels, 1)
        )

    def forward(self, x_center, x_seg):
        x_center = self.center_refine(x_center)
        x_seg = self.seg_refine(x_seg)
        return x_center, x_seg