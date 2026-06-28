from pathlib import Path
import torch

class TPSParams(object):
    def __init__(self, encoder_name = "base"):
        self.image_encoder_name = encoder_name
        self.encoder_weights_path = None

        self.base_hidden_dim = 256

        self.ang1_upscale = torch.pi * (2.5/4)
        self.ang2_upscale = torch.pi * (2.5/4)
        self.plane_dist_upscale = 5.0

        self.transformer_hidden_dim = 384
        self.n_transformer_heads = 6
        self.n_transformer_layers = 8

        # loss weights
        self.cascade_weights = [0.2, 0.3]
        self.seg_cascade_weights = [0.1, 0.1, 0.2]
        self.plane_cascade_weights = [0.1, 0.1, 0.2]
        
        self.w_l_boundary = 0.5
        self.w_l_centerness = 0.5
        self.w_l_seg_ce = 0.5
        self.w_l_seg_miou = 1
        self.w_l_seg_dice = 1
        self.w_l_plane_depth = 0
        self.w_l_plane_naive = 1
        self.w_l_plane_masked = 1.5
        self.w_l_plane_map = 0.5
        self.w_l_plane_cluster = 0.5
        self.w_l_plane_embvar = 0.5
        self.w_l_plane_embdiff = 0
        self.w_l_plane_dist = 1
        pass

def default_tps_params():
    return TPSParams()

def small_tps_params():
    params = TPSParams(encoder_name="vits")
    params.encoder_weights_path = Path(__file__).resolve().parent.parent.joinpath("weights", "depth_anything_v2", "depth_anything_v2_vits.pth")
    params.base_hidden_dim = 256
    params.transformer_hidden_dim = 256
    params.n_transformer_heads = 4
    params.n_transformer_layers = 6
    return params

def base_tps_params():
    params = TPSParams(encoder_name="vitb")
    params.encoder_weights_path = Path(__file__).resolve().parent.parent.joinpath("weights", "depth_anything_v2", "depth_anything_v2_metric_hypersim_vitb.pth")
    params.base_hidden_dim = 256
    params.transformer_hidden_dim = 256
    params.n_transformer_heads = 4
    params.n_transformer_layers = 6
    return params

def reduced_tps_params():
    params = TPSParams(encoder_name="vits")
    params.encoder_weights_path = Path(__file__).resolve().parent.parent.joinpath("weights", "depth_anything_v2", "depth_anything_v2_vits.pth")
    params.base_hidden_dim = 256
    params.transformer_hidden_dim = 256
    params.n_transformer_heads = 4
    params.n_transformer_layers = 6
    return params

def down_scaled_tps_params():
    params = TPSParams(encoder_name="vits")
    # params.encoder_weights_path = Path(__file__).resolve().parent.parent.joinpath("weights", "depth_anything_v2", "depth_anything_v2_metric_hypersim_vits.pth")
    params.encoder_weights_path = Path(__file__).resolve().parent.parent.joinpath("weights", "depth_anything_v2", "depth_anything_v2_vits.pth")
    params.base_hidden_dim = 192
    params.transformer_hidden_dim = 192
    params.n_transformer_heads = 4
    params.n_transformer_layers = 6
    return params

def tps2d_params():
    params = TPSParams(encoder_name="vits")
    params.encoder_weights_path = Path(__file__).resolve().parent.parent.joinpath("weights", "depth_anything_v2", "depth_anything_v2_metric_hypersim_vits.pth")
    # params.w_l_seg_ce = 0.5
    # params.w_l_centerness = 0.4
    return params
