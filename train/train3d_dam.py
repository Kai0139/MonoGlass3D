import os
# os.environ['NCCL_TIMEOUT'] = '3600'  # 设置超时时间为0秒，表示无限制
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from model.tps_params import down_scaled_tps_params

from model.mono_g3d_trainer import MonoG3DTrainer
from data_loaders.gs3d_loader import GS3DDataModule

torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    

    root_dir = Path("/mnt/slurmfs-3090node2/user_data/kzhang740/gs3d_resplit_251213")
    data_module = GS3DDataModule(root_dir=root_dir, batch_size=3)

    train_sub_name = "train_1213"
    checkpoint_dir = Path(__file__).resolve().parent.parent.joinpath("checkpoints", train_sub_name)

    checkpoint_cb = ModelCheckpoint(
        monitor = "val/important_sum",
        dirpath = str(checkpoint_dir),
        filename = "ep{epoch}-val_iou={val/seg_miou:.3f}-val_depth={val/plane_depth_l1_masked:.3f}",
        save_top_k = 5,
        mode = "min",
        save_last=True,
        auto_insert_metric_name=False
    )

    log_path = Path(__file__).resolve().parent.parent.joinpath("tb_logs")
    logger = TensorBoardLogger(str(log_path), name="TPS3D", version=train_sub_name)

    trainer = lightning.Trainer(
        default_root_dir=str(checkpoint_dir),
        check_val_every_n_epoch=1,
        max_epochs=300,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=[0,1,2,3],
        strategy="ddp_find_unused_parameters_true",
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step", log_weight_decay=True)],
        logger=logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True
    )

    tps_params = down_scaled_tps_params()
    model = MonoG3DTrainer(params=tps_params)
    model.load_image_backbone()

    model = model.to("cuda")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    trainer.fit(model, data_module)
