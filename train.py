"""
Training entry point for the camera-space hand motion estimator (stage 1).

This trains HAWOR only. The SLAM stage uses frozen off-the-shelf weights and the
motion infiller is trained separately.

    python train.py --cfg hawor/configs/hawor_train.yaml \
        --video_root datasets/hot3d_trainset_export

Run lib/datasets/hawor_preprocess_train.py over the sequences first.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, default_collate

from hawor.configs import get_config
from lib.datasets.hawor_train_dataset import HaworChunkDataset
from lib.models.hawor import HAWOR


def train_collate(items):
    """HAWOR.training_step reads joint_batch['img'], so the training batch is
    nested one level deeper than the validation batch."""
    return {'img': default_collate(items)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='hawor/configs/hawor_train.yaml')
    parser.add_argument('--video_root', type=str, required=True,
                        help='Root of the exported sequences')
    parser.add_argument('--train_set_file', type=str, default='train.json')
    parser.add_argument('--val_set_file', type=str, default='val.json')
    parser.add_argument('--exp_name', type=str, default='hawor')
    parser.add_argument('--out_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume the full trainer state from')
    parser.add_argument('--devices', type=int, default=torch.cuda.device_count() or 1)
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Stop after this many optimizer steps. Use a small value to '
                             'smoke-test a config before committing to a long run.')
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help='Override config entries, yacs style and last on the command '
                             'line: --opts TRAIN.LR 1e-4 MODEL.BACKBONE.FREEZE False')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.cfg, merge=True, update_cachedir=True)
    if args.opts:
        cfg.defrost()
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        print(f'config overrides: {args.opts}')

    precision = cfg.TRAIN.get('PRECISION', 'bf16-mixed')
    if precision == '16-mixed' and cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
        raise ValueError(
            "PRECISION='16-mixed' cannot be combined with GRAD_CLIP_VAL > 0. "
            "HAWOR uses manual optimization and clips with torch.nn.utils.clip_grad_norm_ "
            "(error_if_nonfinite=True), which sees fp16-scaled gradients and raises on the "
            "first inf. Use 'bf16-mixed' (no grad scaler) or set TRAIN.GRAD_CLIP_VAL: 0.")

    pl.seed_everything(cfg.GENERAL.get('SEED', 42), workers=True)

    train_dataset = HaworChunkDataset(
        args.video_root, args.train_set_file, cfg,
        seq_len=16, stride=cfg.TRAIN.get('CHUNK_STRIDE', 8), train=True)
    val_dataset = HaworChunkDataset(
        args.video_root, args.val_set_file, cfg,
        seq_len=16, stride=cfg.TRAIN.get('VAL_CHUNK_STRIDE', 64), train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.GENERAL.NUM_WORKERS,
        pin_memory=cfg.GENERAL.PIN_MEMORY,
        drop_last=True,
        persistent_workers=cfg.GENERAL.NUM_WORKERS > 0,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.GENERAL.NUM_WORKERS,
        pin_memory=cfg.GENERAL.PIN_MEMORY,
        drop_last=False,
        persistent_workers=cfg.GENERAL.NUM_WORKERS > 0,
    )

    model = HAWOR(cfg)

    log_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    # demo.py / the eval scripts expect a model_config.yaml next to the weights.
    with open(os.path.join(log_dir, 'model_config.yaml'), 'w') as f:
        f.write(cfg.dump())

    logger = TensorBoardLogger(save_dir=args.out_dir, name=args.exp_name,
                               default_hp_metric=False)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(log_dir, 'checkpoints'),
            filename='{epoch}-{step}',
            every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
            save_last=True,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        strategy='ddp' if args.devices > 1 else 'auto',
        precision=precision,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        max_steps=args.max_steps,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.GENERAL.LOG_STEPS,
        val_check_interval=cfg.GENERAL.get('VAL_CHECK_INTERVAL', 1.0),
        num_sanity_val_steps=cfg.GENERAL.get('SANITY_VAL_STEPS', 1),
        # No gradient_clip_val here: HAWOR uses manual optimization and clips
        # inside training_step via TRAIN.GRAD_CLIP_VAL.
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume)


if __name__ == '__main__':
    main()
