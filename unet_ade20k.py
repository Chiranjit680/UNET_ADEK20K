import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
from typing import Tuple, List, Optional
import logging

from UNet import Unet
import utils
import engine
from learning_rate_range_test import LRTest
from ade20k_dataloader import ADE20K, ADE20K_palette

def setup_logging(log_dir: str = './logs'):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_mIoU_batch_efficient(pred: torch.Tensor, label: torch.Tensor, num_classes: int,
                                  ignore_index: int = -1) -> float:
    pred = pred.view(-1)
    label = label.view(-1)

    if ignore_index is not None:
        valid_mask = label != ignore_index
        pred = pred[valid_mask]
        label = label[valid_mask]

    mask = (label >= 0) & (label < num_classes)
    hist = torch.bincount(
        num_classes * label[mask] + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()

    diag = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - diag
    ious = diag / torch.clamp(union, min=1e-8)

    valid_ious = ious[union > 0]
    return valid_ious.mean().item() if len(valid_ious) > 0 else 0.0


class TrainingConfig:
    def __init__(self):
        self.num_classes = 150
        self.ignore_index = -1

        self.num_epochs = 100
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4

        self.crop_size = 512
        self.base_size = 512
        self.num_workers = 4

        self.use_amp = True
        self.log_interval = 10
        self.save_interval = 10

        self.data_dir = './ade/ADEChallengeData2016/images'  # ← fixed path
        self.save_dir = './checkpoints'
        self.log_dir = './logs'


class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = next(model.parameters()).device
        self.logger = setup_logging(config.log_dir)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=config.num_epochs,
            power=0.9
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
        self.scaler = GradScaler() if config.use_amp else None

        self.best_miou = 0.0
        self.train_losses = []
        self.val_mious = []
        self.epoch_times = []

        os.makedirs(config.save_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        running_loss = 0.0
        num_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}"
                )

        epoch_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        avg_loss = running_loss / num_samples

        self.epoch_times.append(epoch_time)
        return avg_loss, epoch_time, peak_memory

    def validate(self) -> float:
        self.model.eval()
        all_ious = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.config.use_amp:
                    with autocast():
                        preds = self.model(images)
                else:
                    preds = self.model(images)

                preds = torch.argmax(preds, dim=1)

                iou = compute_mIoU_batch_efficient(
                    preds, labels, self.config.num_classes, self.config.ignore_index
                )
                all_ious.append(iou)

        return np.mean(all_ious) if all_ious else 0.0

    def save_checkpoint(self, epoch, miou, filepath, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_miou': self.best_miou,
            'current_miou': miou,
            'train_losses': self.train_losses,
            'val_mious': self.val_mious,
            'epoch_times': self.epoch_times
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        if is_best:
            self.logger.info(f'Best model saved to {filepath} with mIoU: {miou:.4f}')
        else:
            self.logger.info(f'Checkpoint saved to {filepath}')

    def train(self, resume_from: str = None) -> float:
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_miou = checkpoint.get('best_miou', 0.0)
            self.logger.info(f'Resuming from epoch {start_epoch}')

        for epoch in range(start_epoch, self.config.num_epochs):
            avg_loss, epoch_time, peak_mem = self.train_epoch(epoch)
            avg_miou = self.validate()

            self.train_losses.append(avg_loss)
            self.val_mious.append(avg_miou)
            self.scheduler.step()

            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} — "
                             f"Loss: {avg_loss:.4f} — mIoU: {avg_miou:.4f} — "
                             f"Peak Mem: {peak_mem:.2f} MB")

            if avg_miou > self.best_miou:
                self.best_miou = avg_miou
                self.save_checkpoint(epoch, avg_miou, os.path.join(self.config.save_dir, "best_model.pth"), is_best=True)

        return self.best_miou


def create_data_loaders(config: TrainingConfig):
    

    train_loader = ADE20K(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        split='training',
        crop_size=config.crop_size,
        base_size=config.base_size,
        scale=True,
        num_workers=config.num_workers,
        shuffle=True,
        flip=True,
        rotate=True,
        blur=True,
        augment=True
    )

    val_loader = ADE20K(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        split='validation',
        crop_size=config.crop_size,
        base_size=config.base_size,
        scale=False,
        num_workers=config.num_workers,
        shuffle=False,
        augment=False,
        val=True
    )

    return train_loader, val_loader


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(channels=[3, 64, 128, 256, 512, 1024],
                 no_classes=config.num_classes,
                 output_size=(config.crop_size, config.crop_size))
    model = model.to(device)

    train_loader, val_loader = create_data_loaders(config)
    trainer = SegmentationTrainer(model, train_loader, val_loader, config)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    best_miou = trainer.train()
    total = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Training done in {total:.2f} sec. Best mIoU: {best_miou:.4f}")
    print(f"Peak GPU Memory: {peak_mem:.2f} MB")


if __name__ == "__main__":
    main()
