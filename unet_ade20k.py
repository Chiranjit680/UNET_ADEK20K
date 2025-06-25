# Main training script with improved organization
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

def setup_logging(log_dir: str = './logs'):
    """Setup logging configuration."""
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
    """
    Efficient batch-wise mIoU computation using confusion matrix.
    
    Args:
        pred: Predicted segmentation masks [N, H, W] or [N*H*W]
        label: Ground truth masks [N, H, W] or [N*H*W]
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        Mean IoU score
    """
    pred = pred.view(-1)
    label = label.view(-1)
    
    # Remove ignore_index pixels
    if ignore_index is not None:
        valid_mask = label != ignore_index
        pred = pred[valid_mask]
        label = label[valid_mask]
    
    # Create confusion matrix
    mask = (label >= 0) & (label < num_classes)
    hist = torch.bincount(
        num_classes * label[mask] + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()
    
    # Compute IoU for each class
    diag = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - diag
    
    # Avoid division by zero
    ious = diag / torch.clamp(union, min=1e-8)
    
    # Return mean IoU, ignoring classes not present
    valid_ious = ious[union > 0]
    return valid_ious.mean().item() if len(valid_ious) > 0 else 0.0

class TrainingConfig:
    """Configuration class for training parameters."""
    def __init__(self):
        # Model parameters
        self.num_classes = 150
        self.ignore_index = -1  # Use 255 for some datasets
        
        # Training parameters
        self.num_epochs = 100
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        
        # Data parameters
        self.crop_size = 512
        self.base_size = 512
        self.num_workers = 4
        
        # Training options
        self.use_amp = True
        self.log_interval = 10
        self.save_interval = 10
        
        # Paths
        self.data_dir = '.~/Desktop/UNET_ADEK20K/ade/ADEChallengeData2016/images'
        self.save_dir = './checkpoints'
        self.log_dir = './logs'

class SegmentationTrainer:
    """Main trainer class for semantic segmentation."""
    
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup logging
        self.logger = setup_logging(config.log_dir)
        
        # Setup optimizer and scheduler
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
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.use_amp else None
        
        # Tracking variables
        self.best_miou = 0.0
        self.train_losses = []
        self.val_mious = []
        self.epoch_times = []
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        running_loss = 0.0
        num_samples = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            batch_size = images.size(0)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
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
            
            running_loss += loss.item() * batch_size
            num_samples += batch_size
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f'Epoch {epoch+1}/{self.config.num_epochs}, '
                    f'Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        epoch_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        avg_loss = running_loss / num_samples
        
        self.epoch_times.append(epoch_time)
        return avg_loss, epoch_time, peak_memory
    
    def validate(self) -> float:
        """Validate the model and compute mIoU."""
        self.model.eval()
        all_ious = []
        
        with torch.no_grad():
            for val_images, val_labels in self.val_loader:
                val_images = val_images.cuda(non_blocking=True)
                val_labels = val_labels.cuda(non_blocking=True)
                
                if self.config.use_amp:
                    with autocast():
                        preds = self.model(val_images)
                else:
                    preds = self.model(val_images)
                
                preds = torch.argmax(preds, dim=1)
                
                iou = compute_mIoU_batch_efficient(
                    preds, val_labels, self.config.num_classes, self.config.ignore_index
                )
                all_ious.append(iou)
        
        return np.mean(all_ious) if all_ious else 0.0
    
    def save_checkpoint(self, epoch: int, miou: float, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
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
            self.logger.info(f'New best model saved to {filepath} with mIoU: {miou:.4f}')
        else:
            self.logger.info(f'Checkpoint saved to {filepath}')
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location='cuda')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_miou = checkpoint.get('best_miou', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_mious = checkpoint.get('val_mious', [])
        self.epoch_times = checkpoint.get('epoch_times', [])
        
        self.logger.info(f'Checkpoint loaded from {filepath}')
        return checkpoint.get('epoch', 0)
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # mIoU curve
        ax2.plot(self.val_mious, label='Validation mIoU', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.set_title('Validation mIoU')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f'Training curves saved to {save_path}')
        
        plt.show()
    
    def train(self, resume_from: str = None) -> float:
        """Main training loop."""
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            self.logger.info(f'Resuming training from epoch {start_epoch}')
        
        self.logger.info(f'Starting training for {self.config.num_epochs} epochs')
        self.logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        
        for epoch in range(start_epoch, self.config.num_epochs):
            # Training
            avg_loss, epoch_time, peak_memory = self.train_epoch(epoch)
            self.train_losses.append(avg_loss)
            
            # Validation
            avg_miou = self.validate()
            self.val_mious.append(avg_miou)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            self.logger.info("-" * 80)
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} Summary:")
            self.logger.info(f"  Train Loss: {avg_loss:.4f}")
            self.logger.info(f"  Val mIoU: {avg_miou:.4f}")
            self.logger.info(f"  Best mIoU: {self.best_miou:.4f}")
            self.logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            self.logger.info(f"  Peak Memory: {peak_memory:.2f} MB")
            self.logger.info(f"  Learning Rate: {current_lr:.2e}")
            self.logger.info("-" * 80)
            
            # Save best model
            if avg_miou > self.best_miou:
                self.best_miou = avg_miou
                best_path = os.path.join(self.config.save_dir, 'best_model.pth')
                self.save_checkpoint(epoch, avg_miou, best_path, is_best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'
                )
                self.save_checkpoint(epoch, avg_miou, checkpoint_path)
        
        # Save final training curves
        curves_path = os.path.join(self.config.save_dir, 'training_curves.png')
        self.plot_training_curves(curves_path)
        
        # Save final model
        final_path = os.path.join(self.config.save_dir, 'final_model.pth')
        self.save_checkpoint(self.config.num_epochs - 1, avg_miou, final_path)
        
        self.logger.info(f"Training completed! Best mIoU: {self.best_miou:.4f}")
        self.logger.info(f"Total training time: {sum(self.epoch_times):.2f}s")
        self.logger.info(f"Average epoch time: {np.mean(self.epoch_times):.2f}s")
        
        return self.best_miou

def create_data_loaders(config: TrainingConfig):
    """Create training and validation data loaders."""
    # This would use your existing ADE20K class
    import ADE20K  # Replace with actual import
    
    # Training loader
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
    
    # Validation loader
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
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create configuration
    config = TrainingConfig()
    
    # Customize config if needed
    config.num_epochs = 100
    config.batch_size = 8
    config.learning_rate = 1e-4
    config.crop_size = 512
    

    model = Unet(num_classes=150)
    
    # For demonstration, using a placeholder
    print("Please uncomment and replace with your actual U-Net model initialization")
    # model = YourUnetModel(num_classes=config.num_classes)
    # model = model.cuda()
    
    # Create data loaders
    print("Please uncomment and replace with your actual data loader creation")
    train_loader, val_loader = create_data_loaders(config)
    
    # Create trainer
    trainer = SegmentationTrainer(model, train_loader, val_loader, config)
    
    # Start training
    torch.cuda.reset_peak_memory_stats()
    # Reset memory stats
    start_time = time.time()
    best_miou = trainer.train()
    end_time = time.time()
    total_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Training completed in {total_time:.2f} seconds. Best mIoU: {best_miou:.4f}")
    print
    
 

if __name__ == "__main__":
    main()