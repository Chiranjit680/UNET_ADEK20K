import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torchvision import transforms
from glob import glob

# ADE20K color palette
ADE20K_PALETTE = [
    0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200,
    3, 120, 120, 80, 140, 140, 140, 204, 5, 255, 230, 230, 230, 4, 250, 7, 224,
    5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82, 143,
    255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255,
    6, 51, 11, 102, 255, 255, 7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9,
    92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6, 10, 255, 71, 255, 41,
    10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8,
    0, 255, 20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0,
    163, 255, 140, 140, 140, 250, 10, 15, 20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224,
    0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255, 31, 0, 255, 11, 200,
    200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255,
    163, 0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0,
    255, 173, 10, 0, 255, 173, 255, 0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245,
    255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0, 31, 255, 0, 255, 61, 0, 71, 255,
    255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0, 194, 255, 0,
    122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163,
    255, 0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184,
    255, 0, 214, 255, 255, 0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163,
    0, 255, 153, 0, 255, 71, 255, 0, 255, 0, 163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255,
    0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0, 10, 190, 212, 214, 255, 0, 0, 204, 255,
    20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0, 255, 41, 255, 0, 173, 0,
    255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0, 133, 255,
    255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255
]

class BaseDataSet(Dataset):
    """Base dataset class for semantic segmentation."""
    
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                 crop_size=321, scale=True, flip=True, rotate=False, blur=False, return_id=False):
        """
        Args:
            root: Root directory of dataset
            split: Dataset split ('training', 'validation', etc.)
            mean: Mean values for normalization
            std: Std values for normalization
            base_size: Base size for scaling
            augment: Whether to apply augmentations
            val: Whether this is validation mode
            crop_size: Size for cropping
            scale: Whether to apply scaling
            flip: Whether to apply horizontal flip
            rotate: Whether to apply rotation
            blur: Whether to apply Gaussian blur
            return_id: Whether to return image ID
        """
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        self.val = val
        self.return_id = return_id
        
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        
        self.files = []
        self._set_files()
        
        # Image preprocessing
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        
        # Set OpenCV threads to 0 to avoid conflicts
        cv2.setNumThreads(0)

    def _set_files(self):
        """Set file paths. To be implemented by subclasses."""
        raise NotImplementedError

    def _load_data(self, index):
        """Load data at given index. To be implemented by subclasses."""
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        """Apply validation-time augmentations (resize and center crop)."""
        if self.crop_size:
            h, w = label.shape
            
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center crop
            h, w = label.shape
            start_h = (h - self.crop_size) // 2
            start_w = (w - self.crop_size) // 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        
        return image, label

    def _augmentation(self, image, label):
        """Apply training-time augmentations."""
        h, w, _ = image.shape
        
        # Scaling - set the bigger side to base size
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            
            if h > w:
                h, w = (longside, int(1.0 * longside * w / h + 0.5))
            else:
                h, w = (int(1.0 * longside * h / w + 0.5), longside)
            
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        h, w, _ = image.shape
        
        # Rotation
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)

        # Padding to ensure we can crop to the desired size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            
            if pad_h > 0 or pad_w > 0:
                pad_kwargs = {
                    "top": 0,
                    "bottom": pad_h,
                    "left": 0,
                    "right": pad_w,
                    "borderType": cv2.BORDER_CONSTANT,
                }
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

            # Random cropping
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random horizontal flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian blur
        if self.blur:
            sigma = random.random() * 1.5  # sigma between 0 and 1.5
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            if ksize > 0:
                image = cv2.GaussianBlur(
                    image, (ksize, ksize), 
                    sigmaX=sigma, sigmaY=sigma, 
                    borderType=cv2.BORDER_REFLECT_101
                )
        
        return image, label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """Get item at index."""
        image, label, image_id = self._load_data(index)
        
        # Apply augmentations
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        # Convert to tensors
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        
        if self.return_id:
            return self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = f"Dataset: {self.__class__.__name__}\n"