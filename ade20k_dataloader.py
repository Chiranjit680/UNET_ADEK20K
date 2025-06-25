import os
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
from torchvision import transforms

class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                crop_size=321, scale=True, flip=True, rotate=False, blur=False, return_id=False):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
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

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    
        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blur (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, val_split = 0.0):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        if val_split: self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else: self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
        
        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)
        
        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

# ADE20K color palette for visualization
ADE20K_palette = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,
                 3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,
                 5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,
                 255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,
                 6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,
                 92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,
                 10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,
                 0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,
                 163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224
                 ,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,
                 200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,
                 163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,
                 255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,
                 255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,
                 255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,
                 122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,
                 255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,
                 255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,
                 0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,
                 0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,
                 20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,
                 255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,
                 255,214,0,25,194,194,102,255,0,92,0,255]

class ADE20KDataset(BaseDataSet):
    """
    ADE20K dataset 
    http://groups.csail.mit.edu/vision/datasets/ADE20K/
    """
    def __init__(self, **kwargs):
        self.num_classes = 150
        self.palette = ADE20K_palette
        super(ADE20KDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ["training", "validation"]:
            self.image_dir = os.path.join(self.root, 'images', self.split)
            self.label_dir = os.path.join(self.root, 'annotations', self.split)
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
        else:
            raise ValueError(f"Invalid split name {self.split}")
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32) - 1  # from -1 to 149
        return image, label, image_id

class ADE20K(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):
        
        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = ADE20KDataset(**kwargs)
        super(ADE20K, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

# Example usage
if __name__ == "__main__":
    # Create training data loader
    train_loader = ADE20K(
        data_dir='./ADE20K',
        batch_size=16,
        split='training',
        crop_size=512,
        base_size=520,
        scale=True,
        num_workers=4,
        shuffle=True,
        flip=True,
        rotate=True,
        blur=True,
        augment=True
    )
    
    # Create validation data loader
    val_loader = ADE20K(
        data_dir='./ADE20K',
        batch_size=16,
        split='validation',
        crop_size=512,
        base_size=520,
        scale=False,
        num_workers=4,
        shuffle=False,
        augment=False,
        val=True
    )
    
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")