import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2

class MedicalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform or Compose([
            Resize(640, 640),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(),
            ToTensorV2()
        ])
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 0-8
        
        aug = self.transform(image=image, mask=mask)
        return aug['image'], aug['mask'].long()  # LongTensor 用于 CrossEntropy