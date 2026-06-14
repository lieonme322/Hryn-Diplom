import torch
from torch.utils.data import Dataset
import cv2
import os
import json

class ShelfDataset(Dataset):
    """Датасет для изображений торговых полок"""
    
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Загрузка изображения
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Загрузка аннотаций
        ann_path = os.path.join(self.annotation_dir, img_name.replace('.jpg', '.json'))
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        
        boxes = torch.as_tensor(annotations['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotations['labels'], dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target
