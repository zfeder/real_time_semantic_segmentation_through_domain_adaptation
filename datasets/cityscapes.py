import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class CityScapes(DataLoader):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images_dir = os.path.join(root, 'images', split)
        self.labels_dir = os.path.join(root, 'gtFine', split)
        self.images = []
        self.labels = []

        # Caricamento delle immagini e delle etichette
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            lbl_dir = os.path.join(self.labels_dir, city)
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(img_dir, file_name))
                    label_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    self.labels.append(os.path.join(lbl_dir, label_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        # Converti l'array numpy in tensore PyTorch
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        
        return image, label
    
print('Cityscapes')