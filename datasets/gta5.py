import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

Label = namedtuple('Label', ['name', 'color', 'trainId'])

gta5_labels = [
    Label('road',        (128, 64, 128),   0),
    Label('sidewalk',    (244, 35, 232),   1),
    Label('building',    (70, 70, 70),     2),
    Label('wall',        (102, 102, 156),  3),
    Label('fence',       (190, 153, 153),  4),
    Label('pole',        (153, 153, 153),  5),
    Label('traffic light', (250, 170, 30), 6),
    Label('traffic sign', (220, 220, 0),   7),
    Label('vegetation',  (107, 142, 35),   8),
    Label('terrain',     (152, 251, 152),  9),
    Label('sky',         (70, 130, 180),  10),
    Label('person',      (220, 20, 60),   11),
    Label('rider',       (255, 0, 0),     12),
    Label('car',         (0, 0, 142),     13),
    Label('truck',       (0, 0, 70),      14),
    Label('bus',         (0, 60, 100),    15),
    Label('train',       (0, 80, 100),    16),
    Label('motorcycle',  (0, 0, 230),     17),
    Label('bicycle',     (119, 11, 32),   18)
]

color_to_class = {label.color: label.trainId for label in gta5_labels}

def rgb_to_class(label_image):
    """
    Converte un'immagine di etichette RGB in un array di valori di classe.
    """
    label_np = np.array(label_image)
    height, width, _ = label_np.shape
    class_label = np.zeros((height, width), dtype=np.uint8)

    for color, class_id in color_to_class.items():
        mask = np.all(label_np == color, axis=-1)
        class_label[mask] = class_id

    return class_label

class GTA5Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, augmentations=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.augmentations = augmentations
        self.images_dir = os.path.join(root, 'images')
        self.labels_dir = os.path.join(root, 'labels')
        self.images = []
        self.labels = []

        for file_name in os.listdir(self.images_dir):
            if file_name.endswith('.png'):
                self.images.append(os.path.join(self.images_dir, file_name))
                label_name = file_name.replace('.png', '.png')
                self.labels.append(os.path.join(self.labels_dir, label_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('RGB')

        if self.augmentations:
            image, label = self.augmentations(image, label)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        label = rgb_to_class(label)
        label = torch.from_numpy(label).long()

        return image, label

    def visualize_sample(self, idx):
        """Visualizza un campione di immagine e maschera con augmentations."""
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('RGB')

        if self.augmentations:
            image, label = self.augmentations(image, label)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Immagine con Augmentations")
        axes[1].imshow(label)
        axes[1].set_title("Maschera con Augmentations")
        plt.show()


    def check_label_format(self, idx):
        """Verifica i valori unici in una etichetta per un determinato indice."""
        label = Image.open(self.labels[idx]).convert('RGB')
        label_np = np.array(label, dtype=np.uint8)
        unique_labels = np.unique(label_np.reshape(-1, label_np.shape[2]), axis=0)
        print(f"Valori unici trovati nell'etichetta {idx}: {unique_labels}")



if __name__ == "__main__":
    dataset = GTA5Dataset(root='/kaggle/input/gta5data/GTA5')
    dataset.visualize_sample(0)
    dataset.check_label_format(0)

