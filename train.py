import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
from itertools import cycle
from config import config 


from utils.metrics import fast_hist, per_class_iou
from models.bisenet import BiSeNet
from models.deeplab import get_deeplab_v2
from models.fc_discriminator import FCDiscriminator
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5Dataset


torch.autograd.set_detect_anomaly(True)  # Enable backpropagation anomaly detection

class SemanticSegmentationPipeline:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.adversarial = config.get("adversarial", False)  # Flag for adversial training
        self.model, self.optimizer, self.loss_fn, self.model_D, self.optimizer_D, self.loss_D = self._initialize_model()
        self.train_loader, self.val_loader, self.data_height, self.data_width = self._initialize_dataloaders()

    def _initialize_model(self):
        model_name = self.config["model_name"]
        num_classes = self.config["num_classes"]

        if model_name == "DeepLabV2":
            model = get_deeplab_v2(
                num_classes=num_classes,
                pretrain=True,
                pretrain_model_path=self.config["DEEPLABV2_PATH"]
            ).to(self.device)
        elif model_name == "BiSeNet":
            model = BiSeNet(num_classes=num_classes, context_path="resnet18").to(self.device)
        else:
            raise ValueError("Only 'DeepLabV2' and 'BiSeNet' are supported in this implementation.")

        if self.config["parallelize"] and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(self.device)

        optimizer = self._initialize_optimizer(model)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.config["ignore_index"])

        
        if self.adversarial:
            model_D = FCDiscriminator(num_classes=num_classes).to(self.device)
            if self.config["parallelize"] and torch.cuda.device_count() > 1:
                model_D = torch.nn.DataParallel(model_D).to(self.device)
            optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-3, betas=(0.9, 0.99))
            loss_D = nn.BCEWithLogitsLoss()
        else:
            model_D, optimizer_D, loss_D = None, None, None

        return model, optimizer, loss_fn, model_D, optimizer_D, loss_D

    def _initialize_optimizer(self, model):
        optimizer_name = self.config["optimizer_name"]
        if optimizer_name == "Adam":
            return torch.optim.Adam(model.parameters(), lr=self.config["lr"])
        elif optimizer_name == "SGD":
            return torch.optim.SGD(model.parameters(), lr=self.config["lr"], momentum=self.config["momentum"],
                                   weight_decay=self.config["weight_decay"])
        else:
            raise ValueError("Only 'Adam' and 'SGD' are supported as optimizers.")

    def _initialize_dataloaders(self):
        train_dataset = self._get_dataset(self.config["train_dataset_name"], split="train")
        val_dataset = self._get_dataset(self.config["val_dataset_name"], split="val")
        
        
        if self.config["train_dataset_name"] == "CityScapes":
            self.data_height, self.data_width = 512, 1024
        elif self.config["train_dataset_name"] == "GTAV":
            self.data_height, self.data_width = 720, 1280
        else:
            raise ValueError(f"Unsupported dataset: {self.config['train_dataset_name']}")
        
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True,
                                  num_workers=self.config["n_workers"])
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=False,
                                num_workers=self.config["n_workers"])
        
        return train_loader, val_loader, self.data_height, self.data_width

    def _get_dataset(self, dataset_name, split):
        transform = self._get_transform(dataset_name)
    
        if dataset_name == "CityScapes":
            root_dir = self.config["CITYSCAPES_PATH"]
            return CityScapes(root_dir, split=split, transform=transform) 
        elif dataset_name == "GTAV":
            root_dir = self.config["GTAV_PATH"]
            augmentations_fn = self._get_augmentations(self.config["augmentations"])
            return GTA5Dataset(root=root_dir, transform=transform, augmentations=augmentations_fn)
        else:
            raise ValueError("Only 'CityScapes' and 'GTAV' are supported as datasets.")


    def _get_transform(self, dataset_name):
        
        if dataset_name == "CityScapes":
            image_height, image_width = 512, 1024
        elif dataset_name == "GTAV":
            image_height, image_width = 720, 1280
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        
        return transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_augmentations(self, augmentations):
        """
        Restituisce una funzione di augmentations in base al parametro configurato.
        """
        def apply_augmentations(image, label):
            if augmentations == 1:  # Horizontal flip
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    label = TF.hflip(label)
            elif augmentations == 2:  # Horizontal flip + Rotazione
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    label = TF.hflip(label)
                angle = random.uniform(-10, 10)  # Angolo di rotazione
                image = TF.rotate(image, angle, resample=Image.BILINEAR)
                label = TF.rotate(label, angle, resample=Image.NEAREST)
            elif augmentations == 3:  # RandomResizeCrop + Horizontal flip + Gaussian blur + Color jitter
                # Random Resize Crop con probabilità 0.5
                if random.random() > 0.5:
                    scale = random.uniform(0.8, 1.0)  
                    ratio = random.uniform(0.9, 1.1) 
                    new_height = int(image.size[1] * scale)
                    new_width = int(image.size[0] * scale)
                    image = TF.resize(image, (new_height, new_width))
                    label = TF.resize(label, (new_height, new_width), interpolation=Image.NEAREST)
                    crop_height = min(new_height, int(image.size[1] * ratio))
                    crop_width = min(new_width, int(image.size[0] * ratio))
                    image = TF.center_crop(image, (crop_height, crop_width))
                    label = TF.center_crop(label, (crop_height, crop_width))
            
                # Horizontal flip
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    label = TF.hflip(label)
            
                # Color jitter
                if random.random() > 0.5:
                    image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                    image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                    image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            
                # Gaussian blur
                if random.random() > 0.5:
                    sigma = random.uniform(0.1, 2.0) 
                    image = TF.gaussian_blur(image, kernel_size=(5, 5), sigma=(sigma, sigma))
    
           
            final_height, final_width = 512, 1024 
            image = TF.resize(image, (final_height, final_width))
            label = TF.resize(label, (final_height, final_width), interpolation=Image.NEAREST)
    
            return image, label
    
        return apply_augmentations


    
    def _save_checkpoint(self, epoch):
        """Salva lo stato del modello, dell'ottimizzatore e dell'epoca corrente."""
        checkpoint_path = os.path.join(self.config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint salvato: {checkpoint_path}")

    def _load_checkpoint(self):
        """Carica il checkpoint più recente se disponibile."""
        checkpoint_files = [f for f in os.listdir(self.config["checkpoint_dir"]) if f.endswith(".pth")]
        if not checkpoint_files:
            print("Nessun checkpoint trovato. Inizio del training da zero.")
            return 0 
        
        
        latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        checkpoint_path = os.path.join(self.config["checkpoint_dir"], latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  
        print(f"Checkpoint caricato: {checkpoint_path}. Ripresa dal'epoca {start_epoch}.")
        return start_epoch


    def _adversarial_training_step(self):
        total_loss, total_miou = 0, 0
        class_ious = np.zeros(self.config["num_classes"])
        class_counts = np.zeros(self.config["num_classes"])
        lambda_adv = 0.001  
    
        self.model.train()
        self.model_D.train()
    
        
        target_iter = iter(self.val_loader)
    
        
        progress_bar = tqdm(self.train_loader, desc="Adversarial Training")
    
        for source_images, source_labels in progress_bar:
           
            try:
                target_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(self.val_loader)
                target_images, _ = next(target_iter)
    
           
            source_images, source_labels = source_images.to(self.device), source_labels.to(self.device)
            target_images = target_images.to(self.device)
    
           
            source_labels = torch.nn.functional.interpolate(
                source_labels.unsqueeze(1).float(),  
                size=(self.data_height, self.data_width),
                mode="nearest"
            ).squeeze(1).long()
    
            # **TRAIN GENERATOR**
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()
    
            # Forward sul dominio source
            source_outputs = self.model(source_images)
            if isinstance(source_outputs, tuple):  # Estrai solo il tensore principale
                source_outputs = source_outputs[0]
    
            # Calcola la segmentation loss
            segmentation_loss = self.loss_fn(source_outputs, source_labels)
    
            # Forward sul dominio target
            target_outputs = self.model(target_images)
            if isinstance(target_outputs, tuple):  # Estrai solo il tensore principale
                target_outputs = target_outputs[0]
    
            # Calcola la adversarial loss sul dominio target
            target_predictions = torch.softmax(target_outputs, dim=1)
            discriminator_target_predictions = self.model_D(target_predictions)
            adv_target_loss = self.loss_D(discriminator_target_predictions, torch.zeros_like(discriminator_target_predictions))
    
            # Loss totale per il generatore
            total_gen_loss = segmentation_loss + lambda_adv * adv_target_loss
            total_gen_loss.backward()
            self.optimizer.step()
    
            # **TRAIN DISCRIMINATOR**
            for param in self.model_D.parameters():
                param.requires_grad = True  # Riabilita i gradienti per il discriminatore
    
            # Forward per il discriminatore sul dominio source
            source_predictions = torch.softmax(source_outputs.detach(), dim=1)  # Detach per fermare i gradienti del generatore
            discriminator_source_predictions = self.model_D(source_predictions)
            source_loss_D = self.loss_D(discriminator_source_predictions, torch.ones_like(discriminator_source_predictions))
    
            # Forward per il discriminatore sul dominio target
            target_predictions = torch.softmax(target_outputs.detach(), dim=1)
            discriminator_target_predictions = self.model_D(target_predictions)
            target_loss_D = self.loss_D(discriminator_target_predictions, torch.zeros_like(discriminator_target_predictions))
    
            # Loss totale per il discriminatore
            total_discriminator_loss = (source_loss_D + target_loss_D) / 2
            total_discriminator_loss.backward()
            self.optimizer_D.step()
    
            # **Calcolo delle metriche**
            mean_iou, per_class_iou = self._compute_miou(source_outputs, source_labels)
            total_loss += segmentation_loss.item()
            total_miou += mean_iou
            class_ious += per_class_iou
            class_counts += (per_class_iou > 0).astype(int)
    
            
            progress_bar.set_postfix({
                "Seg Loss": segmentation_loss.item(),
                "Adv Loss": adv_target_loss.item(),
                "Mean IoU": mean_iou
            })
    
        # Calcolo delle IoU medie per classe
        avg_class_ious = class_ious / np.maximum(class_counts, 1)
        return total_loss / len(self.train_loader), total_miou / len(self.train_loader), avg_class_ious



    def _training_step(self):
        if self.adversarial:
            return self._adversarial_training_step() 
        else:
            return self._standard_training_step() 


    
    def train(self):
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        start_epoch = self._load_checkpoint()  
    
        for epoch in range(start_epoch, self.config["epochs"]):
            train_loss, train_miou, train_class_ious = self._training_step()
            val_loss, val_miou, val_class_ious = self._validation_step()
    
            print(f"Epoch {epoch + 1}/{self.config['epochs']} - Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Train mIoU: {train_miou:.4f}, Val mIoU: {val_miou:.4f}")
    
          
            print("Train IoU per class:")
            for i, iou in enumerate(train_class_ious):
                print(f"Class {i}: {iou:.4f}")
    
            print("Validation IoU per class:")
            for i, iou in enumerate(val_class_ious):
                print(f"Class {i}: {iou:.4f}")
    
           
            self._save_checkpoint(epoch)



    def _standard_training_step(self):
        self.model.train()
        total_loss, total_miou = 0, 0
        class_ious = np.zeros(self.config["num_classes"])  
        class_counts = np.zeros(self.config["num_classes"]) 
    
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
    
            labels[labels >= self.config["num_classes"]] = self.config["ignore_index"]
    
            labels = torch.nn.functional.interpolate(
                labels.unsqueeze(1).float(),
                size=(self.data_height, self.data_width),
                mode="nearest"
            ).squeeze(1).long()
    
            images = torch.nn.functional.interpolate(
                images,
                size=(self.data_height, self.data_width),
                mode="bilinear",
                align_corners=False
            )
    
            self.optimizer.zero_grad()
            outputs = self.model(images)
    
            if isinstance(outputs, tuple):
                outputs = outputs[0]
    
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
    
            total_loss += loss.item()
    
            # Calcolo della IoU
            mean_iou, per_class_iou = self._compute_miou(outputs, labels)
            total_miou += mean_iou
            class_ious += per_class_iou
            class_counts += (per_class_iou > 0).astype(int) 
    
        # Media delle IoU per classe
        avg_class_ious = class_ious / np.maximum(class_counts, 1)
    
        return total_loss / len(self.train_loader), total_miou / len(self.train_loader), avg_class_ious




    def _validation_step(self):
        self.model.eval()
        total_loss, total_miou = 0, 0
        class_ious = np.zeros(self.config["num_classes"])
        class_counts = np.zeros(self.config["num_classes"])
    
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
    
                labels[labels >= self.config["num_classes"]] = self.config["ignore_index"]
    
                labels = torch.nn.functional.interpolate(
                    labels.unsqueeze(1).float(),
                    size=(self.data_height, self.data_width),
                    mode="nearest"
                ).squeeze(1).long()
    
                images = torch.nn.functional.interpolate(
                    images,
                    size=(self.data_height, self.data_width),
                    mode="bilinear",
                    align_corners=False
                )
    
                outputs = self.model(images)
    
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
    
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
    
                # Calcolo della IoU
                mean_iou, per_class_iou = self._compute_miou(outputs, labels)
                total_miou += mean_iou
                class_ious += per_class_iou
                class_counts += (per_class_iou > 0).astype(int)
    
        avg_class_ious = class_ious / np.maximum(class_counts, 1)
        return total_loss / len(self.val_loader), total_miou / len(self.val_loader), avg_class_ious

    
    
    
    def _compute_miou(self, outputs, labels):
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        hist = fast_hist(labels.cpu().numpy(), preds.cpu().numpy(), self.config["num_classes"])
        class_ious = per_class_iou(hist)  
        mean_iou = np.mean(class_ious)  
        return mean_iou, class_ious


pipeline = SemanticSegmentationPipeline(config) 
pipeline.train()
