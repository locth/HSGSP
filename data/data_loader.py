import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class HSGSP_DataLoader:
    """Unified dataset loader for HSGSP project"""

    def __init__(self, config):
        self.config = config

    def _apply_mixup(self, images, labels):
        use_mixup = bool(getattr(self.config, "use_mixup", False))
        mixup_alpha = float(getattr(self.config, "mixup_alpha", 0.0))
        mixup_prob = float(getattr(self.config, "mixup_prob", 0.0))
        if not use_mixup or mixup_alpha <= 0.0 or mixup_prob <= 0.0:
            return images, labels

        if random.random() < mixup_prob:
            batch_size = images.size()[0]
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(images.device)
            images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
            return images, (labels_a, labels_b, lam)
        return images, labels

    def load_cifar10(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess CIFAR-10 dataset.

        Returns:
            train_loader: augmented training dataset (with stochastic transforms)
            val_loader: validation dataset without augmentation
            test_loader: test dataset without augmentation
            train_eval_loader: clean view of the training set (no augmentation) for evaluation
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.4)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.4)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ] if self.config.data_augmentation else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        train_eval_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.config.validation_split * num_train))

        train_idx, val_idx = indices[split:], indices[:split]

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_eval_subset = Subset(train_eval_dataset, train_idx)

        train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        train_eval_loader = DataLoader(train_eval_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, val_loader, test_loader, train_eval_loader

    def load_cifar100(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess CIFAR-100 dataset."""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.4)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.4)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ] if self.config.data_augmentation else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        train_eval_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.config.validation_split * num_train))

        train_idx, val_idx = indices[split:], indices[:split]

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_eval_subset = Subset(train_eval_dataset, train_idx)

        train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        train_eval_loader = DataLoader(train_eval_subset, batch_size=self.config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, val_loader, test_loader, train_eval_loader