import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import Tuple

SEED = 42

class HSGSP_DataLoader:
    """Unified dataset loader for HSGSP project"""

    def __init__(self, config):
        self.config = config

    def _apply_mixup(self, dataloader: DataLoader) -> DataLoader:
        use_mixup = bool(getattr(self.config, "use_mixup", False))
        mixup_alpha = float(getattr(self.config, "mixup_alpha", 0.0))
        mixup_prob = float(getattr(self.config, "mixup_prob", 0.0))
        if not use_mixup or mixup_alpha <= 0.0 or mixup_prob <= 0.0:
            return dataloader

        class MixupWrapper:
            def __init__(self, dl, alpha, prob):
                self.dl = dl
                self.alpha = alpha
                self.prob = prob

            def __iter__(self):
                for batch in self.dl:
                    images, labels = batch
                    labels = labels.long()
                    if torch.rand(1).item() < self.prob:
                        gamma1 = torch.distributions.gamma.Gamma(self.alpha, 1.0).sample()
                        gamma2 = torch.distributions.gamma.Gamma(self.alpha, 1.0).sample()
                        lam = gamma1 / (gamma1 + gamma2)
                        batch_size = images.size(0)
                        indices = torch.randperm(batch_size)
                        shuffled_images = images[indices]
                        shuffled_labels = labels[indices]
                        mixed_images = lam * images + (1.0 - lam) * shuffled_images
                        mixed_labels = lam * torch.nn.functional.one_hot(labels, num_classes=10).float() + (1.0 - lam) * torch.nn.functional.one_hot(shuffled_labels, num_classes=10).float()
                        yield mixed_images, mixed_labels
                    else:
                        yield images, torch.nn.functional.one_hot(labels, num_classes=10).float()

            def __len__(self):
                return len(self.dl)

        return MixupWrapper(dataloader, mixup_alpha, mixup_prob)

    def load_cifar10(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess CIFAR-10 dataset.

        Returns:
            train_dl: augmented training dataloader (with stochastic transforms)
            val_dl: validation dataloader without augmentation
            test_dl: test dataloader without augmentation
            train_eval_dl: clean view of the training set (no augmentation) for evaluation
        """
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.4, saturation=0.4)], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]) if self.config.data_augmentation else transforms.Compose([transforms.ToTensor(), normalize])

        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        full_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        train_clean_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        val_size = int(len(full_train_set) * self.config.validation_split)
        train_size = len(full_train_set) - val_size
        train_set, val_set = random_split(full_train_set, [train_size, val_size])
        train_clean_set, val_clean_set = random_split(train_clean_set, [train_size, val_size])  # But we use train_clean_set for eval

        train_dl = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        val_dl = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        test_dl = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        train_eval_dl = DataLoader(train_clean_set, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

        train_dl = self._apply_mixup(train_dl)

        return train_dl, val_dl, test_dl, train_eval_dl

    def load_cifar100(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Load and preprocess CIFAR-100 dataset.

        Returns:
            train_dl: augmented training dataloader
            val_dl: validation dataloader without augmentation
            test_dl: test dataloader without augmentation
            train_eval_dl: clean training dataloader for evaluation/monitoring
        """
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.4, saturation=0.4)], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]) if self.config.data_augmentation else transforms.Compose([transforms.ToTensor(), normalize])

        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        full_train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        train_clean_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=test_transform)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

        val_size = int(len(full_train_set) * self.config.validation_split)
        train_size = len(full_train_set) - val_size
        train_set, val_set = random_split(full_train_set, [train_size, val_size])
        train_clean_set, val_clean_set = random_split(train_clean_set, [train_size, val_size])

        train_dl = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        val_dl = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        test_dl = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        train_eval_dl = DataLoader(train_clean_set, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

        train_dl = self._apply_mixup(train_dl)

        return train_dl, val_dl, test_dl, train_eval_dl