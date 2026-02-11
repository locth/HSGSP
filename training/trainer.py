import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List, Tuple
import os
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import Config
from utils.logger import Logger
from utils.overfitting_monitor import OverfittingMonitor, AdaptiveRegularization
from training.distillation import Distiller
from training.frequency_regularizer import (
    FrequencyRegularizedModel,
    SpectralEntropyRegularizer,
)
from tqdm import tqdm

class HSGSPTrainer:
    """Training manager for HSGSP pruning"""
    
    def __init__(self, config):
        self.config = config
        self.logger = Logger(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Thêm dòng này

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    @staticmethod
    def _model_outputs_logits(model: Optional[nn.Module]) -> bool:
        last_module = list(model.modules())[-1]
        if isinstance(last_module, nn.Softmax):
            return False
        return True

    def _create_lr_schedule(self, optimizer, epochs: int) -> optim.lr_scheduler._LRScheduler:
        """
        Create learning rate schedule with optional warmup.
        """
        schedule = (self.config.lr_schedule or 'cosine').lower()
        warmup_epochs = max(0, int(getattr(self.config, 'lr_warmup_epochs', 0)))
        total_epochs = max(1, epochs)
        initial_lr = float(self.config.initial_lr)
        min_lr_value = float(getattr(self.config, 'min_lr', 1e-7))

        if schedule == 'cosine':
            min_factor = float(getattr(self.config, 'fine_tune_cosine_min_factor', 0.1))
            target_min_lr = max(min_lr_value, initial_lr * min_factor)
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=target_min_lr)
        elif schedule == 'exponential':
            decay_rate = float(self.config.lr_decay_rate)
            return ExponentialLR(optimizer, gamma=decay_rate)
        elif schedule == 'step':
            decay_steps = max(1, int(self.config.lr_decay_steps))
            decay_rate = float(self.config.lr_decay_rate)
            return StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
        elif schedule == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=self.config.reduce_lr_factor, patience=self.config.reduce_lr_patience)
        return None

    def compile_model(self,
                     model: nn.Module,
                     learning_rate: Optional[float] = None,
                     optimizer: Optional[str] = None) -> optim.Optimizer:
        """
        Create optimizer for the model
        """
        lr = learning_rate or self.config.initial_lr
        opt_name = optimizer or self.config.optimizer
        
        if opt_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif opt_name.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        elif opt_name.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=self.config.momentum, nesterov=True)
        elif opt_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        self.logger.info(f"Model compiled with {opt_name} optimizer, lr={lr}, label_smoothing={self.config.label_smoothing}")
        
        return optimizer

    def compute_loss_and_acc(self, outputs, labels, criterion):
        device = outputs.device
        if labels.dim() == 1:
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
        else:
            log_probs = nn.functional.log_softmax(outputs, dim=1)
            loss = - (labels * log_probs).sum(dim=1).mean()
            _, predicted = outputs.max(1)
            _, true_labels = labels.max(1)
            correct = predicted.eq(true_labels).sum().item()
        return loss, correct

    def train_cifar(self,
                      model: nn.Module,
                      train_dataloader: torch.utils.data.DataLoader,
                      val_dataloader: torch.utils.data.DataLoader,
                      epochs: Optional[int] = None,
                      train_eval_dataloader: Optional[torch.utils.data.DataLoader] = None) -> Dict:
        """
        Train the model with comprehensive anti-overfitting strategies
        """
        epochs = epochs or self.config.default_epochs
        # device = next(model.parameters()).device

        overfitting_monitor = OverfittingMonitor(
            patience=5,
            threshold=0.05
        )

        adaptive_reg = AdaptiveRegularization(
            initial_dropout=self.config.dropout_rate,
            max_dropout=min(0.7, self.config.dropout_rate * 2)
        )

        optimizer = self.compile_model(model, learning_rate=self.config.initial_lr)
        scheduler = self._create_lr_schedule(optimizer, epochs)
        model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        writer = SummaryWriter(log_dir=os.path.join(self.config.tensorboard_dir, datetime.now().strftime("%d%m%Y-%H%M%S")))

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            total = 0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss, correct = self.compute_loss_and_acc(outputs, labels, criterion)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total += labels.size(0)
                train_correct += correct

            train_acc = train_correct / total
            train_loss /= len(train_dataloader)

            model.eval()
            val_loss = 0
            val_correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss, correct = self.compute_loss_and_acc(outputs, labels, criterion)
                    val_loss += loss.item()
                    total += labels.size(0)
                    val_correct += correct

            val_acc = val_correct / total
            val_loss /= len(val_dataloader)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            overfitting_detected = overfitting_monitor.update(train_acc, val_acc)
            if overfitting_detected:
                new_dropout = adaptive_reg.adjust(overfitting_monitor.get_overfitting_score())
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = new_dropout

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(self.config.models_dir, f'best_model_epoch{epoch}_{val_acc:.3f}.pt'))

        writer.close()
        return self.history

    def fine_tune_cifar(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int,
        learning_rate: float,
        log_dir_suffix: str,
        train_eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tuple[nn.Module, Dict]:
        device = next(model.parameters()).device
        optimizer = self.compile_model(model, learning_rate=learning_rate)
        scheduler = self._create_lr_schedule(optimizer, epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        writer = SummaryWriter(log_dir=os.path.join(self.config.tensorboard_dir, log_dir_suffix))

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            total = 0

            for batch in tqdm(train_dataloader, desc=f"Fine-tune Epoch {epoch+1}/{epochs} - Training"):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss, correct = self.compute_loss_and_acc(outputs, labels, criterion)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total += labels.size(0)
                train_correct += correct

            train_acc = train_correct / total
            train_loss /= len(train_dataloader)

            model.eval()
            val_loss = 0
            val_correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Fine-tune Epoch {epoch+1}/{epochs} - Validation"):
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss, correct = self.compute_loss_and_acc(outputs, labels, criterion)
                    val_loss += loss.item()
                    total += labels.size(0)
                    val_correct += correct

            val_acc = val_correct / total
            val_loss /= len(val_dataloader)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        writer.close()
        meta = {'best_model_path': os.path.join(self.config.models_dir, 'fine_tuned_model.pt')}
        torch.save(model.state_dict(), meta['best_model_path'])
        return model, meta