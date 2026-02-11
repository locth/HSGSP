import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from models.model_utils import ModelUtils

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Thêm dòng này
    
    def evaluate_model(self, 
                      model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      dataset_name: str = "") -> Dict:
        """Model evaluation"""
        print(f"\nEvaluating model on {dataset_name}...")

        model.eval()
        model.to(self.device)
        total_loss = 0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += torch.sum(preds == labels).item()
                _, top5_preds = torch.topk(outputs, 5, dim=1)
                total_top5_correct += sum(1 for i in range(labels.size(0)) if labels[i] in top5_preds[i])
                total_samples += inputs.size(0)
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        top5_accuracy = total_top5_correct / total_samples

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        confusion = confusion_matrix(all_labels, all_predictions)
        per_class_acc = np.diag(confusion) / confusion.sum(axis=1)

        inference_time = self._measure_inference_speed(model, dataloader)

        complexity_metrics = self._compute_model_complexity(model)

        results = {
            'dataset': dataset_name,
            'loss': avg_loss,
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'per_class_accuracy': per_class_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time_ms': inference_time,
            'model_size_mb': complexity_metrics['size_mb'],
            'total_params': complexity_metrics['total_params'],
            'flops': complexity_metrics['flops']
        }
        
        return results

    def _measure_inference_speed(self, 
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                num_samples: int = 100) -> float:
        """Measure average inference time"""
        times = []
        batches_to_take = max(1, int(np.ceil(num_samples / float(self.config.batch_size))))
        warmup_batches = min(5, batches_to_take)

        batch_iterator = iter(dataloader)
        measured_batches = 0

        for idx in range(warmup_batches + batches_to_take):
            try:
                x_batch, _ = next(batch_iterator)
            except StopIteration:
                break
            x_batch = x_batch.to(self.device)
            if x_batch.size(0) == 0:
                continue
            if idx < warmup_batches:
                with torch.no_grad():
                    _ = model(x_batch)
                continue

            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(x_batch)
            end_time = time.perf_counter()
            batch_time_ms = (end_time - start_time) * 1000.0
            per_sample_ms = batch_time_ms / float(x_batch.size(0))
            times.append(per_sample_ms)
            measured_batches += 1
            if measured_batches >= batches_to_take:
                break

        return float(np.mean(times)) if times else 0.0
    
    def _compute_model_complexity(self, model: torch.nn.Module) -> Dict:
        """Compute model complexity metrics"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate model size
        size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Estimate FLOPs
        flops = ModelUtils.compute_flops(model)
        
        return {
            'total_params': total_params,
            'size_mb': size_mb,
            'flops': flops
        }

    def benchmark_inference(self,
                            model: torch.nn.Module,
                            batch_size: int = 1,
                            warmup_runs: int = 20,
                            measure_runs: int = 100,
                            dataloader: Optional[torch.utils.data.DataLoader] = None,
                            dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """High-resolution inference benchmark with optional dataloader input."""
        if dataloader is not None:
            ds_iter = iter(dataloader)
            sample_batch = next(ds_iter)[0][:batch_size].to(self.device, dtype=dtype)
        else:
            input_shape = (batch_size, 3, 32, 32)
            sample_batch = torch.randn(input_shape).to(self.device, dtype=dtype)

        # Warmup
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for _ in range(max(0, warmup_runs)):
                _ = model(sample_batch)

        timings = []
        for _ in range(max(1, measure_runs)):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(sample_batch)
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

        batch_latency_ms = float(np.mean(timings))
        per_sample_ms = batch_latency_ms / max(1, sample_batch.size(0))
        throughput = 1000.0 / per_sample_ms if per_sample_ms > 0 else float('inf')

        return {
            'batch_latency_ms': batch_latency_ms,
            'per_sample_latency_ms': per_sample_ms,
            'throughput_samples_per_sec': throughput,
            'batch_size': int(sample_batch.size(0))
        }

    def estimate_accuracy(self,
                          model: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          max_batches: Optional[int] = None) -> Dict[str, float]:
        """Estimate loss/accuracy on (optionally truncated) dataset."""
        if max_batches is not None and max_batches > 0:
            dataloader = list(dataloader)[:max_batches]
            dataloader = torch.utils.data.DataLoader(dataloader, batch_size=self.config.batch_size)

        model.eval()
        model.to(self.device)
        total_loss = 0
        total_correct = 0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def compare_models(self,
                      original_model: torch.nn.Module,
                      pruned_model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      dataset_name: str = "") -> Dict:
        """Compare original and pruned models"""
        print(f"\nComparing models on {dataset_name}...")
        
        # Evaluate both models
        original_results = self.evaluate_model(original_model, dataloader, f"{dataset_name} (Original)")
        pruned_results = self.evaluate_model(pruned_model, dataloader, f"{dataset_name} (Pruned)")
        
        # Compute improvements
        comparison = {
            'accuracy_drop': original_results['accuracy'] - pruned_results['accuracy'],
            'speedup': original_results['inference_time_ms'] / pruned_results['inference_time_ms'],
            'compression_ratio': original_results['total_params'] / pruned_results['total_params'],
            'size_reduction': 1 - (pruned_results['model_size_mb'] / original_results['model_size_mb']),
            'flops_reduction': 1 - (pruned_results['flops'] / original_results['flops'])
        }
        
        return {
            'original': original_results,
            'pruned': pruned_results,
            'comparison': comparison
        }