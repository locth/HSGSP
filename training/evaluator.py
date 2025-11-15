# training/evaluator.py
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional
import time

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_model(self, 
                      model: tf.keras.Model,
                      dataset: tf.data.Dataset,
                      dataset_name: str = "") -> Dict:
        """Model evaluation"""
        print(f"\nEvaluating model on {dataset_name}...")

        # Ensure compiled for metrics (no training impact)
        opt = getattr(model, "optimizer", None) or tf.keras.optimizers.SGD()
        loss = getattr(model, "loss", None) or "categorical_crossentropy"
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
        )
        
        # Basic metrics
        # loss, accuracy = model.evaluate(dataset, verbose=1)
        loss, accuracy, top5_acc = model.evaluate(dataset, verbose=1)
        
        # Per-class metrics
        class_metrics = self._compute_per_class_metrics(model, dataset)
        
        # Inference speed
        inference_time = self._measure_inference_speed(model, dataset)
        
        # Model complexity
        complexity_metrics = self._compute_model_complexity(model)
        
        results = {
            'dataset': dataset_name,
            'loss': loss,
            'accuracy': accuracy,
            'top5_accuracy': top5_acc,
            'per_class_accuracy': class_metrics['per_class_acc'],
            'precision': class_metrics['precision'],
            'recall': class_metrics['recall'],
            'f1_score': class_metrics['f1'],
            'inference_time_ms': inference_time,
            'model_size_mb': complexity_metrics['size_mb'],
            'total_params': complexity_metrics['total_params'],
            'flops': complexity_metrics['flops']
        }
        
        return results
    
    def _compute_per_class_metrics(self, 
                                  model: tf.keras.Model,
                                  dataset: tf.data.Dataset) -> Dict:
        """Compute per-class metrics"""
        all_predictions = []
        all_labels = []
        
        for x_batch, y_batch in dataset:
            predictions = model.predict(x_batch, verbose=0)
            all_predictions.append(np.argmax(predictions, axis=1))
            all_labels.append(np.argmax(y_batch, axis=1))
        
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        # Compute metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        
        # Per-class accuracy
        confusion = confusion_matrix(all_labels, all_predictions)
        per_class_acc = np.diag(confusion) / confusion.sum(axis=1)
        
        return {
            'per_class_acc': per_class_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion
        }
    
    def _measure_inference_speed(self, 
                                model: tf.keras.Model,
                                dataset: tf.data.Dataset,
                                num_samples: int = 100) -> float:
        """Measure average inference time"""
        times = []
        batches_to_take = max(1, int(np.ceil(num_samples / float(self.config.batch_size))))
        warmup_batches = min(5, batches_to_take)

        batch_iterator = dataset.take(warmup_batches + batches_to_take)
        measured_batches = 0

        for idx, (x_batch, _) in enumerate(batch_iterator):
            if x_batch.shape[0] == 0:
                continue
            if idx < warmup_batches:
                _ = model(x_batch, training=False)
                continue

            start_time = time.perf_counter()
            _ = model(x_batch, training=False)
            end_time = time.perf_counter()
            batch_time_ms = (end_time - start_time) * 1000.0
            per_sample_ms = batch_time_ms / float(x_batch.shape[0])
            times.append(per_sample_ms)
            measured_batches += 1
            if measured_batches >= batches_to_take:
                break

        return float(np.mean(times)) if times else 0.0
    
    def _compute_model_complexity(self, model: tf.keras.Model) -> Dict:
        """Compute model complexity metrics"""
        # Count parameters
        total_params = model.count_params()
        
        # Estimate model size
        size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Estimate FLOPs
        from models.model_utils import ModelUtils
        flops = ModelUtils.compute_flops(model)
        
        return {
            'total_params': total_params,
            'size_mb': size_mb,
            'flops': flops
        }

    def benchmark_inference(self,
                            model: tf.keras.Model,
                            batch_size: int = 1,
                            warmup_runs: int = 20,
                            measure_runs: int = 100,
                            dataset: Optional[tf.data.Dataset] = None,
                            dtype: tf.dtypes.DType = tf.float32) -> Dict[str, float]:
        """High-resolution inference benchmark with optional dataset input."""
        if not hasattr(model, 'input_shape'):
            raise ValueError("Model must have a defined input shape for benchmarking.")

        if dataset is not None:
            ds_iter = iter(dataset)
            try:
                sample_batch = next(ds_iter)[0]
            except (StopIteration, TypeError):
                raise ValueError("Dataset must yield (inputs, labels) tuples with non-empty batches.")
            inputs = tf.convert_to_tensor(sample_batch[:batch_size])
        else:
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            if input_shape[0] is None:
                shape = (batch_size,) + tuple(input_shape[1:])
            else:
                shape = (batch_size,) + tuple(input_shape[1:])
            inputs = tf.random.normal(shape, dtype=dtype)

        infer_fn = tf.function(model, autograph=False)
        _ = infer_fn(inputs, training=False)

        for _ in range(max(0, warmup_runs)):
            _ = infer_fn(inputs, training=False)

        timings = []
        for _ in range(max(1, measure_runs)):
            start = time.perf_counter()
            _ = infer_fn(inputs, training=False)
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

        batch_latency_ms = float(np.mean(timings))
        per_sample_ms = batch_latency_ms / max(1, inputs.shape[0])
        throughput = 1000.0 / per_sample_ms if per_sample_ms > 0 else float('inf')

        return {
            'batch_latency_ms': batch_latency_ms,
            'per_sample_latency_ms': per_sample_ms,
            'throughput_samples_per_sec': throughput,
            'batch_size': int(inputs.shape[0])
        }

    def estimate_accuracy(self,
                          model: tf.keras.Model,
                          dataset: tf.data.Dataset,
                          max_batches: Optional[int] = None) -> Dict[str, float]:
        """Estimate loss/accuracy on (optionally truncated) dataset."""
        if max_batches is not None and max_batches > 0:
            dataset = dataset.take(max_batches)

        opt = getattr(model, "optimizer", None) or tf.keras.optimizers.SGD()
        loss = getattr(model, "loss", None) or "categorical_crossentropy"
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["accuracy"],
        )

        results = model.evaluate(dataset, verbose=0)
        metrics_names = model.metrics_names or []

        if isinstance(results, (list, tuple)):
            metrics_map = {
                name: float(results[idx])
                for idx, name in enumerate(metrics_names[:len(results)])
            }
        else:
            key = metrics_names[0] if metrics_names else 'loss'
            metrics_map = {key: float(results)}

        return metrics_map
    
    def compare_models(self,
                      original_model: tf.keras.Model,
                      pruned_model: tf.keras.Model,
                      dataset: tf.data.Dataset,
                      dataset_name: str = "") -> Dict:
        """Compare original and pruned models"""
        print(f"\nComparing models on {dataset_name}...")
        
        # Evaluate both models
        original_results = self.evaluate_model(original_model, dataset, f"{dataset_name} (Original)")
        pruned_results = self.evaluate_model(pruned_model, dataset, f"{dataset_name} (Pruned)")
        
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
