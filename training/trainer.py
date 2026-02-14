import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Callable, List, Tuple, Union
import os
from datetime import datetime
import json
import gc
import shutil

from config import Config
from utils.logger import Logger
from utils.overfitting_monitor import OverfittingMonitor, AdaptiveRegularization
from training.distillation import Distiller
from training.frequency_regularizer import (
    FrequencyRegularizedModel,
    SpectralEntropyRegularizer,
)


class SplitTensorBoard(tf.keras.callbacks.Callback):
    """Write train/val scalars into separate TensorBoard runs."""

    def __init__(self, train_dir: str, val_dir: str):
        super().__init__()
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        self._train_writer = tf.summary.create_file_writer(train_dir)
        self._val_writer = tf.summary.create_file_writer(val_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self._train_writer.as_default():
            for key, value in logs.items():
                if key.startswith('val_') or value is None:
                    continue
                tf.summary.scalar(key, float(value), step=epoch)
        with self._val_writer.as_default():
            for key, value in logs.items():
                if not key.startswith('val_') or value is None:
                    continue
                tag = key[len('val_'):]
                tf.summary.scalar(tag, float(value), step=epoch)
        self._train_writer.flush()
        self._val_writer.flush()

    def on_train_end(self, logs=None):
        self._train_writer.close()
        self._val_writer.close()

class HSGSPTrainer:
    """Training manager for HSGSP pruning"""
    
    def __init__(self, config):
        self.config = config
        self.logger = Logger(config)

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
    def _model_outputs_logits(model: Optional[tf.keras.Model]) -> bool:
        if model is None or not hasattr(model, 'layers') or not model.layers:
            return False
        last_layer = model.layers[-1]
        if isinstance(last_layer, tf.keras.layers.Softmax):
            return False
        activation = getattr(last_layer, 'activation', None)
        if activation is None:
            return True
        return activation != tf.keras.activations.softmax

    def _create_lr_schedule(self, epochs: int) -> Callable:
        """
        Create learning rate schedule with optional warmup.
        """
        schedule = (self.config.lr_schedule or 'cosine').lower()
        warmup_epochs = max(0, int(getattr(self.config, 'lr_warmup_epochs', 0)))
        total_epochs = max(1, epochs)
        initial_lr = float(self.config.initial_lr)
        min_lr_value = float(getattr(self.config, 'min_lr', 1e-7))

        def _warmup(epoch: int) -> float:
            if warmup_epochs <= 0:
                return initial_lr
            ratio = (epoch + 1) / float(warmup_epochs)
            return float(initial_lr * ratio)

        if schedule == 'cosine':
            min_factor = float(getattr(self.config, 'fine_tune_cosine_min_factor', 0.1))
            target_min_lr = max(min_lr_value, initial_lr * min_factor)
            self.logger.info(
                f"Training LR scheduler: cosine annealing (min_lr={target_min_lr:.2e}, warmup={warmup_epochs})"
            )

            def cosine_schedule(epoch: int, current_lr: float) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return _warmup(epoch)
                progress = (epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
                progress = np.clip(progress, 0.0, 1.0)
                cosine = 0.5 * (1 + np.cos(np.pi * progress))
                new_lr = target_min_lr + (initial_lr - target_min_lr) * cosine
                return float(max(min_lr_value, new_lr))

            return cosine_schedule

        if schedule == 'exponential':
            decay_rate = float(self.config.lr_decay_rate)
            self.logger.info(
                f"Training LR scheduler: exponential decay (rate={decay_rate:.4f}, warmup={warmup_epochs})"
            )

            def exp_schedule(epoch: int, current_lr: float) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return _warmup(epoch)
                effective_epoch = epoch - warmup_epochs
                new_lr = initial_lr * (decay_rate ** effective_epoch)
                return float(max(min_lr_value, new_lr))

            return exp_schedule

        if schedule == 'step':
            decay_steps = max(1, int(self.config.lr_decay_steps))
            decay_rate = float(self.config.lr_decay_rate)
            self.logger.info(
                f"Training LR scheduler: step decay (steps={decay_steps}, rate={decay_rate:.3f}, warmup={warmup_epochs})"
            )

            def step_schedule(epoch: int, current_lr: float) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return _warmup(epoch)
                effective_epoch = epoch - warmup_epochs
                drops = max(0, effective_epoch // decay_steps)
                new_lr = initial_lr * (decay_rate ** drops)
                return float(max(min_lr_value, new_lr))

            return step_schedule

        return lambda epoch, current_lr=None: initial_lr
        
    def compile_model(self,
                     model: tf.keras.Model,
                     learning_rate: Optional[float] = None,
                     optimizer: Optional[str] = None) -> tf.keras.Model:
        """
        Compile the model with specified optimizer and loss
        
        Args:
            model: Model to compile
            learning_rate: Learning rate (uses config if None)
            optimizer: Optimizer name (uses config if None)
            
        Returns:
            Compiled model
        """
        lr = learning_rate or self.config.initial_lr
        opt_name = optimizer or self.config.optimizer
        
        # Create optimizer
        if opt_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )
        elif opt_name.lower() == 'adamw':
            # Prefer native Keras AdamW; fallback to Adam if unavailable
            AdamW = tf.keras.optimizers.AdamW
            optimizer = AdamW(
                learning_rate=lr,
                weight_decay=self.config.weight_decay,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )   
        elif opt_name.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=self.config.momentum,
                nesterov=True
            )
        elif opt_name.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=lr,
                rho=0.9,
                epsilon=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        # Create loss function with label smoothing
        if self.config.label_smoothing > 0:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                label_smoothing=self.config.label_smoothing
            )
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc'),
                # tf.keras.metrics.Precision(name='precision'),
                # tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        self.logger.info(f"Model compiled with {opt_name} optimizer, lr={lr}, label_smoothing={self.config.label_smoothing}")
        
        return model

    def train_cifar(self,
                      model: tf.keras.Model,
                      train_dataset: tf.data.Dataset,
                      val_dataset: tf.data.Dataset,
                      epochs: Optional[int] = None,
                      train_eval_dataset: Optional[tf.data.Dataset] = None) -> Dict:
        """
        Train the model with comprehensive anti-overfitting strategies
        
        Args:
            model: Model to train
            train_dataset: Training dataset (should be augmented)
            val_dataset: Validation dataset
            epochs: Number of epochs (uses config if None)
            
        Returns:
            Training history
        """

        epochs = epochs or self.config.default_epochs
        
        # Initialize overfitting monitoring
        overfitting_monitor = OverfittingMonitor(
            patience=5,
            threshold=0.05
        )

        adaptive_reg = AdaptiveRegularization(
            initial_dropout=self.config.dropout_rate,
            max_dropout=min(0.7, self.config.dropout_rate * 2)
        )

        # Callbacks
        callbacks = []

        # Early stopping with restore best weights
        if self.config.early_stopping_patience and self.config.early_stopping_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    mode='max',
                    verbose=1,
                    min_delta=self.config.early_stopping_min_delta
                )
            )

        # Reduce learning rate on plateau
        if self.config.reduce_lr_patience and self.config.reduce_lr_patience > 0:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                    min_delta=self.config.reduce_lr_min_delta,
                    min_lr=self.config.min_lr,
                    mode='auto',
                    verbose=1
                )
            )

        # Model checkpoint
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.models_dir,
                                     'best_model_{epoch:02d}_{val_accuracy:.3f}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        )

        # TensorBoard
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config.tensorboard_dir, datetime.now().strftime("%d%m%Y-%H%M%S")),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )

        if self.config.lr_schedule is not None:
            lr_schedule_fn = self._create_lr_schedule(epochs)
            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(
                    lr_schedule_fn,
                    verbose=1
                )
            )

        # Overfitting monitoring callback (disabled by default but left for reference)
        # callbacks.append(OverfittingCallback(overfitting_monitor, adaptive_reg, self.logger))

        if train_eval_dataset is not None:
            class CleanTrainEvaluation(tf.keras.callbacks.Callback):
                def __init__(self, dataset, logger):
                    super().__init__()
                    self._dataset = dataset
                    self._logger = logger

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    results = self.model.evaluate(self._dataset, verbose=0, return_dict=True)
                    if not isinstance(results, dict):
                        metric_names = self.model.metrics_names
                        results = {name: value for name, value in zip(metric_names, results)}
                    for name, value in results.items():
                        logs[f'clean_{name}'] = float(value)
                    acc = results.get('accuracy')
                    loss = results.get('loss')
                    if acc is not None and loss is not None:
                        self._logger.info(
                            f"Epoch {epoch + 1}: clean_train_loss={loss:.4f}, clean_train_accuracy={acc:.4f}"
                        )

            clean_callback = CleanTrainEvaluation(train_eval_dataset, self.logger)
            tensorboard_indices = [i for i, cb in enumerate(callbacks) if isinstance(cb, tf.keras.callbacks.TensorBoard)]
            if tensorboard_indices:
                callbacks.insert(tensorboard_indices[0], clean_callback)
            else:
                callbacks.append(clean_callback)
        
        # Train model
        self.logger.info("Starting training with anti-overfitting strategies...")
        self.logger.info(f"Label smoothing: {self.config.label_smoothing}")
        self.logger.info(f"Dropout rate: {self.config.dropout_rate}")
        self.logger.info(f"L2 regularization: {self.config.l2_regularization}")

        model = self.compile_model(model, learning_rate=self.config.initial_lr)

        # Add gradient clipping to optimizer
        if hasattr(model.optimizer, 'clipnorm'):
            model.optimizer.clipnorm = 1.0

        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Update training history
        self._update_history(history.history)
        
        # Save final model and history
        self._save_model(model, 'final_model.h5')
        self._save_history()
        
        # Plot overfitting analysis
        overfitting_monitor.plot_overfitting_analysis(
            save_path=os.path.join(self.config.plots_dir, 'overfitting_analysis.png')
        )
        
        # Log final results
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        overfitting_score = overfitting_monitor.get_overfitting_score()

        self.logger.info(f"Training completed:")
        self.logger.info(f"  Final training accuracy: {final_train_acc:.4f}")
        self.logger.info(f"  Final validation accuracy: {final_val_acc:.4f}")
        self.logger.info(f"  Final overfitting score: {overfitting_score:.3f}")

        if train_eval_dataset is not None:
            clean_results = model.evaluate(train_eval_dataset, verbose=0, return_dict=True)
            if not isinstance(clean_results, dict):
                clean_results = {
                    name: value for name, value in zip(model.metrics_names, clean_results)
                }
            clean_acc = clean_results.get('accuracy')
            clean_loss = clean_results.get('loss')
            if clean_acc is not None and clean_loss is not None:
                self.logger.info(
                    f"  Clean training (no augmentation) accuracy: {clean_acc:.4f}, loss: {clean_loss:.4f}"
                )
            history.history.setdefault('clean_train_accuracy', []).append(clean_acc)
            history.history.setdefault('clean_train_loss', []).append(clean_loss)
        
        if overfitting_score > 0.5:
            self.logger.warning("Model shows signs of overfitting. Consider:")
            self.logger.warning("  - Increasing data augmentation")
            self.logger.warning("  - Increasing dropout/regularization")
            self.logger.warning("  - Reducing model capacity")
            self.logger.warning("  - Gathering more training data")
        
        return self.history
    
    def _update_history(self, new_history: Dict):
        """Update training history"""
        for key in new_history:
            if key in self.history:
                self.history[key].extend(new_history[key])
            else:
                self.history[key] = list(new_history[key])

    def _save_model(self, model: tf.keras.Model, filename: str):
        """Save model to file"""
        filepath = os.path.join(self.config.models_dir, filename)
        model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")

    def _save_history(self):
        """Save training history to JSON"""
        filepath = os.path.join(self.config.results_dir, 'training_history.json')
        
        # Create a simple serializable dictionary
        save_dict = {}
        for key, values in self.history.items():
            save_dict[key] = [float(v) for v in values]
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=4)
        self.logger.info(f"Training history saved to {filepath}")

    def load_model(self, filepath: str) -> tf.keras.Model:
        """Load model from file"""
        model = tf.keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        return model
    
    def fine_tune_cifar(self,
                        model: tf.keras.Model,
                        train_dataset: tf.data.Dataset,
                        val_dataset: tf.data.Dataset,
                        epochs: int = 10,
                        learning_rate: Optional[float] = None,
                        log_dir_suffix: Optional[str] = None,
                        stage_log_root: Optional[str] = None,
                        stage_index: Optional[int] = None,
                        train_eval_dataset: Optional[tf.data.Dataset] = None,
                        teacher_model: Optional[tf.keras.Model] = None,
                        kd_alpha: Optional[float] = None,
                        kd_temperature: Optional[float] = None,
                        lr_schedule: Optional[str] = None,
                        frequency_regularizer_config: Optional[Dict[str, object]] = None) -> Tuple[tf.keras.Model, Dict[str, object]]:
        """
        Fine-tune model on a single dataset after pruning

        Args:
            model: Pruned model to fine-tune
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of fine-tuning epochs
            learning_rate: Optional override for the fine-tune learning rate
            log_dir_suffix: Optional name appended to tensorboard log dir
            lr_schedule: Override fine-tune LR scheduler ('plateau' or 'exponential')
            frequency_regularizer_config: Optional dict with keys `layer_names`,
                `targets`, and `beta` to enable spectral-entropy loss.
            
        Returns:
            Tuple of (fine-tuned model, metadata)
        """
        self.logger.info("Starting fine-tuning on single dataset...")
        
        # Setup optimizer with lower learning rate for fine-tuning
        lr = learning_rate if learning_rate is not None else self.config.pruned_growth_lr
        self.logger.info(f"Fine-tune learning rate set to {lr:.2e}")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=self.config.weight_decay
        )
        # Ensure callbacks can access `optimizer.lr`
        if not hasattr(optimizer, 'lr') and hasattr(optimizer, 'learning_rate'):
            try:
                optimizer.lr = optimizer.learning_rate
            except Exception:
                pass
        
        recalibrate_steps = int(getattr(self.config, 'bn_recalibrate_steps', 200))
        recalibrate_steps = max(1, recalibrate_steps)
        first_inputs = None
        for step, (batch_x, _) in enumerate(train_dataset.take(recalibrate_steps)):
            model(batch_x, training=True)
            if first_inputs is None:
                first_inputs = batch_x

        # Decide metrics once for both standard and KD
        metrics = ['accuracy']

        # Compile either plain student or KD distiller
        use_kd = teacher_model is not None
        freq_model = None
        use_freq_reg = False
        if frequency_regularizer_config:
            layer_names = frequency_regularizer_config.get('layer_names') or []
            targets = frequency_regularizer_config.get('targets') or {}
            beta = float(frequency_regularizer_config.get('beta', 0.01))
            if layer_names and targets:
                spectral_regularizer = SpectralEntropyRegularizer(
                    model=model,
                    layer_names=layer_names,
                    target_entropies=targets,
                    beta=beta,
                    layer_weights=frequency_regularizer_config.get('layer_weights'),
                )
                freq_model = FrequencyRegularizedModel(model, spectral_regularizer)
                use_freq_reg = True
                use_kd = False
                self.logger.info(
                    f"Frequency regularization enabled on {len(layer_names)} layer(s) "
                    f"(beta={beta:.3f})."
                )
        distiller = None
        if use_kd:
            alpha = kd_alpha if kd_alpha is not None else self.config.distill_alpha
            temperature = kd_temperature if kd_temperature is not None else self.config.distill_temperature
            self.logger.info(
                f"Knowledge distillation enabled (alpha={alpha:.3f}, temperature={temperature:.2f})"
            )
            distiller = Distiller(
                student=model,
                teacher=teacher_model,
                alpha=alpha,
                temperature=temperature,
            )
            student_loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            distill_loss_fn = tf.keras.losses.KLDivergence()
            distiller.compile(
                optimizer=optimizer,
                metrics=metrics,
                student_loss_fn=student_loss_fn,
                distillation_loss_fn=distill_loss_fn,
                student_from_logits=self._model_outputs_logits(model),
                teacher_from_logits=self._model_outputs_logits(teacher_model),
                alpha=alpha,
                temperature=temperature,
            )
            built_distiller = False
            if first_inputs is not None:
                try:
                    _ = distiller(first_inputs, training=False)
                    built_distiller = True
                except Exception as exc:
                    self.logger.warning(
                        f"Failed to prime distiller with calibration batch: {exc}"
                    )
            if not built_distiller:
                input_shape = getattr(model, 'input_shape', None)
                if input_shape is not None:
                    try:
                        distiller.build(input_shape)
                        built_distiller = True
                    except Exception as exc:
                        self.logger.warning(
                            f"Could not build distiller from input shape {input_shape}: {exc}"
                        )
        else:
            compile_target = freq_model if use_freq_reg else model
            compile_target.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=metrics
            )
        
        if stage_index is not None:
            run_identifier = f"stage{int(stage_index):02d}"
            root = stage_log_root or os.path.join(self.config.tensorboard_dir, "regrow_runs", "stages_shared")
            os.makedirs(root, exist_ok=True)
            train_log_dir = os.path.join(root, f"stage{stage_index}-train")
            val_log_dir = os.path.join(root, f"stage{stage_index}-validation")
        else:
            run_stamp = datetime.now().strftime("%d%m%Y-%H%M%S")
            suffix = (log_dir_suffix or "fine_tune").replace(" ", "_")
            run_identifier = f"{run_stamp}_{suffix}"
            root = os.path.join(
                self.config.tensorboard_dir,
                "regrow_runs",
                f"{run_stamp}_{suffix}"
            )
            train_log_dir = os.path.join(root, "train")
            val_log_dir = os.path.join(root, "validation")

        os.makedirs(train_log_dir, exist_ok=True)
        os.makedirs(val_log_dir, exist_ok=True)

        checkpoint_dir = os.path.join(self.config.models_dir, "fine_tune_checkpoints", run_identifier)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{run_identifier}_best.weights.h5")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                mode='max'
            )
        ]

        lr_schedule_mode = (lr_schedule or getattr(self.config, 'fine_tune_lr_schedule', 'plateau')).lower()
        if lr_schedule_mode == 'exponential':
            decay_rate = float(getattr(self.config, 'fine_tune_exp_decay', 0.96))
            self.logger.info(
                f"Fine-tune LR scheduler: exponential decay (rate={decay_rate:.4f})"
            )

            def exp_decay(epoch: int, current_lr: float) -> float:
                new_lr = lr * (decay_rate ** epoch)
                return float(max(self.config.min_lr, new_lr))

            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(exp_decay, verbose=1)
            )
        elif lr_schedule_mode == 'cosine':
            min_factor = float(getattr(self.config, 'fine_tune_cosine_min_factor', 0.1))
            target_min_lr = max(self.config.min_lr, lr * min_factor)
            self.logger.info(
                f"Fine-tune LR scheduler: cosine annealing (min_lr={target_min_lr:.2e})"
            )

            def cosine_decay(epoch: int, current_lr: float) -> float:
                if epochs <= 1:
                    return float(lr)
                progress = min(epoch, epochs - 1) / float(max(epochs - 1, 1))
                cosine = 0.5 * (1 + np.cos(np.pi * progress))
                new_lr = target_min_lr + (lr - target_min_lr) * cosine
                return float(max(self.config.min_lr, new_lr))

            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(cosine_decay, verbose=1)
            )
        elif lr_schedule_mode == 'linear':
            end_factor = float(getattr(self.config, 'fine_tune_linear_end_factor', 0.1))
            target_end_lr = max(self.config.min_lr, lr * end_factor)
            self.logger.info(
                f"Fine-tune LR scheduler: linear decay (end_lr={target_end_lr:.2e})"
            )

            def linear_decay(epoch: int, current_lr: float) -> float:
                if epochs <= 1:
                    return float(target_end_lr)
                progress = min(epoch, epochs - 1) / float(max(epochs - 1, 1))
                new_lr = lr + (target_end_lr - lr) * progress
                return float(max(self.config.min_lr, new_lr))

            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(linear_decay, verbose=1)
            )
        elif lr_schedule_mode == 'step':
            step_rate = float(getattr(self.config, 'fine_tune_step_decay_rate', 0.5))
            step_epochs = int(getattr(self.config, 'fine_tune_step_decay_epochs', 10))
            step_epochs = max(1, step_epochs)
            self.logger.info(
                f"Fine-tune LR scheduler: step decay (factor={step_rate:.3f} every {step_epochs} epochs)"
            )

            def step_decay(epoch: int, current_lr: float) -> float:
                drops = epoch // step_epochs
                new_lr = lr * (step_rate ** drops)
                return float(max(self.config.min_lr, new_lr))

            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
            )
        else:
            if lr_schedule_mode != 'plateau':
                self.logger.warning(
                    f"Unknown fine-tune LR scheduler '{lr_schedule_mode}', falling back to ReduceLROnPlateau"
                )
            self.logger.info("Fine-tune LR scheduler: ReduceLROnPlateau")
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=self.config.reduce_lr_factor,
                    patience=max(3, self.config.reduce_lr_patience // 2),
                    min_lr=self.config.min_lr,
                    min_delta=self.config.reduce_lr_min_delta,
                    cooldown=2,
                    verbose=1
                )
            )

        callbacks.extend([
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            SplitTensorBoard(train_log_dir, val_log_dir)
        ])

        # Fine-tune
        if use_kd:
            history = distiller.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            # Ensure returned student can be evaluated standalone
            try:
                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0),
                    loss='categorical_crossentropy',
                    metrics=metrics
                )
            except Exception:
                pass
        else:
            train_target = freq_model if use_freq_reg else model
            history = train_target.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            if use_freq_reg:
                try:
                    model.compile(
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0),
                        loss='categorical_crossentropy',
                        metrics=metrics
                    )
                except Exception as exc:
                    self.logger.warning(
                        f"Failed to compile base model after frequency-regularized training: {exc}"
                    )
                try:
                    del freq_model
                except Exception:
                    pass
                gc.collect()

        checkpoint_loaded = False
        target_for_checkpoint = distiller if use_kd else model
        if os.path.exists(checkpoint_path):
            try:
                target_for_checkpoint.load_weights(checkpoint_path)
                checkpoint_loaded = True
                self.logger.info(f"Loaded best fine-tune checkpoint from {checkpoint_path}")
            except Exception as exc:
                self.logger.warning(f"Failed to load fine-tune checkpoint {checkpoint_path}: {exc}")

        best_model_path = None
        best_model_filename = f"{run_identifier}_best_model.h5"
        try:
            self._save_model(model, best_model_filename)
            best_model_path = os.path.join(self.config.models_dir, best_model_filename)
        except Exception as exc:
            self.logger.warning(
                f"Failed to save fine-tuned model with architecture ({best_model_filename}): {exc}"
            )

        self.logger.info(f"Fine-tuning completed. "
                        f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
        if stage_index is not None:
            self.logger.info(f"TensorBoard logs written to {train_log_dir} and {val_log_dir}")
        else:
            self.logger.info(f"TensorBoard logs written to {train_log_dir} and {val_log_dir}")

        clean_metrics = None
        if train_eval_dataset is not None:
            if use_kd:
                eval_model = distiller
            else:
                eval_model = model
            clean_metrics = eval_model.evaluate(train_eval_dataset, verbose=0, return_dict=True)
            if not isinstance(clean_metrics, dict):
                clean_metrics = {
                    name: value for name, value in zip(eval_model.metrics_names, clean_metrics)
                }
            if clean_metrics:
                clean_acc = clean_metrics.get('accuracy')
                clean_loss = clean_metrics.get('loss')
                if clean_acc is not None and clean_loss is not None:
                    self.logger.info(
                        f"Clean training-set evaluation after fine-tune -> loss: {clean_loss:.4f}, accuracy: {clean_acc:.4f}"
                    )

        # Best val metrics snapshot
        best_val_metrics = {}
        try:
            val_acc_hist = history.history.get('val_accuracy')
            if val_acc_hist:
                best_idx = int(np.argmax(val_acc_hist))
                best_val_metrics['val_accuracy'] = float(val_acc_hist[best_idx])
                if 'val_loss' in history.history:
                    best_val_metrics['val_loss'] = float(history.history['val_loss'][best_idx])
                if 'val_top5_acc' in history.history:
                    best_val_metrics['val_top5_accuracy'] = float(history.history['val_top5_acc'][best_idx])
        except Exception:
            pass

        if os.path.isdir(checkpoint_dir):
            try:
                shutil.rmtree(checkpoint_dir)
            except Exception as exc:
                self.logger.warning(f"Failed to remove checkpoint directory {checkpoint_dir}: {exc}")

        metadata = {
            'history': history.history,
            'epochs_ran': len(history.history.get('loss', [])),
            'log_dirs': {
                'train': train_log_dir,
                'validation': val_log_dir
            },
            'clean_train_metrics': clean_metrics,
            'best_val_metrics': best_val_metrics if best_val_metrics else None,
            'checkpoint_path': checkpoint_path if checkpoint_loaded else (checkpoint_path if os.path.exists(checkpoint_path) else None),
            'best_model_path': best_model_path,
        }

        if use_kd:
            student = distiller.student
            del distiller
            gc.collect()
            return student, metadata
        return model, metadata

    def simple_finetune(
        self,
        model_or_path: Union[str, tf.keras.Model],
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        teacher_model: Optional[tf.keras.Model] = None,
        train_eval_dataset: Optional[tf.data.Dataset] = None,
        log_dir_suffix: Optional[str] = None,
        frequency_regularizer_config: Optional[Dict[str, object]] = None,
        kd_alpha: Optional[float] = None,
        kd_temperature: Optional[float] = None,
    ) -> Tuple[tf.keras.Model, Dict[str, object]]:
        """Load a checkpoint (or reuse an in-memory model) and run extra fine-tuning.

        Args:
            model_or_path: Either the model instance to continue training or the
                filesystem path to a saved Keras model.
            train_dataset: Dataset used for fine-tuning.
            val_dataset: Validation dataset.
            epochs: Optional override for the number of epochs (defaults to
                ``config.hybrid_finetune_epochs``).
            learning_rate: Optional override for the fine-tune learning rate
                (defaults to ``config.pruned_growth_lr``).
            use_kd: When ``True`` run fine-tuning with knowledge distillation.
            teacher_path: Optional path to a teacher checkpoint (used if KD enabled).
            teacher_model: Optional in-memory teacher model.
            train_eval_dataset: Optional clean evaluation dataset.
            log_dir_suffix: Optional suffix for TensorBoard logging directories.
            frequency_regularizer_config: Optional spectral-entropy regularizer
                configuration dict (same structure as ``fine_tune_cifar``).
            kd_alpha: Optional KD alpha override.
            kd_temperature: Optional KD temperature override.

        Returns:
            Tuple of (fine-tuned model, metadata dictionary) as produced by
            :meth:`fine_tune_cifar`.
        """

        if isinstance(model_or_path, (str, bytes, os.PathLike)):
            model = self.load_model(str(model_or_path))
        else:
            model = model_or_path

        teacher = None
        if teacher_model is not None:
            teacher = teacher_model

        tuned_model, metadata = self.fine_tune_cifar(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs or self.config.hybrid_finetune_epochs,
            learning_rate=learning_rate or self.config.pruned_growth_lr,
            log_dir_suffix=log_dir_suffix or "manual_finetune",
            train_eval_dataset=train_eval_dataset,
            teacher_model=teacher if bool(teacher) else None,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
            frequency_regularizer_config=frequency_regularizer_config,
        )

        return tuned_model, metadata
