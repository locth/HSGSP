import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import random

import numpy as np
import torch
import torch.nn as nn

from pruning.frequency_rel_analyzer import FrequencyRelevanceAnalyzer
from pruning.pruning_strategy import PruningStrategy, LayerPruningConfig
from models.model_utils import ModelUtils
from utils.logger import Logger
from training.frequency_regularizer import compute_spectral_entropy

class HybridFrequencyBaseline:
    """
    Hybrid frequency-saliency baseline that combines DCT-based energy analysis
    with gradient saliency for iterative channel pruning.
    """

    def __init__(self, config, trainer, evaluator):
        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator
        self.logger = Logger(config)
        self.pruning_util = PruningStrategy(config)
        self.frn_analyzer = FrequencyRelevanceAnalyzer(config)
        self.frequency_layer_names: List[str] = []
        self.entropy_targets: Dict[str, float] = {}
        mode = str(getattr(self.config, "hybrid_mode", "frequency")).lower()
        self.frequency_enabled = mode == "frequency"

    def run_pipeline(
        self,
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        train_eval_dl: Optional[torch.utils.data.DataLoader] = None,
        activation_dl: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tuple[torch.nn.Module, List[Dict]]:
        """
        Execute the iterative pruning baseline.

        Returns:
            pruned_model: Model after pruning iterations.
            iteration_history: metrics per iteration.
        """
        activation_dl = activation_dl or train_eval_dl or train_dl
        frn_model = self._train_frn(model, activation_dl)
        mode_label = "frequency-regularized" if self.frequency_enabled else "original"
        self.logger.info(f"Hybrid baseline mode: {mode_label}")
        self.frequency_layer_names = []
        self.entropy_targets = {}
        if self.frequency_enabled:
            self.frequency_layer_names = self._select_frequency_layers(model)
            if activation_dl is not None and self.frequency_layer_names:
                self.entropy_targets = self._estimate_entropy_targets(
                    model,
                    activation_dl,
                    self.frequency_layer_names,
                    batches=getattr(self.config, "frequency_entropy_target_batches", 8),
                )
                self.logger.info(
                    f"Captured spectral entropy targets for {len(self.entropy_targets)} layer(s)."
                )

        baseline_metrics = self.evaluator.evaluate_model(model, val_dl, "Hybrid Baseline (validation)")
        baseline_loss = float(baseline_metrics.get("loss", 0.0))
        baseline_acc = baseline_metrics.get("accuracy")
        baseline_acc_str = f"{baseline_acc:.4f}" if baseline_acc is not None else "nan"
        self.logger.info(
            f"Hybrid baseline: initial val_loss={baseline_loss:.4f}, "
            f"val_accuracy={baseline_acc_str}"
        )

        iteration = 0
        iteration_history: List[Dict] = []
        kappa_ratio = float(self.config.hybrid_initial_kappa_ratio)
        current_model = model

        target_refresh = int(getattr(self.config, "frequency_entropy_refresh_interval", 0))
        while iteration < self.config.hybrid_iterations:
            iteration += 1
            self.logger.info(f"Hybrid baseline iteration {iteration}/{self.config.hybrid_iterations}...")
            if self.frequency_enabled and target_refresh > 0 and (iteration - 1) % target_refresh == 0 and iteration > 1:
                self._refresh_entropy_targets(current_model, activation_dl)

            activation_stats = None
            use_activation = bool(getattr(self.config, "frn_use_activation_features", True))
            if frn_model is not None and activation_dl is not None and use_activation:
                activation_stats = self._compute_activation_statistics(
                    current_model,
                    activation_dl,
                    max_batches=int(getattr(self.config, "frn_activation_batches", 8)),
                )

            freq_scores = self._compute_frequency_scores(
                current_model,
                kappa_ratio,
                frn_model,
                activation_stats=activation_stats,
            )
            grad_scores = self._compute_gradient_saliency(current_model, train_eval_dl or train_dl)
            hybrid_scores = self._combine_scores(freq_scores, grad_scores)

            layer_pruning_ratios = {name: self.config.hybrid_prune_fraction for name in hybrid_scores}
            current_model, _ = self.pruning_util.prune_model_structured(current_model, layer_pruning_ratios, hybrid_scores)

            warmup_epochs = max(0, int(getattr(self.config, "hybrid_warmup_epochs", 0)))
            if warmup_epochs > 0:
                self.logger.info(f"Warm-up training for {warmup_epochs} epoch(s) before fine-tuning...")
                self.trainer.compile_model(
                    current_model,
                    learning_rate=self.config.hybrid_warmup_lr,
                )
                self.trainer.train_cifar(
                    current_model,
                    train_dataloader=train_dl,
                    val_dataloader=val_dl,
                    epochs=warmup_epochs,
                    train_eval_dataloader=train_eval_dl,
                )

            freq_reg_config = None
            teacher = None
            if self.frequency_enabled and iteration > 1:
                freq_reg_config = self._build_frequency_regularizer_config()
                teacher = model
            current_model, _ = self.trainer.fine_tune_cifar(
                current_model,
                train_dl,
                val_dl,
                epochs=self.config.hybrid_finetune_epochs,
                learning_rate=self.config.pruned_growth_lr,
                log_dir_suffix=f"hybrid_iter_{iteration}",
                train_eval_dataloader=train_eval_dl,
            )

            metrics = self.evaluator.evaluate_model(current_model, val_dl, "Hybrid Baseline (validation)")
            val_acc = metrics.get("accuracy")
            val_loss = float(metrics.get("loss", 0.0))
            param_stats = ModelUtils.count_parameters(current_model)
            flop_count = ModelUtils.compute_flops(current_model)
            iteration_history.append(
                {
                    "iteration": iteration,
                    "kappa_ratio": kappa_ratio,
                    "metrics": metrics,
                    "param_count": param_stats,
                    "flops": flop_count,
                }
            )
            val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "nan"
            self.logger.info(
                f"Iteration {iteration} summary -> val_acc={val_acc_str}, "
                f"params={param_stats['total']:,}, FLOPs={flop_count/1e6:.2f} MFLOPs"
            )

            delta_loss = max(0.0, val_loss - baseline_loss)
            kappa_ratio = max(
                0.05,
                kappa_ratio * (1.0 - self.config.hybrid_kappa_beta * delta_loss),
            )
            baseline_loss = val_loss

        return current_model, iteration_history

    def _select_frequency_layers(self, model: torch.nn.Module) -> List[str]:
        conv_layers = ModelUtils.get_conv_layers(model)
        max_layers = max(0, int(getattr(self.config, "frequency_regularization_layers", 0)))
        if not conv_layers or max_layers <= 0:
            return []
        return [name for name, _ in list(model.named_modules()) if isinstance(_, torch.nn.Conv2d)][-max_layers:]

    def _estimate_entropy_targets(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer_names: List[str],
        batches: int = 8,
        device: str = 'cpu',
    ) -> Dict[str, float]:
        if dataloader is None or not layer_names:
            return {}
        model.eval()
        stats: Dict[str, List[float]] = {name: [] for name in layer_names}
        hooks = []
        activations = {}

        def hook_fn(name):
            def hook(m, i, o):
                activations[name] = o.detach().cpu()
            return hook

        for name in layer_names:
            module = dict(model.named_modules())[name]
            hooks.append(module.register_forward_hook(hook_fn(name)))

        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            with torch.no_grad():
                _ = model(inputs)
            for name in layer_names:
                feat = activations.get(name)
                if feat is not None:
                    entropy = compute_spectral_entropy(feat)
                    stats[name].append(float(entropy))
            if batch_idx + 1 >= batches:
                break

        for h in hooks:
            h.remove()

        return {
            name: float(np.mean(values)) for name, values in stats.items() if values
        }

    def _refresh_entropy_targets(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader],
        batches: Optional[int] = None,
    ) -> None:
        if dataloader is not None and self.frequency_layer_names:
            batch_count = batches or getattr(self.config, "frequency_entropy_target_batches", 8)
            self.entropy_targets = self._estimate_entropy_targets(
                model,
                dataloader,
                self.frequency_layer_names,
                batches=batch_count,
            )
            self.logger.info(
                f"Refreshed spectral entropy targets for {len(self.entropy_targets)} layer(s)."
            )

    def _build_frequency_regularizer_config(self) -> Optional[Dict[str, object]]:
        if not self.frequency_layer_names or not self.entropy_targets:
            return None
        ordered_targets = [
            (name, self.entropy_targets.get(name)) for name in self.frequency_layer_names
        ]
        ordered_targets = [(n, t) for n, t in ordered_targets if t is not None]
        if not ordered_targets:
            return None
        beta = float(getattr(self.config, "frequency_entropy_beta", 0.05))
        layer_weights = getattr(self.config, "frequency_entropy_layer_weights", {}) or {}
        if not layer_weights:
            num_layers = len(ordered_targets)
            if num_layers:
                scales = np.linspace(1.0, 0.4, num_layers)
                layer_weights = {
                    name: float(scale) for (name, _), scale in zip(ordered_targets, scales)
                }
        targets = {name: target for name, target in ordered_targets}
        return {
            "layer_names": list(targets.keys()),
            "targets": targets,
            "beta": beta,
            "layer_weights": layer_weights,
        }

    def _frequency_balanced_regrow_init(self, model: torch.nn.Module, iteration: int) -> None:
        regrow_fraction = float(getattr(self.config, "hybrid_regrow_fraction", 0.0))
        if regrow_fraction <= 0.0:
            return
        if iteration <= 1:
            self.logger.debug("Regrow skipped on first pruning iteration to preserve baseline weights.")
            return
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                weights = layer.weight.data
                h, w, cin, cout = weights.shape
                if cout == 0:
                    continue
                regrow_count = max(1, int(round(cout * regrow_fraction)))
                regrow_count = min(regrow_count, cout)

                dct_kernel = self.frn_analyzer.dct2_ortho(weights)
                band_ratios = self.frn_analyzer.band_ratios(dct_kernel, self.config.frequency_bands)
                if band_ratios.size == 0:
                    continue

                band_names = list(self.config.frequency_bands.keys())
                band_energy = band_ratios.mean(dim=0)
                target_idx = int(torch.argmin(band_energy))
                band_name = band_names[target_idx]
                lo, hi = self.config.frequency_bands[band_name]

                norms = torch.norm(weights.view(-1, cout), dim=0)
                regrow_indices = torch.argsort(norms)[:regrow_count]
                mask = self.frn_analyzer.create_mask(h, w, lo, hi)
                mask_torch = torch.from_numpy(mask).reshape(h, w, 1, 1).to(weights.device).float()
                freq_noise = torch.normal(mean=0.0, std=0.05, size=(h, w, cin, regrow_count)).to(weights.device)
                freq_noise *= mask_torch
                spatial_kernels = self.frn_analyzer.idct2_ortho(freq_noise)

                new_kernel = weights.clone()
                for slot, filt_idx in enumerate(regrow_indices):
                    new_kernel[:, :, :, filt_idx] = spatial_kernels[:, :, :, slot]
                    if layer.bias is not None:
                        layer.bias.data[filt_idx] = 0.0

                layer.weight.data = new_kernel
                self.logger.debug(
                    f"Regrew {len(regrow_indices)} filter(s) in {name} targeting {band_name} band."
                )

    def _validate_and_recalibrate(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader],
    ) -> None:
        device = next(model.parameters()).device
        try:
            input_shape = (1, 3, 32, 32)
            dummy = torch.zeros(input_shape).to(device)
            model(dummy)
        except Exception as exc:
            self.logger.warning("Failed dummy forward after pruning: %s", exc)
        if dataloader is None:
            return
        steps = int(getattr(self.config, "bn_recalibrate_steps", 200))
        steps = max(1, steps)
        model.train()
        for step, (xb, _) in enumerate(dataloader):
            xb = xb.to(device)
            model(xb)
            if step + 1 >= steps:
                break

    def _train_frn(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader],
    ) -> Optional[torch.nn.Module]:
        if dataloader is None:
            self.logger.info("FRN training skipped: no dataset provided.")
            return None

        seed_value = getattr(self.config, "seed", None)
        if seed_value is None:
            seed_value = 42
            self.config.seed = seed_value
        torch.manual_seed(seed_value)

        plot_training = bool(getattr(self.config, "frn_plot_training", False))
        frn_epochs = int(getattr(self.config, "frn_epochs", 15))
        frn_lr = float(getattr(self.config, "frn_initial_lr", 1e-4))
        frn_min_lr = float(getattr(self.config, "frn_min_lr", 1e-5))
        frn_batch_size = int(getattr(self.config, "frn_batch_size", 256))
        frn_val_split = float(getattr(self.config, "frn_validation_split", 0.2))
        frn_min_val = int(getattr(self.config, "frn_min_validation_samples", 128))
        frn_low_vs_rest = bool(getattr(self.config, "frn_low_vs_rest", False))
        num_classes = 2 if frn_low_vs_rest else 3

        X_train, y_train = self.frn_analyzer.build_frn_training_set(model, dataloader)

        if X_train.shape[0] == 0:
            self.logger.warning("Empty FRN training set; skipping FRN training.")
            return None

        train_size = int(len(X_train) * (1 - frn_val_split))
        val_size = max(frn_min_val, len(X_train) - train_size)
        train_size = len(X_train) - val_size

        train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train[:train_size]), torch.from_numpy(y_train[:train_size]))
        val_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train[train_size:]), torch.from_numpy(y_train[train_size:]))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=frn_batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=frn_batch_size, shuffle=False)

        frn_model = self.frn_analyzer.build_frequency_relevance_net()

        optimizer = torch.optim.Adam(frn_model.parameters(), lr=frn_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=frn_epochs, eta_min=frn_min_lr)
        criterion = nn.KLDivLoss(reduction="batchmean") if frn_low_vs_rest else nn.CrossEntropyLoss()

        for epoch in range(frn_epochs):
            frn_model.train()
            for batch in train_loader:
                inputs, targets = batch
                outputs = frn_model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        return frn_model

    def _compute_activation_statistics(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 8,
    ) -> Optional[Dict[str, np.ndarray]]:
        model.eval()
        stats = {}
        hooks = []
        activations = {}

        def hook_fn(name):
            def hook(m, i, o):
                activations[name] = o.detach().cpu().numpy()
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(next(model.parameters()).device)
            with torch.no_grad():
                _ = model(inputs)
            for name in activations:
                feat = activations[name]
                mean = np.mean(feat)
                std = np.std(feat)
                skew = np.mean((feat - mean)**3) / (std**3 + 1e-8)
                if name not in stats:
                    stats[name] = np.array([mean, std, skew])
                else:
                    stats[name] = np.vstack((stats[name], [mean, std, skew]))
            if batch_idx + 1 >= max_batches:
                break

        for h in hooks:
            h.remove()

        for name in stats:
            stats[name] = np.mean(stats[name], axis=0)

        return stats

    def _compute_frequency_scores(
        self,
        model: torch.nn.Module,
        kappa_ratio: float,
        frn_model: Optional[torch.nn.Module],
        activation_stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        freq_scores = {}
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                kernel = layer.weight.data.detach().cpu().numpy()
                band_E, ratios, totals = self.frn_analyzer.band_energies_from_kernel(kernel)
                low = ratios['low']
                mid = ratios['mid']
                high = ratios['high']
                features = np.stack([low, mid, high], axis=1)
                if activation_stats is not None and name in activation_stats:
                    act_features = activation_stats[name]
                    features = np.concatenate([features, act_features[None, :].repeat(features.shape[0], axis=0)], axis=1)
                if frn_model is not None:
                    frn_model.eval()
                    with torch.no_grad():
                        inputs = torch.from_numpy(features).float()
                        outputs = frn_model(inputs)
                        scores = outputs[:, 0].numpy()  # low freq relevance
                else:
                    scores = low
                freq_scores[name] = scores
        return freq_scores

    def _compute_gradient_saliency(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, np.ndarray]:
        model.eval()
        saliency = {}
        device = next(model.parameters()).device
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d):
                    if layer.weight.grad is None:
                        continue
                    grad = layer.weight.grad.abs().mean(dim=[1,2,3]).cpu().numpy()
                    if name not in saliency:
                        saliency[name] = grad
                    else:
                        saliency[name] += grad
            model.zero_grad()
        for name in saliency:
            saliency[name] /= len(dataloader)
        return saliency

    def _combine_scores(self, freq_scores: Dict[str, np.ndarray], grad_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        hybrid = {}
        for name in freq_scores:
            f = freq_scores[name]
            g = grad_scores.get(name, np.ones_like(f))
            hybrid[name] = self.config.hybrid_alpha * f + (1 - self.config.hybrid_alpha) * g
        return hybrid

    def _select_pruning_masks(self, hybrid_scores: Dict[str, np.ndarray], iteration: int) -> Dict[str, np.ndarray]:
        masks = {}
        prune_fraction = self.config.hybrid_prune_fraction
        for name, scores in hybrid_scores.items():
            num_filters = len(scores)
            num_prune = int(num_filters * prune_fraction)
            threshold = np.sort(scores)[num_prune]
            mask = scores > threshold
            masks[name] = mask
        return masks

    def _apply_pruning(self, model: torch.nn.Module, masks: Dict[str, np.ndarray], hybrid_scores: Dict[str, np.ndarray]) -> torch.nn.Module:
        pruning_configs: Dict[str, LayerPruningConfig] = {}
        for name in masks:
            mask = masks[name]
            scores = hybrid_scores[name]
            num_filters = len(scores)
            filters_to_keep = int(np.sum(mask))
            pruning_ratio = (num_filters - filters_to_keep) / num_filters if num_filters > 0 else 0.0
            pruning_configs[name] = LayerPruningConfig(
                layer_name=name,
                original_filters=num_filters,
                filters_to_keep=filters_to_keep,
                pruning_ratio=pruning_ratio,
                importance_scores=scores,
                mask=mask
            )
        pruned_model = self.pruning_util.apply_structured_pruning(model, pruning_configs)
        return pruned_model