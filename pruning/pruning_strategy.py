import copy
import numpy as np
import torch 
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LayerPruningConfig:
    """Configuration for pruning a specific layer"""
    layer_name: str
    original_filters: int
    filters_to_keep: int
    pruning_ratio: float
    importance_scores: np.ndarray
    mask: np.ndarray

class PruningStrategy:
    """
    Implements various pruning strategies for HSGSP
    """

    def __init__(self, config):
        self.config = config
        self.min_filters_per_layer = int(getattr(config, "hybrid_min_filters", 8))
        self.pruning_schedule = self._create_pruning_schedule()

    def _create_pruning_schedule(self) -> Dict[str, float]:
        """
        Create layer-wise pruning schedule
        Different layers may have different sensitivity to pruning
        """
        return {
            'early': 0.8,   # Keep 80% in early layers (more important)
            'middle': 0.6,  # Keep 60% in middle layers
            'late': 0.5,    # Keep 50% in late layers (can prune more)
        }

    def compute_layer_importance(self, 
                                layer_name: str,
                                layer_position: float) -> float:
        """
        Compute importance multiplier for a layer based on its position
        
        Args:
            layer_name: Name of the layer
            layer_position: Normalized position in network (0=first, 1=last)
            
        Returns:
            Importance multiplier (higher = more important)
        """
        if layer_position < 0.3:
            return self.pruning_schedule['early']
        elif layer_position < 0.7:
            return self.pruning_schedule['middle']
        else:
            return self.pruning_schedule['late']

    def select_filters_structured(self,
                                 importance_scores: Dict[str, np.ndarray],
                                 target_pruning_ratio: float,
                                 model: torch.nn.Module) -> Dict[str, LayerPruningConfig]:
        """
        Select filters to prune using structured pruning
        
        Args:
            importance_scores: Filter importance scores per layer
            target_pruning_ratio: Target pruning ratio
            model: Model to prune
            
        Returns:
            Pruning configuration for each layer
        """
        pruning_configs = {}
        total_filters = 0
        total_to_prune = 0
        
        # Get total layer count for position calculation
        conv_layers = [l for name, l in model.named_modules() 
                      if isinstance(l, torch.nn.Conv2d)]
        num_layers = len(conv_layers)
        
        for idx, (name, layer) in enumerate(model.named_modules()):
            if not isinstance(layer, torch.nn.Conv2d):
                continue
            
            scores = importance_scores.get(name)
            if scores is None:
                continue
            
            num_filters = len(scores)
            total_filters += num_filters
            
            # Compute layer-specific pruning ratio
            layer_position = idx / max(num_layers - 1, 1)
            layer_importance = self.compute_layer_importance(
                name, layer_position
            )
            
            # Adjust pruning ratio based on layer importance
            adjusted_ratio = target_pruning_ratio * (2 - layer_importance)
            adjusted_ratio = np.clip(adjusted_ratio, 0, 0.9)
            
            # Calculate filters to keep
            filters_to_keep = max(
                int(num_filters * (1 - adjusted_ratio)),
                self.min_filters_per_layer
            )
            filters_to_prune = num_filters - filters_to_keep
            total_to_prune += filters_to_prune
            
            # Create pruning mask
            if filters_to_prune > 0:
                # Sort filters by importance
                sorted_indices = np.argsort(scores)
                
                # Create mask (True = keep, False = prune)
                mask = np.zeros(num_filters, dtype=bool)
                keep_indices = sorted_indices[-filters_to_keep:]
                mask[keep_indices] = True
            else:
                mask = np.ones(num_filters, dtype=bool)
            
            config = LayerPruningConfig(
                layer_name=name,
                original_filters=num_filters,
                filters_to_keep=filters_to_keep,
                pruning_ratio=filters_to_prune / num_filters,
                importance_scores=scores,
                mask=mask
            )
            
            pruning_configs[name] = config
        
        # Log pruning statistics
        actual_pruning_ratio = total_to_prune / max(total_filters, 1)
        print(f"Structured Pruning: {total_to_prune}/{total_filters} filters "
              f"({actual_pruning_ratio:.1%})")
        
        return pruning_configs
    
    def select_filters_unstructured(self,
                                   importance_scores: Dict[str, np.ndarray],
                                   target_sparsity: float) -> Dict[str, np.ndarray]:
        """
        Select weights to prune using unstructured pruning (weight-level)
        
        Args:
            importance_scores: Weight importance scores
            target_sparsity: Target sparsity level
            
        Returns:
            Binary masks for each layer
        """
        masks = {}
        
        # Collect all scores
        all_scores = []
        layer_info = []
        
        for layer_name, scores in importance_scores.items():
            flat_scores = scores.flatten()
            all_scores.extend(flat_scores)
            layer_info.extend([(layer_name, i) for i in range(len(flat_scores))])
        
        # Compute global threshold
        all_scores = np.array(all_scores)
        threshold = np.percentile(all_scores, target_sparsity * 100)
        
        # Create masks based on threshold
        for layer_name, scores in importance_scores.items():
            mask = scores > threshold
            masks[layer_name] = mask
        
        return masks
    
    def apply_structured_pruning(self,
                                model: torch.nn.Module,
                                pruning_configs: Dict[str, LayerPruningConfig]) -> torch.nn.Module:
        """
        Apply structured pruning with channel-consistency across layers.

        - Prunes Conv2D output channels per configuration.
        - Propagates the kept-channel mask forward to slice the next Conv2D's
          input channels accordingly.
        - Adjusts following BatchNormalization parameters.
        - Adapts Dense input weights when preceded by AdaptiveAvgPool2d
          (common in this repo's VGG models).

        Args:
            model: Original model
            pruning_configs: Dict layer_name -> LayerPruningConfig (with mask)

        Returns:
            A new pruned model with compatible shapes and copied weights.
        """
        pruned_model = copy.deepcopy(model)
        current_mask = None

        modules = list(pruned_model.named_modules())
        for idx, (name, module) in enumerate(modules):
            if isinstance(module, nn.Conv2d):
                config = pruning_configs.get(name)
                if config is not None:
                    out_mask = config.mask
                    new_out_channels = int(np.sum(out_mask))
                else:
                    out_mask = None
                    new_out_channels = module.out_channels

                if current_mask is not None:
                    module.in_channels = int(np.sum(current_mask))
                    weight = module.weight.data
                    weight = weight[:, current_mask, :, :]
                    module.weight.data = weight

                if out_mask is not None:
                    module.out_channels = new_out_channels
                    weight = module.weight.data
                    weight = weight[out_mask, :, :, :]
                    module.weight.data = weight
                    if module.bias is not None:
                        bias = module.bias.data
                        bias = bias[out_mask]
                        module.bias.data = bias
                    current_mask = out_mask
                else:
                    current_mask = np.ones(module.out_channels, dtype=bool)
            elif isinstance(module, nn.BatchNorm2d):
                if current_mask is not None:
                    module.num_features = int(np.sum(current_mask))
                    if module.running_mean is not None:
                        module.running_mean = module.running_mean[current_mask]
                    if module.running_var is not None:
                        module.running_var = module.running_var[current_mask]
                    if module.weight is not None:
                        module.weight.data = module.weight.data[current_mask]
                    if module.bias is not None:
                        module.bias.data = module.bias.data[current_mask]
            elif isinstance(module, nn.Linear):
                if current_mask is not None:
                    module.in_features = int(np.sum(current_mask))
                    weight = module.weight.data
                    weight = weight[:, current_mask]
                    module.weight.data = weight
                    current_mask = None

        return pruned_model

    def prune_model_structured(self,
                               model: torch.nn.Module,
                               layer_pruning_ratios: Dict[str, float],
                               importance_scores: Dict[str, np.ndarray]) -> Tuple[torch.nn.Module, Dict]:
        """
        Convenience API: build per-layer configs from provided ratios and scores,
        then apply structured pruning consistently.

        Args:
            model: model to prune
            layer_pruning_ratios: dict layer_name -> ratio in [0,1]
            importance_scores: dict layer_name -> per-filter importance scores

        Returns:
            (pruned_model, pruning_configs)
        """
        pruning_configs: Dict[str, LayerPruningConfig] = {}
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                if name not in importance_scores:
                    continue
                scores = importance_scores[name]
                num_filters = len(scores)
                ratio = layer_pruning_ratios.get(name, 0.0)
                ratio = float(np.clip(ratio, 0.0, 0.95))
                keep = max(1, int(round(num_filters * (1.0 - ratio))))
                idx_sorted = np.argsort(scores)
                keep_idx = idx_sorted[-keep:]
                mask = np.zeros(num_filters, dtype=bool)
                mask[keep_idx] = True
                pruning_configs[name] = LayerPruningConfig(
                    layer_name=name,
                    original_filters=num_filters,
                    filters_to_keep=keep,
                    pruning_ratio=(num_filters - keep) / max(num_filters, 1),
                    importance_scores=scores,
                    mask=mask,
                )

        pruned_model = self.apply_structured_pruning(model, pruning_configs)
        return pruned_model, pruning_configs

    def compute_pruning_sensitivity(self,
                                   model: torch.nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   max_batches: int = 5) -> Dict[str, float]:
        """Estimate per-layer pruning sensitivity by measuring accuracy drop on small pruning."""
        sensitivities = {}
        baseline = self.evaluator.estimate_accuracy(model, dataloader)
        base_acc = baseline.get('accuracy', 0.0)
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                temp_model = copy.deepcopy(model)
                temp_layer = dict(temp_model.named_modules())[name]
                num_filters = temp_layer.out_channels
                prune_count = max(1, int(num_filters * 0.1))
                scores = np.random.rand(num_filters)
                idx_sorted = np.argsort(scores)
                keep_idx = idx_sorted[prune_count:]
                mask = np.zeros(num_filters, dtype=bool)
                mask[keep_idx] = True
                temp_configs = {name: LayerPruningConfig(
                    name, num_filters, num_filters - prune_count, 0.1, scores, mask
                )}
                temp_pruned = self.apply_structured_pruning(temp_model, temp_configs)
                pruned_acc = self.evaluator.estimate_accuracy(temp_pruned, dataloader).get('accuracy', 0.0)
                drop = base_acc - pruned_acc
                sensitivities[name] = drop
        return sensitivities