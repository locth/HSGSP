import torch
import numpy as np
from typing import List, Dict, Tuple

class ModelUtils:
    """Utilities for model manipulation and analysis"""
    
    @staticmethod
    def get_conv_layers(model: torch.nn.Module) -> List[torch.nn.Conv2d]:
        """Extract all Conv2D layers from model"""
        conv_layers = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)
        return conv_layers
    
    @staticmethod
    def get_layer_weights(layer: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """Get weights and biases from a layer"""
        weights = layer.weight.detach().cpu().numpy() if hasattr(layer, 'weight') else None
        bias = layer.bias.detach().cpu().numpy() if hasattr(layer, 'bias') and layer.bias is not None else None
        return weights, bias
    
    @staticmethod
    def set_layer_weights(layer: torch.nn.Module, 
                         weights: np.ndarray, 
                         bias: np.ndarray = None):
        """Set weights and biases for a layer"""
        if weights is not None:
            layer.weight.data = torch.from_numpy(weights).to(layer.weight.device)
        if bias is not None:
            layer.bias.data = torch.from_numpy(bias).to(layer.bias.device)
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """Count parameters in model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    @staticmethod
    def compute_flops(model: torch.nn.Module) -> int:
        """Estimate FLOPs for the model"""
        total_flops = 0
        input_shape = (1, 3, 32, 32)  # Assuming CIFAR input
        device = next(model.parameters()).device
        
        def hook_fn(module, input, output):
            nonlocal total_flops
            if isinstance(module, torch.nn.Conv2d):
                out_h, out_w = output.shape[2], output.shape[3]
                k_h, k_w = module.kernel_size
                c_in, c_out = input[0].shape[1], output.shape[1]
                flops = 2 * out_h * out_w * k_h * k_w * c_in * c_out
                if module.bias is not None:
                    flops += out_h * out_w * c_out
                total_flops += flops
            elif isinstance(module, torch.nn.Linear):
                in_dim = input[0].shape[1]
                out_dim = output.shape[1]
                flops = 2 * in_dim * out_dim
                if module.bias is not None:
                    flops += out_dim
                total_flops += flops

        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook_fn))
        
        dummy_input = torch.randn(input_shape, device=device)
        
        with torch.no_grad():
            model(dummy_input)
        
        for h in hooks:
            h.remove()
        
        return total_flops