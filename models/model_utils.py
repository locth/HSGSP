# models/model_utils.py
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple

class ModelUtils:
    """Utilities for model manipulation and analysis"""
    
    @staticmethod
    def get_conv_layers(model: tf.keras.Model) -> List[tf.keras.layers.Conv2D]:
        """Extract all Conv2D layers from model"""
        conv_layers = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer)
        return conv_layers
    
    @staticmethod
    def get_layer_weights(layer: tf.keras.layers.Layer) -> Tuple[np.ndarray, np.ndarray]:
        """Get weights and biases from a layer"""
        weights = layer.get_weights()
        if len(weights) == 2:
            return weights[0], weights[1]  # weights, bias
        elif len(weights) == 1:
            return weights[0], None
        else:
            return None, None
    
    @staticmethod
    def set_layer_weights(layer: tf.keras.layers.Layer, 
                         weights: np.ndarray, 
                         bias: np.ndarray = None):
        """Set weights and biases for a layer"""
        if bias is not None:
            layer.set_weights([weights, bias])
        else:
            layer.set_weights([weights])
    
    @staticmethod
    def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
        """Count parameters in model"""
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    @staticmethod
    def compute_flops(model: tf.keras.Model) -> int:
        """Estimate FLOPs for the model"""
        total_flops = 0
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                # FLOPs ≈ 2 * H_out * W_out * K_h * K_w * C_in * C_out
                # Avoid relying on layer.output_shape which may be absent on some TF versions
                try:
                    out_h, out_w, _ = layer.output_shape[-3:]
                except AttributeError:
                    out_shape = tf.keras.backend.int_shape(layer.output)
                    # out_shape is (batch, H, W, C)
                    out_h, out_w = out_shape[1], out_shape[2]

                # Use kernel variable to get (K_h, K_w, C_in, C_out)
                k_h, k_w, c_in, c_out = layer.kernel.shape.as_list()

                if None in (out_h, out_w, k_h, k_w, c_in, c_out):
                    # Skip layers with dynamic shapes we cannot resolve statically
                    continue

                flops = 2 * out_h * out_w * k_h * k_w * c_in * c_out
                # Optionally account for bias add per output activation
                if layer.use_bias:
                    flops += out_h * out_w * c_out

                total_flops += int(flops)
                
            elif isinstance(layer, tf.keras.layers.Dense):
                # FLOPs ≈ 2 * input_dim * output_dim
                in_dim, out_dim = layer.kernel.shape.as_list()
                if None in (in_dim, out_dim):
                    continue
                flops = 2 * in_dim * out_dim
                if layer.use_bias:
                    flops += out_dim
                total_flops += int(flops)
        
        return total_flops
