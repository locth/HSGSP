import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, List, Optional

class FrequencyAnalyzer:
    """Frequency domain analysis for filters"""
    
    def __init__(self, config):
        self.config = config
        self.frequency_bands = config.frequency_bands
    
    def compute_dct_2d(self, filters: np.ndarray) -> np.ndarray:
        """Compute 2D DCT for filters"""
        # filters shape: (height, width, in_channels, out_channels)
        h, w, in_c, out_c = filters.shape
        
        # Reshape for batch processing
        filters_reshaped = filters.transpose(2, 3, 0, 1).reshape(-1, h, w)
        
        # Convert to TensorFlow tensor for DCT
        filters_tf = tf.constant(filters_reshaped, dtype=tf.float32)
        
        # Compute 2D DCT using FFT (TensorFlow doesn't have direct DCT)
        # We'll use FFT as approximation and extract real components
        dct_result = tf.signal.fft2d(tf.cast(filters_tf, tf.complex64))
        dct_result = tf.math.real(dct_result)
        
        # Reshape back
        dct_result = dct_result.numpy().reshape(in_c, out_c, h, w)
        dct_result = dct_result.transpose(2, 3, 0, 1)
        
        return dct_result
    
    def extract_frequency_bands(self, dct_filters: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency bands from DCT coefficients"""
        h, w = dct_filters.shape[:2]
        bands = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Create frequency mask
            mask = self._create_frequency_mask(h, w, low_freq, high_freq)
            
            # Apply mask to extract band
            band_coeffs = dct_filters * mask[..., np.newaxis, np.newaxis]
            bands[band_name] = band_coeffs
        
        return bands
    
    def _create_frequency_mask(self, h: int, w: int, 
                               low_freq: float, high_freq: float) -> np.ndarray:
        """Create a frequency band mask"""
        # Create meshgrid for frequencies
        fx = np.fft.fftfreq(h)[:, np.newaxis]
        fy = np.fft.fftfreq(w)[np.newaxis, :]
        
        # Compute normalized frequency magnitude
        freq_magnitude = np.sqrt(fx**2 + fy**2)
        
        # Create band mask
        mask = ((freq_magnitude >= low_freq) & (freq_magnitude < high_freq)).astype(np.float32)
        
        return mask
    
    def compute_band_energy(self, band_coeffs: np.ndarray) -> np.ndarray:
        """Compute energy for each filter in a frequency band"""
        # Energy per filter (L2 norm of coefficients)
        energy = np.sqrt(np.sum(band_coeffs**2, axis=(0, 1, 2)))
        return energy
    
    def analyze_layer_frequency(self, layer: tf.keras.layers.Conv2D) -> Dict[str, np.ndarray]:
        """Analyze frequency characteristics of a convolutional layer"""
        # Get layer weights
        weights, _ = layer.get_weights()
        
        # Compute DCT
        dct_filters = self.compute_dct_2d(weights)
        
        # Extract frequency bands
        bands = self.extract_frequency_bands(dct_filters)
        
        # Compute band energies
        band_energies = {}
        for band_name, band_coeffs in bands.items():
            band_energies[band_name] = self.compute_band_energy(band_coeffs)
        
        return band_energies

    def frequency_profile(self, model: tf.keras.Model) -> Dict[str, Dict[str, float]]:
        """Compute frequency band energy profile for each Conv2D layer.

        This function does followings:
        - Extracts filters (kernels) from each Conv2D layer
        - Computes a 2D DCT-II (orthonormal) for each filter
        - Partitions coefficients into low/mid/high frequency bands
        - Computes L2 norm (energy) per band per layer

        Args:
            model: tf.keras.Model containing Conv2D layers
            dataset: Unused placeholder to match requested signature

        Returns:
            Dict[str, Dict[str, float]] mapping layer name -> {low, mid, high} energies
        """

        # Default band definitions aligned with Config.frequency_bands
        band_defs = {
            'low': (0.0, 0.25),
            'mid': (0.25, 0.5),
            'high': (0.5, 1.0),
        }

        def create_frequency_mask(h: int, w: int, low: float, high: float) -> np.ndarray:
            # Use DCT index-based "frequency" in [0,1], not FFT frequencies.
            # For DCT-II, indices k,l in [0..H-1],[0..W-1] increase spatial frequency.
            # Normalize indices so max radial is 1.
            u = (np.arange(h) / (h - 1)) if h > 1 else np.zeros(h)
            v = (np.arange(w) / (w - 1)) if w > 1 else np.zeros(w)
            U, V = np.meshgrid(u, v, indexing='ij')
            r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
            return ((r >= low) & (r < high)).astype(np.float32)

        def dct2_ortho(x: tf.Tensor) -> tf.Tensor:
            # Apply DCT-II along spatial dimensions (H, W) with orthonormal scaling
            # x shape: [H, W, Cin, Cout]
            # TensorFlow's tf.signal.dct only supports axis=-1; transpose to move
            # the target axis to the end before each transform.
            # DCT along H (axis 0)
            x_perm_h = tf.transpose(x, [1, 2, 3, 0])           # [W, Cin, Cout, H]
            y_h = tf.signal.dct(x_perm_h, type=2, norm='ortho')  # last axis (H)
            y = tf.transpose(y_h, [3, 0, 1, 2])                # back to [H, W, Cin, Cout]

            # DCT along W (axis 1)
            x_perm_w = tf.transpose(y, [0, 2, 3, 1])            # [H, Cin, Cout, W]
            y_w = tf.signal.dct(x_perm_w, type=2, norm='ortho')   # last axis (W)
            y2 = tf.transpose(y_w, [0, 3, 1, 2])                 # back to [H, W, Cin, Cout]
            return y2

        frequency_maps: Dict[str, Dict[str, float]] = {}

        for layer in getattr(model, 'layers', []):
            # Only process Conv2D layers with kernel weights
            if not isinstance(layer, tf.keras.layers.Conv2D):
                continue
            weights = layer.get_weights()
            if not weights:
                continue
            kernel = weights[0]  # [H, W, Cin, Cout]
            if kernel.ndim != 4:
                continue

            h, w, _, _ = kernel.shape
            k_tf = tf.convert_to_tensor(kernel, dtype=tf.float32)

            # 2D DCT for each filter
            dct_filters = dct2_ortho(k_tf)  # [H, W, Cin, Cout]

            # Compute energies per band
            energies = {}
            for band_name, (lo, hi) in band_defs.items():
                mask = create_frequency_mask(h, w, lo, hi)  # [H, W]
                mask_tf = tf.convert_to_tensor(mask, dtype=dct_filters.dtype)
                mask_tf = tf.reshape(mask_tf, (h, w, 1, 1))  # broadcast over channels
                band_coeffs = dct_filters * mask_tf
                # L2 norm of all band coefficients (scalar)
                energy = tf.norm(band_coeffs).numpy().item()
                energies[band_name] = energy

            frequency_maps[layer.name] = energies

        return frequency_maps

    def _compute_activation_statistics(
        self,
        model: tf.keras.Model,
        dataset: Optional[tf.data.Dataset],
        max_batches: int = 200,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute per-filter activation statistics (mean |activation| and std) for Conv2D layers.
        Returns normalized (z-scored) statistics per layer.
        """
        if dataset is None:
            return {}

        conv_layers = [l for l in getattr(model, "layers", []) if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            return {}

        try:
            activation_model = tf.keras.Model(
                inputs=model.input,
                outputs=[layer.output for layer in conv_layers],
            )
        except Exception:
            # Fallback: unable to build activation model (e.g., multiple inputs)
            return {}

        abs_acc: Dict[str, np.ndarray] = {}
        sq_acc: Dict[str, np.ndarray] = {}
        count_acc: Dict[str, float] = {}

        for layer in conv_layers:
            weights = layer.get_weights()
            if not weights:
                continue
            kernel = weights[0]
            if kernel.ndim != 4:
                continue
            cout = kernel.shape[-1]
            abs_acc[layer.name] = np.zeros((cout,), np.float64)
            sq_acc[layer.name] = np.zeros((cout,), np.float64)
            count_acc[layer.name] = 0.0

        batches = 0
        for xb, _ in dataset.take(max_batches):
            xb = tf.cast(xb, tf.float32)
            activations = activation_model(xb, training=False)
            if isinstance(activations, tf.Tensor):
                activations = [activations]
            for layer, act in zip(conv_layers, activations):
                if layer.name not in abs_acc:
                    continue
                act_np = act.numpy()
                abs_acc[layer.name] += np.sum(np.abs(act_np), axis=(0, 1, 2)).astype(np.float64)
                sq_acc[layer.name] += np.sum(np.square(act_np), axis=(0, 1, 2)).astype(np.float64)
                count_acc[layer.name] += float(act_np.shape[0] * act_np.shape[1] * act_np.shape[2])
            batches += 1

        if batches == 0:
            return {}

        def _normalize(arr: np.ndarray) -> np.ndarray:
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            if sigma < 1e-12:
                return np.zeros_like(arr, dtype=np.float64)
            return (arr - mu) / (sigma + 1e-12)

        stats: Dict[str, Dict[str, np.ndarray]] = {}
        for layer in conv_layers:
            if layer.name not in abs_acc:
                continue
            total_count = count_acc.get(layer.name, 0.0)
            if total_count <= 0.0:
                continue
            mean_abs = abs_acc[layer.name] / total_count
            mean_sq = sq_acc[layer.name] / total_count
            variance = np.maximum(mean_sq - np.square(mean_abs), 0.0)
            std = np.sqrt(variance)
            stats[layer.name] = {
                "mean_abs": _normalize(mean_abs),
                "std": _normalize(std),
            }
        return stats

    def compute_frequency_importance_scores(
        self,
        model: tf.keras.Model,
        band_weights: Dict[str, float] | None = None,
        energy_exponent: float = 0.5,
        normalize_per_layer: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-filter importance scores for each Conv2D layer using DCT-based
        frequency band energies.

        For each Conv2D kernel K [H,W,Cin,Cout]:
          1) Compute 2D DCT-II (orthonormal) over spatial dims (H,W).
          2) Partition coefficients into low/mid/high bands (DCT index radius).
          3) For each output channel j, compute band energies E_b[j] = ||coeffs_b||_2.
          4) Let T[j] = sum_b E_b[j], r_b[j] = E_b[j] / (T[j] + eps).
          5) Importance[j] = (sum_b w_b * r_b[j]) * (T[j] ** energy_exponent).

        - band_weights (default): {'low': 0.6, 'mid': 0.3, 'high': 0.1}
        - energy_exponent: soft boost for overall energy magnitude; 0 disables.
        - normalize_per_layer: min-max normalize importance to [0,1] within a layer.

        Returns:
            Dict[layer_name, importance_scores (np.ndarray of shape [Cout])]
        """
        # Defaults
        if band_weights is None:
            band_weights = {'low': 0.6, 'mid': 0.3, 'high': 0.1}

        # Local helpers (DCT-II via axis=-1 only support)
        def dct2_ortho(x: tf.Tensor) -> tf.Tensor:
            # x: [H, W, Cin, Cout]
            # DCT along H
            x_perm_h = tf.transpose(x, [1, 2, 3, 0])
            y_h = tf.signal.dct(x_perm_h, type=2, norm='ortho')
            y = tf.transpose(y_h, [3, 0, 1, 2])
            # DCT along W
            x_perm_w = tf.transpose(y, [0, 2, 3, 1])
            y_w = tf.signal.dct(x_perm_w, type=2, norm='ortho')
            y2 = tf.transpose(y_w, [0, 3, 1, 2])
            return y2

        def create_mask(h: int, w: int, low: float, high: float) -> np.ndarray:
            u = (np.arange(h) / (h - 1)) if h > 1 else np.zeros(h)
            v = (np.arange(w) / (w - 1)) if w > 1 else np.zeros(w)
            U, V = np.meshgrid(u, v, indexing='ij')
            r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
            return ((r >= low) & (r < high)).astype(np.float32)

        # Band definitions aligned with config defaults
        band_defs = self.config.frequency_bands if hasattr(self.config, 'frequency_bands') and self.config.frequency_bands else {
            'low': (0.0, 0.25),
            'mid': (0.25, 0.5),
            'high': (0.5, 1.0),
        }

        importance: Dict[str, np.ndarray] = {}

        for layer in getattr(model, 'layers', []):
            if not isinstance(layer, tf.keras.layers.Conv2D):
                continue
            weights = layer.get_weights()
            if not weights:
                continue
            kernel = weights[0]  # [H,W,Cin,Cout]
            if kernel.ndim != 4:
                continue

            h, w, _, cout = kernel.shape
            k_tf = tf.convert_to_tensor(kernel, dtype=tf.float32)
            dct_k = dct2_ortho(k_tf)  # [H,W,Cin,Cout]

            band_energy = {}
            total_energy = np.zeros((cout,), dtype=np.float64)
            for bname, (lo, hi) in band_defs.items():
                mask = create_mask(h, w, lo, hi)
                mask_tf = tf.convert_to_tensor(mask, dtype=dct_k.dtype)
                mask_tf = tf.reshape(mask_tf, (h, w, 1, 1))
                coeffs_b = dct_k * mask_tf
                # Energy per output channel
                e_b = tf.sqrt(tf.reduce_sum(tf.square(coeffs_b), axis=(0, 1, 2)))  # [Cout]
                e_b_np = e_b.numpy().astype(np.float64)
                band_energy[bname] = e_b_np
                total_energy += e_b_np

            eps = 1e-12
            # Weighted ratio component
            weighted_ratio = np.zeros((cout,), dtype=np.float64)
            for bname, e_b in band_energy.items():
                w_b = float(band_weights.get(bname, 0.0)) if band_weights is not None else 0.0
                weighted_ratio += w_b * (e_b / (total_energy + eps))

            # Magnitude boost
            mag = np.power(np.maximum(total_energy, 0.0) + eps, float(energy_exponent))
            scores = weighted_ratio * mag

            if normalize_per_layer:
                s_min = float(np.min(scores))
                s_max = float(np.max(scores))
                rng = s_max - s_min
                if rng < 1e-12:
                    scores_norm = np.zeros_like(scores, dtype=np.float64)
                else:
                    scores_norm = (scores - s_min) / (rng + 1e-12)
                importance[layer.name] = scores_norm.astype(np.float32)
            else:
                importance[layer.name] = scores.astype(np.float32)

        return importance

    def build_frequency_relevance_net(hidden_units: int = 32) -> tf.keras.Model:
        """
        TensorFlow/Keras version of FrequencyRelevanceNet.
        Input:  3-dim vector [low, mid, high] (ratios or raw energies)
        Output: 3-dim softmax weights over the three bands
        """
        inputs = tf.keras.Input(shape=(3,), name="band_energies")
        x = tf.keras.layers.Dense(hidden_units, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs, name="FrequencyRelevanceNetTF")
        return model
    
    def compute_frequency_importance_scores_with_frn(
        self,
        model: tf.keras.Model,
        frn_model: Optional[tf.keras.Model] = None,
        use_ratios_as_input: bool = True,
        energy_exponent: float = 0.5,
        normalize_per_layer: bool = True,
        fallback_band_weights: Optional[Dict[str, float]] = None,
        activation_dataset: Optional[tf.data.Dataset] = None,
        activation_max_batches: int = 200,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-filter importance scores for each Conv2D layer using DCT-based
        frequency band energies and an optional TF/Keras FrequencyRelevanceNet.

        For each Conv2D kernel K [H,W,Cin,Cout]:
        1) Compute DCT-II (orthonormal) over (H,W).
        2) Split coefficients into low/mid/high bands via index-radius masks.
        3) Per output channel j, compute band energies E_b[j] (Frobenius over H,W,Cin).
        4) Let T[j] = sum_b E_b[j]; ratios r_b[j] = E_b[j] / (T[j] + eps).
        5) Build FRN input x_j = [r_low, r_mid, r_high] if use_ratios_as_input else [E_low, E_mid, E_high] (log1p).
        6) If frn_model provided: w_j = softmax(FRN(x_j)); else use fallback_band_weights.
        7) score_j = (sum_b w_b[j] * r_b[j]) * (T[j] ** energy_exponent).
        8) Optionally min–max normalize scores within the layer.

        Returns:
            Dict[layer_name, scores_per_filter] with shape [Cout] per layer.
        """
        # ---------- defaults ----------
        if fallback_band_weights is None:
            fallback_band_weights = {'low': 0.6, 'mid': 0.3, 'high': 0.1}

        activation_stats = self._compute_activation_statistics(
            model=model,
            dataset=activation_dataset,
            max_batches=activation_max_batches,
        )

        def dct2_ortho(x: tf.Tensor) -> tf.Tensor:
            # x: [H, W, Cin, Cout]
            # DCT along H
            x_perm_h = tf.transpose(x, [1, 2, 3, 0])
            y_h = tf.signal.dct(x_perm_h, type=2, norm='ortho')
            y = tf.transpose(y_h, [3, 0, 1, 2])
            # DCT along W
            x_perm_w = tf.transpose(y, [0, 2, 3, 1])
            y_w = tf.signal.dct(x_perm_w, type=2, norm='ortho')
            y2 = tf.transpose(y_w, [0, 3, 1, 2])
            return y2  # [H,W,Cin,Cout]

        def create_mask(h: int, w: int, low: float, high: float) -> np.ndarray:
            # DCT-II: low freq at (0,0); use normalized radius in index space
            u = (np.arange(h) / (h - 1)) if h > 1 else np.zeros(h)
            v = (np.arange(w) / (w - 1)) if w > 1 else np.zeros(w)
            U, V = np.meshgrid(u, v, indexing='ij')
            r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
            return ((r >= low) & (r < high)).astype(np.float32)

        band_defs = getattr(self.config, "frequency_bands", None) or {
            'low': (0.0, 0.25),
            'mid': (0.25, 0.5),
            'high': (0.5, 1.0),
        }

        importance: Dict[str, np.ndarray] = {}

        # ---------- iterate Conv2D layers ----------
        for layer in getattr(model, 'layers', []):
            if not isinstance(layer, tf.keras.layers.Conv2D):
                continue
            weights = layer.get_weights()
            if not weights:
                continue
            kernel = weights[0]  # [H,W,Cin,Cout]
            if kernel.ndim != 4:
                continue

            h, w, _, cout = kernel.shape
            k_tf = tf.convert_to_tensor(kernel, dtype=tf.float32)
            dct_k = dct2_ortho(k_tf)  # [H,W,Cin,Cout]

            band_energy = {}
            total_energy = np.zeros((cout,), dtype=np.float64)

            # band energies per output channel
            for bname, (lo, hi) in band_defs.items():
                mask = create_mask(h, w, lo, hi)
                mask_tf = tf.convert_to_tensor(mask, dtype=dct_k.dtype)
                mask_tf = tf.reshape(mask_tf, (h, w, 1, 1))
                coeffs_b = dct_k * mask_tf
                e_b = tf.sqrt(tf.reduce_sum(tf.square(coeffs_b), axis=(0, 1, 2)))  # [Cout]
                e_b_np = e_b.numpy().astype(np.float64)
                band_energy[bname] = e_b_np
                total_energy += e_b_np

            eps = 1e-12
            # ratios r_b[j] = E_b[j] / T[j]
            ratios = {b: e / (total_energy + eps) for b, e in band_energy.items()}

            if use_ratios_as_input:
                feature_core = np.stack(
                    [ratios['low'], ratios['mid'], ratios['high']],
                    axis=1,
                )
            else:
                feature_core = np.stack(
                    [band_energy['low'], band_energy['mid'], band_energy['high']],
                    axis=1,
                )
                feature_core = np.log1p(feature_core)

            expected_dim = None
            if frn_model is not None:
                try:
                    expected_dim = int(frn_model.input_shape[-1])
                except Exception:
                    expected_dim = feature_core.shape[1]

            layer_activation_stats = activation_stats.get(layer.name) if activation_stats else None
            extra_features: Optional[np.ndarray] = None
            if layer_activation_stats is not None:
                extra_features = np.stack(
                    [
                        layer_activation_stats['mean_abs'],
                        layer_activation_stats['std'],
                    ],
                    axis=1,
                )

            if expected_dim is not None:
                needed_extra = max(expected_dim - feature_core.shape[1], 0)
                if extra_features is None:
                    if needed_extra > 0:
                        extra_features = np.zeros((cout, needed_extra), dtype=np.float64)
                else:
                    if extra_features.shape[1] < needed_extra:
                        pad = np.zeros((cout, needed_extra - extra_features.shape[1]), dtype=np.float64)
                        extra_features = np.concatenate([extra_features, pad], axis=1)
                    elif extra_features.shape[1] > needed_extra:
                        extra_features = extra_features[:, :needed_extra]
            else:
                if extra_features is not None:
                    expected_dim = feature_core.shape[1] + extra_features.shape[1]

            if extra_features is not None and extra_features.shape[1] > 0:
                x_mat = np.concatenate([feature_core, extra_features], axis=1)
            else:
                x_mat = feature_core

            # Compute per-filter band weights
            if frn_model is not None:
                # Keras forward pass (no grad)
                x_tf = tf.convert_to_tensor(x_mat.astype(np.float32))
                w_tf = frn_model(x_tf, training=False)     # [Cout,3], softmax
                w_np = w_tf.numpy().astype(np.float64)
                w_low, w_mid, w_high = w_np[:, 0], w_np[:, 1], w_np[:, 2]
            else:
                # Fallback static weights
                w_low  = np.full((cout,), float(fallback_band_weights.get('low',  0.0)), dtype=np.float64)
                w_mid  = np.full((cout,), float(fallback_band_weights.get('mid',  0.0)), dtype=np.float64)
                w_high = np.full((cout,), float(fallback_band_weights.get('high', 0.0)), dtype=np.float64)

            # Weighted mixture over band *ratios*
            weighted_ratio = (
                w_low  * ratios['low'] +
                w_mid  * ratios['mid'] +
                w_high * ratios['high']
            )  # [Cout]

            # Magnitude boost T^alpha
            mag = np.power(np.maximum(total_energy, 0.0) + eps, float(energy_exponent))  # [Cout]
            scores = weighted_ratio * mag  # [Cout]

            # Optional per-layer min–max normalization
            if normalize_per_layer:
                s_min = float(np.min(scores))
                s_max = float(np.max(scores))
                rng = s_max - s_min
                scores = np.zeros_like(scores, dtype=np.float64) if rng < 1e-12 else (scores - s_min) / (rng + 1e-12)

            importance[layer.name] = scores.astype(np.float32)

        return importance
