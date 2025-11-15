import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

from pruning.frequency_analyzer import FrequencyAnalyzer
from pruning.pruning_strategy import PruningStrategy
from utils.logger import Logger
from models.model_utils import ModelUtils

class HSGSP:
    """
    Frequency-Aware CNN Channel Pruning
    """
    
    def __init__(self, config):
        self.config = config
        self.frequency_analyzer = FrequencyAnalyzer(config)
        self.pruning_strategy = PruningStrategy(config)
        self.logger = Logger(config)
    
    def analyze_dataset_complexity(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Analyze the complexity of a single dataset
        
        Args:
            dataset: TensorFlow dataset to analyze
            
        Returns:
            Dictionary containing complexity metrics
        """
        complexities = []
        frequency_energies = []

        # Deterministic sampling: first 10 batches, 5 images per batch
        for batch_images, _ in dataset.take(100):
            # Ensure images are on CPU for .numpy(); or map() beforehand if you prefer TF ops end-to-end
            batch_images_np = batch_images.numpy()
            n = min(10, batch_images_np.shape[0])
            for i in range(n):
                img = batch_images_np[i]

                # 1) Preprocess: luminance in [0,1]
                img = self._resize_for_analysis(img, size=(224,224))
                img01_y = self._to_luminance01(img)

                # 2) Frequency: HF ratio via DCT-II with correct mask
                hf_ratio = self._high_freq_ratio_dct(img01_y, threshold=0.5)
                frequency_energies.append(hf_ratio)

                # 3) Structural: mean Sobel magnitude (robust to noise vs raw gradient)
                c_struct = self._structural_complexity_sobel_tf(img01_y)
                complexities.append(c_struct)

        print("=================================")
        print("Mean Complexities:", np.mean(complexities))
        print("Mean Frequency Energies:", np.mean(frequency_energies))
        print("=================================")
        
        return {
            'mean_complexity': np.mean(complexities),
            'std_complexity': np.std(complexities),
            'high_freq_ratio': np.mean(frequency_energies),
            'complexity_score': self._compute_complexity_score(
                np.mean(complexities), np.mean(frequency_energies)
            )
        }

    def analyze_dataset_comprehensive_complexity(
        self,
        dataset: tf.data.Dataset,
        model: Optional[tf.keras.Model] = None,
        lambda_weights: Optional[Dict[str, float]] = None,
        max_batches: int = 100,
        bins: int = 32,
        spatial_ref: int = 224,
        csr_scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute a comprehensive Dataset Complexity Score (DCS) using:

            DCS = λ1·IC + λ2·FE + λ3·SR + λ4·CSR

        Where:
        - IC (Inter-class Confusion):
            If a model is provided, IC = off-diagonal mass of confusion matrix (≈ 1 - accuracy).
            Otherwise, estimate IC via average pairwise cosine similarity between per-class
            feature histograms (32-bin luminance histograms).
        - FE (Feature Entropy): Normalized Shannon entropy of luminance histogram across the dataset.
        - SR (Spatial Resolution): Normalized sqrt(H*W) versus spatial_ref (default 224).
        - CSR (Class-to-Sample rate): (num_classes / avg_samples_per_class), scaled by csr_scale and clipped to [0,1].

        Args:
            dataset: tf.data.Dataset of (images, one-hot labels)
            model: Optional model to compute true confusion; if None, a proxy is used
            lambda_weights: Optional dict overriding weights; defaults to config.complexity_weights
            max_batches: Limit on batches to sample for computation
            bins: Histogram bins for entropy and centroid features
            spatial_ref: Reference resolution (√area) used for SR normalization
            csr_scale: Scale factor to bring CSR into [0,1] for typical datasets

        Returns:
            Dictionary with components and overall 'complexity_score'
        """
        # Weights (λ's)
        weights = lambda_weights or getattr(self.config, 'complexity_weights', None) or {
            'inter_class_confusion': 0.3,
            'feature_entropy': 0.3,
            'spatial_resolution': 0.2,
            'class_diversity': 0.2,
        }
        # Normalize weights to sum to 1
        w_sum = sum(float(v) for v in weights.values()) or 1.0
        weights = {k: float(v) / w_sum for k, v in weights.items()}

        # Accumulators
        entropy_hist = np.zeros(bins, dtype=np.float64)
        class_histograms: Dict[int, np.ndarray] = {}
        class_counts: Dict[int, int] = {}
        total_samples = 0
        sum_sqrt_area = 0.0

        # Confusion matrix accumulators (if model provided)
        confusion = None
        num_classes = None

        # Sampling loop
        for batch_idx, (images, labels) in enumerate(dataset.take(max_batches)):
            imgs = images.numpy()
            labs = labels.numpy()

            # Determine number of classes from first batch
            if num_classes is None:
                num_classes = int(labs.shape[-1]) if labs.ndim == 2 else int(np.max(labs) + 1)
                if model is not None:
                    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

            # Spatial resolution accumulator
            h, w = imgs.shape[1], imgs.shape[2]
            sum_sqrt_area += float(np.sqrt(h * w) * imgs.shape[0])

            # Convert to luminance [0,1] for histogram-based computations
            imgs01_y = []
            for i in range(imgs.shape[0]):
                y = self._to_luminance01(imgs[i])
                imgs01_y.append(y)
            imgs01_y = np.stack(imgs01_y, axis=0)

            # Global entropy histogram (bins over [0,1])
            hist_batch, _ = np.histogram(imgs01_y, bins=bins, range=(0.0, 1.0))
            entropy_hist += hist_batch.astype(np.float64)

            # Per-class histogram centroids and class counts
            label_indices = np.argmax(labs, axis=1) if labs.ndim == 2 else labs
            for i in range(imgs.shape[0]):
                c = int(label_indices[i])
                class_counts[c] = class_counts.get(c, 0) + 1
                hist_i, _ = np.histogram(imgs01_y[i], bins=bins, range=(0.0, 1.0))
                if c not in class_histograms:
                    class_histograms[c] = hist_i.astype(np.float64)
                else:
                    class_histograms[c] += hist_i.astype(np.float64)

            # If model provided, update confusion
            if model is not None:
                preds = model(images, training=False)
                pred_idx = tf.argmax(preds, axis=1).numpy().astype(int)
                true_idx = label_indices.astype(int)
                for t, p in zip(true_idx, pred_idx):
                    confusion[t, p] += 1

            total_samples += imgs.shape[0]

        # Guard against empty dataset
        if total_samples == 0:
            return {
                'inter_class_confusion': 0.0,
                'feature_entropy': 0.0,
                'spatial_resolution': 0.0,
                'class_diversity': 0.0,
                'num_classes': 0,
                'avg_samples_per_class': 0.0,
                'samples_used': 0,
                'complexity_score': 0.0,
            }

        # Compute SR (normalized)
        avg_sqrt_area = sum_sqrt_area / float(total_samples)
        spatial_res = float(np.clip(avg_sqrt_area / float(spatial_ref), 0.0, 1.0))

        # Compute FE (normalized entropy)
        p = entropy_hist / (np.sum(entropy_hist) + 1e-12)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        max_ent = float(np.log(bins))
        feature_entropy = float(np.clip(ent / (max_ent + 1e-12), 0.0, 1.0))

        # Compute CD
        observed_classes = sorted(class_counts.keys())
        k = len(observed_classes) if num_classes is None else max(len(observed_classes), num_classes)
        avg_per_class = float(total_samples) / float(max(k, 1))
        class2sample_rate_raw = float(k) / float(avg_per_class + 1e-12)  # k^2 / N
        class2sample_rate = float(np.clip(class2sample_rate_raw * csr_scale, 0.0, 1.0))

        # Compute IC
        if model is not None and confusion is not None:
            total = float(confusion.sum())
            off_diag = float(total - np.trace(confusion))
            interClass_confusion = float(off_diag / (total + 1e-12))  # 0..1
        else:
            # Proxy: average pairwise cosine similarity of per-class histogram centroids
            if k <= 1:
                interClass_confusion = 0.0
            else:
                # Build normalized centroids
                centroids = []
                for c in observed_classes:
                    v = class_histograms[c] / (class_counts[c] + 1e-12)
                    v = v.astype(np.float64)
                    n = np.linalg.norm(v) + 1e-12
                    centroids.append(v / n)
                centroids = np.stack(centroids, axis=0)  # [k, bins]
                sim = centroids @ centroids.T  # cosine sim since L2-normalized
                # Average off-diagonal similarity
                off_sum = sim.sum() - np.trace(sim)
                interClass_confusion = float(off_sum / (k * (k - 1) + 1e-12))
                interClass_confusion = float(np.clip(interClass_confusion, 0.0, 1.0))

        # Combine into DCS
        dcs = (
            weights.get('inter_class_confusion', 0.25) * interClass_confusion +
            weights.get('feature_entropy', 0.25) * feature_entropy +
            weights.get('spatial_resolution', 0.25) * spatial_res +
            weights.get('class_diversity', 0.25) * class2sample_rate
        )
        dcs = float(np.clip(dcs, 0.0, 1.0))

        return {
            'inter_class_confusion': interClass_confusion,
            'feature_entropy': feature_entropy,
            'spatial_resolution': spatial_res,
            'class_diversity': class2sample_rate,
            'num_classes': k,
            'avg_samples_per_class': avg_per_class,
            'samples_used': total_samples,
            'complexity_score': dcs,
        }

    def compute_optimal_pruning_ratio(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        min_ratio: float = 0.2,
        max_ratio: float = 0.85,
        weights: Optional[Dict[str, float]] = None,
        structure_norms: Optional[Dict[str, Tuple[float, float]]] = None,
        use_model_for_confusion: bool = True,
        max_batches: int = 80,
        baseline_accuracy: Optional[float] = None,
        target_accuracy_drop: Optional[float] = None,
        return_details: bool = False,
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Estimate an optimal global pruning ratio using dataset complexity and
        model structural capacity/redundancy. Returns a ratio in [min_ratio, max_ratio].

        Method
        1) Dataset Complexity (0..1): computed via analyze_dataset_comprehensive_complexity
           on the provided dataset (optionally with the model to get true confusion).
           Higher complexity -> more conservative pruning.
        2) Model Structure Score (0..1): combines normalized log-params, log-FLOPs,
           conv-depth, and average conv width to estimate prune-ability (capacity/redundancy).
           Higher structure score -> more aggressive pruning possible.
        3) Blend: r = min + (max - min) * clamp(w_struct*S + w_data*(1 - C)).

        Args:
            model: Keras model to analyze
            dataset: tf.data.Dataset used to estimate complexity
            min_ratio: Lower bound for global pruning ratio
            max_ratio: Upper bound for global pruning ratio
            weights: Optional weights for blending and structure components, e.g.:
                {
                  'blend_struct': 0.6, 'blend_data': 0.4,
                  'struct_params': 0.4, 'struct_flops': 0.2,
                  'struct_depth': 0.2, 'struct_width': 0.2
                }
            structure_norms: Optional min/max ranges for normalization, e.g.:
                {
                  'log_params': (5.0, 9.0),     # ~1e5 .. 1e9
                  'log_flops': (7.0, 11.0),     # ~1e7 .. 1e11
                  'depth': (5.0, 30.0),
                  'width': (32.0, 512.0)
                }
            use_model_for_confusion: If True, uses model predictions to compute confusion
            max_batches: Limit batches when computing complexity
            return_details: If True, also return a dict with intermediate components

        Returns:
            ratio or (ratio, details)
        """

        # Defaults
        w = {
            'blend_struct': 0.6, 'blend_data': 0.4,
            'struct_params': 0.4, 'struct_flops': 0.2,
            'struct_depth': 0.2, 'struct_width': 0.2,
        }
        if weights:
            w.update({k: float(v) for k, v in weights.items() if k in w})

        norms = {
            'log_params': (5.0, 9.0),   # 1e5 .. 1e9
            'log_flops': (7.0, 11.0),   # 1e7 .. 1e11
            'depth': (5.0, 30.0),
            'width': (32.0, 512.0),
        }
        if structure_norms:
            for k, rng in structure_norms.items():
                if k in norms and isinstance(rng, (tuple, list)) and len(rng) == 2:
                    norms[k] = (float(rng[0]), float(rng[1]))

        def _norm(x: float, lo: float, hi: float) -> float:
            if np.isnan(x) or np.isinf(x):
                return 0.0
            if hi == lo:
                return 0.0
            return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

        # 1) Dataset complexity (C in 0..1)
        comp = self.analyze_dataset_comprehensive_complexity(
            dataset=dataset,
            model=model if use_model_for_confusion else None,
            max_batches=max_batches,
        )
        C = float(comp.get('complexity_score', 0.0))  # higher => harder dataset
        C = float(np.clip(C, 0.0, 1.0))

        # 2) Model structure score (S in 0..1)
        total_params = float(model.count_params())
        flops = float(ModelUtils.compute_flops(model))

        # Conv layer stats
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        depth = float(len(conv_layers))
        widths = [float(getattr(l, 'filters', 0) or 0) for l in conv_layers]
        avg_width = float(np.mean(widths)) if widths else 0.0

        # Log-scale for params/flops to reduce dynamic range
        log_params = float(np.log10(max(total_params, 1.0)))
        log_flops = float(np.log10(max(flops, 1.0)))

        s_params = _norm(log_params, *norms['log_params'])
        s_flops = _norm(log_flops, *norms['log_flops'])
        s_depth = _norm(depth, *norms['depth'])
        s_width = _norm(avg_width, *norms['width'])

        S = (
            w['struct_params'] * s_params +
            w['struct_flops'] * s_flops +
            w['struct_depth'] * s_depth +
            w['struct_width'] * s_width
        )
        S = float(np.clip(S, 0.0, 1.0))

        # 3) Blend structure capacity and dataset simplicity to a ratio
        simplicity = 1.0 - C
        blend = float(np.clip(w['blend_struct'] * S + w['blend_data'] * simplicity, 0.0, 1.0))

        ratio = float(np.clip(min_ratio + blend * (max_ratio - min_ratio), min_ratio, max_ratio))

        ratio_cap = getattr(self.config, 'max_global_pruning_ratio', None)
        if ratio_cap is not None:
            ratio = min(ratio, float(np.clip(ratio_cap, min_ratio, max_ratio)))

        min_keep = getattr(self.config, 'min_global_keep', None)
        if min_keep is not None:
            min_keep = float(np.clip(min_keep, 0.0, 0.95))
            ratio = min(ratio, 1.0 - min_keep)

        # Extra safety clamps for very complex datasets
        if C > 0.8:
            ratio = min(ratio, (min_ratio + max_ratio) * 0.5)  # cap at midpoint
        elif C > 0.6:
            ratio = min(ratio, max_ratio * 0.7)

        accuracy_guard_multiplier = None
        if baseline_accuracy is not None:
            acc = float(np.clip(baseline_accuracy, 0.0, 1.0))
            tol = float(target_accuracy_drop if target_accuracy_drop is not None else getattr(self.config, 'max_accuracy_drop', 0.03))
            tol = float(np.clip(tol, 0.0, 0.1))
            guard_center = getattr(self.config, 'accuracy_guard_center', 0.7)
            guard_sharpness = getattr(self.config, 'accuracy_guard_sharpness', 0.12)
            acc_margin = max(0.0, acc - (guard_center + tol))
            accuracy_guard_multiplier = float(np.exp(-acc_margin / max(guard_sharpness, 1e-3)))
            accuracy_guard_multiplier = float(np.clip(accuracy_guard_multiplier, 0.4, 1.0))
            ratio *= accuracy_guard_multiplier

        ratio = float(np.clip(ratio, min_ratio, max_ratio))
        if ratio_cap is not None:
            ratio = min(ratio, float(np.clip(ratio_cap, min_ratio, max_ratio)))
        if min_keep is not None:
            ratio = min(ratio, 1.0 - min_keep)

        if not return_details:
            return ratio

        details = {
            'pruning_ratio': ratio,
            'complexity_score': C,
            'simplicity': simplicity,
            'structure_score': S,
            's_params': s_params,
            's_flops': s_flops,
            's_depth': s_depth,
            's_width': s_width,
            'total_params': total_params,
            'flops': flops,
            'conv_depth': depth,
            'avg_conv_width': avg_width,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_guard_multiplier': accuracy_guard_multiplier,
            'ratio_cap': ratio_cap,
            'min_keep': min_keep,
        }
        return ratio, details

    def _resize_for_analysis(self, img, size=(224,224)):
        x = tf.convert_to_tensor(img, dtype=tf.float32)
        x = tf.image.resize(x, size, method=tf.image.ResizeMethod.BICUBIC, antialias=True)
        return x.numpy()
    
    def _create_high_freq_mask(self, h: int, w: int, threshold: float = 0.5) -> np.ndarray:
        """
        High-frequency mask for DCT-II coordinates.
        Low frequency is at (0,0). We define a normalized radius in (u,v) index space:
            r = sqrt( (u/(h-1))^2 + (v/(w-1))^2 )
        Mark HF where r > threshold.
        """
        u = np.arange(h)[:, None]
        v = np.arange(w)[None, :]
        ru = u / max(h - 1, 1)
        rv = v / max(w - 1, 1)
        r = np.sqrt(ru**2 + rv**2)
        mask = (r > threshold).astype(np.float32)
        return mask
    
    def _to_luminance01(self, img: np.ndarray) -> np.ndarray:
        """
        Convert HxW or HxWxC image to luminance in [0,1].
        Assumes input may be [0,255] or [0,1]. Safely rescales if needed.
        """
        x = img.astype(np.float32)
        # If the max suggests 8-bit range, rescale to [0,1]
        if x.max() > 1.5:
            x = x / 255.0

        if x.ndim == 3 and x.shape[-1] == 3:
            # ITU-R BT.601 luma weights (common default)
            x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        elif x.ndim == 2:
            pass  # already grayscale
        else:
            # Handle unexpected shapes by squeezing if possible
            x = np.squeeze(x)
            if x.ndim != 2:
                raise ValueError(f"Unexpected image shape after squeeze: {img.shape} -> {x.shape}")

        # Ensure strictly within [0,1]
        x = np.clip(x, 0.0, 1.0)
        return x
    
    def _structural_complexity_sobel_tf(self, img01_y: np.ndarray) -> float:
        """
        Compute mean Sobel magnitude using TensorFlow conv2d.
        img01_y: HxW numpy array in [0,1] (luminance)
        Returns a scalar float.
        """
        # [1, H, W, 1]
        x = tf.convert_to_tensor(img01_y[None, ..., None], dtype=tf.float32)

        # Sobel kernels
        gx = tf.constant([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=tf.float32)
        gy = tf.transpose(gx)

        gx = gx[:, :, tf.newaxis, tf.newaxis]  # [3,3,1,1]
        gy = gy[:, :, tf.newaxis, tf.newaxis]

        sx = tf.nn.conv2d(x, gx, strides=1, padding="SAME")
        sy = tf.nn.conv2d(x, gy, strides=1, padding="SAME")
        mag = tf.sqrt(tf.maximum(sx*sx + sy*sy, 1e-12))
        return float(tf.reduce_mean(mag).numpy())
    
    def _high_freq_ratio_dct(self, img01_y: np.ndarray, threshold: float = 0.5) -> float:
        """
        Compute HF/Total energy using DCT-II (orthonormal) on luminance in [0,1].
        """

        x = tf.convert_to_tensor(img01_y, dtype=tf.float32)  # [H,W]
        # DCT-II along rows then columns (orthonormal => energy preserved up to numerical eps)
        d0 = tf.signal.dct(x, type=2, norm='ortho')                  # [H,W]
        d2 = tf.signal.dct(tf.transpose(d0), type=2, norm='ortho')   # [W,H]
        X = tf.transpose(d2)                                         # [H,W]
        X = X.numpy()

        h, w = X.shape
        mask = self._create_high_freq_mask(h, w, threshold)
        hf = float(np.sum((np.abs(X) * mask)**2))
        tot = float(np.sum(np.abs(X)**2) + 1e-12)
        return hf / tot
    
    def _compute_complexity_score(self, structural_complexity: float, 
                                 freq_ratio: float) -> float:
        """
        Compute overall complexity score
        
        Higher score = more complex dataset = need more conservative pruning
        """
        # Normalize and combine metrics
        score = (structural_complexity * 0.5 + freq_ratio * 100 * 0.5)
        return np.clip(score / 10, 0, 1)  # Normalize to [0, 1]

    def compute_layer_sensitivity(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        """
        Estimate layer sensitivity from a single dataset using gradient magnitudes.

        For each Conv2D layer l, compute the average Frobenius norm of dL/dW_l over
        a few batches, then z-normalize across layers to make values comparable.

        Args:
            model: Keras model
            dataset: tf.data.Dataset of (images, labels)
            max_batches: number of batches to sample for gradient estimation

        Returns:
            Dict[layer_name, sensitivity_z]: z-scored sensitivity (higher => more sensitive)
        """
        conv_layers = ModelUtils.get_conv_layers(model)
        if not conv_layers:
            return {}

        # Determine if model outputs logits or probabilities (softmax)
        def _from_logits(m: tf.keras.Model) -> bool:
            last = m.layers[-1]
            if isinstance(last, tf.keras.layers.Softmax):
                return False
            act = getattr(last, 'activation', None)
            if act is not None and act == tf.keras.activations.softmax:
                return False
            return True

        from_logits = _from_logits(model)
        loss_cce_cat = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
        loss_cce_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

        # Map conv layer -> index and variable refs
        layer_kernels: List[tf.Variable] = []
        layer_names: List[str] = []
        for l in conv_layers:
            if hasattr(l, 'kernel') and l.kernel is not None:
                layer_kernels.append(l.kernel)
                layer_names.append(l.name)

        if not layer_kernels:
            return {}

        # Accumulators for gradient norms
        accum = np.zeros(len(layer_kernels), dtype=np.float64)
        batches = 0

        for xb, yb in dataset.take(max_batches):
            xb = tf.convert_to_tensor(xb, dtype=tf.float32)
            # choose loss variant by label shape
            sparse = (yb.ndim == 1) or (yb.ndim == 2 and yb.shape[-1] == 1)
            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                if sparse:
                    y_true = tf.cast(tf.reshape(yb, [-1]), tf.int32)
                    loss = loss_cce_sparse(y_true, preds)
                else:
                    y_true = tf.cast(yb, tf.float32)
                    loss = loss_cce_cat(y_true, preds)
            grads = tape.gradient(loss, layer_kernels)
            # Some layers might produce None gradients if unused; treat as zeros
            for i, g in enumerate(grads):
                if g is None:
                    continue
                accum[i] += float(tf.norm(g).numpy())
            batches += 1

        if batches == 0:
            return {name: 0.0 for name in layer_names}

        vals = accum / float(batches)
        # z-score across layers for comparability
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if sigma < 1e-12:
            sens_z = np.zeros_like(vals)
        else:
            sens_z = (vals - mu) / (sigma + 1e-12)

        return {name: float(sens_z[i]) for i, name in enumerate(layer_names)}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def compute_adaptive_pruning_ratios(
        self,
        model: tf.keras.Model,
        dcs: float,
        lcs: Dict[str, float],
        base_ratio: float = 0.5,
        min_clip: float = 0.1,
        max_clip: float = 0.9,
    ) -> Dict[str, float]:
        """
        Compute per-layer adaptive pruning ratios combining dataset complexity (DCS),
        depth factor, and layer sensitivity (LCS).

        ratio_l = clip( base * depth_factor * complexity_factor * sigmoid(LCS_l),
                        min_clip, max_clip )

        depth_factor = 1 - (idx / total_layers) * 0.3
        complexity_factor = 1 + (DCS - 0.5) * 0.4

        Args:
            model: Keras model
            dcs: dataset complexity score in [0,1]
            lcs: dict mapping layer_name -> sensitivity (e.g., z-scored)
            base_ratio: base pruning ratio before adjustments
            min_clip: lower bound for safety
            max_clip: upper bound for safety

        Returns:
            Dict[layer_name, ratio]
        """
        conv_layers = ModelUtils.get_conv_layers(model)
        total_layers = max(len(conv_layers), 1)

        # Precompute complexity factor
        complexity_factor = 1.0 + (float(dcs) - 0.5) * 0.4

        ratios: Dict[str, float] = {}
        for idx, layer in enumerate(conv_layers):
            depth_factor = 1.0 - (idx / total_layers) * 0.3
            lcs_val = float(lcs.get(layer.name, 0.0))
            sens = self._sigmoid(-lcs_val)
            r = base_ratio * depth_factor * complexity_factor * sens
            r = float(np.clip(r, min_clip, max_clip))
            ratios[layer.name] = r
        return ratios

    def adaptive_pruning_ratio(
        self,
        layer_idx: int,
        total_layers: int,
        dcs: float,
        lcs_value: float,
        base_ratio: float = 0.5,
        min_clip: float = 0.2,
        max_clip: float = 0.8,
    ) -> float:
        """
        Convenience helper computing a single layer's adaptive pruning ratio.
        Mirrors the formula discussed in design.
        """
        depth_factor = 1.0 - (float(layer_idx) / max(total_layers, 1)) * 0.3
        complexity_factor = 1.0 + (float(dcs) - 0.5) * 0.4
        sensitivity_factor = self._sigmoid(float(-lcs_value))
        r = base_ratio * depth_factor * complexity_factor * sensitivity_factor
        return float(np.clip(r, min_clip, max_clip))

    def compute_adaptive_pruning_ratios_softmax(
        self,
        model: tf.keras.Model,
        dcs: float,
        lcs: Dict[str, float],
        base_ratio: float = 0.5,
        tau: float = 0.7,
        weights: Optional[Dict[str, float]] = None,
        invert_lcs: bool = True,
        min_clip: float = 0.2,
        max_clip: float = 0.8,
    ) -> Dict[str, float]:
        """
        Allocate per-layer prune ratios via softmax over priorities, preserving
        the average prune ratio (≈ base_ratio) while favoring early and sensitive layers.

        Priority per layer l:
            k_l = w_d*(1 - depth_norm_l) + w_s*sigmoid(±z_lcs_l) + w_c*(1 - DCS)
        q = softmax(k / tau)
        keep_avg = 1 - base_ratio
        keep_l = keep_avg * q / mean(q)   # ensures mean(keep_l) ≈ keep_avg
        ratio_l = clip(1 - keep_l, min_clip, max_clip)

        Notes:
        - By default, uses sigmoid(-z_lcs) so "more sensitive ⇒ prune less".
        - The (1 - DCS) term is constant across layers and cancels in softmax;
          keep it for API symmetry and adjust `base_ratio` externally with DCS if desired.
        - Extreme priorities may cause per-layer keep > 1 before clipping if tau is very small; 
          choose a moderate tau to avoid heavy saturation.

        Args:
            model: Keras model
            dcs: dataset complexity score in [0,1]
            lcs: dict mapping layer_name -> sensitivity (z-scored recommended)
            base_ratio: target average prune ratio across layers
            tau: softmax temperature (lower => sharper)
            weights: optional dict {'depth','sensitivity','data'} summing to ~1
            invert_lcs: if True, uses sigmoid(-lcs); else sigmoid(lcs)
            min_clip, max_clip: safety bounds for per-layer ratios

        Returns:
            Dict[layer_name, prune_ratio]
        """
        conv_layers = ModelUtils.get_conv_layers(model)
        L = len(conv_layers)
        if L == 0:
            return {}

        # Normalize weights
        w = {'depth': 0.5, 'sensitivity': 0.4, 'data': 0.1}
        if weights:
            for k in ('depth', 'sensitivity', 'data'):
                if k in weights:
                    w[k] = float(weights[k])
        s = sum(w.values()) or 1.0
        w = {k: v / s for k, v in w.items()}

        p_data = 1.0 - float(np.clip(dcs, 0.0, 1.0))

        ks = []
        names = []
        for idx, layer in enumerate(conv_layers):
            depth_norm = float(idx) / float(max(L - 1, 1))  # 0..1
            p_depth = 1.0 - depth_norm
            z = float(lcs.get(layer.name, 0.0))
            sens = 1.0 / (1.0 + np.exp(z if not invert_lcs else -z))
            k = w['depth'] * p_depth + w['sensitivity'] * sens + w['data'] * p_data
            ks.append(k)
            names.append(layer.name)

        # Stable softmax with temperature
        k_arr = np.asarray(ks, dtype=np.float64)
        k_arr = k_arr - np.max(k_arr)
        logits = k_arr / max(float(tau), 1e-6)
        exps = np.exp(logits)
        q = exps / (np.sum(exps) + 1e-12)

        keep_avg = float(np.clip(1.0 - base_ratio, 0.0, 1.0))
        mean_q = float(np.mean(q))  # = 1/L
        keep = keep_avg * (q / (mean_q + 1e-12))

        ratios: Dict[str, float] = {}
        # early_break = max(1, int(np.ceil(L * 0.25)))
        # mid_break = max(early_break, int(np.ceil(L * 0.5)))
        for i, name in enumerate(names):
            r = float(np.clip(1.0 - keep[i], min_clip, max_clip))
            # if i < early_break:
            #     r = min(r, 0.4)
            # elif i < mid_break:
            #     r = min(r, 0.55)
            ratios[name] = r

        return ratios
