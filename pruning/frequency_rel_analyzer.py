import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, List, Sequence, Optional


class FrequencyRelevanceAnalyzer:
    def __init__(self, config):
        self.config = config

    def build_frequency_relevance_net(
        self,
        hidden_units: Sequence[int] = (64, 32),
        input_dim: int = 3,
        architecture: Optional[str] = None,
        num_classes: int = 3,
    ) -> tf.keras.Model:
        """
        Build the FRN backbone.

        Args:
            hidden_units: base hidden sizes.
            input_dim: number of input features.
            architecture: 'residual' for residual MLP, 'dense' for vanilla stack.
        """
        arch = (architecture or getattr(self.config, "frn_architecture", "residual")).lower()
        if arch not in {"residual", "dense"}:
            arch = "residual"
        units_seq = tuple(int(u) for u in hidden_units if int(u) > 0) or (64, 32)
        use_batchnorm = bool(getattr(self.config, "frn_use_batchnorm", True))
        dropout_cfg = getattr(self.config, "frn_dropout_rate", 0.05)
        if isinstance(dropout_cfg, (list, tuple)):
            dropout_rates = [float(rate) for rate in dropout_cfg] or [0.05]
        else:
            dropout_rates = [float(dropout_cfg)]
        if len(dropout_rates) < len(units_seq):
            dropout_rates.extend([dropout_rates[-1]] * (len(units_seq) - len(dropout_rates)))
        else:
            dropout_rates = dropout_rates[: len(units_seq)]
        if len(units_seq) > 0:
            final_rate = dropout_rates[-1]
            dropout_rates = [0.0] * (len(units_seq) - 1) + [final_rate]

        inputs = tf.keras.Input(shape=(input_dim,), name="band_features")
        x = inputs

        if arch == "dense":
            for idx, units in enumerate(units_seq, start=1):
                layer_name = f"frn_dense_{idx}"
                x = tf.keras.layers.Dense(
                    units,
                    use_bias=not use_batchnorm,
                    kernel_initializer="he_normal",
                    name=layer_name,
                )(x)
                if use_batchnorm:
                    x = tf.keras.layers.BatchNormalization(name=f"{layer_name}_bn")(x)
                x = tf.keras.layers.Activation("relu", name=f"{layer_name}_relu")(x)
                drop_rate = dropout_rates[idx - 1]
                if drop_rate > 0.0:
                    x = tf.keras.layers.Dropout(drop_rate, name=f"{layer_name}_dropout")(x)
        else:
            prev_units = input_dim
            for idx, units in enumerate(units_seq, start=1):
                layer_name = f"frn_residual_{idx}"
                residual = x
                y = tf.keras.layers.Dense(
                    units,
                    use_bias=not use_batchnorm,
                    kernel_initializer="he_normal",
                    name=f"{layer_name}_dense1",
                )(x)
                if use_batchnorm:
                    y = tf.keras.layers.BatchNormalization(name=f"{layer_name}_bn1")(y)
                y = tf.keras.layers.Activation("relu", name=f"{layer_name}_relu1")(y)
                drop_rate = dropout_rates[idx - 1]
                if drop_rate > 0.0:
                    y = tf.keras.layers.Dropout(drop_rate, name=f"{layer_name}_drop1")(y)
                y = tf.keras.layers.Dense(
                    units,
                    use_bias=not use_batchnorm,
                    kernel_initializer="he_normal",
                    name=f"{layer_name}_dense2",
                )(y)
                if use_batchnorm:
                    y = tf.keras.layers.BatchNormalization(name=f"{layer_name}_bn2")(y)
                if prev_units != units:
                    residual = tf.keras.layers.Dense(
                        units,
                        use_bias=not use_batchnorm,
                        kernel_initializer="he_normal",
                        name=f"{layer_name}_proj",
                    )(residual)
                    if use_batchnorm:
                        residual = tf.keras.layers.BatchNormalization(name=f"{layer_name}_proj_bn")(residual)
                x = tf.keras.layers.Add(name=f"{layer_name}_add")([residual, y])
                x = tf.keras.layers.Activation("relu", name=f"{layer_name}_out")(x)
                if drop_rate > 0.0:
                    x = tf.keras.layers.Dropout(drop_rate, name=f"{layer_name}_dropout")(x)
                prev_units = units
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="frn_logits")(x)
        model_name = "FRN_residual" if arch == "residual" else "FRN_dense"
        return tf.keras.Model(inputs, outputs, name=model_name)

    # ---------------------------- utilities ---------------------------- #
    def dct2_ortho(self, x: tf.Tensor) -> tf.Tensor:
        """2D DCT-II with orthonormal scaling over the spatial axes."""
        x_h = tf.transpose(x, [1, 2, 3, 0])
        y_h = tf.signal.dct(x_h, type=2, norm="ortho")
        y = tf.transpose(y_h, [3, 0, 1, 2])
        x_w = tf.transpose(y, [0, 2, 3, 1])
        y_w = tf.signal.dct(x_w, type=2, norm="ortho")
        return tf.transpose(y_w, [0, 3, 1, 2])  # [H, W, Cin, Cout]

    def idct2_ortho(self, x: tf.Tensor) -> tf.Tensor:
        """Inverse 2D DCT (type-II) with orthonormal scaling."""
        x_w = tf.transpose(x, [0, 2, 3, 1])
        y_w = tf.signal.idct(x_w, type=2, norm="ortho")
        y = tf.transpose(y_w, [0, 3, 1, 2])
        x_h = tf.transpose(y, [1, 2, 3, 0])
        y_h = tf.signal.idct(x_h, type=2, norm="ortho")
        return tf.transpose(y_h, [3, 0, 1, 2])

    def create_mask(self, h: int, w: int, lo: float, hi: float) -> np.ndarray:
        u = (np.arange(h) / (h - 1)) if h > 1 else np.zeros(h)
        v = (np.arange(w) / (w - 1)) if w > 1 else np.zeros(w)
        U, V = np.meshgrid(u, v, indexing="ij")
        r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
        return ((r >= lo) & (r < hi)).astype(np.float32)

    band_defs = {"low": (0.0, 0.25), "mid": (0.25, 0.5), "high": (0.5, 1.0)}

    def band_energies_from_kernel(self, kernel_np: np.ndarray):
        """Return band energies, ratios, and totals for a Conv2D kernel."""
        h, w, _, cout = kernel_np.shape
        k_tf = tf.convert_to_tensor(kernel_np, tf.float32)
        X = self.dct2_ortho(k_tf)  # [H,W,Cin,Cout]
        totals = np.zeros((cout,), np.float64)
        band_E: Dict[str, np.ndarray] = {}
        for band, (lo, hi) in self.band_defs.items():
            mask = self.create_mask(h, w, lo, hi)
            mask_tf = tf.reshape(tf.convert_to_tensor(mask, tf.float32), (h, w, 1, 1))
            coeffs = X * mask_tf
            energy = tf.sqrt(tf.reduce_sum(tf.square(coeffs), axis=(0, 1, 2)))
            energy_np = energy.numpy().astype(np.float64)
            band_E[band] = energy_np
            totals += energy_np
        eps = 1e-12
        ratios = {band: band_E[band] / (totals + eps) for band in band_E}
        return band_E, ratios, totals

    def band_grad_energies_from_batch(
        self,
        model: tf.keras.Model,
        layer: tf.keras.layers.Layer,
        xb: tf.Tensor,
        yb: tf.Tensor,
        from_logits: bool,
    ) -> Dict[str, np.ndarray]:
        """Gradient-based Taylor energies split per frequency band for one batch."""
        sparse = (yb.ndim == 1) or (yb.ndim == 2 and yb.shape[-1] == 1)
        loss_fn = (
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            if sparse
            else tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
        )
        with tf.GradientTape() as tape:
            preds = model(xb, training=True)
            target = (
                tf.reshape(tf.cast(yb, tf.int32), [-1])
                if sparse
                else tf.cast(yb, tf.float32)
            )
            loss = loss_fn(target, preds)

        kernel_var = None
        for v in layer.trainable_variables:
            if "kernel" in v.name:
                kernel_var = v
                break
        if kernel_var is None:
            return {band: np.zeros((layer.filters,), np.float64) for band in self.band_defs}

        grad = tape.gradient(loss, kernel_var)
        if grad is None:
            return {band: np.zeros((layer.filters,), np.float64) for band in self.band_defs}

        G = self.dct2_ortho(grad)
        band_G: Dict[str, np.ndarray] = {}
        for band, (lo, hi) in self.band_defs.items():
            mask = self.create_mask(G.shape[0], G.shape[1], lo, hi)
            mask_tf = tf.reshape(tf.convert_to_tensor(mask, tf.float32), G.shape[:2] + (1, 1))
            coeffs = G * mask_tf
            energy = tf.sqrt(tf.reduce_sum(tf.square(coeffs), axis=(0, 1, 2)))
            band_G[band] = energy.numpy().astype(np.float64)
        return band_G

    # ---------------------------- dataset builders ---------------------------- #
    def build_frn_training_set(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        max_batches: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy smaller builder using only frequency ratios."""
        last = model.layers[-1]
        from_logits = not isinstance(last, tf.keras.layers.Softmax) and \
            getattr(last, "activation", None) != tf.keras.activations.softmax
        binary_targets = bool(getattr(self.config, "frn_low_vs_rest", False))
        num_classes = 2 if binary_targets else 3

        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            return np.zeros((0, num_classes), np.float32), np.zeros((0, num_classes), np.float32)

        grad_acc: Dict[str, Dict[str, np.ndarray]] = {l.name: None for l in conv_layers}

        for xb, yb in dataset.take(max_batches):
            xb = tf.cast(xb, tf.float32)
            for layer in conv_layers:
                band_grads = self.band_grad_energies_from_batch(model, layer, xb, yb, from_logits)
                if grad_acc[layer.name] is None:
                    grad_acc[layer.name] = {band: val.copy() for band, val in band_grads.items()}
                else:
                    for band in self.band_defs:
                        grad_acc[layer.name][band] += band_grads[band]

        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        for layer in conv_layers:
            weights = layer.get_weights()
            if not weights:
                continue
            kernel = weights[0]
            band_E, ratios, _ = self.band_energies_from_kernel(kernel)
            cout = kernel.shape[-1]
            grad_band = grad_acc[layer.name] or {
                band: np.zeros((cout,), np.float64) for band in self.band_defs
            }

            grad_tot = np.zeros((cout,), np.float64)
            for band in self.band_defs:
                grad_tot += grad_band[band]
            eps = 1e-12
            Y = np.stack(
                [
                    grad_band["low"] / (grad_tot + eps),
                    grad_band["mid"] / (grad_tot + eps),
                    grad_band["high"] / (grad_tot + eps),
                ],
                axis=1,
            )
            if binary_targets:
                low_col = Y[:, 0:1]
                other_col = np.sum(Y[:, 1:], axis=1, keepdims=True)
                Y = np.concatenate([low_col, other_col], axis=1)
                Y = Y / (np.sum(Y, axis=1, keepdims=True) + eps)
            X = np.stack(
                [ratios["low"], ratios["mid"], ratios["high"]],
                axis=1,
            )
            mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
            if not np.any(mask):
                continue
            X_list.append(X[mask])
            Y_list.append(Y[mask])

        if not X_list:
            return np.zeros((0, num_classes), np.float32), np.zeros((0, num_classes), np.float32)

        X_all = np.concatenate(X_list, axis=0).astype(np.float32)
        Y_all = np.concatenate(Y_list, axis=0).astype(np.float32)
        return X_all, Y_all

    def build_frn_training_set_stronger(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        band_defs: Dict[str, Tuple[float, float]],
        max_batches: int = 200,
        ema_beta: float = 0.8,
        sharpen_gamma: float = 2.0,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce FRN training data enriched with activation statistics.

        Returns:
            X_all: [N, F] features (band ratios + activation stats).
            Y_all: [N, C] normalized Taylor-based targets.
            W_all: [N]    sample weights for band balancing.
        """
        binary_targets = bool(getattr(self.config, "frn_low_vs_rest", False))
        num_classes = 2 if binary_targets else 3

        def dct2_local(x: tf.Tensor) -> tf.Tensor:
            return self.dct2_ortho(x)

        def create_mask_local(h: int, w: int, lo: float, hi: float) -> np.ndarray:
            return self.create_mask(h, w, lo, hi)

        last = model.layers[-1]
        from_logits = not isinstance(last, tf.keras.layers.Softmax) and \
            getattr(last, "activation", None) != tf.keras.activations.softmax

        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            return (
                np.zeros((0, num_classes), np.float32),
                np.zeros((0, num_classes), np.float32),
                np.zeros((0,), np.float32),
            )

        activation_model = tf.keras.Model(
            inputs=model.input,
            outputs=[layer.output for layer in conv_layers],
        )

        layer_masks: Dict[str, Dict[str, tf.Tensor]] = {}
        taylor_acc: Dict[str, Dict[str, np.ndarray]] = {}
        energy_acc: Dict[str, Dict[str, np.ndarray]] = {}
        totals_energy: Dict[str, np.ndarray] = {}
        act_abs_acc: Dict[str, np.ndarray] = {}
        act_sq_acc: Dict[str, np.ndarray] = {}
        act_count: Dict[str, float] = {}

        for layer in conv_layers:
            weights = layer.get_weights()
            if not weights:
                continue
            kernel = weights[0]
            if kernel.ndim != 4:
                continue
            h, w, _, cout = kernel.shape

            mask_dict: Dict[str, tf.Tensor] = {}
            for band, (lo, hi) in band_defs.items():
                mask = create_mask_local(h, w, lo, hi)
                mask_dict[band] = tf.reshape(tf.convert_to_tensor(mask, tf.float32), (h, w, 1, 1))
            layer_masks[layer.name] = mask_dict

            kernel_tf = tf.convert_to_tensor(kernel, tf.float32)
            kernel_dct = dct2_local(kernel_tf)

            taylor_acc[layer.name] = {
                band: np.zeros((cout,), np.float64) for band in band_defs
            }
            energy_acc[layer.name] = {}
            totals_energy[layer.name] = np.zeros((cout,), np.float64)
            for band, mask_tf in mask_dict.items():
                coeffs = kernel_dct * mask_tf
                energy = tf.sqrt(tf.reduce_sum(tf.square(coeffs), axis=(0, 1, 2)))
                energy_np = energy.numpy().astype(np.float64)
                energy_acc[layer.name][band] = energy_np
                totals_energy[layer.name] += energy_np

            act_abs_acc[layer.name] = np.zeros((cout,), np.float64)
            act_sq_acc[layer.name] = np.zeros((cout,), np.float64)
            act_count[layer.name] = 0.0

        loss_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        loss_cat = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)

        batches = 0
        for xb, yb in dataset.take(max_batches):
            xb = tf.cast(xb, tf.float32)
            sparse = (yb.ndim == 1) or (yb.ndim == 2 and yb.shape[-1] == 1)

            activations = activation_model(xb, training=False)
            if isinstance(activations, tf.Tensor):
                activations = [activations]
            for layer, act in zip(conv_layers, activations):
                act_np = act.numpy()
                abs_sum = np.sum(np.abs(act_np), axis=(0, 1, 2))
                sq_sum = np.sum(np.square(act_np), axis=(0, 1, 2))
                count = float(act_np.shape[0] * act_np.shape[1] * act_np.shape[2])
                act_abs_acc[layer.name] += abs_sum.astype(np.float64)
                act_sq_acc[layer.name] += sq_sum.astype(np.float64)
                act_count[layer.name] += count

            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                if sparse:
                    y_true = tf.reshape(tf.cast(yb, tf.int32), [-1])
                    loss = loss_sparse(y_true, preds)
                else:
                    y_true = tf.cast(yb, tf.float32)
                    loss = loss_cat(y_true, preds)

            grads = tape.gradient(loss, model.trainable_variables)

            def _var_key(var):
                if hasattr(var, "experimental_ref") and callable(getattr(var, "experimental_ref")):
                    try:
                        return var.experimental_ref()
                    except Exception:
                        pass
                if hasattr(var, "ref") and callable(getattr(var, "ref")):
                    try:
                        return var.ref()
                    except Exception:
                        pass
                return id(var)

            var_to_grad = {_var_key(tv): g for tv, g in zip(model.trainable_variables, grads)}

            for layer in conv_layers:
                grad_tensor = None
                for variable in layer.trainable_variables:
                    if "kernel" in variable.name:
                        grad_tensor = var_to_grad.get(_var_key(variable), None)
                        break
                if grad_tensor is None:
                    continue

                kernel_tf = tf.convert_to_tensor(layer.get_weights()[0], tf.float32)
                taylor = tf.abs(kernel_tf * grad_tensor)
                taylor_dct = dct2_local(taylor)
                for band, mask_tf in layer_masks[layer.name].items():
                    coeffs = taylor_dct * mask_tf
                    energy = tf.sqrt(tf.reduce_sum(tf.square(coeffs), axis=(0, 1, 2)))
                    energy_np = energy.numpy().astype(np.float64)
                    if ema_beta > 0:
                        taylor_acc[layer.name][band] = (
                            ema_beta * taylor_acc[layer.name][band]
                            + (1.0 - ema_beta) * energy_np
                        )
                    else:
                        taylor_acc[layer.name][band] += energy_np

            batches += 1

        if batches == 0:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0,), np.float32),
            )

        def _normalize(arr: np.ndarray) -> np.ndarray:
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std < 1e-12:
                return np.zeros_like(arr, dtype=np.float64)
            return (arr - mean) / (std + 1e-12)

        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        argmax_list: List[np.ndarray] = []

        for layer in conv_layers:
            cout = totals_energy[layer.name].shape[0]
            total_energy = totals_energy[layer.name] + eps

            ratios = {
                band: energy_acc[layer.name][band] / total_energy
                for band in band_defs
            }

            taylor_tot = np.zeros((cout,), np.float64)
            for band in band_defs:
                taylor_tot += taylor_acc[layer.name][band]

            targets = np.stack(
                [
                    np.power(taylor_acc[layer.name]["low"] + eps, 1.0 / sharpen_gamma),
                    np.power(taylor_acc[layer.name]["mid"] + eps, 1.0 / sharpen_gamma),
                    np.power(taylor_acc[layer.name]["high"] + eps, 1.0 / sharpen_gamma),
                ],
                axis=1,
            )
            targets = targets / (np.sum(targets, axis=1, keepdims=True) + eps)
            if binary_targets:
                low_col = targets[:, 0:1]
                other_col = np.sum(targets[:, 1:], axis=1, keepdims=True)
                combined = np.concatenate([low_col, other_col], axis=1)
                targets = combined / (np.sum(combined, axis=1, keepdims=True) + eps)

            count = max(act_count[layer.name], eps)
            mean_abs = act_abs_acc[layer.name] / count
            mean_sq = act_sq_acc[layer.name] / count
            variance = np.maximum(mean_sq - np.square(mean_abs), 0.0)
            std = np.sqrt(variance)
            mean_abs_norm = _normalize(mean_abs)
            std_norm = _normalize(std)

            features = np.stack(
                [
                    ratios["low"],
                    ratios["mid"],
                    ratios["high"],
                ],
                axis=1,
            )
            if bool(getattr(self.config, "frn_use_activation_features", True)):
                extra = np.stack([mean_abs_norm, std_norm], axis=1)
                features = np.concatenate([features, extra], axis=1)

            mask = np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1)
            if not np.any(mask):
                continue

            X_list.append(features[mask].astype(np.float32))
            Y_list.append(targets[mask].astype(np.float32))
            argmax_list.append(np.argmax(targets[mask], axis=1))

        if not X_list:
            return (
                np.zeros((0, num_classes), np.float32),
                np.zeros((0, num_classes), np.float32),
                np.zeros((0,), np.float32),
            )

        X_all = np.concatenate(X_list, axis=0)
        Y_all = np.concatenate(Y_list, axis=0)
        argmax_all = np.concatenate(argmax_list, axis=0)

        counts = np.bincount(argmax_all, minlength=num_classes).astype(np.float32)
        inv_freq = 1.0 / (counts + 1e-6)
        inv_freq = inv_freq / np.sum(inv_freq) * 3.0
        w_all = inv_freq[argmax_all]
        max_weight = float(getattr(self.config, "frn_weight_clip", 3.0))
        if max_weight > 0:
            w_all = np.minimum(w_all, max_weight)
        return X_all.astype(np.float32), Y_all.astype(np.float32), w_all.astype(np.float32)
