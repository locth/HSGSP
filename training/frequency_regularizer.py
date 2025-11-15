import tensorflow as tf
from typing import Dict, Iterable, Tuple, Union, Optional


def _dct2(x: tf.Tensor) -> tf.Tensor:
    """Apply orthonormal 2-D DCT (type-II) over the last two axes."""
    x = tf.signal.dct(x, type=2, norm="ortho")
    x = tf.transpose(x, perm=[0, 2, 1])
    x = tf.signal.dct(x, type=2, norm="ortho")
    return tf.transpose(x, perm=[0, 2, 1])


def _reshape_spatial(features: tf.Tensor) -> tf.Tensor:
    """Flatten batch/channel dims so DCT runs per (sample, channel)."""
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    shape = tf.shape(features)
    height = shape[1]
    width = shape[2]
    return tf.reshape(features, [-1, height, width]), height, width


def compute_spectral_entropy(features: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Compute the mean spectral entropy of 4-D activations.

    Args:
        features: Tensor shaped [B, H, W, C].
        eps: Numerical stability constant.

    Returns:
        Scalar tensor with the average entropy across B*C channels.
    """
    flat, height, width = _reshape_spatial(features)
    coeffs = _dct2(flat)
    energy = tf.square(coeffs)
    total = tf.reduce_sum(energy, axis=[1, 2], keepdims=True) + eps
    probs = energy / total
    entropy = -tf.reduce_sum(probs * tf.math.log(tf.maximum(probs, eps)), axis=[1, 2])
    return tf.reduce_mean(entropy)


def frequency_entropy_loss(
    features: tf.Tensor,
    target_entropy: Union[float, tf.Tensor],
    beta: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Penalize deviation from a desired spectral entropy.

    Args:
        features: Activations [B, H, W, C].
        target_entropy: Desired entropy level (scalar).
        beta: Strength of the penalty term.
        eps: Numeric stability constant.

    Returns:
        Tuple of (scalar loss, current entropy).
    """
    entropy = compute_spectral_entropy(features, eps=eps)
    target = tf.convert_to_tensor(target_entropy, dtype=tf.float32)
    loss = beta * tf.square(entropy - target)
    return loss, entropy


class SpectralEntropyRegularizer:
    """
    Helper that measures layer-wise spectral entropy and returns a weighted loss.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        layer_names: Iterable[str],
        target_entropies: Dict[str, float],
        beta: float = 0.01,
        layer_weights: Optional[Dict[str, float]] = None,
    ):
        self.layer_names = list(layer_names)
        self.beta = float(beta)
        self.layer_weights = layer_weights or {}
        outputs = []
        targets = []
        for name in self.layer_names:
            layer = model.get_layer(name)
            outputs.append(layer.output)
            targets.append(float(target_entropies.get(name, 0.0)))
        self._targets = tf.constant(targets, dtype=tf.float32)
        self._extractor = tf.keras.Model(inputs=model.inputs, outputs=outputs)

    def __call__(self, inputs: tf.Tensor, training: bool = False):
        feats = self._extractor(inputs, training=training)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        total_loss = tf.zeros((), dtype=tf.float32)
        entropy_map: Dict[str, tf.Tensor] = {}
        for idx, (layer_name, feat) in enumerate(zip(self.layer_names, feats)):
            layer_scale = float(self.layer_weights.get(layer_name, 1.0))
            layer_beta = layer_scale * self.beta
            layer_loss, entropy = frequency_entropy_loss(
                feat,
                target_entropy=self._targets[idx],
                beta=layer_beta,
            )
            total_loss += layer_loss
            entropy_map[layer_name] = entropy
        return total_loss, entropy_map


class FrequencyRegularizedModel(tf.keras.Model):
    """
    Wraps an existing model and injects spectral-entropy loss into train_step.
    """

    def __init__(
        self,
        base_model: tf.keras.Model,
        spectral_regularizer: SpectralEntropyRegularizer,
    ):
        super().__init__(inputs=base_model.inputs, outputs=base_model.outputs, name=base_model.name)
        self._base_model = base_model
        self._spectral_regularizer = spectral_regularizer

    def call(self, inputs, training=False):
        return self._base_model(inputs, training=training)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self._base_model(x, training=True)
            base_loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self._base_model.losses,
            )
            freq_loss, entropy_map = self._spectral_regularizer(x, training=True)
            total_loss = base_loss + freq_loss

        grads = tape.gradient(total_loss, self._base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self._base_model.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        logs = {m.name: m.result() for m in self.metrics}
        for layer_name, entropy in entropy_map.items():
            logs[f"entropy_{layer_name}"] = entropy
        logs["freq_loss"] = freq_loss
        return logs
