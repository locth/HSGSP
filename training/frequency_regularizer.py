import torch
import torch.nn as nn
from typing import Dict, Iterable, Tuple, Union, Optional

def _dct2(x: torch.Tensor) -> torch.Tensor:
    """Apply orthonormal 2-D DCT (type-II) over the last two axes."""
    return torch.fft.fft2(x, norm="ortho")

def compute_spectral_entropy(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the mean spectral entropy of 4-D activations.
    """
    shape = features.shape
    height, width = shape[1], shape[2]
    features = features.view(-1, height, width)
    coeffs = _dct2(features)
    energy = torch.abs(coeffs)**2  # Corrected: use magnitude squared
    total = energy.sum(dim=[1, 2], keepdim=True) + eps
    probs = energy / total
    entropy = - (probs * torch.log(probs.clamp(min=eps))).sum(dim=[1, 2])
    return entropy.mean()

def frequency_entropy_loss(
    features: torch.Tensor,
    target_entropy: Union[float, torch.Tensor],
    beta: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Penalize deviation from a desired spectral entropy.
    """
    entropy = compute_spectral_entropy(features, eps=eps)
    target = torch.tensor(target_entropy, dtype=torch.float32)
    loss = beta * (entropy - target).pow(2)
    return loss, entropy

class SpectralEntropyRegularizer:
    """
    Helper that measures layer-wise spectral entropy and returns a weighted loss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: Iterable[str],
        target_entropies: Dict[str, float],
        beta: float = 0.01,
        layer_weights: Optional[Dict[str, float]] = None,
    ):
        self.layer_names = list(layer_names)
        self.beta = float(beta)
        self.layer_weights = layer_weights or {}
        self.targets = torch.tensor([target_entropies.get(name, 0.0) for name in self.layer_names], dtype=torch.float32)
        self.extractor = self._build_extractor(model)

    def _build_extractor(self, model: torch.nn.Module) -> nn.Module:
        # In PyTorch, use hooks to extract
        return model  # Placeholder, use hooks in call

    def __call__(self, inputs: torch.Tensor, training: bool = False):
        feats = {}  # Use hooks to get feats
        hooks = []
        def hook_fn(name):
            def hook(m, i, o):
                feats[name] = o
            return hook

        for name in self.layer_names:
            module = dict(self.extractor.named_modules())[name]
            hooks.append(module.register_forward_hook(hook_fn(name)))

        self.extractor(inputs)  # Corrected: use self.extractor instead of undefined model

        for h in hooks:
            h.remove()

        total_loss = torch.zeros((), dtype=torch.float32)
        entropy_map: Dict[str, torch.Tensor] = {}
        for idx, layer_name in enumerate(self.layer_names):
            feat = feats[layer_name]
            layer_scale = float(self.layer_weights.get(layer_name, 1.0))
            layer_beta = layer_scale * self.beta
            layer_loss, entropy = frequency_entropy_loss(
                feat,
                target_entropy=self.targets[idx],
                beta=layer_beta,
            )
            total_loss += layer_loss
            entropy_map[layer_name] = entropy
        return total_loss, entropy_map


class FrequencyRegularizedModel(nn.Module):
    """
    Wraps an existing model and injects spectral-entropy loss into train_step.
    """

    def __init__(
        self,
        base_model: nn.Module,
        spectral_regularizer: SpectralEntropyRegularizer,
    ):
        super().__init__()
        self.base_model = base_model
        self.spectral_regularizer = spectral_regularizer

    def forward(self, inputs, training=False):
        return self.base_model(inputs)

    def train_step(self, data, optimizer, criterion):
        x, y = data
        optimizer.zero_grad()
        y_pred = self.base_model(x)
        base_loss = criterion(y_pred, y)
        freq_loss, _ = self.spectral_regularizer(x)
        total_loss = base_loss + freq_loss
        total_loss.backward()
        optimizer.step()
        return total_loss, base_loss, freq_loss