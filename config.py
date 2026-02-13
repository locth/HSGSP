import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from datetime import datetime

@dataclass
class Config:
    """Configuration optimized for max accuracy on VGG16 Tiny ImageNet"""

    # ========== DATA CONFIGURATION ==========
    task: str = 'imagenet'  # Đặt mặc định cho Tiny ImageNet
    validation_split: float = 0.1  # Giữ để val set ~10k samples
    data_augmentation: bool = True
    batch_size: int = 128  # Giảm xuống 128 để an toàn trên single GPU; tăng 256 nếu multi-GPU

    # Dataset specific
    num_classes_cifar10: int = 10
    num_classes_cifar100: int = 100
    num_classes_imagenet: int = 200
    input_shape_cifar10: Tuple[int, int, int] = (3, 32, 32)
    input_shape_cifar100: Tuple[int, int, int] = (3, 32, 32)
    input_shape_imagenet: Tuple[int, int, int] = (3, 64, 64)

    # ========== TRAINING CONFIGURATION ==========
    default_epochs: int = 200  # Tăng để converge tốt hơn (benchmark: 200-300 cho ~70%+ acc)
    initial_lr: float = 1e-3  # Thích hợp cho Adam + cosine
    pruned_growth_lr: float = 1e-4
    min_lr: float = 1e-5
    momentum: float = 0.9  # Chỉ dùng nếu SGD; không cần cho Adam
    optimizer: str = 'adam'  # Switch sang Adam cho acc cao hơn ~5-10% so với SGD

    # Learning rate schedule
    lr_schedule: str = 'cosine'  # Cosine decay cho smooth và higher final acc
    lr_warmup_epochs: int = 5
    lr_decay_rate: float = 0.1  # Không cần cho cosine, nhưng giữ cho fallback
    lr_decay_steps: int = None  # Không dùng cho cosine

    # Early Stopping
    early_stopping_patience: int = 20  # Tăng để run lâu hơn
    early_stopping_min_delta: float = 1e-4
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    reduce_lr_min_delta: float = 1e-3
    fine_tune_lr_schedule: str = 'cosine'
    fine_tune_exp_decay: float = 0.8
    fine_tune_cosine_min_factor: float = 0.05
    fine_tune_linear_end_factor: float = 0.1
    fine_tune_step_decay_rate: float = 0.5
    fine_tune_step_decay_epochs: int = 5

    # Regularization
    l2_regularization: float = 5e-4  # Tăng để chống overfit trên Tiny
    batch_norm_momentum: float = 0.99

    dropout_rate: float = 0.3  # Tăng nhẹ cho conv layers
    use_spatial_dropout: bool = False
    spatial_dropout_rate: float = 0.0
    fc_dropout_rate1: float = 0.5  # Standard cho FC layers trong VGG
    fc_dropout_rate2: float = 0.5

    weight_decay: float = 5e-4  # Match l2_reg

    # Label Smoothing
    label_smoothing: float = 0.1

    # ========== DISTILLATION CONFIGURATION ==========
    distill_alpha: float = 0.0  # Disable
    distill_temperature: float = 2.5

    # ========== PRUNING CONFIGURATION ==========
    frequency_bands: Dict[str, tuple] = None
    complexity_weights: Dict[str, float] = None
    max_global_pruning_ratio: float = 0.0  # Disable pruning
    min_global_keep: float = 1.0
    max_accuracy_drop: float = 0.0
    accuracy_guard_center: float = 0.92
    accuracy_guard_sharpness: float = 0.5

    simple_finetune_epochs: int = 20
    simple_finetune_lr: float = 1e-5

    # ========== AUGMENTATION CONFIGURATION ==========
    use_mixup: bool = True
    mixup_alpha: float = 0.2  # Thấp hơn để tránh over-aug
    mixup_prob: float = 0.5
    use_cutout: bool = True  # Sử dụng RandomErasing trong pipeline
    cutout_length: int = 8  # Adjust cho 64x64 images

    # ========== HYBRID BASELINE CONFIGURATION ==========
    hybrid_iterations: int = 11  # Disable
    hybrid_prune_fraction: float = 0.0
    hybrid_alpha: float = 0.5
    hybrid_kappa_beta: float = 0.1
    hybrid_initial_kappa_ratio: float = 0.5
    hybrid_mode: str = 'original'
    hybrid_min_filters: int = 8
    hybrid_finetune_epochs: int = 11
    hybrid_warmup_epochs: int = 0
    hybrid_warmup_lr: float = 2e-4
    hybrid_regrow_fraction: float = 0.0

    frequency_regularization_layers: int = 0  # Disable
    frequency_entropy_beta: float = 0.0
    frequency_entropy_target_batches: int = 8
    frequency_entropy_refresh_interval: int = 3
    frequency_entropy_layer_weights: Dict[str, float] = field(default_factory=dict)
   
    frn_validation_split: float = 0.2
    frn_plot_training: bool = True
    frn_feature_count: Optional[int] = 3
    frn_hidden_units: Tuple[int, ...] = (64, 32)
    frn_activation_batches: int = 512
    frn_min_validation_samples: int = 128
    frn_dropout_rate: float = 0.05
    frn_use_batchnorm: bool = False
    frn_use_activation_features: bool = False
    frn_low_vs_rest: bool = False
    frn_architecture: str = "dense"
    frn_ema_beta: float = 0.8
    frn_sharpen_gamma: float = 2.0
    frn_initial_lr: float = 1e-4
    frn_min_lr: float = 1e-5
    frn_epochs: int = 15
    frn_cosine_min_factor: float = 0.1
    frn_batch_size: int = 256
    frn_weight_clip: float = 2.0  

    # ========== PATHs CONFIGURATION ==========
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%d%m%Y_%H%M%S"))
    tensorboard_dir: str = field(init=False)
    results_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    models_dir: str = field(init=False)
    plots_dir: str = field(init=False)

    def __post_init__(self):
        if self.frequency_bands is None:
            self.frequency_bands = {
                'low': (0.0, 0.25),
                'mid': (0.25, 0.5),
                'high': (0.5, 1.0)
            }

        # Build directories using the provided task and a timestamp run_id
        exp_root = f"./EXPERIMENT/{self.run_id}_{self.task}"
        self.tensorboard_dir = os.path.join(exp_root, "tensorboard_logs")
        self.results_dir = os.path.join(exp_root, "results")
        self.logs_dir = os.path.join(exp_root, "logs")
        self.models_dir = os.path.join(exp_root, "models")
        self.plots_dir = os.path.join(exp_root, "plots")

        # Create directories
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        if self.task == 'imagenet':
            self.batch_size = 256  # Hoặc 128 nếu cần
            self.default_epochs = 200
            self.initial_lr = 1e-3
            self.lr_schedule = 'cosine'  # Thay step
            self.lr_decay_steps = None  # Không cần cho cosine
            self.l2_regularization = 5e-4
            self.weight_decay = 5e-4
            self.optimizer = 'adam'  # Thử thay SGD
            self.use_mixup = True
            self.mixup_alpha = 0.2
            self.use_cutout = True  # Implement RandomErasing
            self.cutout_length = 8  # Adjust cho 64x64