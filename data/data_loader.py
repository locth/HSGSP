import os
import shutil
import zipfile
import numpy as np
import tensorflow as tf
from collections import defaultdict
from typing import Dict, List, Tuple

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

class DataLoader:
    """Unified dataset loader for HSGSP project"""

    def __init__(self, config):
        self.config = config

    def _seed(self) -> int:
        return int(getattr(self.config, "seed", SEED))

    @staticmethod
    def _standardize_batch(
        images: tf.Tensor,
        labels: tf.Tensor,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply per-channel mean/std normalization on a batched image tensor."""
        mean_tensor = tf.constant(mean, dtype=images.dtype)
        std_tensor = tf.constant(std, dtype=images.dtype)
        mean_tensor = tf.reshape(mean_tensor, [1, 1, 1, 3])
        std_tensor = tf.reshape(std_tensor, [1, 1, 1, 3])
        images = (images - mean_tensor) / std_tensor
        return images, labels

    def _apply_mixup(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        use_mixup = bool(getattr(self.config, "use_mixup", False))
        mixup_alpha = float(getattr(self.config, "mixup_alpha", 0.0))
        mixup_prob = float(getattr(self.config, "mixup_prob", 0.0))
        if not use_mixup or mixup_alpha <= 0.0 or mixup_prob <= 0.0:
            return dataset

        def _mixup_batch(images, labels):
            labels = tf.cast(labels, tf.float32)
            def _do_mixup():
                alpha = tf.constant(mixup_alpha, dtype=tf.float32)
                gamma1 = tf.random.gamma(shape=[1], alpha=alpha, beta=1.0)[0]
                gamma2 = tf.random.gamma(shape=[1], alpha=alpha, beta=1.0)[0]
                lam = gamma1 / (gamma1 + gamma2)
                lam = tf.cast(lam, images.dtype)
                batch_size = tf.shape(images)[0]
                indices = tf.random.shuffle(tf.range(batch_size))
                shuffled_images = tf.gather(images, indices)
                shuffled_labels = tf.gather(labels, indices)
                mixed_images = lam * images + (1.0 - lam) * shuffled_images
                mixed_labels = lam * labels + (1.0 - lam) * shuffled_labels
                return mixed_images, mixed_labels

            return tf.cond(
                tf.random.uniform([], dtype=tf.float32) < mixup_prob,
                _do_mixup,
                lambda: (images, labels),
            )

        return dataset.map(_mixup_batch, num_parallel_calls=AUTOTUNE)

    @staticmethod
    def _is_image_file(filename: str) -> bool:
        lower = filename.lower()
        return lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")

    def _stratified_split_samples(
        self,
        samples: List[Tuple[str, int]],
        val_fraction: float,
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Split samples into train/val while preserving per-class balance."""
        val_fraction = float(np.clip(val_fraction, 0.0, 1.0))
        if val_fraction <= 0.0:
            return list(samples), []
        if val_fraction >= 1.0:
            return [], list(samples)

        rng = np.random.default_rng(self._seed())
        by_class: Dict[int, List[str]] = defaultdict(list)
        for path, label in samples:
            by_class[int(label)].append(path)

        train_samples: List[Tuple[str, int]] = []
        val_samples: List[Tuple[str, int]] = []

        for label, class_paths in by_class.items():
            paths = np.array(class_paths, dtype=object)
            rng.shuffle(paths)

            val_count = int(round(len(paths) * val_fraction))
            if val_count == 0 and len(paths) > 1:
                val_count = 1
            if val_count >= len(paths):
                val_count = max(0, len(paths) - 1)

            val_paths = paths[:val_count]
            train_paths = paths[val_count:]

            train_samples.extend((str(path), int(label)) for path in train_paths.tolist())
            val_samples.extend((str(path), int(label)) for path in val_paths.tolist())

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        return train_samples, val_samples

    def load_cifar10(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess CIFAR-10 dataset.

        Returns:
            train_ds: augmented training dataset (with stochastic transforms)
            val_ds: validation dataset without augmentation
            test_ds: test dataset without augmentation
            train_eval_ds: clean view of the training set (no augmentation) for evaluation
        """

        cifar10_mean = tuple(getattr(self.config, "cifar10_mean", (0.4914, 0.4822, 0.4465)))
        cifar10_std = tuple(getattr(self.config, "cifar10_std", (0.2470, 0.2435, 0.2616)))

        def _augment(image, label):
            """Data augmentation for training"""
            # Pad + crop to inject spatial jitter similar to standard CIFAR policy
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT')
            image = tf.image.random_crop(image, size=self.config.input_shape_cifar10)

            # Horizontal flip and mild color jitter
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.15)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)

            # Random erasing style cutout (single patch)
            erase_prob = 0.5
            def _cutout():
                mask_size = tf.random.uniform([], 8, 16, dtype=tf.int32)
                offset_x = tf.random.uniform([], 0, 32 - mask_size, dtype=tf.int32)
                offset_y = tf.random.uniform([], 0, 32 - mask_size, dtype=tf.int32)
                mask = tf.ones([mask_size, mask_size, 3], dtype=image.dtype)
                paddings = [[offset_y, 32 - mask_size - offset_y],
                            [offset_x, 32 - mask_size - offset_x],
                            [0, 0]]
                mask = tf.pad(mask, paddings, constant_values=0.0)
                return image * (1.0 - mask)

            image = tf.cond(tf.random.uniform([], 0, 1) < erase_prob, _cutout, lambda: image)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        
        def _prepare_dataset(dataset: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
            """Prepare dataset with batching and prefetching"""
            if is_training:
                dataset = dataset.shuffle(
                    buffer_size=10000,
                    seed=self._seed(),
                    reshuffle_each_iteration=True,
                )
            
            dataset = dataset.batch(self.config.batch_size)
            if is_training:
                dataset = self._apply_mixup(dataset)
            dataset = dataset.map(
                lambda images, labels: self._standardize_batch(images, labels, cifar10_mean, cifar10_std),
                num_parallel_calls=AUTOTUNE,
            )
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset
        
        def _create_dataset(x, y, is_training: bool) -> tf.data.Dataset:
                """Create tf.data dataset from numpy arrays"""
                dataset = tf.data.Dataset.from_tensor_slices((x, y))
                
                if is_training and self.config.data_augmentation:
                    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
                
                return _prepare_dataset(dataset, is_training)

        # Load dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Normalize to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.config.num_classes_cifar10)
        y_test = tf.keras.utils.to_categorical(y_test, self.config.num_classes_cifar10)
        
        # Stratified shuffle before splitting so validation mirrors training distribution
        rng = np.random.default_rng(self._seed())
        indices = np.arange(len(x_train))
        rng.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        val_size = int(len(x_train) * self.config.validation_split)
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]
        
        # Create tf.data datasets
        train_ds = _create_dataset(x_train, y_train, is_training=True)
        val_ds = _create_dataset(x_val, y_val, is_training=False)
        test_ds = _create_dataset(x_test, y_test, is_training=False)

        # Clean view of the training set for evaluation/metrics logging
        train_eval_ds = _create_dataset(x_train, y_train, is_training=False)
        
        return train_ds, val_ds, test_ds, train_eval_ds
    
    def load_cifar100(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess CIFAR-100 dataset.

        Returns:
            train_ds: augmented training dataset
            val_ds: validation dataset without augmentation
            test_ds: test dataset without augmentation
            train_eval_ds: clean training dataset for evaluation/monitoring
        """

        cifar100_mean = tuple(getattr(self.config, "cifar100_mean", (0.5071, 0.4867, 0.4408)))
        cifar100_std = tuple(getattr(self.config, "cifar100_std", (0.2675, 0.2565, 0.2761)))

        def _augment(image, label):
            """Data augmentation for training"""
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT')
            image = tf.image.random_crop(image, size=self.config.input_shape_cifar100)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.15)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)

            erase_prob = 0.5

            def _cutout():
                mask_size = tf.random.uniform([], 8, 16, dtype=tf.int32)
                offset_x = tf.random.uniform([], 0, 32 - mask_size, dtype=tf.int32)
                offset_y = tf.random.uniform([], 0, 32 - mask_size, dtype=tf.int32)
                mask = tf.ones([mask_size, mask_size, 3], dtype=image.dtype)
                paddings = [[offset_y, 32 - mask_size - offset_y],
                            [offset_x, 32 - mask_size - offset_x],
                            [0, 0]]
                mask = tf.pad(mask, paddings, constant_values=0.0)
                return image * (1.0 - mask)

            image = tf.cond(tf.random.uniform([], 0, 1) < erase_prob, _cutout, lambda: image)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        
        def _prepare_dataset(dataset: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
            """Prepare dataset with batching and prefetching"""
            if is_training:
                dataset = dataset.shuffle(
                    buffer_size=10000,
                    seed=self._seed(),
                    reshuffle_each_iteration=True,
                )
            
            dataset = dataset.batch(self.config.batch_size)
            if is_training:
                dataset = self._apply_mixup(dataset)
            dataset = dataset.map(
                lambda images, labels: self._standardize_batch(images, labels, cifar100_mean, cifar100_std),
                num_parallel_calls=AUTOTUNE,
            )
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset
        
        def _create_dataset(x, y, is_training: bool) -> tf.data.Dataset:
                """Create tf.data dataset from numpy arrays"""
                dataset = tf.data.Dataset.from_tensor_slices((x, y))
                
                if is_training and self.config.data_augmentation:
                    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
                
                return _prepare_dataset(dataset, is_training)

        # Load dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        
        # Normalize to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.config.num_classes_cifar100)
        y_test = tf.keras.utils.to_categorical(y_test, self.config.num_classes_cifar100)
        
        rng = np.random.default_rng(self._seed())
        indices = np.arange(len(x_train))
        rng.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        val_size = int(len(x_train) * self.config.validation_split)
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train, y_train = x_train[val_size:], y_train[val_size:]
        
        # Create tf.data datasets
        train_ds = _create_dataset(x_train, y_train, is_training=True)
        val_ds = _create_dataset(x_val, y_val, is_training=False)
        test_ds = _create_dataset(x_test, y_test, is_training=False)

        train_eval_ds = _create_dataset(x_train, y_train, is_training=False)
        
        return train_ds, val_ds, test_ds, train_eval_ds

    def _load_tiny_imagenet_class_ids(self, dataset_root: str) -> List[str]:
        wnids_path = os.path.join(dataset_root, "wnids.txt")
        if os.path.isfile(wnids_path):
            with open(wnids_path, "r", encoding="utf-8") as f:
                wnids = [line.strip() for line in f if line.strip()]
            if wnids:
                return wnids

        train_dir = os.path.join(dataset_root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Could not find Tiny ImageNet classes. Missing {wnids_path} and {train_dir}."
            )
        return sorted(
            [entry for entry in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, entry))]
        )

    def _collect_tiny_imagenet_train_samples(
        self,
        dataset_root: str,
        class_to_index: Dict[str, int],
    ) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        train_dir = os.path.join(dataset_root, "train")
        for class_id, class_idx in class_to_index.items():
            images_dir = os.path.join(train_dir, class_id, "images")
            if not os.path.isdir(images_dir):
                continue
            for file_name in sorted(os.listdir(images_dir)):
                if not self._is_image_file(file_name):
                    continue
                image_path = os.path.join(images_dir, file_name)
                samples.append((image_path, int(class_idx)))
        if not samples:
            raise FileNotFoundError(
                f"No Tiny ImageNet training images found under {train_dir}."
            )
        return samples

    def _collect_tiny_imagenet_val_samples(
        self,
        dataset_root: str,
        class_to_index: Dict[str, int],
    ) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        val_dir = os.path.join(dataset_root, "val")
        annotations_path = os.path.join(val_dir, "val_annotations.txt")
        val_images_dir = os.path.join(val_dir, "images")

        if os.path.isfile(annotations_path) and os.path.isdir(val_images_dir):
            with open(annotations_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    file_name, class_id = parts[0], parts[1]
                    if class_id not in class_to_index:
                        continue
                    image_path = os.path.join(val_images_dir, file_name)
                    if os.path.isfile(image_path):
                        samples.append((image_path, int(class_to_index[class_id])))
            if samples:
                return samples

        # Fallback for variants where val images are already class-folder organized.
        for class_id, class_idx in class_to_index.items():
            images_dir = os.path.join(val_dir, class_id, "images")
            if not os.path.isdir(images_dir):
                images_dir = os.path.join(val_dir, class_id)
                if not os.path.isdir(images_dir):
                    continue
            for file_name in sorted(os.listdir(images_dir)):
                if not self._is_image_file(file_name):
                    continue
                image_path = os.path.join(images_dir, file_name)
                if os.path.isfile(image_path):
                    samples.append((image_path, int(class_idx)))

        if not samples:
            raise FileNotFoundError(
                f"No Tiny ImageNet validation annotations/images found under {val_dir}."
            )
        return samples

    def _create_tiny_imagenet_dataset(
        self,
        samples: List[Tuple[str, int]],
        is_training: bool,
        apply_augmentation: bool,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> tf.data.Dataset:
        if not samples:
            # Build an empty typed dataset for robustness.
            empty_x = tf.constant([], dtype=tf.string)
            empty_y = tf.constant([], dtype=tf.int32)
            dataset = tf.data.Dataset.from_tensor_slices((empty_x, empty_y))
        else:
            paths = np.array([path for path, _ in samples], dtype=np.str_)
            labels = np.array([label for _, label in samples], dtype=np.int32)
            dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        target_height, target_width, _ = input_shape

        def _decode_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            image_bytes = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image_bytes, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(
                image,
                [target_height, target_width],
                method=tf.image.ResizeMethod.BILINEAR,
                antialias=True,
            )
            image = tf.clip_by_value(image, 0.0, 1.0)
            one_hot = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
            return image, one_hot

        def _augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            pad = max(4, target_height // 8)
            image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
            image = tf.image.random_crop(image, [target_height, target_width, 3])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.12)
            image = tf.image.random_contrast(image, 0.85, 1.15)

            erase_prob = 0.35

            def _cutout() -> tf.Tensor:
                min_box = max(8, target_height // 10)
                max_box = max(min_box + 1, target_height // 3)
                mask_size = tf.random.uniform([], min_box, max_box, dtype=tf.int32)
                offset_x = tf.random.uniform([], 0, target_width - mask_size + 1, dtype=tf.int32)
                offset_y = tf.random.uniform([], 0, target_height - mask_size + 1, dtype=tf.int32)
                mask = tf.ones([mask_size, mask_size, 3], dtype=image.dtype)
                paddings = [
                    [offset_y, target_height - mask_size - offset_y],
                    [offset_x, target_width - mask_size - offset_x],
                    [0, 0],
                ]
                mask = tf.pad(mask, paddings, constant_values=0.0)
                return image * (1.0 - mask)

            image = tf.cond(tf.random.uniform([], 0, 1) < erase_prob, _cutout, lambda: image)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label

        if is_training:
            dataset = dataset.shuffle(
                buffer_size=max(10000, len(samples)),
                seed=self._seed(),
                reshuffle_each_iteration=True,
            )
        dataset = dataset.map(_decode_image, num_parallel_calls=AUTOTUNE)
        if is_training and apply_augmentation:
            dataset = dataset.map(_augment, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.config.batch_size)
        if is_training:
            dataset = self._apply_mixup(dataset)
        dataset = dataset.map(
            lambda images, labels: self._standardize_batch(images, labels, mean, std),
            num_parallel_calls=AUTOTUNE,
        )
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    @staticmethod
    def _tiny_imagenet_is_ready(dataset_root: str) -> bool:
        required = [
            os.path.join(dataset_root, "train"),
            os.path.join(dataset_root, "val"),
            os.path.join(dataset_root, "wnids.txt"),
        ]
        return all(os.path.exists(path) for path in required)

    def _ensure_tiny_imagenet_available(self, dataset_root: str) -> str:
        dataset_root = os.path.abspath(dataset_root)
        if self._tiny_imagenet_is_ready(dataset_root):
            return dataset_root

        auto_download = bool(getattr(self.config, "tiny_imagenet_auto_download", False))
        if not auto_download:
            raise FileNotFoundError(
                f"Tiny ImageNet root not found or incomplete: {dataset_root}. "
                "Enable config.tiny_imagenet_auto_download or set config.tiny_imagenet_root to a valid path."
            )

        dataset_url = str(
            getattr(
                self.config,
                "tiny_imagenet_url",
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
            )
        )
        expected_dir_name = "tiny-imagenet-200"
        target_parent = os.path.dirname(dataset_root) or "."
        os.makedirs(target_parent, exist_ok=True)

        archive_name = os.path.basename(dataset_url) or "tiny-imagenet-200.zip"
        archive_name = archive_name.split("?")[0]
        if not archive_name.endswith(".zip"):
            archive_name = f"{archive_name}.zip"

        try:
            archive_path = tf.keras.utils.get_file(
                fname=archive_name,
                origin=dataset_url,
                cache_subdir="datasets",
                extract=False,
            )
        except Exception:
            if dataset_url.startswith("http://"):
                secure_url = "https://" + dataset_url[len("http://"):]
                archive_path = tf.keras.utils.get_file(
                    fname=archive_name,
                    origin=secure_url,
                    cache_subdir="datasets",
                    extract=False,
                )
            else:
                raise

        extracted_default_root = os.path.join(target_parent, expected_dir_name)
        if not self._tiny_imagenet_is_ready(extracted_default_root):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_parent)

        if dataset_root != extracted_default_root and os.path.isdir(extracted_default_root) and not os.path.exists(dataset_root):
            shutil.move(extracted_default_root, dataset_root)

        if self._tiny_imagenet_is_ready(dataset_root):
            return dataset_root
        if self._tiny_imagenet_is_ready(extracted_default_root):
            return extracted_default_root

        raise FileNotFoundError(
            "Tiny ImageNet download/extraction finished but dataset layout is still invalid. "
            f"Checked: {dataset_root} and {extracted_default_root}"
        )

    def load_tiny_imagenet(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load Tiny ImageNet (200 classes, 64x64) with train/val/test datasets."""
        dataset_root = self._ensure_tiny_imagenet_available(
            getattr(self.config, "tiny_imagenet_root", "./data/tiny-imagenet-200")
        )

        class_ids = self._load_tiny_imagenet_class_ids(dataset_root)
        class_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
        num_classes = len(class_ids)
        setattr(self.config, "num_classes_tiny_imagenet", num_classes)

        train_samples_all = self._collect_tiny_imagenet_train_samples(dataset_root, class_to_index)
        official_val_samples = self._collect_tiny_imagenet_val_samples(dataset_root, class_to_index)

        use_official_val_for_test = bool(
            getattr(self.config, "tiny_imagenet_use_official_val_for_test", True)
        )
        if use_official_val_for_test:
            train_samples, val_samples = self._stratified_split_samples(
                train_samples_all,
                float(getattr(self.config, "validation_split", 0.1)),
            )
            test_samples = official_val_samples
        else:
            train_samples = train_samples_all
            val_samples, test_samples = self._stratified_split_samples(
                official_val_samples,
                float(getattr(self.config, "validation_split", 0.1)),
            )
            if not test_samples:
                test_samples = val_samples

        mean = tuple(getattr(self.config, "tiny_imagenet_mean", (0.4802, 0.4481, 0.3975)))
        std = tuple(getattr(self.config, "tiny_imagenet_std", (0.2302, 0.2265, 0.2262)))
        input_shape = tuple(getattr(self.config, "input_shape_tiny_imagenet", (64, 64, 3)))

        train_ds = self._create_tiny_imagenet_dataset(
            train_samples,
            is_training=True,
            apply_augmentation=bool(getattr(self.config, "data_augmentation", True)),
            num_classes=num_classes,
            input_shape=input_shape,
            mean=mean,
            std=std,
        )
        val_ds = self._create_tiny_imagenet_dataset(
            val_samples,
            is_training=False,
            apply_augmentation=False,
            num_classes=num_classes,
            input_shape=input_shape,
            mean=mean,
            std=std,
        )
        test_ds = self._create_tiny_imagenet_dataset(
            test_samples,
            is_training=False,
            apply_augmentation=False,
            num_classes=num_classes,
            input_shape=input_shape,
            mean=mean,
            std=std,
        )
        train_eval_ds = self._create_tiny_imagenet_dataset(
            train_samples,
            is_training=False,
            apply_augmentation=False,
            num_classes=num_classes,
            input_shape=input_shape,
            mean=mean,
            std=std,
        )

        return train_ds, val_ds, test_ds, train_eval_ds
