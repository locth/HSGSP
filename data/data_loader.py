import math
import numpy as np
import tensorflow as tf
from typing import Tuple

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

class DataLoader:
    """Unified dataset loader for HSGSP project"""

    def __init__(self, config):
        self.config = config

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

    def load_cifar10(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess CIFAR-10 dataset.

        Returns:
            train_ds: augmented training dataset (with stochastic transforms)
            val_ds: validation dataset without augmentation
            test_ds: test dataset without augmentation
            train_eval_ds: clean view of the training set (no augmentation) for evaluation
        """

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
                dataset = dataset.shuffle(buffer_size=10000)
            
            dataset = dataset.batch(self.config.batch_size)
            if is_training:
                dataset = self._apply_mixup(dataset)
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
        rng = np.random.default_rng(SEED)
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
                dataset = dataset.shuffle(buffer_size=10000)
            
            dataset = dataset.batch(self.config.batch_size)
            if is_training:
                dataset = self._apply_mixup(dataset)
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
        
        rng = np.random.default_rng(SEED)
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
