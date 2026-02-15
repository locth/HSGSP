import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from typing import Tuple
# from data.augmentation import DataAugmentation
from config import Config

class VGG16:
    """VGG16 model"""

    def __init__(self, config):
        self.config = config

    def build_vgg16_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> Model:
        """Build VGG16 model with BatchNormalization"""
        inputs = layers.Input(shape=input_shape)

        # L2 regularizer
        l2_reg = regularizers.l2(self.config.l2_regularization)

        # Block 1
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(inputs)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 2
        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 3
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 4
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Spatial Dropout
        if self.config.use_spatial_dropout:
            x = layers.SpatialDropout2D(self.config.spatial_dropout_rate)(x)
        
        # Block 5
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Conv2D(512, 3, padding='same', activation='relu',
                         kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.MaxPooling2D(2)(x)
        
        dense_drop1 = getattr(self.config, "fc_dropout_rate1", self.config.dropout_rate)
        dense_drop2 = getattr(self.config, "fc_dropout_rate2", self.config.dropout_rate)

        if num_classes <= 10:
            fc1_units, fc2_units = 256, 512
        elif num_classes <= 100:
            fc1_units, fc2_units = 512, 512
        else:
            fc1_units, fc2_units = 1024, 512

        # Flatten and Dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(fc1_units, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(dense_drop1)(x)
        x = layers.Dense(fc2_units, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(dense_drop2)(x)
        
        # Output layer with label smoothing
        outputs = layers.Dense(num_classes, activation='softmax',
                               kernel_regularizer=l2_reg)(x)
        
        input_h, input_w, _ = input_shape
        model_name = f"vgg16_{input_h}x{input_w}_{num_classes}cls"
        model = Model(inputs, outputs, name=model_name)
        return model
    
    def build_cifar100_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> Model:
        """Backward-compatible alias for shared VGG16 builder."""
        return self.build_vgg16_model(num_classes=num_classes, input_shape=input_shape)
