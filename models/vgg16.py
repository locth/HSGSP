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
        # Flatten and Dense layers
        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 10:
            x = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
        else:
            x = layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(dense_drop1)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(dense_drop2)(x)
        
        # Output layer with label smoothing
        outputs = layers.Dense(num_classes, activation='softmax',
                               kernel_regularizer=l2_reg)(x)
        
        model = Model(inputs, outputs, name='vgg_for_cifar10')
        return model
    
    def build_cifar100_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> Model:
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
        # Flatten and Dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(dense_drop1)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization(momentum=self.config.batch_norm_momentum)(x)
        x = layers.Dropout(dense_drop2)(x)
        
        # Output layer with label smoothing
        outputs = layers.Dense(num_classes, activation='softmax',
                               kernel_regularizer=l2_reg)(x)
        
        model = Model(inputs, outputs, name='vgg_for_cifar100')
        return model
