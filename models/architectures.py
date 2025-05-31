import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import numpy as np

class SkinCancerCNN(keras.Model):
    """Custom CNN architecture optimized for skin cancer detection."""
    
    def __init__(self, num_classes=7, input_shape=(224, 224, 3)):
        super(SkinCancerCNN, self).__init__()
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu')
        self.pool4 = layers.MaxPooling2D((2, 2))
        
        # Global average pooling
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        
        # Classification layers
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        
        x = self.global_avg_pool(x)
        
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        return self.output_layer(x)

class AttentionBlock(layers.Layer):
    """Self-attention mechanism for feature enhancement."""
    
    def __init__(self, units):
        super(AttentionBlock, self).__init__()
        self.units = units
        self.query = layers.Dense(units)
        self.key = layers.Dense(units)
        self.value = layers.Dense(units)
        self.softmax = layers.Softmax()
    
    def call(self, inputs):
        # Reshape for attention computation
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply attention to values
        context = tf.matmul(attention_weights, v)
        
        return context, attention_weights

class FairnessAwareCNN(keras.Model):
    """CNN architecture with fairness-aware training mechanisms."""
    
    def __init__(self, num_classes=7, num_sensitive_groups=6, input_shape=(224, 224, 3)):
        super(FairnessAwareCNN, self).__init__()
        self.num_classes = num_classes
        self.num_sensitive_groups = num_sensitive_groups
        
        # Shared feature extractor
        self.backbone = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Adversarial branch for fairness
        self.adversarial_layers = [
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_sensitive_groups, activation='softmax')
        ]
        
        # Main classification branch
        self.classifier_layers = [
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ]
        
        # Global average pooling
        self.global_avg_pool = layers.GlobalAveragePooling2D()
    
    def call(self, inputs, training=None):
        # Extract features
        features = self.backbone(inputs, training=training)
        features = self.global_avg_pool(features)
        
        # Main classification
        x = features
        for layer in self.classifier_layers:
            x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)
        
        main_output = x
        
        # Adversarial prediction (for fairness)
        y = features
        for layer in self.adversarial_layers:
            y = layer(y, training=training) if hasattr(layer, 'training') else layer(y)
        
        adversarial_output = y
        
        return main_output, adversarial_output

class ExplainableCNN(keras.Model):
    """CNN with built-in explainability features."""
    
    def __init__(self, num_classes=7, input_shape=(224, 224, 3)):
        super(ExplainableCNN, self).__init__()
        self.num_classes = num_classes
        
        # Feature extraction with attention
        self.conv_layers = [
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        ]
        
        # Attention mechanism
        self.attention = AttentionBlock(512)
        
        # Classification head
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Store feature maps for attention visualization
        self.feature_maps = x
        
        # Reshape for attention
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]
        
        x_reshaped = tf.reshape(x, [batch_size, height * width, channels])
        
        # Apply attention
        attended_features, attention_weights = self.attention(x_reshaped)
        self.attention_weights = attention_weights
        
        # Global average pooling
        x = tf.reduce_mean(attended_features, axis=1)
        
        # Classification
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        
        return self.output_layer(x)

class HybridCNNViT(keras.Model):
    """Hybrid architecture combining CNN feature extraction with Vision Transformer."""
    
    def __init__(self, num_classes=7, patch_size=16, num_heads=8, num_layers=6):
        super(HybridCNNViT, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # CNN backbone for feature extraction
        self.cnn_backbone = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Patch embedding
        self.patch_embed = layers.Dense(768)
        
        # Positional encoding
        self.pos_embed = self.add_weight(
            shape=(1, 196, 768),
            initializer='random_normal',
            trainable=True,
            name='pos_embed'
        )
        
        # Transformer layers
        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.append(
                TransformerBlock(768, num_heads)
            )
        
        # Classification head
        self.layer_norm = layers.LayerNormalization()
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        # CNN feature extraction
        cnn_features = self.cnn_backbone(inputs, training=training)
        
        # Convert CNN features to patches
        batch_size = tf.shape(cnn_features)[0]
        height = tf.shape(cnn_features)[1]
        width = tf.shape(cnn_features)[2]
        channels = tf.shape(cnn_features)[3]
        
        # Reshape to patches
        patches = tf.reshape(cnn_features, [batch_size, height * width, channels])
        
        # Patch embedding
        x = self.patch_embed(patches)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Classification
        return self.classifier(x)

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * embed_dim
        
        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        # Multi-head attention
        attention_output = self.attention(inputs, inputs, training=training)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)

class UncertaintyAwareCNN(keras.Model):
    """CNN with uncertainty quantification for clinical decision support."""
    
    def __init__(self, num_classes=7, num_monte_carlo=10, input_shape=(224, 224, 3)):
        super(UncertaintyAwareCNN, self).__init__()
        self.num_classes = num_classes
        self.num_monte_carlo = num_monte_carlo
        
        # Feature extraction
        self.backbone = tf.keras.applications.EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Classification head with dropout for uncertainty
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        features = self.backbone(inputs, training=training)
        features = self.global_avg_pool(features)
        
        x = self.dense1(features)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        return self.output_layer(x)
    
    def predict_with_uncertainty(self, inputs):
        """Predict with uncertainty quantification using Monte Carlo Dropout."""
        predictions = []
        
        for _ in range(self.num_monte_carlo):
            pred = self(inputs, training=True)  # Keep dropout active
            predictions.append(pred)
        
        predictions = tf.stack(predictions)
        
        # Calculate mean and variance
        mean_pred = tf.reduce_mean(predictions, axis=0)
        var_pred = tf.reduce_mean(tf.square(predictions - mean_pred), axis=0)
        
        # Calculate predictive entropy (uncertainty)
        entropy = -tf.reduce_sum(mean_pred * tf.math.log(mean_pred + 1e-8), axis=-1)
        
        return {
            'predictions': mean_pred,
            'variance': var_pred,
            'uncertainty': entropy
        }

def create_ensemble_model(model_configs, num_classes=7):
    """Create an ensemble of different architectures."""
    models = []
    
    for config in model_configs:
        if config['type'] == 'cnn':
            model = SkinCancerCNN(num_classes)
        elif config['type'] == 'fairness_aware':
            model = FairnessAwareCNN(num_classes)
        elif config['type'] == 'explainable':
            model = ExplainableCNN(num_classes)
        elif config['type'] == 'hybrid':
            model = HybridCNNViT(num_classes)
        elif config['type'] == 'uncertainty':
            model = UncertaintyAwareCNN(num_classes)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        models.append(model)
    
    return EnsembleModel(models, num_classes)

class EnsembleModel(keras.Model):
    """Ensemble model that combines predictions from multiple architectures."""
    
    def __init__(self, models, num_classes, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.num_classes = num_classes
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def call(self, inputs, training=None):
        predictions = []
        
        for model in self.models:
            pred = model(inputs, training=training)
            predictions.append(pred)
        
        # Weighted average of predictions
        weighted_preds = []
        for i, pred in enumerate(predictions):
            weighted_preds.append(self.weights[i] * pred)
        
        ensemble_pred = tf.reduce_sum(weighted_preds, axis=0)
        
        return ensemble_pred
    
    def predict_with_confidence(self, inputs):
        """Predict with ensemble confidence metrics."""
        predictions = []
        
        for model in self.models:
            pred = model(inputs, training=False)
            predictions.append(pred)
        
        predictions = tf.stack(predictions)
        
        # Calculate ensemble statistics
        mean_pred = tf.reduce_mean(predictions, axis=0)
        std_pred = tf.math.reduce_std(predictions, axis=0)
        
        # Calculate agreement between models
        max_indices = tf.argmax(predictions, axis=-1)
        agreement = tf.reduce_mean(
            tf.cast(tf.equal(max_indices, tf.expand_dims(tf.argmax(mean_pred, axis=-1), 0)), tf.float32),
            axis=0
        )
        
        return {
            'predictions': mean_pred,
            'std': std_pred,
            'agreement': agreement,
            'individual_predictions': predictions
        }
