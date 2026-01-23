# ==============================
# 📄 training/custom_losses.py
# ==============================
# Custom Loss Functions for Audio Scene Classification
# Upgrade 9: Specialized losses for better training
# ==============================

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABEL_CONFIG, TRAINING_CONFIG


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for handling class imbalance.
    
    Down-weights well-classified examples and focuses on hard examples.
    Especially useful when some classes are rare (e.g., emergency situations).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        from_logits: bool = False,
        **kwargs
    ):
        """
        Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class balancing weight
            from_logits: Whether predictions are logits or probabilities
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        focal_weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'from_logits': self.from_logits
        })
        return config


class LabelSmoothingLoss(tf.keras.losses.Loss):
    """
    Label Smoothing Cross Entropy Loss.
    
    Prevents overconfidence by distributing some probability mass
    to non-target classes. Improves generalization.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        **kwargs
    ):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            smoothing: Amount of smoothing (0 = no smoothing, 1 = uniform)
        """
        super().__init__(**kwargs)
        self.smoothing = smoothing
        
    def call(self, y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        
        # Smooth labels
        smooth_positives = 1.0 - self.smoothing
        smooth_negatives = self.smoothing / num_classes
        
        y_true_smooth = y_true * smooth_positives + smooth_negatives
        
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Cross entropy with smoothed labels
        loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({'smoothing': self.smoothing})
        return config


class HierarchicalLoss(tf.keras.losses.Loss):
    """
    Hierarchical Loss that considers semantic relationships.
    
    Penalizes less for predictions within the same semantic category.
    For example, confusing "Airport" with "Railway" (both transportation)
    is less severe than confusing "Airport" with "Home".
    """
    
    def __init__(
        self,
        hierarchy: Dict[str, List[int]] = None,
        num_classes: int = None,
        intra_category_weight: float = 0.3,
        **kwargs
    ):
        """
        Initialize Hierarchical Loss.
        
        Args:
            hierarchy: Dict mapping category names to class indices
            num_classes: Number of classes
            intra_category_weight: Penalty weight for same-category errors
        """
        super().__init__(**kwargs)
        
        self.hierarchy = hierarchy or LABEL_CONFIG.location_hierarchy
        self.num_classes = num_classes or LABEL_CONFIG.num_locations
        self.intra_weight = intra_category_weight
        
        # Build distance matrix
        self.distance_matrix = self._build_distance_matrix()
        
    def _build_distance_matrix(self) -> tf.Tensor:
        """Build pairwise semantic distance matrix."""
        # Start with ones (full penalty)
        distances = np.ones((self.num_classes, self.num_classes), dtype=np.float32)
        
        # Reduce penalty for same-category pairs
        for category, indices in self.hierarchy.items():
            for i in indices:
                for j in indices:
                    if i != j and i < self.num_classes and j < self.num_classes:
                        distances[i, j] = self.intra_weight
        
        # Zero diagonal (no penalty for correct predictions)
        np.fill_diagonal(distances, 0.0)
        
        return tf.constant(distances, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        # Standard cross entropy
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Get predicted and true classes
        pred_class = tf.argmax(y_pred, axis=-1)
        true_class = tf.argmax(y_true, axis=-1)
        
        # Get semantic penalties
        batch_indices = tf.stack([true_class, pred_class], axis=-1)
        semantic_penalties = tf.gather_nd(self.distance_matrix, batch_indices)
        
        # Weight CE loss by semantic penalty
        # Penalty is 0 for correct, intra_weight for same category, 1 for different category
        weighted_loss = ce_loss * (1.0 + semantic_penalties)
        
        return tf.reduce_mean(weighted_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'intra_category_weight': self.intra_weight
        })
        return config


class ConfidenceCalibrationLoss(tf.keras.losses.Loss):
    """
    Loss that encourages well-calibrated confidence scores.
    
    Adds Expected Calibration Error (ECE) as a regularization term
    to make confidence scores meaningful.
    """
    
    def __init__(
        self,
        n_bins: int = 15,
        ece_weight: float = 0.1,
        **kwargs
    ):
        """
        Initialize Calibration Loss.
        
        Args:
            n_bins: Number of bins for ECE calculation
            ece_weight: Weight for ECE term
        """
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.ece_weight = ece_weight
        
    def expected_calibration_error(
        self,
        confidences: tf.Tensor,
        accuracies: tf.Tensor
    ) -> tf.Tensor:
        """Calculate ECE."""
        ece = 0.0
        n_samples = tf.cast(tf.shape(confidences)[0], tf.float32)
        
        for bin_idx in range(self.n_bins):
            bin_lower = tf.cast(bin_idx, tf.float32) / self.n_bins
            bin_upper = tf.cast(bin_idx + 1, tf.float32) / self.n_bins
            
            # Find samples in this bin
            in_bin = tf.logical_and(
                confidences >= bin_lower,
                confidences < bin_upper
            )
            bin_size = tf.reduce_sum(tf.cast(in_bin, tf.float32))
            
            # Skip empty bins
            if bin_size > 0:
                avg_confidence = tf.reduce_sum(
                    tf.where(in_bin, confidences, 0.0)
                ) / bin_size
                
                avg_accuracy = tf.reduce_sum(
                    tf.where(in_bin, accuracies, 0.0)
                ) / bin_size
                
                ece += (bin_size / n_samples) * tf.abs(avg_confidence - avg_accuracy)
        
        return ece
    
    def call(self, y_true, y_pred):
        # Standard cross entropy
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        ce_loss = tf.reduce_mean(
            -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        )
        
        # Calculate confidence and accuracy
        confidences = tf.reduce_max(y_pred, axis=-1)
        predictions = tf.argmax(y_pred, axis=-1)
        labels = tf.argmax(y_true, axis=-1)
        accuracies = tf.cast(predictions == labels, tf.float32)
        
        # ECE
        ece = self.expected_calibration_error(confidences, accuracies)
        
        return ce_loss + self.ece_weight * ece
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_bins': self.n_bins,
            'ece_weight': self.ece_weight
        })
        return config


class EmergencyFocalLoss(tf.keras.losses.Loss):
    """
    Specialized loss for emergency detection.
    
    Heavily penalizes false negatives (missing emergencies)
    while still penalizing false positives.
    """
    
    def __init__(
        self,
        false_negative_weight: float = 5.0,
        false_positive_weight: float = 1.0,
        gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fn_weight = false_negative_weight
        self.fp_weight = false_positive_weight
        self.gamma = gamma
        
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Binary cross entropy with asymmetric weights
        # For positive examples (emergencies): use FN weight
        # For negative examples: use FP weight
        
        pos_loss = -y_true * self.fn_weight * tf.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)
        neg_loss = -(1 - y_true) * self.fp_weight * tf.pow(y_pred, self.gamma) * tf.math.log(1 - y_pred)
        
        loss = pos_loss + neg_loss
        
        return tf.reduce_mean(loss)


class AuralisLoss(tf.keras.losses.Loss):
    """
    Combined loss function for Auralis multi-task learning.
    
    Combines:
    - Focal loss for location/situation (handles class imbalance)
    - Label smoothing for generalization
    - Hierarchical penalty for semantic errors
    - Emergency focal loss for critical detection
    - Confidence calibration
    """
    
    def __init__(
        self,
        location_weight: float = 1.0,
        situation_weight: float = 1.0,
        confidence_weight: float = 0.3,
        emergency_weight: float = 2.0,
        focal_gamma: float = None,
        label_smoothing: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.location_weight = location_weight
        self.situation_weight = situation_weight
        self.confidence_weight = confidence_weight
        self.emergency_weight = emergency_weight
        
        focal_gamma = focal_gamma or TRAINING_CONFIG.focal_gamma
        label_smoothing = label_smoothing or TRAINING_CONFIG.label_smoothing
        
        # Sub-losses
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        self.hierarchical_loss = HierarchicalLoss()
        self.emergency_loss = EmergencyFocalLoss()
        self.calibration_loss = ConfidenceCalibrationLoss()
        
    def call(self, y_true, y_pred):
        """
        Calculate combined loss.
        
        Args:
            y_true: Dict with 'location', 'situation', 'confidence', 'emergency'
            y_pred: Dict with same keys
        """
        total_loss = 0.0
        
        # Location loss
        if 'location' in y_true and 'location' in y_pred:
            loc_focal = self.focal_loss(y_true['location'], y_pred['location'])
            loc_smooth = self.smoothing_loss(y_true['location'], y_pred['location'])
            loc_hier = self.hierarchical_loss(y_true['location'], y_pred['location'])
            
            location_loss = 0.4 * loc_focal + 0.3 * loc_smooth + 0.3 * loc_hier
            total_loss += self.location_weight * location_loss
        
        # Situation loss
        if 'situation' in y_true and 'situation' in y_pred:
            sit_focal = self.focal_loss(y_true['situation'], y_pred['situation'])
            sit_smooth = self.smoothing_loss(y_true['situation'], y_pred['situation'])
            
            situation_loss = 0.5 * sit_focal + 0.5 * sit_smooth
            total_loss += self.situation_weight * situation_loss
        
        # Confidence loss (MSE with calibration)
        if 'confidence' in y_true and 'confidence' in y_pred:
            conf_mse = tf.reduce_mean(tf.square(y_true['confidence'] - y_pred['confidence']))
            total_loss += self.confidence_weight * conf_mse
        
        # Emergency loss (critical - higher weight)
        if 'emergency' in y_true and 'emergency' in y_pred:
            emerg_loss = self.emergency_loss(y_true['emergency'], y_pred['emergency'])
            total_loss += self.emergency_weight * emerg_loss
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'location_weight': self.location_weight,
            'situation_weight': self.situation_weight,
            'confidence_weight': self.confidence_weight,
            'emergency_weight': self.emergency_weight
        })
        return config


class ContrastiveCenterLoss(tf.keras.losses.Loss):
    """
    Center Loss combined with contrastive learning.
    
    Minimizes intra-class variation while maximizing inter-class variation.
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        alpha: float = 0.5,
        lambda_c: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_c = lambda_c
        
        # Initialize centers
        self.centers = tf.Variable(
            tf.random.normal([num_classes, feature_dim]),
            trainable=False
        )
        
    def call(self, y_true, features):
        """
        Args:
            y_true: One-hot labels
            features: Feature embeddings
        """
        labels = tf.argmax(y_true, axis=-1)
        
        # Get centers for each sample
        centers_batch = tf.gather(self.centers, labels)
        
        # Center loss: minimize distance to class center
        center_loss = tf.reduce_mean(tf.reduce_sum(tf.square(features - centers_batch), axis=-1))
        
        # Update centers
        unique_labels, idx = tf.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            class_features = tf.boolean_mask(features, mask)
            center_update = tf.reduce_mean(class_features, axis=0)
            
            # Exponential moving average update
            new_center = (1 - self.alpha) * self.centers[label] + self.alpha * center_update
            self.centers[label].assign(new_center)
        
        return self.lambda_c * center_loss


# ==============================
# 🛠️ UTILITY FUNCTIONS
# ==============================

def get_class_weights(labels: np.ndarray, num_classes: int) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        labels: Array of class indices
        num_classes: Number of classes
        
    Returns:
        Dict mapping class index to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    
    return {i: w for i, w in enumerate(weights)}


def create_sample_weights(
    labels: np.ndarray,
    emergency_labels: np.ndarray = None,
    emergency_boost: float = 3.0
) -> np.ndarray:
    """
    Create sample weights with emergency boosting.
    
    Args:
        labels: Class labels
        emergency_labels: Binary emergency labels
        emergency_boost: Weight multiplier for emergency samples
        
    Returns:
        Array of sample weights
    """
    weights = np.ones(len(labels), dtype=np.float32)
    
    # Class-based weights
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = len(labels) / (len(unique) * counts)
    
    for cls, weight in zip(unique, class_weights):
        weights[labels == cls] = weight
    
    # Boost emergency samples
    if emergency_labels is not None:
        weights[emergency_labels == 1] *= emergency_boost
    
    return weights


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Custom Losses")
    print("="*60)
    
    batch_size = 8
    num_classes = 13
    
    # Create dummy data
    y_true = np.zeros((batch_size, num_classes), dtype=np.float32)
    for i in range(batch_size):
        y_true[i, np.random.randint(num_classes)] = 1.0
    
    y_pred = np.random.softmax(np.random.randn(batch_size, num_classes), axis=-1).astype(np.float32)
    
    y_true_tf = tf.constant(y_true)
    y_pred_tf = tf.constant(y_pred)
    
    # Test Focal Loss
    print("\n🔧 Testing FocalLoss...")
    focal = FocalLoss(gamma=2.0)
    loss = focal(y_true_tf, y_pred_tf)
    print(f"   Loss: {loss:.4f}")
    
    # Test Label Smoothing
    print("\n🔧 Testing LabelSmoothingLoss...")
    smoothing = LabelSmoothingLoss(smoothing=0.1)
    loss = smoothing(y_true_tf, y_pred_tf)
    print(f"   Loss: {loss:.4f}")
    
    # Test Hierarchical Loss
    print("\n🔧 Testing HierarchicalLoss...")
    hierarchical = HierarchicalLoss()
    loss = hierarchical(y_true_tf, y_pred_tf)
    print(f"   Loss: {loss:.4f}")
    
    # Test Calibration Loss
    print("\n🔧 Testing ConfidenceCalibrationLoss...")
    calibration = ConfidenceCalibrationLoss()
    loss = calibration(y_true_tf, y_pred_tf)
    print(f"   Loss: {loss:.4f}")
    
    # Test Emergency Loss
    print("\n🔧 Testing EmergencyFocalLoss...")
    emergency = EmergencyFocalLoss()
    y_true_binary = tf.constant(np.random.randint(0, 2, batch_size).astype(np.float32))
    y_pred_binary = tf.constant(np.random.rand(batch_size).astype(np.float32))
    loss = emergency(y_true_binary, y_pred_binary)
    print(f"   Loss: {loss:.4f}")
    
    # Test class weights
    print("\n🔧 Testing class weights...")
    labels = np.random.randint(0, num_classes, 1000)
    weights = get_class_weights(labels, num_classes)
    print(f"   Weights: {list(weights.values())[:5]}...")
    
    print("\n✅ Custom Losses test passed!")