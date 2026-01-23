# ==============================
# 📄 training/knowledge_distillation.py
# ==============================
# Knowledge Distillation for Model Compression
# Upgrade 6: Transfer knowledge to smaller models
# ==============================

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABEL_CONFIG, MODEL_CONFIG, TRAINING_CONFIG


class DistillationLoss(tf.keras.losses.Loss):
    """
    Combined distillation and hard label loss.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        # y_true contains: [hard_labels, soft_labels]
        # This is handled in the training loop
        pass


class KnowledgeDistillation(Model):
    """
    Knowledge distillation wrapper.
    Trains a student model to mimic a teacher model.
    
    The student learns from:
    1. Soft targets (teacher's probability distribution)
    2. Hard targets (ground truth labels)
    3. Intermediate feature matching (optional)
    """
    
    def __init__(
        self,
        teacher_model: Model,
        student_model: Model,
        temperature: float = None,
        alpha: float = None,
        feature_matching: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature or TRAINING_CONFIG.distillation_temperature
        self.alpha = alpha or TRAINING_CONFIG.distillation_alpha
        self.feature_matching = feature_matching
        
        # Freeze teacher
        self.teacher.trainable = False
        
        # Feature alignment layers (if dimensions differ)
        if feature_matching:
            self.feature_aligner = layers.Dense(256, name='feature_aligner')
        
        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.hard_loss_tracker = tf.keras.metrics.Mean(name='hard_loss')
        self.soft_loss_tracker = tf.keras.metrics.Mean(name='soft_loss')
        self.student_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name='student_accuracy')
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.hard_loss_tracker,
            self.soft_loss_tracker,
            self.student_acc_tracker
        ]
    
    def distillation_loss(
        self,
        student_logits: tf.Tensor,
        teacher_logits: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute KL divergence between soft targets.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            
        Returns:
            KL divergence loss
        """
        # Soften with temperature
        soft_teacher = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
        soft_student = tf.nn.log_softmax(student_logits / self.temperature, axis=-1)
        
        # KL divergence
        kl_loss = tf.reduce_sum(
            soft_teacher * (tf.math.log(soft_teacher + 1e-10) - soft_student),
            axis=-1
        )
        
        # Scale by temperature squared
        return tf.reduce_mean(kl_loss) * (self.temperature ** 2)
    
    def feature_matching_loss(
        self,
        student_features: tf.Tensor,
        teacher_features: tf.Tensor
    ) -> tf.Tensor:
        """
        MSE loss between intermediate features.
        """
        if hasattr(self, 'feature_aligner'):
            student_features = self.feature_aligner(student_features)
        return tf.reduce_mean(tf.square(student_features - teacher_features))
    
    def call(self, inputs, training=False):
        """Forward pass through student."""
        return self.student(inputs, training=training)
    
    def train_step(self, data):
        """Custom training step with distillation."""
        x, y_true = data
        
        # Get teacher predictions (no gradient)
        teacher_outputs = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Get student predictions
            student_outputs = self.student(x, training=True)
            
            # Calculate losses for each output head
            total_hard_loss = 0
            total_soft_loss = 0
            
            for key in student_outputs:
                if key in teacher_outputs and key in y_true:
                    # Hard label loss
                    hard_loss = tf.keras.losses.categorical_crossentropy(
                        y_true[key],
                        student_outputs[key]
                    )
                    total_hard_loss += tf.reduce_mean(hard_loss)
                    
                    # Soft label (distillation) loss
                    # Convert probabilities back to logits for distillation
                    student_logits = tf.math.log(student_outputs[key] + 1e-10)
                    teacher_logits = tf.math.log(teacher_outputs[key] + 1e-10)
                    
                    soft_loss = self.distillation_loss(student_logits, teacher_logits)
                    total_soft_loss += soft_loss
            
            # Combined loss
            total_loss = (1 - self.alpha) * total_hard_loss + self.alpha * total_soft_loss
        
        # Update student only
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables)
        )
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.hard_loss_tracker.update_state(total_hard_loss)
        self.soft_loss_tracker.update_state(total_soft_loss)
        
        if 'location' in student_outputs and 'location' in y_true:
            self.student_acc_tracker.update_state(
                tf.argmax(y_true['location'], axis=-1),
                student_outputs['location']
            )
        
        return {
            'total_loss': self.total_loss_tracker.result(),
            'hard_loss': self.hard_loss_tracker.result(),
            'soft_loss': self.soft_loss_tracker.result(),
            'student_accuracy': self.student_acc_tracker.result()
        }
    
    def test_step(self, data):
        """Custom test step."""
        x, y_true = data
        
        student_outputs = self.student(x, training=False)
        teacher_outputs = self.teacher(x, training=False)
        
        # Calculate losses
        total_hard_loss = 0
        total_soft_loss = 0
        
        for key in student_outputs:
            if key in teacher_outputs and key in y_true:
                hard_loss = tf.keras.losses.categorical_crossentropy(
                    y_true[key],
                    student_outputs[key]
                )
                total_hard_loss += tf.reduce_mean(hard_loss)
                
                student_logits = tf.math.log(student_outputs[key] + 1e-10)
                teacher_logits = tf.math.log(teacher_outputs[key] + 1e-10)
                soft_loss = self.distillation_loss(student_logits, teacher_logits)
                total_soft_loss += soft_loss
        
        total_loss = (1 - self.alpha) * total_hard_loss + self.alpha * total_soft_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.hard_loss_tracker.update_state(total_hard_loss)
        self.soft_loss_tracker.update_state(total_soft_loss)
        
        return {
            'total_loss': self.total_loss_tracker.result(),
            'hard_loss': self.hard_loss_tracker.result(),
            'soft_loss': self.soft_loss_tracker.result()
        }


def create_compact_student_model(
    num_locations: int = None,
    num_situations: int = None,
    input_shape: Tuple = (None, 80)
) -> Model:
    """
    Create a lightweight student model for deployment.
    
    Architecture is much smaller than teacher but captures
    essential patterns through distillation.
    """
    num_locations = num_locations or LABEL_CONFIG.num_locations
    num_situations = num_situations or LABEL_CONFIG.num_situations
    
    inputs = layers.Input(shape=input_shape, name='mel_input')
    
    # Efficient convolutional encoder
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Shared representation
    shared = layers.Dense(128, activation='relu')(x)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(64, activation='relu')(shared)
    
    # Output heads
    location = layers.Dense(num_locations, activation='softmax', name='location')(shared)
    situation = layers.Dense(num_situations, activation='softmax', name='situation')(shared)
    confidence = layers.Dense(1, activation='sigmoid', name='confidence')(shared)
    emergency = layers.Dense(1, activation='sigmoid', name='emergency')(shared)
    
    model = Model(
        inputs=inputs,
        outputs={
            'location': location,
            'situation': situation,
            'confidence': confidence,
            'emergency': emergency
        },
        name='compact_student'
    )
    
    return model


def create_medium_student_model(
    num_locations: int = None,
    num_situations: int = None,
    input_shape: Tuple = (None, 80)
) -> Model:
    """
    Create a medium-sized student model.
    Balance between efficiency and accuracy.
    """
    num_locations = num_locations or LABEL_CONFIG.num_locations
    num_situations = num_situations or LABEL_CONFIG.num_situations
    
    inputs = layers.Input(shape=input_shape, name='mel_input')
    
    # Convolutional encoder
    x = layers.Conv1D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Simple self-attention
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classifier
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Outputs
    location = layers.Dense(num_locations, activation='softmax', name='location')(x)
    situation = layers.Dense(num_situations, activation='softmax', name='situation')(x)
    confidence = layers.Dense(1, activation='sigmoid', name='confidence')(x)
    emergency = layers.Dense(1, activation='sigmoid', name='emergency')(x)
    
    return Model(
        inputs=inputs,
        outputs={
            'location': location,
            'situation': situation,
            'confidence': confidence,
            'emergency': emergency
        },
        name='medium_student'
    )


def train_with_distillation(
    teacher_model: Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    student_size: str = 'compact',
    epochs: int = 50,
    learning_rate: float = 1e-4,
    save_path: str = None
) -> Tuple[Model, Dict]:
    """
    Train a student model using knowledge distillation.
    
    Args:
        teacher_model: Pre-trained teacher model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        student_size: 'compact' or 'medium'
        epochs: Number of epochs
        learning_rate: Learning rate
        save_path: Path to save student model
        
    Returns:
        Tuple of (trained student model, training history)
    """
    save_path = save_path or f'weights/auralis_{student_size}.keras'
    
    print("\n" + "="*60)
    print("🎓 Knowledge Distillation Training")
    print("="*60)
    print(f"   Teacher params: {teacher_model.count_params():,}")
    
    # Create student
    if student_size == 'compact':
        student = create_compact_student_model()
    else:
        student = create_medium_student_model()
    
    print(f"   Student params: {student.count_params():,}")
    print(f"   Compression ratio: {teacher_model.count_params() / student.count_params():.1f}x")
    print("="*60 + "\n")
    
    # Create distillation wrapper
    distiller = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student,
        temperature=TRAINING_CONFIG.distillation_temperature,
        alpha=TRAINING_CONFIG.distillation_alpha
    )
    
    # Compile
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate)
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_total_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_total_loss',
            factor=0.5,
            patience=5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            save_path.replace('.keras', '_best.keras'),
            monitor='val_student_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/distillation'
        )
    ]
    
    # Train
    history = distiller.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save final student
    student.save(save_path)
    print(f"\n✅ Student model saved to {save_path}")
    
    return student, history.history


class ProgressiveDistillation:
    """
    Progressive knowledge distillation.
    Gradually increases task difficulty during training.
    """
    
    def __init__(
        self,
        teacher_model: Model,
        student_model: Model,
        num_stages: int = 3
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.num_stages = num_stages
        
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs_per_stage: int = 20
    ):
        """Train with progressive difficulty."""
        
        histories = []
        
        for stage in range(self.num_stages):
            print(f"\n📈 Stage {stage + 1}/{self.num_stages}")
            
            # Adjust temperature (higher = easier, lower = harder)
            temperature = 6.0 - stage * 2.0
            
            # Adjust alpha (more hard labels as training progresses)
            alpha = 0.9 - stage * 0.2
            
            distiller = KnowledgeDistillation(
                teacher_model=self.teacher,
                student_model=self.student,
                temperature=temperature,
                alpha=alpha
            )
            
            distiller.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4 * (0.5 ** stage))
            )
            
            history = distiller.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs_per_stage
            )
            
            histories.append(history.history)
        
        return histories


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Knowledge Distillation")
    print("="*60)
    
    # Create dummy teacher
    teacher_input = layers.Input(shape=(None, 80))
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(teacher_input)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    loc = layers.Dense(13, activation='softmax', name='location')(x)
    sit = layers.Dense(15, activation='softmax', name='situation')(x)
    conf = layers.Dense(1, activation='sigmoid', name='confidence')(x)
    emerg = layers.Dense(1, activation='sigmoid', name='emergency')(x)
    
    teacher = Model(
        inputs=teacher_input,
        outputs={'location': loc, 'situation': sit, 'confidence': conf, 'emergency': emerg}
    )
    
    print(f"\n👨‍🏫 Teacher parameters: {teacher.count_params():,}")
    
    # Create compact student
    student = create_compact_student_model()
    print(f"🎓 Compact student parameters: {student.count_params():,}")
    
    # Create medium student
    medium_student = create_medium_student_model()
    print(f"🎓 Medium student parameters: {medium_student.count_params():,}")
    
    # Test distillation model
    distiller = KnowledgeDistillation(teacher, student)
    distiller.compile(optimizer='adam')
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    dummy_input = np.random.randn(batch_size, seq_len, 80).astype(np.float32)
    
    output = distiller(dummy_input)
    print(f"\n📊 Student output shapes:")
    for k, v in output.items():
        print(f"   {k}: {v.shape}")
    
    print("\n✅ Knowledge Distillation test passed!")