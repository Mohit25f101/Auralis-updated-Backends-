# ==============================
# 📄 training/contrastive_learning.py
# ==============================
# Contrastive Learning for Audio Representations
# Upgrade 4: Self-supervised pre-training
# ==============================

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import List, Tuple, Optional, Generator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_CONFIG, MODEL_CONFIG, TRAINING_CONFIG


class ProjectionHead(layers.Layer):
    """
    Projection head for contrastive learning.
    Maps encoder output to contrastive embedding space.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build projection layers
        self.layers_list = []
        for i in range(num_layers - 1):
            self.layers_list.append(layers.Dense(hidden_dim, activation='relu'))
            self.layers_list.append(layers.BatchNormalization())
        
        # Final projection (no activation)
        self.layers_list.append(layers.Dense(output_dim))
        
        # L2 normalization
        self.normalize = layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1)
        )
        
    def call(self, x, training=False):
        for layer in self.layers_list:
            if isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return self.normalize(x)


class AudioContrastiveModel(Model):
    """
    Contrastive learning model for audio (SimCLR-style).
    Learns robust audio representations without labels.
    
    Training process:
    1. Take an audio sample
    2. Create two different augmented views
    3. Encode both views
    4. Train to maximize similarity between views of same audio
       while minimizing similarity with other samples
    """
    
    def __init__(
        self,
        encoder: Model,
        projection_dim: int = None,
        temperature: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.encoder = encoder
        self.temperature = temperature or MODEL_CONFIG.contrastive_temperature
        projection_dim = projection_dim or MODEL_CONFIG.contrastive_projection_dim
        
        # Projection head
        self.projection_head = ProjectionHead(
            hidden_dim=512,
            output_dim=projection_dim,
            num_layers=3
        )
        
        # Metrics
        self.loss_tracker = tf.keras.metrics.Mean(name='contrastive_loss')
        self.accuracy_tracker = tf.keras.metrics.Mean(name='contrastive_accuracy')
        
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]
    
    def contrastive_loss(self, z_i: tf.Tensor, z_j: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
        
        Args:
            z_i: Embeddings from first augmented view [batch, dim]
            z_j: Embeddings from second augmented view [batch, dim]
            
        Returns:
            Tuple of (loss, accuracy)
        """
        batch_size = tf.shape(z_i)[0]
        
        # Concatenate embeddings: [2*batch, dim]
        z = tf.concat([z_i, z_j], axis=0)
        
        # Compute similarity matrix: [2*batch, 2*batch]
        sim_matrix = tf.matmul(z, z, transpose_b=True)
        sim_matrix = sim_matrix / self.temperature
        
        # Create labels: positive pairs are at (i, i+batch_size) and (i+batch_size, i)
        labels = tf.range(batch_size)
        labels = tf.concat([labels + batch_size, labels], axis=0)
        
        # Mask out self-similarity (diagonal)
        mask = tf.eye(2 * batch_size) * -1e9
        sim_matrix = sim_matrix + mask
        
        # Cross entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=sim_matrix
        )
        
        # Calculate accuracy
        predictions = tf.argmax(sim_matrix, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(predictions == tf.cast(labels, tf.int64), tf.float32)
        )
        
        return tf.reduce_mean(loss), accuracy
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Tuple of (view1, view2) augmented spectrograms
            
        Returns:
            Tuple of projected embeddings (z1, z2)
        """
        x1, x2 = inputs
        
        # Encode both views
        h1 = self.encoder(x1, training=training)
        h2 = self.encoder(x2, training=training)
        
        # Project to contrastive space
        z1 = self.projection_head(h1, training=training)
        z2 = self.projection_head(h2, training=training)
        
        return z1, z2
    
    def train_step(self, data):
        """Custom training step."""
        x1, x2 = data
        
        with tf.GradientTape() as tape:
            z1, z2 = self((x1, x2), training=True)
            loss, accuracy = self.contrastive_loss(z1, z2)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        
        return {
            'contrastive_loss': self.loss_tracker.result(),
            'contrastive_accuracy': self.accuracy_tracker.result()
        }
    
    def test_step(self, data):
        """Custom test step."""
        x1, x2 = data
        
        z1, z2 = self((x1, x2), training=False)
        loss, accuracy = self.contrastive_loss(z1, z2)
        
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        
        return {
            'contrastive_loss': self.loss_tracker.result(),
            'contrastive_accuracy': self.accuracy_tracker.result()
        }
    
    def get_encoder(self) -> Model:
        """Get the pre-trained encoder for downstream tasks."""
        return self.encoder


class ContrastiveDataGenerator:
    """
    Generate pairs of augmented views for contrastive learning.
    """
    
    def __init__(
        self,
        audio_files: List[str],
        augmentor,
        mel_extractor,
        batch_size: int = None,
        max_length: int = 300
    ):
        """
        Initialize generator.
        
        Args:
            audio_files: List of audio file paths
            augmentor: AudioAugmentor instance
            mel_extractor: MelSpectrogramExtractor instance
            batch_size: Batch size
            max_length: Maximum spectrogram length
        """
        self.audio_files = audio_files
        self.augmentor = augmentor
        self.mel_extractor = mel_extractor
        self.batch_size = batch_size or TRAINING_CONFIG.contrastive_batch_size
        self.max_length = max_length
        
    def __len__(self):
        return len(self.audio_files) // self.batch_size
    
    def generate(self) -> Generator:
        """Generate batches of augmented pairs."""
        import librosa
        
        while True:
            # Random sample indices
            indices = np.random.choice(
                len(self.audio_files),
                size=self.batch_size,
                replace=False
            )
            
            view1_batch = []
            view2_batch = []
            
            for idx in indices:
                try:
                    # Load audio
                    audio, sr = librosa.load(
                        self.audio_files[idx],
                        sr=AUDIO_CONFIG.sample_rate
                    )
                    
                    # Create two different augmented views
                    aug1 = self.augmentor.augment(audio, intensity='heavy')
                    aug2 = self.augmentor.augment(audio, intensity='heavy')
                    
                    # Extract mel spectrograms
                    mel1 = self.mel_extractor.extract(aug1)
                    mel2 = self.mel_extractor.extract(aug2)
                    
                    view1_batch.append(mel1)
                    view2_batch.append(mel2)
                    
                except Exception as e:
                    # Skip problematic files
                    continue
            
            if len(view1_batch) < self.batch_size // 2:
                continue
            
            # Pad batches
            view1 = self._pad_batch(view1_batch)
            view2 = self._pad_batch(view2_batch)
            
            yield (view1, view2)
    
    def _pad_batch(self, specs: List[np.ndarray]) -> np.ndarray:
        """Pad spectrograms to same length."""
        max_len = min(max(s.shape[0] for s in specs), self.max_length)
        
        padded = []
        for s in specs:
            if s.shape[0] > max_len:
                # Random crop
                start = np.random.randint(0, s.shape[0] - max_len)
                s = s[start:start + max_len]
            elif s.shape[0] < max_len:
                # Pad
                pad_width = ((0, max_len - s.shape[0]), (0, 0))
                s = np.pad(s, pad_width, mode='constant')
            padded.append(s)
        
        return np.array(padded, dtype=np.float32)
    
    def as_dataset(self) -> tf.data.Dataset:
        """Convert to tf.data.Dataset."""
        return tf.data.Dataset.from_generator(
            self.generate,
            output_signature=(
                tf.TensorSpec(shape=(None, None, AUDIO_CONFIG.n_mels), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, AUDIO_CONFIG.n_mels), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)


def pretrain_contrastive(
    audio_files: List[str],
    encoder: Model,
    augmentor,
    mel_extractor,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = 1e-3,
    save_path: str = None
) -> Model:
    """
    Pre-train encoder with contrastive learning.
    
    Args:
        audio_files: List of audio file paths
        encoder: Encoder model to pre-train
        augmentor: AudioAugmentor instance
        mel_extractor: MelSpectrogramExtractor instance
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save pre-trained encoder
        
    Returns:
        Pre-trained encoder
    """
    epochs = epochs or TRAINING_CONFIG.contrastive_epochs
    batch_size = batch_size or TRAINING_CONFIG.contrastive_batch_size
    save_path = save_path or 'weights/encoder_pretrained.keras'
    
    print("\n" + "="*60)
    print("🔄 Starting Contrastive Pre-training")
    print("="*60)
    print(f"   Audio files: {len(audio_files)}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print("="*60 + "\n")
    
    # Create contrastive model
    contrastive_model = AudioContrastiveModel(encoder)
    
    # Compile
    contrastive_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate)
    )
    
    # Create data generator
    data_gen = ContrastiveDataGenerator(
        audio_files=audio_files,
        augmentor=augmentor,
        mel_extractor=mel_extractor,
        batch_size=batch_size
    )
    
    steps_per_epoch = len(audio_files) // batch_size
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='contrastive_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='contrastive_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/contrastive',
            histogram_freq=1
        )
    ]
    
    # Train
    history = contrastive_model.fit(
        data_gen.generate(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )
    
    # Save pre-trained encoder
    encoder = contrastive_model.get_encoder()
    encoder.save(save_path)
    print(f"\n✅ Pre-trained encoder saved to {save_path}")
    
    return encoder


class MomentumContrastiveModel(Model):
    """
    MoCo-style contrastive learning with momentum encoder.
    More memory efficient than SimCLR for large batch sizes.
    """
    
    def __init__(
        self,
        encoder: Model,
        projection_dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.encoder_q = encoder
        self.encoder_k = tf.keras.models.clone_model(encoder)
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        
        # Projection heads
        self.proj_q = ProjectionHead(output_dim=projection_dim)
        self.proj_k = ProjectionHead(output_dim=projection_dim)
        
        # Initialize key encoder with query encoder weights
        self._copy_weights()
        
        # Queue for negative samples
        self.queue = tf.Variable(
            tf.nn.l2_normalize(tf.random.normal([projection_dim, queue_size]), axis=0),
            trainable=False
        )
        self.queue_ptr = tf.Variable(0, trainable=False)
        
    def _copy_weights(self):
        """Copy weights from query encoder to key encoder."""
        for q_layer, k_layer in zip(
            self.encoder_q.layers + self.proj_q.layers,
            self.encoder_k.layers + self.proj_k.layers
        ):
            k_layer.set_weights(q_layer.get_weights())
    
    @tf.function
    def _momentum_update(self):
        """Momentum update of key encoder."""
        for q_layer, k_layer in zip(
            self.encoder_q.trainable_variables + self.proj_q.trainable_variables,
            self.encoder_k.trainable_variables + self.proj_k.trainable_variables
        ):
            k_layer.assign(self.momentum * k_layer + (1 - self.momentum) * q_layer)
    
    @tf.function
    def _dequeue_and_enqueue(self, keys):
        """Update queue with new keys."""
        batch_size = tf.shape(keys)[0]
        
        ptr = self.queue_ptr
        
        # Replace oldest keys
        indices = tf.range(ptr, ptr + batch_size) % self.queue_size
        self.queue.scatter_nd_update(
            tf.expand_dims(indices, 1),
            tf.transpose(keys)
        )
        
        # Update pointer
        self.queue_ptr.assign((ptr + batch_size) % self.queue_size)
    
    def call(self, inputs, training=False):
        x_q, x_k = inputs
        
        # Query encoding
        q = self.encoder_q(x_q, training=training)
        q = self.proj_q(q, training=training)
        
        # Key encoding (no gradient)
        k = self.encoder_k(x_k, training=False)
        k = self.proj_k(k, training=False)
        
        return q, k
    
    def train_step(self, data):
        x_q, x_k = data
        
        with tf.GradientTape() as tape:
            q, k = self((x_q, x_k), training=True)
            
            # Positive logits
            l_pos = tf.reduce_sum(q * k, axis=1, keepdims=True)
            
            # Negative logits from queue
            l_neg = tf.matmul(q, self.queue)
            
            # Logits
            logits = tf.concat([l_pos, l_neg], axis=1) / self.temperature
            
            # Labels (positive is always first)
            labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int64)
            
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
            loss = tf.reduce_mean(loss)
        
        # Update query encoder
        gradients = tape.gradient(
            loss,
            self.encoder_q.trainable_variables + self.proj_q.trainable_variables
        )
        self.optimizer.apply_gradients(zip(
            gradients,
            self.encoder_q.trainable_variables + self.proj_q.trainable_variables
        ))
        
        # Momentum update key encoder
        self._momentum_update()
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return {'loss': loss}


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Contrastive Learning")
    print("="*60)
    
    # Create dummy encoder
    encoder = tf.keras.Sequential([
        layers.Input(shape=(None, 80)),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(256)
    ])
    
    # Create contrastive model
    model = AudioContrastiveModel(encoder)
    model.compile(optimizer='adam')
    
    # Create dummy data
    batch_size = 8
    seq_len = 100
    view1 = np.random.randn(batch_size, seq_len, 80).astype(np.float32)
    view2 = np.random.randn(batch_size, seq_len, 80).astype(np.float32)
    
    # Forward pass
    z1, z2 = model((view1, view2), training=False)
    print(f"\n📊 Embedding shapes: z1={z1.shape}, z2={z2.shape}")
    
    # Test loss calculation
    loss, acc = model.contrastive_loss(z1, z2)
    print(f"📉 Initial loss: {loss:.4f}, accuracy: {acc:.4f}")
    
    # Test training step
    result = model.train_step((view1, view2))
    print(f"🏋️ Training step: loss={result['contrastive_loss']:.4f}")
    
    print("\n✅ Contrastive Learning test passed!")