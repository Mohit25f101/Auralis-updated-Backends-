# ==============================
# 📄 data/data_loader.py
# ==============================
# Efficient Data Loading Pipeline
# TFRecord and tf.data integration
# ==============================

import tensorflow as tf
import numpy as np
import librosa
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_CONFIG, LABEL_CONFIG, TRAINING_CONFIG, DATASET_DIR


class TFRecordLoader:
    """
    Create and load TFRecord datasets for efficient training.
    """
    
    def __init__(
        self,
        sample_rate: int = None,
        n_mels: int = None,
        max_duration: float = None
    ):
        """
        Initialize TFRecord loader.
        
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            max_duration: Maximum audio duration
        """
        self.sample_rate = sample_rate or AUDIO_CONFIG.sample_rate
        self.n_mels = n_mels or AUDIO_CONFIG.n_mels
        self.max_duration = max_duration or AUDIO_CONFIG.max_duration
        
        # Feature description for parsing
        self.feature_description = {
            'audio': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'mel': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'mel_shape': tf.io.FixedLenFeature([2], tf.int64),
            'location_idx': tf.io.FixedLenFeature([], tf.int64),
            'situation_idx': tf.io.FixedLenFeature([], tf.int64),
            'duration': tf.io.FixedLenFeature([], tf.float32),
        }
    
    def create_tfrecord(
        self,
        audio_files: List[str],
        labels: List[Dict],
        output_path: str,
        extract_mel: bool = True
    ):
        """
        Create TFRecord file from audio files.
        
        Args:
            audio_files: List of audio file paths
            labels: List of label dictionaries
            output_path: Output TFRecord path
            extract_mel: Whether to extract mel spectrograms
        """
        print(f"📦 Creating TFRecord: {output_path}")
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for i, (audio_path, label) in enumerate(zip(audio_files, labels)):
                try:
                    # Load audio
                    audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                    
                    # Truncate if needed
                    max_samples = int(self.max_duration * self.sample_rate)
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                    
                    # Extract mel spectrogram
                    if extract_mel:
                        mel = librosa.feature.melspectrogram(
                            y=audio,
                            sr=self.sample_rate,
                            n_mels=self.n_mels,
                            n_fft=AUDIO_CONFIG.n_fft,
                            hop_length=AUDIO_CONFIG.hop_length
                        )
                        mel = np.log(mel + 1e-6).T.flatten()
                        mel_shape = [mel.shape[0] // self.n_mels, self.n_mels]
                    else:
                        mel = np.zeros(1)
                        mel_shape = [1, 1]
                    
                    # Create feature dict
                    feature = {
                        'audio': tf.train.Feature(
                            float_list=tf.train.FloatList(value=audio)
                        ),
                        'mel': tf.train.Feature(
                            float_list=tf.train.FloatList(value=mel)
                        ),
                        'mel_shape': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=mel_shape)
                        ),
                        'location_idx': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label['location_idx']])
                        ),
                        'situation_idx': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label['situation_idx']])
                        ),
                        'duration': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[len(audio) / self.sample_rate])
                        ),
                    }
                    
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    
                except Exception as e:
                    print(f"⚠️ Error processing {audio_path}: {e}")
                    continue
        
        print(f"✅ Created TFRecord with {len(audio_files)} samples")
    
    def parse_example(self, serialized_example):
        """Parse a single TFRecord example"""
        example = tf.io.parse_single_example(serialized_example, self.feature_description)
        
        # Reshape mel spectrogram
        mel_shape = example['mel_shape']
        mel = tf.reshape(example['mel'], mel_shape)
        
        # Create label dict
        labels = {
            'location': tf.one_hot(example['location_idx'], LABEL_CONFIG.num_locations),
            'situation': tf.one_hot(example['situation_idx'], LABEL_CONFIG.num_situations),
        }
        
        return mel, labels
    
    def load_dataset(
        self,
        tfrecord_path: str,
        batch_size: int = None,
        shuffle: bool = True,
        cache: bool = True
    ) -> tf.data.Dataset:
        """
        Load TFRecord dataset.
        
        Args:
            tfrecord_path: Path to TFRecord file
            batch_size: Batch size
            shuffle: Whether to shuffle
            cache: Whether to cache dataset
            
        Returns:
            tf.data.Dataset
        """
        batch_size = batch_size or TRAINING_CONFIG.batch_size
        
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(self.parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        
        if cache:
            dataset = dataset.cache()
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [None, self.n_mels],
                {
                    'location': [LABEL_CONFIG.num_locations],
                    'situation': [LABEL_CONFIG.num_situations],
                }
            )
        )
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class AudioDataPipeline:
    """
    Complete data pipeline for audio scene classification.
    Handles loading, preprocessing, augmentation, and batching.
    """
    
    def __init__(
        self,
        metadata_path: str = None,
        audio_dir: str = None,
        sample_rate: int = None,
        n_mels: int = None,
        augment: bool = True
    ):
        """
        Initialize data pipeline.
        
        Args:
            metadata_path: Path to metadata CSV
            audio_dir: Base directory for audio files
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            augment: Whether to apply augmentation
        """
        self.metadata_path = metadata_path or str(DATASET_DIR / 'metadata' / 'dataset.csv')
        self.audio_dir = Path(audio_dir) if audio_dir else DATASET_DIR / 'audio'
        self.sample_rate = sample_rate or AUDIO_CONFIG.sample_rate
        self.n_mels = n_mels or AUDIO_CONFIG.n_mels
        self.augment = augment
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, List]:
        """Load and organize metadata by split"""
        import pandas as pd
        
        if not os.path.exists(self.metadata_path):
            print(f"⚠️ Metadata not found at {self.metadata_path}")
            return {'train': [], 'val': [], 'test': []}
        
        df = pd.read_csv(self.metadata_path)
        
        metadata = {
            'train': df[df['split'] == 'train'].to_dict('records'),
            'val': df[df['split'] == 'val'].to_dict('records'),
            'test': df[df['split'] == 'test'].to_dict('records'),
        }
        
        print(f"📊 Loaded metadata:")
        for split, items in metadata.items():
            print(f"   {split}: {len(items)} samples")
        
        return metadata
    
    def load_audio(self, filepath: str) -> np.ndarray:
        """Load and preprocess audio file"""
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    
    def extract_mel(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=AUDIO_CONFIG.n_fft,
            hop_length=AUDIO_CONFIG.hop_length
        )
        
        # Log scale and normalize
        mel = np.log(mel + 1e-6)
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-8)
        
        return mel.T  # [time, mels]
    
    def create_tf_dataset(
        self,
        split: str,
        batch_size: int = None,
        shuffle: bool = None,
        augment: bool = None
    ) -> tf.data.Dataset:
        """
        Create tf.data.Dataset for a split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            batch_size: Batch size
            shuffle: Whether to shuffle
            augment: Whether to augment
            
        Returns:
            tf.data.Dataset
        """
        batch_size = batch_size or TRAINING_CONFIG.batch_size
        shuffle = shuffle if shuffle is not None else (split == 'train')
        augment = augment if augment is not None else (split == 'train' and self.augment)
        
        items = self.metadata[split]
        
        if not items:
            print(f"⚠️ No data for split: {split}")
            return None
        
        # Create generators
        def generator():
            for item in items:
                try:
                    audio_path = str(self.audio_dir / split / item['filename'])
                    audio = self.load_audio(audio_path)
                    mel = self.extract_mel(audio)
                    
                    yield (
                        mel,
                        {
                            'location': item['location_idx'],
                            'situation': item['situation_idx'],
                        }
                    )
                except Exception as e:
                    continue
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, self.n_mels), dtype=tf.float32),
                {
                    'location': tf.TensorSpec(shape=(), dtype=tf.int64),
                    'situation': tf.TensorSpec(shape=(), dtype=tf.int64),
                }
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(items), 1000))
        
        # Convert labels to one-hot
        def to_one_hot(mel, labels):
            return (
                mel,
                {
                    'location': tf.one_hot(labels['location'], LABEL_CONFIG.num_locations),
                    'situation': tf.one_hot(labels['situation'], LABEL_CONFIG.num_situations),
                }
            )
        
        dataset = dataset.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch with padding
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                [None, self.n_mels],
                {
                    'location': [LABEL_CONFIG.num_locations],
                    'situation': [LABEL_CONFIG.num_situations],
                }
            )
        )
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def create_train_dataset(
    metadata_path: str = None,
    audio_dir: str = None,
    batch_size: int = None
) -> tf.data.Dataset:
    """Convenience function to create training dataset"""
    pipeline = AudioDataPipeline(metadata_path, audio_dir)
    return pipeline.create_tf_dataset('train', batch_size=batch_size)


def create_val_dataset(
    metadata_path: str = None,
    audio_dir: str = None,
    batch_size: int = None
) -> tf.data.Dataset:
    """Convenience function to create validation dataset"""
    pipeline = AudioDataPipeline(metadata_path, audio_dir)
    return pipeline.create_tf_dataset('val', batch_size=batch_size, augment=False)


def create_test_dataset(
    metadata_path: str = None,
    audio_dir: str = None,
    batch_size: int = None
) -> tf.data.Dataset:
    """Convenience function to create test dataset"""
    pipeline = AudioDataPipeline(metadata_path, audio_dir)
    return pipeline.create_tf_dataset('test', batch_size=batch_size, shuffle=False, augment=False)


class DatasetInfo:
    """
    Get information about a dataset.
    """
    
    @staticmethod
    def analyze(metadata_path: str) -> Dict:
        """Analyze dataset and return statistics"""
        import pandas as pd
        
        df = pd.read_csv(metadata_path)
        
        info = {
            'total_samples': len(df),
            'splits': df['split'].value_counts().to_dict(),
            'locations': df['location'].value_counts().to_dict(),
            'situations': df['situation'].value_counts().to_dict(),
            'duration_stats': {
                'total_hours': df['duration'].sum() / 3600,
                'mean_seconds': df['duration'].mean(),
                'min_seconds': df['duration'].min(),
                'max_seconds': df['duration'].max(),
            }
        }
        
        return info
    
    @staticmethod
    def print_info(metadata_path: str):
        """Print dataset information"""
        info = DatasetInfo.analyze(metadata_path)
        
        print("\n" + "="*60)
        print("📊 DATASET INFORMATION")
        print("="*60)
        print(f"\n📁 Total: {info['total_samples']} samples")
        print(f"⏱️  Duration: {info['duration_stats']['total_hours']:.2f} hours")
        
        print("\n📂 Splits:")
        for split, count in info['splits'].items():
            print(f"   {split}: {count}")
        
        print("\n📍 Top Locations:")
        for loc, count in list(info['locations'].items())[:5]:
            print(f"   {loc}: {count}")
        
        print("="*60)


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Data Loader")
    print("="*60)
    
    # Test TFRecordLoader
    print("\n🔧 Testing TFRecordLoader...")
    loader = TFRecordLoader()
    print(f"   Sample rate: {loader.sample_rate}")
    print(f"   Mel bands: {loader.n_mels}")
    
    # Test AudioDataPipeline (without actual data)
    print("\n🔧 Testing AudioDataPipeline...")
    try:
        pipeline = AudioDataPipeline()
        print("   Pipeline initialized successfully")
    except Exception as e:
        print(f"   Pipeline initialization (expected without data): {e}")
    
    print("\n✅ Data Loader test complete!")