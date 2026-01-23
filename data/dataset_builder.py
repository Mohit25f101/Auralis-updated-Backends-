# ==============================
# 📄 data/dataset_builder.py
# ==============================
# Comprehensive Dataset Builder
# Upgrade 10: Build high-quality custom dataset
# ==============================

import os
import json
import hashlib
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import soundfile as sf
import requests
import zipfile
import tarfile
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    AUDIO_CONFIG, LABEL_CONFIG, DATASET_DIR, 
    AUDIO_DIR, FEATURES_DIR, METADATA_DIR
)


class AuralisDatasetBuilder:
    """
    Build a comprehensive audio scene dataset from multiple sources.
    Handles data collection, preprocessing, and organization.
    """
    
    def __init__(
        self,
        output_dir: str = None,
        sample_rate: int = None
    ):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Output directory for dataset
            sample_rate: Target sample rate for audio
        """
        self.output_dir = Path(output_dir) if output_dir else DATASET_DIR
        self.sample_rate = sample_rate or AUDIO_CONFIG.sample_rate
        self.metadata = []
        
        # Setup directories
        self._setup_directories()
        
        # Location and situation mappings
        self.locations = LABEL_CONFIG.locations
        self.situations = LABEL_CONFIG.situations
        
    def _setup_directories(self):
        """Create organized directory structure"""
        directories = [
            self.output_dir / 'audio' / 'train',
            self.output_dir / 'audio' / 'val',
            self.output_dir / 'audio' / 'test',
            self.output_dir / 'features' / 'train',
            self.output_dir / 'features' / 'val',
            self.output_dir / 'features' / 'test',
            self.output_dir / 'metadata',
            self.output_dir / 'raw',
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Dataset directories created at: {self.output_dir}")
    
    def add_audio_files(
        self,
        source_path: str,
        location: str,
        situation: str,
        split: str = 'train',
        recursive: bool = True
    ) -> int:
        """
        Add audio files from a source directory.
        
        Args:
            source_path: Path to source audio files
            location: Location label for these files
            situation: Situation label for these files
            split: Dataset split (train/val/test)
            recursive: Whether to search subdirectories
            
        Returns:
            Number of files added
        """
        source = Path(source_path)
        
        if not source.exists():
            print(f"❌ Source path not found: {source_path}")
            return 0
        
        # Find audio files
        extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
        audio_files = []
        
        for ext in extensions:
            if recursive:
                audio_files.extend(source.rglob(ext))
            else:
                audio_files.extend(source.glob(ext))
        
        print(f"📁 Found {len(audio_files)} audio files in {source_path}")
        
        added = 0
        for audio_file in tqdm(audio_files, desc="Processing files"):
            try:
                self._process_audio_file(audio_file, location, situation, split)
                added += 1
            except Exception as e:
                print(f"⚠️ Error processing {audio_file}: {e}")
                
        print(f"✅ Added {added} files")
        return added
    
    def _process_audio_file(
        self,
        audio_path: Path,
        location: str,
        situation: str,
        split: str
    ):
        """Process and save a single audio file"""
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        
        # Skip very short or silent audio
        if len(audio) < self.sample_rate * AUDIO_CONFIG.min_duration:
            return
        
        if np.max(np.abs(audio)) < 0.01:
            return
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
        
        # Truncate if too long
        max_samples = int(AUDIO_CONFIG.max_duration * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Generate unique filename
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()[:12]
        location_short = location.replace('/', '_').replace(' ', '_')[:20]
        situation_short = situation.replace('/', '_').replace(' ', '_')[:20]
        filename = f"{location_short}_{situation_short}_{audio_hash}.wav"
        
        # Save to appropriate split directory
        output_path = self.output_dir / 'audio' / split / filename
        sf.write(str(output_path), audio, self.sample_rate)
        
        # Add metadata entry
        self.metadata.append({
            'filename': filename,
            'split': split,
            'location': location,
            'situation': situation,
            'location_idx': self.locations.index(location) if location in self.locations else -1,
            'situation_idx': self.situations.index(situation) if situation in self.situations else -1,
            'duration': len(audio) / self.sample_rate,
            'original_path': str(audio_path),
            'sample_rate': self.sample_rate
        })
    
    def add_labeled_csv(
        self,
        csv_path: str,
        audio_column: str,
        location_column: str,
        situation_column: str,
        audio_base_path: str = None,
        split: str = 'train'
    ) -> int:
        """
        Add audio files using a CSV file with labels.
        
        Args:
            csv_path: Path to CSV file
            audio_column: Column name containing audio file paths
            location_column: Column name containing location labels
            situation_column: Column name containing situation labels
            audio_base_path: Base path for audio files
            split: Dataset split
            
        Returns:
            Number of files added
        """
        df = pd.read_csv(csv_path)
        
        print(f"📊 Loading {len(df)} entries from CSV")
        
        added = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
            try:
                audio_path = row[audio_column]
                if audio_base_path:
                    audio_path = os.path.join(audio_base_path, audio_path)
                
                location = row[location_column]
                situation = row[situation_column]
                
                self._process_audio_file(
                    Path(audio_path),
                    location,
                    situation,
                    split
                )
                added += 1
            except Exception as e:
                continue
                
        print(f"✅ Added {added} files from CSV")
        return added
    
    def generate_synthetic_data(
        self,
        n_samples: int = 1000,
        split: str = 'train'
    ) -> int:
        """
        Generate synthetic audio scenes by combining sounds.
        
        Args:
            n_samples: Number of synthetic samples to generate
            split: Dataset split
            
        Returns:
            Number of samples generated
        """
        print(f"🔄 Generating {n_samples} synthetic audio samples...")
        
        generator = SyntheticAudioGenerator(self.sample_rate)
        
        for i in tqdm(range(n_samples), desc="Generating synthetic data"):
            # Random scene parameters
            location = np.random.choice(self.locations)
            situation = np.random.choice(self.situations)
            
            # Generate synthetic audio
            audio = generator.generate_scene(location, situation)
            
            # Generate filename
            filename = f"synthetic_{location.replace(' ', '_')}_{situation.replace(' ', '_')}_{i:06d}.wav"
            
            # Save
            output_path = self.output_dir / 'audio' / split / filename
            sf.write(str(output_path), audio, self.sample_rate)
            
            # Add metadata
            self.metadata.append({
                'filename': filename,
                'split': split,
                'location': location,
                'situation': situation,
                'location_idx': self.locations.index(location) if location in self.locations else -1,
                'situation_idx': self.situations.index(situation) if situation in self.situations else -1,
                'duration': len(audio) / self.sample_rate,
                'original_path': 'synthetic',
                'sample_rate': self.sample_rate,
                'synthetic': True
            })
        
        print(f"✅ Generated {n_samples} synthetic samples")
        return n_samples
    
    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True
    ):
        """
        Split existing data into train/val/test sets.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            stratify: Whether to stratify by labels
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # Get all files currently labeled as 'train'
        train_files = [m for m in self.metadata if m['split'] == 'train']
        
        if stratify:
            # Group by location + situation
            groups = {}
            for m in train_files:
                key = (m['location'], m['situation'])
                if key not in groups:
                    groups[key] = []
                groups[key].append(m)
            
            # Split each group
            for key, items in groups.items():
                np.random.shuffle(items)
                n = len(items)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                for i, item in enumerate(items):
                    if i < n_train:
                        item['split'] = 'train'
                    elif i < n_train + n_val:
                        item['split'] = 'val'
                    else:
                        item['split'] = 'test'
        else:
            np.random.shuffle(train_files)
            n = len(train_files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            for i, item in enumerate(train_files):
                if i < n_train:
                    item['split'] = 'train'
                elif i < n_train + n_val:
                    item['split'] = 'val'
                else:
                    item['split'] = 'test'
        
        # Move files to correct directories
        self._reorganize_files()
        
        print("✅ Dataset split complete")
        self.print_statistics()
    
    def _reorganize_files(self):
        """Move files to their correct split directories"""
        for item in self.metadata:
            src = self.output_dir / 'audio' / 'train' / item['filename']
            dst = self.output_dir / 'audio' / item['split'] / item['filename']
            
            if src.exists() and src != dst:
                src.rename(dst)
    
    def extract_features(
        self,
        feature_type: str = 'mel',
        n_jobs: int = 4
    ):
        """
        Extract features from all audio files.
        
        Args:
            feature_type: Type of features ('mel', 'mfcc', 'both')
            n_jobs: Number of parallel jobs
        """
        print(f"🔄 Extracting {feature_type} features...")
        
        def process_file(item):
            try:
                audio_path = self.output_dir / 'audio' / item['split'] / item['filename']
                audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
                
                features = {}
                
                if feature_type in ['mel', 'both']:
                    mel = librosa.feature.melspectrogram(
                        y=audio,
                        sr=sr,
                        n_mels=AUDIO_CONFIG.n_mels,
                        n_fft=AUDIO_CONFIG.n_fft,
                        hop_length=AUDIO_CONFIG.hop_length
                    )
                    features['mel'] = np.log(mel + 1e-6).T
                
                if feature_type in ['mfcc', 'both']:
                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sr,
                        n_mfcc=40
                    )
                    features['mfcc'] = mfcc.T
                
                # Save features
                feature_path = self.output_dir / 'features' / item['split'] / f"{item['filename']}.npz"
                np.savez_compressed(str(feature_path), **features)
                
                return True
            except Exception as e:
                return False
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_file, item) for item in self.metadata]
            
            success = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
                if future.result():
                    success += 1
        
        print(f"✅ Extracted features for {success}/{len(self.metadata)} files")
    
    def save_metadata(self):
        """Save dataset metadata to files"""
        # Save as CSV
        df = pd.DataFrame(self.metadata)
        csv_path = self.output_dir / 'metadata' / 'dataset.csv'
        df.to_csv(str(csv_path), index=False)
        
        # Save as JSON
        json_path = self.output_dir / 'metadata' / 'dataset.json'
        with open(str(json_path), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save statistics
        stats = self.get_statistics()
        stats_path = self.output_dir / 'metadata' / 'statistics.json'
        with open(str(stats_path), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save label mappings
        labels_path = self.output_dir / 'metadata' / 'labels.json'
        with open(str(labels_path), 'w') as f:
            json.dump({
                'locations': self.locations,
                'situations': self.situations
            }, f, indent=2)
        
        print(f"✅ Metadata saved to {self.output_dir / 'metadata'}")
    
    def load_metadata(self):
        """Load existing metadata"""
        csv_path = self.output_dir / 'metadata' / 'dataset.csv'
        if csv_path.exists():
            df = pd.read_csv(str(csv_path))
            self.metadata = df.to_dict('records')
            print(f"✅ Loaded {len(self.metadata)} metadata entries")
        else:
            print("⚠️ No existing metadata found")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        df = pd.DataFrame(self.metadata)
        
        stats = {
            'total_samples': len(self.metadata),
            'total_duration_hours': df['duration'].sum() / 3600 if 'duration' in df else 0,
            'splits': df['split'].value_counts().to_dict() if 'split' in df else {},
            'locations': df['location'].value_counts().to_dict() if 'location' in df else {},
            'situations': df['situation'].value_counts().to_dict() if 'situation' in df else {},
            'synthetic_count': df['synthetic'].sum() if 'synthetic' in df else 0,
            'avg_duration': df['duration'].mean() if 'duration' in df else 0,
        }
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        print(f"\n📁 Total Samples: {stats['total_samples']}")
        print(f"⏱️  Total Duration: {stats['total_duration_hours']:.2f} hours")
        print(f"📏 Avg Duration: {stats['avg_duration']:.2f} seconds")
        
        print("\n📂 Splits:")
        for split, count in stats['splits'].items():
            print(f"   {split}: {count}")
        
        print("\n📍 Locations (top 5):")
        for loc, count in list(stats['locations'].items())[:5]:
            print(f"   {loc}: {count}")
        
        print("\n🎯 Situations (top 5):")
        for sit, count in list(stats['situations'].items())[:5]:
            print(f"   {sit}: {count}")
        
        print("="*60 + "\n")


class SyntheticAudioGenerator:
    """
    Generate synthetic audio scenes by combining various sound elements.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        
    def generate_scene(
        self,
        location: str,
        situation: str,
        duration: float = 5.0
    ) -> np.ndarray:
        """
        Generate a synthetic audio scene.
        
        Args:
            location: Location type
            situation: Situation type
            duration: Duration in seconds
            
        Returns:
            Synthetic audio waveform
        """
        n_samples = int(duration * self.sr)
        audio = np.zeros(n_samples, dtype=np.float32)
        
        # Add base ambient noise
        audio += self._generate_ambient(location, n_samples)
        
        # Add location-specific sounds
        audio += self._generate_location_sounds(location, n_samples)
        
        # Add situation-specific elements
        audio += self._generate_situation_sounds(situation, n_samples)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
        
        return audio
    
    def _generate_ambient(self, location: str, n_samples: int) -> np.ndarray:
        """Generate ambient background noise"""
        location_lower = location.lower()
        
        if 'airport' in location_lower or 'station' in location_lower:
            # Large indoor space reverb
            noise = np.random.randn(n_samples) * 0.05
            # Add low rumble
            t = np.linspace(0, n_samples / self.sr, n_samples)
            rumble = 0.03 * np.sin(2 * np.pi * 50 * t)
            return noise + rumble
        
        elif 'street' in location_lower or 'road' in location_lower:
            # Traffic-like noise
            noise = np.random.randn(n_samples) * 0.08
            return self._apply_lowpass(noise, 2000)
        
        elif 'home' in location_lower or 'office' in location_lower:
            # Quiet indoor
            return np.random.randn(n_samples) * 0.02
        
        else:
            # Generic ambient
            return np.random.randn(n_samples) * 0.04
    
    def _generate_location_sounds(self, location: str, n_samples: int) -> np.ndarray:
        """Generate location-specific sounds"""
        audio = np.zeros(n_samples)
        location_lower = location.lower()
        
        if 'airport' in location_lower:
            # PA announcement beeps
            audio += self._generate_beeps(n_samples, freq=880, n_beeps=2)
            # Distant aircraft
            audio += self._generate_aircraft_noise(n_samples) * 0.1
            
        elif 'railway' in location_lower or 'train' in location_lower:
            # Train horn in distance
            if np.random.random() < 0.3:
                audio += self._generate_horn(n_samples, freq=300)
            # Platform sounds
            audio += self._generate_crowd_murmur(n_samples) * 0.15
            
        elif 'hospital' in location_lower:
            # Quiet beeps
            audio += self._generate_beeps(n_samples, freq=1000, n_beeps=1) * 0.3
            
        elif 'mall' in location_lower or 'shopping' in location_lower:
            # Background music hint
            audio += self._generate_music_hint(n_samples) * 0.1
            # Crowd
            audio += self._generate_crowd_murmur(n_samples) * 0.2
            
        elif 'street' in location_lower or 'road' in location_lower:
            # Traffic
            audio += self._generate_traffic(n_samples)
            
        return audio
    
    def _generate_situation_sounds(self, situation: str, n_samples: int) -> np.ndarray:
        """Generate situation-specific sounds"""
        audio = np.zeros(n_samples)
        situation_lower = situation.lower()
        
        if 'emergency' in situation_lower:
            # Siren
            audio += self._generate_siren(n_samples) * 0.4
            
        elif 'busy' in situation_lower or 'crowded' in situation_lower:
            # More crowd noise
            audio += self._generate_crowd_murmur(n_samples) * 0.3
            
        elif 'announcement' in situation_lower:
            # PA tones
            audio += self._generate_beeps(n_samples, freq=660, n_beeps=3)
            
        elif 'traffic' in situation_lower:
            audio += self._generate_traffic(n_samples) * 0.3
            
        return audio
    
    def _generate_beeps(
        self,
        n_samples: int,
        freq: float = 880,
        n_beeps: int = 2
    ) -> np.ndarray:
        """Generate beep tones"""
        audio = np.zeros(n_samples)
        beep_len = int(0.2 * self.sr)
        
        for i in range(n_beeps):
            start = int(np.random.uniform(0.1, 0.5) * n_samples)
            if start + beep_len < n_samples:
                t = np.linspace(0, 0.2, beep_len)
                beep = np.sin(2 * np.pi * freq * t)
                # Apply envelope
                envelope = np.exp(-3 * t)
                beep = beep * envelope * 0.1
                audio[start:start+beep_len] += beep
                
        return audio
    
    def _generate_siren(self, n_samples: int) -> np.ndarray:
        """Generate siren sound"""
        t = np.linspace(0, n_samples / self.sr, n_samples)
        # Frequency modulation for siren
        freq = 500 + 300 * np.sin(2 * np.pi * 2 * t)
        phase = 2 * np.pi * np.cumsum(freq) / self.sr
        siren = np.sin(phase)
        
        # Random amplitude variation
        envelope = 0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
        
        return siren * envelope
    
    def _generate_crowd_murmur(self, n_samples: int) -> np.ndarray:
        """Generate crowd murmur sound"""
        # Multiple filtered noise sources
        murmur = np.zeros(n_samples)
        
        for _ in range(5):
            noise = np.random.randn(n_samples)
            # Bandpass for speech frequencies
            filtered = self._apply_bandpass(noise, 200, 3000)
            # Random amplitude
            filtered *= np.random.uniform(0.1, 0.3)
            murmur += filtered
            
        return murmur / 5
    
    def _generate_traffic(self, n_samples: int) -> np.ndarray:
        """Generate traffic noise"""
        audio = np.zeros(n_samples)
        
        # Base traffic rumble
        rumble = np.random.randn(n_samples)
        rumble = self._apply_lowpass(rumble, 500) * 0.2
        audio += rumble
        
        # Occasional car passes
        n_cars = np.random.randint(1, 4)
        for _ in range(n_cars):
            audio += self._generate_car_pass(n_samples) * 0.15
            
        return audio
    
    def _generate_car_pass(self, n_samples: int) -> np.ndarray:
        """Generate car passing sound"""
        t = np.linspace(0, n_samples / self.sr, n_samples)
        
        # Random pass time
        center = np.random.uniform(0.3, 0.7) * n_samples / self.sr
        
        # Doppler-like effect
        envelope = np.exp(-10 * (t - center) ** 2)
        
        # Engine sound
        freq = 100 + 50 * (1 - 2 * (t - center) / (n_samples / self.sr))
        engine = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
        
        return engine * envelope
    
    def _generate_aircraft_noise(self, n_samples: int) -> np.ndarray:
        """Generate distant aircraft noise"""
        noise = np.random.randn(n_samples)
        # Low frequency rumble
        filtered = self._apply_lowpass(noise, 300)
        
        # Slow amplitude modulation
        t = np.linspace(0, n_samples / self.sr, n_samples)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
        
        return filtered * modulation
    
    def _generate_horn(self, n_samples: int, freq: float = 300) -> np.ndarray:
        """Generate horn sound"""
        audio = np.zeros(n_samples)
        
        # Random position
        start = int(np.random.uniform(0.2, 0.6) * n_samples)
        duration = int(1.5 * self.sr)
        
        if start + duration < n_samples:
            t = np.linspace(0, 1.5, duration)
            horn = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * 1.5 * t)
            
            # Envelope
            envelope = np.ones(duration)
            attack = int(0.1 * self.sr)
            release = int(0.2 * self.sr)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-release:] = np.linspace(1, 0, release)
            
            audio[start:start+duration] = horn * envelope * 0.2
            
        return audio
    
    def _generate_music_hint(self, n_samples: int) -> np.ndarray:
        """Generate hint of background music"""
        t = np.linspace(0, n_samples / self.sr, n_samples)
        
        # Simple chord
        freqs = [220, 277, 330]  # A minor
        music = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
        
        # Apply heavy filtering to sound distant
        music = self._apply_lowpass(music, 1000)
        
        return music * 0.1
    
    def _apply_lowpass(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply lowpass filter"""
        from scipy import signal
        nyquist = self.sr / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio).astype(np.float32)
    
    def _apply_bandpass(
        self,
        audio: np.ndarray,
        low: float,
        high: float
    ) -> np.ndarray:
        """Apply bandpass filter"""
        from scipy import signal
        nyquist = self.sr / 2
        low_norm = max(low / nyquist, 0.01)
        high_norm = min(high / nyquist, 0.99)
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        return signal.filtfilt(b, a, audio).astype(np.float32)


class AudioSceneDataset:
    """
    PyTorch-style dataset for audio scenes.
    """
    
    def __init__(
        self,
        metadata_path: str,
        audio_dir: str,
        split: str = 'train',
        transform = None
    ):
        """
        Initialize dataset.
        
        Args:
            metadata_path: Path to metadata CSV
            audio_dir: Base directory for audio files
            split: Dataset split to use
            transform: Optional transform to apply
        """
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        self.data = df[df['split'] == split].to_dict('records')
        
        print(f"📊 Loaded {len(self.data)} samples for {split} split")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio
        audio_path = self.audio_dir / item['split'] / item['filename']
        audio, sr = librosa.load(str(audio_path), sr=AUDIO_CONFIG.sample_rate)
        
        # Apply transform
        if self.transform:
            audio = self.transform(audio)
        
        # Get labels
        labels = {
            'location': item['location_idx'],
            'situation': item['situation_idx'],
        }
        
        return audio, labels


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Dataset Builder")
    print("="*60)
    
    # Create builder
    builder = AuralisDatasetBuilder(output_dir="test_dataset")
    
    # Test synthetic generation
    print("\n🔧 Testing synthetic data generation...")
    generator = SyntheticAudioGenerator()
    
    test_scenes = [
        ("Airport Terminal", "Normal/Quiet"),
        ("Street/Road", "Traffic"),
        ("Hospital", "Emergency"),
        ("Shopping Mall", "Busy/Crowded"),
    ]
    
    for location, situation in test_scenes:
        audio = generator.generate_scene(location, situation, duration=3.0)
        print(f"   ✅ {location} + {situation}: {len(audio)} samples")
    
    # Generate a small synthetic dataset
    print("\n🔄 Generating small synthetic dataset...")
    builder.generate_synthetic_data(n_samples=10, split='train')
    
    # Save metadata
    builder.save_metadata()
    
    # Print statistics
    builder.print_statistics()
    
    print("\n✅ Dataset Builder test passed!")