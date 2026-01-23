# ==============================
# 📄 training/audio_augmentation.py
# ==============================
# Audio Data Augmentation Pipeline
# Upgrade 2: Increase training data diversity
# ==============================

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import random
from typing import List, Tuple, Optional, Callable
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_CONFIG, TRAINING_CONFIG


class AudioAugmentor:
    """
    Professional audio augmentation for robust training.
    Simulates real-world audio conditions including noise,
    room acoustics, and various distortions.
    """
    
    def __init__(self, sample_rate: int = None):
        """
        Initialize audio augmentor.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sr = sample_rate or AUDIO_CONFIG.sample_rate
        
        # Precomputed noise samples for efficiency
        self._noise_cache = {}
        
    def time_stretch(
        self,
        audio: np.ndarray,
        rate_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Speed up or slow down audio without changing pitch.
        
        Args:
            audio: Input audio waveform
            rate_range: Range of stretch rates (min, max)
            
        Returns:
            Time-stretched audio
        """
        rate = random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(
        self,
        audio: np.ndarray,
        steps_range: Tuple[float, float] = (-4, 4)
    ) -> np.ndarray:
        """
        Shift pitch up or down.
        
        Args:
            audio: Input audio waveform
            steps_range: Range of semitones to shift
            
        Returns:
            Pitch-shifted audio
        """
        steps = random.uniform(*steps_range)
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)
    
    def add_gaussian_noise(
        self,
        audio: np.ndarray,
        noise_level_range: Tuple[float, float] = (0.001, 0.015)
    ) -> np.ndarray:
        """
        Add Gaussian white noise.
        
        Args:
            audio: Input audio waveform
            noise_level_range: Range of noise standard deviation
            
        Returns:
            Noisy audio
        """
        noise_level = random.uniform(*noise_level_range)
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise.astype(audio.dtype)
    
    def add_colored_noise(
        self,
        audio: np.ndarray,
        noise_type: str = 'pink',
        snr_range: Tuple[float, float] = (10, 30)
    ) -> np.ndarray:
        """
        Add colored noise (pink, brown, etc.).
        
        Args:
            audio: Input audio waveform
            noise_type: Type of noise ('pink', 'brown', 'blue')
            snr_range: Signal-to-noise ratio range in dB
            
        Returns:
            Noisy audio
        """
        n_samples = len(audio)
        snr = random.uniform(*snr_range)
        
        # Generate noise based on type
        if noise_type == 'pink':
            # Pink noise: 1/f spectrum
            freqs = np.fft.rfftfreq(n_samples)
            freqs[0] = 1  # Avoid division by zero
            pink_filter = 1 / np.sqrt(freqs)
            white = np.random.randn(n_samples)
            noise = np.fft.irfft(np.fft.rfft(white) * pink_filter, n_samples)
        elif noise_type == 'brown':
            # Brown noise: 1/f^2 spectrum
            noise = np.cumsum(np.random.randn(n_samples))
            noise = noise - np.mean(noise)
        else:
            noise = np.random.randn(n_samples)
        
        # Normalize noise
        noise = noise / (np.std(noise) + 1e-8)
        
        # Calculate scaling for target SNR
        audio_power = np.mean(audio ** 2) + 1e-8
        noise_power = np.mean(noise ** 2) + 1e-8
        scale = np.sqrt(audio_power / (noise_power * (10 ** (snr / 10))))
        
        return audio + (scale * noise).astype(audio.dtype)
    
    def add_background_noise(
        self,
        audio: np.ndarray,
        noise_audio: np.ndarray,
        snr_range: Tuple[float, float] = (5, 20)
    ) -> np.ndarray:
        """
        Mix with background noise audio at random SNR.
        
        Args:
            audio: Input audio waveform
            noise_audio: Background noise waveform
            snr_range: Signal-to-noise ratio range in dB
            
        Returns:
            Mixed audio
        """
        snr = random.uniform(*snr_range)
        
        # Adjust noise length to match audio
        if len(noise_audio) < len(audio):
            # Tile noise to match length
            repeats = int(np.ceil(len(audio) / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)
        
        # Randomly select segment
        if len(noise_audio) > len(audio):
            start = random.randint(0, len(noise_audio) - len(audio))
            noise_audio = noise_audio[start:start + len(audio)]
        
        # Calculate scaling for target SNR
        audio_power = np.mean(audio ** 2) + 1e-8
        noise_power = np.mean(noise_audio ** 2) + 1e-8
        scale = np.sqrt(audio_power / (noise_power * (10 ** (snr / 10))))
        
        return audio + (scale * noise_audio).astype(audio.dtype)
    
    def room_reverb(
        self,
        audio: np.ndarray,
        room_size_range: Tuple[float, float] = (0.1, 0.6),
        damping_range: Tuple[float, float] = (0.2, 0.8)
    ) -> np.ndarray:
        """
        Simulate room acoustics with reverb.
        
        Args:
            audio: Input audio waveform
            room_size_range: Range of room sizes (affects reverb length)
            damping_range: Range of damping factors
            
        Returns:
            Reverbed audio
        """
        room_size = random.uniform(*room_size_range)
        damping = random.uniform(*damping_range)
        
        # Create simple impulse response
        impulse_len = int(self.sr * room_size)
        t = np.linspace(0, 10, impulse_len)
        impulse = np.exp(-damping * t) * np.random.randn(impulse_len)
        impulse = impulse / (np.sum(np.abs(impulse)) + 1e-8)
        
        # Convolve with audio
        reverbed = signal.convolve(audio, impulse, mode='same')
        
        # Mix dry and wet
        wet_ratio = random.uniform(0.1, 0.4)
        return ((1 - wet_ratio) * audio + wet_ratio * reverbed).astype(audio.dtype)
    
    def frequency_mask(
        self,
        audio: np.ndarray,
        num_masks: int = 2,
        mask_width_range: Tuple[int, int] = (20, 100)
    ) -> np.ndarray:
        """
        Mask random frequency bands (SpecAugment style).
        
        Args:
            audio: Input audio waveform
            num_masks: Number of frequency masks
            mask_width_range: Range of mask widths in frequency bins
            
        Returns:
            Frequency-masked audio
        """
        stft = librosa.stft(audio)
        
        for _ in range(num_masks):
            mask_width = random.randint(*mask_width_range)
            max_start = max(0, stft.shape[0] - mask_width)
            mask_start = random.randint(0, max_start) if max_start > 0 else 0
            stft[mask_start:mask_start + mask_width, :] = 0
        
        return librosa.istft(stft, length=len(audio))
    
    def time_mask(
        self,
        audio: np.ndarray,
        num_masks: int = 2,
        max_mask_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Mask random time segments.
        
        Args:
            audio: Input audio waveform
            num_masks: Number of time masks
            max_mask_ratio: Maximum ratio of audio to mask
            
        Returns:
            Time-masked audio
        """
        result = audio.copy()
        max_mask_len = int(len(audio) * max_mask_ratio)
        
        for _ in range(num_masks):
            mask_len = random.randint(0, max_mask_len)
            mask_start = random.randint(0, max(0, len(audio) - mask_len))
            result[mask_start:mask_start + mask_len] = 0
        
        return result
    
    def telephone_effect(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulate telephone audio quality (bandpass + distortion).
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Telephone-quality audio
        """
        # Bandpass filter (300-3400 Hz for telephone)
        nyquist = self.sr / 2
        low = 300 / nyquist
        high = min(3400 / nyquist, 0.99)
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        
        # Add slight distortion
        filtered = np.tanh(filtered * 1.5) * 0.8
        
        # Add subtle noise
        noise = np.random.randn(len(filtered)) * 0.01
        
        return (filtered + noise).astype(audio.dtype)
    
    def low_quality_mic(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulate low-quality microphone recording.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Degraded audio
        """
        # Low-pass filter
        nyquist = self.sr / 2
        cutoff = random.uniform(3000, 6000) / nyquist
        b, a = signal.butter(3, cutoff, btype='low')
        filtered = signal.filtfilt(b, a, audio)
        
        # Add hum (50/60 Hz)
        hum_freq = random.choice([50, 60])
        t = np.arange(len(audio)) / self.sr
        hum = np.sin(2 * np.pi * hum_freq * t) * 0.02
        
        # Add noise
        noise = np.random.randn(len(audio)) * 0.02
        
        return (filtered + hum + noise).astype(audio.dtype)
    
    def volume_change(
        self,
        audio: np.ndarray,
        gain_range: Tuple[float, float] = (0.5, 1.5)
    ) -> np.ndarray:
        """
        Random volume change.
        
        Args:
            audio: Input audio waveform
            gain_range: Range of gain multipliers
            
        Returns:
            Volume-adjusted audio
        """
        gain = random.uniform(*gain_range)
        result = audio * gain
        
        # Clip to prevent clipping
        return np.clip(result, -1.0, 1.0).astype(audio.dtype)
    
    def dynamic_range_compression(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        ratio: float = 4.0
    ) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Args:
            audio: Input audio waveform
            threshold: Compression threshold
            ratio: Compression ratio
            
        Returns:
            Compressed audio
        """
        # Simple compression
        compressed = np.sign(audio) * (
            threshold + (np.abs(audio) - threshold) / ratio
        ) * (np.abs(audio) > threshold) + audio * (np.abs(audio) <= threshold)
        
        return compressed.astype(audio.dtype)
    
    def speed_perturb(
        self,
        audio: np.ndarray,
        speed_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """
        Change playback speed (affects both tempo and pitch).
        
        Args:
            audio: Input audio waveform
            speed_range: Range of speed factors
            
        Returns:
            Speed-perturbed audio
        """
        speed = random.uniform(*speed_range)
        
        # Resample to change speed
        new_length = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)
    
    def augment(
        self,
        audio: np.ndarray,
        intensity: str = 'medium'
    ) -> np.ndarray:
        """
        Apply random augmentations based on intensity level.
        
        Args:
            audio: Input audio waveform
            intensity: Augmentation intensity ('light', 'medium', 'heavy')
            
        Returns:
            Augmented audio
        """
        # Define augmentation probabilities for each intensity
        augmentations = {
            'light': [
                (self.add_gaussian_noise, {'noise_level_range': (0.001, 0.005)}, 0.3),
                (self.pitch_shift, {'steps_range': (-1, 1)}, 0.2),
                (self.volume_change, {'gain_range': (0.8, 1.2)}, 0.3),
            ],
            'medium': [
                (self.add_gaussian_noise, {'noise_level_range': (0.001, 0.01)}, 0.4),
                (self.pitch_shift, {'steps_range': (-2, 2)}, 0.3),
                (self.time_stretch, {'rate_range': (0.9, 1.1)}, 0.2),
                (self.room_reverb, {'room_size_range': (0.1, 0.3)}, 0.3),
                (self.volume_change, {'gain_range': (0.7, 1.3)}, 0.3),
                (self.time_mask, {'num_masks': 1, 'max_mask_ratio': 0.05}, 0.2),
            ],
            'heavy': [
                (self.add_gaussian_noise, {'noise_level_range': (0.005, 0.02)}, 0.5),
                (self.add_colored_noise, {'noise_type': 'pink', 'snr_range': (10, 25)}, 0.3),
                (self.pitch_shift, {'steps_range': (-4, 4)}, 0.4),
                (self.time_stretch, {'rate_range': (0.8, 1.2)}, 0.3),
                (self.room_reverb, {'room_size_range': (0.1, 0.5)}, 0.4),
                (self.frequency_mask, {'num_masks': 2, 'mask_width_range': (10, 50)}, 0.3),
                (self.time_mask, {'num_masks': 2, 'max_mask_ratio': 0.1}, 0.3),
                (self.telephone_effect, {}, 0.15),
                (self.low_quality_mic, {}, 0.15),
                (self.volume_change, {'gain_range': (0.5, 1.5)}, 0.4),
            ]
        }
        
        result = audio.copy()
        
        for aug_func, kwargs, prob in augmentations.get(intensity, augmentations['medium']):
            if random.random() < prob:
                try:
                    result = aug_func(result, **kwargs)
                except Exception as e:
                    # Skip failed augmentation
                    pass
        
        # Ensure output is normalized
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.99
        
        return result.astype(np.float32)


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.
    Applied directly to mel spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        """
        Initialize SpecAugment.
        
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            n_freq_masks: Number of frequency masks
            n_time_masks: Number of time masks
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram [time, freq]
            
        Returns:
            Augmented mel spectrogram
        """
        result = mel_spec.copy()
        time_len, freq_len = result.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, freq_len - 1))
            f0 = random.randint(0, freq_len - f)
            result[:, f0:f0 + f] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_len - 1))
            t0 = random.randint(0, time_len - t)
            result[t0:t0 + t, :] = 0
        
        return result


class AugmentedDataGenerator:
    """
    Generate augmented training batches with on-the-fly augmentation.
    """
    
    def __init__(
        self,
        audio_files: List[str],
        labels: List[dict],
        augmentor: AudioAugmentor,
        mel_extractor,
        batch_size: int = 32,
        augmentation_probability: float = None,
        intensity: str = None,
        shuffle: bool = True,
        spec_augment: bool = True
    ):
        """
        Initialize augmented data generator.
        
        Args:
            audio_files: List of audio file paths
            labels: List of label dictionaries
            augmentor: AudioAugmentor instance
            mel_extractor: MelSpectrogramExtractor instance
            batch_size: Batch size
            augmentation_probability: Probability of applying augmentation
            intensity: Augmentation intensity
            shuffle: Whether to shuffle data
            spec_augment: Whether to apply SpecAugment
        """
        self.audio_files = audio_files
        self.labels = labels
        self.augmentor = augmentor
        self.mel_extractor = mel_extractor
        self.batch_size = batch_size
        self.aug_prob = augmentation_probability or TRAINING_CONFIG.augmentation_probability
        self.intensity = intensity or TRAINING_CONFIG.augmentation_intensity
        self.shuffle = shuffle
        self.spec_augment = SpecAugment() if spec_augment else None
        
    def __len__(self):
        return len(self.audio_files) // self.batch_size
    
    def __iter__(self):
        indices = np.arange(len(self.audio_files))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            
            batch_mels = []
            batch_labels = []
            
            for idx in batch_indices:
                # Load audio
                audio, sr = librosa.load(
                    self.audio_files[idx],
                    sr=AUDIO_CONFIG.sample_rate
                )
                
                # Apply augmentation with probability
                if random.random() < self.aug_prob:
                    intensity = random.choice(['light', 'medium', 'heavy'])
                    audio = self.augmentor.augment(audio, intensity=intensity)
                
                # Extract mel spectrogram
                mel = self.mel_extractor.extract(audio)
                
                # Apply SpecAugment
                if self.spec_augment and random.random() < 0.5:
                    mel = self.spec_augment(mel)
                
                batch_mels.append(mel)
                batch_labels.append(self.labels[idx])
            
            # Pad batch to same length
            max_len = max(m.shape[0] for m in batch_mels)
            padded_mels = []
            for m in batch_mels:
                if m.shape[0] < max_len:
                    pad_width = ((0, max_len - m.shape[0]), (0, 0))
                    m = np.pad(m, pad_width, mode='constant')
                padded_mels.append(m)
            
            yield np.array(padded_mels), batch_labels


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Audio Augmentation")
    print("="*60)
    
    # Create augmentor
    augmentor = AudioAugmentor()
    
    # Generate test audio (3 seconds of sine wave + noise)
    duration = 3.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    audio = audio.astype(np.float32)
    
    print(f"\n📊 Original audio: {len(audio)} samples ({duration}s)")
    
    # Test individual augmentations
    print("\n🔧 Testing individual augmentations:")
    
    augmentations = [
        ("Time Stretch", lambda a: augmentor.time_stretch(a)),
        ("Pitch Shift", lambda a: augmentor.pitch_shift(a)),
        ("Gaussian Noise", lambda a: augmentor.add_gaussian_noise(a)),
        ("Colored Noise", lambda a: augmentor.add_colored_noise(a)),
        ("Room Reverb", lambda a: augmentor.room_reverb(a)),
        ("Frequency Mask", lambda a: augmentor.frequency_mask(a)),
        ("Time Mask", lambda a: augmentor.time_mask(a)),
        ("Telephone Effect", lambda a: augmentor.telephone_effect(a)),
        ("Low Quality Mic", lambda a: augmentor.low_quality_mic(a)),
        ("Volume Change", lambda a: augmentor.volume_change(a)),
    ]
    
    for name, func in augmentations:
        try:
            result = func(audio)
            print(f"   ✅ {name}: {len(result)} samples")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    # Test combined augmentation
    print("\n🎯 Testing combined augmentation:")
    for intensity in ['light', 'medium', 'heavy']:
        result = augmentor.augment(audio, intensity=intensity)
        print(f"   {intensity.capitalize()}: {len(result)} samples, range [{result.min():.3f}, {result.max():.3f}]")
    
    # Test SpecAugment
    print("\n📈 Testing SpecAugment:")
    spec_aug = SpecAugment()
    mel = np.random.randn(100, 80)  # Fake mel spectrogram
    augmented_mel = spec_aug(mel)
    print(f"   Input shape: {mel.shape}")
    print(f"   Output shape: {augmented_mel.shape}")
    print(f"   Zeros added: {np.sum(augmented_mel == 0) - np.sum(mel == 0)}")
    
    print("\n✅ Audio Augmentation test passed!")