# ==============================
# services/audio_preprocessor_windows.py
# ==============================
"""
Windows-Compatible Audio Preprocessor
No webrtcvad, no noisereduce - pure Python only!
"""

import numpy as np
import librosa
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from config import SAMPLE_RATE


class AudioPreprocessor:
    """Windows-compatible audio preprocessing"""
    
    def __init__(self):
        """Initialize audio preprocessor"""
        self.sample_rate = SAMPLE_RATE
        
    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio (Windows-compatible version)
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of input audio
            normalize: Whether to normalize audio
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )
            sample_rate = self.sample_rate
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Normalize audio
        if normalize:
            audio = self._normalize_audio(audio)
        
        # Remove silence from beginning and end
        audio = self._trim_silence(audio)
        
        return audio, sample_rate
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to have consistent volume
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio
        """
        # Peak normalization
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 0.1
        if rms > 0:
            audio = audio * (target_rms / rms)
        
        # Clip to prevent distortion
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _trim_silence(
        self,
        audio: np.ndarray,
        top_db: int = 30
    ) -> np.ndarray:
        """
        Trim silence from beginning and end
        
        Args:
            audio: Input audio array
            top_db: Threshold for silence detection
            
        Returns:
            Trimmed audio
        """
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return trimmed
        except:
            return audio
    
    def enhance_for_speech(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, int]:
        """
        Simple enhancement for speech
        
        Args:
            audio: Input audio array
            sr: Sample rate
            
        Returns:
            Tuple of (enhanced_audio, sample_rate)
        """
        return self.preprocess(audio, sr, normalize=True)
    
    def chunk_audio(
        self,
        audio: np.ndarray,
        chunk_duration: float = 30.0
    ) -> list:
        """
        Split long audio into chunks
        
        Args:
            audio: Input audio array
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * self.sample_rate)
        chunks = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > self.sample_rate * 0.5:
                chunks.append(chunk)
        
        return chunks


def preprocess_audio_file(
    file_path: str,
    enhance: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file
    
    Args:
        file_path: Path to audio file
        enhance: Whether to apply enhancement
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    preprocessor = AudioPreprocessor()
    
    # Load audio
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # Enhance if requested
    if enhance:
        audio, sr = preprocessor.enhance_for_speech(audio, sr)
    
    return audio, sr