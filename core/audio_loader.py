"""
============================================
AURALIS v5.0 - Audio Loader
============================================
Robust audio loading with multiple fallback methods.
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np

from config import settings
from utils.logger import logger
from utils.helpers import (
    normalize_audio, trim_silence, convert_to_wav,
    check_ffmpeg, setup_ffmpeg_path
)


class AudioLoader:
    """
    Robust audio loader supporting multiple formats.
    
    Features:
    - Multiple loading backends (librosa, soundfile, ffmpeg)
    - Automatic format conversion
    - Audio preprocessing and normalization
    - Sample rate conversion
    """
    
    def __init__(self, sample_rate: int = None):
        """
        Initialize audio loader.
        
        Args:
            sample_rate: Target sample rate (default from settings)
        """
        self.sample_rate = sample_rate or settings.sample_rate
        
        # Setup FFmpeg
        self.ffmpeg_available = setup_ffmpeg_path(settings.ffmpeg_path)
        
        # Import audio libraries
        self._librosa = None
        self._soundfile = None
        self._wavfile = None
        
        self._import_libraries()
    
    def _import_libraries(self):
        """Import audio libraries with graceful fallback."""
        try:
            import librosa
            self._librosa = librosa
            logger.debug("Librosa available")
        except ImportError:
            logger.warning("Librosa not available")
        
        try:
            import soundfile
            self._soundfile = soundfile
            logger.debug("Soundfile available")
        except ImportError:
            logger.warning("Soundfile not available")
        
        try:
            from scipy.io import wavfile
            self._wavfile = wavfile
            logger.debug("Scipy wavfile available")
        except ImportError:
            logger.warning("Scipy wavfile not available")
    
    def load(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Load audio file with automatic format detection.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, duration_seconds)
            
        Raises:
            Exception: If audio cannot be loaded
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("Audio file is empty")
        
        logger.info(f"Loading audio: {file_path} ({file_size / 1024:.1f} KB)")
        
        # Try loading methods in order
        audio = None
        methods = [
            ("Librosa", self._load_librosa),
            ("Soundfile", self._load_soundfile),
            ("FFmpeg", self._load_ffmpeg),
        ]
        
        for method_name, method_func in methods:
            try:
                audio, sr = method_func(file_path)
                if audio is not None and len(audio) > 0:
                    # Resample if needed
                    if sr != self.sample_rate:
                        audio = self._resample(audio, sr)
                    
                    # Convert to mono
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1 if audio.shape[1] < audio.shape[0] else 0)
                    
                    # Preprocess
                    audio = self._preprocess(audio)
                    
                    duration = len(audio) / self.sample_rate
                    logger.info(f"Loaded with {method_name}: {duration:.2f}s")
                    
                    return audio, duration
            except Exception as e:
                logger.debug(f"{method_name} failed: {e}")
                continue
        
        raise RuntimeError("Failed to load audio with all available methods")
    
    def _load_librosa(self, path: str) -> Tuple[np.ndarray, int]:
        """Load using librosa."""
        if self._librosa is None:
            raise ImportError("Librosa not available")
        
        audio, sr = self._librosa.load(path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32), sr
    
    def _load_soundfile(self, path: str) -> Tuple[np.ndarray, int]:
        """Load using soundfile."""
        if self._soundfile is None:
            raise ImportError("Soundfile not available")
        
        audio, sr = self._soundfile.read(path)
        return audio.astype(np.float32), sr
    
    def _load_ffmpeg(self, path: str) -> Tuple[np.ndarray, int]:
        """Load by converting to WAV with FFmpeg."""
        if not self.ffmpeg_available:
            raise RuntimeError("FFmpeg not available")
        
        if self._wavfile is None:
            raise ImportError("Scipy wavfile not available")
        
        # Create temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            if not convert_to_wav(path, temp_path, self.sample_rate):
                raise RuntimeError("FFmpeg conversion failed")
            
            sr, audio = self._wavfile.read(temp_path)
            
            # Convert to float32
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128.0
            else:
                audio = audio.astype(np.float32)
            
            return audio, sr
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == self.sample_rate:
            return audio
        
        if self._librosa is not None:
            return self._librosa.resample(
                audio, 
                orig_sr=orig_sr, 
                target_sr=self.sample_rate
            )
        
        # Simple linear interpolation fallback
        ratio = self.sample_rate / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for analysis."""
        # Normalize
        audio = normalize_audio(audio)
        
        # Trim silence
        audio = trim_silence(audio, sample_rate=self.sample_rate)
        
        return audio
    
    def save_wav(self, audio: np.ndarray, path: str) -> str:
        """
        Save audio as WAV file.
        
        Args:
            audio: Audio array
            path: Output path
            
        Returns:
            Path to saved file
        """
        if self._wavfile is not None:
            audio_int = (audio * 32767).astype(np.int16)
            self._wavfile.write(path, self.sample_rate, audio_int)
        elif self._soundfile is not None:
            self._soundfile.write(path, audio, self.sample_rate)
        else:
            raise RuntimeError("No audio writer available")
        
        return path
    
    def get_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract basic audio features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary of audio features
        """
        features = {
            "duration": len(audio) / self.sample_rate,
            "samples": len(audio),
            "sample_rate": self.sample_rate,
            "rms_energy": float(np.sqrt(np.mean(audio ** 2))),
            "max_amplitude": float(np.max(np.abs(audio))),
            "zero_crossing_rate": float(
                np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
            ),
        }
        
        # Estimate if audio contains speech
        features["likely_has_speech"] = (
            0.02 < features["zero_crossing_rate"] < 0.15 and
            features["rms_energy"] > 0.01
        )
        
        return features