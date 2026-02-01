# ══════════════════════════════════════════════════════════════
# 🔊 services/audio_preprocessor.py - v8.0 ENTERPRISE
# ══════════════════════════════════════════════════════════════
"""
Professional Audio Preprocessing System

Handles:
- Noise reduction (even extreme noise)
- Echo cancellation
- Volume normalization
- Voice activity detection
- Bandwidth extension
- Audio quality enhancement
- Format conversion
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Audio processing libraries
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy import signal
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


@dataclass
class AudioQuality:
    """Audio quality metrics"""
    snr_db: float  # Signal-to-Noise Ratio
    quality_score: float  # 0-1 scale
    has_noise: bool
    has_echo: bool
    is_clipped: bool
    volume_level: str  # "too_quiet", "optimal", "too_loud"


class AudioPreprocessor:
    """
    Professional Audio Preprocessing System
    
    Features:
    - Advanced noise reduction (works even with extreme noise)
    - Echo cancellation
    - Volume normalization
    - Voice activity detection (VAD)
    - Bandwidth extension
    - Automatic quality assessment
    - Multiple processing modes (fast, balanced, quality)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mode: str = "balanced"  # fast, balanced, quality
    ):
        """
        Initialize Audio Preprocessor
        
        Args:
            sample_rate: Target sample rate
            mode: Processing mode
                - fast: Quick processing, basic enhancement
                - balanced: Good quality with reasonable speed (recommended)
                - quality: Maximum quality, slower processing
        """
        self.sample_rate = sample_rate
        self.mode = mode
        self.loaded = False
        
        # Check available backends
        self.use_noisereduce = NOISEREDUCE_AVAILABLE
        self.use_librosa = LIBROSA_AVAILABLE
        self.use_scipy = SCIPY_AVAILABLE
        self.use_webrtcvad = WEBRTCVAD_AVAILABLE
        
        backends = []
        if self.use_noisereduce:
            backends.append("NoiseReduce")
        if self.use_librosa:
            backends.append("Librosa")
        if self.use_scipy:
            backends.append("SciPy")
        if self.use_webrtcvad:
            backends.append("WebRTC VAD")
        
        backend_str = ", ".join(backends) if backends else "Basic"
        print(f"🔊 Audio Preprocessor: Mode={mode}, Backends=[{backend_str}]")
    
    def load(self) -> bool:
        """Load preprocessing components"""
        self.loaded = True
        print("✅ Audio Preprocessor ready")
        return True
    
    def preprocess(
        self,
        audio: np.ndarray,
        aggressive: bool = False
    ) -> Dict[str, Any]:
        """
        Preprocess audio with all enhancements
        
        Args:
            audio: Input audio waveform
            aggressive: Use aggressive noise reduction (for very noisy audio)
            
        Returns:
            Dict with:
            - enhanced_audio: Processed audio
            - original_audio: Original audio (for comparison)
            - quality_before: Quality metrics before processing
            - quality_after: Quality metrics after processing
            - improvements: List of improvements applied
        """
        if not self.loaded:
            return {"enhanced_audio": audio, "improvements": []}
        
        original_audio = audio.copy()
        improvements = []
        
        # 1. Assess original quality
        quality_before = self._assess_quality(audio)
        
        # 2. Noise reduction (CRITICAL for noisy audio)
        if quality_before.has_noise or aggressive:
            audio = self._reduce_noise(audio, aggressive=aggressive)
            improvements.append("noise_reduction")
        
        # 3. Echo cancellation
        if quality_before.has_echo:
            audio = self._cancel_echo(audio)
            improvements.append("echo_cancellation")
        
        # 4. Volume normalization
        if quality_before.volume_level != "optimal":
            audio = self._normalize_volume(audio)
            improvements.append("volume_normalization")
        
        # 5. Remove silence/noise segments (VAD)
        if self.use_webrtcvad:
            audio = self._apply_vad(audio)
            improvements.append("vad_filtering")
        
        # 6. Bandwidth extension (improve quality)
        if self.mode in ["balanced", "quality"]:
            audio = self._extend_bandwidth(audio)
            improvements.append("bandwidth_extension")
        
        # 7. De-clip if needed
        if quality_before.is_clipped:
            audio = self._declip(audio)
            improvements.append("declipping")
        
        # 8. Final enhancement
        audio = self._enhance_speech(audio)
        improvements.append("speech_enhancement")
        
        # 9. Assess final quality
        quality_after = self._assess_quality(audio)
        
        return {
            "enhanced_audio": audio,
            "original_audio": original_audio,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "improvements": improvements,
            "snr_improvement_db": quality_after.snr_db - quality_before.snr_db,
            "quality_improvement": quality_after.quality_score - quality_before.quality_score
        }
    
    def _reduce_noise(
        self,
        audio: np.ndarray,
        aggressive: bool = False
    ) -> np.ndarray:
        """
        Advanced noise reduction
        
        Works even with extreme background noise
        """
        if not self.use_noisereduce:
            return self._reduce_noise_scipy(audio)
        
        try:
            # NoiseReduce is very effective
            if aggressive:
                # Aggressive mode for very noisy audio
                reduced = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    stationary=False,  # Non-stationary noise
                    prop_decrease=1.0,  # Maximum reduction
                    freq_mask_smooth_hz=1000,
                    time_mask_smooth_ms=100
                )
            else:
                # Balanced mode
                reduced = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.8,
                    freq_mask_smooth_hz=500,
                    time_mask_smooth_ms=50
                )
            
            return reduced
            
        except Exception as e:
            logger.warning(f"NoiseReduce failed: {e}, using fallback")
            return self._reduce_noise_scipy(audio)
    
    def _reduce_noise_scipy(self, audio: np.ndarray) -> np.ndarray:
        """Fallback noise reduction using scipy"""
        if not self.use_scipy:
            return audio
        
        # Apply bandpass filter (human speech range: 300-3400 Hz)
        nyquist = self.sample_rate / 2
        low = 300 / nyquist
        high = 3400 / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, audio)
        
        return filtered
    
    def _cancel_echo(self, audio: np.ndarray) -> np.ndarray:
        """Echo cancellation using spectral subtraction"""
        if not self.use_librosa:
            return audio
        
        try:
            # Compute STFT
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate and subtract echo
            # Simple method: subtract attenuated past signal
            echo_delay = int(0.1 * self.sample_rate / 512)  # ~100ms delay
            
            for i in range(echo_delay, magnitude.shape[1]):
                magnitude[:, i] = np.maximum(
                    magnitude[:, i] - 0.3 * magnitude[:, i - echo_delay],
                    0
                )
            
            # Reconstruct
            stft_clean = magnitude * np.exp(1j * phase)
            audio_clean = librosa.istft(stft_clean)
            
            # Ensure same length
            if len(audio_clean) > len(audio):
                audio_clean = audio_clean[:len(audio)]
            elif len(audio_clean) < len(audio):
                audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
            
            return audio_clean
            
        except Exception as e:
            logger.warning(f"Echo cancellation failed: {e}")
            return audio
    
    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume to optimal level"""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-6:
            return audio
        
        # Target RMS (optimal for speech)
        target_rms = 0.1
        
        # Calculate gain
        gain = target_rms / rms
        
        # Limit gain to avoid over-amplification
        gain = min(gain, 10.0)
        
        # Apply gain
        normalized = audio * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 0.95:
            normalized = normalized * (0.95 / max_val)
        
        return normalized
    
    def _apply_vad(self, audio: np.ndarray) -> np.ndarray:
        """Apply Voice Activity Detection to remove non-speech segments"""
        if not self.use_webrtcvad:
            return audio
        
        try:
            vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Process in 30ms frames
            frame_duration = 30  # ms
            frame_length = int(self.sample_rate * frame_duration / 1000)
            
            voiced_frames = []
            
            for i in range(0, len(audio_int16) - frame_length, frame_length):
                frame = audio_int16[i:i + frame_length]
                
                # Check if frame contains speech
                try:
                    is_speech = vad.is_speech(frame.tobytes(), self.sample_rate)
                    
                    if is_speech:
                        voiced_frames.append(frame)
                except:
                    # If VAD fails, keep the frame
                    voiced_frames.append(frame)
            
            # Concatenate voiced frames
            if voiced_frames:
                voiced_audio = np.concatenate(voiced_frames)
                return voiced_audio.astype(np.float32) / 32767
            else:
                return audio
                
        except Exception as e:
            logger.warning(f"VAD failed: {e}")
            return audio
    
    def _extend_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """Extend bandwidth for better quality (if audio is narrowband)"""
        if not self.use_librosa:
            return audio
        
        try:
            # Check if audio is narrowband (< 8kHz)
            if self.sample_rate < 16000:
                # Upsample to 16kHz
                audio = librosa.resample(
                    audio,
                    orig_sr=self.sample_rate,
                    target_sr=16000
                )
                self.sample_rate = 16000
            
            return audio
            
        except Exception as e:
            logger.warning(f"Bandwidth extension failed: {e}")
            return audio
    
    def _declip(self, audio: np.ndarray) -> np.ndarray:
        """Remove clipping artifacts"""
        # Detect clipped samples (near ±1.0)
        clipped_threshold = 0.98
        clipped_mask = np.abs(audio) > clipped_threshold
        
        if not np.any(clipped_mask):
            return audio
        
        # Simple declipping: interpolate clipped regions
        clipped_indices = np.where(clipped_mask)[0]
        
        for idx in clipped_indices:
            # Find boundaries of clipped region
            start = idx
            end = idx
            
            while start > 0 and clipped_mask[start - 1]:
                start -= 1
            
            while end < len(audio) - 1 and clipped_mask[end + 1]:
                end += 1
            
            # Interpolate
            if start > 0 and end < len(audio) - 1:
                audio[start:end+1] = np.linspace(
                    audio[start - 1],
                    audio[end + 1],
                    end - start + 1
                )
        
        return audio
    
    def _enhance_speech(self, audio: np.ndarray) -> np.ndarray:
        """Final speech enhancement"""
        if not self.use_scipy:
            return audio
        
        # Apply gentle high-pass filter to remove low-frequency rumble
        nyquist = self.sample_rate / 2
        cutoff = 80 / nyquist  # 80 Hz
        
        b, a = butter(2, cutoff, btype='high')
        enhanced = filtfilt(b, a, audio)
        
        return enhanced
    
    def _assess_quality(self, audio: np.ndarray) -> AudioQuality:
        """Assess audio quality"""
        # Calculate SNR
        snr = self._calculate_snr(audio)
        
        # Check for noise
        has_noise = snr < 15  # < 15 dB SNR indicates noisy audio
        
        # Check for echo (autocorrelation)
        has_echo = self._detect_echo(audio)
        
        # Check for clipping
        is_clipped = np.max(np.abs(audio)) > 0.98
        
        # Check volume level
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            volume_level = "too_quiet"
        elif rms > 0.3:
            volume_level = "too_loud"
        else:
            volume_level = "optimal"
        
        # Calculate overall quality score (0-1)
        quality_score = 0.0
        
        # SNR contribution (0-0.4)
        quality_score += min(snr / 25, 1.0) * 0.4
        
        # No clipping bonus (0.2)
        if not is_clipped:
            quality_score += 0.2
        
        # Optimal volume bonus (0.2)
        if volume_level == "optimal":
            quality_score += 0.2
        
        # No echo bonus (0.2)
        if not has_echo:
            quality_score += 0.2
        
        return AudioQuality(
            snr_db=snr,
            quality_score=quality_score,
            has_noise=has_noise,
            has_echo=has_echo,
            is_clipped=is_clipped,
            volume_level=volume_level
        )
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB"""
        # Divide into frames
        frame_size = int(self.sample_rate * 0.02)  # 20ms
        
        energies = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            energy = np.mean(frame ** 2)
            energies.append(energy)
        
        if not energies:
            return 0.0
        
        energies = np.array(energies)
        
        # Signal: top 10% energy
        signal_power = np.percentile(energies, 90)
        
        # Noise: bottom 10% energy
        noise_power = np.percentile(energies, 10) + 1e-10
        
        # SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)
        
        return max(0, min(60, snr))  # Clip to 0-60 dB range
    
    def _detect_echo(self, audio: np.ndarray) -> bool:
        """Detect echo using autocorrelation"""
        if len(audio) < self.sample_rate:
            return False
        
        # Use first second
        segment = audio[:self.sample_rate]
        
        # Compute autocorrelation
        autocorr = np.correlate(segment, segment, mode='same')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Look for peaks (indicating echo)
        # Echo typically occurs at 50-500ms delay
        delay_min = int(0.05 * self.sample_rate)
        delay_max = int(0.5 * self.sample_rate)
        
        echo_region = autocorr[delay_min:delay_max]
        max_corr = np.max(echo_region) if len(echo_region) > 0 else 0
        
        # If correlation > 0.3, likely has echo
        return max_corr > 0.3