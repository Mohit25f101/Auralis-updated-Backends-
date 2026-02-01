# ==============================
# 📄 services/ambient_analyzer.py
# ==============================
"""
Ambient Sound & Acoustic Feature Analyzer
Extracts acoustic properties for location detection
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class AmbientAnalyzer:
    """
    Ambient Sound & Acoustic Feature Analyzer
    
    Analyzes:
    - Reverb/echo characteristics
    - Background noise floor
    - Outdoor vs indoor detection
    - Crowd density estimation
    - Environmental acoustic signatures
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.loaded = False
    
    def load(self) -> bool:
        """Load the analyzer"""
        self.loaded = True
        print("✅ Ambient Analyzer loaded")
        return True
    
    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio for ambient/acoustic features
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Dictionary with acoustic feature analysis
        """
        if not self.loaded or len(audio) < self.sample_rate:
            return self._empty_result()
        
        try:
            # Calculate various acoustic features
            reverb_estimate = self._estimate_reverb(audio)
            noise_floor = self._estimate_noise_floor(audio)
            is_outdoor = self._detect_outdoor(audio)
            crowd_density = self._estimate_crowd_density(audio)
            acoustic_signature = self._get_acoustic_signature(audio)
            energy_profile = self._get_energy_profile(audio)
            
            return {
                "reverb_estimate": round(reverb_estimate, 3),
                "noise_floor": round(noise_floor, 3),
                "is_outdoor": is_outdoor,
                "crowd_density": crowd_density,
                "acoustic_signature": acoustic_signature,
                "energy_profile": energy_profile,
                "snr_estimate": round(self._estimate_snr(audio), 2),
                "dynamic_range": round(self._get_dynamic_range(audio), 3)
            }
            
        except Exception as e:
            print(f"   ⚠️ Ambient analysis error: {e}")
            return self._empty_result()
    
    def _estimate_reverb(self, audio: np.ndarray) -> float:
        """
        Estimate reverb/echo level
        Higher values indicate more reverberant spaces (stations, halls)
        Lower values indicate dry spaces (outdoors, small rooms)
        """
        # Calculate autocorrelation for reverb estimation
        frame_size = int(self.sample_rate * 0.05)  # 50ms frames
        hop_size = int(self.sample_rate * 0.025)   # 25ms hop
        
        reverb_scores = []
        
        for i in range(0, len(audio) - frame_size * 2, hop_size):
            frame1 = audio[i:i + frame_size]
            frame2 = audio[i + frame_size:i + frame_size * 2]
            
            # Cross-correlation
            if np.std(frame1) > 0.01 and np.std(frame2) > 0.01:
                correlation = np.correlate(frame1, frame2, mode='valid')[0]
                normalized = correlation / (np.std(frame1) * np.std(frame2) * len(frame1))
                reverb_scores.append(abs(normalized))
        
        if not reverb_scores:
            return 0.5
        
        # High correlation between consecutive frames indicates reverb
        return min(np.mean(reverb_scores) * 2, 1.0)
    
    def _estimate_noise_floor(self, audio: np.ndarray) -> float:
        """
        Estimate background noise floor
        0 = very quiet, 1 = very noisy
        """
        # Calculate RMS energy in small frames
        frame_size = int(self.sample_rate * 0.02)  # 20ms
        
        energies = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            energies.append(rms)
        
        if not energies:
            return 0.5
        
        energies = np.array(energies)
        
        # Noise floor is estimated from the lower percentile of energy
        noise_floor_energy = np.percentile(energies, 10)
        
        # Normalize (typical speech RMS is around 0.1)
        normalized = min(noise_floor_energy / 0.05, 1.0)
        
        return normalized
    
    def _detect_outdoor(self, audio: np.ndarray) -> bool:
        """
        Detect if recording is outdoors
        Uses reverb, noise variability, and frequency characteristics
        """
        reverb = self._estimate_reverb(audio)
        noise_floor = self._estimate_noise_floor(audio)
        
        # Calculate energy variability
        frame_size = int(self.sample_rate * 0.1)
        energies = []
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            energies.append(np.sqrt(np.mean(frame ** 2)))
        
        if not energies:
            return False
        
        energy_variability = np.std(energies) / (np.mean(energies) + 1e-6)
        
        # Outdoor characteristics:
        # - Low reverb
        # - Variable noise (wind, traffic)
        # - High energy variability
        
        outdoor_score = 0
        
        if reverb < 0.3:
            outdoor_score += 0.4
        if energy_variability > 0.5:
            outdoor_score += 0.3
        if noise_floor > 0.3:
            outdoor_score += 0.3
        
        return outdoor_score > 0.5
    
    def _estimate_crowd_density(self, audio: np.ndarray) -> str:
        """
        Estimate crowd density from audio
        Returns: "none", "sparse", "moderate", "dense", "very_dense"
        """
        # Calculate spectral characteristics
        frame_size = int(self.sample_rate * 0.1)
        
        speech_activity = []
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            
            # Zero crossing rate (higher for multiple speakers)
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            
            # Energy
            energy = np.sqrt(np.mean(frame ** 2))
            
            # Combined activity metric
            if energy > 0.02:
                speech_activity.append(zcr)
        
        if not speech_activity:
            return "none"
        
        avg_activity = np.mean(speech_activity)
        variability = np.std(speech_activity)
        
        # High activity + high variability = crowd
        crowd_score = avg_activity * 10 + variability * 5
        
        if crowd_score < 0.5:
            return "none"
        elif crowd_score < 1.0:
            return "sparse"
        elif crowd_score < 2.0:
            return "moderate"
        elif crowd_score < 3.0:
            return "dense"
        else:
            return "very_dense"
    
    def _get_acoustic_signature(self, audio: np.ndarray) -> Dict[str, Any]:
        """Get acoustic signature for location matching"""
        return {
            "low_freq_energy": self._get_band_energy(audio, 0, 300),
            "mid_freq_energy": self._get_band_energy(audio, 300, 2000),
            "high_freq_energy": self._get_band_energy(audio, 2000, 8000),
            "spectral_centroid": self._get_spectral_centroid(audio),
            "spectral_rolloff": self._get_spectral_rolloff(audio)
        }
    
    def _get_band_energy(
        self,
        audio: np.ndarray,
        low_freq: int,
        high_freq: int
    ) -> float:
        """Calculate energy in frequency band"""
        try:
            # Simple FFT-based band energy
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_energy = np.sum(np.abs(fft[mask]) ** 2)
            total_energy = np.sum(np.abs(fft) ** 2)
            
            if total_energy > 0:
                return float(band_energy / total_energy)
            return 0.0
        except:
            return 0.33  # Default
    
    def _get_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid (brightness)"""
        try:
            fft = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            
            if np.sum(fft) > 0:
                centroid = np.sum(freqs * fft) / np.sum(fft)
                # Normalize to 0-1 range (assuming max 8kHz is bright)
                return min(centroid / 4000, 1.0)
            return 0.5
        except:
            return 0.5
    
    def _get_spectral_rolloff(self, audio: np.ndarray) -> float:
        """Calculate spectral rolloff (frequency below which 85% of energy lies)"""
        try:
            fft = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            
            cumsum = np.cumsum(fft ** 2)
            total = cumsum[-1]
            
            if total > 0:
                rolloff_idx = np.where(cumsum >= 0.85 * total)[0]
                if len(rolloff_idx) > 0:
                    rolloff_freq = freqs[rolloff_idx[0]]
                    return min(rolloff_freq / 8000, 1.0)
            return 0.5
        except:
            return 0.5
    
    def _get_energy_profile(self, audio: np.ndarray) -> Dict[str, float]:
        """Get energy profile over time"""
        # Divide audio into 10 segments
        segment_size = len(audio) // 10
        
        energies = []
        for i in range(10):
            start = i * segment_size
            end = start + segment_size
            segment = audio[start:end]
            rms = np.sqrt(np.mean(segment ** 2))
            energies.append(float(rms))
        
        return {
            "mean": round(np.mean(energies), 4),
            "std": round(np.std(energies), 4),
            "max": round(max(energies), 4),
            "min": round(min(energies), 4),
            "trend": "increasing" if energies[-1] > energies[0] * 1.2 else (
                "decreasing" if energies[-1] < energies[0] * 0.8 else "stable"
            )
        }
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio in dB"""
        frame_size = int(self.sample_rate * 0.02)
        
        energies = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            energies.append(np.mean(frame ** 2))
        
        if not energies:
            return 20.0
        
        energies = np.array(energies)
        
        # Signal power (top 10%)
        signal_power = np.percentile(energies, 90)
        
        # Noise power (bottom 10%)
        noise_power = np.percentile(energies, 10) + 1e-10
        
        snr = 10 * np.log10(signal_power / noise_power)
        
        return max(0, min(60, snr))
    
    def _get_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range (0 = flat, 1 = highly dynamic)"""
        frame_size = int(self.sample_rate * 0.05)
        
        energies = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            energies.append(np.sqrt(np.mean(frame ** 2)))
        
        if not energies or max(energies) == 0:
            return 0.5
        
        dynamic_range = (max(energies) - min(energies)) / max(energies)
        
        return dynamic_range
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "reverb_estimate": 0.5,
            "noise_floor": 0.5,
            "is_outdoor": False,
            "crowd_density": "unknown",
            "acoustic_signature": {},
            "energy_profile": {},
            "snr_estimate": 0,
            "dynamic_range": 0.5
        }