# ==============================
# services/audio_loader.py - FIXED
# ==============================
import os
import subprocess
import tempfile
from typing import Tuple, Optional
import numpy as np
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None
try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    wavfile = None
class AudioLoader:
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm', '.aac', '.wma'}
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.ffmpeg_available = self._check_ffmpeg()
        self.loaded = True
        backends = []
        if LIBROSA_AVAILABLE:
            backends.append("Librosa")
        if SOUNDFILE_AVAILABLE:
            backends.append("Soundfile")
        if SCIPY_AVAILABLE:
            backends.append("Scipy")
        if self.ffmpeg_available:
            backends.append("FFmpeg")
        if backends:
            print(f"? Audio Loader initialized (backends: {', '.join(backends)})")
        else:
            print("?? Audio Loader: No backends available!")
    def _check_ffmpeg(self) -> bool:
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    def load(self, path: str) -> Tuple[np.ndarray, float]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED_FORMATS}")
        audio = None
        method_used = None
        # Try Librosa
        if LIBROSA_AVAILABLE and audio is None:
            try:
                audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
                method_used = "Librosa"
            except Exception as e:
                pass
        # Try Soundfile
        if SOUNDFILE_AVAILABLE and audio is None:
            try:
                data, orig_sr = sf.read(path)
                audio = self._process_audio(data, orig_sr)
                method_used = "Soundfile"
            except:
                pass
        # Try FFmpeg
        if self.ffmpeg_available and audio is None:
            try:
                audio = self._load_with_ffmpeg(path)
                if audio is not None:
                    method_used = "FFmpeg"
            except:
                pass
        # Try Scipy for WAV
        if SCIPY_AVAILABLE and audio is None and ext == '.wav':
            try:
                orig_sr, data = wavfile.read(path)
                audio = self._process_audio(data, orig_sr)
                method_used = "Scipy"
            except:
                pass
        if audio is None:
            raise RuntimeError("Failed to load audio")
        audio = self._normalize(audio)
        duration = len(audio) / self.sample_rate
        print(f"   ? Loaded with {method_used} ({duration:.2f}s)")
        return audio, duration
    def _process_audio(self, data: np.ndarray, orig_sr: int) -> np.ndarray:
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        else:
            audio = data.astype(np.float32)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if orig_sr != self.sample_rate and LIBROSA_AVAILABLE:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
        return audio
    def _load_with_ffmpeg(self, path: str) -> Optional[np.ndarray]:
        temp_wav = os.path.join(tempfile.gettempdir(), f"auralis_{os.getpid()}.wav")
        try:
            cmd = ["ffmpeg", "-y", "-i", path, "-ar", str(self.sample_rate), "-ac", "1", "-f", "wav", temp_wav]
            subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.exists(temp_wav) and SCIPY_AVAILABLE:
                orig_sr, data = wavfile.read(temp_wav)
                return self._process_audio(data, orig_sr)
            return None
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32)
        audio = audio - np.mean(audio)
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx * 0.95
        return audio
    def save_wav(self, audio: np.ndarray, path: str) -> str:
        audio_int16 = (audio * 32767).astype(np.int16)
        if SCIPY_AVAILABLE:
            wavfile.write(path, self.sample_rate, audio_int16)
        elif SOUNDFILE_AVAILABLE:
            sf.write(path, audio_int16, self.sample_rate)
        return path
