# ==============================
# 📄 utils/audio_utils.py
# ==============================
# Audio Processing Utilities
# ==============================

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union, List
import subprocess
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_CONFIG


def load_audio(
    filepath: Union[str, Path],
    sample_rate: int = None,
    mono: bool = True,
    duration: float = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with various options.
    
    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate (None for original)
        mono: Convert to mono
        duration: Duration to load in seconds
        offset: Start time offset in seconds
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    filepath = Path(filepath)
    sr = sample_rate or AUDIO_CONFIG.sample_rate
    
    audio, loaded_sr = librosa.load(
        str(filepath),
        sr=sr,
        mono=mono,
        duration=duration,
        offset=offset
    )
    
    return audio.astype(np.float32), loaded_sr


def save_audio(
    audio: np.ndarray,
    filepath: Union[str, Path],
    sample_rate: int = None,
    format: str = None
) -> str:
    """
    Save audio to file.
    
    Args:
        audio: Audio array
        filepath: Output path
        sample_rate: Sample rate
        format: Output format (wav, flac, mp3, etc.)
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    sr = sample_rate or AUDIO_CONFIG.sample_rate
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize to prevent clipping
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio)) * 0.99
    
    sf.write(str(filepath), audio, sr, format=format)
    
    return str(filepath)


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -3.0,
    method: str = 'peak'
) -> np.ndarray:
    """
    Normalize audio level.
    
    Args:
        audio: Input audio
        target_db: Target level in dB
        method: 'peak' or 'rms' normalization
        
    Returns:
        Normalized audio
    """
    if method == 'peak':
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = 10 ** (target_db / 20)
            audio = audio * (target_peak / peak)
    elif method == 'rms':
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)
    
    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio.astype(np.float32)


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        audio: Input audio
        threshold_db: Silence threshold in dB
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Tuple of (trimmed_audio, (start_sample, end_sample))
    """
    trimmed, (start, end) = librosa.effects.trim(
        audio,
        top_db=-threshold_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return trimmed.astype(np.float32), (start, end)


def split_audio(
    audio: np.ndarray,
    sample_rate: int,
    segment_duration: float,
    overlap: float = 0.0
) -> List[np.ndarray]:
    """
    Split audio into fixed-length segments.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        segment_duration: Segment duration in seconds
        overlap: Overlap ratio (0.0 to 1.0)
        
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    # Handle last segment
    if start < len(audio):
        last_segment = np.zeros(segment_samples, dtype=np.float32)
        remaining = audio[start:]
        last_segment[:len(remaining)] = remaining
        segments.append(last_segment)
    
    return segments


def merge_audio(
    segments: List[np.ndarray],
    crossfade_samples: int = 0
) -> np.ndarray:
    """
    Merge audio segments with optional crossfade.
    
    Args:
        segments: List of audio segments
        crossfade_samples: Number of samples for crossfade
        
    Returns:
        Merged audio
    """
    if not segments:
        return np.array([], dtype=np.float32)
    
    if crossfade_samples == 0:
        return np.concatenate(segments)
    
    # Calculate total length
    total_length = sum(len(s) for s in segments) - crossfade_samples * (len(segments) - 1)
    merged = np.zeros(total_length, dtype=np.float32)
    
    position = 0
    for i, segment in enumerate(segments):
        if i == 0:
            merged[position:position + len(segment)] = segment
            position += len(segment) - crossfade_samples
        else:
            # Create crossfade
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
            
            # Apply crossfade
            merged[position:position + crossfade_samples] *= fade_out
            merged[position:position + crossfade_samples] += segment[:crossfade_samples] * fade_in
            
            # Add rest of segment
            merged[position + crossfade_samples:position + len(segment)] = segment[crossfade_samples:]
            position += len(segment) - crossfade_samples
    
    return merged


def get_audio_info(filepath: Union[str, Path]) -> dict:
    """
    Get audio file information.
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    filepath = Path(filepath)
    
    info = sf.info(str(filepath))
    
    return {
        'filepath': str(filepath),
        'filename': filepath.name,
        'format': info.format,
        'subtype': info.subtype,
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'duration': info.duration,
        'frames': info.frames,
        'file_size_mb': filepath.stat().st_size / (1024 * 1024)
    }


def convert_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = None,
    channels: int = None,
    format: str = None
) -> str:
    """
    Convert audio file format.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        sample_rate: Target sample rate
        channels: Target number of channels
        format: Output format
        
    Returns:
        Path to converted file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Try using ffmpeg if available
    try:
        cmd = ['ffmpeg', '-y', '-i', str(input_path)]
        
        if sample_rate:
            cmd.extend(['-ar', str(sample_rate)])
        if channels:
            cmd.extend(['-ac', str(channels)])
        
        cmd.append(str(output_path))
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to librosa
        audio, sr = load_audio(input_path, sample_rate=sample_rate, mono=(channels == 1))
        save_audio(audio, output_path, sample_rate=sr, format=format)
        return str(output_path)


def compute_features(
    audio: np.ndarray,
    sample_rate: int = None,
    feature_type: str = 'mel'
) -> np.ndarray:
    """
    Compute audio features.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        feature_type: Type of features ('mel', 'mfcc', 'chroma', 'spectral')
        
    Returns:
        Feature array
    """
    sr = sample_rate or AUDIO_CONFIG.sample_rate
    
    if feature_type == 'mel':
        features = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=AUDIO_CONFIG.n_mels,
            n_fft=AUDIO_CONFIG.n_fft,
            hop_length=AUDIO_CONFIG.hop_length
        )
        features = librosa.power_to_db(features, ref=np.max)
        
    elif feature_type == 'mfcc':
        features = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40,
            n_fft=AUDIO_CONFIG.n_fft,
            hop_length=AUDIO_CONFIG.hop_length
        )
        
    elif feature_type == 'chroma':
        features = librosa.feature.chroma_stft(
            y=audio,
            sr=sr,
            n_fft=AUDIO_CONFIG.n_fft,
            hop_length=AUDIO_CONFIG.hop_length
        )
        
    elif feature_type == 'spectral':
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features = np.vstack([centroid, bandwidth, rolloff])
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return features.T.astype(np.float32)


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Audio Utils")
    print("="*60)
    
    # Generate test audio
    duration = 3.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    print(f"\n📊 Test audio: {len(audio)} samples ({duration}s)")
    
    # Test save and load
    print("\n💾 Testing save/load...")
    save_audio(audio, "./test_audio.wav", sr)
    loaded, loaded_sr = load_audio("./test_audio.wav")
    print(f"   Saved and loaded: {len(loaded)} samples at {loaded_sr}Hz")
    
    # Test normalization
    print("\n🔊 Testing normalization...")
    quiet_audio = audio * 0.1
    normalized = normalize_audio(quiet_audio, target_db=-3.0)
    print(f"   Before peak: {np.max(np.abs(quiet_audio)):.4f}")
    print(f"   After peak: {np.max(np.abs(normalized)):.4f}")
    
    # Test splitting
    print("\n✂️ Testing split...")
    segments = split_audio(audio, sr, segment_duration=1.0, overlap=0.5)
    print(f"   Split into {len(segments)} segments")
    
    # Test merging
    print("\n🔗 Testing merge...")
    merged = merge_audio(segments, crossfade_samples=int(0.1 * sr))
    print(f"   Merged length: {len(merged)} samples")
    
    # Test features
    print("\n📈 Testing feature extraction...")
    for feat_type in ['mel', 'mfcc', 'chroma']:
        features = compute_features(audio, sr, feat_type)
        print(f"   {feat_type}: {features.shape}")
    
    # Cleanup
    os.remove("./test_audio.wav")
    
    print("\n✅ Audio Utils test passed!")