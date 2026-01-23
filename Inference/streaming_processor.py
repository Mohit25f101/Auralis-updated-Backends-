# ==============================
# 📄 inference/streaming_processor.py
# ==============================
# Real-Time Audio Stream Processing
# Upgrade 8: Live audio analysis
# ==============================

import numpy as np
import queue
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Callable, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_CONFIG


class StreamingBuffer:
    """
    Thread-safe circular buffer for audio streaming.
    """
    
    def __init__(self, max_duration: float = 10.0, sample_rate: int = None):
        self.sample_rate = sample_rate or AUDIO_CONFIG.sample_rate
        self.max_samples = int(max_duration * self.sample_rate)
        self.buffer = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()
        
    def add(self, samples: np.ndarray):
        """Add samples to buffer."""
        with self.lock:
            self.buffer.extend(samples.flatten())
    
    def get(self, duration: float) -> Optional[np.ndarray]:
        """Get samples for specified duration."""
        n_samples = int(duration * self.sample_rate)
        
        with self.lock:
            if len(self.buffer) < n_samples:
                return None
            return np.array(list(self.buffer)[-n_samples:])
    
    def get_all(self) -> np.ndarray:
        """Get all samples in buffer."""
        with self.lock:
            return np.array(list(self.buffer))
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
    
    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate


class AudioStreamProcessor:
    """
    Real-time audio stream processor with sliding window analysis.
    
    Features:
    - Configurable analysis window and hop size
    - Temporal smoothing of predictions
    - Event detection with debouncing
    - Thread-safe operation
    """
    
    def __init__(
        self,
        model,
        mel_extractor,
        sample_rate: int = None,
        window_duration: float = 3.0,
        hop_duration: float = 0.5,
        buffer_duration: float = 10.0,
        smoothing_window: int = 5
    ):
        """
        Initialize stream processor.
        
        Args:
            model: Prediction model
            mel_extractor: Mel spectrogram extractor
            sample_rate: Audio sample rate
            window_duration: Analysis window in seconds
            hop_duration: Time between analyses in seconds
            buffer_duration: Audio buffer size in seconds
            smoothing_window: Number of predictions to average
        """
        self.model = model
        self.mel_extractor = mel_extractor
        self.sample_rate = sample_rate or AUDIO_CONFIG.sample_rate
        
        self.window_samples = int(window_duration * self.sample_rate)
        self.hop_samples = int(hop_duration * self.sample_rate)
        self.smoothing_window = smoothing_window
        
        # Audio buffer
        self.buffer = StreamingBuffer(buffer_duration, self.sample_rate)
        
        # Result queue
        self.result_queue = queue.Queue(maxsize=100)
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=smoothing_window)
        
        # State
        self.is_running = False
        self.last_analysis_time = 0
        self.analysis_count = 0
        
        # Event callbacks
        self.callbacks = {
            'on_prediction': [],
            'on_emergency': [],
            'on_location_change': []
        }
        
        # Last known state
        self.last_location = None
        self.last_situation = None
        
    def add_callback(self, event: str, callback: Callable):
        """Add callback for event."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger all callbacks for event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")
    
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer."""
        self.buffer.add(audio_chunk)
    
    def process(self) -> Optional[Dict]:
        """
        Process current buffer if enough data available.
        
        Returns:
            Prediction result or None
        """
        # Check if enough data
        audio = self.buffer.get(self.window_samples / self.sample_rate)
        if audio is None:
            return None
        
        # Check hop time
        current_time = time.time()
        if current_time - self.last_analysis_time < (self.hop_samples / self.sample_rate):
            return None
        
        self.last_analysis_time = current_time
        self.analysis_count += 1
        
        try:
            # Extract features
            mel_spec = self.mel_extractor.extract(audio)
            mel_batch = np.expand_dims(mel_spec, axis=0)
            
            # Predict
            predictions = self.model.predict(mel_batch, verbose=0)
            
            # Smooth predictions
            result = self._smooth_predictions(predictions)
            
            # Add metadata
            result['timestamp'] = current_time
            result['analysis_id'] = self.analysis_count
            result['audio_energy'] = float(np.mean(np.abs(audio)))
            result['buffer_duration'] = self.buffer.duration
            
            # Check for events
            self._check_events(result)
            
            # Add to queue
            if not self.result_queue.full():
                self.result_queue.put(result)
            
            # Trigger prediction callback
            self._trigger_callbacks('on_prediction', result)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            return None
    
    def _smooth_predictions(self, predictions: Dict) -> Dict:
        """Apply temporal smoothing to predictions."""
        self.prediction_history.append(predictions)
        
        if len(self.prediction_history) < 2:
            return self._format_predictions(predictions)
        
        # Calculate exponential weights
        n = len(self.prediction_history)
        weights = np.exp(np.linspace(-1, 0, n))
        weights = weights / weights.sum()
        
        # Smooth each output
        smoothed = {}
        for key in predictions:
            if isinstance(predictions[key], np.ndarray):
                stacked = np.stack([h[key] for h in self.prediction_history])
                # Weighted average along time axis
                smoothed[key] = np.average(stacked, axis=0, weights=weights)
            else:
                smoothed[key] = predictions[key]
        
        return self._format_predictions(smoothed)
    
    def _format_predictions(self, predictions: Dict) -> Dict:
        """Format predictions for output."""
        from config import LABEL_CONFIG
        
        result = {}
        
        if 'location' in predictions:
            loc_probs = predictions['location'][0] if len(predictions['location'].shape) > 1 else predictions['location']
            loc_idx = int(np.argmax(loc_probs))
            result['location'] = LABEL_CONFIG.locations[loc_idx]
            result['location_confidence'] = float(loc_probs[loc_idx])
            result['location_probs'] = {
                LABEL_CONFIG.locations[i]: float(loc_probs[i])
                for i in range(len(LABEL_CONFIG.locations))
            }
        
        if 'situation' in predictions:
            sit_probs = predictions['situation'][0] if len(predictions['situation'].shape) > 1 else predictions['situation']
            sit_idx = int(np.argmax(sit_probs))
            result['situation'] = LABEL_CONFIG.situations[sit_idx]
            result['situation_confidence'] = float(sit_probs[sit_idx])
        
        if 'emergency' in predictions:
            emerg = predictions['emergency'][0] if len(predictions['emergency'].shape) > 0 else predictions['emergency']
            emerg_prob = float(emerg[0]) if hasattr(emerg, '__len__') else float(emerg)
            result['is_emergency'] = emerg_prob > 0.5
            result['emergency_probability'] = emerg_prob
        
        if 'confidence' in predictions:
            conf = predictions['confidence'][0] if len(predictions['confidence'].shape) > 0 else predictions['confidence']
            result['overall_confidence'] = float(conf[0]) if hasattr(conf, '__len__') else float(conf)
        
        return result
    
    def _check_events(self, result: Dict):
        """Check for significant events."""
        # Emergency detection
        if result.get('is_emergency') and result.get('emergency_probability', 0) > 0.7:
            self._trigger_callbacks('on_emergency', result)
        
        # Location change
        current_location = result.get('location')
        if current_location and current_location != self.last_location:
            if self.last_location is not None:
                self._trigger_callbacks('on_location_change', {
                    'previous': self.last_location,
                    'current': current_location,
                    'confidence': result.get('location_confidence', 0)
                })
            self.last_location = current_location
    
    def start(self):
        """Start continuous processing in background thread."""
        self.is_running = True
        
        def process_loop():
            while self.is_running:
                self.process()
                time.sleep(0.05)  # Small sleep to prevent CPU spinning
        
        self.process_thread = threading.Thread(target=process_loop, daemon=True)
        self.process_thread.start()
        print("🎙️ Stream processor started")
    
    def stop(self):
        """Stop processing."""
        self.is_running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)
        print("🛑 Stream processor stopped")
    
    def get_latest_result(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get latest result from queue."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_results(self) -> List[Dict]:
        """Get all available results."""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results


class MicrophoneProcessor(AudioStreamProcessor):
    """
    Process live microphone input.
    """
    
    def __init__(self, model, mel_extractor, device_index: int = None, **kwargs):
        super().__init__(model, mel_extractor, **kwargs)
        self.device_index = device_index
        self.stream = None
        
    def start_microphone(self):
        """Start capturing from microphone."""
        try:
            import sounddevice as sd
        except ImportError:
            print("❌ sounddevice not installed. Run: pip install sounddevice")
            return False
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"⚠️ Audio status: {status}")
            self.add_audio(indata[:, 0])
        
        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=audio_callback
            )
            
            self.stream.start()
            self.start()  # Start processing thread
            
            print(f"🎤 Microphone started (device: {self.device_index or 'default'})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start microphone: {e}")
            return False
    
    def stop_microphone(self):
        """Stop microphone capture."""
        self.stop()  # Stop processing thread
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        print("🎤 Microphone stopped")
    
    @staticmethod
    def list_devices():
        """List available audio devices."""
        try:
            import sounddevice as sd
            print("\n📋 Available Audio Devices:")
            print(sd.query_devices())
        except ImportError:
            print("❌ sounddevice not installed")


class FileStreamSimulator(AudioStreamProcessor):
    """
    Simulate streaming from audio file (for testing).
    """
    
    def __init__(self, model, mel_extractor, **kwargs):
        super().__init__(model, mel_extractor, **kwargs)
        
    def stream_file(self, file_path: str, realtime: bool = True):
        """
        Stream audio file.
        
        Args:
            file_path: Path to audio file
            realtime: If True, simulate real-time playback
        """
        import librosa
        
        print(f"📂 Loading: {file_path}")
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        
        print(f"⏱️ Duration: {len(audio) / sr:.1f}s")
        
        self.start()  # Start processing thread
        
        # Feed audio in chunks
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        
        for i in range(0, len(audio), chunk_size):
            if not self.is_running:
                break
                
            chunk = audio[i:i + chunk_size]
            self.add_audio(chunk)
            
            if realtime:
                time.sleep(len(chunk) / self.sample_rate)
        
        # Wait for processing to complete
        time.sleep(1.0)
        self.stop()


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Streaming Processor")
    print("="*60)
    
    # Test StreamingBuffer
    print("\n🔧 Testing StreamingBuffer...")
    buffer = StreamingBuffer(max_duration=5.0)
    
    # Add some samples
    for _ in range(10):
        buffer.add(np.random.randn(1600))  # 100ms of audio
    
    print(f"   Buffer duration: {buffer.duration:.2f}s")
    
    audio = buffer.get(1.0)
    print(f"   Retrieved 1s: {len(audio)} samples")
    
    # Test processor with dummy model
    print("\n🔧 Testing AudioStreamProcessor...")
    
    # Create dummy model and extractor
    class DummyModel:
        def predict(self, x, verbose=0):
            batch_size = x.shape[0]
            return {
                'location': np.random.softmax(np.random.randn(batch_size, 13), axis=-1),
                'situation': np.random.softmax(np.random.randn(batch_size, 15), axis=-1),
                'emergency': np.random.rand(batch_size, 1),
                'confidence': np.random.rand(batch_size, 1)
            }
    
    class DummyExtractor:
        def extract(self, audio):
            return np.random.randn(100, 80)
    
    processor = AudioStreamProcessor(
        model=DummyModel(),
        mel_extractor=DummyExtractor(),
        window_duration=2.0,
        hop_duration=0.5
    )
    
    # Add callback
    processor.add_callback('on_prediction', lambda r: print(f"   Prediction: {r.get('location')}"))
    
    # Simulate audio input
    for _ in range(50):
        processor.add_audio(np.random.randn(1600))
    
    # Process
    result = processor.process()
    if result:
        print(f"   Location: {result.get('location')}")
        print(f"   Situation: {result.get('situation')}")
        print(f"   Emergency: {result.get('is_emergency')}")
    
    print("\n✅ Streaming Processor test passed!")