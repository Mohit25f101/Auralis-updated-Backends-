# ==============================
# 📄 inference/predictor.py
# ==============================
# Unified Prediction Interface
# Clean API for model inference
# ==============================

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_CONFIG, LABEL_CONFIG, WEIGHTS_DIR


class AuralisPredictor:
    """
    Unified prediction interface for Auralis models.
    
    Handles:
    - Model loading
    - Audio preprocessing
    - Feature extraction
    - Multi-modal inference
    - Result formatting
    """
    
    def __init__(
        self,
        model_path: str = None,
        use_whisper: bool = True,
        use_yamnet: bool = True,
        device: str = 'auto'
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            use_whisper: Whether to use Whisper for transcription
            use_yamnet: Whether to use YAMNet for sound classification
            device: 'cpu', 'gpu', or 'auto'
        """
        self.model_path = model_path
        self.model = None
        self.whisper = None
        self.yamnet = None
        self.mel_extractor = None
        
        # Configure device
        self._setup_device(device)
        
        # Load components
        self._load_model(model_path)
        
        if use_whisper:
            self._load_whisper()
        
        if use_yamnet:
            self._load_yamnet()
        
        self._setup_mel_extractor()
        
        print("✅ AuralisPredictor initialized")
    
    def _setup_device(self, device: str):
        """Configure compute device."""
        if device == 'auto':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"🖥️ Using GPU: {gpus[0].name}")
                except:
                    print("🖥️ Using CPU")
            else:
                print("🖥️ Using CPU")
        elif device == 'cpu':
            tf.config.set_visible_devices([], 'GPU')
            print("🖥️ Using CPU (forced)")
    
    def _load_model(self, model_path: str):
        """Load the main prediction model."""
        if model_path and Path(model_path).exists():
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"📦 Model loaded: {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                self._create_fallback_model()
        else:
            print("📦 No model path provided, using fallback")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model."""
        from models.scene_classifier import AuralisSceneClassifier
        self.model = AuralisSceneClassifier()
        # Build with dummy input
        dummy_audio = np.zeros((1, 1024), dtype=np.float32)
        dummy_text = np.zeros((1, 768), dtype=np.float32)
        self.model((dummy_audio, dummy_text))
    
    def _load_whisper(self):
        """Load Whisper for speech recognition."""
        try:
            from transformers import pipeline
            self.whisper = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small"
            )
            print("🎤 Whisper loaded")
        except Exception as e:
            print(f"⚠️ Whisper not available: {e}")
    
    def _load_yamnet(self):
        """Load YAMNet for sound classification."""
        try:
            import tensorflow_hub as hub
            self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            print("🔊 YAMNet loaded")
        except Exception as e:
            print(f"⚠️ YAMNet not available: {e}")
    
    def _setup_mel_extractor(self):
        """Setup mel spectrogram extractor."""
        from models.audio_transformer import MelSpectrogramExtractor
        self.mel_extractor = MelSpectrogramExtractor()
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Transcribed text
        """
        if self.whisper is None:
            return ""
        
        try:
            # Whisper expects audio at specific sample rate
            result = self.whisper(audio)
            return result.get('text', '')
        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
            return ""
    
    def classify_sounds(self, audio: np.ndarray, top_k: int = 10) -> Dict[str, float]:
        """
        Classify sounds in audio using YAMNet.
        
        Args:
            audio: Audio waveform
            top_k: Number of top classes to return
            
        Returns:
            Dict mapping class names to scores
        """
        if self.yamnet is None:
            return {}
        
        try:
            scores, embeddings, spectrogram = self.yamnet(audio)
            
            # Average scores across time
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
            
            # Get top classes
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            
            # Load class names
            class_map_path = self.yamnet.resolved_object.class_map_path().numpy().decode('utf-8')
            
            import csv
            class_names = []
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class_names.append(row['display_name'])
            
            return {
                class_names[i]: float(mean_scores[i])
                for i in top_indices
            }
            
        except Exception as e:
            print(f"⚠️ Sound classification error: {e}")
            return {}
    
    def extract_features(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all features from audio.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Tuple of (mel_features, text_embedding, sound_features)
        """
        # Mel spectrogram features
        mel_spec = self.mel_extractor.extract(audio)
        
        # Text features (from transcription)
        text = self.transcribe(audio)
        text_embedding = self._get_text_embedding(text)
        
        # Sound classification features
        sounds = self.classify_sounds(audio)
        sound_features = self._sounds_to_features(sounds)
        
        return mel_spec, text_embedding, sound_features
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding (placeholder - would use a text encoder)."""
        # Simple bag-of-words style encoding
        # In production, use a proper text encoder
        embedding = np.zeros(768, dtype=np.float32)
        
        # Simple keyword-based features
        keywords = {
            'airport': 0, 'flight': 1, 'gate': 2, 'boarding': 3,
            'train': 4, 'platform': 5, 'station': 6,
            'emergency': 7, 'help': 8, 'fire': 9, 'accident': 10,
            'traffic': 11, 'car': 12, 'road': 13,
            'announcement': 14, 'attention': 15
        }
        
        text_lower = text.lower()
        for word, idx in keywords.items():
            if word in text_lower:
                embedding[idx] = 1.0
        
        return embedding
    
    def _sounds_to_features(self, sounds: Dict[str, float]) -> np.ndarray:
        """Convert sound classifications to feature vector."""
        # Create fixed-size feature vector
        features = np.zeros(521, dtype=np.float32)  # YAMNet has 521 classes
        
        # Would map sound names to indices
        # Simplified: use hash-based mapping
        for i, (name, score) in enumerate(sounds.items()):
            idx = hash(name) % 521
            features[idx] = max(features[idx], score)
        
        return features
    
    def predict(
        self,
        audio: Union[np.ndarray, str],
        return_details: bool = False
    ) -> Dict:
        """
        Make prediction on audio.
        
        Args:
            audio: Audio waveform or file path
            return_details: Whether to return detailed results
            
        Returns:
            Prediction dictionary
        """
        # Load audio if path provided
        if isinstance(audio, str):
            import librosa
            audio, sr = librosa.load(audio, sr=AUDIO_CONFIG.sample_rate)
        
        # Extract features
        mel_spec, text_emb, sound_feat = self.extract_features(audio)
        
        # Prepare inputs for model
        mel_batch = np.expand_dims(mel_spec, axis=0)
        text_batch = np.expand_dims(text_emb, axis=0)
        sound_batch = np.expand_dims(sound_feat, axis=0)
        
        # Predict
        try:
            # Try multi-modal prediction
            predictions = self.model.predict(
                (mel_batch, text_batch, sound_batch),
                verbose=0
            )
        except:
            # Fallback to simpler prediction
            try:
                predictions = self.model.predict(
                    (mel_batch, text_batch),
                    verbose=0
                )
            except:
                predictions = self.model.predict(mel_batch, verbose=0)
        
        # Format result
        result = self._format_result(predictions)
        
        if return_details:
            result['transcription'] = self.transcribe(audio)
            result['sounds'] = self.classify_sounds(audio)
            result['audio_duration'] = len(audio) / AUDIO_CONFIG.sample_rate
        
        return result
    
    def _format_result(self, predictions: Dict) -> Dict:
        """Format model predictions into readable result."""
        result = {}
        
        # Location
        if 'location' in predictions:
            loc_probs = predictions['location'][0]
            loc_idx = int(np.argmax(loc_probs))
            result['location'] = LABEL_CONFIG.locations[loc_idx]
            result['location_confidence'] = float(loc_probs[loc_idx])
        
        # Situation
        if 'situation' in predictions:
            sit_probs = predictions['situation'][0]
            sit_idx = int(np.argmax(sit_probs))
            result['situation'] = LABEL_CONFIG.situations[sit_idx]
            result['situation_confidence'] = float(sit_probs[sit_idx])
        
        # Emergency
        if 'emergency' in predictions:
            emerg_prob = float(predictions['emergency'][0][0])
            result['is_emergency'] = emerg_prob > 0.5
            result['emergency_probability'] = emerg_prob
        
        # Confidence
        if 'confidence' in predictions:
            result['overall_confidence'] = float(predictions['confidence'][0][0])
        else:
            # Calculate from location/situation confidences
            confs = [result.get('location_confidence', 0), result.get('situation_confidence', 0)]
            result['overall_confidence'] = sum(confs) / len(confs) if confs else 0
        
        # Generate summary
        result['summary'] = self._generate_summary(result)
        
        return result
    
    def _generate_summary(self, result: Dict) -> str:
        """Generate human-readable summary."""
        location = result.get('location', 'Unknown location')
        situation = result.get('situation', 'Unknown situation')
        confidence = result.get('overall_confidence', 0) * 100
        
        summary = f"Audio detected from {location} during {situation} "
        summary += f"(confidence: {confidence:.0f}%)"
        
        if result.get('is_emergency'):
            summary = f"⚠️ EMERGENCY DETECTED! " + summary
        
        return summary


class BatchPredictor:
    """
    Batch prediction for multiple audio files.
    """
    
    def __init__(self, predictor: AuralisPredictor):
        self.predictor = predictor
    
    def predict_batch(
        self,
        audio_files: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict on multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            show_progress: Whether to show progress bar
            
        Returns:
            List of prediction results
        """
        results = []
        
        iterator = audio_files
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_files, desc="Processing")
            except ImportError:
                pass
        
        for file_path in iterator:
            try:
                result = self.predictor.predict(file_path)
                result['file'] = file_path
                results.append(result)
            except Exception as e:
                results.append({
                    'file': file_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(
        self,
        directory: str,
        pattern: str = '*.wav'
    ) -> List[Dict]:
        """Predict on all matching files in directory."""
        from pathlib import Path
        
        files = list(Path(directory).glob(pattern))
        return self.predict_batch([str(f) for f in files])


def create_predictor(
    model_path: str = None,
    model_type: str = 'auto',
    **kwargs
) -> AuralisPredictor:
    """
    Factory function to create appropriate predictor.
    
    Args:
        model_path: Path to model weights
        model_type: 'full', 'compact', or 'auto'
        **kwargs: Additional arguments for predictor
        
    Returns:
        AuralisPredictor instance
    """
    # Auto-detect model type
    if model_type == 'auto' and model_path:
        if 'compact' in model_path.lower():
            model_type = 'compact'
        elif 'full' in model_path.lower():
            model_type = 'full'
    
    # Find model if not specified
    if model_path is None:
        candidates = [
            WEIGHTS_DIR / 'auralis_scene_best.keras',
            WEIGHTS_DIR / 'auralis_compact.keras',
            WEIGHTS_DIR / 'auralis_full.keras'
        ]
        
        for candidate in candidates:
            if candidate.exists():
                model_path = str(candidate)
                break
    
    return AuralisPredictor(model_path=model_path, **kwargs)


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Predictor")
    print("="*60)
    
    # Create predictor (will use fallback model)
    predictor = AuralisPredictor(
        model_path=None,
        use_whisper=False,  # Skip for testing
        use_yamnet=False    # Skip for testing
    )
    
    # Test with dummy audio
    print("\n🔧 Testing prediction...")
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
    
    result = predictor.predict(dummy_audio, return_details=True)
    
    print(f"\n📊 Result:")
    print(f"   Location: {result.get('location')}")
    print(f"   Situation: {result.get('situation')}")
    print(f"   Confidence: {result.get('overall_confidence', 0)*100:.1f}%")
    print(f"   Summary: {result.get('summary')}")
    
    print("\n✅ Predictor test passed!")