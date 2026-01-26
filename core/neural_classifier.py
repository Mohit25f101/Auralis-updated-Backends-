# ==============================
# 📄 core/neural_classifier.py
# ==============================
# Neural Network Scene Classifier
# ==============================

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from config import MODEL, LABELS, CLASSIFIER_MODEL, get_device


class SceneClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural network for scene classification.
    Combines text embeddings, audio features, and sound embeddings.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 80,  # Mel bands
        sound_dim: int = 1024,  # YAMNet embedding
        hidden_dims: List[int] = None,
        num_locations: int = None,
        num_situations: int = None,
        dropout: float = None
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        hidden_dims = hidden_dims or MODEL.classifier_hidden_dims
        num_locations = num_locations or LABELS.num_locations
        num_situations = num_situations or LABELS.num_situations
        dropout = dropout or MODEL.classifier_dropout
        
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.sound_dim = sound_dim
        
        if not TORCH_AVAILABLE:
            return
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Audio encoder (for mel spectrogram)
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Sound encoder (for YAMNet embeddings)
        self.sound_encoder = nn.Sequential(
            nn.Linear(sound_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Fusion layers
        fusion_input_dim = hidden_dims[0] * 3
        fusion_layers = []
        
        prev_dim = fusion_input_dim
        for dim in hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        
        self.fusion = nn.Sequential(*fusion_layers)
        
        # Output heads
        self.location_head = nn.Linear(hidden_dims[-1], num_locations)
        self.situation_head = nn.Linear(hidden_dims[-1], num_situations)
        self.emergency_head = nn.Linear(hidden_dims[-1], 1)
        self.confidence_head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        sound_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text_features: [batch, text_dim]
            audio_features: [batch, audio_dim]
            sound_features: [batch, sound_dim]
        
        Returns:
            Dict with location, situation, emergency, confidence predictions
        """
        # Encode each modality
        text_enc = self.text_encoder(text_features)
        audio_enc = self.audio_encoder(audio_features)
        sound_enc = self.sound_encoder(sound_features)
        
        # Concatenate and fuse
        combined = torch.cat([text_enc, audio_enc, sound_enc], dim=-1)
        fused = self.fusion(combined)
        
        # Output predictions
        return {
            "location": F.softmax(self.location_head(fused), dim=-1),
            "situation": F.softmax(self.situation_head(fused), dim=-1),
            "emergency": torch.sigmoid(self.emergency_head(fused)),
            "confidence": torch.sigmoid(self.confidence_head(fused)),
        }


class NeuralClassifier:
    """
    Wrapper for neural scene classifier with inference utilities.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(CLASSIFIER_MODEL)
        self.device = get_device()
        self.model: Optional[SceneClassifier] = None
        self.is_loaded = False
    
    def load(self) -> bool:
        """Load or initialize model."""
        if not TORCH_AVAILABLE:
            print("❌ NeuralClassifier: PyTorch not available")
            return False
        
        try:
            self.model = SceneClassifier()
            
            # Load weights if available
            if CLASSIFIER_MODEL.exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Loaded classifier from {self.model_path}")
            else:
                print("ℹ️ Classifier initialized with random weights")
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ NeuralClassifier failed: {e}")
            return False
    
    def predict(
        self,
        text_features: np.ndarray,
        audio_features: np.ndarray,
        sound_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Make predictions.
        
        Args:
            text_features: Text embedding [text_dim]
            audio_features: Audio features [audio_dim]
            sound_features: Sound embedding [sound_dim]
        
        Returns:
            Dict with location, situation, emergency, confidence
        """
        if not self.is_loaded or self.model is None:
            return self._fallback_prediction()
        
        try:
            with torch.no_grad():
                # Prepare inputs
                text_t = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                audio_t = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                sound_t = torch.tensor(sound_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs = self.model(text_t, audio_t, sound_t)
                
                # Get predictions
                location_probs = outputs["location"][0].cpu().numpy()
                situation_probs = outputs["situation"][0].cpu().numpy()
                emergency_prob = float(outputs["emergency"][0].cpu().numpy())
                confidence = float(outputs["confidence"][0].cpu().numpy())
                
                # Get top predictions
                location_idx = int(np.argmax(location_probs))
                situation_idx = int(np.argmax(situation_probs))
                
                return {
                    "location": LABELS.idx_to_location(location_idx),
                    "location_confidence": float(location_probs[location_idx]),
                    "location_probs": location_probs.tolist(),
                    "situation": LABELS.idx_to_situation(situation_idx),
                    "situation_confidence": float(situation_probs[situation_idx]),
                    "situation_probs": situation_probs.tolist(),
                    "is_emergency": emergency_prob > 0.5,
                    "emergency_probability": emergency_prob,
                    "model_confidence": confidence,
                }
                
        except Exception as e:
            print(f"⚠️ Neural prediction failed: {e}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self) -> Dict[str, Any]:
        """Return fallback when model unavailable."""
        return {
            "location": "Unknown",
            "location_confidence": 0.0,
            "situation": "Unknown",
            "situation_confidence": 0.0,
            "is_emergency": False,
            "emergency_probability": 0.0,
            "model_confidence": 0.0,
            "fallback": True,
        }
    
    def save(self, path: str = None):
        """Save model weights."""
        if self.model is not None:
            save_path = path or self.model_path
            torch.save(self.model.state_dict(), save_path)
            print(f"💾 Saved classifier to {save_path}")


# ==============================
# 🧪 TESTING
# ==============================
if __name__ == "__main__":
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        classifier = NeuralClassifier()
        if classifier.load():
            # Test with dummy features
            text_feat = np.random.randn(768).astype(np.float32)
            audio_feat = np.random.randn(80).astype(np.float32)
            sound_feat = np.random.randn(1024).astype(np.float32)
            
            result = classifier.predict(text_feat, audio_feat, sound_feat)
            print(f"Location: {result['location']} ({result['location_confidence']:.2f})")
            print(f"Situation: {result['situation']} ({result['situation_confidence']:.2f})")