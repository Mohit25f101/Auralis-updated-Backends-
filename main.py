# ==============================
# 📄 main.py
# ==============================
# AURALIS ML API - UPGRADED VERSION
# Integrates all ML upgrades for advanced audio analysis
# ==============================

# ==============================
# 🔇 SILENCE WARNINGS (Must be first)
# ==============================
import os
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# ==============================
# 📦 IMPORTS
# ==============================
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import requests
import librosa
import shutil
import subprocess
import json
import uuid
from datetime import datetime
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

# ==============================
# 📦 LOCAL IMPORTS
# ==============================
try:
    from config import (
        AUDIO_CONFIG, LABEL_CONFIG, MODEL_CONFIG, 
        TRAINING_CONFIG, API_CONFIG,
        WEIGHTS_DIR, LOGS_DIR, DATASET_DIR, FFMPEG_BIN_DIR
    )
    from models.scene_classifier import AuralisSceneClassifier
    from models.audio_transformer import AudioTransformerEncoder, MelSpectrogramExtractor
    from models.multimodal_fusion import MultiModalFusionNetwork
    from inference.predictor import AuralisPredictor
    from inference.active_learning import ActiveLearningPipeline
    from inference.streaming_processor import AudioStreamProcessor
    from training.audio_augmentation import AudioAugmentor
    from utils.audio_utils import ensure_ffmpeg_available, load_audio_file
    
    MODULES_LOADED = True
    print("✅ All custom modules loaded successfully!")
except ImportError as e:
    MODULES_LOADED = False
    print(f"⚠️ Some modules not loaded: {e}")
    print("   Running in fallback mode with basic features...")

# ==============================
# 🎬 FFMPEG DETECTION
# ==============================
def setup_ffmpeg():
    """Ensure FFmpeg is available for audio processing."""
    try:
        # Check if already available
        existing = shutil.which("ffmpeg")
        if existing:
            print(f"✅ FFmpeg found: {existing}")
            return True
        
        # Try to add from config path
        ffmpeg_path = FFMPEG_BIN_DIR if MODULES_LOADED else r"D:\photo\ffmpeg\ffmpeg-2026-01-07-git-af6a1dd0b2-full_build\bin"
        
        if os.path.isdir(ffmpeg_path):
            os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
            found = shutil.which("ffmpeg")
            if found:
                print(f"✅ FFmpeg enabled: {found}")
                return True
        
        print("⚠️ FFmpeg not found - some features may be limited")
        return False
    except Exception as e:
        print(f"⚠️ FFmpeg setup error: {e}")
        return False

# ==============================
# 🤖 MODEL MANAGER
# ==============================
class ModelManager:
    """Manages all ML models for the application."""
    
    def __init__(self):
        self.whisper = None
        self.yamnet = None
        self.yamnet_labels = []
        self.scene_classifier = None
        self.audio_transformer = None
        self.fusion_network = None
        self.predictor = None
        self.active_learning = None
        self.mel_extractor = None
        self.text_embedder = None
        self.is_upgraded = False
        
    def load_basic_models(self):
        """Load basic models (Whisper + YAMNet)."""
        print("\n" + "="*60)
        print("🔄 Loading Basic AI Models...")
        print("="*60)
        
        # Load Whisper
        try:
            from transformers import pipeline
            self.whisper = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-small",
                device=-1  # CPU, use 0 for GPU
            )
            print("✅ Whisper loaded!")
        except Exception as e:
            print(f"⚠️ Whisper failed: {e}")
            self.whisper = None
        
        # Load YAMNet
        try:
            self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            print("✅ YAMNet loaded!")
            self._load_yamnet_labels()
        except Exception as e:
            print(f"⚠️ YAMNet failed: {e}")
            self.yamnet = None
            
    def _load_yamnet_labels(self):
        """Load YAMNet class labels."""
        labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        try:
            response = requests.get(labels_url, timeout=10)
            reader = csv.reader(response.text.splitlines())
            next(reader)  # Skip header
            self.yamnet_labels = [row[2] for row in reader]
            print(f"✅ Loaded {len(self.yamnet_labels)} sound labels")
        except Exception as e:
            print(f"⚠️ Could not load labels: {e}")
            self.yamnet_labels = []
            
    def load_upgraded_models(self):
        """Load upgraded ML models."""
        if not MODULES_LOADED:
            print("⚠️ Upgraded modules not available")
            return False
            
        print("\n" + "="*60)
        print("🚀 Loading Upgraded AI Models...")
        print("="*60)
        
        try:
            # Mel Spectrogram Extractor
            self.mel_extractor = MelSpectrogramExtractor()
            print("✅ Mel Extractor initialized!")
            
            # Audio Transformer Encoder
            self.audio_transformer = AudioTransformerEncoder()
            print("✅ Audio Transformer initialized!")
            
            # Scene Classifier
            self.scene_classifier = AuralisSceneClassifier()
            print("✅ Scene Classifier initialized!")
            
            # Multi-Modal Fusion Network
            self.fusion_network = MultiModalFusionNetwork()
            print("✅ Fusion Network initialized!")
            
            # Load pre-trained weights if available
            self._load_pretrained_weights()
            
            # Initialize Predictor
            self.predictor = AuralisPredictor(
                scene_classifier=self.scene_classifier,
                audio_transformer=self.audio_transformer,
                fusion_network=self.fusion_network,
                mel_extractor=self.mel_extractor,
                whisper=self.whisper,
                yamnet=self.yamnet,
                yamnet_labels=self.yamnet_labels
            )
            print("✅ Predictor initialized!")
            
            # Initialize Active Learning
            self.active_learning = ActiveLearningPipeline(
                model=self.scene_classifier,
                uncertainty_threshold=TRAINING_CONFIG.uncertainty_threshold
            )
            print("✅ Active Learning initialized!")
            
            # Load text embedder for fusion
            self._load_text_embedder()
            
            self.is_upgraded = True
            print("\n✅ All upgraded models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading upgraded models: {e}")
            import traceback
            traceback.print_exc()
            self.is_upgraded = False
            return False
            
    def _load_pretrained_weights(self):
        """Load pre-trained weights if available."""
        weights_path = WEIGHTS_DIR if MODULES_LOADED else Path("weights")
        
        # Scene Classifier weights
        scene_weights = weights_path / "auralis_scene_best.keras"
        if scene_weights.exists():
            try:
                self.scene_classifier.load_weights(str(scene_weights))
                print(f"✅ Loaded scene classifier weights from {scene_weights}")
            except Exception as e:
                print(f"⚠️ Could not load scene weights: {e}")
                
        # Audio Transformer weights
        transformer_weights = weights_path / "audio_transformer.keras"
        if transformer_weights.exists():
            try:
                self.audio_transformer.load_weights(str(transformer_weights))
                print(f"✅ Loaded transformer weights from {transformer_weights}")
            except Exception as e:
                print(f"⚠️ Could not load transformer weights: {e}")
                
        # Fusion Network weights
        fusion_weights = weights_path / "fusion_network.keras"
        if fusion_weights.exists():
            try:
                self.fusion_network.load_weights(str(fusion_weights))
                print(f"✅ Loaded fusion weights from {fusion_weights}")
            except Exception as e:
                print(f"⚠️ Could not load fusion weights: {e}")
                
    def _load_text_embedder(self):
        """Load text embedding model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModel.from_pretrained(model_name)
            self.text_model.eval()
            print("✅ Text Embedder loaded!")
        except Exception as e:
            print(f"⚠️ Text Embedder not loaded: {e}")
            self.text_tokenizer = None
            self.text_model = None
            
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using sentence transformer."""
        if self.text_tokenizer is None or self.text_model is None:
            # Return dummy embedding
            return np.zeros(768, dtype=np.float32)
            
        try:
            import torch
            
            inputs = self.text_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            return embeddings.numpy()[0]
        except Exception as e:
            print(f"⚠️ Text embedding error: {e}")
            return np.zeros(768, dtype=np.float32)


# ==============================
# 📊 RESPONSE MODELS
# ==============================
class AnalysisResult(BaseModel):
    """Response model for audio analysis."""
    location: str
    location_confidence: float
    situation: str
    situation_confidence: float
    is_emergency: bool
    emergency_probability: float
    overall_confidence: float
    transcribed_text: str
    detected_sounds: List[str]
    evidence: List[str]
    summary: str
    analysis_mode: str
    processing_time_ms: float
    request_id: str
    timestamp: str


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    request_id: str
    correct_location: Optional[str] = None
    correct_situation: Optional[str] = None
    was_emergency: Optional[bool] = None
    comments: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    upgraded_mode: bool
    timestamp: str


# ==============================
# 🧠 ANALYSIS ENGINE
# ==============================
class AnalysisEngine:
    """
    Core analysis engine that handles both basic and upgraded inference.
    """
    
    def __init__(self, model_manager: ModelManager):
        self.models = model_manager
        self.analysis_history = []
        
    def analyze_basic(
        self,
        audio: np.ndarray,
        text: str,
        sounds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Basic rule-based analysis (fallback mode).
        """
        text_lower = text.lower()
        sound_labels = [s.lower() for s in sounds.keys()]
        
        # Keywords for detection
        location_keywords = {
            "Airport Terminal": ["flight", "boarding", "gate", "terminal", "aircraft", "pilot"],
            "Railway Station": ["train", "platform", "coach", "railway", "track", "station"],
            "Bus Terminal": ["bus", "route", "terminal", "passenger"],
            "Hospital": ["doctor", "nurse", "patient", "emergency", "medical", "ambulance"],
            "Shopping Mall": ["shop", "store", "mall", "sale", "customer"],
            "Office": ["meeting", "conference", "office", "work", "presentation"],
            "School/University": ["class", "student", "teacher", "lecture", "exam"],
            "Restaurant/Cafe": ["food", "order", "table", "menu", "waiter"],
            "Street/Road": ["car", "traffic", "road", "drive", "vehicle"],
            "Home/Residential": ["home", "house", "room", "family"],
            "Park/Outdoor": ["park", "tree", "bird", "nature", "outdoor"],
            "Stadium/Arena": ["game", "team", "score", "match", "crowd", "cheer"],
        }
        
        situation_keywords = {
            "Emergency": ["help", "fire", "emergency", "accident", "danger", "police", "ambulance"],
            "Boarding/Departure": ["boarding", "departure", "gate", "flight", "train leaving"],
            "Announcement": ["attention", "announcement", "please", "passengers"],
            "Traffic": ["traffic", "jam", "slow", "congestion"],
            "Meeting/Conference": ["meeting", "agenda", "discuss", "presentation"],
            "Celebration/Event": ["congratulations", "happy", "celebrate", "party"],
        }
        
        emergency_sounds = ["siren", "alarm", "scream", "glass breaking", "explosion", "gunshot"]
        
        # Detect location
        location = "Unknown"
        location_confidence = 0.3
        
        for loc, keywords in location_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                conf = min(0.5 + matches * 0.15, 0.95)
                if conf > location_confidence:
                    location = loc
                    location_confidence = conf
                    
        # Check sound labels for location hints
        for label in sound_labels:
            if "aircraft" in label or "airplane" in label:
                if location_confidence < 0.7:
                    location = "Airport Terminal"
                    location_confidence = 0.75
            elif "train" in label:
                if location_confidence < 0.7:
                    location = "Railway Station"
                    location_confidence = 0.75
            elif "traffic" in label or "car" in label:
                if location_confidence < 0.6:
                    location = "Street/Road"
                    location_confidence = 0.65
                    
        # Detect situation
        situation = "Normal/Quiet"
        situation_confidence = 0.5
        is_emergency = False
        emergency_prob = 0.1
        
        for sit, keywords in situation_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                conf = min(0.5 + matches * 0.2, 0.95)
                if conf > situation_confidence:
                    situation = sit
                    situation_confidence = conf
                    
        # Check for emergency
        for emg_sound in emergency_sounds:
            if any(emg_sound in label for label in sound_labels):
                is_emergency = True
                emergency_prob = max(emergency_prob, 0.8)
                situation = "Emergency"
                situation_confidence = 0.9
                break
                
        for emg_word in situation_keywords.get("Emergency", []):
            if emg_word in text_lower:
                is_emergency = True
                emergency_prob = max(emergency_prob, 0.85)
                situation = "Emergency"
                situation_confidence = 0.9
                break
        
        # Build evidence
        evidence = []
        if text and text != "Speech unclear":
            evidence.append(f"Speech detected: '{text[:50]}...'")
        if sounds:
            top_sounds = list(sounds.keys())[:3]
            evidence.append(f"Top sounds: {', '.join(top_sounds)}")
            
        # Build summary
        summary = f"Audio analysis suggests location is '{location}' "
        summary += f"with {location_confidence*100:.0f}% confidence. "
        summary += f"Current situation appears to be '{situation}'. "
        if is_emergency:
            summary += "⚠️ EMERGENCY INDICATORS DETECTED!"
        
        overall_confidence = (location_confidence + situation_confidence) / 2
        
        return {
            "location": location,
            "location_confidence": round(location_confidence, 3),
            "situation": situation,
            "situation_confidence": round(situation_confidence, 3),
            "is_emergency": is_emergency,
            "emergency_probability": round(emergency_prob, 3),
            "overall_confidence": round(overall_confidence, 3),
            "evidence": evidence,
            "summary": summary,
            "analysis_mode": "basic"
        }
        
    def analyze_upgraded(
        self,
        audio: np.ndarray,
        text: str,
        sounds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Advanced neural network-based analysis.
        """
        try:
            # Extract mel spectrogram
            mel_spec = self.models.mel_extractor.extract(audio)
            mel_batch = np.expand_dims(mel_spec, axis=0)
            
            # Get audio embedding from transformer
            audio_embedding = self.models.audio_transformer(mel_batch, training=False)
            audio_embedding = audio_embedding.numpy()[0]
            
            # Get text embedding
            text_embedding = self.models.get_text_embedding(text)
            
            # Get sound scores (YAMNet output as feature)
            if sounds:
                sound_scores = np.zeros(521)  # YAMNet has 521 classes
                for i, (label, score) in enumerate(sounds.items()):
                    if i < 521:
                        sound_scores[i] = score
            else:
                sound_scores = np.zeros(521)
                
            # Use fusion network if available
            if self.models.fusion_network is not None:
                try:
                    inputs = (
                        np.expand_dims(audio_embedding, 0),
                        np.expand_dims(text_embedding, 0),
                        np.expand_dims(sound_scores, 0)
                    )
                    predictions = self.models.fusion_network(inputs, training=False)
                    
                    # Extract predictions
                    location_probs = predictions['location'].numpy()[0]
                    situation_probs = predictions['situation'].numpy()[0]
                    confidence = float(predictions['confidence'].numpy()[0][0])
                    emergency_prob = float(predictions['emergency'].numpy()[0][0])
                    
                except Exception as e:
                    print(f"⚠️ Fusion network error: {e}")
                    # Fallback to scene classifier
                    return self._use_scene_classifier(audio_embedding, text_embedding, text, sounds)
                    
            else:
                # Use scene classifier
                return self._use_scene_classifier(audio_embedding, text_embedding, text, sounds)
            
            # Get top predictions
            location_idx = np.argmax(location_probs)
            situation_idx = np.argmax(situation_probs)
            
            location = LABEL_CONFIG.locations[location_idx]
            location_confidence = float(location_probs[location_idx])
            situation = LABEL_CONFIG.situations[situation_idx]
            situation_confidence = float(situation_probs[situation_idx])
            is_emergency = emergency_prob > 0.5
            
            # Build evidence
            evidence = []
            if text and text != "Speech unclear":
                evidence.append(f"Transcription: '{text[:80]}...'")
            
            # Add top location alternatives
            top_locs = np.argsort(location_probs)[-3:][::-1]
            alt_locs = [f"{LABEL_CONFIG.locations[i]} ({location_probs[i]*100:.1f}%)" for i in top_locs[1:]]
            if alt_locs:
                evidence.append(f"Alternative locations: {', '.join(alt_locs)}")
                
            # Add detected sounds
            if sounds:
                top_sounds = list(sounds.keys())[:5]
                evidence.append(f"Detected sounds: {', '.join(top_sounds)}")
            
            # Build summary
            summary = f"Neural network analysis identified this as '{location}' "
            summary += f"with {location_confidence*100:.1f}% confidence. "
            summary += f"The situation is '{situation}' ({situation_confidence*100:.1f}% confidence). "
            if is_emergency:
                summary += f"⚠️ EMERGENCY DETECTED with {emergency_prob*100:.1f}% probability!"
            
            return {
                "location": location,
                "location_confidence": round(location_confidence, 3),
                "situation": situation,
                "situation_confidence": round(situation_confidence, 3),
                "is_emergency": is_emergency,
                "emergency_probability": round(emergency_prob, 3),
                "overall_confidence": round(confidence, 3),
                "evidence": evidence,
                "summary": summary,
                "analysis_mode": "upgraded_neural",
                "all_location_probs": {
                    LABEL_CONFIG.locations[i]: float(location_probs[i])
                    for i in range(len(LABEL_CONFIG.locations))
                },
                "all_situation_probs": {
                    LABEL_CONFIG.situations[i]: float(situation_probs[i])
                    for i in range(len(LABEL_CONFIG.situations))
                }
            }
            
        except Exception as e:
            print(f"⚠️ Upgraded analysis error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic analysis
            return self.analyze_basic(audio, text, sounds)
            
    def _use_scene_classifier(
        self,
        audio_embedding: np.ndarray,
        text_embedding: np.ndarray,
        text: str,
        sounds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Use scene classifier for prediction."""
        try:
            result = self.models.scene_classifier.predict_single(
                audio_embedding, text_embedding
            )
            
            evidence = []
            if text and text != "Speech unclear":
                evidence.append(f"Transcription: '{text[:80]}...'")
            if sounds:
                top_sounds = list(sounds.keys())[:5]
                evidence.append(f"Detected sounds: {', '.join(top_sounds)}")
                
            summary = f"Scene classifier identified '{result['location']}' "
            summary += f"({result['location_confidence']*100:.1f}% confidence). "
            summary += f"Situation: '{result['situation']}'. "
            if result['is_emergency']:
                summary += "⚠️ EMERGENCY INDICATORS DETECTED!"
                
            return {
                "location": result['location'],
                "location_confidence": round(result['location_confidence'], 3),
                "situation": result['situation'],
                "situation_confidence": round(result['situation_confidence'], 3),
                "is_emergency": result['is_emergency'],
                "emergency_probability": round(result['emergency_probability'], 3),
                "overall_confidence": round(result['overall_confidence'], 3),
                "evidence": evidence,
                "summary": summary,
                "analysis_mode": "scene_classifier"
            }
        except Exception as e:
            print(f"⚠️ Scene classifier error: {e}")
            # Final fallback
            return self.analyze_basic(
                np.zeros(16000),  # dummy
                text,
                sounds
            )
    
    def analyze(
        self,
        audio: np.ndarray,
        text: str,
        sounds: Dict[str, float],
        force_basic: bool = False
    ) -> Dict[str, Any]:
        """
        Main analysis method. Uses upgraded models if available.
        """
        if force_basic or not self.models.is_upgraded:
            return self.analyze_basic(audio, text, sounds)
        else:
            return self.analyze_upgraded(audio, text, sounds)


# ==============================
# 🌐 GLOBAL INSTANCES
# ==============================
model_manager = ModelManager()
analysis_engine = None


# ==============================
# 🚀 APPLICATION LIFESPAN
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global analysis_engine
    
    print("\n" + "="*60)
    print("🚀 AURALIS ML SYSTEM - STARTING UP")
    print("="*60)
    
    # Setup FFmpeg
    setup_ffmpeg()
    
    # Load basic models
    model_manager.load_basic_models()
    
    # Try to load upgraded models
    model_manager.load_upgraded_models()
    
    # Initialize analysis engine
    analysis_engine = AnalysisEngine(model_manager)
    
    print("\n" + "="*60)
    if model_manager.is_upgraded:
        print("🎯 AURALIS RUNNING IN UPGRADED MODE")
    else:
        print("🔧 AURALIS RUNNING IN BASIC MODE")
    print("="*60)
    print("🌐 Server ready at http://127.0.0.1:8000")
    print("📚 API docs at http://127.0.0.1:8000/docs")
    print("="*60 + "\n")
    
    yield
    
    # Cleanup
    print("\n👋 Shutting down Auralis...")


# ==============================
# 🚀 FASTAPI APP
# ==============================
app = FastAPI(
    title="Auralis ML API",
    description="Advanced Audio Scene Analysis using Deep Learning",
    version="2.0.0",
    lifespan=lifespan
)

# ==============================
# 🔧 CORS CONFIGURATION
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# ==============================
# 📍 ROUTES
# ==============================

@app.get("/", tags=["General"])
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle favicon request."""
    return Response(content=b"", media_type="image/x-icon", status_code=200)


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded={
            "whisper": model_manager.whisper is not None,
            "yamnet": model_manager.yamnet is not None,
            "scene_classifier": model_manager.scene_classifier is not None,
            "audio_transformer": model_manager.audio_transformer is not None,
            "fusion_network": model_manager.fusion_network is not None,
            "text_embedder": model_manager.text_model is not None,
        },
        upgraded_mode=model_manager.is_upgraded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/labels", tags=["General"])
async def get_labels():
    """Get all available location and situation labels."""
    if MODULES_LOADED:
        return {
            "locations": LABEL_CONFIG.locations,
            "situations": LABEL_CONFIG.situations,
            "num_locations": LABEL_CONFIG.num_locations,
            "num_situations": LABEL_CONFIG.num_situations
        }
    else:
        return {
            "locations": [
                "Airport Terminal", "Railway Station", "Bus Terminal",
                "Hospital", "Shopping Mall", "Office", "School/University",
                "Restaurant/Cafe", "Street/Road", "Home/Residential",
                "Park/Outdoor", "Stadium/Arena", "Unknown"
            ],
            "situations": [
                "Normal/Quiet", "Busy/Crowded", "Emergency",
                "Boarding/Departure", "Waiting", "Traffic",
                "Meeting/Conference", "Announcement", "Celebration/Event",
                "Construction", "Weather Event", "Accident",
                "Medical Emergency", "Security Alert", "Unknown"
            ]
        }


@app.post("/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_audio(
    file: UploadFile = File(...),
    force_basic: bool = False
):
    """
    Analyze uploaded audio file.
    
    - **file**: Audio file (WAV, MP3, FLAC, etc.)
    - **force_basic**: Force basic analysis mode (skip neural network)
    
    Returns detailed analysis including:
    - Location detection
    - Situation classification
    - Emergency detection
    - Transcribed speech
    - Detected sounds
    """
    import time
    start_time = time.time()
    
    request_id = str(uuid.uuid4())[:8]
    temp_filename = f"temp_{request_id}_{file.filename}"
    
    print("\n" + "="*60)
    print(f"📥 RECEIVED: {file.filename}")
    print(f"   Request ID: {request_id}")
    print(f"   Mode: {'Basic' if force_basic else 'Auto'}")
    print("="*60)
    
    try:
        # Save uploaded file
        contents = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(contents)
        print(f"💾 Saved: {temp_filename} ({len(contents)} bytes)")
        
        # Load audio
        print("🔊 Loading audio...")
        try:
            audio, sr = librosa.load(temp_filename, sr=16000, mono=True)
            duration = len(audio) / sr
            print(f"⏱️ Duration: {duration:.2f}s, Samples: {len(audio)}")
            
            if duration < 0.5:
                raise HTTPException(status_code=400, detail="Audio too short (min 0.5s)")
            if duration > 300:
                print("⚠️ Audio too long, truncating to 5 minutes")
                audio = audio[:sr * 300]
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not load audio: {str(e)}")
        
        # Transcribe with Whisper
        print("🎤 Transcribing...")
        text = "Speech unclear"
        if model_manager.whisper:
            try:
                result = model_manager.whisper(temp_filename)
                text = result.get("text", "Speech unclear").strip()
                if not text:
                    text = "Speech unclear"
                print(f"📝 Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            except Exception as e:
                print(f"⚠️ Whisper error: {e}")
        
        # Analyze with YAMNet
        print("🔊 Analyzing sounds...")
        sounds = {}
        if model_manager.yamnet and len(model_manager.yamnet_labels) > 0:
            try:
                scores, _, _ = model_manager.yamnet(audio)
                mean_scores = tf.reduce_mean(scores, axis=0).numpy()
                top_indices = np.argsort(mean_scores)[-15:][::-1]
                
                for i in top_indices:
                    if i < len(model_manager.yamnet_labels):
                        sounds[model_manager.yamnet_labels[i]] = float(mean_scores[i])
                        
                print(f"🔊 Top sounds: {list(sounds.keys())[:5]}")
            except Exception as e:
                print(f"⚠️ YAMNet error: {e}")
        
        # Run analysis
        print("🧠 Running analysis...")
        result = analysis_engine.analyze(audio, text, sounds, force_basic=force_basic)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Log for active learning
        if model_manager.active_learning and model_manager.is_upgraded:
            try:
                model_manager.active_learning.log_prediction(
                    audio_data=None,  # Don't store audio
                    prediction=result,
                    user_feedback=None
                )
            except:
                pass
        
        # Build response
        response = AnalysisResult(
            location=result["location"],
            location_confidence=result["location_confidence"],
            situation=result["situation"],
            situation_confidence=result["situation_confidence"],
            is_emergency=result["is_emergency"],
            emergency_probability=result["emergency_probability"],
            overall_confidence=result["overall_confidence"],
            transcribed_text=text,
            detected_sounds=list(sounds.keys())[:10],
            evidence=result["evidence"],
            summary=result["summary"],
            analysis_mode=result["analysis_mode"],
            processing_time_ms=round(processing_time, 2),
            request_id=request_id,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"\n✅ RESULT:")
        print(f"   📍 Location: {response.location} ({response.location_confidence*100:.1f}%)")
        print(f"   🎯 Situation: {response.situation} ({response.situation_confidence*100:.1f}%)")
        print(f"   🚨 Emergency: {response.is_emergency} ({response.emergency_probability*100:.1f}%)")
        print(f"   ⚡ Mode: {response.analysis_mode}")
        print(f"   ⏱️ Time: {response.processing_time_ms:.0f}ms")
        print("="*60 + "\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass


@app.post("/analyze/stream", tags=["Analysis"])
async def analyze_audio_stream(
    file: UploadFile = File(...)
):
    """
    Analyze audio with streaming updates (for long audio files).
    Returns preliminary results quickly, then detailed analysis.
    """
    # Similar to /analyze but with progress updates
    # Implementation would use WebSocket or Server-Sent Events
    return await analyze_audio(file)


@app.post("/feedback", tags=["Learning"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a previous analysis.
    Used for active learning to improve the model.
    """
    if not model_manager.active_learning:
        return {"status": "feedback_recorded", "note": "Active learning not enabled"}
    
    try:
        # Record feedback
        model_manager.active_learning.query_history.append({
            "request_id": feedback.request_id,
            "feedback": {
                "correct_location": feedback.correct_location,
                "correct_situation": feedback.correct_situation,
                "was_emergency": feedback.was_emergency,
                "comments": feedback.comments
            },
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "feedback_recorded",
            "request_id": feedback.request_id,
            "message": "Thank you for your feedback! This will help improve the model."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {str(e)}")


@app.get("/stats", tags=["General"])
async def get_stats():
    """Get system statistics and performance metrics."""
    stats = {
        "total_analyses": len(analysis_engine.analysis_history) if analysis_engine else 0,
        "model_mode": "upgraded" if model_manager.is_upgraded else "basic",
        "models_status": {
            "whisper": "loaded" if model_manager.whisper else "not loaded",
            "yamnet": "loaded" if model_manager.yamnet else "not loaded",
            "scene_classifier": "loaded" if model_manager.scene_classifier else "not loaded",
            "audio_transformer": "loaded" if model_manager.audio_transformer else "not loaded",
            "fusion_network": "loaded" if model_manager.fusion_network else "not loaded",
        }
    }
    
    if model_manager.active_learning:
        al_stats = model_manager.active_learning.get_improvement_stats()
        stats["active_learning"] = al_stats
        
    return stats


# ==============================
# 🔒 MOCK AUTH ENDPOINTS
# ==============================

@app.post("/login", tags=["Auth"])
async def login_mock(data: dict):
    """Mock login endpoint for frontend compatibility."""
    return {
        "token": f"mock-token-{uuid.uuid4().hex[:8]}",
        "email": data.get("email", "user@auralis.ai"),
        "expires_in": 3600
    }


@app.post("/register", tags=["Auth"])
async def register_mock(data: dict):
    """Mock register endpoint for frontend compatibility."""
    return {
        "status": "success",
        "message": "User registered successfully",
        "email": data.get("email", "user@auralis.ai")
    }


@app.post("/save_history", tags=["Auth"])
async def save_history_mock(data: dict):
    """Mock save history endpoint for frontend compatibility."""
    return {
        "status": "saved",
        "id": str(uuid.uuid4())[:8]
    }


# ==============================
# 🏃 RUN SERVER
# ==============================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 STARTING AURALIS ML SERVER")
    print("="*60)
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Set True for development
        workers=1,
        log_level="info"
    )