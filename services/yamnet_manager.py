# ==============================
# services/yamnet_manager.py - FIXED
# ==============================
"""
YAMNet Sound Classification Manager
Detects 521 audio event classes
"""

import csv
import requests
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np

# Attempt imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

try:
    import tensorflow_hub as hub
    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False
    hub = None


class YAMNetManager:
    """
    YAMNet Sound Classification Manager
    
    Features:
    - 521 audio event classes
    - Sound-to-location mapping
    - Sound-to-situation mapping
    - Intelligent filtering
    """
    
    # Sounds to filter out
    FILTER_OUT = {
        "silence", "white noise", "pink noise", "static",
        "noise", "hum", "buzz"
    }
    
    # Generic sounds (lower priority)
    GENERIC_SOUNDS = {
        "speech", "narration", "monologue", "conversation",
        "male speech", "female speech", "child speech",
        "inside", "outside", "room", "small room", "large room"
    }
    
    # Sound to location mappings
    SOUND_LOCATIONS = {
        # Aviation
        "aircraft": ["Airport Terminal"],
        "aircraft engine": ["Airport Terminal"],
        "airplane": ["Airport Terminal"],
        "jet engine": ["Airport Terminal"],
        "helicopter": ["Airport Terminal"],
        "propeller": ["Airport Terminal"],
        
        # Railways
        "train": ["Railway Station"],
        "railroad car": ["Railway Station"],
        "rail transport": ["Railway Station"],
        "train horn": ["Railway Station"],
        "train whistle": ["Railway Station"],
        "subway": ["Metro/Subway"],
        "metro": ["Metro/Subway"],
        
        # Vehicles
        "car": ["Street/Road", "Parking Area"],
        "vehicle": ["Street/Road"],
        "traffic": ["Street/Road"],
        "car horn": ["Street/Road"],
        "engine": ["Street/Road", "Parking Area"],
        "bus": ["Bus Terminal", "Street/Road"],
        "motorcycle": ["Street/Road"],
        "truck": ["Street/Road"],
        
        # Crowds & Public
        "crowd": ["Airport Terminal", "Railway Station", "Shopping Mall", "Stadium/Arena"],
        "chatter": ["Restaurant/Cafe", "Shopping Mall"],
        "applause": ["Stadium/Arena", "Cinema/Theater"],
        "cheering": ["Stadium/Arena"],
        
        # Emergency
        "siren": ["Hospital", "Street/Road"],
        "ambulance": ["Hospital"],
        "fire engine": ["Street/Road"],
        "police car": ["Street/Road"],
        "alarm": ["Office Building", "Shopping Mall"],
        
        # Nature & Outdoor
        "bird": ["Park/Outdoor"],
        "bird vocalization": ["Park/Outdoor"],
        "water": ["Park/Outdoor"],
        "rain": ["Park/Outdoor", "Street/Road"],
        "thunder": ["Park/Outdoor"],
        "wind": ["Park/Outdoor"],
        
        # Indoor
        "typing": ["Office Building"],
        "keyboard": ["Office Building"],
        "telephone": ["Office Building"],
        "door": ["Office Building", "Home/Residential"],
        "doorbell": ["Home/Residential"],
        
        # Religious
        "bell": ["Religious Place"],
        "church bell": ["Religious Place"],
        "singing bowl": ["Religious Place"],
        "chant": ["Religious Place"],
        
        # Construction
        "power tool": ["Construction Site"],
        "hammer": ["Construction Site"],
        "drill": ["Construction Site"],
        "sawing": ["Construction Site"],
        "jackhammer": ["Construction Site"],
        
        # Sports
        "whistle": ["Stadium/Arena", "Gym/Sports Center"],
        "bouncing": ["Gym/Sports Center"],
        
        # Dining
        "dishes": ["Restaurant/Cafe"],
        "cutlery": ["Restaurant/Cafe"],
        "glass": ["Restaurant/Cafe"],
    }
    
    # Sound to situation mappings
    SOUND_SITUATIONS = {
        # Emergency
        "siren": ["Emergency"],
        "alarm": ["Emergency", "Security Alert"],
        "scream": ["Emergency"],
        "gunshot": ["Emergency"],
        "explosion": ["Emergency"],
        
        # Medical
        "ambulance": ["Medical Emergency"],
        
        # Busy/Crowded
        "crowd": ["Busy/Crowded"],
        "chatter": ["Busy/Crowded"],
        
        # Celebrations
        "applause": ["Celebration/Event"],
        "cheering": ["Celebration/Event", "Sports Event"],
        "fireworks": ["Celebration/Event"],
        "music": ["Celebration/Event", "Concert/Music"],
        
        # Traffic
        "traffic": ["Traffic", "Rush Hour"],
        "car horn": ["Traffic"],
        "engine idling": ["Traffic"],
        
        # Construction
        "power tool": ["Construction"],
        "hammer": ["Construction"],
        "drill": ["Construction"],
        
        # Weather
        "thunder": ["Weather Event"],
        "rain": ["Weather Event"],
        "wind": ["Weather Event"],
        
        # Transport
        "train horn": ["Boarding/Departure"],
        "train whistle": ["Boarding/Departure"],
        "aircraft": ["Boarding/Departure"],
    }
    
    YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
    LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    
    def __init__(self):
        """Initialize YAMNet manager"""
        self.model = None
        self.labels = []
        self.loaded = False
    
    def load(self) -> bool:
        """Load YAMNet model"""
        if self.loaded:
            return True
        
        if not TF_AVAILABLE or not HUB_AVAILABLE:
            print("❌ YAMNet: TensorFlow/Hub not available")
            return False
        
        try:
            print("🔄 Loading YAMNet...")
            
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')
            
            # Load model
            self.model = hub.load(self.YAMNET_URL)
            
            # Load labels
            self._load_labels()
            
            self.loaded = True
            print(f"✅ YAMNet loaded ({len(self.labels)} classes)")
            return True
            
        except Exception as e:
            print(f"❌ YAMNet loading failed: {e}")
            return False
    
    def _load_labels(self):
        """Load class labels"""
        try:
            response = requests.get(self.LABELS_URL, timeout=10)
            response.raise_for_status()
            
            reader = csv.reader(response.text.splitlines())
            next(reader)  # Skip header
            
            self.labels = [row[2] for row in reader if len(row) > 2]
            
        except Exception as e:
            print(f"   ⚠️ Failed to load labels: {e}")
            self.labels = [f"Sound_{i}" for i in range(521)]
    
    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio for sound events
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Dictionary with detected sounds and hints
        """
        if not self.loaded:
            return self._empty_result()
        
        try:
            # Ensure float32
            audio_tensor = audio.astype(np.float32)
            
            # Run inference
            scores, embeddings, spectrogram = self.model(audio_tensor)
            
            # Average scores across time
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
            
            # Get all detected sounds
            all_sounds = {}
            for i, score in enumerate(mean_scores):
                if score > 0.01 and i < len(self.labels):
                    all_sounds[self.labels[i]] = float(score)
            
            # Filter sounds
            filtered_sounds = self._filter_sounds(all_sounds)
            
            # Get location hints
            location_hints = self._get_location_hints(all_sounds)
            
            # Get situation hints
            situation_hints = self._get_situation_hints(all_sounds)
            
            # Get top sounds for summary
            top_sounds = sorted(
                filtered_sounds.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                "sounds": dict(top_sounds),
                "all_sounds": all_sounds,
                "location_hints": location_hints,
                "situation_hints": situation_hints,
                "sound_count": len(filtered_sounds),
                "embeddings": embeddings.numpy() if embeddings is not None else None
            }
            
        except Exception as e:
            print(f"   ⚠️ YAMNet analysis error: {e}")
            return self._empty_result()
    
    def _filter_sounds(self, sounds: Dict[str, float]) -> Dict[str, float]:
        """Filter and prioritize sounds"""
        result = {}
        generic = {}
        
        for sound, score in sounds.items():
            sound_lower = sound.lower()
            
            # Skip filtered sounds
            if any(f in sound_lower for f in self.FILTER_OUT):
                continue
            
            # Separate generic sounds
            if any(g in sound_lower for g in self.GENERIC_SOUNDS):
                if score > 0.05:
                    generic[sound] = score
            elif score > 0.02:
                result[sound] = round(score, 4)
        
        # Sort by score
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:15])
        
        # If no specific sounds, use generic
        if not result and generic:
            result = dict(sorted(generic.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return result
    
    def _get_location_hints(self, sounds: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get location hints from sounds"""
        hints = defaultdict(float)
        
        for sound, score in sounds.items():
            sound_lower = sound.lower()
            
            for keyword, locations in self.SOUND_LOCATIONS.items():
                if keyword in sound_lower:
                    for location in locations:
                        hints[location] = max(hints[location], score)
        
        # Sort by score
        sorted_hints = sorted(hints.items(), key=lambda x: x[1], reverse=True)
        return sorted_hints[:5]
    
    def _get_situation_hints(self, sounds: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get situation hints from sounds"""
        hints = defaultdict(float)
        
        for sound, score in sounds.items():
            sound_lower = sound.lower()
            
            for keyword, situations in self.SOUND_SITUATIONS.items():
                if keyword in sound_lower:
                    for situation in situations:
                        hints[situation] = max(hints[situation], score)
        
        # Sort by score
        sorted_hints = sorted(hints.items(), key=lambda x: x[1], reverse=True)
        return sorted_hints[:5]
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "sounds": {},
            "all_sounds": {},
            "location_hints": [],
            "situation_hints": [],
            "sound_count": 0,
            "embeddings": None
        }
    
    def get_label(self, index: int) -> str:
        """Get label for class index"""
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return f"Unknown_{index}"
    
    def get_all_labels(self) -> List[str]:
        """Get all class labels"""
        return self.labels.copy()