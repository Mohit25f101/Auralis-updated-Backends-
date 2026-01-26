"""
============================================
AURALIS v5.0 - YAMNet Engine
============================================
Sound classification with location/situation hints.
"""

import csv
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import requests

from config import settings
from utils.logger import logger
from utils.constants import (
    SOUND_CATEGORIES, SOUND_LOCATION_MAP, SOUND_SITUATION_MAP
)


class YAMNetEngine:
    """
    YAMNet-based sound classification engine.
    
    Features:
    - 521 sound class detection
    - Sound filtering (removes generic sounds)
    - Location hints from sounds
    - Situation hints from sounds
    """
    
    # Sounds to always filter out
    FILTER_OUT = {
        "silence", "white noise", "pink noise", "static", "noise",
    }
    
    # Generic sounds (lower priority)
    GENERIC_SOUNDS = {
        "speech", "narration", "monologue", "conversation",
        "male speech, man speaking", "female speech, woman speaking",
        "child speech, kid speaking", "inside, small room",
        "inside, large room or hall", "outside, urban or manmade",
        "outside, rural or natural",
    }
    
    def __init__(self, model_url: str = None):
        """
        Initialize YAMNet engine.
        
        Args:
            model_url: YAMNet model URL (default from settings)
        """
        self.model_url = model_url or settings.yamnet_model
        self.model = None
        self.labels: List[str] = []
        self.loaded = False
    
    def load(self) -> bool:
        """
        Load YAMNet model and labels.
        
        Returns:
            True if loaded successfully
        """
        if self.loaded:
            return True
        
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            
            logger.info("Loading YAMNet model...")
            
            self.model = hub.load(self.model_url)
            self._load_labels()
            
            self.loaded = True
            logger.info(f"YAMNet loaded with {len(self.labels)} classes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YAMNet: {e}")
            return False
    
    def _load_labels(self):
        """Load YAMNet class labels."""
        labels_url = (
            "https://raw.githubusercontent.com/tensorflow/models/"
            "master/research/audioset/yamnet/yamnet_class_map.csv"
        )
        
        try:
            response = requests.get(labels_url, timeout=10)
            reader = csv.reader(response.text.splitlines())
            next(reader)  # Skip header
            self.labels = [row[2] for row in reader]
            logger.debug(f"Loaded {len(self.labels)} YAMNet labels")
        except Exception as e:
            logger.warning(f"Could not load labels online: {e}")
            self.labels = [f"Sound_{i}" for i in range(521)]
    
    def analyze(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio for sounds.
        
        Args:
            audio: Audio array (float32, 16kHz)
            
        Returns:
            Dictionary with sound analysis results
        """
        if not self.loaded:
            return self._empty_result()
        
        try:
            import tensorflow as tf
            
            # Run YAMNet
            scores, embeddings, spectrogram = self.model(audio.astype(np.float32))
            
            # Average scores across time
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
            
            # Get all sounds above threshold
            all_sounds = {}
            for idx in range(len(mean_scores)):
                if mean_scores[idx] > 0.01 and idx < len(self.labels):
                    all_sounds[self.labels[idx]] = float(mean_scores[idx])
            
            # Filter sounds
            filtered_sounds = self._filter_sounds(all_sounds)
            
            # Get hints
            location_hints = self._get_location_hints(all_sounds)
            situation_hints = self._get_situation_hints(all_sounds)
            
            # Categorize sounds
            categories = self._categorize_sounds(all_sounds)
            
            return {
                "sounds": filtered_sounds,
                "all_sounds": all_sounds,
                "categories": categories,
                "location_hints": location_hints,
                "situation_hints": situation_hints,
                "top_category": max(categories, key=categories.get) if categories else None,
                "embeddings": embeddings.numpy() if embeddings is not None else None,
            }
            
        except Exception as e:
            logger.error(f"YAMNet analysis error: {e}")
            return self._empty_result()
    
    def _filter_sounds(self, sounds: Dict[str, float]) -> Dict[str, float]:
        """Filter sounds to show most informative ones."""
        specific = {}
        generic = {}
        
        for sound, score in sounds.items():
            sound_lower = sound.lower()
            
            # Skip filtered sounds
            if any(f in sound_lower for f in self.FILTER_OUT):
                continue
            
            # Separate generic sounds
            if any(g.lower() in sound_lower for g in self.GENERIC_SOUNDS):
                if score > 0.05:
                    generic[sound] = round(score, 4)
            else:
                if score > 0.02:
                    specific[sound] = round(score, 4)
        
        # Sort by score
        specific = dict(sorted(specific.items(), key=lambda x: x[1], reverse=True)[:12])
        
        # If no specific sounds, use generic
        if not specific and generic:
            specific = dict(sorted(generic.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return specific
    
    def _get_location_hints(self, sounds: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get location hints from detected sounds."""
        hints = defaultdict(float)
        
        for sound, score in sounds.items():
            sound_lower = sound.lower()
            for keyword, locations in SOUND_LOCATION_MAP.items():
                if keyword in sound_lower:
                    for loc in locations:
                        hints[loc] = max(hints[loc], score)
        
        return sorted(hints.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_situation_hints(self, sounds: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get situation hints from detected sounds."""
        hints = defaultdict(float)
        
        for sound, score in sounds.items():
            sound_lower = sound.lower()
            for keyword, situations in SOUND_SITUATION_MAP.items():
                if keyword in sound_lower:
                    for sit in situations:
                        hints[sit] = max(hints[sit], score)
        
        return sorted(hints.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _categorize_sounds(self, sounds: Dict[str, float]) -> Dict[str, float]:
        """Categorize sounds into meaningful groups."""
        categories = {}
        
        for category, keywords in SOUND_CATEGORIES.items():
            max_score = 0.0
            for sound, score in sounds.items():
                sound_lower = sound.lower()
                if any(kw in sound_lower for kw in keywords):
                    max_score = max(max_score, score)
            
            if max_score > 0.02:
                categories[category] = round(max_score, 4)
        
        return categories
    
    def _empty_result(self) -> Dict[str, Any]:
        """Create empty result."""
        return {
            "sounds": {},
            "all_sounds": {},
            "categories": {},
            "location_hints": [],
            "situation_hints": [],
            "top_category": None,
            "embeddings": None,
        }
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.loaded