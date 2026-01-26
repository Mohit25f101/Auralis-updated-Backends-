"""
============================================
AURALIS v5.0 - Learning Service
============================================
Active learning from user feedback.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import settings
from utils.logger import logger


class LearningService:
    """
    Active learning service for pattern learning from feedback.
    
    Features:
    - Keyword-based boost learning
    - Sound association learning
    - Persistent storage
    - Auto-save on threshold
    """
    
    def __init__(self, storage_path: Path = None):
        """
        Initialize learning service.
        
        Args:
            storage_path: Path to store learned patterns
        """
        self.storage_path = storage_path or settings.learned_patterns_path
        
        # Learning data
        self.boosts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.patterns: Dict[str, Dict] = {}
        self.feedback_count = 0
        
        # Configuration
        self.save_threshold = 3
        self.max_boost = 0.35
        self.keyword_boost_increment = 0.10
        self.sound_boost_increment = 0.12
        
        self._load()
    
    def _load(self):
        """Load learned patterns from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.boosts = defaultdict(
                lambda: defaultdict(float),
                {k: defaultdict(float, v) for k, v in data.get('boosts', {}).items()}
            )
            self.patterns = data.get('patterns', {})
            self.feedback_count = data.get('feedback_count', 0)
            
            if self.feedback_count > 0:
                logger.info(f"Loaded {self.feedback_count} learned corrections")
                
        except Exception as e:
            logger.warning(f"Could not load learned patterns: {e}")
    
    def save(self):
        """Save learned patterns to storage."""
        try:
            data = {
                'boosts': {k: dict(v) for k, v in self.boosts.items()},
                'patterns': self.patterns,
                'feedback_count': self.feedback_count,
                'last_updated': datetime.now().isoformat()
            }
            
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Saved learned patterns")
            
        except Exception as e:
            logger.error(f"Could not save learned patterns: {e}")
    
    def learn(
        self,
        category: str,
        correct_value: str,
        text: str,
        sounds: List[str]
    ):
        """
        Learn from a correction.
        
        Args:
            category: Category type ('location' or 'situation')
            correct_value: The correct label
            text: Transcribed text
            sounds: Detected sounds
        """
        # Extract keywords from text
        keywords = self._extract_keywords(text)
        
        # Boost keywords
        for keyword in keywords:
            key = f"{correct_value}::{keyword}"
            self.boosts[category][key] += self.keyword_boost_increment
        
        # Boost sounds
        for sound in sounds[:5]:
            key = f"{correct_value}::sound::{sound.lower()}"
            self.boosts[category][key] += self.sound_boost_increment
        
        # Store pattern
        pattern_key = f"{category}::{correct_value}"
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = {
                'keywords': [],
                'sounds': [],
                'count': 0
            }
        
        self.patterns[pattern_key]['keywords'].extend(keywords[:10])
        self.patterns[pattern_key]['sounds'].extend(sounds[:5])
        self.patterns[pattern_key]['count'] += 1
        
        self.feedback_count += 1
        
        # Auto-save
        if self.feedback_count % self.save_threshold == 0:
            self.save()
        
        logger.info(f"Learned correction for {category}: {correct_value}")
    
    def get_boost(
        self,
        category: str,
        value: str,
        text: str,
        sounds: List[str]
    ) -> float:
        """
        Get learned boost for a prediction.
        
        Args:
            category: Category type
            value: Predicted value
            text: Input text
            sounds: Detected sounds
            
        Returns:
            Boost value (0.0 to max_boost)
        """
        boost = 0.0
        
        # Check keywords
        keywords = self._extract_keywords(text)
        for keyword in keywords:
            key = f"{value}::{keyword}"
            boost += self.boosts[category].get(key, 0)
        
        # Check sounds
        for sound in sounds:
            key = f"{value}::sound::{sound.lower()}"
            boost += self.boosts[category].get(key, 0)
        
        return min(boost, self.max_boost)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return []
        
        # Find words with 4+ characters
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        keywords = []
        for word in words:
            if word not in seen:
                seen.add(word)
                keywords.append(word)
        
        return keywords
    
    def record_feedback(
        self,
        request_id: str,
        original_result: Dict[str, Any],
        correct_location: Optional[str],
        correct_situation: Optional[str],
        transcribed_text: str,
        detected_sounds: List[str]
    ) -> bool:
        """
        Record user feedback and learn from it.
        
        Args:
            request_id: Request identifier
            original_result: Original prediction
            correct_location: Correct location (if different)
            correct_situation: Correct situation (if different)
            transcribed_text: Text that was transcribed
            detected_sounds: Sounds that were detected
            
        Returns:
            True if learning occurred
        """
        learned = False
        
        if correct_location and correct_location != original_result.get('location'):
            self.learn('location', correct_location, transcribed_text, detected_sounds)
            learned = True
        
        if correct_situation and correct_situation != original_result.get('situation'):
            self.learn('situation', correct_situation, transcribed_text, detected_sounds)
            learned = True
        
        return learned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'feedback_count': self.feedback_count,
            'pattern_count': len(self.patterns),
            'location_boosts': len(self.boosts.get('location', {})),
            'situation_boosts': len(self.boosts.get('situation', {})),
        }
    
    def reset(self):
        """Reset all learned patterns."""
        self.boosts = defaultdict(lambda: defaultdict(float))
        self.patterns = {}
        self.feedback_count = 0
        self.save()
        logger.info("Reset all learned patterns")