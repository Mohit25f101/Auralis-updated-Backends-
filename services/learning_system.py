# ==============================
# 📄 services/learning_system.py
# ==============================
"""
Adaptive Learning System
Learns from user corrections to improve accuracy
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class LearningSystem:
    """
    Adaptive Learning System
    
    Features:
    - Learns from user corrections
    - Keyword-based pattern boosting
    - Sound-based pattern boosting
    - Persistent storage
    - Statistics tracking
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize learning system
        
        Args:
            storage_path: Path to store learned data
        """
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "data",
                "learned_data.json"
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Learning data
        self.boosts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.corrections = 0
        self.location_corrections = 0
        self.situation_corrections = 0
        self.emotion_corrections = 0
        self.last_saved: Optional[str] = None
        
        # Load existing data
        self._load()
        
        self.loaded = True
    
    def _load(self):
        """Load learned data from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore boosts
                self.boosts = defaultdict(lambda: defaultdict(float))
                for category, patterns in data.get('boosts', {}).items():
                    self.boosts[category] = defaultdict(float, patterns)
                
                # Restore counters
                self.corrections = data.get('corrections', 0)
                self.location_corrections = data.get('location_corrections', 0)
                self.situation_corrections = data.get('situation_corrections', 0)
                self.emotion_corrections = data.get('emotion_corrections', 0)
                self.last_saved = data.get('last_saved')
                
                if self.corrections > 0:
                    print(f"📚 Loaded {self.corrections} learned corrections")
                    
        except Exception as e:
            print(f"   ⚠️ Learning data load error: {e}")
    
    def save(self):
        """Save learned data to storage"""
        try:
            data = {
                'boosts': {
                    category: dict(patterns)
                    for category, patterns in self.boosts.items()
                },
                'corrections': self.corrections,
                'location_corrections': self.location_corrections,
                'situation_corrections': self.situation_corrections,
                'emotion_corrections': self.emotion_corrections,
                'last_saved': datetime.now().isoformat(),
                'version': '5.0.0'
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.last_saved = data['last_saved']
            
        except Exception as e:
            print(f"   ⚠️ Learning data save error: {e}")
    
    def learn(
        self,
        category: str,
        correct_value: str,
        text: str,
        sounds: List[str]
    ):
        """
        Learn from a correction
        
        Args:
            category: Type of correction ('location', 'situation', 'emotion')
            correct_value: The correct value
            text: Transcribed text that was analyzed
            sounds: List of detected sounds
        """
        # Extract keywords from text
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        
        # Learn word patterns
        for word in words:
            key = f"{correct_value}::word::{word}"
            self.boosts[category][key] += 0.08
        
        # Learn sound patterns
        for sound in sounds[:8]:
            key = f"{correct_value}::sound::{sound.lower()}"
            self.boosts[category][key] += 0.10
        
        # Learn bigrams
        words_list = text.lower().split()
        for i in range(len(words_list) - 1):
            bigram = f"{words_list[i]}_{words_list[i+1]}"
            if len(bigram) > 8:
                key = f"{correct_value}::bigram::{bigram}"
                self.boosts[category][key] += 0.12
        
        # Update counters
        self.corrections += 1
        if category == 'location':
            self.location_corrections += 1
        elif category == 'situation':
            self.situation_corrections += 1
        elif category == 'emotion':
            self.emotion_corrections += 1
        
        # Auto-save periodically
        if self.corrections % 5 == 0:
            self.save()
    
    def get_boost(
        self,
        category: str,
        value: str,
        text: str,
        sounds: List[str]
    ) -> float:
        """
        Get learned boost for a value
        
        Args:
            category: Category to check
            value: Value to get boost for
            text: Current text
            sounds: Current sounds
            
        Returns:
            Boost value (0.0 to max_boost)
        """
        boost = 0.0
        max_boost = 0.35
        
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        
        # Check word patterns
        for word in words:
            key = f"{value}::word::{word}"
            boost += self.boosts[category].get(key, 0)
        
        # Check sound patterns
        for sound in sounds:
            key = f"{value}::sound::{sound.lower()}"
            boost += self.boosts[category].get(key, 0)
        
        # Check bigram patterns
        words_list = text.lower().split()
        for i in range(len(words_list) - 1):
            bigram = f"{words_list[i]}_{words_list[i+1]}"
            key = f"{value}::bigram::{bigram}"
            boost += self.boosts[category].get(key, 0)
        
        return min(boost, max_boost)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        # Count patterns
        pattern_counts = {}
        for category, patterns in self.boosts.items():
            pattern_counts[category] = len(patterns)
        
        # Get top patterns
        top_patterns = []
        for category, patterns in self.boosts.items():
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for pattern, score in sorted_patterns:
                top_patterns.append({
                    "category": category,
                    "pattern": pattern,
                    "score": round(score, 3)
                })
        
        # Sort top patterns
        top_patterns.sort(key=lambda x: x["score"], reverse=True)
        
        # Get file size
        file_size_kb = 0
        if os.path.exists(self.storage_path):
            file_size_kb = os.path.getsize(self.storage_path) / 1024
        
        return {
            "total_corrections": self.corrections,
            "location_corrections": self.location_corrections,
            "situation_corrections": self.situation_corrections,
            "emotion_corrections": self.emotion_corrections,
            "pattern_counts": pattern_counts,
            "top_patterns": top_patterns[:10],
            "last_saved": self.last_saved or "never",
            "file_size_kb": round(file_size_kb, 2)
        }
    
    def reset(self):
        """Reset all learned data"""
        self.boosts = defaultdict(lambda: defaultdict(float))
        self.corrections = 0
        self.location_corrections = 0
        self.situation_corrections = 0
        self.emotion_corrections = 0
        
        # Delete storage file
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        
        print("🗑️ Learning data reset")
    
    def export_patterns(self) -> Dict[str, Any]:
        """Export all learned patterns"""
        return {
            category: dict(patterns)
            for category, patterns in self.boosts.items()
        }
    
    def import_patterns(self, patterns: Dict[str, Dict[str, float]]):
        """Import patterns from export"""
        for category, cat_patterns in patterns.items():
            for pattern, score in cat_patterns.items():
                self.boosts[category][pattern] = score
        
        self.save()