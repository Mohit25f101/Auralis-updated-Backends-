# ==============================
# 📄 services/situation_classifier.py
# ==============================
"""
Situation Classification Service
Classifies the situation/context in audio
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from enum import Enum


class Situation(Enum):
    """Situation types"""
    NORMAL = "Normal/Quiet"
    BUSY = "Busy/Crowded"
    EMERGENCY = "Emergency"
    MEDICAL_EMERGENCY = "Medical Emergency"
    BOARDING = "Boarding/Departure"
    WAITING = "Waiting"
    TRAFFIC = "Traffic"
    RUSH_HOUR = "Rush Hour"
    MEETING = "Meeting/Conference"
    ANNOUNCEMENT = "Announcement"
    CELEBRATION = "Celebration/Event"
    CONSTRUCTION = "Construction"
    WEATHER = "Weather Event"
    ACCIDENT = "Accident"
    SECURITY = "Security Alert"
    FLIGHT_DELAY = "Flight Delay"
    TRAIN_DELAY = "Train Delay"
    SPORTS = "Sports Event"
    CONCERT = "Concert/Music"
    UNKNOWN = "Unknown"


class SituationClassifier:
    """
    Situation Classification Service
    
    Classifies:
    - Current situation/context
    - Emergency status
    - Urgency level
    - Crowd density
    """
    
    # Situation keywords with weights
    SITUATION_KEYWORDS = {
        Situation.EMERGENCY: {
            "high": ["emergency", "fire", "help", "evacuate", "danger", "critical"],
            "medium": ["urgent", "immediately", "alert", "warning"],
            "weight": 1.5
        },
        Situation.MEDICAL_EMERGENCY: {
            "high": ["heart attack", "stroke", "unconscious", "bleeding", "injured", "ambulance", "cpr"],
            "medium": ["doctor", "nurse", "medical help", "first aid"],
            "weight": 1.4
        },
        Situation.BOARDING: {
            "high": ["now boarding", "final call", "proceed to gate", "proceed to platform", "boarding"],
            "medium": ["departing", "departure", "arriving", "arrival", "gate closes"],
            "weight": 1.2
        },
        Situation.FLIGHT_DELAY: {
            "high": ["flight delayed", "flight cancelled", "rescheduled flight"],
            "medium": ["delay", "postponed", "waiting for crew", "technical issue"],
            "weight": 1.1
        },
        Situation.TRAIN_DELAY: {
            "high": ["train delayed", "signal failure", "train cancelled"],
            "medium": ["running late", "behind schedule", "expected delay"],
            "weight": 1.1
        },
        Situation.TRAFFIC: {
            "high": ["heavy traffic", "traffic jam", "congestion", "gridlock"],
            "medium": ["slow moving", "stuck", "blocked"],
            "weight": 1.0
        },
        Situation.RUSH_HOUR: {
            "high": ["rush hour", "peak time", "peak hours"],
            "medium": ["busy period", "crowded", "packed"],
            "weight": 0.9
        },
        Situation.ANNOUNCEMENT: {
            "high": ["attention please", "announcement", "ladies and gentlemen"],
            "medium": ["please note", "kindly note", "inform"],
            "weight": 0.9
        },
        Situation.BUSY: {
            "high": ["very crowded", "extremely busy", "packed"],
            "medium": ["crowded", "busy", "queue", "waiting line"],
            "weight": 0.8
        },
        Situation.MEETING: {
            "high": ["meeting started", "conference call", "presentation begins"],
            "medium": ["meeting", "conference", "discussion", "agenda"],
            "weight": 0.8
        },
        Situation.CELEBRATION: {
            "high": ["celebration", "festival", "ceremony", "wedding"],
            "medium": ["party", "event", "gathering", "function"],
            "weight": 0.8
        },
        Situation.SPORTS: {
            "high": ["goal", "score", "match", "game on", "tournament"],
            "medium": ["team", "player", "stadium", "fans"],
            "weight": 0.8
        },
        Situation.CONCERT: {
            "high": ["concert", "live performance", "on stage"],
            "medium": ["music", "band", "singer", "performer"],
            "weight": 0.8
        },
        Situation.CONSTRUCTION: {
            "high": ["construction zone", "work in progress"],
            "medium": ["construction", "renovation", "repair work"],
            "weight": 0.7
        },
        Situation.WEATHER: {
            "high": ["storm warning", "heavy rain", "flood alert", "cyclone"],
            "medium": ["rain", "thunder", "weather warning", "bad weather"],
            "weight": 0.9
        },
        Situation.SECURITY: {
            "high": ["security alert", "suspicious", "lockdown", "threat"],
            "medium": ["security check", "increased security"],
            "weight": 1.3
        },
        Situation.WAITING: {
            "high": [],
            "medium": ["please wait", "waiting area", "in queue", "your turn"],
            "weight": 0.6
        }
    }
    
    # Sound-based situation hints
    SOUND_SITUATIONS = {
        "siren": [Situation.EMERGENCY, Situation.MEDICAL_EMERGENCY],
        "alarm": [Situation.EMERGENCY, Situation.SECURITY],
        "crowd": [Situation.BUSY, Situation.CELEBRATION],
        "applause": [Situation.CELEBRATION, Situation.SPORTS],
        "cheering": [Situation.SPORTS, Situation.CELEBRATION],
        "traffic": [Situation.TRAFFIC, Situation.RUSH_HOUR],
        "car horn": [Situation.TRAFFIC],
        "construction": [Situation.CONSTRUCTION],
        "power tool": [Situation.CONSTRUCTION],
        "thunder": [Situation.WEATHER],
        "rain": [Situation.WEATHER],
        "music": [Situation.CONCERT, Situation.CELEBRATION],
    }
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the classifier"""
        self.loaded = True
        print("✅ Situation Classifier loaded")
        return True
    
    def classify(
        self,
        text: str,
        sounds: Dict[str, float],
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify the situation
        
        Args:
            text: Transcribed text
            sounds: Detected sounds with confidence
            location: Detected location (optional)
            
        Returns:
            Situation classification results
        """
        if not self.loaded:
            return self._empty_result()
        
        text_lower = text.lower() if text else ""
        
        # Score situations from text
        text_scores = self._score_from_text(text_lower)
        
        # Score situations from sounds
        sound_scores = self._score_from_sounds(sounds)
        
        # Combine scores
        combined_scores = self._combine_scores(text_scores, sound_scores)
        
        # Determine primary situation
        situation, confidence = self._determine_situation(combined_scores)
        
        # Check emergency
        is_emergency, emergency_prob = self._check_emergency(
            text_lower, combined_scores, sounds
        )
        
        # Determine urgency
        urgency = self._determine_urgency(text_lower, situation, is_emergency)
        
        # Get evidence
        evidence = self._get_evidence(text_lower, sounds, situation)
        
        return {
            "situation": situation.value,
            "confidence": round(confidence, 3),
            "is_emergency": is_emergency,
            "emergency_probability": round(emergency_prob, 3),
            "urgency_level": urgency,
            "all_scores": {s.value: round(sc, 3) for s, sc in combined_scores.items() if sc > 0.1},
            "evidence": evidence,
            "sound_based": len(sound_scores) > 0,
            "text_based": len(text_scores) > 0
        }
    
    def _score_from_text(self, text: str) -> Dict[Situation, float]:
        """Score situations based on text"""
        scores = defaultdict(float)
        
        for situation, data in self.SITUATION_KEYWORDS.items():
            weight = data.get("weight", 1.0)
            
            # High confidence keywords
            for keyword in data.get("high", []):
                if keyword in text:
                    scores[situation] += weight * 1.5
            
            # Medium confidence keywords
            for keyword in data.get("medium", []):
                if keyword in text:
                    scores[situation] += weight * 0.8
        
        return dict(scores)
    
    def _score_from_sounds(self, sounds: Dict[str, float]) -> Dict[Situation, float]:
        """Score situations based on sounds"""
        scores = defaultdict(float)
        
        for sound, confidence in sounds.items():
            sound_lower = sound.lower()
            
            for keyword, situations in self.SOUND_SITUATIONS.items():
                if keyword in sound_lower:
                    for situation in situations:
                        scores[situation] = max(scores[situation], confidence)
        
        return dict(scores)
    
    def _combine_scores(
        self,
        text_scores: Dict[Situation, float],
        sound_scores: Dict[Situation, float]
    ) -> Dict[Situation, float]:
        """Combine text and sound scores"""
        combined = defaultdict(float)
        
        # Weight text higher
        text_weight = 0.7
        sound_weight = 0.3
        
        for situation, score in text_scores.items():
            combined[situation] += score * text_weight
        
        for situation, score in sound_scores.items():
            combined[situation] += score * sound_weight
        
        return dict(combined)
    
    def _determine_situation(
        self,
        scores: Dict[Situation, float]
    ) -> Tuple[Situation, float]:
        """Determine primary situation"""
        if not scores:
            return Situation.NORMAL, 0.5
        
        # Priority order for emergencies
        priority_situations = [
            Situation.EMERGENCY,
            Situation.MEDICAL_EMERGENCY,
            Situation.SECURITY,
            Situation.ACCIDENT
        ]
        
        # Check priority situations first
        for sit in priority_situations:
            if sit in scores and scores[sit] > 0.5:
                confidence = 0.7 + min(scores[sit] * 0.2, 0.28)
                return sit, confidence
        
        # Get highest scoring situation
        best = max(scores, key=scores.get)
        confidence = 0.6 + min(scores[best] * 0.25, 0.38)
        
        return best, confidence
    
    def _check_emergency(
        self,
        text: str,
        scores: Dict[Situation, float],
        sounds: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Check for emergency status"""
        emergency_keywords = [
            "emergency", "fire", "help", "evacuate", "danger",
            "accident", "injured", "attack", "critical"
        ]
        
        prob = 0.02  # Base probability
        
        # Check text
        for keyword in emergency_keywords:
            if keyword in text:
                prob += 0.15
        
        # Check situation scores
        for sit in [Situation.EMERGENCY, Situation.MEDICAL_EMERGENCY, Situation.SECURITY]:
            if sit in scores:
                prob += scores[sit] * 0.3
        
        # Check sounds
        for sound in sounds:
            if any(kw in sound.lower() for kw in ["siren", "alarm", "scream"]):
                prob += sounds[sound] * 0.2
        
        prob = min(prob, 0.98)
        is_emergency = prob > 0.5
        
        return is_emergency, prob
    
    def _determine_urgency(
        self,
        text: str,
        situation: Situation,
        is_emergency: bool
    ) -> str:
        """Determine urgency level"""
        if is_emergency:
            return "critical"
        
        if situation in [Situation.EMERGENCY, Situation.MEDICAL_EMERGENCY, Situation.SECURITY]:
            return "critical"
        
        critical_words = ["immediately", "now", "urgent", "asap", "right away"]
        high_words = ["soon", "quickly", "hurry", "fast"]
        
        for word in critical_words:
            if word in text:
                return "critical"
        
        for word in high_words:
            if word in text:
                return "high"
        
        if situation in [Situation.BOARDING, Situation.FLIGHT_DELAY, Situation.TRAIN_DELAY]:
            return "medium"
        
        return "normal"
    
    def _get_evidence(
        self,
        text: str,
        sounds: Dict[str, float],
        situation: Situation
    ) -> List[str]:
        """Get evidence for classification"""
        evidence = []
        
        # Add keyword matches
        if situation in self.SITUATION_KEYWORDS:
            keywords = (
                self.SITUATION_KEYWORDS[situation].get("high", []) +
                self.SITUATION_KEYWORDS[situation].get("medium", [])
            )
            matches = [kw for kw in keywords if kw in text]
            if matches:
                evidence.append(f"Keywords: {', '.join(matches[:3])}")
        
        # Add sound matches
        for sound, conf in list(sounds.items())[:3]:
            evidence.append(f"Sound: {sound} ({conf*100:.0f}%)")
        
        return evidence[:5]
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "situation": Situation.UNKNOWN.value,
            "confidence": 0.0,
            "is_emergency": False,
            "emergency_probability": 0.0,
            "urgency_level": "normal",
            "all_scores": {},
            "evidence": [],
            "sound_based": False,
            "text_based": False
        }