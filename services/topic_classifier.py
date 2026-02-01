# ==============================
# 📄 services/topic_classifier.py
# ==============================
"""
Topic Classification Service
Identifies topics and subjects discussed in speech
"""

import re
from typing import Dict, List, Any, Optional
from collections import defaultdict


class TopicClassifier:
    """
    Topic Classification Service
    
    Identifies:
    - Main topics discussed
    - Subject areas
    - Keywords by topic
    """
    
    # Topic patterns and keywords
    TOPICS = {
        "travel": {
            "keywords": ["travel", "journey", "trip", "tour", "vacation", "holiday"],
            "weight": 1.0
        },
        "transport": {
            "keywords": ["train", "flight", "bus", "metro", "car", "taxi", "uber", "vehicle"],
            "weight": 1.0
        },
        "time": {
            "keywords": ["time", "schedule", "delay", "late", "early", "arriving", "departing", "hours", "minutes"],
            "weight": 0.9
        },
        "safety": {
            "keywords": ["safety", "security", "emergency", "caution", "warning", "danger", "careful"],
            "weight": 1.1
        },
        "service": {
            "keywords": ["service", "help", "assistance", "support", "customer", "complaint"],
            "weight": 0.9
        },
        "location": {
            "keywords": ["platform", "gate", "terminal", "station", "airport", "stop", "exit", "entrance"],
            "weight": 1.0
        },
        "payment": {
            "keywords": ["ticket", "fare", "payment", "price", "cost", "fee", "charge", "rupees", "dollars"],
            "weight": 0.9
        },
        "weather": {
            "keywords": ["weather", "rain", "storm", "thunder", "temperature", "hot", "cold", "sunny"],
            "weight": 0.8
        },
        "food": {
            "keywords": ["food", "restaurant", "cafe", "meal", "drink", "coffee", "tea", "breakfast", "lunch", "dinner"],
            "weight": 0.8
        },
        "health": {
            "keywords": ["health", "medical", "doctor", "hospital", "medicine", "sick", "ill", "pain"],
            "weight": 0.9
        },
        "entertainment": {
            "keywords": ["movie", "music", "concert", "show", "performance", "event", "game", "sports"],
            "weight": 0.8
        },
        "business": {
            "keywords": ["meeting", "conference", "office", "work", "project", "deadline", "presentation"],
            "weight": 0.9
        },
        "education": {
            "keywords": ["school", "college", "university", "class", "exam", "student", "teacher", "lecture"],
            "weight": 0.8
        },
        "shopping": {
            "keywords": ["shop", "store", "buy", "purchase", "sale", "discount", "mall", "market"],
            "weight": 0.8
        },
        "accommodation": {
            "keywords": ["hotel", "room", "booking", "reservation", "check-in", "check-out", "stay"],
            "weight": 0.8
        },
        "communication": {
            "keywords": ["call", "phone", "message", "email", "contact", "number"],
            "weight": 0.7
        },
        "directions": {
            "keywords": ["direction", "way", "route", "turn", "left", "right", "straight", "near", "far"],
            "weight": 0.8
        },
        "announcements": {
            "keywords": ["attention", "announcement", "notice", "inform", "please note"],
            "weight": 0.9
        }
    }
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the classifier"""
        self.loaded = True
        print("✅ Topic Classifier loaded")
        return True
    
    def classify(
        self,
        text: str,
        location: Optional[str] = None,
        situation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify topics in text
        
        Args:
            text: Input text
            location: Detected location
            situation: Detected situation
            
        Returns:
            Topic classification results
        """
        if not self.loaded or not text:
            return self._empty_result()
        
        text_lower = text.lower()
        
        # Score topics
        topic_scores = self._score_topics(text_lower)
        
        # Add context-based topics
        if location:
            context_topics = self._topics_from_location(location)
            for topic in context_topics:
                if topic not in topic_scores:
                    topic_scores[topic] = 0.3
        
        # Get top topics
        sorted_topics = sorted(
            topic_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_topics = [t[0] for t in sorted_topics if t[1] > 0.2][:6]
        
        # Get primary topic
        primary = top_topics[0] if top_topics else "general"
        
        # Get keywords found
        keywords_found = self._get_keywords_found(text_lower, top_topics)
        
        return {
            "topics": top_topics,
            "primary_topic": primary,
            "topic_scores": {t: round(s, 3) for t, s in sorted_topics[:10] if s > 0.1},
            "keywords_found": keywords_found,
            "topic_count": len(top_topics),
            "is_multi_topic": len(top_topics) > 2
        }
    
    def _score_topics(self, text: str) -> Dict[str, float]:
        """Score topics based on keywords"""
        scores = defaultdict(float)
        
        for topic, data in self.TOPICS.items():
            weight = data.get("weight", 1.0)
            
            for keyword in data.get("keywords", []):
                if keyword in text:
                    scores[topic] += weight * 0.3
                    
                    # Bonus for multiple occurrences
                    count = text.count(keyword)
                    if count > 1:
                        scores[topic] += weight * 0.1 * min(count - 1, 3)
        
        # Normalize
        max_score = max(scores.values()) if scores else 1
        if max_score > 0:
            scores = {k: min(v / max_score, 1.0) for k, v in scores.items()}
        
        return dict(scores)
    
    def _topics_from_location(self, location: str) -> List[str]:
        """Infer topics from location"""
        location_topics = {
            "Airport Terminal": ["travel", "transport", "announcements"],
            "Railway Station": ["travel", "transport", "time"],
            "Hospital": ["health", "safety"],
            "Shopping Mall": ["shopping", "entertainment"],
            "Restaurant/Cafe": ["food"],
            "Office Building": ["business"],
            "School/University": ["education"],
            "Hotel/Lodge": ["accommodation", "travel"],
            "Stadium/Arena": ["entertainment", "sports"],
            "Street/Road": ["transport", "directions"],
        }
        
        return location_topics.get(location, [])
    
    def _get_keywords_found(
        self,
        text: str,
        topics: List[str]
    ) -> Dict[str, List[str]]:
        """Get keywords found for each topic"""
        found = {}
        
        for topic in topics:
            if topic in self.TOPICS:
                keywords = self.TOPICS[topic].get("keywords", [])
                matches = [kw for kw in keywords if kw in text]
                if matches:
                    found[topic] = matches[:5]
        
        return found
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "topics": [],
            "primary_topic": "unknown",
            "topic_scores": {},
            "keywords_found": {},
            "topic_count": 0,
            "is_multi_topic": False
        }