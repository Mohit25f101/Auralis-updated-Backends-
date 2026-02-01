# ==============================
# 📄 services/summarizer.py
# ==============================
"""
Content Summarization Service
Generates concise summaries of speech content

PROVIDES:
- One-line summary
- Key points extraction
- TL;DR version
- Action summary
"""

import re
from typing import Dict, List, Any, Optional


class Summarizer:
    """
    Content Summarization Service
    
    Generates:
    - Brief one-line summary
    - Extended summary
    - Key points list
    - Action items summary
    """
    
    # Important keywords to include in summary
    IMPORTANT_KEYWORDS = [
        "train", "flight", "bus", "metro",
        "platform", "gate", "terminal",
        "arriving", "departing", "delayed", "cancelled",
        "emergency", "warning", "attention",
        "time", "number", "please",
    ]
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the summarizer"""
        self.loaded = True
        print("✅ Summarizer loaded")
        return True
    
    def summarize(
        self,
        text: str,
        intent: str = "unknown",
        entities: Dict = None,
        max_length: int = 100
    ) -> Dict[str, Any]:
        """
        Generate summary of text
        
        Args:
            text: Input text
            intent: Detected intent
            entities: Extracted entities
            max_length: Maximum summary length
            
        Returns:
            Summary results
        """
        if not self.loaded or not text:
            return self._empty_result()
        
        entities = entities or {}
        sentences = self._split_sentences(text)
        
        # Generate different summary types
        one_line = self._generate_one_line(text, intent, entities)
        key_points = self._extract_key_points(sentences)
        action_summary = self._generate_action_summary(text, intent)
        extended = self._generate_extended(sentences, entities)
        tldr = self._generate_tldr(text, intent, entities)
        
        return {
            "one_line_summary": one_line,
            "extended_summary": extended,
            "tldr": tldr,
            "key_points": key_points,
            "action_summary": action_summary,
            "word_count_original": len(text.split()),
            "word_count_summary": len(one_line.split()),
            "compression_ratio": round(
                len(one_line.split()) / max(len(text.split()), 1), 2
            )
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
    def _generate_one_line(
        self,
        text: str,
        intent: str,
        entities: Dict
    ) -> str:
        """Generate one-line summary"""
        # Start with intent-based prefix
        prefixes = {
            "announcement": "Announcement:",
            "instruction": "Instructions:",
            "warning": "⚠️ Warning:",
            "emergency": "🚨 EMERGENCY:",
            "question": "Question:",
            "information": "Info:",
            "greeting": "Greeting:",
        }
        prefix = prefixes.get(intent, "")
        
        # Extract key information
        parts = []
        
        # Add entity information
        if entities.get("train_numbers"):
            parts.append(f"Train {entities['train_numbers'][0]}")
        elif entities.get("flight_numbers"):
            parts.append(f"Flight {entities['flight_numbers'][0]}")
        
        if entities.get("platform_numbers"):
            parts.append(entities['platform_numbers'][0])
        elif entities.get("gate_numbers"):
            parts.append(entities['gate_numbers'][0])
        
        if entities.get("times"):
            parts.append(f"at {entities['times'][0]}")
        
        # If no entities, use first sentence
        if not parts:
            sentences = self._split_sentences(text)
            if sentences:
                # Take first sentence, truncate if needed
                first = sentences[0]
                if len(first) > 80:
                    first = first[:77] + "..."
                parts.append(first)
        
        summary = " ".join(parts)
        
        if prefix:
            summary = f"{prefix} {summary}"
        
        return summary if summary else "Summary not available"
    
    def _extract_key_points(self, sentences: List[str]) -> List[str]:
        """Extract key points from sentences"""
        key_points = []
        
        for sentence in sentences:
            # Check for importance markers
            importance_score = 0
            
            for keyword in self.IMPORTANT_KEYWORDS:
                if keyword in sentence.lower():
                    importance_score += 1
            
            # Check for numbers
            if re.search(r'\d+', sentence):
                importance_score += 1
            
            # Check for action words
            if re.search(r'\b(?:please|must|should|will|can)\b', sentence, re.IGNORECASE):
                importance_score += 1
            
            if importance_score >= 2:
                key_points.append(sentence)
        
        # If no key points found, use first and last sentences
        if not key_points and sentences:
            key_points = [sentences[0]]
            if len(sentences) > 1:
                key_points.append(sentences[-1])
        
        return key_points[:5]
    
    def _generate_action_summary(self, text: str, intent: str) -> str:
        """Generate action-focused summary"""
        actions = []
        
        # Extract action phrases
        action_patterns = [
            (r'(?:please|kindly)\s+([^.!?]+)', "{}"),
            (r'(?:you\s+)?(?:should|must|need\s+to)\s+([^.!?]+)', "{}"),
            (r'(?:proceed|go|move)\s+to\s+([^.!?]+)', "Go to {}"),
            (r'(?:wait\s+(?:at|for))\s+([^.!?]+)', "Wait {}"),
        ]
        
        for pattern, template in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:
                action = template.format(match.strip())
                if len(action) < 60:
                    actions.append(action)
        
        if actions:
            return "Actions: " + "; ".join(actions[:3])
        
        # Default based on intent
        defaults = {
            "announcement": "Listen to the announcement for details",
            "instruction": "Follow the given instructions",
            "warning": "Pay attention and take precautions",
            "emergency": "Take immediate action as directed",
            "question": "Answer or respond to the question",
        }
        
        return defaults.get(intent, "No specific action required")
    
    def _generate_extended(
        self,
        sentences: List[str],
        entities: Dict
    ) -> str:
        """Generate extended summary"""
        if not sentences:
            return "No content to summarize"
        
        # Take first 2-3 important sentences
        selected = []
        
        for sentence in sentences[:5]:
            importance = sum(1 for kw in self.IMPORTANT_KEYWORDS if kw in sentence.lower())
            if importance >= 1 or len(selected) < 2:
                selected.append(sentence)
                if len(selected) >= 3:
                    break
        
        return ". ".join(selected) + "."
    
    def _generate_tldr(
        self,
        text: str,
        intent: str,
        entities: Dict
    ) -> str:
        """Generate TL;DR version"""
        parts = []
        
        # Intent
        intent_tldr = {
            "announcement": "📢 Announcement",
            "instruction": "📋 Instructions",
            "warning": "⚠️ Warning",
            "emergency": "🚨 Emergency",
            "question": "❓ Question",
            "information": "ℹ️ Info",
            "greeting": "👋 Greeting",
        }
        parts.append(intent_tldr.get(intent, "💬 Message"))
        
        # Key entity
        if entities.get("train_numbers"):
            parts.append(f"Train {entities['train_numbers'][0]}")
        elif entities.get("flight_numbers"):
            parts.append(f"Flight {entities['flight_numbers'][0]}")
        
        if entities.get("platform_numbers"):
            parts.append(entities['platform_numbers'][0])
        
        if entities.get("times"):
            parts.append(entities['times'][0])
        
        return " | ".join(parts)
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "one_line_summary": "No content to summarize",
            "extended_summary": "",
            "tldr": "No content",
            "key_points": [],
            "action_summary": "No actions",
            "word_count_original": 0,
            "word_count_summary": 0,
            "compression_ratio": 0
        }