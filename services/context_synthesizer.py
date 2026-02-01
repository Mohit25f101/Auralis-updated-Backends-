# ==============================
# 📄 services/context_synthesizer.py
# ==============================
"""
Contextual Synthesis Engine
Generates actionable improvement suggestions based on analysis
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(Enum):
    CLARITY = "clarity"
    PACE = "pace"
    EMOTION = "emotion"
    CONFIDENCE = "confidence"
    STRUCTURE = "structure"
    ENGAGEMENT = "engagement"
    CONTEXT = "context"
    LANGUAGE = "language"


@dataclass
class Suggestion:
    category: str
    priority: str
    suggestion: str
    rationale: str
    example: Optional[str] = None


class ContextSynthesizer:
    """
    Contextual Synthesis Engine
    
    Generates personalized improvement suggestions based on:
    - Detected emotions
    - Confidence scores
    - Location/situation context
    - Speech patterns
    - Asian cultural context
    """
    
    def __init__(self):
        self.loaded = False
        self._init_suggestion_templates()
        self._init_context_tips()
        self._init_asian_context()
    
    def _init_suggestion_templates(self):
        """Initialize suggestion templates"""
        self.templates = {
            # Clarity suggestions
            "unclear_speech": Suggestion(
                category=Category.CLARITY.value,
                priority=Priority.HIGH.value,
                suggestion="Speak more clearly by enunciating each word distinctly",
                rationale="Some words were difficult to transcribe accurately",
                example="Instead of rushing, pause slightly between key phrases"
            ),
            "filler_words": Suggestion(
                category=Category.CLARITY.value,
                priority=Priority.MEDIUM.value,
                suggestion="Reduce filler words like 'um', 'uh', 'like', 'you know'",
                rationale="Filler words reduce perceived confidence and clarity",
                example="Replace 'um' with a brief pause for emphasis"
            ),
            "repetition": Suggestion(
                category=Category.CLARITY.value,
                priority=Priority.LOW.value,
                suggestion="Avoid unnecessary repetition of words or phrases",
                rationale="Repetition can indicate hesitation or reduce impact",
                example="State the key point once with conviction"
            ),
            
            # Pace suggestions
            "too_fast": Suggestion(
                category=Category.PACE.value,
                priority=Priority.HIGH.value,
                suggestion="Slow down your speaking pace for better comprehension",
                rationale="Rapid speech can be difficult to follow, especially in announcements",
                example="Aim for 120-150 words per minute in formal settings"
            ),
            "too_slow": Suggestion(
                category=Category.PACE.value,
                priority=Priority.MEDIUM.value,
                suggestion="Increase your speaking pace to maintain engagement",
                rationale="Very slow speech can lose audience attention",
                example="Practice with a timer to find optimal pace"
            ),
            "monotonous": Suggestion(
                category=Category.PACE.value,
                priority=Priority.MEDIUM.value,
                suggestion="Vary your pace to emphasize important points",
                rationale="Monotonous delivery reduces engagement and retention",
                example="Slow down for key information, speed up for transitions"
            ),
            
            # Emotion suggestions
            "too_neutral": Suggestion(
                category=Category.EMOTION.value,
                priority=Priority.MEDIUM.value,
                suggestion="Add appropriate emotional expression to your delivery",
                rationale="Neutral delivery may not convey the message's importance",
                example="Match your tone to the content - urgent for alerts, warm for welcomes"
            ),
            "too_intense": Suggestion(
                category=Category.EMOTION.value,
                priority=Priority.MEDIUM.value,
                suggestion="Modulate emotional intensity for the context",
                rationale="Overly intense delivery can cause anxiety or distrust",
                example="Reserve high intensity for genuine emergencies"
            ),
            "mismatched_emotion": Suggestion(
                category=Category.EMOTION.value,
                priority=Priority.HIGH.value,
                suggestion="Align emotional tone with the message content",
                rationale="Mismatched emotions confuse the audience",
                example="Serious news requires serious tone, good news allows warmth"
            ),
            
            # Confidence suggestions
            "low_confidence": Suggestion(
                category=Category.CONFIDENCE.value,
                priority=Priority.HIGH.value,
                suggestion="Project more confidence through stronger vocal delivery",
                rationale="Uncertain delivery undermines message credibility",
                example="End statements firmly, avoid upward inflection on facts"
            ),
            "hedging_language": Suggestion(
                category=Category.CONFIDENCE.value,
                priority=Priority.MEDIUM.value,
                suggestion="Replace hedging language with definitive statements",
                rationale="Words like 'maybe', 'perhaps', 'I think' reduce authority",
                example="Instead of 'I think the train will arrive', say 'The train will arrive'"
            ),
            
            # Structure suggestions
            "poor_structure": Suggestion(
                category=Category.STRUCTURE.value,
                priority=Priority.MEDIUM.value,
                suggestion="Organize information in a clear, logical sequence",
                rationale="Well-structured messages are easier to understand and remember",
                example="Start with what, then when, then where, then any special instructions"
            ),
            "missing_context": Suggestion(
                category=Category.STRUCTURE.value,
                priority=Priority.HIGH.value,
                suggestion="Provide essential context at the beginning",
                rationale="Audience needs context to process subsequent information",
                example="Begin with 'Attention passengers...' before the details"
            ),
            
            # Engagement suggestions
            "low_engagement": Suggestion(
                category=Category.ENGAGEMENT.value,
                priority=Priority.LOW.value,
                suggestion="Use more engaging language and tone",
                rationale="Engaging delivery improves attention and compliance",
                example="Address the audience directly: 'You can now...' instead of 'Passengers may...'"
            )
        }
    
    def _init_context_tips(self):
        """Initialize context-specific tips"""
        self.context_tips = {
            "Airport Terminal": [
                "Use standard aviation phraseology for clarity",
                "Repeat flight numbers and gate numbers for accuracy",
                "Speak in both local language and English for international terminals",
                "Use calm, reassuring tone even for delays"
            ],
            "Railway Station": [
                "Clearly state train number before train name",
                "Specify platform number with 'number' prefix (Platform Number 3)",
                "Repeat key departure times",
                "Use standard Indian Railways terminology if applicable"
            ],
            "Hospital": [
                "Maintain calm, professional tone always",
                "Avoid alarming language unless necessary",
                "Be precise with department and room numbers",
                "Respect patient privacy in announcements"
            ],
            "Emergency": [
                "Lead with the action required",
                "Be direct and unambiguous",
                "Repeat essential instructions",
                "Provide clear directions and landmarks"
            ],
            "Shopping Mall": [
                "Keep announcements brief and pleasant",
                "Avoid disrupting shopping experience unnecessarily",
                "Use friendly, welcoming tone",
                "Clearly state store names and locations"
            ],
            "Office Building": [
                "Maintain professional tone",
                "Be concise - respect work time",
                "State department/floor clearly",
                "For meetings, state room and time"
            ]
        }
    
    def _init_asian_context(self):
        """Initialize Asian context patterns"""
        self.asian_patterns = {
            "india": {
                "railway_terms": ["bogey", "coach", "platform", "Rajdhani", "Shatabdi", "local"],
                "respectful_address": ["yatriyon", "passengers", "ladies and gentlemen"],
                "common_phrases": ["kripya dhyan dijiye", "attention please"],
                "suggestions": [
                    "Use 'Yatriyon kripya dhyan dijiye' for Hindi announcements",
                    "State train number in Indian Railways format",
                    "Include both Hindi and English for major stations"
                ]
            },
            "japan": {
                "announcement_style": "formal, polite, apologetic for delays",
                "suggestions": [
                    "Use formal Japanese honorifics",
                    "Apologize sincerely for any inconvenience",
                    "Provide precise timing information"
                ]
            },
            "general_asia": {
                "suggestions": [
                    "Consider multilingual announcements for diverse audiences",
                    "Respect cultural norms around directness",
                    "Use appropriate honorifics and formal language"
                ]
            }
        }
    
    def load(self) -> bool:
        """Initialize the synthesizer"""
        self.loaded = True
        print("✅ Context Synthesizer loaded")
        return True
    
    def synthesize(
        self,
        text: str,
        emotions: Optional[Dict[str, Any]],
        confidence: Optional[Dict[str, Any]],
        location: str,
        situation: str
    ) -> Dict[str, Any]:
        """
        Generate improvement suggestions based on analysis
        
        Args:
            text: Transcribed text
            emotions: Emotion analysis results
            confidence: Confidence scoring results
            location: Detected location
            situation: Detected situation
            
        Returns:
            Dictionary with suggestions, strengths, and areas for improvement
        """
        if not self.loaded:
            return self._empty_result()
        
        suggestions = []
        strengths = []
        areas_for_improvement = []
        
        # Analyze text patterns
        text_analysis = self._analyze_text(text)
        
        # Generate suggestions based on text
        suggestions.extend(self._text_based_suggestions(text_analysis))
        
        # Generate suggestions based on emotions
        if emotions:
            emotion_suggestions = self._emotion_based_suggestions(emotions)
            suggestions.extend(emotion_suggestions["suggestions"])
            strengths.extend(emotion_suggestions["strengths"])
        
        # Generate suggestions based on confidence
        if confidence:
            confidence_suggestions = self._confidence_based_suggestions(confidence)
            suggestions.extend(confidence_suggestions["suggestions"])
            if confidence.get("overall_score", 5) >= 7:
                strengths.append("Strong speaker confidence")
        
        # Get context-specific tips
        contextual_tips = self._get_contextual_tips(location, situation)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            text_analysis, emotions, confidence
        )
        
        # Identify areas for improvement
        areas_for_improvement = self._identify_improvement_areas(
            text_analysis, emotions, confidence
        )
        
        # Sort suggestions by priority
        suggestions.sort(key=lambda x: self._priority_order(x["priority"]))
        
        return {
            "suggestions": suggestions[:8],  # Top 8 suggestions
            "overall_quality_score": quality_score,
            "strengths": strengths[:5],
            "areas_for_improvement": areas_for_improvement[:5],
            "contextual_tips": contextual_tips,
            "asian_context": self._get_asian_context(location)
        }
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for patterns"""
        if not text:
            return {"has_text": False}
        
        words = text.split()
        word_count = len(words)
        
        # Detect filler words
        fillers = ["um", "uh", "like", "you know", "basically", "actually", "so", "well"]
        filler_count = sum(1 for w in words if w.lower() in fillers)
        
        # Detect hedging language
        hedges = ["maybe", "perhaps", "i think", "i guess", "kind of", "sort of", "might"]
        hedge_count = sum(1 for hedge in hedges if hedge in text.lower())
        
        # Detect repetition
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        # Estimate speaking rate (rough)
        # Assuming average audio is 10 seconds for this text length
        estimated_wpm = word_count * 6  # Very rough estimate
        
        return {
            "has_text": True,
            "word_count": word_count,
            "filler_count": filler_count,
            "filler_ratio": filler_count / max(word_count, 1),
            "hedge_count": hedge_count,
            "unique_ratio": unique_ratio,
            "estimated_wpm": estimated_wpm,
            "has_structure": self._check_structure(text)
        }
    
    def _check_structure(self, text: str) -> bool:
        """Check if text has good structure"""
        # Check for common structural elements
        structure_indicators = [
            "attention", "please", "now", "will", "at", 
            "platform", "gate", "time", "number"
        ]
        matches = sum(1 for ind in structure_indicators if ind in text.lower())
        return matches >= 3
    
    def _text_based_suggestions(self, analysis: Dict[str, Any]) -> List[Dict]:
        """Generate suggestions based on text analysis"""
        suggestions = []
        
        if not analysis.get("has_text"):
            suggestions.append({
                "category": Category.CLARITY.value,
                "priority": Priority.HIGH.value,
                "suggestion": "Speech was not clearly detected - ensure clear pronunciation",
                "rationale": "The audio did not contain recognizable speech patterns",
                "example": None
            })
            return suggestions
        
        # Filler words
        if analysis.get("filler_ratio", 0) > 0.05:
            suggestions.append(self.templates["filler_words"].__dict__)
        
        # Hedging
        if analysis.get("hedge_count", 0) > 2:
            suggestions.append(self.templates["hedging_language"].__dict__)
        
        # Repetition
        if analysis.get("unique_ratio", 1) < 0.6:
            suggestions.append(self.templates["repetition"].__dict__)
        
        # Speaking rate
        wpm = analysis.get("estimated_wpm", 150)
        if wpm > 180:
            suggestions.append(self.templates["too_fast"].__dict__)
        elif wpm < 100:
            suggestions.append(self.templates["too_slow"].__dict__)
        
        # Structure
        if not analysis.get("has_structure"):
            suggestions.append(self.templates["poor_structure"].__dict__)
        
        return suggestions
    
    def _emotion_based_suggestions(self, emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate suggestions based on emotion analysis"""
        suggestions = []
        strengths = []
        
        primary = emotions.get("primary_emotion", "neutral")
        intensity = emotions.get("intensity", 5)
        micro = emotions.get("micro_emotions", {})
        
        # Check for neutral emotion
        if primary == "neutral" and intensity < 4:
            suggestions.append(self.templates["too_neutral"].__dict__)
        
        # Check for high intensity
        if intensity > 8:
            suggestions.append(self.templates["too_intense"].__dict__)
        elif intensity >= 5 and intensity <= 7:
            strengths.append("Good emotional engagement")
        
        # Check micro-emotions
        urgency = micro.get("urgency", 0)
        hesitation = micro.get("hesitation", 0)
        confidence_level = micro.get("confidence", 0)
        
        if hesitation > 0.6:
            suggestions.append({
                "category": Category.CONFIDENCE.value,
                "priority": Priority.MEDIUM.value,
                "suggestion": "Reduce hesitation markers in speech",
                "rationale": f"High hesitation detected ({hesitation:.0%})",
                "example": "Practice the announcement to build familiarity"
            })
        
        if urgency > 0.7:
            strengths.append("Appropriate sense of urgency conveyed")
        
        if confidence_level > 0.7:
            strengths.append("Confident vocal delivery")
        
        return {"suggestions": suggestions, "strengths": strengths}
    
    def _confidence_based_suggestions(self, confidence: Dict[str, Any]) -> Dict[str, Any]:
        """Generate suggestions based on confidence scoring"""
        suggestions = []
        
        score = confidence.get("overall_score", 5)
        breakdown = confidence.get("breakdown", {})
        
        if score < 5:
            suggestions.append(self.templates["low_confidence"].__dict__)
        
        # Check specific breakdown areas
        if breakdown.get("vocal_stability", 0.5) < 0.5:
            suggestions.append({
                "category": Category.CONFIDENCE.value,
                "priority": Priority.MEDIUM.value,
                "suggestion": "Work on vocal stability and consistency",
                "rationale": "Voice shows variability that may indicate nervousness",
                "example": "Take a deep breath before speaking and maintain steady breathing"
            })
        
        if breakdown.get("pitch_variation", 0.5) < 0.3:
            suggestions.append(self.templates["monotonous"].__dict__)
        
        return {"suggestions": suggestions}
    
    def _get_contextual_tips(self, location: str, situation: str) -> List[str]:
        """Get context-specific tips"""
        tips = []
        
        # Location-based tips
        location_tips = self.context_tips.get(location, [])
        tips.extend(location_tips[:2])
        
        # Situation-based tips
        situation_tips = self.context_tips.get(situation, [])
        tips.extend(situation_tips[:2])
        
        # Default tips if none found
        if not tips:
            tips = [
                "Speak clearly and at a measured pace",
                "Ensure key information is repeated for clarity"
            ]
        
        return tips[:4]
    
    def _get_asian_context(self, location: str) -> Dict[str, Any]:
        """Get Asian context adaptations"""
        # Default to Indian context for now
        # Can be extended based on detected language or user preference
        return self.asian_patterns.get("india", self.asian_patterns.get("general_asia", {}))
    
    def _calculate_quality_score(
        self,
        text_analysis: Dict[str, Any],
        emotions: Optional[Dict[str, Any]],
        confidence: Optional[Dict[str, Any]]
    ) -> int:
        """Calculate overall delivery quality score (1-10)"""
        scores = []
        
        # Text quality score
        if text_analysis.get("has_text"):
            text_score = 5
            if text_analysis.get("filler_ratio", 0) < 0.03:
                text_score += 1
            if text_analysis.get("unique_ratio", 0) > 0.7:
                text_score += 1
            if text_analysis.get("has_structure"):
                text_score += 1
            scores.append(min(text_score, 10))
        
        # Emotion score
        if emotions:
            emotion_score = emotions.get("intensity", 5)
            # Penalize extremes
            if emotion_score > 8 or emotion_score < 3:
                emotion_score = max(emotion_score - 2, 3)
            scores.append(emotion_score)
        
        # Confidence score
        if confidence:
            scores.append(confidence.get("overall_score", 5))
        
        if not scores:
            return 5
        
        return round(sum(scores) / len(scores))
    
    def _identify_improvement_areas(
        self,
        text_analysis: Dict[str, Any],
        emotions: Optional[Dict[str, Any]],
        confidence: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify main areas for improvement"""
        areas = []
        
        if text_analysis.get("filler_ratio", 0) > 0.05:
            areas.append("Reducing filler words")
        
        if text_analysis.get("unique_ratio", 1) < 0.6:
            areas.append("Avoiding repetition")
        
        if not text_analysis.get("has_structure"):
            areas.append("Message structure and organization")
        
        if emotions:
            if emotions.get("intensity", 5) < 4:
                areas.append("Emotional engagement")
            micro = emotions.get("micro_emotions", {})
            if micro.get("hesitation", 0) > 0.5:
                areas.append("Reducing hesitation")
        
        if confidence:
            if confidence.get("overall_score", 5) < 5:
                areas.append("Building speaker confidence")
        
        return areas
    
    def _priority_order(self, priority: str) -> int:
        """Get priority order for sorting"""
        order = {
            Priority.HIGH.value: 0,
            Priority.MEDIUM.value: 1,
            Priority.LOW.value: 2
        }
        return order.get(priority, 3)
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "suggestions": [],
            "overall_quality_score": 5,
            "strengths": [],
            "areas_for_improvement": [],
            "contextual_tips": [],
            "asian_context": {}
        }