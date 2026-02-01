# ==============================
# 📄 services/intent_detector.py
# ==============================
"""
Speech Intent Detection Service
Determines the PURPOSE of speech

DETECTS:
- Announcements
- Instructions
- Questions
- Requests
- Warnings
- Greetings
- Emergencies
"""

import re
from typing import Dict, List, Any, Tuple
from enum import Enum
from collections import defaultdict


class Intent(Enum):
    """Speech intent types"""
    ANNOUNCEMENT = "announcement"
    INSTRUCTION = "instruction"
    INFORMATION = "information"
    QUESTION = "question"
    REQUEST = "request"
    WARNING = "warning"
    EMERGENCY = "emergency"
    GREETING = "greeting"
    FAREWELL = "farewell"
    COMPLAINT = "complaint"
    CONFIRMATION = "confirmation"
    EXPLANATION = "explanation"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


class IntentDetector:
    """
    Speech Intent Detection Service
    
    Determines WHY the speaker is speaking:
    - To announce something?
    - To give instructions?
    - To ask a question?
    - To warn about danger?
    """
    
    # Intent patterns with weights
    PATTERNS = {
        Intent.ANNOUNCEMENT: {
            "high": [
                r"attention\s+(?:please|all|passengers)",
                r"(?:ladies\s+and\s+)?gentlemen",
                r"(?:we\s+)?(?:are\s+)?(?:pleased\s+)?to\s+announce",
                r"this\s+is\s+(?:a|an)\s+(?:public\s+)?announcement",
                r"(?:may\s+i\s+have\s+)?your\s+attention\s+please",
                r"(?:passengers|customers|guests)\s+are\s+(?:informed|requested)",
            ],
            "medium": [
                r"please\s+note",
                r"kindly\s+(?:note|be\s+informed)",
                r"for\s+your\s+(?:information|attention)",
                r"we\s+(?:would\s+like\s+to\s+)?inform",
            ],
        },
        Intent.INSTRUCTION: {
            "high": [
                r"(?:please\s+)?(?:proceed|go|move|come)\s+to",
                r"you\s+(?:must|should|need\s+to|have\s+to)",
                r"(?:do\s+not|don't|never)\s+\w+",
                r"(?:make\s+sure|ensure)\s+(?:to|that)",
                r"(?:follow|take)\s+(?:the|these)\s+(?:steps|instructions)",
            ],
            "medium": [
                r"(?:please|kindly)\s+\w+",
                r"(?:keep|remain|stay)\s+\w+",
                r"step\s+(?:back|forward|aside)",
            ],
        },
        Intent.WARNING: {
            "high": [
                r"(?:warning|caution|danger|alert)",
                r"(?:be\s+)?careful",
                r"(?:emergency|urgent)\s+\w+",
                r"for\s+your\s+(?:safety|security)",
                r"(?:hazard|risk|threat)",
            ],
            "medium": [
                r"(?:do\s+not|don't|avoid)\s+\w+",
                r"(?:prohibited|forbidden|not\s+allowed)",
            ],
        },
        Intent.EMERGENCY: {
            "high": [
                r"(?:emergency|fire|help)",
                r"(?:evacuate|escape)\s+(?:now|immediately)?",
                r"(?:call|dial)\s+(?:911|100|101|102|108|112)",
                r"(?:someone\s+)?(?:needs?\s+help|hurt|injured)",
                r"(?:medical\s+)?emergency",
            ],
        },
        Intent.QUESTION: {
            "high": [
                r"\?$",
                r"^(?:what|where|when|why|how|who|which)",
                r"^(?:is|are|was|were|do|does|did|can|could|will|would)",
            ],
            "medium": [
                r"(?:can|could)\s+you\s+(?:tell|help)",
                r"(?:do\s+you\s+know|any\s+idea)",
                r"(?:i\s+)?(?:wonder|wondering)",
            ],
        },
        Intent.REQUEST: {
            "high": [
                r"(?:can|could|would)\s+you\s+(?:please)?",
                r"(?:i\s+)?(?:need|want|would\s+like)",
                r"(?:requesting|request\s+for)",
            ],
            "medium": [
                r"(?:please|kindly)\s+(?:help|assist)",
                r"(?:if\s+possible|if\s+you\s+can)",
            ],
        },
        Intent.GREETING: {
            "high": [
                r"(?:hello|hi|hey|good\s+(?:morning|afternoon|evening|day))",
                r"(?:welcome\s+to|welcome\s+aboard|welcome\s+back)",
                r"(?:greetings|namaste|namaskar)",
            ],
        },
        Intent.FAREWELL: {
            "high": [
                r"(?:goodbye|bye|farewell|see\s+you)",
                r"(?:thank\s+you|thanks)\s+(?:for|and)",
                r"(?:have\s+a\s+)?(?:good|nice|safe)\s+(?:day|trip|journey|flight)",
            ],
        },
        Intent.INFORMATION: {
            "medium": [
                r"(?:the|this)\s+(?:train|flight|bus)\s+(?:is|will)",
                r"(?:arriving|departing|scheduled)\s+(?:at|for)",
                r"(?:located|available|open)\s+(?:at|in|on)",
                r"(?:the\s+)?(?:time|temperature|weather)\s+is",
            ],
        },
    }
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the intent detector"""
        self.loaded = True
        print("✅ Intent Detector loaded")
        return True
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect speech intent
        
        Args:
            text: Input text
            
        Returns:
            Intent detection results
        """
        if not self.loaded or not text:
            return self._empty_result()
        
        text_lower = text.lower().strip()
        
        # Score each intent
        scores = defaultdict(float)
        matches = defaultdict(list)
        
        for intent, priority_patterns in self.PATTERNS.items():
            for priority, patterns in priority_patterns.items():
                weight = 2.0 if priority == "high" else 1.0
                
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        scores[intent] += weight
                        matches[intent].append(pattern)
        
        # Determine primary intent
        if not scores:
            primary_intent = Intent.UNKNOWN
            confidence = 0.4
        else:
            primary_intent = max(scores, key=scores.get)
            max_score = scores[primary_intent]
            confidence = min(0.5 + (max_score * 0.12), 0.98)
        
        # Get secondary intents
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary_intents = [
            {"intent": i.value, "score": round(s, 2)}
            for i, s in sorted_intents[1:4]
            if s > 0
        ]
        
        # Generate explanation
        explanation = self._explain_intent(primary_intent, text)
        
        return {
            "primary_intent": primary_intent.value,
            "confidence": round(confidence, 3),
            "secondary_intents": secondary_intents,
            "explanation": explanation,
            "all_scores": {i.value: round(s, 2) for i, s in scores.items()},
            "matched_patterns": {i.value: m for i, m in matches.items()},
            "is_actionable": primary_intent in [
                Intent.INSTRUCTION, Intent.WARNING, Intent.EMERGENCY, Intent.REQUEST
            ],
            "requires_response": primary_intent in [
                Intent.QUESTION, Intent.REQUEST
            ],
            "is_urgent": primary_intent in [
                Intent.EMERGENCY, Intent.WARNING
            ]
        }
    
    def _explain_intent(self, intent: Intent, text: str) -> str:
        """Generate human-readable explanation of intent"""
        explanations = {
            Intent.ANNOUNCEMENT: "The speaker is making a public announcement to inform listeners.",
            Intent.INSTRUCTION: "The speaker is giving instructions that should be followed.",
            Intent.INFORMATION: "The speaker is sharing factual information.",
            Intent.QUESTION: "The speaker is asking a question and expects an answer.",
            Intent.REQUEST: "The speaker is making a request for something.",
            Intent.WARNING: "The speaker is warning about potential danger or issue.",
            Intent.EMERGENCY: "⚠️ This is an EMERGENCY communication requiring immediate attention!",
            Intent.GREETING: "The speaker is greeting or welcoming someone.",
            Intent.FAREWELL: "The speaker is saying goodbye or thanking.",
            Intent.UNKNOWN: "The intent of this speech could not be clearly determined.",
        }
        return explanations.get(intent, "Unknown intent")
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "primary_intent": "unknown",
            "confidence": 0,
            "secondary_intents": [],
            "explanation": "No text provided for analysis",
            "all_scores": {},
            "matched_patterns": {},
            "is_actionable": False,
            "requires_response": False,
            "is_urgent": False
        }