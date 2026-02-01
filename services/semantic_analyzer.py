# ==============================
# 📄 services/semantic_analyzer.py
# ==============================
"""
Semantic Intelligence Engine
Understands WHAT speakers mean, not just what they say

PROVIDES:
- Intent/Purpose detection
- Meaning extraction
- Key information identification
- Context understanding
- Semantic summary
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum


class SpeechIntent(Enum):
    """Types of speech intent"""
    ANNOUNCEMENT = "announcement"           # Public announcements
    INSTRUCTION = "instruction"             # Telling someone to do something
    INFORMATION = "information"             # Sharing information
    QUESTION = "question"                   # Asking something
    REQUEST = "request"                     # Asking for something
    WARNING = "warning"                     # Warning about danger
    GREETING = "greeting"                   # Hello, welcome, etc.
    FAREWELL = "farewell"                   # Goodbye, thank you, etc.
    COMPLAINT = "complaint"                 # Expressing dissatisfaction
    CONFIRMATION = "confirmation"           # Confirming something
    DENIAL = "denial"                       # Denying something
    EXPLANATION = "explanation"             # Explaining something
    NARRATION = "narration"                 # Telling a story
    CONVERSATION = "conversation"           # General conversation
    EMERGENCY = "emergency"                 # Emergency communication
    ADVERTISEMENT = "advertisement"         # Promotional content
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Types of content"""
    TRAVEL_ANNOUNCEMENT = "travel_announcement"
    SAFETY_INSTRUCTION = "safety_instruction"
    GENERAL_INFORMATION = "general_information"
    COMMERCIAL_MESSAGE = "commercial_message"
    PERSONAL_CONVERSATION = "personal_conversation"
    OFFICIAL_STATEMENT = "official_statement"
    NEWS_UPDATE = "news_update"
    ENTERTAINMENT = "entertainment"
    EDUCATIONAL = "educational"
    EMERGENCY_ALERT = "emergency_alert"
    OTHER = "other"


@dataclass
class SemanticResult:
    """Semantic analysis result"""
    intent: str
    intent_confidence: float
    meaning_summary: str
    detailed_meaning: str
    key_points: List[str]
    entities: Dict[str, List[str]]
    topics: List[str]
    action_items: List[str]
    questions: List[str]
    content_type: str
    sentiment: str
    urgency_level: str
    target_audience: str
    context_clues: List[str]


class SemanticAnalyzer:
    """
    Semantic Intelligence Engine
    
    Analyzes speech to understand:
    1. WHAT is being said (transcription) ✓ Already done by Whisper
    2. WHAT it MEANS (semantic meaning)
    3. WHY it's being said (intent/purpose)
    4. WHO it's for (target audience)
    5. WHAT to do about it (action items)
    6. KEY INFORMATION (entities, numbers, names)
    """
    
    # Intent patterns
    INTENT_PATTERNS = {
        SpeechIntent.ANNOUNCEMENT: {
            "patterns": [
                r"attention\s+(?:please|all)",
                r"(?:ladies\s+and\s+)?gentlemen",
                r"(?:we\s+)?(?:would\s+like\s+to\s+)?(?:announce|inform)",
                r"(?:this\s+is\s+)?(?:a|an)\s+(?:public\s+)?announcement",
                r"(?:may\s+i\s+have\s+)?your\s+attention",
                r"passengers\s+are\s+(?:requested|informed)",
                r"kindly\s+note",
                r"please\s+be\s+(?:informed|advised)",
            ],
            "weight": 1.0
        },
        SpeechIntent.INSTRUCTION: {
            "patterns": [
                r"please\s+(?:proceed|go|come|move|stand|wait|board)",
                r"(?:you\s+)?(?:must|should|need\s+to|have\s+to)",
                r"(?:do\s+not|don't)\s+\w+",
                r"(?:make\s+sure|ensure)\s+(?:to|that)",
                r"(?:keep|remain|stay)\s+\w+",
                r"(?:follow|take|use)\s+the",
                r"step\s+(?:back|forward|aside)",
            ],
            "weight": 0.9
        },
        SpeechIntent.WARNING: {
            "patterns": [
                r"(?:warning|caution|danger|alert)",
                r"(?:be\s+)?careful",
                r"(?:do\s+not|don't|never)\s+\w+",
                r"(?:emergency|urgent|immediately)",
                r"(?:evacuate|leave|exit)\s+(?:now|immediately)?",
                r"(?:risk|hazard|threat)",
                r"for\s+your\s+(?:safety|security)",
            ],
            "weight": 1.2
        },
        SpeechIntent.QUESTION: {
            "patterns": [
                r"\?$",
                r"^(?:what|where|when|why|how|who|which|whose)",
                r"^(?:is|are|was|were|do|does|did|can|could|will|would|should)",
                r"(?:can|could)\s+you\s+(?:tell|help|show)",
                r"(?:do\s+you\s+know|any\s+idea)",
            ],
            "weight": 0.8
        },
        SpeechIntent.REQUEST: {
            "patterns": [
                r"(?:please|kindly)\s+\w+",
                r"(?:can|could|would)\s+you\s+(?:please)?",
                r"(?:i\s+)?(?:need|want|would\s+like)",
                r"(?:requesting|request\s+for)",
                r"(?:help\s+me|assist\s+me)",
            ],
            "weight": 0.8
        },
        SpeechIntent.INFORMATION: {
            "patterns": [
                r"(?:the|this)\s+(?:train|flight|bus|metro)\s+(?:number|is)",
                r"(?:arriving|departing)\s+(?:at|from)",
                r"(?:platform|gate|terminal)\s+(?:number)?\s*\d+",
                r"(?:scheduled|expected)\s+(?:at|for)",
                r"(?:the\s+)?(?:time|temperature|weather)\s+is",
                r"(?:located|available)\s+(?:at|in|on)",
            ],
            "weight": 0.7
        },
        SpeechIntent.GREETING: {
            "patterns": [
                r"(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))",
                r"(?:welcome\s+to|welcome\s+aboard)",
                r"(?:greetings|namaste|namaskar)",
                r"(?:how\s+are\s+you|nice\s+to\s+meet)",
            ],
            "weight": 0.6
        },
        SpeechIntent.FAREWELL: {
            "patterns": [
                r"(?:goodbye|bye|farewell|see\s+you)",
                r"(?:thank\s+you|thanks)\s+(?:for|and)",
                r"(?:have\s+a\s+)?(?:good|nice|safe)\s+(?:day|trip|journey)",
                r"(?:take\s+care|until\s+next\s+time)",
            ],
            "weight": 0.6
        },
        SpeechIntent.EMERGENCY: {
            "patterns": [
                r"(?:emergency|help|fire|accident)",
                r"(?:call|dial)\s+(?:911|100|101|102|108)",
                r"(?:evacuate|escape|run)",
                r"(?:medical\s+)?emergency",
                r"(?:someone\s+)?(?:needs\s+help|is\s+hurt|injured)",
            ],
            "weight": 1.5
        }
    }
    
    # Content type patterns
    CONTENT_PATTERNS = {
        ContentType.TRAVEL_ANNOUNCEMENT: [
            r"(?:train|flight|bus|metro)\s+(?:number|no\.?)",
            r"(?:platform|gate|terminal)\s+(?:number|no\.?)?\s*\d+",
            r"(?:arriving|departing|boarding|delayed)",
            r"(?:passengers|travelers)",
        ],
        ContentType.SAFETY_INSTRUCTION: [
            r"(?:safety|security|emergency)",
            r"(?:evacuation|fire\s+exit|assembly\s+point)",
            r"(?:do\s+not|prohibited|forbidden)",
            r"(?:for\s+your\s+safety|safety\s+first)",
        ],
        ContentType.COMMERCIAL_MESSAGE: [
            r"(?:sale|discount|offer|buy|purchase)",
            r"(?:limited\s+time|special\s+offer|exclusive)",
            r"(?:call\s+now|order\s+now|visit\s+us)",
            r"(?:free|bonus|gift|prize)",
        ],
        ContentType.EMERGENCY_ALERT: [
            r"(?:emergency|urgent|critical|immediate)",
            r"(?:warning|alert|danger|caution)",
            r"(?:evacuate|shelter|take\s+cover)",
        ]
    }
    
    # Target audience patterns
    AUDIENCE_PATTERNS = {
        "passengers": [r"passengers", r"travelers", r"commuters"],
        "customers": [r"customers", r"shoppers", r"buyers", r"guests"],
        "employees": [r"employees", r"staff", r"workers", r"colleagues"],
        "students": [r"students", r"learners", r"class"],
        "patients": [r"patients", r"visitors", r"attendants"],
        "general_public": [r"everyone", r"all", r"public", r"citizens"],
        "specific_person": [r"mr\.?", r"mrs\.?", r"ms\.?", r"dr\.?"],
    }
    
    # Urgency indicators
    URGENCY_PATTERNS = {
        "critical": [r"emergency", r"immediately", r"now", r"urgent", r"critical"],
        "high": [r"as\s+soon\s+as", r"quickly", r"hurry", r"right\s+away", r"asap"],
        "medium": [r"soon", r"shortly", r"please", r"kindly"],
        "low": [r"when\s+possible", r"at\s+your\s+convenience", r"later"],
    }
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the semantic analyzer"""
        self.loaded = True
        print("✅ Semantic Analyzer loaded")
        return True
    
    def analyze(
        self,
        text: str,
        detected_language: str = "en",
        location: str = "Unknown",
        situation: str = "Unknown",
        sounds: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis
        
        Args:
            text: Transcribed text (in English)
            detected_language: Original language
            location: Detected location
            situation: Detected situation
            sounds: Detected sounds
            
        Returns:
            Complete semantic analysis
        """
        if not self.loaded or not text:
            return self._empty_result()
        
        text = text.strip()
        text_lower = text.lower()
        sounds = sounds or []
        
        # 1. Detect Intent (WHY they're speaking)
        intent, intent_conf = self._detect_intent(text_lower)
        
        # 2. Detect Content Type
        content_type = self._detect_content_type(text_lower, intent)
        
        # 3. Extract Entities (WHO, WHAT, WHERE, WHEN)
        entities = self._extract_entities(text)
        
        # 4. Extract Key Points
        key_points = self._extract_key_points(text, intent)
        
        # 5. Extract Topics
        topics = self._extract_topics(text_lower, location, situation)
        
        # 6. Extract Action Items
        action_items = self._extract_actions(text)
        
        # 7. Detect Questions
        questions = self._detect_questions(text)
        
        # 8. Detect Target Audience
        audience = self._detect_audience(text_lower)
        
        # 9. Assess Urgency
        urgency = self._assess_urgency(text_lower, intent)
        
        # 10. Analyze Sentiment
        sentiment = self._analyze_sentiment(text_lower)
        
        # 11. Generate Meaning Summary
        meaning_summary = self._generate_summary(
            text, intent, entities, key_points, audience
        )
        
        # 12. Generate Detailed Meaning
        detailed_meaning = self._generate_detailed_meaning(
            text, intent, entities, key_points, 
            action_items, questions, audience, location, situation
        )
        
        # 13. Extract Context Clues
        context_clues = self._extract_context_clues(text, sounds, location)
        
        return {
            # Core Understanding
            "intent": intent.value,
            "intent_confidence": round(intent_conf, 3),
            "content_type": content_type.value,
            
            # Meaning
            "meaning_summary": meaning_summary,
            "detailed_meaning": detailed_meaning,
            "what_they_mean": self._explain_meaning(text, intent, entities, action_items),
            
            # Key Information
            "key_points": key_points,
            "entities": entities,
            "topics": topics,
            
            # Actions & Questions
            "action_items": action_items,
            "questions_asked": questions,
            "has_questions": len(questions) > 0,
            "has_actions": len(action_items) > 0,
            
            # Context
            "target_audience": audience,
            "urgency_level": urgency,
            "sentiment": sentiment,
            "context_clues": context_clues,
            
            # Analysis Metadata
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "is_complete_thought": self._is_complete_thought(text),
            
            # Recommendations
            "listener_should": self._get_listener_recommendations(
                intent, action_items, urgency, audience
            )
        }
    
    def _detect_intent(self, text: str) -> Tuple[SpeechIntent, float]:
        """Detect the primary intent of the speech"""
        scores = defaultdict(float)
        
        for intent, data in self.INTENT_PATTERNS.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[intent] += data["weight"]
        
        if not scores:
            return SpeechIntent.UNKNOWN, 0.5
        
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        
        # Normalize confidence
        confidence = min(0.5 + (max_score * 0.15), 0.98)
        
        return best_intent, confidence
    
    def _detect_content_type(
        self,
        text: str,
        intent: SpeechIntent
    ) -> ContentType:
        """Detect the type of content"""
        for content_type, patterns in self.CONTENT_PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
            if matches >= 2:
                return content_type
        
        # Infer from intent
        intent_to_content = {
            SpeechIntent.ANNOUNCEMENT: ContentType.GENERAL_INFORMATION,
            SpeechIntent.WARNING: ContentType.SAFETY_INSTRUCTION,
            SpeechIntent.EMERGENCY: ContentType.EMERGENCY_ALERT,
            SpeechIntent.ADVERTISEMENT: ContentType.COMMERCIAL_MESSAGE,
        }
        
        return intent_to_content.get(intent, ContentType.OTHER)
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            "numbers": [],
            "times": [],
            "locations": [],
            "names": [],
            "organizations": [],
            "dates": [],
            "durations": [],
            "money": [],
            "contact": []
        }
        
        # Numbers (train numbers, flight numbers, platform numbers)
        entities["numbers"] = re.findall(
            r'\b(?:number|no\.?|#)?\s*(\d{1,5})\b', text, re.IGNORECASE
        )
        
        # Times
        entities["times"] = re.findall(
            r'\b(\d{1,2}[:.]\d{2}\s*(?:am|pm|AM|PM)?|\d{1,2}\s*(?:am|pm|AM|PM)|\d{4}\s*(?:hours|hrs))\b',
            text
        )
        
        # Platform/Gate/Terminal numbers
        platform_matches = re.findall(
            r'(?:platform|gate|terminal|track|lane)\s*(?:number|no\.?)?\s*(\d+)',
            text, re.IGNORECASE
        )
        if platform_matches:
            entities["locations"].extend([f"Platform {n}" for n in platform_matches])
        
        # Durations
        entities["durations"] = re.findall(
            r'(\d+\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?))',
            text, re.IGNORECASE
        )
        
        # Money
        entities["money"] = re.findall(
            r'(?:₹|rs\.?|inr|usd|\$|€|£)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            text, re.IGNORECASE
        )
        
        # Phone numbers
        entities["contact"] = re.findall(
            r'\b(\d{3}[-.]?\d{3}[-.]?\d{4}|\d{10}|\d{4})\b',
            text
        )
        
        # Proper nouns (capitalized words, potential names)
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Skip if it's the start of a sentence
                if i == 0 or words[i-1].endswith('.'):
                    continue
                # Skip common words
                if word.lower() not in ['the', 'a', 'an', 'this', 'that', 'please']:
                    entities["names"].append(word)
        
        # Clean up duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))[:5]
        
        return entities
    
    def _extract_key_points(self, text: str, intent: SpeechIntent) -> List[str]:
        """Extract key points from the text"""
        key_points = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Check for important markers
            importance_markers = [
                r'\b(?:important|please\s+note|attention|remember|must|should)\b',
                r'\b(?:first|second|finally|lastly)\b',
                r'\b(?:platform|gate|terminal|time|number)\s+\d+',
                r'\b(?:will|is|are)\s+(?:be\s+)?(?:arriving|departing|boarding|closing)',
            ]
            
            for marker in importance_markers:
                if re.search(marker, sentence, re.IGNORECASE):
                    key_points.append(sentence)
                    break
        
        # If no key points found, use first and last sentences
        if not key_points and sentences:
            if sentences[0].strip():
                key_points.append(sentences[0].strip())
            if len(sentences) > 1 and sentences[-1].strip():
                key_points.append(sentences[-1].strip())
        
        return key_points[:5]
    
    def _extract_topics(
        self,
        text: str,
        location: str,
        situation: str
    ) -> List[str]:
        """Extract discussion topics"""
        topics = []
        
        topic_patterns = {
            "travel": r'\b(?:train|flight|bus|metro|journey|travel|trip)\b',
            "time": r'\b(?:time|schedule|delay|late|early|arriving|departing)\b',
            "safety": r'\b(?:safety|security|emergency|caution|warning)\b',
            "service": r'\b(?:service|help|assistance|support|customer)\b',
            "location": r'\b(?:platform|gate|terminal|station|airport|stop)\b',
            "payment": r'\b(?:ticket|fare|payment|price|cost|fee)\b',
            "weather": r'\b(?:weather|rain|storm|temperature|hot|cold)\b',
            "food": r'\b(?:food|restaurant|cafe|meal|drink|coffee)\b',
            "health": r'\b(?:health|medical|doctor|hospital|emergency)\b',
            "event": r'\b(?:event|celebration|festival|ceremony|meeting)\b',
        }
        
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                topics.append(topic)
        
        # Add inferred topics from location/situation
        location_topics = {
            "Airport Terminal": ["travel", "flight"],
            "Railway Station": ["travel", "train"],
            "Hospital": ["health", "medical"],
            "Shopping Mall": ["shopping", "retail"],
            "Restaurant/Cafe": ["food", "dining"],
        }
        
        if location in location_topics:
            for t in location_topics[location]:
                if t not in topics:
                    topics.append(t)
        
        return topics[:6]
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract action items from text"""
        actions = []
        
        action_patterns = [
            (r'(?:please|kindly)\s+([^.!?]+)', "Please {}"),
            (r'(?:must|should|need\s+to)\s+([^.!?]+)', "Must {}"),
            (r'(?:proceed\s+to|go\s+to|move\s+to)\s+([^.!?]+)', "Go to {}"),
            (r'(?:call|contact|dial)\s+([^.!?]+)', "Contact {}"),
            (r'(?:wait\s+(?:at|for|until))\s+([^.!?]+)', "Wait {}"),
            (r'(?:board|enter|exit)\s+([^.!?]+)', "Board/Enter {}"),
            (r'(?:collect|pick\s+up|take)\s+([^.!?]+)', "Collect {}"),
        ]
        
        for pattern, template in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                action = template.format(match.strip()[:50])
                if action not in actions:
                    actions.append(action)
        
        return actions[:5]
    
    def _detect_questions(self, text: str) -> List[str]:
        """Detect questions in the text"""
        questions = []
        
        # Explicit questions (ending with ?)
        explicit = re.findall(r'([^.!?]*\?)', text)
        questions.extend([q.strip() for q in explicit if len(q.strip()) > 5])
        
        # Implicit questions
        implicit_patterns = [
            r'((?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|help|show)[^.!?]*)',
            r'((?:do\s+you\s+know|any\s+idea)[^.!?]*)',
            r'((?:where|when|what|how|who)\s+(?:is|are|was|were|do|does|can|will)[^.!?]*)',
        ]
        
        for pattern in implicit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            questions.extend([m.strip() for m in matches if len(m.strip()) > 5])
        
        return list(set(questions))[:5]
    
    def _detect_audience(self, text: str) -> str:
        """Detect target audience"""
        for audience, patterns in self.AUDIENCE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return audience
        
        return "general_public"
    
    def _assess_urgency(self, text: str, intent: SpeechIntent) -> str:
        """Assess urgency level"""
        if intent in [SpeechIntent.EMERGENCY, SpeechIntent.WARNING]:
            return "critical"
        
        for level, patterns in self.URGENCY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        
        return "normal"
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze overall sentiment"""
        positive_words = [
            "welcome", "thank", "please", "happy", "good", "great",
            "excellent", "wonderful", "pleasure", "glad", "appreciate"
        ]
        
        negative_words = [
            "sorry", "unfortunately", "delay", "cancel", "problem",
            "issue", "regret", "inconvenience", "apolog"
        ]
        
        neutral_words = [
            "attention", "note", "inform", "announce", "request"
        ]
        
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        
        if pos_count > neg_count + 1:
            return "positive"
        elif neg_count > pos_count + 1:
            return "negative"
        else:
            return "neutral"
    
    def _generate_summary(
        self,
        text: str,
        intent: SpeechIntent,
        entities: Dict,
        key_points: List[str],
        audience: str
    ) -> str:
        """Generate a concise meaning summary"""
        intent_verbs = {
            SpeechIntent.ANNOUNCEMENT: "announces",
            SpeechIntent.INSTRUCTION: "instructs",
            SpeechIntent.WARNING: "warns",
            SpeechIntent.QUESTION: "asks",
            SpeechIntent.REQUEST: "requests",
            SpeechIntent.INFORMATION: "informs",
            SpeechIntent.GREETING: "greets",
            SpeechIntent.FAREWELL: "says goodbye to",
            SpeechIntent.EMERGENCY: "alerts about emergency for",
        }
        
        verb = intent_verbs.get(intent, "communicates to")
        
        # Build summary
        summary_parts = [f"Speaker {verb} {audience}"]
        
        # Add key entity info
        if entities.get("numbers"):
            summary_parts.append(f"about number(s) {', '.join(entities['numbers'][:2])}")
        if entities.get("times"):
            summary_parts.append(f"at {entities['times'][0]}")
        if entities.get("locations"):
            summary_parts.append(f"regarding {entities['locations'][0]}")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_detailed_meaning(
        self,
        text: str,
        intent: SpeechIntent,
        entities: Dict,
        key_points: List[str],
        action_items: List[str],
        questions: List[str],
        audience: str,
        location: str,
        situation: str
    ) -> str:
        """Generate detailed meaning explanation"""
        parts = []
        
        # Context
        parts.append(f"📍 Context: This appears to be at/about {location} during {situation}.")
        
        # Intent
        intent_explanations = {
            SpeechIntent.ANNOUNCEMENT: "This is a public announcement meant to inform listeners.",
            SpeechIntent.INSTRUCTION: "The speaker is giving instructions that should be followed.",
            SpeechIntent.WARNING: "This is a warning that requires attention and possibly action.",
            SpeechIntent.QUESTION: "The speaker is asking question(s) and seeking information.",
            SpeechIntent.REQUEST: "The speaker is making a request.",
            SpeechIntent.INFORMATION: "The speaker is sharing information.",
            SpeechIntent.EMERGENCY: "⚠️ This is an emergency communication requiring immediate attention.",
        }
        parts.append(f"🎯 Purpose: {intent_explanations.get(intent, 'General communication.')}")
        
        # Key Information
        if key_points:
            parts.append(f"📋 Key Info: {key_points[0]}")
        
        # Actions
        if action_items:
            parts.append(f"✅ Actions Required: {'; '.join(action_items[:2])}")
        
        # Questions
        if questions:
            parts.append(f"❓ Questions Asked: {questions[0]}")
        
        return " | ".join(parts)
    
    def _explain_meaning(
        self,
        text: str,
        intent: SpeechIntent,
        entities: Dict,
        action_items: List[str]
    ) -> str:
        """
        Generate a human-friendly explanation of what the speaker means
        THIS IS THE KEY "WHAT THEY MEAN" OUTPUT
        """
        explanations = []
        
        # Start with the basic meaning
        if intent == SpeechIntent.ANNOUNCEMENT:
            explanations.append("The speaker is making an announcement to inform listeners")
        elif intent == SpeechIntent.INSTRUCTION:
            explanations.append("The speaker is telling you what to do")
        elif intent == SpeechIntent.WARNING:
            explanations.append("The speaker is warning you about something important")
        elif intent == SpeechIntent.QUESTION:
            explanations.append("The speaker is asking a question")
        elif intent == SpeechIntent.EMERGENCY:
            explanations.append("⚠️ The speaker is alerting about an EMERGENCY")
        elif intent == SpeechIntent.INFORMATION:
            explanations.append("The speaker is sharing information")
        else:
            explanations.append("The speaker is communicating")
        
        # Add specifics
        if entities.get("numbers"):
            numbers = entities["numbers"][:2]
            explanations.append(f"mentioning number(s): {', '.join(numbers)}")
        
        if entities.get("times"):
            explanations.append(f"about timing: {entities['times'][0]}")
        
        if entities.get("locations"):
            explanations.append(f"at/about: {entities['locations'][0]}")
        
        if action_items:
            explanations.append(f"You should: {action_items[0]}")
        
        # Join into readable format
        result = explanations[0]
        if len(explanations) > 1:
            result += " " + ", ".join(explanations[1:])
        
        return result + "."
    
    def _extract_context_clues(
        self,
        text: str,
        sounds: List[str],
        location: str
    ) -> List[str]:
        """Extract context clues that help understand the situation"""
        clues = []
        
        # From sounds
        if sounds:
            clues.append(f"🔊 Background sounds: {', '.join(sounds[:3])}")
        
        # From text patterns
        if re.search(r'train|railway|platform', text, re.IGNORECASE):
            clues.append("🚂 Train/Railway context detected")
        if re.search(r'flight|airport|gate|boarding', text, re.IGNORECASE):
            clues.append("✈️ Airport/Flight context detected")
        if re.search(r'emergency|help|urgent', text, re.IGNORECASE):
            clues.append("🚨 Emergency context detected")
        
        # From location
        if location and location != "Unknown":
            clues.append(f"📍 Location context: {location}")
        
        return clues[:5]
    
    def _is_complete_thought(self, text: str) -> bool:
        """Check if text represents a complete thought"""
        # Has subject and verb pattern
        if len(text.split()) < 3:
            return False
        
        # Ends with proper punctuation
        if text[-1] in '.!?':
            return True
        
        return len(text.split()) >= 5
    
    def _get_listener_recommendations(
        self,
        intent: SpeechIntent,
        action_items: List[str],
        urgency: str,
        audience: str
    ) -> List[str]:
        """Get recommendations for what the listener should do"""
        recommendations = []
        
        if urgency == "critical":
            recommendations.append("🚨 Take immediate action as this is urgent")
        
        if intent == SpeechIntent.INSTRUCTION:
            recommendations.append("📋 Follow the instructions given")
        
        if intent == SpeechIntent.WARNING:
            recommendations.append("⚠️ Pay attention and take precautions")
        
        if intent == SpeechIntent.QUESTION:
            recommendations.append("💬 A response or answer is expected")
        
        if action_items:
            recommendations.append(f"✅ Complete action: {action_items[0]}")
        
        if not recommendations:
            recommendations.append("ℹ️ This is informational - no immediate action required")
        
        return recommendations
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "intent": "unknown",
            "intent_confidence": 0,
            "content_type": "other",
            "meaning_summary": "Unable to analyze - no text provided",
            "detailed_meaning": "",
            "what_they_mean": "No speech content to analyze",
            "key_points": [],
            "entities": {},
            "topics": [],
            "action_items": [],
            "questions_asked": [],
            "has_questions": False,
            "has_actions": False,
            "target_audience": "unknown",
            "urgency_level": "normal",
            "sentiment": "neutral",
            "context_clues": [],
            "word_count": 0,
            "sentence_count": 0,
            "is_complete_thought": False,
            "listener_should": []
        }