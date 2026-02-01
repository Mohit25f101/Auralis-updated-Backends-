# ==============================
# 📄 services/question_detector.py
# ==============================
"""
Question Detection Service
Detects and analyzes questions in speech
"""

import re
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass


class QuestionType(Enum):
    """Types of questions"""
    WHAT = "what"           # What is...?
    WHERE = "where"         # Where is...?
    WHEN = "when"           # When does...?
    WHY = "why"             # Why is...?
    HOW = "how"             # How do...?
    WHO = "who"             # Who is...?
    WHICH = "which"         # Which one...?
    YES_NO = "yes_no"       # Is it...? Do you...?
    CHOICE = "choice"       # A or B?
    RHETORICAL = "rhetorical"
    REQUEST = "request"     # Can you...? Could you...?
    CONFIRMATION = "confirmation"  # ...right? ...isn't it?
    UNKNOWN = "unknown"


@dataclass
class Question:
    """Represents a detected question"""
    text: str
    question_type: QuestionType
    is_explicit: bool  # Has question mark
    subject: Optional[str]
    expected_answer_type: str


class QuestionDetector:
    """
    Question Detection Service
    
    Detects:
    - Explicit questions (with ?)
    - Implicit questions
    - Question types (what, where, when, etc.)
    - Expected answer types
    """
    
    # Question patterns
    QUESTION_PATTERNS = {
        QuestionType.WHAT: [
            r'^what\s+(?:is|are|was|were|do|does|did|can|could|will|would|should)',
            r'^what\s+\w+',
            r'\bwhat\s+(?:is|are)\s+\w+\?',
        ],
        QuestionType.WHERE: [
            r'^where\s+(?:is|are|was|were|do|does|did|can|could)',
            r'^where\s+\w+',
            r'\bwhere\s+(?:is|are)\s+\w+\?',
        ],
        QuestionType.WHEN: [
            r'^when\s+(?:is|are|was|were|do|does|did|will|would)',
            r'^when\s+\w+',
            r'\bwhen\s+(?:does|will)\s+\w+\?',
        ],
        QuestionType.WHY: [
            r'^why\s+(?:is|are|was|were|do|does|did|can|could)',
            r'^why\s+\w+',
        ],
        QuestionType.HOW: [
            r'^how\s+(?:is|are|was|were|do|does|did|can|could|much|many|long|far)',
            r'^how\s+\w+',
        ],
        QuestionType.WHO: [
            r'^who\s+(?:is|are|was|were|do|does|did|can|could)',
            r'^who\s+\w+',
        ],
        QuestionType.WHICH: [
            r'^which\s+(?:is|are|one|platform|gate|train|flight)',
            r'^which\s+\w+',
        ],
        QuestionType.YES_NO: [
            r'^(?:is|are|was|were|do|does|did|can|could|will|would|should|have|has)\s+',
            r'^(?:am|is|are)\s+(?:i|you|we|they|it|this|that)',
        ],
        QuestionType.REQUEST: [
            r'^(?:can|could|would|will)\s+you\s+(?:please\s+)?',
            r'^(?:may|might)\s+i\s+',
        ],
        QuestionType.CONFIRMATION: [
            r'\b(?:right|correct|isn\'t\s+it|aren\'t\s+they|don\'t\s+you)\s*\?',
            r'\b(?:okay|ok|yeah)\s*\?',
        ],
    }
    
    # Expected answer types
    ANSWER_TYPES = {
        QuestionType.WHAT: "information",
        QuestionType.WHERE: "location",
        QuestionType.WHEN: "time",
        QuestionType.WHY: "reason",
        QuestionType.HOW: "method/manner",
        QuestionType.WHO: "person",
        QuestionType.WHICH: "choice",
        QuestionType.YES_NO: "yes/no",
        QuestionType.REQUEST: "action/help",
        QuestionType.CONFIRMATION: "confirmation",
    }
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the detector"""
        self.loaded = True
        print("✅ Question Detector loaded")
        return True
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect questions in text
        
        Args:
            text: Input text
            
        Returns:
            Detection results
        """
        if not self.loaded or not text:
            return self._empty_result()
        
        questions = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for explicit question (ends with ?)
            is_explicit = sentence.endswith('?')
            
            # Detect question type
            q_type = self._detect_question_type(sentence)
            
            if is_explicit or q_type != QuestionType.UNKNOWN:
                questions.append(Question(
                    text=sentence,
                    question_type=q_type,
                    is_explicit=is_explicit,
                    subject=self._extract_subject(sentence),
                    expected_answer_type=self.ANSWER_TYPES.get(q_type, "unknown")
                ))
        
        # Also detect implicit questions
        implicit = self._detect_implicit_questions(text)
        for q in implicit:
            if q not in [qu.text for qu in questions]:
                questions.append(Question(
                    text=q,
                    question_type=QuestionType.REQUEST,
                    is_explicit=False,
                    subject=None,
                    expected_answer_type="information"
                ))
        
        # Convert to dict format
        question_list = [
            {
                "question": q.text,
                "type": q.question_type.value,
                "is_explicit": q.is_explicit,
                "subject": q.subject,
                "expected_answer": q.expected_answer_type
            }
            for q in questions
        ]
        
        # Get question type counts
        type_counts = {}
        for q in questions:
            t = q.question_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "questions": question_list,
            "has_questions": len(questions) > 0,
            "question_count": len(questions),
            "explicit_count": sum(1 for q in questions if q.is_explicit),
            "implicit_count": sum(1 for q in questions if not q.is_explicit),
            "question_types": type_counts,
            "primary_type": max(type_counts, key=type_counts.get) if type_counts else None,
            "requires_response": any(
                q.question_type in [QuestionType.REQUEST, QuestionType.YES_NO]
                for q in questions
            )
        }
    
    def _detect_question_type(self, sentence: str) -> QuestionType:
        """Detect the type of question"""
        sentence_lower = sentence.lower().strip()
        
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    return q_type
        
        # If ends with ? but no pattern matched
        if sentence.endswith('?'):
            return QuestionType.UNKNOWN
        
        return QuestionType.UNKNOWN
    
    def _extract_subject(self, sentence: str) -> Optional[str]:
        """Extract the subject of the question"""
        # Remove question word and extract main subject
        patterns = [
            r'(?:what|where|when|who|which)\s+(?:is|are)\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
            r'(?:is|are)\s+(?:the|this|that)\s+(\w+(?:\s+\w+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _detect_implicit_questions(self, text: str) -> List[str]:
        """Detect implicit questions (without ?)"""
        implicit = []
        
        patterns = [
            r'(?:i\s+)?(?:wonder|wondering)\s+(?:if|whether|what|where|when|why|how)\s+[^.!]+',
            r'(?:can|could)\s+you\s+(?:please\s+)?(?:tell|help|show)\s+[^.!?]+',
            r'(?:do\s+you\s+know|any\s+idea)\s+(?:if|whether|what|where|when)\s+[^.!?]+',
            r'(?:i\'d\s+like\s+to\s+know|tell\s+me)\s+[^.!?]+',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            implicit.extend(matches)
        
        return implicit[:5]
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "questions": [],
            "has_questions": False,
            "question_count": 0,
            "explicit_count": 0,
            "implicit_count": 0,
            "question_types": {},
            "primary_type": None,
            "requires_response": False
        }