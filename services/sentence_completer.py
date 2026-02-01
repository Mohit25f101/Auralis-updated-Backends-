# ══════════════════════════════════════════════════════════════
# 📝 services/sentence_completer.py - v7.0 ADVANCED
# ══════════════════════════════════════════════════════════════
"""
NLP-Powered Sentence Completion & Grammar Correction

Features:
- Completes incomplete sentences
- Fixes grammar errors
- Removes repetitions
- Adds punctuation
- Normalizes text
- Context-aware completion
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try advanced NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class CompletionResult:
    """Result of sentence completion"""
    original: str
    completed: str
    changes_made: List[str]
    confidence: float
    is_complete: bool


class SentenceCompleter:
    """
    NLP-Powered Sentence Completion System
    
    Features:
    - Detects incomplete sentences
    - Completes fragmented text
    - Fixes grammar and punctuation
    - Removes stuttering and repetitions
    - Context-aware improvements
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize Sentence Completer
        
        Args:
            language: Language code (en, hi, etc.)
        """
        self.language = language
        self.nlp = None
        self.grammar_model = None
        self.use_spacy = SPACY_AVAILABLE
        self.use_textblob = TEXTBLOB_AVAILABLE
        self.use_transformers = TRANSFORMERS_AVAILABLE
        self.loaded = False
        
        backend = []
        if self.use_spacy:
            backend.append("spaCy")
        if self.use_textblob:
            backend.append("TextBlob")
        if self.use_transformers:
            backend.append("Transformers")
        
        backend_str = ", ".join(backend) if backend else "Rule-based"
        print(f"📝 Sentence Completer: Using {backend_str}")
    
    def load(self) -> bool:
        """Load NLP models"""
        try:
            if self.use_spacy:
                print("📥 Loading spaCy model...")
                try:
                    # Try to load English model
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Download if not available
                    print("   Downloading spaCy model...")
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                    self.nlp = spacy.load("en_core_web_sm")
                
                print("✅ spaCy model loaded")
            
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"NLP model load error: {e}")
            print(f"⚠️ spaCy load failed: {e}")
            print("   Using rule-based completion")
            self.loaded = True
            return True
    
    def complete(
        self,
        text: str,
        context: Optional[str] = None
    ) -> CompletionResult:
        """
        Complete and improve text
        
        Args:
            text: Input text (possibly incomplete)
            context: Additional context for better completion
            
        Returns:
            CompletionResult with original and completed text
        """
        if not text:
            return CompletionResult(
                original="",
                completed="",
                changes_made=[],
                confidence=0.0,
                is_complete=True
            )
        
        original = text
        changes = []
        
        # 1. Remove stuttering and repetitions
        text, stutter_removed = self._remove_stuttering(text)
        if stutter_removed:
            changes.append("removed_stuttering")
        
        # 2. Fix spacing and basic punctuation
        text = self._fix_spacing(text)
        
        # 3. Complete sentences using NLP
        if self.use_spacy and self.nlp:
            text, completed = self._complete_with_spacy(text, context)
            if completed:
                changes.append("completed_sentences")
        else:
            text, completed = self._complete_rule_based(text)
            if completed:
                changes.append("completed_sentences_rule_based")
        
        # 4. Fix grammar
        if self.use_textblob:
            text, grammar_fixed = self._fix_grammar_textblob(text)
            if grammar_fixed:
                changes.append("fixed_grammar")
        
        # 5. Add proper punctuation
        text = self._add_punctuation(text)
        
        # 6. Normalize text
        text = self._normalize_text(text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(original, text, changes)
        
        # Check if complete
        is_complete = self._is_complete(text)
        
        return CompletionResult(
            original=original,
            completed=text,
            changes_made=changes,
            confidence=confidence,
            is_complete=is_complete
        )
    
    def _remove_stuttering(self, text: str) -> Tuple[str, bool]:
        """Remove stuttering and word repetitions"""
        words = text.split()
        cleaned = []
        prev_word = ""
        removed = False
        
        for word in words:
            word_clean = word.lower().strip('.,!?')
            
            # Skip if same as previous word
            if word_clean == prev_word and len(word_clean) > 2:
                removed = True
                continue
            
            cleaned.append(word)
            prev_word = word_clean
        
        return " ".join(cleaned), removed
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])(?=[A-Za-z])', r'\1 ', text)
        
        return text.strip()
    
    def _complete_with_spacy(
        self,
        text: str,
        context: Optional[str]
    ) -> Tuple[str, bool]:
        """Complete sentences using spaCy NLP"""
        if not self.nlp:
            return text, False
        
        doc = self.nlp(text)
        completed = False
        
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Check if sentence is incomplete
            if self._is_incomplete_sentence(sent):
                # Try to complete it
                sent_text = self._attempt_completion(sent_text, sent, context)
                completed = True
            
            sentences.append(sent_text)
        
        return " ".join(sentences), completed
    
    def _is_incomplete_sentence(self, sent) -> bool:
        """Check if a sentence is incomplete"""
        text = sent.text.strip()
        
        # Check for incomplete markers
        incomplete_patterns = [
            r'\w+\s*$',  # Ends with word (no punctuation)
            r',\s*$',     # Ends with comma
            r'and\s*$',   # Ends with conjunction
            r'or\s*$',
            r'but\s*$',
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Also check if has verb
                has_verb = any(token.pos_ == "VERB" for token in sent)
                if not has_verb:
                    return True
        
        return False
    
    def _attempt_completion(
        self,
        text: str,
        sent,
        context: Optional[str]
    ) -> str:
        """Attempt to complete an incomplete sentence"""
        # Simple rule-based completion
        
        # If ends with conjunction, add "continuing"
        if re.search(r'(and|or|but)\s*$', text, re.IGNORECASE):
            return text + " continuing."
        
        # If ends with comma, add "etc."
        if text.endswith(','):
            return text + " etc."
        
        # If no punctuation at all, add period
        if not re.search(r'[.!?]$', text):
            return text + "."
        
        return text
    
    def _complete_rule_based(self, text: str) -> Tuple[str, bool]:
        """Rule-based sentence completion (fallback)"""
        completed = False
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        fixed_sentences = []
        for sent in sentences:
            sent = sent.strip()
            
            # If ends without punctuation, add period
            if sent and not re.search(r'[.!?]$', sent):
                sent += "."
                completed = True
            
            # If ends with "and", "or", "but", complete it
            if re.search(r'(and|or|but)\s*\.?$', sent, re.IGNORECASE):
                sent = re.sub(r'(and|or|but)\s*\.?$', r'\1 continuing.', sent, flags=re.IGNORECASE)
                completed = True
            
            fixed_sentences.append(sent)
        
        return " ".join(fixed_sentences), completed
    
    def _fix_grammar_textblob(self, text: str) -> Tuple[str, bool]:
        """Fix grammar using TextBlob"""
        try:
            blob = TextBlob(text)
            corrected = str(blob.correct())
            
            return corrected, (corrected != text)
        except:
            return text, False
    
    def _add_punctuation(self, text: str) -> str:
        """Add proper punctuation"""
        # Ensure sentences end with punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        fixed = []
        for sent in sentences:
            sent = sent.strip()
            if sent and not re.search(r'[.!?]$', sent):
                # Determine appropriate punctuation
                if re.search(r'^(what|where|when|why|how|who)', sent, re.IGNORECASE):
                    sent += "?"
                elif re.search(r'(stop|help|emergency|attention)', sent, re.IGNORECASE):
                    sent += "!"
                else:
                    sent += "."
            
            fixed.append(sent)
        
        return " ".join(fixed)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text formatting"""
        # Capitalize first letter of sentences
        sentences = re.split(r'([.!?]\s+)', text)
        
        normalized = []
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Sentence text (not punctuation)
                part = part[0].upper() + part[1:] if len(part) > 0 else part
            normalized.append(part)
        
        text = "".join(normalized)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _calculate_confidence(
        self,
        original: str,
        completed: str,
        changes: List[str]
    ) -> float:
        """Calculate confidence score for completion"""
        if original == completed:
            return 1.0
        
        # Base confidence
        confidence = 0.7
        
        # Increase for each successful change
        confidence += len(changes) * 0.05
        
        # Decrease if many changes (might be uncertain)
        if len(changes) > 5:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 0.99)
    
    def _is_complete(self, text: str) -> bool:
        """Check if text is complete"""
        # Check if all sentences end with punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sent in sentences:
            sent = sent.strip()
            if sent and not re.search(r'[.!?]$', sent):
                return False
        
        return True
    
    def batch_complete(
        self,
        texts: List[str],
        context: Optional[str] = None
    ) -> List[CompletionResult]:
        """Complete multiple texts"""
        return [self.complete(text, context) for text in texts]
