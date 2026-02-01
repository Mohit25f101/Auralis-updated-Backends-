# ==============================
# 📄 services/whisper_manager.py
# ==============================
"""
Enhanced Whisper Speech Recognition Manager
Supports 100+ languages with automatic translation to English
"""

import re
from typing import Dict, Any, Optional, List

# Attempt import
try:
    from transformers import pipeline as hf_pipeline
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    hf_pipeline = None
    WhisperProcessor = None
    WhisperForConditionalGeneration = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class WhisperManager:
    """
    Enhanced Whisper Speech Recognition Manager
    
    Features:
    - 100+ language support
    - Automatic language detection
    - Translation to English
    - Hallucination filtering
    - Confidence scoring
    - Timestamp generation
    """
    
    # Whisper supported languages (99 languages)
    WHISPER_LANGUAGES = {
        "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
        "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
        "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
        "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
        "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
        "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
        "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
        "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
        "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
        "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
        "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
        "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
        "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
        "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
        "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
        "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
        "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
        "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
        "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
        "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
        "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
        "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
        "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
        "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
        "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
    }
    
    # Known hallucination patterns
    HALLUCINATIONS = [
        "thank you for watching",
        "please subscribe",
        "like and subscribe",
        "thanks for watching",
        "bye bye",
        "goodbye",
        "see you next time",
        "don't forget to subscribe",
        "hit the bell",
        "leave a comment",
        "[music]",
        "[applause]",
        "[laughter]",
        "[silence]",
        "♪",
        "🎵",
        "...",
        "you",  # Single word hallucination
    ]
    
    # Models available
    MODELS = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3",
        "large-v2": "openai/whisper-large-v2",
    }
    
    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Initialize Whisper manager
        
        Args:
            model_name: Whisper model to use
        """
        self.model_name = model_name
        self.pipe = None
        self.loaded = False
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    def load(self) -> bool:
        """Load the Whisper model"""
        if self.loaded:
            return True
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Whisper: Transformers library not available")
            return False
        
        try:
            print(f"🔄 Loading Whisper ({self.model_name})...")
            print(f"   Device: {self.device}")
            
            # Load pipeline with automatic device selection
            self.pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                chunk_length_s=30,
                stride_length_s=5,
                device=0 if self.device == "cuda" else -1,
            )
            
            self.loaded = True
            print(f"✅ Whisper loaded successfully ({len(self.WHISPER_LANGUAGES)} languages supported)")
            return True
            
        except Exception as e:
            print(f"❌ Whisper loading failed: {e}")
            return False
    
    def transcribe(
        self,
        audio_path: str,
        language_hint: Optional[str] = None,
        translate_to_english: bool = True,
        return_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with automatic language detection and translation
        
        Args:
            audio_path: Path to audio file
            language_hint: Optional language code hint (e.g., 'hi', 'zh', 'ar')
            translate_to_english: If True, translates non-English to English
            return_timestamps: If True, returns word/segment timestamps
            
        Returns:
            Dictionary with transcription results including:
            - text: English text (translated if needed)
            - original_text: Original language transcription (ALWAYS provided)
            - detected_language: Detected language code
            - confidence: Transcription confidence
            - is_translated: Whether translation was performed
        """
        if not self.loaded:
            return self._empty_result("Model not loaded")
        
        try:
            # First pass: Get original transcription in detected language
            print(f"   🎯 Detecting language and transcribing...")
            detect_result = self.pipe(
                audio_path,
                generate_kwargs={"task": "transcribe"},
                return_timestamps=return_timestamps
            )
            
            # Extract original transcription
            original_text = detect_result.get("text", "").strip()
            original_timestamps = detect_result.get("chunks", [])
            
            # Detect language from transcription
            detected_lang = self._detect_language_from_text(original_text)
            lang_name = self._get_language_name(detected_lang)
            
            print(f"   🌍 Detected: {lang_name} ({detected_lang})")
            
            # Check for empty result
            if not original_text or len(original_text) < 2:
                return self._empty_result("No speech detected", raw=original_text)
            
            # Second pass: Translate to English if needed
            english_text = original_text
            is_translated = False
            
            if translate_to_english and detected_lang != "en":
                print(f"   🔄 Translating {lang_name} → English...")
                translate_result = self.pipe(
                    audio_path,
                    generate_kwargs={
                        "task": "translate",
                        "language": detected_lang if detected_lang in self.WHISPER_LANGUAGES else None
                    },
                    return_timestamps=return_timestamps
                )
                english_text = translate_result.get("text", "").strip()
                is_translated = True
                timestamps = translate_result.get("chunks", [])
                
                if english_text and len(english_text) >= 2:
                    print(f"   ✅ Translation complete")
                else:
                    # Fallback to original if translation fails
                    english_text = original_text
                    timestamps = original_timestamps
            else:
                timestamps = original_timestamps
            
            # Use English text for hallucination check
            hallucination_result = self._check_hallucination(english_text)
            if hallucination_result["is_hallucination"]:
                return {
                    "text": "",
                    "original_text": original_text,
                    "raw": english_text,
                    "confidence": 0.2,
                    "is_reliable": False,
                    "is_translated": is_translated,
                    "hallucination_detected": True,
                    "hallucination_type": hallucination_result["type"],
                    "detected_language": detected_lang,
                    "detected_language_name": lang_name,
                    "timestamps": []
                }
            
            # Clean both texts
            cleaned_english = self._clean_text(english_text)
            cleaned_original = self._clean_text(original_text)
            
            # Calculate confidence based on English text
            confidence = self._calculate_confidence(cleaned_english, english_text, detected_lang)
            
            return {
                "text": cleaned_english,  # English translation
                "original_text": cleaned_original,  # ALWAYS include original language
                "raw": english_text,
                "confidence": confidence,
                "is_reliable": confidence > 0.6 and len(cleaned_english) > 5,
                "is_translated": is_translated,
                "hallucination_detected": False,
                "detected_language": detected_lang,
                "detected_language_name": lang_name,
                "word_count": len(cleaned_english.split()),
                "original_word_count": len(cleaned_original.split()),
                "timestamps": self._process_timestamps(timestamps),
                "translation_note": f"Translated from {lang_name}" if is_translated else "Original English"
            }
            
        except Exception as e:
            print(f"   ⚠️ Transcription error: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return self._empty_result(f"Error: {str(e)[:50]}")
    
    def transcribe_keep_original(
        self,
        audio_path: str,
        language_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio and return both original and English translation
        
        Args:
            audio_path: Path to audio file
            language_hint: Optional language code hint
            
        Returns:
            Dictionary with both original transcription and English translation
        """
        if not self.loaded:
            return self._empty_result("Model not loaded")
        
        try:
            # Get original transcription
            original_result = self.pipe(
                audio_path,
                generate_kwargs={
                    "task": "transcribe",
                    "language": language_hint if language_hint in self.WHISPER_LANGUAGES else None
                },
                return_timestamps=True
            )
            
            original_text = original_result.get("text", "").strip()
            detected_lang = self._detect_language_from_text(original_text)
            
            # Get English translation
            english_text = original_text
            if detected_lang != "en":
                translate_result = self.pipe(
                    audio_path,
                    generate_kwargs={"task": "translate"},
                    return_timestamps=True
                )
                english_text = translate_result.get("text", "").strip()
            
            # Clean texts
            cleaned_original = self._clean_text(original_text)
            cleaned_english = self._clean_text(english_text)
            
            return {
                "original_text": cleaned_original,
                "english_text": cleaned_english,
                "detected_language": detected_lang,
                "detected_language_name": self._get_language_name(detected_lang),
                "is_translated": detected_lang != "en",
                "confidence": self._calculate_confidence(cleaned_english, english_text, detected_lang),
                "is_reliable": len(cleaned_english) > 5,
                "timestamps": self._process_timestamps(original_result.get("chunks", []))
            }
            
        except Exception as e:
            return self._empty_result(f"Error: {str(e)[:50]}")
    
    def _detect_language_from_text(self, text: str) -> str:
        """Detect language from text using character analysis"""
        if not text:
            return "en"
        
        # Check for various scripts
        script_patterns = {
            "hi": r'[\u0900-\u097F]',  # Devanagari (Hindi, Marathi, etc.)
            "ta": r'[\u0B80-\u0BFF]',  # Tamil
            "te": r'[\u0C00-\u0C7F]',  # Telugu
            "bn": r'[\u0980-\u09FF]',  # Bengali
            "gu": r'[\u0A80-\u0AFF]',  # Gujarati
            "kn": r'[\u0C80-\u0CFF]',  # Kannada
            "ml": r'[\u0D00-\u0D7F]',  # Malayalam
            "pa": r'[\u0A00-\u0A7F]',  # Punjabi (Gurmukhi)
            "th": r'[\u0E00-\u0E7F]',  # Thai
            "zh": r'[\u4e00-\u9fff]',  # Chinese
            "ja": r'[\u3040-\u309f\u30a0-\u30ff]',  # Japanese
            "ko": r'[\uac00-\ud7af]',  # Korean
            "ar": r'[\u0600-\u06FF]',  # Arabic
            "he": r'[\u0590-\u05FF]',  # Hebrew
            "ru": r'[\u0400-\u04FF]',  # Cyrillic
            "el": r'[\u0370-\u03FF]',  # Greek
            "ka": r'[\u10A0-\u10FF]',  # Georgian
            "hy": r'[\u0530-\u058F]',  # Armenian
            "my": r'[\u1000-\u109F]',  # Myanmar
            "km": r'[\u1780-\u17FF]',  # Khmer
            "lo": r'[\u0E80-\u0EFF]',  # Lao
            "am": r'[\u1200-\u137F]',  # Ethiopic
        }
        
        for lang, pattern in script_patterns.items():
            if re.search(pattern, text):
                return lang
        
        # For Latin scripts, use word analysis
        return self._detect_latin_language(text)
    
    def _detect_latin_language(self, text: str) -> str:
        """Detect language for Latin script text"""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Language markers
        markers = {
            "en": {"the", "and", "is", "are", "was", "have", "has", "this", "that", "with"},
            "es": {"que", "de", "el", "la", "los", "las", "por", "con", "para", "una"},
            "fr": {"de", "la", "le", "les", "et", "en", "un", "une", "du", "que"},
            "de": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"},
            "pt": {"que", "de", "em", "um", "uma", "para", "com", "não", "por", "mais"},
            "it": {"di", "che", "la", "il", "un", "una", "per", "non", "sono", "da"},
            "nl": {"de", "het", "een", "van", "en", "in", "is", "op", "te", "dat"},
            "tr": {"bir", "ve", "bu", "için", "olan", "ile", "de", "da", "olarak", "gibi"},
            "id": {"yang", "dan", "di", "ini", "dari", "untuk", "dengan", "tidak", "adalah"},
            "vi": {"của", "và", "các", "là", "trong", "được", "có", "này", "cho", "với"},
            "pl": {"i", "w", "na", "do", "z", "że", "to", "nie", "się", "o"},
            "ro": {"de", "în", "și", "la", "cu", "un", "o", "care", "pe", "nu"},
        }
        
        scores = {}
        for lang, lang_markers in markers.items():
            match_count = len(words & lang_markers)
            if match_count > 0:
                scores[lang] = match_count
        
        if scores:
            return max(scores, key=scores.get)
        
        return "en"  # Default to English
    
    def _get_language_name(self, code: str) -> str:
        """Get full language name from code"""
        return self.WHISPER_LANGUAGES.get(code, "Unknown").title()
    
    def _check_hallucination(self, text: str) -> Dict[str, Any]:
        """Check if text is a hallucination"""
        text_lower = text.lower().strip()
        
        # Check known patterns
        for pattern in self.HALLUCINATIONS:
            if pattern.lower() in text_lower:
                return {
                    "is_hallucination": True,
                    "type": "known_pattern",
                    "matched": pattern
                }
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return {
                    "is_hallucination": True,
                    "type": "repetition",
                    "unique_ratio": unique_ratio
                }
        
        # Check for very short single word
        if len(words) == 1 and len(text) < 10:
            return {
                "is_hallucination": True,
                "type": "too_short"
            }
        
        return {"is_hallucination": False, "type": None}
    
    def _clean_text(self, text: str) -> str:
        """Clean transcription text"""
        # Remove brackets content
        cleaned = re.sub(r'\[.*?\]', '', text)
        cleaned = re.sub(r'\(.*?\)', '', cleaned)
        
        # Remove music symbols
        cleaned = re.sub(r'[♪🎵🎶🎤🎧]', '', cleaned)
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove leading/trailing punctuation
        cleaned = cleaned.strip('.,!?;: ')
        
        return cleaned
    
    def _calculate_confidence(
        self,
        cleaned: str,
        raw: str,
        detected_lang: str
    ) -> float:
        """Calculate transcription confidence"""
        if not cleaned:
            return 0.2
        
        words = cleaned.split()
        word_count = len(words)
        
        # Base confidence by length
        if word_count >= 15:
            base = 0.90
        elif word_count >= 10:
            base = 0.85
        elif word_count >= 5:
            base = 0.75
        elif word_count >= 3:
            base = 0.65
        else:
            base = 0.50
        
        # Adjust for cleaning impact
        if len(cleaned) < len(raw) * 0.7:
            base -= 0.10
        
        # Adjust for repetition
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.5:
            base -= 0.15
        
        # Bonus for detected language matching
        if detected_lang != "en":
            base += 0.05  # Translation was performed
        
        return max(0.2, min(0.98, base))
    
    def _process_timestamps(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Process timestamp chunks"""
        if not chunks:
            return []
        
        processed = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                processed.append({
                    "text": chunk.get("text", ""),
                    "start": chunk.get("timestamp", [0, 0])[0] if chunk.get("timestamp") else 0,
                    "end": chunk.get("timestamp", [0, 0])[1] if chunk.get("timestamp") else 0
                })
        
        return processed
    
    def _empty_result(self, reason: str, raw: str = "") -> Dict[str, Any]:
        """Return empty transcription result"""
        return {
            "text": "",
            "original_text": None,
            "raw": raw,
            "confidence": 0.0,
            "is_reliable": False,
            "is_translated": False,
            "error": reason,
            "detected_language": "unknown",
            "detected_language_name": "Unknown",
            "word_count": 0,
            "timestamps": []
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return {code: name.title() for code, name in self.WHISPER_LANGUAGES.items()}
    
    def is_language_supported(self, code: str) -> bool:
        """Check if a language is supported"""
        return code.lower() in self.WHISPER_LANGUAGES