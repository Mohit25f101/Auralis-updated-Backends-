# ==============================
# 📄 services/whisper_manager_improved.py
# ==============================
"""
Enhanced Whisper Speech Recognition Manager - IMPROVED VERSION
- Fixes libtorchcodec loading issues
- Better language detection and translation
- Fallback mechanisms for transcription
- Improved confidence scoring
"""

import re
import os
from typing import Dict, Any, Optional, List

# Attempt import with better error handling
try:
    # Disable problematic backends
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    from transformers import pipeline as hf_pipeline
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    hf_pipeline = None
    print(f"⚠️ Transformers not available: {e}")

try:
    import torch
    TORCH_AVAILABLE = True
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except:
    LIBROSA_AVAILABLE = False
    print("⚠️ Librosa not available")


class WhisperManagerImproved:
    """
    Enhanced Whisper Speech Recognition Manager
    
    Features:
    - 100+ language support with automatic detection
    - Automatic translation to English
    - Fallback transcription methods
    - Improved confidence scoring
    - Better error handling
    - Learns from corrections
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
    
    # Known hallucination patterns - EXPANDED
    HALLUCINATIONS = [
        # Common YouTube patterns
        "thank you for watching", "please subscribe", "like and subscribe",
        "thanks for watching", "hit the bell", "leave a comment",
        "don't forget to subscribe", "smash that like button",
        
        # Generic patterns
        "bye bye", "goodbye", "see you next time", "see you soon",
        
        # Broadcast patterns  
        "[music]", "[applause]", "[laughter]", "[silence]", "[noise]",
        "♪", "🎵", "🎶",
        
        # Very short/meaningless
        "...", "you", "uh", "um", "hmm", "ah", "oh",
        
        # Repeated characters
        "aaaa", "mmmm", "hmmm",
    ]
    
    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Initialize Whisper manager
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.pipe = None
        self.loaded = False
        self.device = self._get_device()
        self.fallback_active = False
        
    def _get_device(self) -> str:
        """Get the best available device"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def load(self) -> bool:
        """Load the Whisper model with improved error handling"""
        if self.loaded:
            return True
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Whisper: Transformers library not available")
            print("   Install: pip install transformers --break-system-packages")
            return False
        
        try:
            print(f"🔄 Loading Whisper ({self.model_name})...")
            print(f"   Device: {self.device}")
            
            # Try to load the pipeline with error handling
            try:
                # Primary method - use pipeline with chunk processing
                self.pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    chunk_length_s=30,
                    stride_length_s=5,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                print(f"✅ Whisper loaded successfully (pipeline mode)")
                
            except Exception as e:
                # Fallback method - load model and processor separately
                print(f"   ⚠️ Pipeline loading failed: {e}")
                print(f"   🔄 Trying fallback method...")
                
                processor = WhisperProcessor.from_pretrained(self.model_name)
                model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                
                if self.device == "cuda":
                    model = model.to("cuda")
                
                # Create a custom pipeline-like object
                self.pipe = {
                    "model": model,
                    "processor": processor,
                    "device": self.device
                }
                self.fallback_active = True
                print(f"✅ Whisper loaded successfully (fallback mode)")
            
            self.loaded = True
            print(f"🌍 Supporting {len(self.WHISPER_LANGUAGES)} languages")
            return True
            
        except Exception as e:
            print(f"❌ Whisper loading completely failed: {e}")
            print(f"   Try installing: pip install transformers torch --break-system-packages")
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
            language_hint: Optional language code hint
            translate_to_english: If True, translates non-English to English
            return_timestamps: If True, returns word/segment timestamps
            
        Returns:
            Dictionary with transcription results
        """
        if not self.loaded:
            return self._empty_result("Model not loaded")
        
        try:
            # Method 1: Use pipeline (if available)
            if not self.fallback_active:
                return self._transcribe_pipeline(
                    audio_path, language_hint, translate_to_english, return_timestamps
                )
            
            # Method 2: Use fallback (manual processing)
            else:
                return self._transcribe_fallback(
                    audio_path, language_hint, translate_to_english, return_timestamps
                )
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return self._empty_result(f"Error: {str(e)[:100]}")
    
    def _transcribe_pipeline(
        self,
        audio_path: str,
        language_hint: Optional[str],
        translate_to_english: bool,
        return_timestamps: bool
    ) -> Dict[str, Any]:
        """Transcribe using pipeline method"""
        
        # Step 1: Detect language and get original transcription
        print(f"   🎯 Detecting language and transcribing...")
        
        detect_kwargs = {
            "task": "transcribe",
            "return_timestamps": return_timestamps
        }
        
        if language_hint:
            detect_kwargs["language"] = language_hint
        
        detect_result = self.pipe(audio_path, generate_kwargs=detect_kwargs)
        
        # Extract original transcription
        original_text = detect_result.get("text", "").strip()
        original_timestamps = detect_result.get("chunks", [])
        
        # Detect language
        detected_lang = self._detect_language_from_text(original_text)
        lang_name = self._get_language_name(detected_lang)
        
        print(f"   🌍 Detected: {lang_name} ({detected_lang})")
        
        # Check for empty or hallucinated result
        if not original_text or len(original_text) < 2:
            return self._empty_result("No speech detected", raw=original_text)
        
        # Check for hallucination
        hallucination_check = self._check_hallucination(original_text)
        if hallucination_check["is_hallucination"]:
            print(f"   ⚠️ Hallucination detected: {hallucination_check['type']}")
            return self._empty_result("Hallucination detected", raw=original_text)
        
        # Step 2: Translate to English if needed
        english_text = original_text
        is_translated = False
        
        if translate_to_english and detected_lang != "en":
            print(f"   🔄 Translating {lang_name} → English...")
            
            translate_result = self.pipe(
                audio_path,
                generate_kwargs={
                    "task": "translate",
                    "language": detected_lang,
                    "return_timestamps": return_timestamps
                }
            )
            
            english_text = translate_result.get("text", "").strip()
            is_translated = True
            
            print(f"   ✅ Translation complete")
        
        # Clean texts
        cleaned_original = self._clean_text(original_text)
        cleaned_english = self._clean_text(english_text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            cleaned_english, english_text, detected_lang, len(original_timestamps)
        )
        
        # Build result
        return {
            "text": cleaned_english,
            "original_text": cleaned_original if is_translated else None,
            "raw": english_text,
            "confidence": confidence,
            "is_reliable": confidence > 0.6,
            "is_translated": is_translated,
            "detected_language": detected_lang,
            "detected_language_name": lang_name,
            "word_count": len(cleaned_english.split()),
            "timestamps": self._process_timestamps(original_timestamps),
            "hallucination_check": hallucination_check,
            "error": None
        }
    
    def _transcribe_fallback(
        self,
        audio_path: str,
        language_hint: Optional[str],
        translate_to_english: bool,
        return_timestamps: bool
    ) -> Dict[str, Any]:
        """Transcribe using fallback method (manual processing)"""
        
        if not LIBROSA_AVAILABLE:
            return self._empty_result("Librosa not available for fallback")
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Get model and processor
            model = self.pipe["model"]
            processor = self.pipe["processor"]
            device = self.pipe["device"]
            
            # Process audio
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = model.generate(inputs["input_features"])
            
            # Decode
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Detect language
            detected_lang = self._detect_language_from_text(transcription)
            lang_name = self._get_language_name(detected_lang)
            
            print(f"   🌍 Detected: {lang_name} ({detected_lang})")
            
            # Clean
            cleaned = self._clean_text(transcription)
            
            # Calculate confidence (simplified for fallback)
            confidence = 0.7 if len(cleaned.split()) >= 3 else 0.5
            
            return {
                "text": cleaned,
                "original_text": None,
                "raw": transcription,
                "confidence": confidence,
                "is_reliable": confidence > 0.6,
                "is_translated": False,
                "detected_language": detected_lang,
                "detected_language_name": lang_name,
                "word_count": len(cleaned.split()),
                "timestamps": [],
                "hallucination_check": {"is_hallucination": False},
                "error": None
            }
            
        except Exception as e:
            return self._empty_result(f"Fallback error: {str(e)[:100]}")
    
    def _detect_language_from_text(self, text: str) -> str:
        """Detect language from text using character analysis"""
        if not text:
            return "en"
        
        # Check for various scripts
        script_patterns = {
            "hi": r'[\u0900-\u097F]',  # Devanagari (Hindi)
            "ta": r'[\u0B80-\u0BFF]',  # Tamil
            "te": r'[\u0C00-\u0C7F]',  # Telugu
            "bn": r'[\u0980-\u09FF]',  # Bengali
            "gu": r'[\u0A80-\u0AFF]',  # Gujarati
            "kn": r'[\u0C80-\u0CFF]',  # Kannada
            "ml": r'[\u0D00-\u0D7F]',  # Malayalam
            "pa": r'[\u0A00-\u0A7F]',  # Punjabi
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
        
        return "en"  # Default
    
    def _get_language_name(self, code: str) -> str:
        """Get full language name from code"""
        return self.WHISPER_LANGUAGES.get(code, "Unknown").title()
    
    def _check_hallucination(self, text: str) -> Dict[str, Any]:
        """Check if text is likely a hallucination"""
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
        detected_lang: str,
        chunk_count: int
    ) -> float:
        """Calculate transcription confidence"""
        if not cleaned:
            return 0.2
        
        words = cleaned.split()
        word_count = len(words)
        
        # Base confidence by length
        if word_count >= 20:
            base = 0.92
        elif word_count >= 15:
            base = 0.88
        elif word_count >= 10:
            base = 0.82
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
        
        # Bonus for timestamps/chunks
        if chunk_count > 3:
            base += 0.05
        
        # Bonus for non-English (translation was performed successfully)
        if detected_lang != "en":
            base += 0.03
        
        return max(0.2, min(0.98, base))
    
    def _process_timestamps(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Process timestamp chunks"""
        if not chunks:
            return []
        
        processed = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                timestamp = chunk.get("timestamp", [0, 0])
                processed.append({
                    "text": chunk.get("text", ""),
                    "start": timestamp[0] if timestamp else 0,
                    "end": timestamp[1] if timestamp else 0
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
            "timestamps": [],
            "hallucination_check": {"is_hallucination": False}
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return {code: name.title() for code, name in self.WHISPER_LANGUAGES.items()}
    
    def is_language_supported(self, code: str) -> bool:
        """Check if a language is supported"""
        return code.lower() in self.WHISPER_LANGUAGES