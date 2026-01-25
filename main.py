# ==============================
# 📄 main.py - AURALIS v4.1 STANDALONE
# ==============================
# COMPLETE STANDALONE VERSION
# NO EXTERNAL LOCAL IMPORTS
# ==============================

import os
import sys
import logging
import warnings

# Silence everything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# ==============================
# 📦 IMPORTS
# ==============================
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
from datetime import datetime
from collections import defaultdict
import numpy as np
import csv
import requests
import shutil
import json
import uuid
import re
import time
import subprocess
import tempfile

# ==============================
# 📦 ML IMPORTS
# ==============================
print("\n" + "="*60)
print("🔄 Loading ML Libraries...")
print("="*60)

# TensorFlow
tf = None
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    print(f"   ✅ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"   ⚠️ TensorFlow: {e}")

# TensorFlow Hub
hub = None
try:
    import tensorflow_hub as hub
    print("   ✅ TensorFlow Hub")
except Exception as e:
    print(f"   ⚠️ TensorFlow Hub: {e}")

# Librosa
librosa = None
try:
    import librosa
    print(f"   ✅ Librosa {librosa.__version__}")
except Exception as e:
    print(f"   ⚠️ Librosa: {e}")

# Soundfile
sf = None
try:
    import soundfile as sf
    print("   ✅ Soundfile")
except:
    pass

# Scipy
wavfile = None
try:
    from scipy.io import wavfile
    print("   ✅ Scipy")
except:
    pass

# Transformers
pipeline_func = None
try:
    from transformers import pipeline as pipeline_func
    print("   ✅ Transformers")
except Exception as e:
    print(f"   ⚠️ Transformers: {e}")

# PyTorch
torch = None
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
except:
    pass

print("="*60)


# ==============================
# ⚙️ CONFIGURATION (INLINE)
# ==============================
SAMPLE_RATE = 16000
MAX_DURATION = 60.0
MIN_DURATION = 0.5

LOCATIONS = [
    "Airport Terminal", "Railway Station", "Bus Terminal", "Metro/Subway",
    "Hospital", "Shopping Mall", "Office Building", "School/University",
    "Restaurant/Cafe", "Street/Road", "Home/Residential", "Park/Outdoor",
    "Stadium/Arena", "Parking Area", "Construction Site", "Factory/Industrial",
    "Religious Place", "Government Office", "Bank", "Hotel/Lodge",
    "Cinema/Theater", "Gym/Sports Center", "Market/Bazaar", "Unknown"
]

SITUATIONS = [
    "Normal/Quiet", "Busy/Crowded", "Emergency", "Boarding/Departure",
    "Waiting", "Traffic", "Meeting/Conference", "Announcement",
    "Celebration/Event", "Construction", "Weather Event", "Accident",
    "Medical Emergency", "Security Alert", "Rush Hour", "Flight Delay",
    "Train Delay", "Sports Event", "Concert/Music", "Unknown"
]

FFMPEG_PATH = r"D:\photo\ffmpeg\ffmpeg-2026-01-07-git-af6a1dd0b2-full_build\bin"
LEARNING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_data.json")


# ==============================
# 🎬 FFMPEG
# ==============================
def setup_ffmpeg():
    try:
        if shutil.which("ffmpeg"):
            print("✅ FFmpeg found")
            return True
        if os.path.isdir(FFMPEG_PATH):
            os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ffmpeg"):
                print("✅ FFmpeg enabled")
                return True
        print("⚠️ FFmpeg not found")
        return False
    except:
        return False


def convert_audio(input_path, output_path):
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
        subprocess.run(cmd, capture_output=True, timeout=60)
        return os.path.exists(output_path)
    except:
        return False


# ==============================
# 📚 LEARNING SYSTEM
# ==============================
class LearningSystem:
    def __init__(self):
        self.boosts = defaultdict(lambda: defaultdict(float))
        self.corrections = 0
        self._load()
    
    def _load(self):
        try:
            if os.path.exists(LEARNING_FILE):
                with open(LEARNING_FILE, 'r') as f:
                    data = json.load(f)
                    self.boosts = defaultdict(lambda: defaultdict(float),
                        {k: defaultdict(float, v) for k, v in data.get('boosts', {}).items()})
                    self.corrections = data.get('corrections', 0)
                if self.corrections > 0:
                    print(f"📚 Loaded {self.corrections} learned corrections")
        except:
            pass
    
    def save(self):
        try:
            with open(LEARNING_FILE, 'w') as f:
                json.dump({
                    'boosts': {k: dict(v) for k, v in self.boosts.items()},
                    'corrections': self.corrections
                }, f)
        except:
            pass
    
    def learn(self, category, value, text, sounds):
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        for w in words:
            self.boosts[category][f"{value}::{w}"] += 0.1
        for s in sounds[:5]:
            self.boosts[category][f"{value}::sound::{s.lower()}"] += 0.12
        self.corrections += 1
        if self.corrections % 3 == 0:
            self.save()
    
    def get_boost(self, category, value, text, sounds):
        boost = 0.0
        words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        for w in words:
            boost += self.boosts[category].get(f"{value}::{w}", 0)
        for s in sounds:
            boost += self.boosts[category].get(f"{value}::sound::{s.lower()}", 0)
        return min(boost, 0.3)


# ==============================
# 🔊 AUDIO LOADER
# ==============================
class AudioLoader:
    def __init__(self):
        self.sr = SAMPLE_RATE
    
    def load(self, path):
        if not os.path.exists(path):
            raise Exception("File not found")
        
        audio = None
        
        # Try librosa
        if librosa and audio is None:
            try:
                audio, _ = librosa.load(path, sr=self.sr, mono=True)
                print("   ✅ Loaded with Librosa")
            except:
                pass
        
        # Try soundfile
        if sf and audio is None:
            try:
                data, orig_sr = sf.read(path)
                audio = data.astype(np.float32)
                if orig_sr != self.sr and librosa:
                    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                print("   ✅ Loaded with Soundfile")
            except:
                pass
        
        # Try FFmpeg
        if audio is None:
            temp = path + ".converted.wav"
            try:
                if convert_audio(path, temp) and wavfile:
                    orig_sr, data = wavfile.read(temp)
                    if data.dtype == np.int16:
                        audio = data.astype(np.float32) / 32768.0
                    elif data.dtype == np.float32:
                        audio = data
                    else:
                        audio = data.astype(np.float32)
                    if orig_sr != self.sr and librosa:
                        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)
                    print("   ✅ Loaded with FFmpeg")
            except Exception as e:
                print(f"   ⚠️ FFmpeg error: {e}")
            finally:
                if os.path.exists(temp):
                    os.remove(temp)
        
        if audio is None:
            raise Exception("Could not load audio")
        
        # Normalize
        audio = audio.astype(np.float32)
        audio = audio - np.mean(audio)
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx * 0.95
        
        return audio, len(audio) / self.sr
    
    def save_wav(self, audio, path):
        if wavfile:
            wavfile.write(path, self.sr, (audio * 32767).astype(np.int16))
        return path


# ==============================
# 🗣️ WHISPER MANAGER
# ==============================
class WhisperManager:
    HALLUCINATIONS = [
        "thank you for watching", "please subscribe", "like and subscribe",
        "thanks for watching", "bye bye", "goodbye", "[music]", "[applause]",
    ]
    
    def __init__(self):
        self.pipe = None
        self.loaded = False
    
    def load(self):
        if self.loaded:
            return True
        if not pipeline_func:
            print("❌ Whisper: Transformers not available")
            return False
        try:
            print("🔄 Loading Whisper...")
            self.pipe = pipeline_func(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                chunk_length_s=30,
            )
            self.loaded = True
            print("✅ Whisper loaded!")
            return True
        except Exception as e:
            print(f"❌ Whisper error: {e}")
            return False
    
    def transcribe(self, path):
        if not self.loaded:
            return {"text": "", "confidence": 0, "is_reliable": False}
        
        try:
            result = self.pipe(path, generate_kwargs={"task": "translate", "language": "en"})
            raw = result.get("text", "").strip()
            
            if not raw or len(raw) < 3:
                return {"text": "", "confidence": 0.2, "is_reliable": False}
            
            # Check hallucination
            lower = raw.lower()
            for h in self.HALLUCINATIONS:
                if h in lower:
                    return {"text": "", "raw": raw, "confidence": 0.2, "is_reliable": False}
            
            # Check repetition
            words = raw.split()
            if len(words) > 5 and len(set(words)) / len(words) < 0.35:
                return {"text": "", "raw": raw, "confidence": 0.25, "is_reliable": False}
            
            # Clean
            clean = ' '.join(raw.split())
            clean = re.sub(r'\[.*?\]', '', clean).strip()
            
            conf = 0.90 if len(words) >= 10 else (0.85 if len(words) >= 5 else 0.75)
            
            return {"text": clean, "raw": raw, "confidence": conf, "is_reliable": True}
        
        except Exception as e:
            print(f"   ⚠️ Transcription error: {str(e)[:50]}")
            return {"text": "", "confidence": 0, "is_reliable": False, "error": str(e)[:50]}


# ==============================
# 🔊 YAMNET MANAGER
# ==============================
class YAMNetManager:
    FILTER_OUT = {"silence", "white noise", "pink noise", "static"}
    
    GENERIC = {
        "speech", "narration", "monologue", "conversation",
        "male speech", "female speech", "inside", "outside", "room",
    }
    
    SOUND_LOCATIONS = {
        "aircraft": ["Airport Terminal"], "airplane": ["Airport Terminal"],
        "jet engine": ["Airport Terminal"], "helicopter": ["Airport Terminal"],
        "train": ["Railway Station"], "railroad": ["Railway Station"],
        "rail transport": ["Railway Station"], "subway": ["Metro/Subway"],
        "car": ["Street/Road"], "vehicle": ["Street/Road"],
        "traffic": ["Street/Road"], "horn": ["Street/Road"],
        "siren": ["Hospital", "Street/Road"], "ambulance": ["Hospital"],
        "crowd": ["Airport Terminal", "Railway Station", "Shopping Mall", "Stadium/Arena"],
        "applause": ["Stadium/Arena"], "cheering": ["Stadium/Arena"],
        "bell": ["Religious Place"], "church bell": ["Religious Place"],
        "construction": ["Construction Site"], "hammer": ["Construction Site"],
        "bird": ["Park/Outdoor"], "water": ["Park/Outdoor"],
    }
    
    SOUND_SITUATIONS = {
        "siren": ["Emergency"], "alarm": ["Emergency"],
        "ambulance": ["Medical Emergency"], "crowd": ["Busy/Crowded"],
        "applause": ["Celebration/Event"], "cheering": ["Sports Event"],
        "traffic": ["Traffic"], "horn": ["Traffic"],
        "construction": ["Construction"], "thunder": ["Weather Event"],
    }
    
    def __init__(self):
        self.model = None
        self.labels = []
        self.loaded = False
    
    def load(self):
        if self.loaded:
            return True
        if not hub or not tf:
            print("❌ YAMNet: TensorFlow not available")
            return False
        try:
            print("🔄 Loading YAMNet...")
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            self._load_labels()
            self.loaded = True
            print(f"✅ YAMNet loaded ({len(self.labels)} classes)")
            return True
        except Exception as e:
            print(f"❌ YAMNet error: {e}")
            return False
    
    def _load_labels(self):
        try:
            url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
            r = requests.get(url, timeout=10)
            reader = csv.reader(r.text.splitlines())
            next(reader)
            self.labels = [row[2] for row in reader]
        except:
            self.labels = [f"Sound_{i}" for i in range(521)]
    
    def analyze(self, audio):
        if not self.loaded:
            return {"sounds": {}, "location_hints": [], "situation_hints": []}
        
        try:
            scores, _, _ = self.model(audio.astype(np.float32))
            mean = tf.reduce_mean(scores, axis=0).numpy()
            
            all_sounds = {}
            for i in range(len(mean)):
                if mean[i] > 0.01 and i < len(self.labels):
                    all_sounds[self.labels[i]] = float(mean[i])
            
            filtered = self._filter(all_sounds)
            loc_hints = self._location_hints(all_sounds)
            sit_hints = self._situation_hints(all_sounds)
            
            return {
                "sounds": filtered,
                "all": all_sounds,
                "location_hints": loc_hints,
                "situation_hints": sit_hints,
            }
        except Exception as e:
            print(f"   ⚠️ YAMNet error: {e}")
            return {"sounds": {}, "location_hints": [], "situation_hints": []}
    
    def _filter(self, sounds):
        result = {}
        generic = {}
        for s, score in sounds.items():
            lower = s.lower()
            if any(f in lower for f in self.FILTER_OUT):
                continue
            if any(g in lower for g in self.GENERIC):
                if score > 0.05:
                    generic[s] = score
            elif score > 0.02:
                result[s] = round(score, 3)
        
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:12])
        if not result and generic:
            result = dict(sorted(generic.items(), key=lambda x: x[1], reverse=True)[:5])
        return result
    
    def _location_hints(self, sounds):
        hints = defaultdict(float)
        for s, score in sounds.items():
            lower = s.lower()
            for kw, locs in self.SOUND_LOCATIONS.items():
                if kw in lower:
                    for loc in locs:
                        hints[loc] = max(hints[loc], score)
        return sorted(hints.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _situation_hints(self, sounds):
        hints = defaultdict(float)
        for s, score in sounds.items():
            lower = s.lower()
            for kw, sits in self.SOUND_SITUATIONS.items():
                if kw in lower:
                    for sit in sits:
                        hints[sit] = max(hints[sit], score)
        return sorted(hints.items(), key=lambda x: x[1], reverse=True)[:5]


# ==============================
# 🧠 ANALYZER
# ==============================
class Analyzer:
    def __init__(self, learning):
        self.learning = learning
        self._init_keywords()
    
    def _init_keywords(self):
        self.loc_kw = {
            "Airport Terminal": {
                "airport": 5, "terminal": 5, "flight": 5, "airline": 5, "aircraft": 5,
                "boarding": 4.5, "gate": 4, "runway": 5, "takeoff": 5, "landing": 5,
                "captain": 4, "pilot": 4.5, "cabin": 4, "heathrow": 5, "lufthansa": 5,
                "emirates": 5, "british airways": 5, "air india": 5, "indigo": 4.5,
                "economy": 3.5, "business class": 4, "passport": 3.5, "baggage": 3.5,
                "departure": 4, "arrival": 4, "passengers": 2.5,
            },
            "Railway Station": {
                "railway": 5, "train": 5, "platform": 5, "station": 3.5, "locomotive": 5,
                "coach": 4, "compartment": 4, "bogey": 4.5, "rail": 4.5, "track": 3.5,
                "rajdhani": 5, "shatabdi": 5, "indian railways": 5, "irctc": 5,
                "express": 3, "local train": 4, "reservation": 3.5, "engine": 3,
            },
            "Hospital": {
                "hospital": 5, "doctor": 5, "nurse": 5, "patient": 5, "medical": 4.5,
                "emergency": 4, "ambulance": 5, "surgery": 5, "ward": 4.5, "icu": 5,
                "clinic": 4, "medicine": 3.5, "treatment": 4,
            },
            "Shopping Mall": {
                "mall": 5, "shopping": 5, "shop": 4, "store": 4, "sale": 3.5,
                "discount": 3.5, "customer": 2.5, "escalator": 4.5, "food court": 5,
            },
            "Street/Road": {
                "traffic": 5, "road": 4.5, "highway": 5, "street": 4, "car": 3.5,
                "vehicle": 3.5, "signal": 4, "crossing": 3.5, "jam": 4, "horn": 3.5,
            },
            "Office Building": {
                "office": 5, "meeting": 4.5, "conference": 5, "presentation": 4.5,
                "manager": 4, "corporate": 4.5, "company": 3.5, "project": 3.5,
            },
            "Stadium/Arena": {
                "stadium": 5, "arena": 5, "match": 5, "game": 4, "team": 3.5,
                "score": 4.5, "goal": 5, "cricket": 5, "football": 5, "ipl": 5,
            },
            "Religious Place": {
                "temple": 5, "church": 5, "mosque": 5, "prayer": 4.5, "worship": 4.5,
                "aarti": 5, "namaz": 5, "mandir": 5, "masjid": 5,
            },
            "Restaurant/Cafe": {
                "restaurant": 5, "cafe": 5, "food": 3.5, "menu": 4.5, "waiter": 4.5,
                "chef": 5, "table": 3, "order": 3.5,
            },
            "Park/Outdoor": {
                "park": 5, "garden": 5, "outdoor": 4, "nature": 4.5, "tree": 3.5,
                "bird": 3.5, "walk": 2.5, "bench": 4,
            },
            "Metro/Subway": {
                "metro": 5, "subway": 5, "underground": 5, "delhi metro": 5,
                "mumbai metro": 5, "token": 3.5,
            },
            "Construction Site": {
                "construction": 5, "cement": 4.5, "crane": 5, "scaffold": 5,
                "building site": 5, "labor": 3.5,
            },
        }
        
        self.sit_kw = {
            "Emergency": {
                "emergency": 5, "help": 4.5, "fire": 5, "accident": 5, "danger": 5,
                "police": 4.5, "ambulance": 5, "injured": 5, "attack": 5, "urgent": 4.5,
            },
            "Boarding/Departure": {
                "boarding": 5, "now boarding": 5, "final call": 5, "departure": 4.5,
                "proceed to": 4.5, "gate": 3.5, "platform": 3.5, "all passengers": 4,
                "economy customers": 5, "continue boarding": 5,
            },
            "Flight Delay": {
                "delayed": 5, "delay": 5, "postponed": 4.5, "rescheduled": 4.5,
                "inconvenience": 4, "regret": 4, "technical": 4,
            },
            "Announcement": {
                "attention": 4.5, "announcement": 5, "ladies and gentlemen": 5,
                "attention please": 5, "information": 4, "kindly": 3.5, "passengers": 3,
            },
            "Traffic": {
                "traffic": 5, "jam": 4.5, "congestion": 5, "heavy traffic": 5,
                "stuck": 4, "slow": 3.5,
            },
            "Busy/Crowded": {
                "crowded": 5, "busy": 4, "rush": 4, "queue": 4, "packed": 4.5,
                "rush hour": 5,
            },
            "Sports Event": {
                "goal": 5, "score": 4.5, "match": 4.5, "win": 4, "tournament": 4.5,
            },
            "Normal/Quiet": {},
        }
    
    def analyze(self, trans, sounds):
        text = trans.get("text", "").lower()
        raw = trans.get("raw", text).lower()
        reliable = trans.get("is_reliable", False)
        
        detected = sounds.get("sounds", {})
        loc_hints = sounds.get("location_hints", [])
        sit_hints = sounds.get("situation_hints", [])
        
        full_text = f"{text} {raw}".lower()
        sound_list = list(detected.keys())
        
        # Weights
        tw = 0.70 if reliable and text else 0.25
        sw = 1.0 - tw
        
        # Detect
        loc, loc_conf, loc_ev = self._detect_loc(full_text, loc_hints, tw, sw, sound_list)
        sit, sit_conf, is_emg, sit_ev = self._detect_sit(full_text, sit_hints, tw, sw, sound_list)
        
        emg_prob = 0.98 if is_emg else self._calc_emg(full_text, sit_hints)
        overall = (loc_conf + sit_conf) / 2
        
        evidence = self._build_ev(text, reliable, loc_ev, sit_ev, detected, loc_hints)
        summary = self._build_sum(loc, loc_conf, sit, sit_conf, is_emg, emg_prob, text)
        
        return {
            "location": loc,
            "location_confidence": round(loc_conf, 3),
            "situation": sit,
            "situation_confidence": round(sit_conf, 3),
            "is_emergency": is_emg,
            "emergency_probability": round(emg_prob, 3),
            "overall_confidence": round(overall, 3),
            "evidence": evidence,
            "summary": summary,
            "detected_sounds": sound_list[:10],
            "text_reliable": reliable,
        }
    
    def _detect_loc(self, text, hints, tw, sw, sounds):
        scores = {}
        evidence = {}
        
        for loc, kws in self.loc_kw.items():
            score = 0.0
            ev = []
            matched = []
            
            kw_score = sum(w for k, w in kws.items() if k in text)
            if kw_score > 0:
                matched = [k for k in kws if k in text][:4]
                score += min(kw_score / 10, 1.5) * tw
                if matched:
                    ev.append(f"Keywords: {', '.join(matched)}")
            
            for hint_loc, hint_score in hints:
                if hint_loc == loc and hint_score > 0.05:
                    score += hint_score * sw * 2
                    ev.append(f"Sound: {loc}")
            
            boost = self.learning.get_boost('location', loc, text, sounds)
            if boost > 0:
                score += boost
            
            if score > 0.1:
                scores[loc] = score
                evidence[loc] = ev
        
        if not scores:
            if hints:
                best = hints[0]
                return best[0], 0.55 + best[1] * 0.3, [f"Sound suggests: {best[0]}"]
            return "Unknown", 0.40, ["No clear indicators"]
        
        best = max(scores, key=scores.get)
        conf = 0.70 + min(scores[best] / 2, 0.28)
        return best, round(conf, 3), evidence.get(best, [])
    
    def _detect_sit(self, text, hints, tw, sw, sounds):
        candidates = []
        
        for sit, kws in self.sit_kw.items():
            score = 0.0
            ev = []
            
            kw_score = sum(w for k, w in kws.items() if k in text)
            if kw_score > 0:
                matched = [k for k in kws if k in text][:3]
                score += min(kw_score / 8, 1.5) * tw
                if matched:
                    ev.append(f"Keywords: {', '.join(matched)}")
            
            for hint_sit, hint_score in hints:
                if hint_sit == sit and hint_score > 0.05:
                    score += hint_score * sw * 2
                    ev.append(f"Sound: {sit}")
            
            boost = self.learning.get_boost('situation', sit, text, sounds)
            score += boost
            
            is_emg = sit in ["Emergency", "Medical Emergency", "Accident"]
            priority = 10 if is_emg else (7 if sit == "Boarding/Departure" else 5)
            
            if score > 0.1:
                candidates.append({"sit": sit, "score": score, "priority": priority, 
                                  "is_emg": is_emg, "ev": ev})
        
        if not candidates:
            if hints:
                best = hints[0]
                return best[0], 0.60, False, [f"Sound: {best[0]}"]
            return "Normal/Quiet", 0.55, False, ["No specific situation"]
        
        candidates.sort(key=lambda x: (x["priority"], x["score"]), reverse=True)
        best = candidates[0]
        conf = 0.68 + min(best["score"] / 1.5, 0.28)
        return best["sit"], round(conf, 3), best["is_emg"], best["ev"]
    
    def _calc_emg(self, text, hints):
        prob = 0.03
        for w, b in {"help": 0.15, "emergency": 0.2, "fire": 0.2, "accident": 0.18}.items():
            if w in text:
                prob += b
        for sit, score in hints:
            if sit in ["Emergency", "Medical Emergency"]:
                prob += score * 0.4
        return min(prob, 0.95)
    
    def _build_ev(self, text, reliable, loc_ev, sit_ev, sounds, hints):
        ev = []
        if reliable and text:
            preview = text[:80] + "..." if len(text) > 80 else text
            ev.append(f'🗣️ Speech: "{preview}"')
        else:
            ev.append("🗣️ Speech unclear - using sound analysis")
        
        for e in loc_ev[:2]:
            ev.append(f"📍 {e}")
        for e in sit_ev[:2]:
            ev.append(f"🎯 {e}")
        
        if sounds:
            ev.append(f"🔊 Sounds: {', '.join(list(sounds.keys())[:5])}")
        
        if hints:
            hint_str = ", ".join([f"{h[0]} ({h[1]*100:.0f}%)" for h in hints[:2]])
            ev.append(f"💡 Hints: {hint_str}")
        
        return ev[:6]
    
    def _build_sum(self, loc, loc_conf, sit, sit_conf, is_emg, emg_prob, text):
        parts = [
            f"📍 **{loc}** ({loc_conf*100:.0f}%)",
            f"🎯 **{sit}** ({sit_conf*100:.0f}%)"
        ]
        if is_emg:
            parts.append(f"🚨 **EMERGENCY** ({emg_prob*100:.0f}%)")
        if text and len(text) > 10:
            parts.append(f'💬 "{text[:50]}..."')
        return " | ".join(parts)


# ==============================
# 📊 MODELS
# ==============================
class AnalysisResult(BaseModel):
    location: str
    location_confidence: float
    situation: str
    situation_confidence: float
    is_emergency: bool
    emergency_probability: float
    overall_confidence: float
    transcribed_text: str
    text_reliable: bool
    detected_sounds: List[str]
    evidence: List[str]
    summary: str
    processing_time_ms: float
    request_id: str
    timestamp: str
    audio_duration: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models: Dict[str, bool]
    timestamp: str


class FeedbackRequest(BaseModel):
    request_id: str
    correct_location: Optional[str] = None
    correct_situation: Optional[str] = None


# ==============================
# 🌐 GLOBALS
# ==============================
loader = AudioLoader()
whisper = WhisperManager()
yamnet = YAMNetManager()
learning = LearningSystem()
analyzer = Analyzer(learning)
history = {}


# ==============================
# 🚀 LIFESPAN
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("🚀 AURALIS v4.1 - STANDALONE")
    print("="*60)
    
    setup_ffmpeg()
    whisper.load()
    yamnet.load()
    
    print("\n" + "="*60)
    print("✅ READY")
    print("="*60)
    print(f"🌐 http://127.0.0.1:8000")
    print(f"📚 http://127.0.0.1:8000/docs")
    print(f"📍 {len(LOCATIONS)} locations | 🎯 {len(SITUATIONS)} situations")
    print(f"🔊 {len(yamnet.labels)} sounds | 📚 {learning.corrections} corrections")
    print("="*60 + "\n")
    
    yield
    
    learning.save()
    print("💾 Saved | 👋 Bye")


# ==============================
# 🚀 APP
# ==============================
app = FastAPI(title="Auralis", version="4.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ==============================
# 📍 ROUTES
# ==============================
@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy", version="4.1.0",
        models={"whisper": whisper.loaded, "yamnet": yamnet.loaded},
        timestamp=datetime.now().isoformat()
    )

@app.get("/labels")
async def labels():
    return {"locations": LOCATIONS, "situations": SITUATIONS, "sounds": len(yamnet.labels)}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    start = time.time()
    rid = uuid.uuid4().hex[:8]
    
    tmp = tempfile.gettempdir()
    ext = os.path.splitext(file.filename)[1] or ".wav"
    orig = os.path.join(tmp, f"aur_{rid}{ext}")
    wav = os.path.join(tmp, f"aur_{rid}.wav")
    
    print("\n" + "="*60)
    print(f"📥 {file.filename} | ID: {rid}")
    print("="*60)
    
    try:
        content = await file.read()
        with open(orig, "wb") as f:
            f.write(content)
        print(f"💾 {len(content)/1024:.1f} KB")
        
        print("🔊 Loading...")
        audio, dur = loader.load(orig)
        print(f"⏱️ {dur:.2f}s")
        
        if dur < MIN_DURATION:
            raise HTTPException(400, "Too short")
        if dur > MAX_DURATION:
            audio = audio[:int(SAMPLE_RATE * MAX_DURATION)]
            dur = MAX_DURATION
        
        loader.save_wav(audio, wav)
        
        print("🎤 Transcribing...")
        trans = whisper.transcribe(wav)
        if trans.get("is_reliable"):
            print(f"📝 ✓ '{trans['text'][:70]}...'")
        else:
            print(f"📝 ⚠️ Unreliable")
        
        print("🔊 Sounds...")
        snd = yamnet.analyze(audio)
        if snd["sounds"]:
            print(f"🔊 {list(snd['sounds'].keys())[:5]}")
        if snd["location_hints"]:
            print(f"💡 {[h[0] for h in snd['location_hints'][:3]]}")
        
        print("🧠 Analyzing...")
        res = analyzer.analyze(trans, snd)
        
        history[rid] = {"trans": trans, "sounds": snd, "result": res}
        if len(history) > 500:
            del history[list(history.keys())[0]]
        
        proc = (time.time() - start) * 1000
        
        response = AnalysisResult(
            location=res["location"],
            location_confidence=res["location_confidence"],
            situation=res["situation"],
            situation_confidence=res["situation_confidence"],
            is_emergency=res["is_emergency"],
            emergency_probability=res["emergency_probability"],
            overall_confidence=res["overall_confidence"],
            transcribed_text=trans.get("text", ""),
            text_reliable=trans.get("is_reliable", False),
            detected_sounds=res["detected_sounds"],
            evidence=res["evidence"],
            summary=res["summary"],
            processing_time_ms=round(proc, 1),
            request_id=rid,
            timestamp=datetime.now().isoformat(),
            audio_duration=round(dur, 2)
        )
        
        print(f"\n✅ RESULT:")
        print(f"   📍 {response.location} ({response.location_confidence*100:.0f}%)")
        print(f"   🎯 {response.situation} ({response.situation_confidence*100:.0f}%)")
        print(f"   🚨 Emergency: {response.is_emergency}")
        print(f"   ⏱️ {response.processing_time_ms:.0f}ms")
        print("="*60 + "\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        for f in [orig, wav]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.request_id not in history:
        return {"status": "not_found"}
    h = history[req.request_id]
    if req.correct_location:
        learning.learn('location', req.correct_location, h['trans'].get('text', ''),
                      list(h['sounds'].get('sounds', {}).keys()))
    if req.correct_situation:
        learning.learn('situation', req.correct_situation, h['trans'].get('text', ''),
                      list(h['sounds'].get('sounds', {}).keys()))
    return {"status": "learned", "corrections": learning.corrections}


@app.get("/stats")
async def stats():
    return {"version": "4.1.0", "analyses": len(history), 
            "models": {"whisper": whisper.loaded, "yamnet": yamnet.loaded},
            "corrections": learning.corrections}


# Auth mock
@app.post("/login")
async def login(d: dict): return {"token": uuid.uuid4().hex[:16]}
@app.post("/register") 
async def register(d: dict): return {"status": "ok"}
@app.post("/save_history")
async def save_hist(d: dict): return {"id": uuid.uuid4().hex[:8]}


# ==============================
# 🏃 RUN
# ==============================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Auralis v4.1...\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")