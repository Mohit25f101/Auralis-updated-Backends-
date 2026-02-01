# ==============================
# 📍 services/ai_location_detector.py - ENHANCED v6.0
# ==============================
"""
AI-Powered Location Detection with Nearby Suggestions
- Exact location detection
- Nearby location suggestions
- GPS coordinate support
- Semantic similarity matching
- Multi-language location names
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False


class AILocationDetector:
    """
    AI-Powered Location Detector
    
    Features:
    - Exact location detection from keywords
    - Nearby location suggestions
    - GPS coordinate matching
    - Semantic similarity (AI-powered)
    - Fuzzy string matching
    - Multi-language support
    """
    
    # Location categories with embeddings
    LOCATION_TYPES = {
        "Airport Terminal": {
            "keywords": ["airport", "terminal", "flight", "gate", "runway", "aircraft", "airline"],
            "subcategories": ["International Airport", "Domestic Terminal", "Private Airport"],
            "related": ["Railway Station", "Bus Terminal"]
        },
        "Railway Station": {
            "keywords": ["railway", "train", "platform", "station", "locomotive", "rail"],
            "subcategories": ["Metro Station", "Junction", "Terminal Station"],
            "related": ["Metro/Subway", "Bus Terminal"]
        },
        "Metro/Subway": {
            "keywords": ["metro", "subway", "underground", "tube"],
            "subcategories": ["Metro Station", "Interchange Station"],
            "related": ["Railway Station", "Bus Terminal"]
        },
        "Hospital": {
            "keywords": ["hospital", "clinic", "medical", "doctor", "emergency"],
            "subcategories": ["Government Hospital", "Private Hospital", "Emergency Room"],
            "related": ["Pharmacy", "Medical Center"]
        },
        "Shopping Mall": {
            "keywords": ["mall", "shopping", "store", "retail"],
            "subcategories": ["Shopping Center", "Department Store", "Outlet Mall"],
            "related": ["Market/Bazaar", "Restaurant/Cafe"]
        },
        "Restaurant/Cafe": {
            "keywords": ["restaurant", "cafe", "food", "dining", "eatery"],
            "subcategories": ["Fast Food", "Fine Dining", "Cafe", "Food Court"],
            "related": ["Shopping Mall", "Hotel/Lodge"]
        },
        "Office Building": {
            "keywords": ["office", "building", "corporate", "business"],
            "subcategories": ["Corporate Office", "Coworking Space", "Business Park"],
            "related": ["Industrial Area", "Commercial Complex"]
        },
        "School/University": {
            "keywords": ["school", "university", "college", "education", "campus"],
            "subcategories": ["Primary School", "High School", "University", "Institute"],
            "related": ["Library", "Educational Complex"]
        },
        "Park/Outdoor": {
            "keywords": ["park", "garden", "outdoor", "playground"],
            "subcategories": ["Public Park", "Botanical Garden", "Playground"],
            "related": ["Stadium/Arena", "Beach"]
        },
        "Stadium/Arena": {
            "keywords": ["stadium", "arena", "sports", "field", "ground"],
            "subcategories": ["Cricket Stadium", "Football Stadium", "Indoor Arena"],
            "related": ["Gym/Sports Center", "Park/Outdoor"]
        },
        "Hotel/Lodge": {
            "keywords": ["hotel", "lodge", "resort", "accommodation"],
            "subcategories": ["5-Star Hotel", "Budget Hotel", "Resort"],
            "related": ["Restaurant/Cafe", "Tourist Spot"]
        },
        "Street/Road": {
            "keywords": ["street", "road", "highway", "lane", "avenue"],
            "subcategories": ["Main Road", "Highway", "Residential Street"],
            "related": ["Parking Area", "Traffic Junction"]
        },
        "Market/Bazaar": {
            "keywords": ["market", "bazaar", "vendors", "stalls"],
            "subcategories": ["Vegetable Market", "Flea Market", "Street Market"],
            "related": ["Shopping Mall", "Street/Road"]
        },
        "Religious Place": {
            "keywords": ["temple", "church", "mosque", "gurudwara", "monastery"],
            "subcategories": ["Temple", "Church", "Mosque", "Gurudwara", "Synagogue"],
            "related": ["Tourist Spot", "Cultural Center"]
        },
        "Government Office": {
            "keywords": ["government", "municipal", "office", "department"],
            "subcategories": ["Municipal Office", "Government Building", "Post Office"],
            "related": ["Court", "Police Station"]
        },
        "Bank": {
            "keywords": ["bank", "atm", "financial", "branch"],
            "subcategories": ["Commercial Bank", "ATM", "Branch Office"],
            "related": ["Office Building", "Shopping Mall"]
        },
        "Construction Site": {
            "keywords": ["construction", "building", "site", "work"],
            "subcategories": ["Building Construction", "Road Construction", "Infrastructure"],
            "related": ["Factory/Industrial", "Office Building"]
        },
        "Factory/Industrial": {
            "keywords": ["factory", "industrial", "manufacturing", "plant"],
            "subcategories": ["Manufacturing Plant", "Warehouse", "Industrial Area"],
            "related": ["Construction Site", "Office Building"]
        },
        "Parking Area": {
            "keywords": ["parking", "garage", "lot"],
            "subcategories": ["Multi-level Parking", "Open Parking", "Underground Parking"],
            "related": ["Shopping Mall", "Office Building"]
        },
        "Home/Residential": {
            "keywords": ["home", "house", "apartment", "residential"],
            "subcategories": ["Apartment", "Villa", "Residential Complex"],
            "related": ["Neighborhood", "Street/Road"]
        },
    }
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize AI Location Detector
        
        Args:
            use_ai: Use AI-powered semantic matching (requires sentence-transformers)
        """
        self.use_ai = use_ai and SENTENCE_TRANSFORMERS_AVAILABLE
        self.use_fuzzy = FUZZYWUZZY_AVAILABLE
        self.use_geocoding = GEOPY_AVAILABLE
        
        self.model = None
        self.location_embeddings = {}
        self.geocoder = None
        
        if self.use_geocoding:
            try:
                self.geocoder = Nominatim(user_agent="auralis_location_detector")
            except:
                self.use_geocoding = False
        
        self.loaded = False
        
        print(f"📍 AI Location Detector: AI={'✅' if self.use_ai else '❌'}, Fuzzy={'✅' if self.use_fuzzy else '❌'}, Geocoding={'✅' if self.use_geocoding else '❌'}")
    
    def load(self) -> bool:
        """Load the AI model"""
        try:
            if self.use_ai:
                print("📥 Loading AI location model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & accurate
                
                # Pre-compute embeddings for all locations
                self._compute_location_embeddings()
                
                print("✅ AI Location Model loaded")
            
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"⚠️ AI model loading failed: {e}")
            print("✅ Falling back to keyword matching")
            self.use_ai = False
            self.loaded = True
            return True
    
    def _compute_location_embeddings(self):
        """Pre-compute embeddings for all location types"""
        if not self.use_ai or not self.model:
            return
        
        for loc_type, info in self.LOCATION_TYPES.items():
            # Combine keywords and location name
            text = f"{loc_type} {' '.join(info['keywords'])}"
            embedding = self.model.encode(text)
            self.location_embeddings[loc_type] = embedding
    
    def detect(
        self,
        text: str,
        sounds: Dict[str, float],
        acoustic_features: Optional[Dict] = None,
        user_location: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect location with AI-powered analysis
        
        Args:
            text: Transcribed text
            sounds: Detected sounds with confidence
            acoustic_features: Acoustic analysis results
            user_location: User's GPS coordinates (latitude, longitude)
            
        Returns:
            Dict with:
            - exact_location: Best match location
            - confidence: Confidence score
            - nearby_locations: List of nearby/similar locations
            - coordinates: GPS coordinates (if available)
            - distance_km: Distance from user (if GPS provided)
        """
        if not self.loaded:
            return self._empty_result()
        
        # Score locations using multiple signals
        scores = defaultdict(float)
        
        # 1. AI Semantic Matching (40% weight)
        if self.use_ai and text:
            ai_scores = self._ai_semantic_match(text)
            for loc, score in ai_scores.items():
                scores[loc] += score * 0.4
        
        # 2. Keyword Matching (30% weight)
        keyword_scores = self._keyword_match(text)
        for loc, score in keyword_scores.items():
            scores[loc] += score * 0.3
        
        # 3. Sound Analysis (20% weight)
        sound_scores = self._sound_match(sounds)
        for loc, score in sound_scores.items():
            scores[loc] += score * 0.2
        
        # 4. Acoustic Features (10% weight)
        if acoustic_features:
            acoustic_scores = self._acoustic_match(acoustic_features)
            for loc, score in acoustic_scores.items():
                scores[loc] += score * 0.1
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_scores:
            return self._empty_result()
        
        # Get top location
        exact_location, confidence = sorted_scores[0]
        
        # Get nearby/similar locations (top 5)
        nearby = self._get_nearby_locations(exact_location, sorted_scores[1:6])
        
        # Try to get GPS coordinates
        coordinates = None
        distance_km = None
        
        if self.use_geocoding and text:
            coordinates = self._extract_coordinates(text, exact_location)
            
            if coordinates and user_location:
                distance_km = self._calculate_distance(user_location, coordinates)
        
        return {
            "exact_location": exact_location,
            "confidence": round(confidence, 3),
            "nearby_locations": nearby,
            "all_scores": {loc: round(score, 3) for loc, score in sorted_scores[:10]},
            "coordinates": coordinates,
            "distance_km": distance_km,
            "detection_methods": {
                "ai_semantic": self.use_ai,
                "keyword_matching": True,
                "sound_analysis": len(sounds) > 0,
                "acoustic_features": acoustic_features is not None
            }
        }
    
    def _ai_semantic_match(self, text: str) -> Dict[str, float]:
        """AI-powered semantic matching"""
        if not self.model or not text:
            return {}
        
        # Encode input text
        text_embedding = self.model.encode(text)
        
        # Calculate similarity with each location
        scores = {}
        for loc_type, loc_embedding in self.location_embeddings.items():
            similarity = cosine_similarity(
                [text_embedding],
                [loc_embedding]
            )[0][0]
            
            # Convert to 0-1 range
            scores[loc_type] = max(0, min(1, similarity))
        
        return scores
    
    def _keyword_match(self, text: str) -> Dict[str, float]:
        """Keyword-based location matching"""
        if not text:
            return {}
        
        text_lower = text.lower()
        scores = {}
        
        for loc_type, info in self.LOCATION_TYPES.items():
            score = 0
            keywords = info["keywords"]
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Count occurrences
                    count = text_lower.count(keyword)
                    score += count * 0.2
            
            if score > 0:
                scores[loc_type] = min(score, 1.0)
        
        return scores
    
    def _sound_match(self, sounds: Dict[str, float]) -> Dict[str, float]:
        """Match sounds to locations"""
        sound_location_map = {
            "train": ["Railway Station", "Metro/Subway"],
            "aircraft": ["Airport Terminal"],
            "car": ["Street/Road", "Parking Area"],
            "traffic": ["Street/Road"],
            "crowd": ["Shopping Mall", "Stadium/Arena", "Market/Bazaar"],
            "music": ["Restaurant/Cafe", "Shopping Mall"],
            "announcement": ["Airport Terminal", "Railway Station"],
            "construction": ["Construction Site"],
            "siren": ["Street/Road", "Hospital"],
        }
        
        scores = defaultdict(float)
        
        for sound, confidence in sounds.items():
            sound_lower = sound.lower()
            
            for keyword, locations in sound_location_map.items():
                if keyword in sound_lower:
                    for loc in locations:
                        scores[loc] = max(scores[loc], confidence)
        
        return dict(scores)
    
    def _acoustic_match(self, features: Dict) -> Dict[str, float]:
        """Match acoustic features to locations"""
        scores = {}
        
        reverb = features.get("reverb_estimate", 0.5)
        noise_floor = features.get("noise_floor", 0.5)
        is_outdoor = features.get("is_outdoor", False)
        
        # High reverb = large indoor spaces
        if reverb > 0.6:
            scores["Airport Terminal"] = 0.3
            scores["Railway Station"] = 0.3
            scores["Shopping Mall"] = 0.2
        
        # Low reverb = small spaces or outdoor
        if reverb < 0.3:
            if is_outdoor:
                scores["Park/Outdoor"] = 0.4
                scores["Street/Road"] = 0.3
            else:
                scores["Home/Residential"] = 0.3
                scores["Office Building"] = 0.2
        
        # High noise = busy places
        if noise_floor > 0.6:
            scores["Market/Bazaar"] = 0.3
            scores["Street/Road"] = 0.3
        
        return scores
    
    def _get_nearby_locations(
        self,
        exact_location: str,
        other_scores: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        """Get nearby/similar locations"""
        nearby = []
        
        # Add related locations from exact match
        if exact_location in self.LOCATION_TYPES:
            related = self.LOCATION_TYPES[exact_location].get("related", [])
            for loc in related:
                nearby.append({
                    "location": loc,
                    "similarity": 0.8,
                    "reason": "Related location type"
                })
        
        # Add high-scoring alternatives
        for loc, score in other_scores:
            if score > 0.3 and len(nearby) < 5:
                nearby.append({
                    "location": loc,
                    "similarity": round(score, 3),
                    "reason": "High confidence match"
                })
        
        return nearby[:5]
    
    def _extract_coordinates(
        self,
        text: str,
        location_type: str
    ) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates using geocoding"""
        if not self.use_geocoding or not self.geocoder:
            return None
        
        try:
            # Look for location names in text
            location_query = f"{location_type} {text}"
            
            location = self.geocoder.geocode(location_query, timeout=5)
            
            if location:
                return (location.latitude, location.longitude)
        except:
            pass
        
        return None
    
    def _calculate_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate distance between two GPS points in kilometers"""
        try:
            return round(geodesic(point1, point2).kilometers, 2)
        except:
            return None
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "exact_location": "Unknown",
            "confidence": 0.4,
            "nearby_locations": [],
            "all_scores": {},
            "coordinates": None,
            "distance_km": None,
            "detection_methods": {
                "ai_semantic": False,
                "keyword_matching": False,
                "sound_analysis": False,
                "acoustic_features": False
            }
        }
