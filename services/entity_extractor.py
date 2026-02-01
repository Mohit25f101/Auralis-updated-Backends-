# ==============================
# 📄 services/entity_extractor.py
# ==============================
"""
Named Entity Extraction Service
Extracts structured information from speech

EXTRACTS:
- Numbers (train/flight numbers, platform numbers, etc.)
- Times and dates
- Locations and places
- Person names
- Organizations
- Money/prices
- Contact information
- Durations
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Entity:
    """Represents an extracted entity"""
    type: str
    value: str
    original: str
    confidence: float
    context: str


class EntityExtractor:
    """
    Named Entity Extraction Service
    
    Specialized for:
    - Transport announcements (train/flight/bus numbers)
    - Location references (platforms, gates, terminals)
    - Time expressions
    - Indian/Asian naming conventions
    """
    
    # Entity patterns
    PATTERNS = {
        # ═══ NUMBERS ═══
        "train_number": [
            (r'\b(?:train\s+(?:number|no\.?|#)?)\s*(\d{4,5})\b', "Train {}"),
            (r'\b(\d{5})\s+(?:express|mail|superfast|rajdhani|shatabdi|duronto)\b', "Train {}"),
        ],
        "flight_number": [
            (r'\b(?:flight\s+(?:number|no\.?|#)?)\s*([A-Z]{2}\s*\d{3,4})\b', "Flight {}"),
            (r'\b([A-Z]{2}\s*\d{3,4})\s+(?:to|from|flight)\b', "Flight {}"),
            (r'\b(indigo|spicejet|air\s*india|vistara|emirates)\s+(\d{3,4})\b', "{} {}"),
        ],
        "platform_number": [
            (r'\b(?:platform|track)\s+(?:number|no\.?)?\s*(\d{1,2})\b', "Platform {}"),
        ],
        "gate_number": [
            (r'\b(?:gate|boarding\s+gate)\s+(?:number|no\.?)?\s*([A-Z]?\d{1,3})\b', "Gate {}"),
        ],
        "terminal_number": [
            (r'\b(?:terminal)\s+(?:number|no\.?)?\s*(\d{1})\b', "Terminal {}"),
        ],
        "coach_number": [
            (r'\b(?:coach|car|bogie|bogey)\s+(?:number|no\.?)?\s*([A-Z]?\d{1,2})\b', "Coach {}"),
        ],
        "seat_number": [
            (r'\b(?:seat)\s+(?:number|no\.?)?\s*(\d{1,3}[A-Z]?)\b', "Seat {}"),
        ],
        "generic_number": [
            (r'\b(?:number|no\.?|#)\s*(\d+)\b', "Number {}"),
        ],
        
        # ═══ TIME ═══
        "time_12h": [
            (r'\b(\d{1,2}[:.]\d{2}\s*(?:am|pm|a\.m\.|p\.m\.))\b', "{}"),
            (r'\b(\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.))\b', "{}"),
        ],
        "time_24h": [
            (r'\b(\d{2}[:.]\d{2})\s*(?:hours?|hrs?)?\b', "{} hours"),
            (r'\b(\d{4})\s*(?:hours?|hrs?)\b', "{} hours"),
        ],
        "duration": [
            (r'\b(\d+)\s*(?:minutes?|mins?)\b', "{} minutes"),
            (r'\b(\d+)\s*(?:hours?|hrs?)\b', "{} hours"),
            (r'\b(\d+)\s*(?:seconds?|secs?)\b', "{} seconds"),
            (r'\b(\d+)\s*(?:days?)\b', "{} days"),
        ],
        "delay": [
            (r'(?:delayed?\s+(?:by)?)\s*(\d+\s*(?:minutes?|mins?|hours?|hrs?))', "Delayed by {}"),
            (r'(\d+\s*(?:minutes?|mins?|hours?|hrs?))\s*(?:late|delay)', "{} late"),
        ],
        
        # ═══ DATE ═══
        "date": [
            (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', "Date: {}"),
            (r'\b(\d{1,2})\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', "Date: {} of month"),
            (r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b', "Date: {} of month"),
        ],
        
        # ═══ LOCATIONS ═══
        "station": [
            (r'\b(\w+(?:\s+\w+)?)\s+(?:station|junction|terminal)\b', "{} Station"),
        ],
        "airport": [
            (r'\b(\w+(?:\s+\w+)?)\s+(?:airport|airfield)\b', "{} Airport"),
        ],
        "city": [
            (r'\b(?:to|from|at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', "City: {}"),
        ],
        
        # ═══ MONEY ═══
        "money_inr": [
            (r'\b(?:₹|rs\.?|inr)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b', "₹{}"),
            (r'\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rupees?|rs\.?)\b', "₹{}"),
        ],
        "money_usd": [
            (r'\b(?:\$|usd)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b', "${}"),
        ],
        
        # ═══ CONTACT ═══
        "phone": [
            (r'\b(\d{10})\b', "Phone: {}"),
            (r'\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b', "Phone: {}"),
            (r'\b(?:call|dial|phone)\s*[:.]?\s*(\d+)\b', "Phone: {}"),
        ],
        "emergency_number": [
            (r'\b(100|101|102|108|112|911)\b', "Emergency: {}"),
        ],
        
        # ═══ NAMES (Indian context) ═══
        "title_name": [
            (r'\b((?:mr\.?|mrs\.?|ms\.?|dr\.?|shri|smt\.?|kumar)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', "{}"),
        ],
    }
    
    # Indian/Asian name patterns
    INDIAN_NAME_PATTERNS = [
        r'\b(Shri|Smt\.?|Kumar|Sharma|Singh|Patel|Gupta|Kumar|Verma|Joshi|Rao|Reddy|Nair|Menon|Iyer)\b',
    ]
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the entity extractor"""
        self.loaded = True
        print("✅ Entity Extractor loaded")
        return True
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with categorized entities
        """
        if not self.loaded or not text:
            return self._empty_result()
        
        entities = {
            # Transport
            "train_numbers": [],
            "flight_numbers": [],
            "platform_numbers": [],
            "gate_numbers": [],
            "terminal_numbers": [],
            "coach_numbers": [],
            "seat_numbers": [],
            
            # Time
            "times": [],
            "durations": [],
            "delays": [],
            "dates": [],
            
            # Location
            "stations": [],
            "airports": [],
            "cities": [],
            
            # Other
            "money": [],
            "phone_numbers": [],
            "emergency_numbers": [],
            "person_names": [],
            "generic_numbers": [],
        }
        
        # Extract using patterns
        for entity_type, patterns in self.PATTERNS.items():
            for pattern, template in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        value = template.format(*match)
                    else:
                        value = template.format(match)
                    
                    # Map to output category
                    category = self._map_to_category(entity_type)
                    if category and value not in entities[category]:
                        entities[category].append(value)
        
        # Extract names
        entities["person_names"] = self._extract_names(text)
        
        # Generate summary
        summary = self._generate_summary(entities)
        
        return {
            "entities": entities,
            "summary": summary,
            "total_entities": sum(len(v) for v in entities.values()),
            "has_transport_info": bool(
                entities["train_numbers"] or 
                entities["flight_numbers"] or 
                entities["platform_numbers"]
            ),
            "has_time_info": bool(entities["times"] or entities["durations"]),
            "has_contact_info": bool(entities["phone_numbers"]),
            "has_location_info": bool(
                entities["stations"] or 
                entities["airports"] or 
                entities["cities"]
            ),
        }
    
    def _map_to_category(self, entity_type: str) -> Optional[str]:
        """Map entity type to output category"""
        mapping = {
            "train_number": "train_numbers",
            "flight_number": "flight_numbers",
            "platform_number": "platform_numbers",
            "gate_number": "gate_numbers",
            "terminal_number": "terminal_numbers",
            "coach_number": "coach_numbers",
            "seat_number": "seat_numbers",
            "time_12h": "times",
            "time_24h": "times",
            "duration": "durations",
            "delay": "delays",
            "date": "dates",
            "station": "stations",
            "airport": "airports",
            "city": "cities",
            "money_inr": "money",
            "money_usd": "money",
            "phone": "phone_numbers",
            "emergency_number": "emergency_numbers",
            "title_name": "person_names",
            "generic_number": "generic_numbers",
        }
        return mapping.get(entity_type)
    
    def _extract_names(self, text: str) -> List[str]:
        """Extract person names"""
        names = []
        
        # Title + Name pattern
        title_pattern = r'\b((?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Shri|Smt\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        matches = re.findall(title_pattern, text)
        names.extend(matches)
        
        # Passenger pattern
        passenger_pattern = r'\b(?:passenger|pax)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        matches = re.findall(passenger_pattern, text, re.IGNORECASE)
        names.extend(matches)
        
        return list(set(names))[:5]
    
    def _generate_summary(self, entities: Dict) -> str:
        """Generate entity summary"""
        parts = []
        
        if entities["train_numbers"]:
            parts.append(f"🚂 Train: {entities['train_numbers'][0]}")
        if entities["flight_numbers"]:
            parts.append(f"✈️ Flight: {entities['flight_numbers'][0]}")
        if entities["platform_numbers"]:
            parts.append(f"🚏 {entities['platform_numbers'][0]}")
        if entities["gate_numbers"]:
            parts.append(f"🚪 {entities['gate_numbers'][0]}")
        if entities["times"]:
            parts.append(f"🕐 Time: {entities['times'][0]}")
        if entities["delays"]:
            parts.append(f"⏰ {entities['delays'][0]}")
        
        return " | ".join(parts) if parts else "No key entities found"
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "entities": {
                "train_numbers": [],
                "flight_numbers": [],
                "platform_numbers": [],
                "gate_numbers": [],
                "terminal_numbers": [],
                "coach_numbers": [],
                "seat_numbers": [],
                "times": [],
                "durations": [],
                "delays": [],
                "dates": [],
                "stations": [],
                "airports": [],
                "cities": [],
                "money": [],
                "phone_numbers": [],
                "emergency_numbers": [],
                "person_names": [],
                "generic_numbers": [],
            },
            "summary": "No entities found",
            "total_entities": 0,
            "has_transport_info": False,
            "has_time_info": False,
            "has_contact_info": False,
            "has_location_info": False,
        }