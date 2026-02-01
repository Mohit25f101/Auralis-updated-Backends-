# ==============================
# 📄 services/action_extractor.py
# ==============================
"""
Action Item Extraction Service
Extracts actionable items from speech
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ActionPriority(Enum):
    """Action priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Action:
    """Represents an action item"""
    action: str
    priority: str
    verb: str
    target: Optional[str]
    source_phrase: str


class ActionExtractor:
    """
    Action Item Extraction Service
    
    Extracts:
    - Required actions
    - Priority levels
    - Target locations/objects
    - Deadlines
    """
    
    # Action patterns with priority
    ACTION_PATTERNS = {
        ActionPriority.CRITICAL: [
            (r'(?:must|have\s+to)\s+([^.!?]+)', "must"),
            (r'immediately\s+([^.!?]+)', "immediately"),
            (r'(?:evacuate|leave|exit)\s+([^.!?]*(?:now|immediately)?)', "evacuate"),
            (r'(?:call|dial)\s+(911|100|101|102|108|112|emergency)', "call_emergency"),
        ],
        ActionPriority.HIGH: [
            (r'(?:should|need\s+to)\s+([^.!?]+)', "should"),
            (r'(?:please|kindly)\s+([^.!?]+)', "please"),
            (r'(?:proceed|go|move)\s+to\s+([^.!?]+)', "proceed"),
            (r'(?:board|enter)\s+([^.!?]+)', "board"),
        ],
        ActionPriority.MEDIUM: [
            (r'(?:wait|stay)\s+(?:at|for|until)\s+([^.!?]+)', "wait"),
            (r'(?:keep|remain)\s+([^.!?]+)', "keep"),
            (r'(?:collect|pick\s+up|take)\s+([^.!?]+)', "collect"),
            (r'(?:check|verify|confirm)\s+([^.!?]+)', "check"),
        ],
        ActionPriority.LOW: [
            (r'(?:may|can|could)\s+([^.!?]+)', "may"),
            (r'(?:consider|think\s+about)\s+([^.!?]+)', "consider"),
            (r'(?:remember|don\'t\s+forget)\s+([^.!?]+)', "remember"),
        ]
    }
    
    # Negative patterns (things NOT to do)
    NEGATIVE_PATTERNS = [
        (r'(?:do\s+not|don\'t|never)\s+([^.!?]+)', ActionPriority.HIGH),
        (r'(?:avoid|refrain\s+from)\s+([^.!?]+)', ActionPriority.MEDIUM),
        (r'(?:prohibited|forbidden|not\s+allowed)\s+([^.!?]*)', ActionPriority.HIGH),
    ]
    
    def __init__(self):
        self.loaded = False
    
    def load(self) -> bool:
        """Load the extractor"""
        self.loaded = True
        print("✅ Action Extractor loaded")
        return True
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract action items from text
        
        Args:
            text: Input text
            
        Returns:
            Extraction results with actions and priorities
        """
        if not self.loaded or not text:
            return self._empty_result()
        
        text_lower = text.lower()
        
        actions = []
        
        # Extract positive actions
        for priority, patterns in self.ACTION_PATTERNS.items():
            for pattern, verb in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    action = self._clean_action(match)
                    if action and len(action) > 3:
                        actions.append(Action(
                            action=action,
                            priority=priority.value,
                            verb=verb,
                            target=self._extract_target(action),
                            source_phrase=match[:100]
                        ))
        
        # Extract negative actions (things NOT to do)
        negative_actions = []
        for pattern, priority in self.NEGATIVE_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                action = self._clean_action(match)
                if action and len(action) > 3:
                    negative_actions.append({
                        "action": f"DO NOT: {action}",
                        "priority": priority.value,
                        "type": "prohibition"
                    })
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        actions.sort(key=lambda x: priority_order.get(x.priority, 4))
        
        # Convert to dict format
        action_list = [
            {
                "action": a.action,
                "priority": a.priority,
                "verb": a.verb,
                "target": a.target
            }
            for a in actions[:10]
        ]
        
        return {
            "actions": action_list,
            "negative_actions": negative_actions[:5],
            "total_actions": len(action_list),
            "has_critical": any(a["priority"] == "critical" for a in action_list),
            "has_prohibitions": len(negative_actions) > 0,
            "action_summary": self._generate_summary(action_list, negative_actions)
        }
    
    def _clean_action(self, action: str) -> str:
        """Clean and normalize action text"""
        # Remove extra whitespace
        action = ' '.join(action.split())
        
        # Remove trailing punctuation
        action = action.rstrip('.,;:!?')
        
        # Capitalize first letter
        if action:
            action = action[0].upper() + action[1:]
        
        return action
    
    def _extract_target(self, action: str) -> Optional[str]:
        """Extract target location/object from action"""
        # Common target patterns
        target_patterns = [
            r'to\s+(?:the\s+)?([^,.\s]+(?:\s+[^,.\s]+)?)',
            r'at\s+(?:the\s+)?([^,.\s]+(?:\s+[^,.\s]+)?)',
            r'platform\s+(\d+)',
            r'gate\s+([A-Z]?\d+)',
            r'terminal\s+(\d)',
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _generate_summary(
        self,
        actions: List[Dict],
        negative: List[Dict]
    ) -> str:
        """Generate action summary"""
        parts = []
        
        # Critical actions
        critical = [a for a in actions if a["priority"] == "critical"]
        if critical:
            parts.append(f"🚨 URGENT: {critical[0]['action']}")
        
        # High priority
        high = [a for a in actions if a["priority"] == "high"]
        if high:
            parts.append(f"⚠️ {high[0]['action']}")
        
        # Prohibitions
        if negative:
            parts.append(f"🚫 {negative[0]['action']}")
        
        return " | ".join(parts) if parts else "No specific actions required"
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "actions": [],
            "negative_actions": [],
            "total_actions": 0,
            "has_critical": False,
            "has_prohibitions": False,
            "action_summary": "No actions found"
        }