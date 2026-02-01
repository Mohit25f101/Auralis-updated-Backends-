# ==============================
# 📄 tests/test_semantic.py
# ==============================
"""
Tests for Semantic Analysis Components
Tests intent detection, meaning extraction, and summarization
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer service"""
    
    def test_semantic_analyzer_initialization(self):
        """Test SemanticAnalyzer initializes correctly"""
        from services.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        analyzer.load()
        
        assert analyzer.loaded == True
    
    def test_announcement_intent(self):
        """Test detection of announcement intent"""
        from services.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        analyzer.load()
        
        text = "Attention please, ladies and gentlemen. Flight AI302 is now boarding at gate 15."
        
        result = analyzer.analyze(text)
        
        assert result["intent"] == "announcement"
        assert result["intent_confidence"] > 0.7
        assert result["target_audience"] in ["passengers", "general_public"]
    
    def test_instruction_intent(self):
        """Test detection of instruction intent"""
        from services.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        analyzer.load()
        
        text = "Please proceed to platform 3 immediately. You must board the train now."
        
        result = analyzer.analyze(text)
        
        assert result["intent"] == "instruction"
        assert len(result["action_items"]) > 0
    
    def test_question_intent(self):
        """Test detection of question intent"""
        from services.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        analyzer.load()
        
        text = "Where is the ticket counter? Can you help me find platform 5?"
        
        result = analyzer.analyze(text)
        
        assert result["intent"] == "question"
        assert result["has_questions"] == True
        assert len(result["questions_asked"]) > 0
    
    def test_emergency_intent(self):
        """Test detection of emergency intent"""
        from services.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        analyzer.load()
        
        text = "Emergency! Fire in the terminal! Call 911 immediately! Everyone evacuate now!"
        
        result = analyzer.analyze(text)
        
        assert result["intent"] == "emergency"
        assert result["urgency_level"] == "critical"
        assert "emergency" in result["topics"] or result["intent"] == "emergency"
    
    def test_meaning_extraction(self):
        """Test extraction of meaning"""
        from services.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        analyzer.load()
        
        text = "Train number 12345 Rajdhani Express arriving at platform 3 at 10:30 AM"        
        
        result = analyzer.analyze(text)
        
        assert "what_they_mean" in result
        assert len(result["what_they_mean"]) > 10
        
        # Updated assertion - check for meaningful content rather than specific keywords
        what_they_mean_lower = result["what_they_mean"].lower()
        assert "platform" in what_they_mean_lower or "timing" in what_they_mean_lower
        assert "10:30" in result["what_they_mean"]

    def test_key_points_extraction(self):
        """Test extraction of key points"""
        from services.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        analyzer.load()
        
        text = """
        Attention please. Train number 12345 is arriving at platform 3. 
        Passengers are requested to keep their luggage ready. 
        Please do not cross the yellow line. 
        Thank you for traveling with us.
        """
        
        result = analyzer.analyze(text)
        
        assert len(result["key_points"]) > 0
        assert len(result["key_points"]) <= 5
    
    def test_listener_recommendations(self):
        """Test listener recommendations generation"""
        from services.semantic_analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        analyzer.load()
        
        text = "Please proceed to gate 15 for boarding. Your flight will depart in 30 minutes."
        
        result = analyzer.analyze(text)
        
        assert "listener_should" in result
        assert len(result["listener_should"]) > 0


class TestIntentDetector:
    """Tests for IntentDetector service"""
    
    def test_intent_detector_initialization(self):
        """Test IntentDetector initializes correctly"""
        from services.intent_detector import IntentDetector
        
        detector = IntentDetector()
        detector.load()
        
        assert detector.loaded == True
    
    def test_warning_intent(self):
        """Test warning intent detection"""
        from services.intent_detector import IntentDetector
        
        detector = IntentDetector()
        detector.load()
        
        text = "Warning! Caution! For your safety, please stand behind the yellow line."
        
        result = detector.detect(text)
        
        assert result["primary_intent"] == "warning"
        assert result["is_urgent"] == True
    
    def test_greeting_intent(self):
        """Test greeting intent detection"""
        from services.intent_detector import IntentDetector
        
        detector = IntentDetector()
        detector.load()
        
        text = "Good morning everyone! Welcome to Mumbai Central station."
        
        result = detector.detect(text)
        
        assert result["primary_intent"] == "greeting"
    
    def test_farewell_intent(self):
        """Test farewell intent detection"""
        from services.intent_detector import IntentDetector
        
        detector = IntentDetector()
        detector.load()
        
        text = "Thank you for traveling with us. Have a safe journey. Goodbye!"
        
        result = detector.detect(text)
        
        assert result["primary_intent"] == "farewell"


class TestSummarizer:
    """Tests for Summarizer service"""
    
    def test_summarizer_initialization(self):
        """Test Summarizer initializes correctly"""
        from services.summarizer import Summarizer
        
        summarizer = Summarizer()
        summarizer.load()
        
        assert summarizer.loaded == True
    
    def test_one_line_summary(self):
        """Test one-line summary generation"""
        from services.summarizer import Summarizer
        
        summarizer = Summarizer()
        summarizer.load()
        
        text = """
        Attention all passengers. This is an important announcement. 
        Train number 12345 Rajdhani Express from New Delhi to Mumbai 
        is now arriving at platform number 3. Passengers are requested 
        to proceed to the platform with their valid tickets.
        """
        
        result = summarizer.summarize(text, "announcement")
        
        assert "one_line_summary" in result
        assert len(result["one_line_summary"]) < 150
    
    def test_tldr_generation(self):
        """Test TL;DR generation"""
        from services.summarizer import Summarizer
        
        summarizer = Summarizer()
        summarizer.load()
        
        text = "Flight AI302 to London departing from gate 15 at 2:30 PM"
        entities = {
            "flight_numbers": ["Flight AI302"],
            "gate_numbers": ["Gate 15"],
            "times": ["2:30 PM"]
        }
        
        result = summarizer.summarize(text, "announcement", entities)
        
        assert "tldr" in result
        assert len(result["tldr"]) < 100


class TestTopicClassifier:
    """Tests for TopicClassifier service"""
    
    def test_topic_classifier_initialization(self):
        """Test TopicClassifier initializes correctly"""
        from services.topic_classifier import TopicClassifier
        
        classifier = TopicClassifier()
        classifier.load()
        
        assert classifier.loaded == True
    
    def test_travel_topic_detection(self):
        """Test travel topic detection"""
        from services.topic_classifier import TopicClassifier
        
        classifier = TopicClassifier()
        classifier.load()
        
        text = "The train will depart from platform 3. Please board the coach."
        
        result = classifier.classify(text)
        
        assert "travel" in result["topics"] or "transport" in result["topics"]
    
    def test_multiple_topics(self):
        """Test multiple topic detection"""
        from services.topic_classifier import TopicClassifier
        
        classifier = TopicClassifier()
        classifier.load()
        
        text = "For your safety during the journey, please keep your luggage secure. Food is available in the dining car."
        
        result = classifier.classify(text)
        
        assert len(result["topics"]) >= 2


# ==============================
# RUN TESTS
# ==============================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])