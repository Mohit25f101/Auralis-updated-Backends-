# ==============================
# 📄 tests/test_entities.py
# ==============================
"""
Tests for Entity Extraction Components
Tests extraction of numbers, times, locations, names, etc.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEntityExtractor:
    """Tests for EntityExtractor service"""
    
    def test_entity_extractor_initialization(self):
        """Test EntityExtractor initializes correctly"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        assert extractor.loaded == True
    
    def test_train_number_extraction(self):
        """Test train number extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "Train number 12345 Rajdhani Express is arriving"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["train_numbers"]) > 0
        assert "12345" in str(result["entities"]["train_numbers"])
    
    def test_flight_number_extraction(self):
        """Test flight number extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "Flight AI302 is now boarding at gate 15"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["flight_numbers"]) > 0
        assert "AI302" in str(result["entities"]["flight_numbers"])
    
    def test_platform_number_extraction(self):
        """Test platform number extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "Please proceed to platform number 3"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["platform_numbers"]) > 0
        assert "3" in str(result["entities"]["platform_numbers"])
    
    def test_time_extraction_12h(self):
        """Test 12-hour time extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "The train will arrive at 10:30 AM"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["times"]) > 0
        assert "10:30" in str(result["entities"]["times"])
    
    def test_time_extraction_24h(self):
        """Test 24-hour time extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "Departure scheduled at 1430 hours"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["times"]) > 0
    
    def test_duration_extraction(self):
        """Test duration extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "The journey will take approximately 5 hours and 30 minutes"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["durations"]) > 0
    
    def test_delay_extraction(self):
        """Test delay extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "The flight has been delayed by 45 minutes"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["delays"]) > 0
        assert "45" in str(result["entities"]["delays"])
    
    def test_money_extraction_inr(self):
        """Test Indian Rupee extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "The ticket costs Rs. 500 or ₹500"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["money"]) > 0
    
    def test_phone_number_extraction(self):
        """Test phone number extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "For assistance, call 9876543210 or dial 1800-123-4567"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["phone_numbers"]) > 0
    
    def test_emergency_number_extraction(self):
        """Test emergency number extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "In case of emergency, dial 100 or 112"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["emergency_numbers"]) > 0
    
    def test_person_name_extraction(self):
        """Test person name extraction"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "Mr. Sharma and Dr. Patel are requested to proceed to the information desk"
        
        result = extractor.extract(text)
        
        assert len(result["entities"]["person_names"]) > 0
    
    def test_has_transport_info_flag(self):
        """Test transport info flag"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "Train 12345 arriving at platform 3"
        
        result = extractor.extract(text)
        
        assert result["has_transport_info"] == True
    
    def test_summary_generation(self):
        """Test entity summary generation"""
        from services.entity_extractor import EntityExtractor
        
        extractor = EntityExtractor()
        extractor.load()
        
        text = "Train 12345 arriving at platform 3 at 10:30 AM"
        
        result = extractor.extract(text)
        
        assert "summary" in result
        assert len(result["summary"]) > 5


class TestQuestionDetector:
    """Tests for QuestionDetector service"""
    
    def test_question_detector_initialization(self):
        """Test QuestionDetector initializes correctly"""
        from services.question_detector import QuestionDetector
        
        detector = QuestionDetector()
        detector.load()
        
        assert detector.loaded == True
    
    def test_explicit_question_detection(self):
        """Test explicit question detection (with ?)"""
        from services.question_detector import QuestionDetector
        
        detector = QuestionDetector()
        detector.load()
        
        text = "Where is the ticket counter? What time does the train arrive?"
        
        result = detector.detect(text)
        
        assert result["has_questions"] == True
        assert result["question_count"] >= 2
    
    def test_implicit_question_detection(self):
        """Test implicit question detection"""
        from services.question_detector import QuestionDetector
        
        detector = QuestionDetector()
        detector.load()
        
        text = "I wonder if you could tell me where platform 3 is"
        
        result = detector.detect(text)
        
        assert result["has_questions"] == True
    
    def test_question_type_classification(self):
        """Test question type classification"""
        from services.question_detector import QuestionDetector
        
        detector = QuestionDetector()
        detector.load()
        
        text = "Where is the exit? When does the train leave?"
        
        result = detector.detect(text)
        
        assert "where" in result["question_types"] or "location" in str(result)
        assert "when" in result["question_types"] or "time" in str(result)


class TestActionExtractor:
    """Tests for ActionExtractor service"""
    
    def test_action_extractor_initialization(self):
        """Test ActionExtractor initializes correctly"""
        from services.action_extractor import ActionExtractor
        
        extractor = ActionExtractor()
        extractor.load()
        
        assert extractor.loaded == True
    
    def test_please_action_extraction(self):
        """Test 'please' action extraction"""
        from services.action_extractor import ActionExtractor
        
        extractor = ActionExtractor()
        extractor.load()
        
        text = "Please proceed to platform 3. Kindly keep your tickets ready."
        
        result = extractor.extract(text)
        
        assert len(result["actions"]) > 0
    
    def test_must_should_extraction(self):
        """Test 'must/should' action extraction"""
        from services.action_extractor import ActionExtractor
        
        extractor = ActionExtractor()
        extractor.load()
        
        text = "Passengers must board the train now. You should keep your luggage secure."
        
        result = extractor.extract(text)
        
        assert len(result["actions"]) > 0
    
    def test_action_priority(self):
        """Test action priority assignment"""
        from services.action_extractor import ActionExtractor
        
        extractor = ActionExtractor()
        extractor.load()
        
        text = "You must evacuate immediately. Please keep calm."
        
        result = extractor.extract(text)
        
        assert any(a["priority"] == "high" for a in result["actions"])


# ==============================
# RUN TESTS
# ==============================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])