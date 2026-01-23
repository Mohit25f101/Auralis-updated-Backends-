# ==============================
# 📄 inference/active_learning.py
# ==============================
# Active Learning Pipeline
# Upgrade 7: Continuous model improvement
# ==============================

import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABEL_CONFIG, TRAINING_CONFIG


class UncertaintySampler:
    """
    Uncertainty-based sample selection for active learning.
    """
    
    def __init__(self):
        self.methods = {
            'entropy': self._entropy_uncertainty,
            'margin': self._margin_uncertainty,
            'least_confidence': self._least_confidence,
            'bald': self._bald_uncertainty
        }
    
    def _entropy_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Entropy-based uncertainty."""
        entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=-1)
        max_entropy = np.log(predictions.shape[-1])
        return entropy / max_entropy
    
    def _margin_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Margin-based uncertainty (1 - margin between top 2)."""
        sorted_preds = np.sort(predictions, axis=-1)
        margin = sorted_preds[:, -1] - sorted_preds[:, -2]
        return 1 - margin
    
    def _least_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Least confidence uncertainty."""
        return 1 - np.max(predictions, axis=-1)
    
    def _bald_uncertainty(
        self,
        predictions_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Bayesian Active Learning by Disagreement.
        Requires multiple forward passes with dropout.
        """
        predictions = np.stack(predictions_list, axis=0)
        
        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)
        
        # Predictive entropy
        pred_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)
        
        # Expected entropy
        entropies = -np.sum(predictions * np.log(predictions + 1e-10), axis=-1)
        expected_entropy = np.mean(entropies, axis=0)
        
        # BALD = predictive entropy - expected entropy
        return pred_entropy - expected_entropy
    
    def compute_uncertainty(
        self,
        predictions: np.ndarray,
        method: str = 'entropy'
    ) -> np.ndarray:
        """
        Compute uncertainty scores.
        
        Args:
            predictions: Model predictions [batch, classes]
            method: Uncertainty method
            
        Returns:
            Uncertainty scores [batch]
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        return self.methods[method](predictions)
    
    def combined_uncertainty(
        self,
        predictions: np.ndarray,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Combine multiple uncertainty measures.
        
        Args:
            predictions: Model predictions
            weights: Weights for each method
            
        Returns:
            Combined uncertainty scores
        """
        if weights is None:
            weights = {'entropy': 0.4, 'margin': 0.3, 'least_confidence': 0.3}
        
        combined = np.zeros(len(predictions))
        for method, weight in weights.items():
            combined += weight * self.compute_uncertainty(predictions, method)
        
        return combined


class FeedbackCollector:
    """
    Collect and manage user feedback for active learning.
    """
    
    def __init__(self, storage_path: str = 'feedback'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.feedback_file = self.storage_path / 'feedback_log.json'
        self.feedback_data = self._load_feedback()
        
    def _load_feedback(self) -> List[Dict]:
        """Load existing feedback."""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback(self):
        """Save feedback to file."""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def add_feedback(
        self,
        sample_id: str,
        prediction: Dict,
        correct_label: Dict = None,
        is_correct: bool = None,
        notes: str = None
    ):
        """
        Add user feedback for a prediction.
        
        Args:
            sample_id: Unique identifier for the sample
            prediction: Model's prediction
            correct_label: Correct label (if provided)
            is_correct: Whether prediction was correct
            notes: Additional notes
        """
        feedback_entry = {
            'sample_id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'correct_label': correct_label,
            'is_correct': is_correct,
            'notes': notes
        }
        
        self.feedback_data.append(feedback_entry)
        self._save_feedback()
    
    def get_corrections(self) -> List[Dict]:
        """Get all corrections (wrong predictions)."""
        return [
            f for f in self.feedback_data
            if f.get('is_correct') == False and f.get('correct_label')
        ]
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics."""
        total = len(self.feedback_data)
        correct = sum(1 for f in self.feedback_data if f.get('is_correct'))
        incorrect = sum(1 for f in self.feedback_data if f.get('is_correct') == False)
        
        return {
            'total_feedback': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': correct / total if total > 0 else 0,
            'corrections_available': len(self.get_corrections())
        }


class ActiveLearningPipeline:
    """
    Complete active learning pipeline for continuous improvement.
    
    Workflow:
    1. Model makes predictions
    2. Uncertain predictions are flagged for review
    3. User provides feedback
    4. Model is retrained with new labeled data
    """
    
    def __init__(
        self,
        model,
        uncertainty_threshold: float = None,
        storage_path: str = 'active_learning'
    ):
        self.model = model
        self.threshold = uncertainty_threshold or TRAINING_CONFIG.uncertainty_threshold
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.sampler = UncertaintySampler()
        self.feedback = FeedbackCollector(str(self.storage_path / 'feedback'))
        
        # Prediction log
        self.prediction_log = []
        self.unlabeled_pool = []
        
    def predict_with_uncertainty(
        self,
        inputs: np.ndarray,
        return_details: bool = False
    ) -> Dict:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            inputs: Input features
            return_details: Whether to return detailed uncertainty info
            
        Returns:
            Predictions with uncertainty
        """
        # Get predictions
        predictions = self.model.predict(inputs, verbose=0)
        
        # Compute uncertainty for each output head
        uncertainties = {}
        needs_review = False
        
        for key, pred in predictions.items():
            if len(pred.shape) > 1 and pred.shape[-1] > 1:
                # Classification output
                unc = self.sampler.combined_uncertainty(pred)
                uncertainties[key] = float(unc[0]) if len(unc) == 1 else unc.tolist()
                
                if np.any(unc > self.threshold):
                    needs_review = True
        
        result = {
            'predictions': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in predictions.items()},
            'uncertainty': uncertainties,
            'needs_review': needs_review,
            'confidence': 1 - np.mean(list(uncertainties.values())) if uncertainties else 1.0
        }
        
        # Log prediction
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'result': result,
            'reviewed': False
        }
        self.prediction_log.append(log_entry)
        
        if needs_review:
            self.unlabeled_pool.append({
                'inputs': inputs.tolist() if isinstance(inputs, np.ndarray) else inputs,
                'prediction': result
            })
        
        if return_details:
            result['uncertainty_breakdown'] = {
                key: {
                    'entropy': float(self.sampler._entropy_uncertainty(pred)[0]),
                    'margin': float(self.sampler._margin_uncertainty(pred)[0]),
                    'least_conf': float(self.sampler._least_confidence(pred)[0])
                }
                for key, pred in predictions.items()
                if len(pred.shape) > 1 and pred.shape[-1] > 1
            }
        
        return result
    
    def select_for_labeling(
        self,
        n_samples: int = 10,
        strategy: str = 'uncertainty'
    ) -> List[Dict]:
        """
        Select most informative samples for labeling.
        
        Args:
            n_samples: Number of samples to select
            strategy: Selection strategy
            
        Returns:
            List of samples needing labels
        """
        if not self.unlabeled_pool:
            return []
        
        if strategy == 'uncertainty':
            # Sort by uncertainty (highest first)
            sorted_pool = sorted(
                self.unlabeled_pool,
                key=lambda x: 1 - x['prediction']['confidence'],
                reverse=True
            )
            return sorted_pool[:n_samples]
        
        elif strategy == 'diverse':
            # TODO: Implement diversity sampling
            return self.unlabeled_pool[:n_samples]
        
        elif strategy == 'random':
            indices = np.random.choice(
                len(self.unlabeled_pool),
                size=min(n_samples, len(self.unlabeled_pool)),
                replace=False
            )
            return [self.unlabeled_pool[i] for i in indices]
        
        return self.unlabeled_pool[:n_samples]
    
    def add_labeled_sample(
        self,
        sample_id: str,
        inputs: Any,
        labels: Dict,
        prediction: Dict = None
    ):
        """
        Add a newly labeled sample.
        
        Args:
            sample_id: Unique sample identifier
            inputs: Input data
            labels: Correct labels
            prediction: Original prediction (if available)
        """
        # Record feedback
        is_correct = None
        if prediction:
            # Compare prediction with labels
            pred_loc = np.argmax(prediction.get('location', [0]))
            true_loc = np.argmax(labels.get('location', [0]))
            is_correct = pred_loc == true_loc
        
        self.feedback.add_feedback(
            sample_id=sample_id,
            prediction=prediction,
            correct_label=labels,
            is_correct=is_correct
        )
        
        # Store for retraining
        labeled_path = self.storage_path / 'labeled_samples.json'
        
        labeled_samples = []
        if labeled_path.exists():
            with open(labeled_path, 'r') as f:
                labeled_samples = json.load(f)
        
        labeled_samples.append({
            'sample_id': sample_id,
            'inputs': inputs if isinstance(inputs, list) else inputs.tolist(),
            'labels': labels,
            'timestamp': datetime.now().isoformat()
        })
        
        with open(labeled_path, 'w') as f:
            json.dump(labeled_samples, f)
    
    def get_retraining_data(self) -> Tuple[List, List]:
        """
        Get accumulated labeled data for retraining.
        
        Returns:
            Tuple of (inputs, labels)
        """
        labeled_path = self.storage_path / 'labeled_samples.json'
        
        if not labeled_path.exists():
            return [], []
        
        with open(labeled_path, 'r') as f:
            labeled_samples = json.load(f)
        
        inputs = [s['inputs'] for s in labeled_samples]
        labels = [s['labels'] for s in labeled_samples]
        
        return inputs, labels
    
    def retrain_model(
        self,
        epochs: int = 5,
        learning_rate: float = 1e-5
    ):
        """
        Retrain model with accumulated labeled data.
        
        Args:
            epochs: Number of fine-tuning epochs
            learning_rate: Fine-tuning learning rate
        """
        import tensorflow as tf
        
        inputs, labels = self.get_retraining_data()
        
        if len(inputs) < 10:
            print("⚠️ Not enough labeled samples for retraining (need at least 10)")
            return
        
        print(f"\n🔄 Retraining with {len(inputs)} new samples...")
        
        # Convert to tensors
        inputs_array = np.array(inputs)
        labels_dict = {
            key: np.array([l[key] for l in labels])
            for key in labels[0].keys()
        }
        
        # Compile with low learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=self.model.loss if hasattr(self.model, 'loss') else 'categorical_crossentropy'
        )
        
        # Fine-tune
        self.model.fit(
            inputs_array,
            labels_dict,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        print("✅ Retraining complete!")
    
    def get_improvement_stats(self) -> Dict:
        """Get statistics on model improvement over time."""
        stats = self.feedback.get_statistics()
        
        # Calculate accuracy trend
        if len(self.feedback.feedback_data) > 0:
            window_size = 50
            windows = []
            
            for i in range(0, len(self.feedback.feedback_data), window_size):
                window = self.feedback.feedback_data[i:i + window_size]
                correct = sum(1 for f in window if f.get('is_correct'))
                windows.append(correct / len(window) if window else 0)
            
            stats['accuracy_trend'] = windows
            stats['improvement'] = windows[-1] - windows[0] if len(windows) > 1 else 0
        
        stats['unlabeled_pool_size'] = len(self.unlabeled_pool)
        stats['total_predictions'] = len(self.prediction_log)
        
        return stats
    
    def save_state(self):
        """Save pipeline state."""
        state = {
            'prediction_log': self.prediction_log[-1000:],  # Keep last 1000
            'unlabeled_pool': self.unlabeled_pool[-500:],   # Keep last 500
            'threshold': self.threshold
        }
        
        state_path = self.storage_path / 'pipeline_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f)
    
    def load_state(self):
        """Load pipeline state."""
        state_path = self.storage_path / 'pipeline_state.json'
        
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.prediction_log = state.get('prediction_log', [])
            self.unlabeled_pool = state.get('unlabeled_pool', [])
            self.threshold = state.get('threshold', self.threshold)


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Active Learning")
    print("="*60)
    
    # Test uncertainty sampler
    print("\n🔧 Testing UncertaintySampler...")
    sampler = UncertaintySampler()
    
    # Confident prediction
    confident = np.array([[0.9, 0.05, 0.05]])
    print(f"   Confident: entropy={sampler._entropy_uncertainty(confident)[0]:.3f}")
    
    # Uncertain prediction
    uncertain = np.array([[0.33, 0.33, 0.34]])
    print(f"   Uncertain: entropy={sampler._entropy_uncertainty(uncertain)[0]:.3f}")
    
    # Test feedback collector
    print("\n🔧 Testing FeedbackCollector...")
    collector = FeedbackCollector('test_feedback')
    collector.add_feedback(
        sample_id='test_001',
        prediction={'location': 'Airport'},
        is_correct=True
    )
    print(f"   Statistics: {collector.get_statistics()}")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_feedback', ignore_errors=True)
    
    print("\n✅ Active Learning test passed!")