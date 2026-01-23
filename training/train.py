# ==============================
# 📄 training/train.py
# ==============================
# Main Training Script for Auralis
# Orchestrates the complete training pipeline
# ==============================

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np

from config import (
    AUDIO_CONFIG, LABEL_CONFIG, MODEL_CONFIG, 
    TRAINING_CONFIG, WEIGHTS_DIR, LOGS_DIR, DATASET_DIR
)


def setup_gpu():
    """Configure GPU settings."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")


def load_datasets(data_dir: str, batch_size: int):
    """
    Load training and validation datasets.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from data.data_loader import AuralisDataLoader
    
    loader = AuralisDataLoader(data_dir)
    
    train_dataset = loader.load_tfrecord_dataset(
        split='train',
        batch_size=batch_size,
        shuffle=True,
        augment=True
    )
    
    val_dataset = loader.load_tfrecord_dataset(
        split='val',
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    return train_dataset, val_dataset


def create_model(model_type: str = 'full'):
    """
    Create model based on type.
    
    Args:
        model_type: 'full', 'transformer', 'fusion', 'compact'
        
    Returns:
        Model instance
    """
    if model_type == 'full':
        from models.multimodal_fusion import MultiModalFusionNetwork
        return MultiModalFusionNetwork()
    
    elif model_type == 'transformer':
        from models.audio_transformer import AudioTransformerEncoder
        from models.scene_classifier import AuralisSceneClassifier
        
        encoder = AudioTransformerEncoder()
        return AuralisSceneClassifier()
    
    elif model_type == 'compact':
        from training.knowledge_distillation import create_compact_student_model
        return create_compact_student_model()
    
    else:
        from models.scene_classifier import AuralisSceneClassifier
        return AuralisSceneClassifier()


def train_phase1_contrastive(
    audio_files: list,
    epochs: int = 50,
    batch_size: int = 64
):
    """
    Phase 1: Contrastive pre-training.
    """
    print("\n" + "="*60)
    print("📚 PHASE 1: Contrastive Pre-training")
    print("="*60)
    
    from models.audio_transformer import AudioTransformerEncoder, MelSpectrogramExtractor
    from training.audio_augmentation import AudioAugmentor
    from training.contrastive_learning import pretrain_contrastive
    
    # Create encoder
    encoder = AudioTransformerEncoder()
    
    # Create augmentor and extractor
    augmentor = AudioAugmentor()
    mel_extractor = MelSpectrogramExtractor()
    
    # Pre-train
    encoder = pretrain_contrastive(
        audio_files=audio_files,
        encoder=encoder,
        augmentor=augmentor,
        mel_extractor=mel_extractor,
        epochs=epochs,
        batch_size=batch_size,
        save_path=str(WEIGHTS_DIR / 'encoder_pretrained.keras')
    )
    
    return encoder


def train_phase2_supervised(
    train_dataset,
    val_dataset,
    encoder=None,
    epochs: int = 100
):
    """
    Phase 2: Supervised training with pre-trained encoder.
    """
    print("\n" + "="*60)
    print("🎯 PHASE 2: Supervised Training")
    print("="*60)
    
    from models.scene_classifier import AuralisSceneClassifier, SceneClassifierTrainer
    from training.custom_losses import AuralisLoss
    
    # Create model
    model = AuralisSceneClassifier()
    
    # Load pre-trained encoder if available
    if encoder is not None:
        print("📥 Loading pre-trained encoder weights...")
        # Transfer relevant weights
    
    # Create trainer
    trainer = SceneClassifierTrainer(model)
    trainer.compile_model(
        learning_rate=TRAINING_CONFIG.learning_rate,
        use_focal_loss=True
    )
    
    # Train
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        save_path=str(WEIGHTS_DIR / 'auralis_scene_best.keras')
    )
    
    return model, history


def train_phase3_distillation(
    teacher_model,
    train_dataset,
    val_dataset,
    student_size: str = 'compact',
    epochs: int = 50
):
    """
    Phase 3: Knowledge distillation for deployment.
    """
    print("\n" + "="*60)
    print("🎓 PHASE 3: Knowledge Distillation")
    print("="*60)
    
    from training.knowledge_distillation import train_with_distillation
    
    student, history = train_with_distillation(
        teacher_model=teacher_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        student_size=student_size,
        epochs=epochs,
        save_path=str(WEIGHTS_DIR / f'auralis_{student_size}.keras')
    )
    
    return student, history


def evaluate_model(model, test_dataset):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("📊 Evaluation")
    print("="*60)
    
    results = model.evaluate(test_dataset, return_dict=True)
    
    for metric, value in results.items():
        print(f"   {metric}: {value:.4f}")
    
    return results


def save_training_report(
    history: dict,
    eval_results: dict,
    save_path: str
):
    """Save training report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_history': history,
        'evaluation_results': eval_results,
        'config': {
            'audio': vars(AUDIO_CONFIG),
            'model': vars(MODEL_CONFIG),
            'training': vars(TRAINING_CONFIG)
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Report saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Auralis ML Model')
    
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--model-type', type=str, default='full',
                        choices=['full', 'transformer', 'compact'],
                        help='Model architecture type')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--pretrain', action='store_true',
                        help='Run contrastive pre-training')
    parser.add_argument('--distill', action='store_true',
                        help='Run knowledge distillation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup
    setup_gpu()
    
    print("\n" + "="*60)
    print("🚀 AURALIS ML TRAINING PIPELINE")
    print("="*60)
    print(f"   Data Directory: {args.data_dir}")
    print(f"   Model Type: {args.model_type}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print("="*60)
    
    # Create directories
    WEIGHTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    try:
        # Load datasets
        print("\n📂 Loading datasets...")
        train_dataset, val_dataset = load_datasets(args.data_dir, args.batch_size)
        
        encoder = None
        
        # Phase 1: Contrastive pre-training (optional)
        if args.pretrain:
            # Get audio files for contrastive learning
            audio_dir = Path(args.data_dir) / 'audio' / 'train'
            audio_files = list(audio_dir.glob('*.wav'))
            
            if audio_files:
                encoder = train_phase1_contrastive(
                    audio_files=[str(f) for f in audio_files],
                    epochs=args.epochs // 2
                )
        
        # Phase 2: Supervised training
        model, history = train_phase2_supervised(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            encoder=encoder,
            epochs=args.epochs
        )
        
        # Phase 3: Knowledge distillation (optional)
        if args.distill:
            student, _ = train_phase3_distillation(
                teacher_model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=args.epochs // 2
            )
            model = student  # Use student for evaluation
        
        # Evaluate
        print("\n📊 Final Evaluation...")
        eval_results = evaluate_model(model, val_dataset)
        
        # Save report
        report_path = LOGS_DIR / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        save_training_report(
            history=history.history if hasattr(history, 'history') else history,
            eval_results=eval_results,
            save_path=str(report_path)
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"   Model saved to: {WEIGHTS_DIR}")
        print(f"   Report saved to: {report_path}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()