# ==============================
# 📄 training/__init__.py
# ==============================
# Training package initialization
# ==============================

from .audio_augmentation import (
    AudioAugmentor,
    SpecAugment,
    AugmentedDataGenerator
)

from .contrastive_learning import (
    AudioContrastiveModel,
    ContrastiveDataGenerator,
    pretrain_contrastive
)

from .knowledge_distillation import (
    KnowledgeDistillation,
    train_with_distillation,
    create_compact_student_model
)

from .custom_losses import (
    FocalLoss,
    LabelSmoothingLoss,
    HierarchicalLoss,
    ConfidenceCalibrationLoss,
    AuralisLoss
)

__all__ = [
    # Audio Augmentation
    'AudioAugmentor',
    'SpecAugment',
    'AugmentedDataGenerator',
    
    # Contrastive Learning
    'AudioContrastiveModel',
    'ContrastiveDataGenerator',
    'pretrain_contrastive',
    
    # Knowledge Distillation
    'KnowledgeDistillation',
    'train_with_distillation',
    'create_compact_student_model',
    
    # Custom Losses
    'FocalLoss',
    'LabelSmoothingLoss',
    'HierarchicalLoss',
    'ConfidenceCalibrationLoss',
    'AuralisLoss',
]