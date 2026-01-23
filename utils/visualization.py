# ==============================
# 📄 utils/visualization.py
# ==============================
# Visualization Utilities
# ==============================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import librosa
import librosa.display
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_CONFIG, LABEL_CONFIG

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = None,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (12, 4),
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot audio waveform.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show plot
        
    Returns:
        Matplotlib figure
    """
    sr = sample_rate or AUDIO_CONFIG.sample_rate
    
    fig, ax = plt.subplots(figsize=figsize)
    
    times = np.arange(len(audio)) / sr
    ax.plot(times, audio, linewidth=0.5, color='#2E86AB')
    ax.fill_between(times, audio, alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, times[-1])
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int = None,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None,
    show: bool = True,
    cmap: str = 'magma'
) -> plt.Figure:
    """
    Plot audio spectrogram.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show plot
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    sr = sample_rate or AUDIO_CONFIG.sample_rate
    
    fig, ax = plt.subplots(figsize=figsize)
    
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )
    
    img = librosa.display.specshow(
        D,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=ax,
        cmap=cmap
    )
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Amplitude (dB)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_mel_spectrogram(
    audio: np.ndarray = None,
    mel_spec: np.ndarray = None,
    sample_rate: int = None,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None,
    show: bool = True,
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Plot mel spectrogram.
    
    Args:
        audio: Audio array (if mel_spec not provided)
        mel_spec: Pre-computed mel spectrogram
        sample_rate: Sample rate
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show plot
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    sr = sample_rate or AUDIO_CONFIG.sample_rate
    
    if mel_spec is None and audio is not None:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=AUDIO_CONFIG.n_mels,
            n_fft=AUDIO_CONFIG.n_fft,
            hop_length=AUDIO_CONFIG.hop_length
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    elif mel_spec is not None and mel_spec.ndim == 2:
        # Handle [time, freq] format
        if mel_spec.shape[0] > mel_spec.shape[1]:
            mel_spec = mel_spec.T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    img = librosa.display.specshow(
        mel_spec,
        sr=sr,
        hop_length=AUDIO_CONFIG.hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap=cmap
    )
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Amplitude (dB)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mel Frequency', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_training_history(
    history: Dict,
    metrics: List[str] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 10),
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        metrics: Metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show plot
        
    Returns:
        Matplotlib figure
    """
    # Get metrics to plot
    if metrics is None:
        metrics = [k for k in history.keys() if not k.startswith('val_')]
    
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Training metric
        if metric in history:
            ax.plot(
                history[metric],
                label=f'Train',
                color=colors[0],
                linewidth=2
            )
        
        # Validation metric
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(
                history[val_metric],
                label=f'Validation',
                color=colors[1],
                linewidth=2,
                linestyle='--'
            )
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
    save_path: str = None,
    show: bool = True,
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show plot
        normalize: Whether to normalize
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
    
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=8
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_attention_weights(
    attention_weights: np.ndarray,
    x_labels: List[str] = None,
    y_labels: List[str] = None,
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot attention weight heatmap.
    
    Args:
        attention_weights: Attention weights [query, key]
        x_labels: Labels for x-axis (keys)
        y_labels: Labels for y-axis (queries)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(attention_weights, aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Attention Weight')
    
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_prediction_distribution(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, List[str]] = None,
    title: str = "Prediction Distribution",
    figsize: Tuple[int, int] = (14, 6),
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot prediction probability distributions.
    
    Args:
        predictions: Dictionary of predictions
        labels: Dictionary of label names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show plot
        
    Returns:
        Matplotlib figure
    """
    n_outputs = len(predictions)
    fig, axes = plt.subplots(1, n_outputs, figsize=figsize)
    
    if n_outputs == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    for ax, (name, probs) in zip(axes, predictions.items()):
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        
        probs = probs.mean(axis=0)  # Average if batch
        
        if labels and name in labels:
            x_labels = labels[name]
        else:
            x_labels = [str(i) for i in range(len(probs))]
        
        # Only show top N
        top_n = min(10, len(probs))
        top_indices = np.argsort(probs)[-top_n:][::-1]
        
        bars = ax.barh(
            range(top_n),
            probs[top_indices],
            color=colors[:top_n]
        )
        
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([x_labels[i] for i in top_indices])
        ax.set_xlabel('Probability')
        ax.set_title(name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add percentage labels
        for bar, prob in zip(bars, probs[top_indices]):
            ax.text(
                prob + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%',
                va='center', fontsize=9
            )
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_audio_report(
    audio: np.ndarray,
    predictions: Dict,
    sample_rate: int = None,
    title: str = "Audio Analysis Report",
    save_path: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comprehensive audio analysis report.
    
    Args:
        audio: Audio array
        predictions: Model predictions
        sample_rate: Sample rate
        title: Report title
        save_path: Path to save report
        show: Whether to show
        
    Returns:
        Matplotlib figure
    """
    sr = sample_rate or AUDIO_CONFIG.sample_rate
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Waveform
    ax1 = fig.add_subplot(gs[0, :2])
    times = np.arange(len(audio)) / sr
    ax1.plot(times, audio, linewidth=0.5, color='#2E86AB')
    ax1.fill_between(times, audio, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Waveform', fontweight='bold')
    ax1.set_xlim(0, times[-1])
    
    # Mel spectrogram
    ax2 = fig.add_subplot(gs[1, :2])
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(
        mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='viridis'
    )
    ax2.set_title('Mel Spectrogram', fontweight='bold')
    
    # Location predictions
    ax3 = fig.add_subplot(gs[0, 2])
    if 'location' in predictions:
        probs = predictions['location']
        if probs.ndim > 1:
            probs = probs[0]
        top_5 = np.argsort(probs)[-5:][::-1]
        ax3.barh(range(5), probs[top_5], color='#E74C3C')
        ax3.set_yticks(range(5))
        ax3.set_yticklabels([LABEL_CONFIG.locations[i] for i in top_5], fontsize=9)
        ax3.set_xlim(0, 1)
        ax3.set_title('Location Prediction', fontweight='bold')
    
    # Situation predictions
    ax4 = fig.add_subplot(gs[1, 2])
    if 'situation' in predictions:
        probs = predictions['situation']
        if probs.ndim > 1:
            probs = probs[0]
        top_5 = np.argsort(probs)[-5:][::-1]
        ax4.barh(range(5), probs[top_5], color='#27AE60')
        ax4.set_yticks(range(5))
        ax4.set_yticklabels([LABEL_CONFIG.situations[i] for i in top_5], fontsize=9)
        ax4.set_xlim(0, 1)
        ax4.set_title('Situation Prediction', fontweight='bold')
    
    # Summary text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary text
    summary_lines = [
        f"📊 Audio Duration: {len(audio)/sr:.2f} seconds",
        f"🎵 Sample Rate: {sr} Hz",
    ]
    
    if 'location' in predictions:
        loc_probs = predictions['location']
        if loc_probs.ndim > 1:
            loc_probs = loc_probs[0]
        top_loc = LABEL_CONFIG.locations[np.argmax(loc_probs)]
        summary_lines.append(f"📍 Predicted Location: {top_loc} ({np.max(loc_probs)*100:.1f}%)")
    
    if 'situation' in predictions:
        sit_probs = predictions['situation']
        if sit_probs.ndim > 1:
            sit_probs = sit_probs[0]
        top_sit = LABEL_CONFIG.situations[np.argmax(sit_probs)]
        summary_lines.append(f"🎯 Predicted Situation: {top_sit} ({np.max(sit_probs)*100:.1f}%)")
    
    if 'emergency' in predictions:
        emerg = predictions['emergency']
        if emerg.ndim > 1:
            emerg = emerg[0]
        is_emergency = emerg[0] > 0.5
        emoji = "🚨" if is_emergency else "✅"
        summary_lines.append(f"{emoji} Emergency: {'Yes' if is_emergency else 'No'} ({emerg[0]*100:.1f}%)")
    
    if 'confidence' in predictions:
        conf = predictions['confidence']
        if conf.ndim > 1:
            conf = conf[0]
        summary_lines.append(f"📈 Overall Confidence: {conf[0]*100:.1f}%")
    
    summary_text = "\n".join(summary_lines)
    ax5.text(
        0.5, 0.5, summary_text,
        transform=ax5.transAxes,
        fontsize=14,
        verticalalignment='center',
        horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        family='monospace'
    )
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# ==============================
# 🧪 TESTING
# ==============================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Visualization Utils")
    print("="*60)
    
    # Generate test audio
    duration = 3.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    audio = audio.astype(np.float32)
    
    print("\n📊 Testing plots (will show windows)...")
    
    # Test waveform
    plot_waveform(audio, sr, title="Test Waveform", show=False)
    print("   ✅ Waveform plot")
    
    # Test spectrogram
    plot_spectrogram(audio, sr, title="Test Spectrogram", show=False)
    print("   ✅ Spectrogram plot")
    
    # Test mel spectrogram
    plot_mel_spectrogram(audio=audio, sample_rate=sr, title="Test Mel Spectrogram", show=False)
    print("   ✅ Mel spectrogram plot")
    
    # Test training history
    fake_history = {
        'loss': np.random.rand(50) * 0.5,
        'val_loss': np.random.rand(50) * 0.6,
        'accuracy': 0.5 + np.random.rand(50) * 0.4,
        'val_accuracy': 0.4 + np.random.rand(50) * 0.4,
    }
    plot_training_history(fake_history, show=False)
    print("   ✅ Training history plot")
    
    # Test prediction distribution
    fake_preds = {
        'location': np.random.rand(13),
        'situation': np.random.rand(15),
    }
    fake_preds['location'] /= fake_preds['location'].sum()
    fake_preds['situation'] /= fake_preds['situation'].sum()
    
    plot_prediction_distribution(
        fake_preds,
        labels={
            'location': LABEL_CONFIG.locations,
            'situation': LABEL_CONFIG.situations
        },
        show=False
    )
    print("   ✅ Prediction distribution plot")
    
    # Test audio report
    fake_full_preds = {
        **fake_preds,
        'emergency': np.array([[0.15]]),
        'confidence': np.array([[0.85]])
    }
    create_audio_report(audio, fake_full_preds, sr, show=False)
    print("   ✅ Audio report")
    
    plt.close('all')
    
    print("\n✅ Visualization Utils test passed!")