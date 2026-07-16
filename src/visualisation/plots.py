import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

FS = 12000
CLASS_NAMES = ["Healthy", "Ball", "Inner", "Outer"]
os.makedirs("results", exist_ok=True)

# ============================================
# TIME DOMAIN PLOT
# ============================================
def plot_time_domain(signals_dict, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for i, (name, signal) in enumerate(signals_dict.items()):
        axes[i].plot(signal[:1000], color=["steelblue","tomato","green","orange"][i])
        axes[i].set_title(f"{name} - Time Domain")
        axes[i].set_xlabel("Samples")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)
    plt.suptitle("Time Domain Signals - All Classes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("results/time_domain.png", dpi=150)
        print("Saved: results/time_domain.png")
    plt.show()

# ============================================
# FFT PLOT
# ============================================
def plot_fft(signals_dict, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    colors = ["steelblue", "tomato", "green", "orange"]
    for i, (name, signal) in enumerate(signals_dict.items()):
        x = signal - np.mean(signal)
        N = len(x)
        fft_vals = np.abs(np.fft.rfft(x)) / N
        freqs = np.fft.rfftfreq(N, 1/FS)
        mask = freqs < 3000
        axes[i].plot(freqs[mask], fft_vals[mask], color=colors[i])
        axes[i].set_title(f"{name} - FFT")
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)
    plt.suptitle("FFT Comparison - All Classes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("results/fft_comparison.png", dpi=150)
        print("Saved: results/fft_comparison.png")
    plt.show()

# ============================================
# CONFUSION MATRIX
# ============================================
def plot_confusion_matrix(y_test, y_pred, save=True):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save:
        plt.savefig("results/confusion_matrix.png", dpi=150)
        print("Saved: results/confusion_matrix.png")
    plt.show()

# ============================================
# FAULT FREQUENCY PLOT
# ============================================
def plot_fault_frequencies(signal, fault_freqs, title="FFT with Fault Frequencies", save=True):
    x = signal - np.mean(signal)
    N = len(x)
    fft_vals = np.abs(np.fft.rfft(x)) / N
    freqs = np.fft.rfftfreq(N, 1/FS)
    mask = freqs < 1000
    plt.figure(figsize=(12, 5))
    plt.plot(freqs[mask], fft_vals[mask], color="steelblue", label="FFT")
    colors = {"BPFO": "red", "BPFI": "green", "BSF": "orange", "FTF": "purple"}
    for fname, fval in fault_freqs.items():
        if fname in colors:
            plt.axvline(x=fval, color=colors[fname],
                       linestyle="--", alpha=0.8, label=f"{fname}={fval}Hz")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig("results/fault_frequencies.png", dpi=150)
        print("Saved: results/fault_frequencies.png")
    plt.show()

# ============================================
# FEATURE IMPORTANCE
# ============================================
def plot_feature_importance(model, feature_names, top_n=20, save=True):
    if not hasattr(model, "feature_importances_"):
        print("This model does not support feature importance.")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n),
            importances[indices],
            color="steelblue", alpha=0.8)
    plt.xticks(range(top_n),
               [feature_names[i] for i in indices],
               rotation=45, ha="right", fontsize=9)
    plt.title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    if save:
        plt.savefig("results/feature_importance.png", dpi=150)
        print("Saved: results/feature_importance.png")
    plt.show()

# ============================================
# HEALTH INDEX PLOT
# ============================================
def plot_health_index(health_scores, save=True):
    plt.figure(figsize=(12, 4))
    colors = ["green" if h >= 80 else "orange" if h >= 50 else "red"
              for h in health_scores]
    plt.bar(range(len(health_scores)), health_scores, color=colors, alpha=0.7)
    plt.axhline(y=80, color="green", linestyle="--", label="Safe (80%)")
    plt.axhline(y=50, color="orange", linestyle="--", label="Warning (50%)")
    plt.title("Health Index per Segment", fontsize=13, fontweight="bold")
    plt.xlabel("Segment")
    plt.ylabel("Health Index (%)")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("results/health_index.png", dpi=150)
        print("Saved: results/health_index.png")
    plt.show()