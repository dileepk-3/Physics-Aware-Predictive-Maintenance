import numpy as np
import shap
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

def run_shap(model, X_train, X_test, feature_names, max_samples=100):
    print("Running SHAP analysis...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:max_samples])
        print("TreeExplainer used.")
    except Exception:
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X_train, 50)
        )
        shap_values = explainer.shap_values(X_test[:max_samples])
        print("KernelExplainer used.")
    return explainer, shap_values

def plot_shap_summary(shap_values, X_test, feature_names, max_samples=100, save=True):
    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values,
            X_test[:max_samples],
            feature_names=feature_names,
            class_names=["Healthy","Ball","Inner","Outer"],
            show=False
        )
    else:
        shap.summary_plot(
            shap_values,
            X_test[:max_samples],
            feature_names=feature_names,
            show=False
        )
    if save:
        plt.savefig("results/shap_summary.png", dpi=150, bbox_inches="tight")
        print("Saved: results/shap_summary.png")
    plt.show()

def plot_shap_bar(shap_values, feature_names, save=True):
    feature_names = list(feature_names)

    # Handle list of arrays (multiclass)
    if isinstance(shap_values, list):
        all_shap = np.array([np.abs(sv) for sv in shap_values])
        mean_shap = all_shap.mean(axis=0).mean(axis=0)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            mean_shap = np.abs(shap_values).mean(axis=0).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
    else:
        mean_shap = np.abs(np.array(shap_values)).mean(axis=0)

    mean_shap = mean_shap.flatten()
    n = min(15, len(mean_shap), len(feature_names))

    indices = np.argsort(mean_shap)[-n:].tolist()
    labels = [feature_names[i] for i in indices]
    values = [float(mean_shap[i]) for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color="steelblue", alpha=0.8)
    plt.title("SHAP Feature Importance (Top 15)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()

    if save:
        plt.savefig("results/shap_bar.png", dpi=150, bbox_inches="tight")
        print("Saved: results/shap_bar.png")
    plt.show()