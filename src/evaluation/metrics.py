import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

CLASS_NAMES = ["Healthy", "Ball", "Inner", "Outer"]

# ============================================
# MODEL EVALUATION
# ============================================
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec  = recall_score(y_test, preds, average="weighted")
    f1   = f1_score(y_test, preds, average="weighted")
    cm   = confusion_matrix(y_test, preds)
    rep  = classification_report(y_test, preds, target_names=CLASS_NAMES)

    print("="*45)
    print("MODEL EVALUATION RESULTS")
    print("="*45)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(rep)
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": preds
    }

# ============================================
# HEALTH INDEX
# ============================================
def compute_health_index(proba, class_idx=0):
    healthy_prob = proba[:, class_idx]
    health_index = healthy_prob * 100
    return health_index

def maintenance_recommendation(health_index):
    if health_index >= 80:
        return "Continue Monitoring"
    elif health_index >= 50:
        return "Inspect Soon"
    else:
        return "Replace Bearing Immediately"

# ============================================
# SHUFFLE TEST
# ============================================
def shuffle_test(X_feat, y, scaler, model):
    print("\n" + "="*45)
    print("SHUFFLE TEST")
    print("="*45)

    y_shuffled = y.copy()
    np.random.seed(42)
    np.random.shuffle(y_shuffled)

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_feat, y_shuffled,
        test_size=0.2,
        random_state=42,
        stratify=y_shuffled
    )

    X_train_s = scaler.fit_transform(X_train_s)
    X_test_s  = scaler.transform(X_test_s)

    model.fit(X_train_s, y_train_s)
    acc = accuracy_score(y_test_s, model.predict(X_test_s))

    print(f"Shuffled label accuracy : {acc:.4f}")
    print(f"Expected (random chance): ~0.25")
    if acc < 0.35:
        print("RESULT: PASSED - Model is genuine.")
    else:
        print("RESULT: WARNING - Check for data leakage.")
    return acc

# ============================================
# K-FOLD CROSS VALIDATION
# ============================================
def kfold_cross_validation(X_feat, y, k=5):
    from sklearn.ensemble import RandomForestClassifier
    print("\n" + "="*45)
    print(f"{k}-FOLD CROSS VALIDATION")
    print("="*45)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_feat, y, cv=cv, scoring="accuracy")

    print(f"Fold scores : {[round(s,4) for s in scores]}")
    print(f"Mean        : {scores.mean():.4f}")
    print(f"Std         : {scores.std():.4f}")
    return scores

# ============================================
# ROC CURVES
# ============================================
def plot_roc_curves(model, X_test, y_test, save=True):
    print("\nGenerating ROC curves...")
    classes = [0, 1, 2, 3]
    colors  = ["steelblue", "tomato", "green", "orange"]

    y_bin  = label_binarize(y_test, classes=classes)

    if not hasattr(model, "predict_proba"):
        print("Model does not support predict_proba. Skipping ROC.")
        return

    y_prob = model.predict_proba(X_test)

    plt.figure(figsize=(10, 7))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0,1], [0,1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Classes", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/roc_curves.png", dpi=150)
        print("Saved: results/roc_curves.png")
    plt.show()