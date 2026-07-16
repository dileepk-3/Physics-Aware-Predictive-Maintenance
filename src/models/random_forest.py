import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os
import time

MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),
    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
}

def train_and_evaluate(X_feat, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = round(time.time() - start, 2)

        preds = model.predict(X_test_scaled)
        acc   = accuracy_score(y_test, preds)
        prec  = precision_score(y_test, preds, average="weighted")
        rec   = recall_score(y_test, preds, average="weighted")
        f1    = f1_score(y_test, preds, average="weighted")
        report = classification_report(
            y_test, preds,
            target_names=["Healthy","Ball","Inner","Outer"]
        )

        results[name] = {
            "model":       model,
            "accuracy":    acc,
            "precision":   prec,
            "recall":      rec,
            "f1":          f1,
            "train_time":  train_time,
            "predictions": preds,
            "report":      report
        }
        print(f"  Accuracy: {acc:.4f} | Time: {train_time}s")
        print(report)

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest model: {best_name} ({results[best_name]['accuracy']:.4f})")

    os.makedirs("results", exist_ok=True)
    joblib.dump(results[best_name]["model"], "results/best_model.pkl")
    joblib.dump(scaler, "results/scaler.pkl")
    joblib.dump(results, "results/all_model_results.pkl")
    print("Models saved to results/")

    return results, scaler, X_test_scaled, y_test, best_name