import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.data_loader import load_dataset
from src.features.feature_extractor import build_feature_matrix, FEATURE_NAMES
from src.models.random_forest import train_and_evaluate
from src.evaluation.metrics import (
    evaluate_model, shuffle_test,
    kfold_cross_validation, plot_roc_curves
)
from src.visualisation.plots import (
    plot_time_domain, plot_fft,
    plot_confusion_matrix, plot_fault_frequencies,
    plot_feature_importance, plot_health_index
)
from src.physics.bearing_physics import compute_fault_frequencies
from src.explainability.shap_explainer import run_shap, plot_shap_summary, plot_shap_bar
from src.health.health_monitor import batch_health_report, health_summary

print("="*50)
print("PHYSICS-AWARE PREDICTIVE MAINTENANCE")
print("="*50)

# STEP 1: Load data
X_raw, y, file_names = load_dataset()

# STEP 2: Extract features
X_feat = build_feature_matrix(X_raw)

# STEP 3: Train all models
results, scaler, X_test_scaled, y_test, best_name = train_and_evaluate(X_feat, y)

# STEP 4: Evaluate best model
best_model = results[best_name]["model"]
metrics = evaluate_model(best_model, X_test_scaled, y_test)

# STEP 5: Visualizations
print("\nGenerating visualizations...")
fault_freqs = compute_fault_frequencies(1797)

signals_dict = {}
class_map = {"Healthy": 0, "Ball": 1, "Inner": 2, "Outer": 3}
for cname, label in class_map.items():
    idx = np.where(y == label)[0][0]
    signals_dict[cname] = X_raw[idx]

plot_time_domain(signals_dict)
plot_fft(signals_dict)
plot_confusion_matrix(y_test, metrics["predictions"])
plot_fault_frequencies(
    X_raw[np.where(y == 3)[0][0]],
    fault_freqs,
    title="Outer Race Fault - FFT with Fault Frequencies"
)

# STEP 6: Feature importance
if best_name in ["RandomForest", "XGBoost"]:
    plot_feature_importance(best_model, FEATURE_NAMES)

# STEP 7: SHAP
print("\nRunning SHAP...")
X_train_full, X_test_full, _, _ = train_test_split(
    X_feat, y, test_size=0.2, random_state=42, stratify=y
)
X_train_scaled = scaler.transform(X_train_full)
explainer, shap_values = run_shap(best_model, X_train_scaled, X_test_scaled, FEATURE_NAMES)
plot_shap_summary(shap_values, X_test_scaled, FEATURE_NAMES)
plot_shap_bar(shap_values, FEATURE_NAMES)

# STEP 8: Health index
print("\nComputing health index...")
if hasattr(best_model, "predict_proba"):
    probas = best_model.predict_proba(X_test_scaled)
    report = batch_health_report(probas, metrics["predictions"])
    summary = health_summary(report)
    health_scores = [r["health"] for r in report]
    plot_health_index(health_scores)

# STEP 9: Validation
print("\nRunning validation tests...")
shuffle_test(
    X_feat, y,
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
kfold_cross_validation(X_feat, y)
plot_roc_curves(best_model, X_test_scaled, y_test)

print("\nAll done! Check results/ folder for all saved plots.")
# STEP 10: Generate PDF Report
print("\nGenerating PDF report...")
from src.reports.report_generator import generate_report
from src.physics.bearing_physics import physics_validation

sample_signal = X_raw[np.where(y == 3)[0][0]]
sample_pred = 3
phys_val = physics_validation(sample_signal, sample_pred)

sample_features = list(results[best_name]["model"].predict_proba(X_test_scaled[:1])[0])
feat_vals = [float(v) for v in X_test_scaled[0]]

generate_report(
    prediction=sample_pred,
    confidence=max(results[best_name]["model"].predict_proba(X_test_scaled[:1])[0]),
    health_index=results[best_name]["model"].predict_proba(X_test_scaled[:1])[0][0]*100,
    physics_validation=phys_val,
    metrics=metrics,
    fault_freqs=fault_freqs,
    feature_names=FEATURE_NAMES,
    feature_values=feat_vals,
    output_path="results/engineering_report.pdf"
)
print("PDF Report saved to results/engineering_report.pdf")