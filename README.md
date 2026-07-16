# Physics-Guided Predictive Maintenance Framework for Rotating Machinery

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)
![Models](https://img.shields.io/badge/Models-RF%20%7C%20SVM%20%7C%20XGBoost-orange)
![Dataset](https://img.shields.io/badge/Dataset-CWRU-yellow)

A complete physics-guided predictive maintenance framework that combines
vibration signal processing, bearing kinematics, machine learning,
and explainable AI to detect and diagnose bearing faults in rotating machinery.

---

## Overview

Most bearing fault detection projects use only statistical or frequency
features and train a single classifier. This framework adds a complete
**Physics Engine** that computes bearing fault characteristic frequencies
(BPFO, BPFI, BSF, FTF), harmonic energies, and sideband energies —
then **validates the AI prediction against physics** to give a physics
confidence score.

The system also introduces an **AI vs Physics Agreement** check — if the
ML model and the physics engine disagree on the fault type, the system
flags it for manual inspection. This is how real industrial diagnostic
systems work.

---

## Key Features

### Multi-Domain Feature Extraction (47 features)
- **Statistical (11):** RMS, Peak, Std, Kurtosis, Skewness, Crest Factor, Peak-to-Peak, Impulse Factor, Shape Factor, Margin Factor, Mean
- **Frequency (8):** Dominant Frequency, Spectral Centroid, Spectral Energy, Spectral Entropy, Low/Mid/High Band Ratio, Peak Amplitude
- **Wavelet (16):** DWT db4 level 3 — Energy, MeanAbs, Std, Entropy per level
- **Physics (12):** BPFO/BPFI/BSF/FTF energy, harmonic energy 1x/2x/3x, sideband energy, fault energy ratio

### Physics Engine
- Bearing kinematics: BPFO, BPFI, BSF, FTF from bearing geometry
- Harmonic analysis: energy at 1x, 2x, 3x fault frequencies
- Sideband analysis: energy at fault_freq ± k × shaft_hz
- Physics validation: confidence score comparing AI prediction to physics

### Machine Learning
- Three models benchmarked: Random Forest, SVM, XGBoost
- Best model: **SVM at 100% accuracy**
- Stratified 80/20 train-test split with StandardScaler

### Explainable AI
- SHAP values for feature importance
- Top features: RMS, Peak, BPFO Energy, Spectral features

### Health Monitoring
- Health index 0–100% based on class probabilities
- Status: GOOD / WARNING / CRITICAL
- Maintenance recommendation: Continue / Inspect / Replace

### Validation
- Shuffle test: 32.6% — PASSED, model is genuine
- 5-Fold Cross Validation: 99.76% mean accuracy, 0.0015 std
- ROC AUC: 1.0 for all 4 classes

### AI vs Physics Agreement
- Cross-validates ML prediction with physics engine
- Shows AGREE or DISAGREE
- Flags disagreement for manual inspection

### Streamlit Dashboard
- Upload any .mat vibration file
- View time domain signal, FFT with fault frequencies
- Fault prediction, confidence, health index
- Individual BPFO/BPFI/BSF/FTF match status
- AI vs Physics Agreement
- Download PDF engineering report

### Auto PDF Engineering Report
- Executive summary, bearing kinematics table
- Model performance metrics
- Physics validation results
- All signal analysis plots
- Maintenance recommendation with sign-off section

---

## Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Random Forest | 99.53% | 99.53% | 99.53% | 99.53% |
| SVM | 100.00% | 100.00% | 100.00% | 100.00% |
| XGBoost | 99.53% | 99.53% | 99.53% | 99.53% |

### Validation

| Test | Result | Status |
|---|---|---|
| Shuffle Test | 32.6% | PASSED |
| 5-Fold CV Mean | 99.76% | PASSED |
| 5-Fold CV Std | 0.0015 | PASSED |
| ROC AUC Healthy | 1.0 | PASSED |
| ROC AUC Ball | 1.0 | PASSED |
| ROC AUC Inner | 1.0 | PASSED |
| ROC AUC Outer | 1.0 | PASSED |

### Physics Validation

| Parameter | Value |
|---|---|
| Target Frequency BPFO | 110.67 Hz |
| Physics Confidence | 100% |
| AI vs Physics | AGREE |

---

## Dataset

**Case Western Reserve University (CWRU) Bearing Dataset**

| Class | Label | Files |
|---|---|---|
| Healthy | 0 | 97, 98, 99, 100 |
| Ball Fault | 1 | 118-121, 185-188 |
| Inner Race Fault | 2 | 105-108, 169-172 |
| Outer Race Fault | 3 | 294-297, 313, 315 |

- Total segments: 2126
- Segment size: 2048 samples
- Sampling frequency: 12000 Hz
- Source: https://engineering.case.edu/bearingdatacenter

---

## Bearing Physics

CWRU Bearing Specifications:
- Pitch diameter: 1.748 inches
- Ball diameter: 0.3126 inches
- Number of balls: 9

Fault frequencies at 1797 RPM:

| Frequency | Symbol | Value |
|---|---|---|
| Ball Pass Frequency Outer Race | BPFO | 110.67 Hz |
| Ball Pass Frequency Inner Race | BPFI | 158.88 Hz |
| Ball Spin Frequency | BSF | 81.06 Hz |
| Fundamental Train Frequency | FTF | 12.30 Hz |

---

## Folder Structure

- data/CWRU/Healthy, Ball, Inner, Outer
- results/ — all plots, models, PDF report
- src/preprocessing/ — data_loader.py
- src/features/ — statistical, frequency, wavelet, feature_extractor
- src/physics/ — bearing_physics.py
- src/models/ — random_forest.py
- src/evaluation/ — metrics.py
- src/visualisation/ — plots.py
- src/explainability/ — shap_explainer.py
- src/health/ — health_monitor.py
- src/reports/ — report_generator.py
- src/dashboard/ — app.py
- main.py

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib pywavelets xgboost shap streamlit reportlab seaborn pandas
```

---

## How to Run

```bash
python main.py
```

```bash
streamlit run src/dashboard/app.py
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Signal Processing | NumPy, SciPy, PyWavelets |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Report Generation | ReportLab |
| Data Handling | Pandas |

---

## References

1. [A physics-informed deep learning approach for bearing fault detection](https://www.sciencedirect.com/science/article/abs/pii/S0952197621001421) — ScienceDirect 2021
2. [Research on bearing fault diagnosis based on machine learning and SHAP interpretability](https://www.nature.com/articles/s41598-025-25083-4) — Scientific Reports 2025
3. [SHAP for Efficient Feature Selection in Rolling Bearing Fault Diagnosis](https://www.researchgate.net/publication/378063501_SHapley_Additive_exPlanations_SHAP_for_Efficient_Feature_Selection_in_Rolling_Bearing_Fault_Diagnosis) — ResearchGate 2024
4. [Machine Learning and Deep Learning Algorithms for Bearing Fault Diagnostics](https://arxiv.org/pdf/1901.08247v1) — arXiv 2019
5. [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter) — Case Western Reserve University

---

## Author

**Sai Dileep**

Mechanical Engineering, NIT Calicut

Specialisation: Vibration Analysis, Physics-Guided ML, Predictive Maintenance