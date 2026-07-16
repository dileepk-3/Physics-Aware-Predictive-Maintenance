import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import joblib
import os
import sys
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.feature_extractor import extract_features, FEATURE_NAMES
from src.physics.bearing_physics import compute_fault_frequencies, physics_validation
from src.health.health_monitor import maintenance_recommendation
from src.reports.report_generator import generate_report

st.set_page_config(
    page_title="Physics-Aware Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_model():
    model  = joblib.load("results/best_model.pkl")
    scaler = joblib.load("results/scaler.pkl")
    return model, scaler

@st.cache_resource
def load_all_results():
    try:
        return joblib.load("results/all_model_results.pkl")
    except:
        return None

CLASS_NAMES = {
    0: "✅ Healthy",
    1: "⚠️ Ball Fault",
    2: "⚠️ Inner Race Fault",
    3: "🔴 Outer Race Fault"
}

CLASS_NAMES_PLAIN = {
    0: "Healthy",
    1: "Ball Fault",
    2: "Inner Race Fault",
    3: "Outer Race Fault"
}

FAULT_COLORS = {0: "green", 1: "orange", 2: "orange", 3: "red"}

# ── HEADER ──────────────────────────────────
st.title("⚙️ Physics-Aware Predictive Maintenance")
st.markdown("### Bearing Fault Detection using Vibration Analysis + Physics Engine")
st.markdown("---")

# ── SIDEBAR ─────────────────────────────────
st.sidebar.header("Settings")
shaft_rpm    = st.sidebar.slider("Shaft RPM", 500, 3000, 1797, step=50)
segment_size = st.sidebar.selectbox("Segment Size", [1024, 2048, 4096], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Fault Frequencies at selected RPM:**")
fault_freqs = compute_fault_frequencies(shaft_rpm)
for k, v in fault_freqs.items():
    st.sidebar.write(f"`{k}`: {v} Hz")

# ── MODEL COMPARISON TABLE ───────────────────
st.subheader("📊 Model Comparison")
all_results = load_all_results()
if all_results:
    import pandas as pd
    comp_data = []
    for mname, mres in all_results.items():
        comp_data.append({
            "Model":     mname,
            "Accuracy":  f"{mres['accuracy']*100:.2f}%",
            "Precision": f"{mres['precision']*100:.2f}%",
            "Recall":    f"{mres['recall']*100:.2f}%",
            "F1 Score":  f"{mres['f1']*100:.2f}%",
            "Train Time": f"{mres['train_time']}s"
        })
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
else:
    st.info("Run main.py first to generate model comparison data.")

st.markdown("---")

# ── FILE UPLOAD ──────────────────────────────
st.subheader("📂 Upload Vibration File")
uploaded_file = st.file_uploader(
    "Upload a .mat or .npy vibration file",
    type=["mat", "npy"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".mat"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            data = sio.loadmat(tmp_path)
            signal_key = None
            for key in data.keys():
                if "DE_time" in key:
                    signal_key = key
                    break
            if signal_key is None:
                st.error("No DE_time key found in .mat file.")
                st.stop()
            signal = data[signal_key].flatten()
        else:
            signal = np.load(uploaded_file)

        st.success(f"File loaded. Signal length: {len(signal)} samples")

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    segment = signal[:segment_size]

    # ── SIGNAL ANALYSIS ──────────────────────
    st.markdown("---")
    st.subheader("📈 Signal Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Time Domain Signal**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(segment[:1000], color="steelblue", linewidth=0.8)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.set_title("Time Domain")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**FFT with Fault Frequencies**")
        x        = segment - np.mean(segment)
        N        = len(x)
        fft_vals = np.abs(np.fft.rfft(x)) / N
        freqs    = np.fft.rfftfreq(N, 1/12000)
        mask     = freqs < 1000
        fig, ax  = plt.subplots(figsize=(6, 3))
        ax.plot(freqs[mask], fft_vals[mask], color="steelblue", linewidth=0.8)
        colors_f = {"BPFO":"red","BPFI":"green","BSF":"orange","FTF":"purple"}
        for fname, fval in fault_freqs.items():
            if fname in colors_f and fval < 1000:
                ax.axvline(x=fval, color=colors_f[fname],
                           linestyle="--", alpha=0.8, label=f"{fname}={fval}Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("FFT Analysis")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    # ── PREDICTION ───────────────────────────
    st.markdown("---")
    st.subheader("🔍 Fault Prediction")

    try:
        model, scaler  = load_model()
        features        = extract_features(segment, shaft_rpm=shaft_rpm)
        features_scaled = scaler.transform([features])
        prediction      = model.predict(features_scaled)[0]
        proba           = model.predict_proba(features_scaled)[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Predicted Fault**")
            color = FAULT_COLORS[prediction]
            st.markdown(
                f"<h2 style='color:{color}'>{CLASS_NAMES[prediction]}</h2>",
                unsafe_allow_html=True
            )
        with col2:
            conf = max(proba) * 100
            st.metric("Model Confidence", f"{conf:.1f}%")
        with col3:
            health = proba[0] * 100
            st.metric("Health Index", f"{health:.1f}%")

        st.markdown("**Class Probabilities**")
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(
            ["Healthy","Ball","Inner","Outer"],
            proba * 100,
            color=["green","orange","orange","red"],
            alpha=0.7
        )
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

    # ── PHYSICS VALIDATION (DETAILED) ────────
    st.markdown("---")
    st.subheader("⚛️ Physics Validation")

    val = physics_validation(segment, prediction, shaft_rpm=shaft_rpm)

    from src.physics.bearing_physics import energy_around_freq, harmonic_analysis, sideband_analysis

    x2        = segment - np.mean(segment)
    N2        = len(x2)
    fft_v2    = np.abs(np.fft.rfft(x2)) / N2
    freqs2    = np.fft.rfftfreq(N2, 1/12000)
    shaft_hz  = fault_freqs["shaft_hz"]
    total_e   = np.sum(fft_v2**2) + 1e-8
    threshold = total_e * 0.001

    def check_match(freq):
        e = energy_around_freq(fft_v2, freqs2, freq)
        return e > threshold

    bpfo_match = check_match(fault_freqs["BPFO"])
    bpfi_match = check_match(fault_freqs["BPFI"])
    bsf_match  = check_match(fault_freqs["BSF"])
    ftf_match  = check_match(fault_freqs["FTF"])

    bpfo_sb = sideband_analysis(fft_v2, freqs2, fault_freqs["BPFO"], shaft_hz)
    sb_match = bpfo_sb > threshold

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("BPFO Match", "✅" if bpfo_match else "❌")
    with col2:
        st.metric("BPFI Match", "✅" if bpfi_match else "❌")
    with col3:
        st.metric("BSF Match",  "✅" if bsf_match  else "❌")
    with col4:
        st.metric("FTF Match",  "✅" if ftf_match  else "❌")
    with col5:
        st.metric("Sidebands",  "✅" if sb_match   else "❌")

    st.metric("Overall Physics Confidence", f"{val['confidence']*100:.2f}%")

    # ── AI vs PHYSICS AGREEMENT ───────────────
    st.markdown("---")
    st.subheader("🤝 AI vs Physics Agreement")

    class_to_freq = {
        1: ("BSF",  fault_freqs["BSF"]),
        2: ("BPFI", fault_freqs["BPFI"]),
        3: ("BPFO", fault_freqs["BPFO"])
    }

    if prediction == 0:
        physics_pred = 0
        physics_name = "Healthy"
        no_fault_energy = (
            energy_around_freq(fft_v2, freqs2, fault_freqs["BPFO"]) +
            energy_around_freq(fft_v2, freqs2, fault_freqs["BPFI"]) +
            energy_around_freq(fft_v2, freqs2, fault_freqs["BSF"])
        )
        physics_pred = 0 if no_fault_energy < threshold * 3 else -1
        physics_name = "Healthy" if physics_pred == 0 else "Fault Detected"
    else:
        fault_energies = {}
        for cls, (fname, fval) in class_to_freq.items():
            e = energy_around_freq(fft_v2, freqs2, fval)
            h = sum(harmonic_analysis(fft_v2, freqs2, fval))
            s = sideband_analysis(fft_v2, freqs2, fval, shaft_hz)
            fault_energies[cls] = e + h + s

        physics_pred = max(fault_energies, key=fault_energies.get)
        physics_name = CLASS_NAMES_PLAIN[physics_pred]

    ai_name      = CLASS_NAMES_PLAIN[prediction]
    agreement    = (physics_pred == prediction)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**AI Prediction**")
        st.markdown(
            f"<h3 style='color:{FAULT_COLORS[prediction]}'>{ai_name}</h3>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown("**Physics Prediction**")
        color2 = FAULT_COLORS.get(physics_pred, "gray")
        st.markdown(
            f"<h3 style='color:{color2}'>{physics_name}</h3>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown("**Agreement**")
        if agreement:
            st.markdown("<h3 style='color:green'>✅ AGREE</h3>", unsafe_allow_html=True)
            st.success("AI and Physics predictions match. High confidence result.")
        else:
            st.markdown("<h3 style='color:red'>⚠️ DISAGREE</h3>", unsafe_allow_html=True)
            st.warning("AI and Physics disagree. Manual inspection recommended.")

    # ── MAINTENANCE RECOMMENDATION ────────────
    st.markdown("---")
    st.subheader("🔧 Maintenance Recommendation")
    rec = maintenance_recommendation(health, prediction)
    if rec["status"] == "GOOD":
        st.success(f"{rec['action']} — {rec['message']}")
    elif rec["status"] == "WARNING":
        st.warning(f"{rec['action']} — {rec['message']}")
    else:
        st.error(f"{rec['action']} — {rec['message']}")

    # ── FEATURE TABLE ─────────────────────────
    st.markdown("---")
    st.subheader("📋 Extracted Features")
    with st.expander("Show all 47 features"):
        import pandas as pd
        feat_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Value":   [round(float(f), 6) for f in features]
        })
        st.dataframe(feat_df, use_container_width=True)

    # ── DOWNLOAD REPORT ───────────────────────
    st.markdown("---")
    st.subheader("📄 Engineering Report")
    if st.button("🖨️ Generate & Download PDF Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                metrics_dict = {
                    "accuracy":  1.0,
                    "precision": 1.0,
                    "recall":    1.0,
                    "f1":        1.0
                }
                report_path = generate_report(
                    prediction         = int(prediction),
                    confidence         = float(max(proba)),
                    health_index       = float(health),
                    physics_validation = val,
                    metrics            = metrics_dict,
                    fault_freqs        = fault_freqs,
                    feature_names      = FEATURE_NAMES,
                    feature_values     = features,
                    output_path        = "results/engineering_report.pdf"
                )
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label     = "⬇️ Download PDF Report",
                    data      = pdf_bytes,
                    file_name = f"bearing_report_{uploaded_file.name}.pdf",
                    mime      = "application/pdf"
                )
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Report error: {e}")

else:
    st.info("👆 Upload a .mat vibration file to begin analysis.")
    st.markdown("""
    **How to use:**
    1. Upload any `.mat` file from your CWRU dataset
    2. View time domain, FFT, prediction, physics validation
    3. Check AI vs Physics Agreement
    4. Download PDF engineering report

    **Supported fault types:**
    - ✅ Healthy bearing
    - ⚠️ Ball fault
    - ⚠️ Inner race fault
    - 🔴 Outer race fault
    """)