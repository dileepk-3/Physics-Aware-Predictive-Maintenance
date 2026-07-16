import os
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, HRFlowable,
    PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ============================================
# COLORS
# ============================================
DARK_BLUE  = colors.HexColor("#1a237e")
MID_BLUE   = colors.HexColor("#1565c0")
LIGHT_BLUE = colors.HexColor("#e3f2fd")
GREEN      = colors.HexColor("#2e7d32")
RED        = colors.HexColor("#c62828")
ORANGE     = colors.HexColor("#e65100")
GRAY       = colors.HexColor("#f5f5f5")
DARK_GRAY  = colors.HexColor("#424242")

CLASS_NAMES = {
    0: "Healthy",
    1: "Ball Fault",
    2: "Inner Race Fault",
    3: "Outer Race Fault"
}

STATUS_COLORS = {
    "GOOD":     GREEN,
    "WARNING":  ORANGE,
    "CRITICAL": RED
}

# ============================================
# STYLES
# ============================================
def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontSize=22,
        fontName="Helvetica-Bold",
        textColor=DARK_BLUE,
        alignment=TA_CENTER,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontSize=12,
        fontName="Helvetica",
        textColor=DARK_GRAY,
        alignment=TA_CENTER,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=14,
        fontName="Helvetica-Bold",
        textColor=DARK_BLUE,
        spaceBefore=12,
        spaceAfter=6,
        borderPad=4
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        fontSize=10,
        fontName="Helvetica",
        textColor=DARK_GRAY,
        spaceAfter=4,
        leading=14
    ))
    styles.add(ParagraphStyle(
        name="SmallText",
        fontSize=8,
        fontName="Helvetica",
        textColor=DARK_GRAY,
    ))
    styles.add(ParagraphStyle(
        name="ResultGood",
        fontSize=16,
        fontName="Helvetica-Bold",
        textColor=GREEN,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="ResultCritical",
        fontSize=16,
        fontName="Helvetica-Bold",
        textColor=RED,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="ResultWarning",
        fontSize=16,
        fontName="Helvetica-Bold",
        textColor=ORANGE,
        alignment=TA_CENTER
    ))
    return styles

# ============================================
# HEADER / FOOTER
# ============================================
def add_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4

    # Header bar
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, height - 45, width, 45, fill=True, stroke=False)

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(1.5*cm, height - 28, "Physics-Aware Predictive Maintenance System")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(width - 1.5*cm, height - 28,
                           datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Footer
    canvas.setFillColor(DARK_BLUE)
    canvas.rect(0, 0, width, 30, fill=True, stroke=False)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.5*cm, 10, "Confidential Engineering Report | CWRU Bearing Dataset")
    canvas.drawRightString(width - 1.5*cm, 10, f"Page {doc.page}")

    canvas.restoreState()

# ============================================
# GENERATE REPORT
# ============================================
def generate_report(
    prediction,
    confidence,
    health_index,
    physics_validation,
    metrics,
    fault_freqs,
    feature_names,
    feature_values,
    output_path="results/engineering_report.pdf"
):
    os.makedirs("results", exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm
    )

    styles = get_styles()
    story  = []
    width  = 17*cm

    # ── TITLE PAGE ──────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("ENGINEERING INSPECTION REPORT", styles["ReportTitle"]))
    story.append(Paragraph("Physics-Guided Bearing Fault Diagnosis", styles["ReportSubtitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        styles["ReportSubtitle"]
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width=width, color=DARK_BLUE, thickness=2))
    story.append(Spacer(1, 0.5*cm))

    # ── EXECUTIVE SUMMARY ───────────────────────
    story.append(Paragraph("1. Executive Summary", styles["SectionHeader"]))

    fault_name = CLASS_NAMES.get(prediction, "Unknown")
    health_pct = round(health_index, 1)

    if health_pct >= 80:
        status = "GOOD"
        action = "Continue Monitoring"
    elif health_pct >= 50:
        status = "WARNING"
        action = "Inspect Soon"
    else:
        status = "CRITICAL"
        action = "Replace Bearing Immediately"

    style_key = f"Result{status.capitalize()}" if status != "GOOD" else "ResultGood"

    summary_data = [
        ["Parameter", "Value"],
        ["Detected Fault", fault_name],
        ["Model Confidence", f"{confidence*100:.1f}%"],
        ["Health Index", f"{health_pct:.1f}%"],
        ["Physics Confidence", f"{physics_validation.get('confidence', 0)*100:.1f}%"],
        ["System Status", status],
        ["Recommended Action", action],
    ]

    summary_table = Table(summary_data, colWidths=[7*cm, 10*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,0), 11),
        ("BACKGROUND",   (0,1), (0,-1), LIGHT_BLUE),
        ("FONTNAME",     (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,1), (-1,-1), 10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, GRAY]),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING",      (0,0), (-1,-1), 8),
        ("ALIGN",        (0,0), (-1,-1), "LEFT"),
        ("TEXTCOLOR",    (1,6), (1,6),
         GREEN if status=="GOOD" else ORANGE if status=="WARNING" else RED),
        ("FONTNAME",     (1,6), (1,6), "Helvetica-Bold"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.5*cm))

    # ── BEARING FAULT FREQUENCIES ───────────────
    story.append(HRFlowable(width=width, color=MID_BLUE, thickness=1))
    story.append(Paragraph("2. Bearing Kinematics & Fault Frequencies", styles["SectionHeader"]))
    story.append(Paragraph(
        "The following fault characteristic frequencies were computed from bearing geometry "
        "and shaft speed using ISO bearing kinematics formulas.",
        styles["BodyText2"]
    ))

    freq_data = [["Frequency", "Symbol", "Value (Hz)", "Description"]]
    freq_info = {
        "BPFO": "Ball Pass Frequency Outer Race",
        "BPFI": "Ball Pass Frequency Inner Race",
        "BSF":  "Ball Spin Frequency",
        "FTF":  "Fundamental Train Frequency",
        "shaft_hz": "Shaft Rotation Frequency"
    }
    for k, v in fault_freqs.items():
        freq_data.append([freq_info.get(k, k), k, f"{v:.2f}", "Fault indicator frequency"])

    freq_table = Table(freq_data, colWidths=[6*cm, 2.5*cm, 3*cm, 5.5*cm])
    freq_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), MID_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRAY]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.grey),
        ("FONTSIZE",      (0,1),(-1,-1), 9),
        ("PADDING",       (0,0),(-1,-1), 7),
    ]))
    story.append(freq_table)
    story.append(Spacer(1, 0.5*cm))

    # ── MODEL PERFORMANCE ───────────────────────
    story.append(HRFlowable(width=width, color=MID_BLUE, thickness=1))
    story.append(Paragraph("3. Model Performance Metrics", styles["SectionHeader"]))

    perf_data = [
        ["Metric", "Value", "Benchmark"],
        ["Accuracy",  f"{metrics.get('accuracy',0)*100:.2f}%",  "≥ 95%"],
        ["Precision", f"{metrics.get('precision',0)*100:.2f}%", "≥ 95%"],
        ["Recall",    f"{metrics.get('recall',0)*100:.2f}%",    "≥ 95%"],
        ["F1 Score",  f"{metrics.get('f1',0)*100:.2f}%",        "≥ 95%"],
    ]

    perf_table = Table(perf_data, colWidths=[5*cm, 5*cm, 7*cm])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), MID_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRAY]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.grey),
        ("FONTSIZE",      (0,0),(-1,-1), 10),
        ("PADDING",       (0,0),(-1,-1), 8),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("TEXTCOLOR",     (1,1),(-1,-1), GREEN),
        ("FONTNAME",      (1,1),(-1,-1), "Helvetica-Bold"),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.5*cm))

    # ── PHYSICS VALIDATION ──────────────────────
    story.append(HRFlowable(width=width, color=MID_BLUE, thickness=1))
    story.append(Paragraph("4. Physics Validation", styles["SectionHeader"]))
    story.append(Paragraph(
        "The AI prediction was cross-validated against physics-based bearing kinematics. "
        "Energy was analysed at fault characteristic frequencies, harmonics, and sidebands.",
        styles["BodyText2"]
    ))

    phys_conf = physics_validation.get("confidence", 0) * 100
    phys_msg  = physics_validation.get("message", "N/A")

    phys_data = [
        ["Physics Parameter", "Result"],
        ["Target Frequency",
         f"{physics_validation.get('target_freq', 'N/A')} Hz"
         if physics_validation.get('target_freq') else "N/A (Healthy)"],
        ["Physics Confidence", f"{phys_conf:.2f}%"],
        ["Validation Message", phys_msg],
        ["Harmonic Analysis",  "BPFO × 1,2,3 | BPFI × 1,2,3 | BSF × 1,2,3"],
        ["Sideband Analysis",  "Sidebands at fault_freq ± k × shaft_hz"],
    ]

    phys_table = Table(phys_data, colWidths=[6*cm, 11*cm])
    phys_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), MID_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND",    (0,1), (0,-1), LIGHT_BLUE),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRAY]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.grey),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("PADDING",       (0,0),(-1,-1), 7),
    ]))
    story.append(phys_table)
    story.append(Spacer(1, 0.5*cm))

    # ── PLOTS ───────────────────────────────────
    story.append(HRFlowable(width=width, color=MID_BLUE, thickness=1))
    story.append(Paragraph("5. Signal Analysis Plots", styles["SectionHeader"]))

    plot_files = [
        ("results/time_domain.png",       "Time Domain Signals - All Classes"),
        ("results/fft_comparison.png",    "FFT Comparison - All Classes"),
        ("results/confusion_matrix.png",  "Confusion Matrix"),
        ("results/fault_frequencies.png", "FFT with Fault Frequencies"),
        ("results/shap_bar.png",          "SHAP Feature Importance"),
        ("results/roc_curves.png",        "ROC Curves - All Classes"),
        ("results/health_index.png",      "Health Index per Segment"),
    ]

    for fpath, caption in plot_files:
        if os.path.exists(fpath):
            story.append(Paragraph(caption, styles["BodyText2"]))
            img = Image(fpath, width=16*cm, height=7*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))

    # ── TOP FEATURES ────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("6. Top Extracted Features", styles["SectionHeader"]))
    story.append(Paragraph(
        "47 features extracted across statistical, frequency, wavelet and physics domains.",
        styles["BodyText2"]
    ))

    feat_data = [["#", "Feature Name", "Value", "Domain"]]
    domains = (
        ["Statistical"]*11 +
        ["Frequency"]*8 +
        ["Wavelet"]*16 +
        ["Physics"]*12
    )

    for i, (name, val) in enumerate(zip(feature_names[:20], feature_values[:20])):
        domain = domains[i] if i < len(domains) else "Physics"
        feat_data.append([str(i), name, f"{float(val):.6f}", domain])

    feat_table = Table(feat_data, colWidths=[1*cm, 6*cm, 5*cm, 5*cm])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), MID_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, GRAY]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.grey),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("PADDING",       (0,0),(-1,-1), 5),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 0.5*cm))

    # ── MAINTENANCE RECOMMENDATION ──────────────
    story.append(HRFlowable(width=width, color=MID_BLUE, thickness=1))
    story.append(Paragraph("7. Maintenance Recommendation", styles["SectionHeader"]))

    rec_color = STATUS_COLORS.get(status, DARK_GRAY)
    rec_data  = [
        ["Field", "Details"],
        ["Fault Detected",        fault_name],
        ["Health Index",          f"{health_pct:.1f}%"],
        ["Status",                status],
        ["Recommended Action",    action],
        ["Next Inspection",
         "Within 24 hours" if status=="CRITICAL"
         else "Within 1 week" if status=="WARNING"
         else "Routine schedule"],
        ["Engineer Sign-off",     "___________________________"],
        ["Date",                  datetime.now().strftime("%Y-%m-%d")],
    ]

    rec_table = Table(rec_data, colWidths=[6*cm, 11*cm])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), DARK_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND",  (0,1), (0,-1), LIGHT_BLUE),
        ("FONTNAME",    (0,1), (0,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, GRAY]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("PADDING",     (0,0), (-1,-1), 8),
        ("TEXTCOLOR",   (1,3), (1,3),  rec_color),
        ("FONTNAME",    (1,3), (1,3),  "Helvetica-Bold"),
        ("FONTSIZE",    (1,3), (1,3),  12),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 1*cm))

    # ── FOOTER NOTE ─────────────────────────────
    story.append(HRFlowable(width=width, color=DARK_BLUE, thickness=1))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "This report was automatically generated by the Physics-Aware Predictive Maintenance System. "
        "Dataset: CWRU Bearing Dataset (Case Western Reserve University). "
        "Models: Random Forest, SVM, XGBoost. "
        "Physics Engine: Bearing kinematics + harmonic + sideband analysis.",
        styles["SmallText"]
    ))

    # ── BUILD ───────────────────────────────────
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(f"Report saved: {output_path}")
    return output_path