import numpy as np

CLASS_NAMES = {
    0: "Healthy",
    1: "Ball Fault",
    2: "Inner Race Fault",
    3: "Outer Race Fault"
}

# ============================================
# HEALTH INDEX
# ============================================
def compute_health_index(proba):
    """
    Computes health index 0-100 based on
    probability of being Healthy class (class 0).
    """
    healthy_prob = proba[:, 0]
    health_index = healthy_prob * 100
    return health_index

# ============================================
# HEALTH STATUS
# ============================================
def health_status(health_index):
    if health_index >= 80:
        return "🟢 GOOD"
    elif health_index >= 50:
        return "🟡 WARNING"
    else:
        return "🔴 CRITICAL"

# ============================================
# MAINTENANCE RECOMMENDATION
# ============================================
def maintenance_recommendation(health_index, predicted_class):
    fault = CLASS_NAMES.get(predicted_class, "Unknown")

    if health_index >= 80:
        return {
            "action": "✅ Continue Monitoring",
            "fault": fault,
            "health": round(health_index, 2),
            "status": "GOOD",
            "message": "Bearing is healthy. No action needed."
        }
    elif health_index >= 50:
        return {
            "action": "⚠️  Inspect Soon",
            "fault": fault,
            "health": round(health_index, 2),
            "status": "WARNING",
            "message": f"Early signs of {fault} detected. Schedule inspection."
        }
    else:
        return {
            "action": "🔴 Replace Bearing Immediately",
            "fault": fault,
            "health": round(health_index, 2),
            "status": "CRITICAL",
            "message": f"Severe {fault} detected. Immediate replacement required."
        }

# ============================================
# BATCH HEALTH REPORT
# ============================================
def batch_health_report(probas, predictions):
    health_indices = compute_health_index(probas)
    report = []
    for i, (hi, pred) in enumerate(zip(health_indices, predictions)):
        rec = maintenance_recommendation(hi, pred)
        rec["segment"] = i
        report.append(rec)
    return report

# ============================================
# SUMMARY STATISTICS
# ============================================
def health_summary(report):
    statuses = [r["status"] for r in report]
    total = len(statuses)
    good     = statuses.count("GOOD")
    warning  = statuses.count("WARNING")
    critical = statuses.count("CRITICAL")

    print("=" * 40)
    print("HEALTH SUMMARY")
    print("=" * 40)
    print(f"Total segments : {total}")
    print(f"Good           : {good}  ({100*good/total:.1f}%)")
    print(f"Warning        : {warning}  ({100*warning/total:.1f}%)")
    print(f"Critical       : {critical}  ({100*critical/total:.1f}%)")
    print("=" * 40)

    return {
        "total": total,
        "good": good,
        "warning": warning,
        "critical": critical
    }