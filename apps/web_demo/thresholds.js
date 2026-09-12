/**
 * AuraPulse Clinical Thresholds & Evaluation Ruleset.
 * Centralized clinical bounds mapping physiological biomarkers to color-coded states.
 */

const CLINICAL_THRESHOLDS = {
  heartRate: {
    name: "Heart Rate",
    unit: "BPM",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", subtext: "Awaiting signal lock" };
      }
      if (val > 115) {
        return { state: "critical", label: "🔴 High Alert", color: "#ef4444", subtext: "Severe Tachycardia alert (>115 BPM)" };
      }
      if (val < 50) {
        return { state: "critical", label: "🔴 High Alert", color: "#ef4444", subtext: "Severe Bradycardia alert (<50 BPM)" };
      }
      if (val > 100 || val < 60) {
        return { state: "warning", label: "🟡 Borderline", color: "#eab308", subtext: val > 100 ? "Elevated resting heart rate" : "Mild resting bradycardia" };
      }
      return { state: "optimal", label: "🟢 Normal", color: "#10b981", subtext: "Optimal resting range (60-100 BPM)" };
    }
  },

  respirationRate: {
    name: "Respiration Rate",
    unit: "RPM",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", subtext: "Awaiting respiratory rhythm" };
      }
      if (val < 9) {
        return { state: "critical", label: "🔴 Bradypnea", color: "#ef4444", subtext: "Critical low breathing rate (<9 RPM)" };
      }
      if (val > 24) {
        return { state: "critical", label: "🔴 Tachypnea", color: "#ef4444", subtext: "Critical high breathing rate (>24 RPM)" };
      }
      if (val < 12 || val > 20) {
        return { state: "warning", label: "🟡 Borderline", color: "#eab308", subtext: val > 20 ? "Mildly elevated breathing" : "Slow breathing rhythm" };
      }
      return { state: "optimal", label: "🟢 Normal", color: "#10b981", subtext: "Healthy eupnea range (12-20 RPM)" };
    }
  },

  rmssd: {
    name: "Pulse Rate Variability (RMSSD)",
    unit: "ms",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", subtext: "Computing sub-sample PPIs" };
      }
      if (val < 20) {
        return { state: "critical", label: "🔴 High Stress", color: "#ef4444", subtext: "Low autonomic resilience (<20 ms)" };
      }
      if (val < 50) {
        return { state: "warning", label: "🟡 Moderate", color: "#eab308", subtext: "Moderate parasympathetic tone (20-50 ms)" };
      }
      return { state: "optimal", label: "🟢 Good Recovery", color: "#10b981", subtext: "Robust vagal parasympathetic tone (>50 ms)" };
    }
  },

  stressIndex: {
    name: "Physiological Stress Score",
    unit: "/ 100",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val === "--") {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", subtext: "Evaluating autonomic balance" };
      }
      const num = parseFloat(val);
      if (num >= 61) {
        return { state: "critical", label: "🔴 High Stress", color: "#ef4444", subtext: "High sympathetic autonomic load" };
      }
      if (num >= 26) {
        return { state: "warning", label: "🟡 Moderate", color: "#eab308", subtext: "Moderate sympathetic arousal" };
      }
      return { state: "optimal", label: "🟢 Low Stress", color: "#10b981", subtext: "Optimal relaxed homeostatic state" };
    }
  },

  bloodPressure: {
    name: "Blood Pressure",
    unit: "mmHg",
    evaluate: (sbp, dbp) => {
      if (!sbp || !dbp || isNaN(sbp) || isNaN(dbp) || sbp <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", subtext: "Analyzing SDPPG arterial contour" };
      }
      if (sbp >= 140 || dbp >= 90) {
        return { state: "critical", label: "🔴 Hypertension", color: "#ef4444", subtext: "Stage 1/2 Hypertension threshold (≥140/90)" };
      }
      if (sbp >= 120 || dbp >= 80) {
        return { state: "warning", label: "🟡 Elevated", color: "#eab308", subtext: "Prehypertension / Elevated vascular tone" };
      }
      return { state: "optimal", label: "🟢 Optimal", color: "#10b981", subtext: "Normal hemodynamic pressure (<120/<80)" };
    }
  },

  spo2: {
    name: "Oxygen Saturation",
    unit: "%",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", subtext: "Multi-spectral optical ratio" };
      }
      if (val < 90) {
        return { state: "critical", label: "🔴 Hypoxemia", color: "#ef4444", subtext: "Critical blood oxygen desaturation (<90%)" };
      }
      if (val < 95) {
        return { state: "warning", label: "🟡 Mild Hypoxia", color: "#eab308", subtext: "Borderline oxygenation (90-94%)" };
      }
      return { state: "optimal", label: "🟢 Normal", color: "#10b981", subtext: "Optimal arterial saturation (95-100%)" };
    }
  },

  cardiacWorkload: {
    name: "Cardiac Workload (RPP)",
    unit: "RPP",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", subtext: "Rate-Pressure Product" };
      }
      if (val > 125) {
        return { state: "critical", label: "🔴 High Load", color: "#ef4444", subtext: "Elevated myocardial oxygen demand (>125)" };
      }
      if (val > 100) {
        return { state: "warning", label: "🟡 Moderate", color: "#eab308", subtext: "Moderate cardiac workload (101-125)" };
      }
      return { state: "optimal", label: "🟢 Normal", color: "#10b981", subtext: "Healthy resting myocardial workload (≤100)" };
    }
  },

  cvdRisk: {
    name: "10-Year CVD Risk",
    unit: "%",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", subtext: "Framingham / WHO-ISH model" };
      }
      if (val >= 20.0) {
        return { state: "critical", label: "🔴 High Risk", color: "#ef4444", subtext: "High 10-year cardiovascular projection (≥20%)" };
      }
      if (val >= 10.0) {
        return { state: "warning", label: "🟡 Moderate", color: "#eab308", subtext: "Moderate 10-year risk profile (10-20%)" };
      }
      return { state: "optimal", label: "🟢 Low Risk", color: "#10b981", subtext: "Low 10-year cardiovascular projection (<10%)" };
    }
  }
};

if (typeof window !== "undefined") {
  window.CLINICAL_THRESHOLDS = CLINICAL_THRESHOLDS;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = CLINICAL_THRESHOLDS;
}
