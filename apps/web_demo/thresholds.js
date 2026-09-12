/**
 * AuraPulse Standardized Clinical Thresholds & 3-Tier Badging Engine.
 * 🟢 Normal (Optimal / Within range)
 * 🟡 Moderate (Elevated / Borderline / Monitor)
 * 🔴 Alert (High Risk / Low / Out of normal limits)
 */

const CLINICAL_THRESHOLDS = {
  heartRate: {
    name: "Heart Rate",
    unit: "BPM",
    normalRange: "60 - 100 BPM",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Awaiting optical lock" };
      }
      if (val > 115) {
        return { state: "critical", label: "High Alert", color: "#ef4444", badge: "🔴 Alert", subtext: "Severe Tachycardia alert (>115 BPM)" };
      }
      if (val < 50) {
        return { state: "critical", label: "High Alert", color: "#ef4444", badge: "🔴 Alert", subtext: "Severe Bradycardia alert (<50 BPM)" };
      }
      if (val > 100 || val < 60) {
        return { state: "warning", label: "Moderate", color: "#eab308", badge: "🟡 Moderate", subtext: val > 100 ? "Elevated resting heart rate (101-115 BPM)" : "Mild resting bradycardia (50-59 BPM)" };
      }
      return { state: "optimal", label: "Normal", color: "#10b981", badge: "🟢 Normal", subtext: "Optimal physiological resting range (60-100 BPM)" };
    }
  },

  rmssd: {
    name: "HRV (RMSSD)",
    unit: "ms",
    normalRange: "> 50 ms (Age 20-40: 55-105 ms)",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Extracting PPI variance" };
      }
      if (val < 20) {
        return { state: "critical", label: "Alert", color: "#ef4444", badge: "🔴 Alert", subtext: "Depressed autonomic vagal tone (<20 ms)" };
      }
      if (val < 50) {
        return { state: "warning", label: "Moderate", color: "#eab308", badge: "🟡 Moderate", subtext: "Moderate parasympathetic reserves (20-50 ms)" };
      }
      return { state: "optimal", label: "Normal", color: "#10b981", badge: "🟢 Normal", subtext: "Robust vagal recovery & autonomic resilience (>50 ms)" };
    }
  },

  sdnn: {
    name: "HRV (SDNN)",
    unit: "ms",
    normalRange: "> 50 ms",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Computing total variance" };
      }
      if (val < 30) {
        return { state: "critical", label: "Alert", color: "#ef4444", badge: "🔴 Alert", subtext: "Significantly low overall HRV (<30 ms)" };
      }
      if (val < 50) {
        return { state: "warning", label: "Moderate", color: "#eab308", badge: "🟡 Moderate", subtext: "Moderate overall HRV capacity (30-50 ms)" };
      }
      return { state: "optimal", label: "Normal", color: "#10b981", badge: "🟢 Normal", subtext: "Optimal total autonomic regulatory capacity (>50 ms)" };
    }
  },

  respirationRate: {
    name: "Respiration Rate",
    unit: "RPM",
    normalRange: "12 - 20 RPM",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Awaiting breathing rhythm" };
      }
      if (val < 9 || val > 24) {
        return { state: "critical", label: "Alert", color: "#ef4444", badge: "🔴 Alert", subtext: val > 24 ? "Marked Tachypnea (>24 RPM)" : "Severe Bradypnea (<9 RPM)" };
      }
      if (val < 12 || val > 20) {
        return { state: "warning", label: "Moderate", color: "#eab308", badge: "🟡 Moderate", subtext: val > 20 ? "Mildly elevated breathing rhythm" : "Borderline low breathing rhythm" };
      }
      return { state: "optimal", label: "Normal", color: "#10b981", badge: "🟢 Normal", subtext: "Healthy eupnea breathing range (12-20 RPM)" };
    }
  },

  bloodPressure: {
    name: "Blood Pressure",
    unit: "mmHg",
    normalRange: "Systolic < 120, Diastolic < 80 mmHg",
    evaluate: (sbp, dbp) => {
      if (!sbp || !dbp || isNaN(sbp) || isNaN(dbp) || sbp <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Analyzing arterial contour" };
      }
      if (sbp >= 140 || dbp >= 90) {
        return { state: "critical", label: "Hypertension Alert", color: "#ef4444", badge: "🔴 Alert", subtext: "Stage 1/2 Hypertension threshold (≥140/90 mmHg)" };
      }
      if (sbp >= 120 || dbp >= 80) {
        return { state: "warning", label: "Elevated BP", color: "#eab308", badge: "🟡 Moderate", subtext: "Prehypertension / Elevated vascular tone (120-139 / 80-89)" };
      }
      return { state: "optimal", label: "Normal", color: "#10b981", badge: "🟢 Normal", subtext: "Optimal hemodynamic pressure (<120/<80 mmHg)" };
    }
  },

  spo2: {
    name: "Oxygen Saturation",
    unit: "%",
    normalRange: "95% - 100%",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Dual-ratio optical calculation" };
      }
      if (val < 90) {
        return { state: "critical", label: "Hypoxemia Alert", color: "#ef4444", badge: "🔴 Alert", subtext: "Critical blood oxygen desaturation (<90%)" };
      }
      if (val < 95) {
        return { state: "warning", label: "Mild Hypoxia", color: "#eab308", badge: "🟡 Moderate", subtext: "Borderline peripheral oxygenation (90-94%)" };
      }
      return { state: "optimal", label: "Normal", color: "#10b981", badge: "🟢 Normal", subtext: "Optimal arterial saturation (95-100%)" };
    }
  },

  prq: {
    name: "Pulse-Respiration Quotient (PRQ)",
    unit: "ratio",
    normalRange: "3.5 - 5.0",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "HR / RR synchronization" };
      }
      if (val < 3.0 || val > 5.5) {
        return { state: "critical", label: "Alert", color: "#ef4444", badge: "🔴 Alert", subtext: val > 5.5 ? "Cardiorespiratory dyssynchrony / High HR load" : "Respiratory dominance / Bradypnea" };
      }
      if (val < 3.5 || val > 5.0) {
        return { state: "warning", label: "Moderate", color: "#eab308", badge: "🟡 Moderate", subtext: "Borderline cardiorespiratory coupling (3.0-3.5 or 5.0-5.5)" };
      }
      return { state: "optimal", label: "Normal", color: "#10b981", badge: "🟢 Normal", subtext: "Balanced cardiorespiratory synchronization (3.5 - 5.0)" };
    }
  },

  baevskyStress: {
    name: "Baevsky Stress Index",
    unit: "SI",
    normalRange: "50 - 150 (Resting Eustress)",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Histogram mode extraction" };
      }
      if (val > 300) {
        return { state: "critical", label: "High Distress", color: "#ef4444", badge: "🔴 Alert", subtext: "High sympathoadrenal activation & regulatory tension (>300)" };
      }
      if (val > 150) {
        return { state: "warning", label: "Moderate Tension", color: "#eab308", badge: "🟡 Moderate", subtext: "Moderate compensatory autonomic stress (151-300)" };
      }
      return { state: "optimal", label: "Normal Eustress", color: "#10b981", badge: "🟢 Normal", subtext: "Optimal calm homeostatic balance (50-150)" };
    }
  },

  pnsRecovery: {
    name: "PNS Recovery Tone",
    unit: "/ 100",
    normalRange: "50 - 100",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Evaluating vagal tone" };
      }
      if (val < 25) {
        return { state: "critical", label: "Suppressed", color: "#ef4444", badge: "🔴 Alert", subtext: "Significantly blunted parasympathetic rest tone (<25)" };
      }
      if (val < 50) {
        return { state: "warning", label: "Moderate", color: "#eab308", badge: "🟡 Moderate", subtext: "Moderate vagal recovery reserves (25-49)" };
      }
      return { state: "optimal", label: "Optimal Vagal", color: "#10b981", badge: "🟢 Normal", subtext: "Robust parasympathetic regenerative capacity (≥50)" };
    }
  },

  snsZone: {
    name: "SNS Arousal Zone",
    unit: "/ 100",
    normalRange: "< 35 (Calm)",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Evaluating sympathetic drive" };
      }
      if (val > 65) {
        return { state: "critical", label: "High Sympathetic", color: "#ef4444", badge: "🔴 Alert", subtext: "Elevated fight-or-flight adrenergic arousal (>65)" };
      }
      if (val >= 35) {
        return { state: "warning", label: "Moderate Sympathetic", color: "#eab308", badge: "🟡 Moderate", subtext: "Moderate sympathetic arousal (35-65)" };
      }
      return { state: "optimal", label: "Calm Sympathetic", color: "#10b981", badge: "🟢 Normal", subtext: "Optimal low resting sympathetic tone (<35)" };
    }
  },

  lfHfRatio: {
    name: "Autonomic LF/HF Ratio",
    unit: "ratio",
    normalRange: "0.8 - 2.5",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Spectral decomposition" };
      }
      if (val > 4.0 || val < 0.4) {
        return { state: "critical", label: "Alert", color: "#ef4444", badge: "🔴 Alert", subtext: val > 4.0 ? "Severe sympathetic dominance (>4.0)" : "Extreme parasympathetic predominance (<0.4)" };
      }
      if (val > 2.5 || val < 0.8) {
        return { state: "warning", label: "Moderate Shift", color: "#eab308", badge: "🟡 Moderate", subtext: "Mild sympathovagal asymmetry (0.4-0.8 or 2.6-4.0)" };
      }
      return { state: "optimal", label: "Normal Balance", color: "#10b981", badge: "🟢 Normal", subtext: "Well-balanced sympathovagal equilibrium (0.8 - 2.5)" };
    }
  },

  cardiacWorkload: {
    name: "Cardiac Workload (RPP)",
    unit: "RPP",
    normalRange: "≤ 100 (Resting)",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Rate-Pressure Product" };
      }
      if (val > 125) {
        return { state: "critical", label: "High Workload", color: "#ef4444", badge: "🔴 Alert", subtext: "Elevated myocardial oxygen consumption (>125)" };
      }
      if (val > 100) {
        return { state: "warning", label: "Moderate Workload", color: "#eab308", badge: "🟡 Moderate", subtext: "Moderate myocardial oxygen demand (101-125)" };
      }
      return { state: "optimal", label: "Normal Resting", color: "#10b981", badge: "🟢 Normal", subtext: "Healthy baseline myocardial load (≤100)" };
    }
  },

  vascularAge: {
    name: "Vascular Heart Age Delta",
    unit: "years",
    normalRange: "Δ ≤ +1.0 years",
    evaluate: (ageDelta) => {
      if (ageDelta === null || ageDelta === undefined || isNaN(ageDelta)) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Arterial compliance curve" };
      }
      if (ageDelta > 5.0) {
        return { state: "critical", label: "Accelerated Stiffening", color: "#ef4444", badge: "🔴 Alert", subtext: `Arterial stiffness exceeds age by +${ageDelta.toFixed(1)} yrs` };
      }
      if (ageDelta > 1.0) {
        return { state: "warning", label: "Mild Stiffening", color: "#eab308", badge: "🟡 Moderate", subtext: `Arterial tone +${ageDelta.toFixed(1)} yrs above chronological` };
      }
      return { state: "optimal", label: "Optimal Compliance", color: "#10b981", badge: "🟢 Normal", subtext: `Arterial compliance youthfully aligned (${ageDelta >= 0 ? '+' : ''}${ageDelta.toFixed(1)} yrs)` };
    }
  },

  cvdRisk: {
    name: "10-Year ASCVD Event Risk",
    unit: "%",
    normalRange: "< 10% (Low Risk)",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Framingham Cox model" };
      }
      if (val >= 20.0) {
        return { state: "critical", label: "High ASCVD Risk", color: "#ef4444", badge: "🔴 Alert", subtext: "High 10-year cardiovascular event projection (≥20%)" };
      }
      if (val >= 10.0) {
        return { state: "warning", label: "Moderate ASCVD Risk", color: "#eab308", badge: "🟡 Moderate", subtext: "Moderate 10-year risk profile (10-20%)" };
      }
      return { state: "optimal", label: "Low ASCVD Risk", color: "#10b981", badge: "🟢 Normal", subtext: "Low 10-year cardiovascular event risk (<10%)" };
    }
  },

  fbg: {
    name: "Fasting Glucose Risk",
    unit: "mg/dL",
    normalRange: "< 100 mg/dL",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Metabolic risk projection" };
      }
      if (val > 125.0) {
        return { state: "critical", label: "Elevated Glucose Risk", color: "#ef4444", badge: "🔴 Alert", subtext: "Estimated FBG > 125 mg/dL (Diabetes range)" };
      }
      if (val >= 100.0) {
        return { state: "warning", label: "Impaired Glucose Risk", color: "#eab308", badge: "🟡 Moderate", subtext: "Estimated FBG 100-125 mg/dL (Prediabetes range)" };
      }
      return { state: "optimal", label: "Normal Glucose Risk", color: "#10b981", badge: "🟢 Normal", subtext: "Estimated FBG < 100 mg/dL (Optimal range)" };
    }
  },

  hba1c: {
    name: "HbA1c Glycemic Risk",
    unit: "%",
    normalRange: "< 5.7%",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Glycated hemoglobin proxy" };
      }
      if (val > 6.4) {
        return { state: "critical", label: "Diabetes Risk Tier", color: "#ef4444", badge: "🔴 Alert", subtext: "Estimated HbA1c > 6.4% (Diabetic tier)" };
      }
      if (val >= 5.7) {
        return { state: "warning", label: "Prediabetes Risk Tier", color: "#eab308", badge: "🟡 Moderate", subtext: "Estimated HbA1c 5.7 - 6.4% (Prediabetic tier)" };
      }
      return { state: "optimal", label: "Normal Glycemic Tier", color: "#10b981", badge: "🟢 Normal", subtext: "Estimated HbA1c < 5.7% (Normal tier)" };
    }
  },

  hemoglobin: {
    name: "Hemoglobin (Hb)",
    unit: "g/dL",
    normalRange: "Men: 14-18, Women: 12-16 g/dL",
    evaluate: (val, isMale = true) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Multi-spectral optical absorption" };
      }
      const lowThresh = isMale ? 13.5 : 12.0;
      const highThresh = isMale ? 18.0 : 16.0;
      if (val < lowThresh) {
        return { state: "critical", label: "Low (Anemia Warning)", color: "#ef4444", badge: "🔴 Alert", subtext: `Estimated Hb ${val.toFixed(1)} g/dL below ${lowThresh} g/dL threshold` };
      }
      if (val > highThresh) {
        return { state: "warning", label: "Elevated Hb", color: "#eab308", badge: "🟡 Moderate", subtext: `Estimated Hb ${val.toFixed(1)} g/dL above ${highThresh} g/dL reference` };
      }
      return { state: "optimal", label: "Normal Hb", color: "#10b981", badge: "🟢 Normal", subtext: `Optimal blood hemoglobin level (${val.toFixed(1)} g/dL)` };
    }
  },

  bmi: {
    name: "Body Mass Index (BMI)",
    unit: "kg/m²",
    normalRange: "18.5 - 24.9 kg/m²",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "neutral", label: "Waiting", color: "#94a3b8", badge: "⚪ Waiting", subtext: "Anthropometric ratio" };
      }
      if (val >= 30.0 || val < 18.5) {
        return { state: "critical", label: val < 18.5 ? "Underweight" : "Obese Tier", color: "#ef4444", badge: "🔴 Alert", subtext: val < 18.5 ? "BMI < 18.5 kg/m²" : "BMI ≥ 30.0 kg/m² (Increased chronic risk)" };
      }
      if (val >= 25.0) {
        return { state: "warning", label: "Overweight Tier", color: "#eab308", badge: "🟡 Moderate", subtext: "BMI 25.0 - 29.9 kg/m² (Borderline metabolic load)" };
      }
      return { state: "optimal", label: "Normal BMI", color: "#10b981", badge: "🟢 Normal", subtext: "Healthy anthropometric weight range (18.5 - 24.9 kg/m²)" };
    }
  }
};

if (typeof window !== "undefined") {
  window.CLINICAL_THRESHOLDS = CLINICAL_THRESHOLDS;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = CLINICAL_THRESHOLDS;
}
