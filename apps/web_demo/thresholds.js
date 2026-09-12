/**
 * AuraPulse 4.0 Standardized Clinical Thresholds & Badging Ruleset.
 *
 * Color Specification:
 * - Normal:        #22C55E  [🟢 Normal]
 * - Elevated:      #F59E0B  [🟡 Elevated / Borderline]
 * - Abnormal:      #EF4444  [🔴 Alert / Out of limits]
 * - Unavailable:   #64748B  [⚪ Unavailable]
 * - Experimental:  #8B5CF6  [🔬 Experimental / Not Validated]
 * - Informational: #06B6D4  [🔵 Physiological Ratio]
 */

const CLINICAL_THRESHOLDS = {
  heartRate: {
    name: "Heart Rate",
    unit: "BPM",
    normalRange: "60 - 100 BPM (Resting Adult)",
    validationStatus: "PRIMARY_MVP",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Awaiting optical pulse lock" };
      }
      if (val > 115) {
        return { state: "critical", label: "Tachycardia Alert", color: "#EF4444", badge: "🔴 Alert", subtext: "Marked resting tachycardia (>115 BPM)" };
      }
      if (val < 50) {
        return { state: "critical", label: "Bradycardia Alert", color: "#EF4444", badge: "🔴 Alert", subtext: "Marked resting bradycardia (<50 BPM)" };
      }
      if (val > 100 || val < 60) {
        return { state: "warning", label: "Borderline", color: "#F59E0B", badge: "🟡 Borderline", subtext: val > 100 ? "Elevated resting pulse (101-115 BPM)" : "Mild resting bradycardia (50-59 BPM)" };
      }
      return { state: "optimal", label: "Normal", color: "#22C55E", badge: "🟢 Normal", subtext: "Typical adult resting baseline (60-100 BPM)" };
    }
  },

  rmssd: {
    name: "Pulse Rate Variability (PRV - RMSSD)",
    unit: "ms",
    normalRange: "Contextual (Age 20-25: 55-105 ms; Age 60-65: 25-45 ms)",
    validationStatus: "PRIMARY_MVP",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Extracting optical PPI variance" };
      }
      if (val < 20) {
        return { state: "critical", label: "Low Variability", color: "#EF4444", badge: "🔴 Low", subtext: "Depressed optical pulse interval variability (<20 ms)" };
      }
      if (val < 45) {
        return { state: "warning", label: "Moderate", color: "#F59E0B", badge: "🟡 Moderate", subtext: "Moderate pulse rate variability (20-45 ms)" };
      }
      return { state: "optimal", label: "Optimal", color: "#22C55E", badge: "🟢 Optimal", subtext: "Robust pulse interval variability (>45 ms)" };
    }
  },

  sdnn: {
    name: "Pulse Rate Variability (PRV - SDNN)",
    unit: "ms",
    normalRange: "> 45 ms",
    validationStatus: "PRIMARY_MVP",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Calculating total variance" };
      }
      if (val < 25) {
        return { state: "critical", label: "Low", color: "#EF4444", badge: "🔴 Low", subtext: "Low total pulse interval standard deviation (<25 ms)" };
      }
      if (val < 45) {
        return { state: "warning", label: "Moderate", color: "#F59E0B", badge: "🟡 Moderate", subtext: "Moderate total variability capacity (25-45 ms)" };
      }
      return { state: "optimal", label: "Normal", color: "#22C55E", badge: "🟢 Normal", subtext: "Healthy total variability capacity (>45 ms)" };
    }
  },

  respirationRate: {
    name: "Respiration Rate",
    unit: "breaths/min",
    normalRange: "12 - 20 breaths/min",
    validationStatus: "PRIMARY_MVP",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Awaiting breathing rhythm lock" };
      }
      if (val < 9 || val > 24) {
        return { state: "critical", label: val > 24 ? "Tachypnea" : "Bradypnea", color: "#EF4444", badge: "🔴 Alert", subtext: val > 24 ? "Elevated breathing rate (>24 breaths/min)" : "Slow breathing rate (<9 breaths/min)" };
      }
      if (val < 12 || val > 20) {
        return { state: "warning", label: "Borderline", color: "#F59E0B", badge: "🟡 Borderline", subtext: val > 20 ? "Mildly elevated breathing rhythm" : "Slow breathing rhythm" };
      }
      return { state: "optimal", label: "Normal", color: "#22C55E", badge: "🟢 Normal", subtext: "Typical resting breathing frequency (12-20 breaths/min)" };
    }
  },

  prq: {
    name: "Pulse-Respiration Quotient (PRQ)",
    unit: "ratio",
    normalRange: "3.5 - 5.0 (Resting baseline)",
    validationStatus: "PHYSIOLOGICAL_RATIO",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Requires valid simultaneous HR & RR" };
      }
      if (val < 3.0 || val > 5.5) {
        return { state: "warning", label: "Atypical Ratio", color: "#F59E0B", badge: "🟡 Deviated", subtext: val > 5.5 ? "Elevated pulse-to-breath quotient" : "Low pulse-to-breath quotient" };
      }
      return { state: "optimal", label: "Balanced", color: "#22C55E", badge: "🟢 Balanced", subtext: "Expected resting cardiorespiratory coupling (3.5 - 5.0)" };
    }
  },

  arrhythmia: {
    name: "Pulse Rhythm & AFib Screening",
    unit: "% CV",
    normalRange: "Regular Sinus Rhythm (CV < 12%)",
    validationStatus: "RESEARCH_ONLY",
    evaluate: (isIrregular, cvPct, status) => {
      if (cvPct === null || cvPct === undefined || isNaN(cvPct)) {
        return { state: "unavailable", label: "Analyzing Rhythm", color: "#64748B", badge: "⚪ In Progress", subtext: "Extracting Poincaré IBI scatter" };
      }
      if (isIrregular || cvPct >= 12.0) {
        return { state: "critical", label: "Irregular Rhythm", color: "#EF4444", badge: "🔴 Irregular (AFib Risk)", subtext: `Elevated beat interval dispersion (${cvPct.toFixed(1)}% CV)` };
      }
      if (cvPct >= 8.0) {
        return { state: "warning", label: "Borderline", color: "#F59E0B", badge: "🟡 Borderline", subtext: `Mild interval variability (${cvPct.toFixed(1)}% CV)` };
      }
      return { state: "optimal", label: "Regular Sinus", color: "#22C55E", badge: "🟢 Regular Rhythm", subtext: `Normal rhythmic interval pacing (${cvPct.toFixed(1)}% CV)` };
    }
  },

  vo2Max: {
    name: "Cardiorespiratory Fitness (VO2 max)",
    unit: "mL/kg/min",
    normalRange: "Normative (Age/Gender Stratified)",
    validationStatus: "MODEL_DEPENDENT",
    evaluate: (val, tier) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Non-exercise Jackson-Pollock regression" };
      }
      if (tier === "Superior" || tier === "Excellent") {
        return { state: "optimal", label: tier, color: "#22C55E", badge: `🟢 ${tier}`, subtext: `${val.toFixed(1)} mL/kg/min (High Aerobic Capacity)` };
      }
      if (tier === "Good") {
        return { state: "optimal", label: "Good", color: "#22C55E", badge: "🟢 Good", subtext: `${val.toFixed(1)} mL/kg/min (Average Aerobic Capacity)` };
      }
      if (tier === "Fair") {
        return { state: "warning", label: "Fair", color: "#F59E0B", badge: "🟡 Fair", subtext: `${val.toFixed(1)} mL/kg/min (Moderate Capacity)` };
      }
      return { state: "critical", label: "Poor", color: "#EF4444", badge: "🔴 Poor", subtext: `${val.toFixed(1)} mL/kg/min (Low Aerobic Fitness)` };
    }
  },

  coherence: {
    name: "Cardiorespiratory Coherence",
    unit: "%",
    normalRange: "> 40% (Coherent Breathing)",
    validationStatus: "RESEARCH_ONLY",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Cross-spectral respiratory-HRV coupling" };
      }
      if (val >= 70.0) {
        return { state: "optimal", label: "High Coherence", color: "#22C55E", badge: "🟢 High (70-100%)", subtext: "Strong autonomic-respiratory phase sync" };
      }
      if (val >= 40.0) {
        return { state: "warning", label: "Moderate", color: "#F59E0B", badge: "🟡 Moderate (40-69%)", subtext: "Moderate cardiorespiratory coupling" };
      }
      return { state: "critical", label: "Low", color: "#64748B", badge: "⚪ Low (<40%)", subtext: "Desynchronized autonomic rhythm" };
    }
  },

  baevskyStress: {
    name: "Baevsky Stress Index",
    unit: "SI",
    normalRange: "50 - 150 SI (Resting Eustress)",
    validationStatus: "RESEARCH_ONLY",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Evaluating 50ms histogram mode" };
      }
      if (val > 300) {
        return { state: "experimental", label: "High Tension", color: "#8B5CF6", badge: "🔬 High Load", subtext: "High regulatory tension index (>300 SI)" };
      }
      if (val > 150) {
        return { state: "experimental", label: "Moderate Tension", color: "#8B5CF6", badge: "🔬 Moderate", subtext: "Compensatory regulatory load (151-300 SI)" };
      }
      return { state: "experimental", label: "Eustress", color: "#8B5CF6", badge: "🔬 Eustress", subtext: "Optimal resting homeostatic balance (50-150 SI)" };
    }
  },

  lfHfRatio: {
    name: "Autonomic Spectral LF/HF",
    unit: "ratio",
    normalRange: "0.8 - 2.5",
    validationStatus: "RESEARCH_ONLY",
    evaluate: (val) => {
      if (val === null || val === undefined || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Spectral decomposition" };
      }
      return { state: "experimental", label: "Spectral Feature", color: "#8B5CF6", badge: "🔬 Research", subtext: `LF (0.04-0.15Hz) / HF (0.15-0.40Hz) = ${val.toFixed(2)}` };
    }
  },

  bloodPressure: {
    name: "Blood Pressure (SDPPG)",
    unit: "mmHg",
    normalRange: "Reference: Systolic < 120, Diastolic < 80 mmHg",
    validationStatus: "NOT_VALIDATED",
    evaluate: (sbp, dbp) => {
      if (!sbp || !dbp || isNaN(sbp) || isNaN(dbp) || sbp <= 0) {
        return { state: "unavailable", label: "Not Validated", color: "#64748B", badge: "⚪ Not Validated", subtext: "Camera BP is experimental / not clinically validated" };
      }
      return { state: "experimental", label: "Experimental", color: "#8B5CF6", badge: "🔬 Experimental", subtext: `Contour estimate: ${Math.round(sbp)}/${Math.round(dbp)} mmHg (Not a clinical measurement)` };
    }
  },

  spo2: {
    name: "Oxygen Saturation (SpO2)",
    unit: "%",
    normalRange: "Reference: 95% - 100%",
    validationStatus: "NOT_VALIDATED",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Not Validated", color: "#64748B", badge: "⚪ Not Validated", subtext: "Multi-spectral optical estimation is experimental" };
      }
      return { state: "experimental", label: "Experimental", color: "#8B5CF6", badge: "🔬 Experimental", subtext: `Chromatic estimate: ${val.toFixed(1)}% (Not a clinical pulse oximeter)` };
    }
  },

  hemoglobin: {
    name: "Hemoglobin (Hb)",
    unit: "g/dL",
    normalRange: "Reference: Men 14-18, Women 12-16 g/dL",
    validationStatus: "NOT_VALIDATED",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Not Validated", color: "#64748B", badge: "⚪ Not Validated", subtext: "Optical density model is experimental" };
      }
      return { state: "experimental", label: "Experimental", color: "#8B5CF6", badge: "🔬 Experimental", subtext: `Extinction estimate: ${val.toFixed(1)} g/dL (Not a lab blood test)` };
    }
  },

  fbg: {
    name: "Fasting Glucose Risk",
    unit: "mg/dL",
    normalRange: "Target: < 100 mg/dL",
    validationStatus: "NOT_VALIDATED",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Not Validated", color: "#64748B", badge: "⚪ Not Validated", subtext: "Glycemic projection is experimental" };
      }
      return { state: "experimental", label: "Experimental", color: "#8B5CF6", badge: "🔬 Experimental", subtext: `Multivariate proxy: ~${Math.round(val)} mg/dL (Not a clinical lab value)` };
    }
  },

  hba1c: {
    name: "HbA1c Glycemic Risk",
    unit: "%",
    normalRange: "Target: < 5.7%",
    validationStatus: "NOT_VALIDATED",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Not Validated", color: "#64748B", badge: "⚪ Not Validated", subtext: "Glycated hemoglobin proxy is experimental" };
      }
      return { state: "experimental", label: "Experimental", color: "#8B5CF6", badge: "🔬 Experimental", subtext: `Proxy estimate: ~${val.toFixed(1)}% (Not a clinical lab value)` };
    }
  },

  cvdRisk: {
    name: "10-Year ASCVD Risk",
    unit: "%",
    normalRange: "< 10% (Low Risk Profile)",
    validationStatus: "CLINICAL_RISK_MODEL",
    evaluate: (val) => {
      if (!val || isNaN(val) || val <= 0) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Requires user-verified clinical profile" };
      }
      if (val >= 20.0) {
        return { state: "critical", label: "High Risk", color: "#EF4444", badge: "🔴 High Risk", subtext: "Framingham 10-year projection ≥ 20%" };
      }
      if (val >= 10.0) {
        return { state: "warning", label: "Moderate Risk", color: "#F59E0B", badge: "🟡 Moderate", subtext: "Framingham 10-year projection 10-20%" };
      }
      return { state: "optimal", label: "Low Risk", color: "#22C55E", badge: "🟢 Low Risk", subtext: "Framingham 10-year projection < 10%" };
    }
  },

  vascularAge: {
    name: "Vascular Heart Age",
    unit: "years",
    normalRange: "Delta ≤ +1.0 years",
    validationStatus: "MODEL_DEPENDENT",
    evaluate: (ageDelta) => {
      if (ageDelta === null || ageDelta === undefined || isNaN(ageDelta)) {
        return { state: "unavailable", label: "Unavailable", color: "#64748B", badge: "⚪ Unavailable", subtext: "Requires hemodynamic baseline" };
      }
      if (ageDelta > 5.0) {
        return { state: "warning", label: "Accelerated", color: "#F59E0B", badge: "🟡 Stiffening", subtext: `Arterial compliance exceeds age by +${ageDelta.toFixed(1)} yrs` };
      }
      return { state: "optimal", label: "Aligned", color: "#22C55E", badge: "🟢 Aligned", subtext: `Arterial compliance aligned (${ageDelta >= 0 ? '+' : ''}${ageDelta.toFixed(1)} yrs)` };
    }
  }
};

if (typeof window !== "undefined") {
  window.CLINICAL_THRESHOLDS = CLINICAL_THRESHOLDS;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = CLINICAL_THRESHOLDS;
}
