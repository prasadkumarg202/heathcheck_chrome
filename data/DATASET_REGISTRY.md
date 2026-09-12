# AuraPulse Open Research Dataset License & Benchmark Registry

> **Policy**: Zero commercial infringement. Public research datasets must strictly adhere to their original author licenses and cannot be utilized for direct proprietary commercial bundling unless explicitly allowed by an open-access license (e.g., CC BY 4.0, MIT).

---

## 1. Public Research Datasets Registry

| Dataset | Origin & Authors | License / Terms | Commercial Use | Sensor Modalities & Ground Truth | Demographics & Scope | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PURE** | Stricker et al. (TU Ilmenau, 2014) | Academic Research Use Only | **Prohibited** | RGB (30 FPS, uncompressed PNG) + Contec CMS50E Pulse Oximeter (PPG & SpO2) | 10 subjects (8M/2F), 6 tasks (steady, talking, head rotation) | Small cohort, limited Fitzpatrick range (mostly I–III) |
| **UBFC-rPPG** | Bobbia et al. (Univ. of Burgundy, 2019) | Academic Research Only | **Prohibited** | RGB (30 FPS webcams) + CMS50E PPG & HR | 42 subjects, resting & mathematical stress tasks | Variable indoor ambient lighting, light to medium skin |
| **UBFC-Phys** | Meziati et al. (Univ. of Burgundy, 2021) | Academic Research Only | **Prohibited** | RGB (1024x1024 @ 35 FPS) + Empatica E4 (BVP, EDA, Temp) + Biopac ECG | 56 subjects, Trier Social Stress Test (TSST) | Focused on acute stress; uncalibrated RGB color spaces |
| **VIPL-HR** | Niu et al. (CAS / VIPL, 2018) | Academic Non-Commercial | **Prohibited** | Multi-camera (RGB + NIR), FDA-cleared CONTEC CMS60C | 107 subjects, 9 scenarios (motion, dark lighting, mobile) | High compression artifacts on some webcams |
| **COHFACE** | Heusch et al. (Idiap Research Institute, 2017) | Idiap Non-Commercial License | **Prohibited** | RGB (HD webcam) + Thought Technologies PPG & Respiration | 40 subjects, studio & natural lighting | Highly compressed MJPEG video stream |
| **MMSE-HR** | Tulyakov et al. (Binghamton Univ., 2016) | Academic Non-Commercial | **Prohibited** | High-speed 1040x1392 RGB (25 FPS) + Biopac ECG | 40 subjects, spontaneous emotion induction | Large storage size, non-standard resolutions |

---

## 2. Proprietary Data Collection Architecture & Protocol

To build fully owned, commercially unencumbered models and algorithms, AuraPulse operates its own data collection infrastructure under strict IRB / IEC ethics guidelines.

### Metadata Collected (Non-PII)
- **Synchronized Ground Truth**: Medical-grade 3-lead ECG (sampling rate $\ge 250\text{ Hz}$), clinical finger PPG pulse oximeter ($\ge 100\text{ Hz}$), respiration chest belt.
- **Optical Conditions**: Calibrated illuminance meter (Lux: 50 lx, 250 lx, 750 lx, 2000 lx), color temperature (2700K warm, 4000K neutral, 6500K daylight).
- **Demographics Matrix**: Fitzpatrick Skin Types I through VI (balanced representation target: $\ge 15\%$ per class).
- **Physical Dynamics**: Static baseline, small head movements ($\pm 15^\circ$), reading aloud, physical recovery post-exercise.
