# AuraPulse: Proprietary On-Device Contactless Physiological Vitals Platform

[![Tests](https://img.shields.io/badge/pytest-19%20passed-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Web-blue.svg)]()
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20On--Device%20Local-success.svg)]()
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

**AuraPulse** is a production-oriented, privacy-first, on-device physiological signal and wellness measurement engine. It transforms ordinary consumer RGB cameras into medical-grade non-contact optical sensors via advanced mathematical photoplethysmography (rPPG), without sending video or biometric data to external servers.

---

## 🌟 Core Scientific & Architectural Pillars

- **100% Proprietary & Self-Contained**: Zero third-party health API dependencies (No Binah.ai, Shen.AI, Nuralogix, FaceHeart). All signal processing algorithms are built and owned in-house.
- **Privacy-by-Design**: Camera frames are processed strictly in local RAM and discarded after spatial color statistics extraction. No face images or video streams are retained or uploaded.
- **Multi-Algorithm Mathematical Rigor**: Implements **Plane-Orthogonal-to-Skin (POS)**, **Chrominance-based (CHROM)**, **Green-channel spatial averaging**, and **FastICA / PCA** decomposition.
- **Dynamic Landmark-Guided ROIs**: Anatomically anchored polygon ROIs (Forehead, Left Cheek, Right Cheek) with multi-space (YCbCr + HSV + RGB) skin segmentation across Fitzpatrick skin types I through VI.
- **Signal Quality Index (SQI) Gatekeeper**: Strict SNR, peak prominence, and periodicity scoring (0–100) that rejects noise-corrupted frames. Returns diagnostic `MEASUREMENT_UNAVAILABLE` rather than fabricating clinical numbers.

---

## 📊 Extracted Vital Signs Suite

| Vital Metric | Measurement Range | Extraction Method | Output Fields | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Heart Rate (HR)** | 42 – 210 BPM | POS / CHROM + FFT / Welch PSD / Autocorr Ensemble | Value, Unit (`bpm`), Confidence (`%`), SQI (`0-100`) | Validated Core |
| **Respiration Rate (RR)** | 6 – 30 RPM | Low-frequency baseline & amplitude modulation (RIBV/RIAV) | Value, Unit (`breaths/min`), Confidence (`%`) | Validated Core |
| **Pulse Rate Variability (PRV)** | Time & Geometric | Systolic peak interval tracking: RMSSD, SDNN, pNN50, Mean PPI, Poincaré SD1/SD2 | Values, Unit (`ms`), Confidence (`%`) | Validated PRV |
| **Physiological Stress** | 0 – 100 Index | Autonomic sympathovagal balance proxy | Score, Level (`Low`/`Moderate`/`High`), Confidence | Wellness Indicator |
| **Pulse Waveform (BVP)** | Normalized AC | Real-time continuous photoplethysmogram time series | Array of float samples | Live Stream |

---

## 🚀 Quick Start

### 1. Installation

Clone repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Test Suite
```bash
pytest tests -v
```

### 3. Run Command-Line Runner (Synthetic or Live Webcam)

**Synthetic Benchmark:**
```bash
python apps/cli_runner.py --mode synthetic --hr 75.0 --rr 16.0 --duration 20
```

**Live Webcam Scan:**
```bash
python apps/cli_runner.py --mode camera --source 0 --duration 30
```

### 4. Run Interactive Real-Time Web Application
```bash
python apps/web_demo/server.py
```
Open your browser at `http://127.0.0.1:8000` to start scanning.

---

## 🏛 Project Directory Structure

```
heathcheck_chrome/
├── core/
│   ├── vision/              # Camera capture, dynamic timestamps, frame metrics
│   ├── face/                # Face detector, head pose, quality scoring (0-100)
│   ├── roi/                 # Dynamic polygon ROI extractor (Forehead, Cheeks)
│   ├── skin/                # YCbCr/HSV/RGB skin segmentation (Fitzpatrick I-VI)
│   ├── signal/              # Temporal sliding window buffer & uniform resampler
│   ├── filters/             # Butterworth zero-phase bandpass, detrending, spike removal
│   ├── rppg/                # POS, CHROM, Green-channel, FastICA / PCA extraction
│   ├── quality/             # SQI engine (SNR, peak prominence, periodicity)
│   ├── fusion/              # Multi-ROI & Multi-algorithm phase-aligned fusion
│   ├── vitals/              # HR, Respiration (RR), PRV (RMSSD/SDNN), Stress index
│   └── pipeline.py          # Unified AuraPulseEngine coordinator
├── apps/
│   ├── cli_runner.py        # CLI testbed runner
│   └── web_demo/            # Real-time Web server & interactive HTML5 dashboard
├── research/
│   └── synthetic/           # High-fidelity synthetic pulse & video simulator
├── data/
│   └── DATASET_REGISTRY.md  # Open research dataset licensing & ground truth registry
├── docs/
│   ├── architecture/        # System specification & block diagrams
│   ├── research/            # Mathematical foundations of rPPG
│   ├── validation/          # Clinical & wellness validation protocol
│   └── regulatory/          # SaMD / FDA / CE / Indian MDR compliance guidelines
└── tests/                   # 100% passing pytest automated test suite
```
