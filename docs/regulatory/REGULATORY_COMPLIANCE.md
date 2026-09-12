# AuraPulse Regulatory Design & Software as a Medical Device (SaMD) Strategy

## 1. Dual-Track Regulatory Positioning

AuraPulse is engineered with a strict two-tier architecture:

### Tier 1: General Wellness & Fitness (Current Release)
- **Classification**: General Wellness Device (FDA Guidance 2019 / EU MDR Recital 19).
- **Intended Use**: Real-time fitness tracking, stress management, autonomic health awareness.
- **Labeling Requirements**: All user interfaces and SDK outputs explicitly include non-diagnostic disclaimers:
  > *"This product is intended for general wellness and fitness purposes and is not a medical device. It should not be used to diagnose, treat, cure, or prevent any medical condition."*

### Tier 2: Software as a Medical Device (SaMD) / Class II Roadmap
- **Target Clearances**: FDA 510(k) (USA), CE Mark MDR Class IIa (EU), CDSCO (India).
- **QMS Standards**: ISO 13485, IEC 62304 (Medical Device Software Lifecycle), ISO 14971 (Risk Management).
- **Demographic Equity**: FDA 2024 Guidance on Pulse Oximeter and Optical Sensor Demographic Performance across Fitzpatrick scale I–VI.

---

## 2. Privacy & Biometric Protection

1. **Local Frame Destruction**: All RGB video buffers are ephemeral in volatile memory and destroyed after spatial color averaging.
2. **GDPR / HIPAA / DPDP Act (India 2023)**: Zero personally identifiable biometric templates or face embeddings stored or transmitted without explicit signed consent.
