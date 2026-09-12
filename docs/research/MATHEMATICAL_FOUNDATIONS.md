# Mathematical Foundations of AuraPulse rPPG Algorithms

## 1. Skin Optical Model & Dichromatic Reflection

Light reflected from human facial skin follows the Dichromatic Reflection Model:

$$\mathbf{C}(t) = I(t) \cdot (\mathbf{v}_s(t) + \mathbf{v}_d(t)) + \mathbf{n}(t)$$

Where:
- $I(t)$: Ambient light intensity.
- $\mathbf{v}_s(t)$: Specular reflection from the skin surface (contains no pulsatile information).
- $\mathbf{v}_d(t) = \mathbf{u}_d \cdot d_0 + \mathbf{u}_p \cdot p(t)$: Diffuse reflection through dermis and capillary beds.
- $\mathbf{u}_p$: Hemoglobin absorption vector.
- $p(t)$: Pulsatile Blood Volume Pulse (BVP).

---

## 2. Plane-Orthogonal-to-Skin (POS) Formulation

Reference: Wang et al., IEEE Transactions on Biomedical Engineering (2017).

1. **Temporal Normalization**:
   $$c_n(t) = \frac{\mathbf{C}(t)}{\mu_{\mathbf{C}}}$$

2. **Orthogonal Projection Planes**:
   $$S_1(t) = G_n(t) - B_n(t)$$
   $$S_2(t) = -2 R_n(t) + G_n(t) + B_n(t)$$

3. **Adaptive Standard Deviation Ratio**:
   $$\alpha = \frac{\sigma(S_1)}{\sigma(S_2)}$$

4. **Extracted Blood Volume Pulse**:
   $$h(t) = S_1(t) + \alpha \cdot S_2(t)$$

---

## 3. Chrominance-Based (CHROM) Formulation

Reference: de Haan & Jeanne, IEEE TBME (2013).

1. **Color Difference Signals**:
   $$X_s(t) = 3 R_n(t) - 2 G_n(t)$$
   $$Y_s(t) = 1.5 R_n(t) + G_n(t) - 1.5 B_n(t)$$

2. **Adaptive Weight**:
   $$\alpha = \frac{\sigma(X_s)}{\sigma(Y_s)}$$

3. **Output**:
   $$\text{BVP}(t) = X_s(t) - \alpha \cdot Y_s(t)$$

---

## 4. Signal Quality Index (SQI) Metric Formulation

$$\text{SQI} = w_1 \cdot \text{SNR}_{\text{norm}} + w_2 \cdot \text{Periodicity}_{\text{ac}} + w_3 \cdot \text{Prominence}_{\text{peak}} + w_4 \cdot \text{Coherence}_{\text{ROI}}$$

Where:
$$\text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{\int_{f_{\text{peak}}-\delta}^{f_{\text{peak}}+\delta} P(f)df + \int_{2f_{\text{peak}}-\delta}^{2f_{\text{peak}}+\delta} P(f)df}{\int_{0.5}^{4.0} P(f)df - \text{Harmonics}} \right)$$
