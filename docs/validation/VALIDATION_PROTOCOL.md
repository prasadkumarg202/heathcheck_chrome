# AuraPulse Physiological Validation Protocol & Benchmarking Standards

## 1. Statistical Validation Framework

All vital sign extraction algorithms are evaluated against synchronized gold-standard references (FDA-cleared 3-lead ECG, CMS50E finger PPG, calibrated chest belt).

### Metrics:
1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{M} \sum_{i=1}^{M} | \hat{y}_i - y_i |$$

2. **Root Mean Square Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2}$$

3. **Pearson Correlation Coefficient ($r$)**:
   $$r = \frac{\sum (\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}{\sqrt{\sum (\hat{y}_i - \bar{\hat{y}})^2 \sum (y_i - \bar{y})^2}}$$

4. **Bland-Altman 95% Limits of Agreement (LoA)**:
   $$\text{Bias} = \bar{d}, \quad \text{LoA} = \bar{d} \pm 1.96 \cdot s_d$$

---

## 2. Demographic Subgroup Stratification

Validation MUST be reported across stratified subgroups, not solely aggregated averages:
- **Skin Pigmentation**: Fitzpatrick Skin Types I, II, III, IV, V, VI.
- **Illumination Levels**: Dim (50 lx), Office (300 lx), High (1000+ lx).
- **Physical Dynamics**: Still resting pose vs. head movement ($\pm 15^\circ$ yaw/pitch).
