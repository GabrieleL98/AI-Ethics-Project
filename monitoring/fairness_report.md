# Loan Approval — Fairness Monitor
*Generated: 2026-05-22 10:15 UTC*

---

## Executive Summary

| | |
|---|---|
| **Batches processed** | 50 |
| **Active alerts** | 3 |
| **Critical alerts** | 3 |
| **High alerts** | 0 |

### ⛔ Status: CRITICAL — Immediate action required.

## Current Fairness Metrics

- **Demographic Parity**: `0.0073`
- **Equalized Odds**: `0.0135`
- **Predictive Parity**: `0.0044`

## Group Positive Rates (latest window)

| Group | Positive Rate | Group Size |
|---|---|---|
| False | 0.6961 | 1649 |
| True | 0.7020 | 1678 |

## Active Alerts

| Time | Metric | Severity | KS Stat | p-value | Message |
|---|---|---|---|---|---|
| 2024-06-02 09:00:00 | demographic_parity | **CRITICAL** | 0.7500 | 0.0003 | [CRITICAL] Drift detected in 'demographic_parity' (KS=0.750, p=0.0003, threshold=0.05). Wavelet: N/A. |
| 2024-06-02 09:00:00 | predictive_parity | **CRITICAL** | 0.8571 | 0.0000 | [CRITICAL] Drift detected in 'predictive_parity' (KS=0.857, p=0.0000, threshold=0.07). Wavelet: N/A. |
| 2024-06-02 09:00:00 | equalized_odds | **CRITICAL** | 0.3214 | 0.4050 | [CRITICAL] Drift detected in 'equalized_odds' (KS=0.321, p=0.4050, threshold=0.07). Wavelet: N/A. |

## Recommendations

- 🔴 **Investigate immediately**: critical drift in fairness metrics detected.
- Consider rolling back the model or applying a post-processing correction.

---
*Report generated automatically by FairnessReportingDashboard.*