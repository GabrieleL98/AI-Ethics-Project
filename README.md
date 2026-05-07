# FairnessAnalyzer
**Unified Fairness Measurement Module**
*Developer Reference & Usage Guide*

---

## Overview
**FairnessAnalyzer** è una classe Python progettata per fungere da punto di ingresso unico per l'audit della fairness nei modelli di machine learning. Integra le funzionalità di tre librerie leader del settore — **Fairlearn**, **AIF360**, e **Aequitas** — sotto un'unica interfaccia coerente.

Ogni metodo pubblico restituisce una dataclass `FairnessResult` (o un dizionario di esse) contenente:
*   **`value`**: Stima puntuale della metrica.
*   **`confidence_interval`**: Intervallo di confidenza al 95% (calcolato tramite bootstrap).
*   **`effect_size`**: Risk Ratio o rapporto worst/best.
*   **`sample_sizes`**: Conteggio dei campioni per ogni gruppo sensibile.

---

## Installazione

### Requisiti
È richiesto Python 3.9 o superiore.
```bash
pip install fairlearn aif360 aequitas
pip install scikit-learn pandas numpy statsmodels
# Opzionali
pip install mlflow matplotlib