# Dual Economic Complexity: Digital Capabilities and GDP Prediction

**Repository:** [github.com/yerassyltus/dual-economic-complexity](https://github.com/yerassyltus/dual-economic-complexity)

## Overview

Extension of Hausmann-Hidalgo's Economic Complexity Index (ECI) incorporating digital capabilities. Tested on 18 countries (2010–2023) using ABM + KNN.

### Key Finding

DCI is **conditionally informative**: it improves GDP prediction for countries with active digital transformation, confirmed by placebo test and statistical significance.

| Country | TC-only | TC+Digital | Improvement | p-value |
|---------|---------|------------|-------------|---------|
| **Kazakhstan** | 12.5% | **3.5%** | 3.6× better | 0.031* |
| Turkey | 23.3% | 6.8% | 3.4× better | 0.031* |
| Armenia | 22.6% | 8.5% | 2.7× better | 0.031* |
| Poland | 27.8% | 13.7% | 2.0× better | 0.031* |
| Malaysia | 10.6% | 3.8% | 2.8× better | 0.094 |

*Paired Wilcoxon signed-rank test, \*p < 0.05*

### Robustness

- **Placebo test**: Non-digital time-varying features (urban %, health expenditure, capital formation) give weaker improvement (KAZ: 7.8% vs 3.5%). Effect is digital-specific.
- **COVID robustness**: Results unchanged excluding 2020 (KAZ improvement: +8.9pp with or without).
- **Data-driven discovery**: Blind screening of 1,221 WDI indicators ranks ICT service exports #2 by correlation with TC-only prediction residual (r = +0.635).
- **Pooled negative result**: TC-only outperforms TC+Digital at aggregate level (p = 0.001). Reported transparently.

## Structure

```
src/model_v3.jl            ABM model (Julia/Agents.jl)
data/                      CSV panels (6-country ABM + 18-country KNN)
scripts/
  experiment_v3.jl         ABM experiments
  calibration_v3.jl        ABM calibration (λ grid search)
  knn_analysis.py          Basic KNN comparison
  knn_analysis_v2.py       Full pipeline: KNN + placebo + significance + COVID
results/                   Output CSVs
figures/                   Generated plots
docs/                      Methodology documentation
```

## Reproduce

```bash
# KNN analysis (Python)
pip install scikit-learn numpy matplotlib scipy
python scripts/knn_analysis_v2.py

# ABM (Julia)
julia --project=. scripts/experiment_v3.jl
julia --project=. scripts/calibration_v3.jl
```

## Data Sources

- [World Bank WDI](https://data.worldbank.org): GDP, digital indicators, placebo indicators (2010–2023)
- [Atlas of Economic Complexity](https://atlas.cid.harvard.edu): Product Space RCA (2024)

## DCI Formula

```
DCI(t) = ω · TC + (1 − ω) · DC(t)
```

TC = traditional complexity (Atlas RCA), DC = digital complexity (WDI), ω ∈ [0,1].

## Countries

**ABM (6):** KAZ, BLR, AZE, GEO, ARM, NOR  
**KNN (18):** Above + RUS, UZB, KGZ, MDA, EST, CHL, MNG, MYS, TUR, POL, ROU, THA

## License

MIT

## Acknowledgments

Data: World Bank, Harvard Growth Lab. AI tools (Claude, Anthropic) used for code editing and formatting; all research decisions by the author.
