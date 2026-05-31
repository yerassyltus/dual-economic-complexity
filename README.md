# Dual Economic Complexity: Digital Capabilities and GDP Prediction

## Overview

This repository contains code and data for the paper:

> **"Dual Economic Complexity: Digital Capabilities and GDP Prediction in Resource-Dependent Economies"**

We propose the **Dual Complexity Index (DCI)** extending Hausmann-Hidalgo's ECI by incorporating digital capabilities. Tested on 18 countries (2010–2023) using ABM and KNN.

### Key Finding

DCI is **conditionally informative**: it improves GDP prediction for countries with active digital transformation but not universally.

| Country | TC-only MAPE | TC+Digital MAPE | Improvement |
|---------|-------------|-----------------|-------------|
| **Kazakhstan** | 12.5% | **3.5%** | 3.6× better |
| Malaysia | 10.6% | 3.8% | 2.8× better |
| Turkey | 23.3% | 6.8% | 3.4× better |
| Armenia | 22.6% | 8.5% | 2.7× better |

Placebo test confirms digital-specificity: non-digital time-varying features (urban %, health expenditure, capital formation) give weaker improvement (KAZ: 7.8% vs 3.5%).

## Structure

```
src/            Julia ABM model (Agents.jl)
data/           CSV panels (6-country ABM + 18-country KNN)
scripts/        Experiment, calibration, KNN analysis
results/        Output CSVs
figures/        Generated plots
docs/           Methodology documentation
```

## Reproduce

**KNN (Python):**
```bash
pip install scikit-learn numpy matplotlib
python scripts/knn_analysis.py
```

**ABM (Julia):**
```bash
julia --project=. scripts/experiment_v3.jl
julia --project=. scripts/calibration_v3.jl
```

## Data Sources

- [World Bank WDI](https://data.worldbank.org): GDP, Secure servers, Internet users, Broadband (2010–2023)
- [Atlas of Economic Complexity](https://atlas.cid.harvard.edu): Product Space RCA (2024)

## DCI Formula

```
DCI(t) = ω · TC + (1 − ω) · DC(t)
```

TC = traditional complexity (Atlas RCA), DC = digital complexity (WDI indicators), ω ∈ [0,1].

## Countries

**ABM (6):** KAZ, BLR, AZE, GEO, ARM, NOR

**KNN (18):** Above + RUS, UZB, KGZ, MDA, EST, CHL, MNG, MYS, TUR, POL, ROU, THA

## License

MIT

## Acknowledgments

Data: World Bank, Harvard Growth Lab. AI tools (Claude, Anthropic) used for code editing and formatting; all research decisions by the author.
