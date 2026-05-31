"""
KNN Analysis: TC-only vs TC+Digital vs Placebo
================================================
Validates the Dual Complexity Index using non-parametric
k-nearest neighbors regression on 18-country panel.

Usage: python knn_analysis.py
Reads: data/full_panel_20countries.csv
Output: results/, figures/
"""

import csv, numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

# Paths
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Load panel
with open(os.path.join(DATA, 'full_panel_20countries.csv'), 'r') as f:
    panel = list(csv.DictReader(f))

for r in panel:
    for k in ['gdp','tc1','tc2','tc3','srv','internet','bb']:
        r[k] = float(r[k])
    r['year'] = int(r['year'])

print(f"Loaded: {len(panel)} observations, {len(set(r['country'] for r in panel))} countries")

# Build arrays
data = np.array([[r['tc1'],r['tc2'],r['tc3'],r['srv'],r['internet'],r['bb'],r['gdp'],r['year']] for r in panel])
country_arr = np.array([r['country'] for r in panel])

X_tc = data[:, :3]
X_all = data[:, :6]
y = data[:, 6]
years = data[:, 7]
train = years <= 2017
test = years >= 2018
k = 5

# === Per-country comparison ===
countries = sorted(set(country_arr))
results = []

for c in countries:
    c_test = (country_arr == c) & test
    if sum(c_test) == 0:
        continue
    row = {'country': c}
    for name, X in [('tc_only', X_tc), ('tc_digital', X_all)]:
        sc = StandardScaler().fit(X[train])
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(sc.transform(X[train]), y[train])
        row[name] = mean_absolute_percentage_error(y[c_test], knn.predict(sc.transform(X[c_test]))) * 100
    row['improvement'] = row['tc_only'] - row['tc_digital']
    results.append(row)

results.sort(key=lambda r: r['improvement'], reverse=True)

print(f"\n{'Country':<6} {'TC-only':>10} {'TC+Digital':>10} {'Δ (pp)':>10}")
for r in results:
    marker = ' ★' if r['improvement'] > 0 else ''
    print(f"{r['country']:<6} {r['tc_only']:>9.1f}% {r['tc_digital']:>9.1f}% {r['improvement']:>+9.1f}{marker}")

# Save CSV
with open(os.path.join(RESULTS, 'knn_results_summary.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['country','mape_tc_only','mape_tc_digital','improvement_pp','winner'])
    for r in results:
        w.writerow([r['country'], f"{r['tc_only']:.2f}", f"{r['tc_digital']:.2f}",
                     f"{r['improvement']:.2f}", 'TC+Digital' if r['improvement'] > 0 else 'TC-only'])

# === Figures ===
# Fig 1: Per-country bars
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(results))
tc_vals = [r['tc_only'] for r in results]
td_vals = [r['tc_digital'] for r in results]
ax.bar(x - 0.175, tc_vals, 0.35, label='TC-only', color='#2196F3', alpha=0.8)
ax.bar(x + 0.175, td_vals, 0.35, label='TC+Digital', color='#4CAF50', alpha=0.8)
for i, r in enumerate(results):
    if r['improvement'] > 0:
        ax.annotate('★', (i, min(r['tc_only'], r['tc_digital']) - 2), ha='center', fontsize=14, color='#4CAF50')
ax.set_xticks(x)
ax.set_xticklabels([r['country'] for r in results], rotation=45, ha='right')
ax.set_ylabel('Test MAPE (%)')
ax.set_title('KNN (k=5): Per-Country Test MAPE\n★ = TC+Digital wins')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'knn_per_country.png'), dpi=150, bbox_inches='tight')

# Fig 2: KAZ trajectory
for name, X in [('TC-only', X_tc), ('TC+Digital', X_all)]:
    sc = StandardScaler().fit(X[train])
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(sc.transform(X[train]), y[train])

fig, ax = plt.subplots(figsize=(12, 6))
kaz = country_arr == 'KAZ'
for name, X, style in [('TC-only', X_tc, 's--'), ('TC+Digital', X_all, '^--')]:
    sc = StandardScaler().fit(X[train])
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(sc.transform(X[train]), y[train])
    pred = knn.predict(sc.transform(X[kaz]))
    color = '#2196F3' if 'only' in name else '#4CAF50'
    ax.plot(years[kaz], pred, style, label=f'KNN {name}', color=color, linewidth=2, markersize=5)
ax.plot(years[kaz], y[kaz], 'ko-', label='Real GDP', linewidth=2.5, markersize=6)
ax.axvline(x=2017.5, color='red', linestyle=':', alpha=0.7, label='Train/Test split')
ax.set_xlabel('Year')
ax.set_ylabel('GDP per capita (PPP)')
ax.set_title('Kazakhstan: KNN Predictions vs Reality')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'knn_kaz_trajectory.png'), dpi=150, bbox_inches='tight')

print(f"\nSaved: results/knn_results_summary.csv")
print(f"Saved: figures/knn_per_country.png")
print(f"Saved: figures/knn_kaz_trajectory.png")
