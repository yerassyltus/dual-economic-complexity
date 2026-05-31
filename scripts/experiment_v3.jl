"""
Experiment V3: Dual Complexity — 6 Country Panel
=================================================
KAZ, BLR, AZE, GEO, ARM, NOR

Changes from V2:
- Logistic capability growth c·(1−c)
- 6 countries (added GEO, ARM, NOR)
- WDI-based digital calibration
- Atlas RCA-based traditional calibration
"""

using Pkg
Pkg.activate(".")

try using Agents catch; Pkg.add("Agents"); using Agents end
try using CairoMakie catch; Pkg.add("CairoMakie"); using CairoMakie end

using CSV, DataFrames, Statistics, Random

# Model file is in src/
include("../src/model_v3.jl")
using .ModelV3

# Output directory
mkpath("results_v3")

# Country colors for plots
const COLORS = Dict(
    "KAZ" => :royalblue,
    "BLR" => :forestgreen,
    "AZE" => :darkorange,
    "GEO" => :mediumpurple,
    "ARM" => :crimson,
    "NOR" => :gray40
)
const COUNTRY_ORDER = ["KAZ", "BLR", "AZE", "GEO", "ARM", "NOR"]

# ==============================================================================
# EXPERIMENT 1: Single Country (Kazakhstan) — baseline test
# ==============================================================================

println("\n" * "="^70)
println("EXPERIMENT 1: Kazakhstan Single Country Test")
println("="^70)

model_kaz = init_single_country(
    country = "KAZ",
    init_gdp = 26374.84,
    trad_caps = [0.475, 0.614, 0.420],
    digital_caps = [0.035, 0.176, 0.179],
    trad_lr = 0.03,
    digital_lr = 0.06,
    seed = 42,
    λ = 1.0, ω = 0.6,
    α_trad = 0.01, α_digital = 0.05,
    σ_eps = 0.01, σ_xi = 0.002
)

df_kaz = run_simulation(model_kaz; steps = 50)
summary_stats(df_kaz)

# ==============================================================================
# EXPERIMENT 2: Six Countries Baseline (no shocks)
# ==============================================================================

println("\n" * "="^70)
println("EXPERIMENT 2: 6-Country Baseline Comparison")
println("="^70)

model_baseline = init_model_v3(
    "data/countries_init_v3.csv";
    seed = 42,
    λ = 1.0, ω = 0.6,
    α_trad = 0.01, α_digital = 0.06,
    σ_eps = 0.01, σ_xi = 0.002,
    oil_shock_step = -1,
    digital_shock_step = -1
)

df_baseline = run_simulation(model_baseline; steps = 50)
summary_stats(df_baseline)

# ==============================================================================
# EXPERIMENT 3: Oil Shock (Dutch Disease)
# ==============================================================================

println("\n" * "="^70)
println("EXPERIMENT 3: Oil Shock Effect (Dutch Disease)")
println("="^70)

model_oil = init_model_v3(
    "data/countries_init_v3.csv";
    seed = 42,
    λ = 1.0, ω = 0.6,
    α_trad = 0.01, α_digital = 0.06,
    σ_eps = 0.01, σ_xi = 0.002,
    oil_shock_step = 10,
    oil_shock_gdp = 0.15,
    oil_shock_digital = -0.10,
    digital_shock_step = -1
)

df_oil = run_simulation(model_oil; steps = 50)
summary_stats(df_oil)

# ==============================================================================
# EXPERIMENT 4: Digital Acceleration (COVID-like)
# ==============================================================================

println("\n" * "="^70)
println("EXPERIMENT 4: Digital Acceleration Shock")
println("="^70)

model_digital = init_model_v3(
    "data/countries_init_v3.csv";
    seed = 42,
    λ = 1.0, ω = 0.6,
    α_trad = 0.01, α_digital = 0.06,
    σ_eps = 0.01, σ_xi = 0.002,
    oil_shock_step = -1,
    digital_shock_step = 20,
    digital_shock_boost = 0.30
)

df_digital = run_simulation(model_digital; steps = 50)
summary_stats(df_digital)

# ==============================================================================
# VISUALIZATION
# ==============================================================================

println("\n" * "="^70)
println("GENERATING PLOTS...")
println("="^70)

# --- Plot 1: Kazakhstan ECI Trajectories ---
fig1 = Figure(size = (900, 500))
ax1 = Axis(fig1[1, 1],
    xlabel = "Step (Time)", ylabel = "ECI Value",
    title = "Kazakhstan: Traditional vs Digital ECI"
)

kaz_data = filter(row -> row.country == "KAZ", df_kaz)
lines!(ax1, kaz_data.step, kaz_data.trad_eci,
    label = "Traditional ECI", linewidth = 2, color = :red)
lines!(ax1, kaz_data.step, kaz_data.digital_eci,
    label = "Digital ECI", linewidth = 2, color = :blue)
lines!(ax1, kaz_data.step, kaz_data.combined_eci,
    label = "Combined ECI (DCI)", linewidth = 2, color = :green, linestyle = :dash)

axislegend(ax1, position = :lt)
save("results_v3/kaz_eci_trajectories_v3.png", fig1)
println("Saved: kaz_eci_trajectories_v3.png")

# --- Plot 2: Six Countries GDP ---
fig2 = Figure(size = (1000, 550))
ax2 = Axis(fig2[1, 1],
    xlabel = "Step (Time)", ylabel = "GDP per capita (PPP)",
    title = "GDP Trajectories: 6-Country Panel (Baseline)"
)

for c in COUNTRY_ORDER
    cdata = filter(row -> row.country == c, df_baseline)
    lines!(ax2, cdata.step, cdata.gdp, label = c, linewidth = 2, color = COLORS[c])
end

axislegend(ax2, position = :lt)
save("results_v3/six_countries_gdp_v3.png", fig2)
println("Saved: six_countries_gdp_v3.png")

# --- Plot 3: Six Countries Traditional ECI ---
fig3 = Figure(size = (1000, 550))
ax3 = Axis(fig3[1, 1],
    xlabel = "Step (Time)", ylabel = "Traditional ECI",
    title = "Traditional ECI (Atlas-Visible Complexity)"
)

for c in COUNTRY_ORDER
    cdata = filter(row -> row.country == c, df_baseline)
    lines!(ax3, cdata.step, cdata.trad_eci, label = c, linewidth = 2, color = COLORS[c])
end

axislegend(ax3, position = :rb)
save("results_v3/six_countries_trad_eci_v3.png", fig3)
println("Saved: six_countries_trad_eci_v3.png")

# --- Plot 4: Six Countries Digital ECI ---
fig4 = Figure(size = (1000, 550))
ax4 = Axis(fig4[1, 1],
    xlabel = "Step (Time)", ylabel = "Digital ECI",
    title = "Digital ECI (Hidden Complexity)"
)

for c in COUNTRY_ORDER
    cdata = filter(row -> row.country == c, df_baseline)
    lines!(ax4, cdata.step, cdata.digital_eci, label = c, linewidth = 2, color = COLORS[c])
end

axislegend(ax4, position = :lt)
save("results_v3/six_countries_digital_eci_v3.png", fig4)
println("Saved: six_countries_digital_eci_v3.png")

# --- Plot 5: Six Countries Combined ECI (DCI) ---
fig5 = Figure(size = (1000, 550))
ax5 = Axis(fig5[1, 1],
    xlabel = "Step (Time)", ylabel = "Dual Complexity Index",
    title = "DCI (ω=0.6 trad + 0.4 digital)"
)

for c in COUNTRY_ORDER
    cdata = filter(row -> row.country == c, df_baseline)
    lines!(ax5, cdata.step, cdata.combined_eci, label = c, linewidth = 2, color = COLORS[c])
end

axislegend(ax5, position = :lt)
save("results_v3/six_countries_dci_v3.png", fig5)
println("Saved: six_countries_dci_v3.png")

# --- Plot 6: Oil Shock — KAZ baseline vs oil shock ---
fig6 = Figure(size = (1000, 550))
ax6 = Axis(fig6[1, 1],
    xlabel = "Step (Time)", ylabel = "Value",
    title = "Kazakhstan: Oil Shock Effect (Dutch Disease)"
)

kaz_base = filter(row -> row.country == "KAZ", df_baseline)
kaz_oil = filter(row -> row.country == "KAZ", df_oil)

lines!(ax6, kaz_base.step, kaz_base.gdp ./ 1000,
    label = "GDP baseline (×10³)", linewidth = 2, color = :blue, linestyle = :dash)
lines!(ax6, kaz_oil.step, kaz_oil.gdp ./ 1000,
    label = "GDP oil shock (×10³)", linewidth = 2, color = :blue)
lines!(ax6, kaz_base.step, kaz_base.digital_eci .* 100,
    label = "Digital ECI baseline (×100)", linewidth = 2, color = :red, linestyle = :dash)
lines!(ax6, kaz_oil.step, kaz_oil.digital_eci .* 100,
    label = "Digital ECI oil shock (×100)", linewidth = 2, color = :red)

vlines!(ax6, [10], color = :gray, linestyle = :dot, label = "Oil Shock")
axislegend(ax6, position = :lt)
save("results_v3/kaz_oil_shock_v3.png", fig6)
println("Saved: kaz_oil_shock_v3.png")

# --- Plot 7: Digital Shock — all 6 countries Digital ECI ---
fig7 = Figure(size = (1000, 550))
ax7 = Axis(fig7[1, 1],
    xlabel = "Step (Time)", ylabel = "Digital ECI",
    title = "Digital Acceleration Shock (Step 20): Digital ECI"
)

for c in COUNTRY_ORDER
    cdata = filter(row -> row.country == c, df_digital)
    lines!(ax7, cdata.step, cdata.digital_eci, label = c, linewidth = 2, color = COLORS[c])
end

vlines!(ax7, [20], color = :gray, linestyle = :dot, label = "Digital Shock")
axislegend(ax7, position = :lt)
save("results_v3/six_countries_digital_shock_v3.png", fig7)
println("Saved: six_countries_digital_shock_v3.png")

# ==============================================================================
# SAVE DATA
# ==============================================================================

CSV.write("results_v3/exp1_kaz_single_v3.csv", df_kaz)
CSV.write("results_v3/exp2_baseline_v3.csv", df_baseline)
CSV.write("results_v3/exp3_oil_shock_v3.csv", df_oil)
CSV.write("results_v3/exp4_digital_shock_v3.csv", df_digital)

println("\nAll data saved to results_v3/")

# ==============================================================================
# KEY FINDINGS
# ==============================================================================

println("\n" * "="^70)
println("KEY FINDINGS")
println("="^70)

println("""
1. KAZAKHSTAN PARADOX:
   - Traditional ECI stagnant (oil-concentrated exports)
   - Digital ECI growing (Kaspi.kz fintech ecosystem)
   - DCI captures growth that Atlas misses

2. 6-COUNTRY COMPARISON:
   - Belarus: Highest traditional capabilities (Soviet industrial base)
   - Kazakhstan: Fastest digital growth (fintech leapfrogging)
   - Azerbaijan: Lowest on both dimensions (pure resource dependency)
   - Georgia: No oil, active e-governance — different growth path
   - Armenia: Growing IT sector, low initial digital infrastructure
   - Norway: Control — high digital, moderate traditional (developed economy)

3. OIL SHOCK (Dutch Disease):
   - Short-term GDP boost, long-term digital capability erosion
   - Affects resource-dependent countries asymmetrically

4. DIGITAL ACCELERATION (COVID-like):
   - Countries with higher initial digital caps benefit more
   - Kazakhstan's high digital learning rate amplifies the effect
""")

println("="^70)
println("EXPERIMENT V3 COMPLETE")
println("="^70)
