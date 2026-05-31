"""
Calibration V3: Grid Search λ + Out-of-Sample Validation
=========================================================
1. Load real GDP 2010-2023 for 6 countries
2. Grid search λ ∈ [0.20, 3.00] to minimize MAPE (2010-2017 training)
3. Out-of-sample MAPE on 2018-2023
4. Compare DCI (ω=0.6) vs TC-only (ω=1.0)
5. ω-sweep for interior optimum

Author: [Your name]
Date: May 2026
"""

using Pkg
Pkg.activate(".")

try using Agents catch; Pkg.add("Agents"); using Agents end

using CSV, DataFrames, Statistics, Random

include("../src/model_v3.jl")
using .ModelV3

mkpath("results_v3")

# ==============================================================================
# LOAD REAL GDP DATA
# ==============================================================================

println("Loading real GDP data...")
real_gdp_df = CSV.read("data/real_gdp_2010_2023.csv", DataFrame)

# Convert to dict: country => [gdp_2010, gdp_2011, ..., gdp_2023]
const COUNTRIES = ["KAZ", "BLR", "AZE", "GEO", "ARM", "NOR"]
const YEARS = 2010:2023
const N_YEARS = length(YEARS)  # 14 data points, 13 steps

real_gdp = Dict{String, Vector{Float64}}()
for c in COUNTRIES
    real_gdp[c] = Float64[real_gdp_df[i, Symbol(c)] for i in 1:nrow(real_gdp_df)]
end

println("Real GDP loaded: $(length(COUNTRIES)) countries, $N_YEARS years")
for c in COUNTRIES
    cagr = (real_gdp[c][end] / real_gdp[c][1])^(1/13) - 1
    println("  $c: $(round(real_gdp[c][1], digits=0)) → $(round(real_gdp[c][end], digits=0)) (CAGR $(round(cagr*100, digits=2))%)")
end

# ==============================================================================
# HELPER: Run model and compute MAPE
# ==============================================================================

"""
    run_and_mape(; λ, ω, seed, year_range)

Run 13-step simulation, compare to real GDP.
Returns per-country MAPE dict for specified year range.
"""
function run_and_mape(;
    λ::Float64,
    ω::Float64,
    seed::Int,
    α_trad::Float64 = 0.01,
    α_digital::Float64 = 0.05,
    σ_eps::Float64 = 0.008,
    σ_xi::Float64 = 0.002
)
    model = init_model_v3(
        "data/countries_init_v3.csv";
        seed = seed,
        λ = λ,
        ω = ω,
        α_trad = α_trad,
        α_digital = α_digital,
        σ_eps = σ_eps,
        σ_xi = σ_xi,
        oil_shock_step = -1,
        digital_shock_step = -1
    )
    
    # Run for 13 steps (2010 → 2023)
    df = run_simulation(model; steps = 13)
    
    # Compute per-country, per-year APE
    apes = Dict{String, Vector{Float64}}()
    
    for c in COUNTRIES
        c_data = filter(row -> row.country == c, df)
        country_apes = Float64[]
        
        for (i, step) in enumerate(0:13)
            model_gdp = c_data[i, :gdp]
            real_val = real_gdp[c][i]
            ape = abs(model_gdp - real_val) / real_val * 100
            push!(country_apes, ape)
        end
        
        apes[c] = country_apes
    end
    
    return apes, df
end

"""
    compute_mape(apes, year_range)

Compute MAPE for given year indices (1-based: 1=2010, 8=2017, 14=2023)
"""
function compute_mape(apes::Dict{String, Vector{Float64}}, idx_range)
    mapes = Dict{String, Float64}()
    all_apes = Float64[]
    
    for c in COUNTRIES
        country_apes = apes[c][idx_range]
        mapes[c] = mean(country_apes)
        append!(all_apes, country_apes)
    end
    
    mapes["POOLED"] = mean(all_apes)
    return mapes
end

# ==============================================================================
# STEP 1: GRID SEARCH λ (training period 2010-2017)
# ==============================================================================

println("\n" * "="^70)
println("STEP 1: Grid Search λ (training: 2010-2017, ω=0.6)")
println("="^70)

# Training indices: years 2010-2017 = indices 1:8
# Test indices: years 2018-2023 = indices 9:14
const TRAIN_IDX = 1:8
const TEST_IDX = 9:14
const N_SEEDS = 20

λ_grid = collect(0.20:0.05:3.00)
best_λ = 0.0
best_mape = Inf
results_λ = DataFrame(λ = Float64[], mape_train = Float64[], mape_test = Float64[])

for λ in λ_grid
    train_mapes = Float64[]
    test_mapes = Float64[]
    
    for s in 1:N_SEEDS
        apes, _ = run_and_mape(λ = λ, ω = 0.6, seed = s)
        
        train_m = compute_mape(apes, TRAIN_IDX)
        test_m = compute_mape(apes, TEST_IDX)
        
        push!(train_mapes, train_m["POOLED"])
        push!(test_mapes, test_m["POOLED"])
    end
    
    mean_train = mean(train_mapes)
    mean_test = mean(test_mapes)
    
    push!(results_λ, (λ, mean_train, mean_test))
    
    if mean_train < best_mape
        global best_mape = mean_train
        global best_λ = λ
    end
end

println("\nBest λ = $(best_λ) (training MAPE = $(round(best_mape, digits=2))%)")

# Show top 10
sort!(results_λ, :mape_train)
println("\nTop 10 λ values:")
println("  λ      MAPE_train  MAPE_test")
for i in 1:min(10, nrow(results_λ))
    r = results_λ[i, :]
    println("  $(round(r.λ, digits=2))    $(round(r.mape_train, digits=2))%     $(round(r.mape_test, digits=2))%")
end

CSV.write("results_v3/calibration_lambda_grid.csv", results_λ)

# ==============================================================================
# STEP 2: DETAILED RESULTS WITH BEST λ
# ==============================================================================

println("\n" * "="^70)
println("STEP 2: Detailed Results with λ=$(best_λ)")
println("="^70)

# Run N_SEEDS replications with best λ, collect per-country MAPE
train_by_country = Dict(c => Float64[] for c in COUNTRIES)
test_by_country = Dict(c => Float64[] for c in COUNTRIES)

for s in 1:N_SEEDS
    apes, _ = run_and_mape(λ = best_λ, ω = 0.6, seed = s)
    
    train_m = compute_mape(apes, TRAIN_IDX)
    test_m = compute_mape(apes, TEST_IDX)
    
    for c in COUNTRIES
        push!(train_by_country[c], train_m[c])
        push!(test_by_country[c], test_m[c])
    end
end

println("\nPer-country MAPE ($(N_SEEDS) seeds, λ=$(best_λ), ω=0.6):")
println("  Country  MAPE_train (mean±std)  MAPE_test (mean±std)")
for c in COUNTRIES
    tr_m = mean(train_by_country[c])
    tr_s = std(train_by_country[c])
    te_m = mean(test_by_country[c])
    te_s = std(test_by_country[c])
    println("  $c       $(round(tr_m, digits=2))% ± $(round(tr_s, digits=2))       $(round(te_m, digits=2))% ± $(round(te_s, digits=2))")
end

# ==============================================================================
# STEP 3: DCI (ω=0.6) vs TC-ONLY (ω=1.0) COMPARISON
# ==============================================================================

println("\n" * "="^70)
println("STEP 3: DCI (ω=0.6) vs TC-only (ω=1.0)")
println("="^70)
println("Using same λ=$(best_λ) for both (fair comparison)")

for (label, ω) in [("DCI (ω=0.6)", 0.6), ("TC-only (ω=1.0)", 1.0)]
    train_pooled = Float64[]
    test_pooled = Float64[]
    
    for s in 1:N_SEEDS
        apes, _ = run_and_mape(λ = best_λ, ω = ω, seed = s)
        
        train_m = compute_mape(apes, TRAIN_IDX)
        test_m = compute_mape(apes, TEST_IDX)
        
        push!(train_pooled, train_m["POOLED"])
        push!(test_pooled, test_m["POOLED"])
    end
    
    println("\n  $label:")
    println("    Train MAPE: $(round(mean(train_pooled), digits=2))% ± $(round(std(train_pooled), digits=2))%")
    println("    Test MAPE:  $(round(mean(test_pooled), digits=2))% ± $(round(std(test_pooled), digits=2))%")
end

# ==============================================================================
# STEP 4: ω-SWEEP (interior optimum search)
# ==============================================================================

println("\n" * "="^70)
println("STEP 4: ω-sweep (λ=$(best_λ), $(N_SEEDS) seeds)")
println("="^70)

ω_grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results_ω = DataFrame(ω = Float64[], mape_train = Float64[], mape_test = Float64[], 
                       mape_train_std = Float64[], mape_test_std = Float64[])

for ω in ω_grid
    train_vals = Float64[]
    test_vals = Float64[]
    
    for s in 1:N_SEEDS
        apes, _ = run_and_mape(λ = best_λ, ω = ω, seed = s)
        push!(train_vals, compute_mape(apes, TRAIN_IDX)["POOLED"])
        push!(test_vals, compute_mape(apes, TEST_IDX)["POOLED"])
    end
    
    push!(results_ω, (ω, mean(train_vals), mean(test_vals), std(train_vals), std(test_vals)))
end

println("\n  ω      MAPE_train (±std)    MAPE_test (±std)")
for r in eachrow(results_ω)
    marker = r.ω == 0.6 ? " ←" : (r.ω == 1.0 ? " ← TC-only" : "")
    println("  $(round(r.ω, digits=1))    $(round(r.mape_train, digits=2))% ± $(round(r.mape_train_std, digits=2))     $(round(r.mape_test, digits=2))% ± $(round(r.mape_test_std, digits=2))$marker")
end

CSV.write("results_v3/calibration_omega_sweep.csv", results_ω)

# ==============================================================================
# STEP 5: SAVE BEST-FIT TRAJECTORY FOR PLOTTING
# ==============================================================================

println("\n" * "="^70)
println("STEP 5: Saving best-fit trajectory")
println("="^70)

# Run one simulation with best λ, seed=42
_, df_best = run_and_mape(λ = best_λ, ω = 0.6, seed = 42)

# Merge with real GDP for comparison
comparison = DataFrame(
    year = Int[], country = String[],
    model_gdp = Float64[], real_gdp = Float64[], ape = Float64[]
)

for c in COUNTRIES
    c_data = filter(row -> row.country == c, df_best)
    for (i, yr) in enumerate(2010:2023)
        mgdp = c_data[i, :gdp]
        rgdp = real_gdp[c][i]
        ape = abs(mgdp - rgdp) / rgdp * 100
        push!(comparison, (yr, c, mgdp, rgdp, ape))
    end
end

CSV.write("results_v3/calibration_best_fit.csv", comparison)
CSV.write("results_v3/calibration_best_trajectory.csv", df_best)

println("Saved: calibration_best_fit.csv")
println("Saved: calibration_best_trajectory.csv")

# Print year-by-year comparison for KAZ
println("\nKAZ year-by-year (λ=$(best_λ), ω=0.6, seed=42):")
println("  Year   Model      Real     APE")
kaz_comp = filter(row -> row.country == "KAZ", comparison)
for r in eachrow(kaz_comp)
    println("  $(r.year)  $(round(r.model_gdp, digits=0))   $(round(r.real_gdp, digits=0))   $(round(r.ape, digits=1))%")
end

# ==============================================================================
# SUMMARY
# ==============================================================================

println("\n" * "="^70)
println("CALIBRATION COMPLETE")
println("="^70)
println("""
Results:
  Best λ = $(best_λ) (calibrated on 2010-2017 training set)
  
  Files saved:
  - results_v3/calibration_lambda_grid.csv (λ grid search results)
  - results_v3/calibration_omega_sweep.csv (ω sweep results)  
  - results_v3/calibration_best_fit.csv (year-by-year comparison)
  - results_v3/calibration_best_trajectory.csv (full model output)
  
  Next steps:
  - Check if DCI (ω=0.6) beats TC-only (ω=1.0) on out-of-sample MAPE
  - If yes → strong claim for digital complexity dimension
  - If no → reframe as descriptive framework
  - Compute BIC/AIC penalized comparison (Step 5 from plan)
""")
