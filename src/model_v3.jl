"""
Model V3: Dual Economic Complexity (6 countries, logistic dynamics)
==================================================================
Extension of V2:
- Logistic capability growth c·(1−c) instead of linear c
- 6-country panel: KAZ, BLR, AZE, GEO, ARM, NOR
- WDI-based digital capability calibration
- Atlas RCA-based traditional capability calibration

Author: [Your name]
Date: May 2026
"""

module ModelV3

using Agents
using Random
using CSV
using DataFrames
using Statistics

# ==============================================================================
# AGENT DEFINITION
# ==============================================================================

@agent struct CountryAgent(GridAgent{2})
    country::String
    
    # TRADITIONAL capabilities (Atlas-visible: resources, intermediate, advanced)
    trad_caps::Vector{Float64}
    
    # DIGITAL capabilities (hidden: infrastructure, engagement, broadband)
    digital_caps::Vector{Float64}
    
    # Learning rates
    trad_learning_rate::Float64
    digital_learning_rate::Float64
    
    # GDP per capita PPP
    gdp::Float64
    
    # ECI metrics (updated each step)
    trad_eci::Float64
    digital_eci::Float64
    combined_eci::Float64
end

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================

"""
    init_model_v3(path; kwargs...)

Initialize model from CSV with dual capabilities.

CSV columns: id, country, init_gdp, trad_lr, digital_lr,
             trad_cap1..3, digital_cap1..3
"""
function init_model_v3(
    path::String;
    seed::Int = 42,
    # GDP growth parameters
    λ::Float64 = 1.0,
    ω::Float64 = 0.6,
    σ_eps::Float64 = 0.01,
    # Capability growth parameters
    α_trad::Float64 = 0.01,
    α_digital::Float64 = 0.05,
    σ_xi::Float64 = 0.002,
    # Shocks
    oil_shock_step::Int = -1,
    oil_shock_gdp::Float64 = 0.10,
    oil_shock_digital::Float64 = -0.05,
    digital_shock_step::Int = -1,
    digital_shock_boost::Float64 = 0.20
)
    Random.seed!(seed)
    
    df = CSV.read(path, DataFrame)
    
    space = GridSpace((10, 10))
    
    properties = Dict{Symbol, Any}(
        :λ => λ,
        :ω => ω,
        :σ_eps => σ_eps,
        :α_trad => α_trad,
        :α_digital => α_digital,
        :σ_xi => σ_xi,
        :oil_shock_step => oil_shock_step,
        :oil_shock_gdp => oil_shock_gdp,
        :oil_shock_digital => oil_shock_digital,
        :digital_shock_step => digital_shock_step,
        :digital_shock_boost => digital_shock_boost,
        :t => 0
    )
    
    model = StandardABM(CountryAgent, space; properties = properties, rng = Random.MersenneTwister(seed))
    
    for (i, row) in enumerate(eachrow(df))
        pos = (rand(1:10), rand(1:10))
        
        trad_caps = Float64[row.trad_cap1, row.trad_cap2, row.trad_cap3]
        digital_caps = Float64[row.digital_cap1, row.digital_cap2, row.digital_cap3]
        
        trad_eci = mean(trad_caps)
        digital_eci = mean(digital_caps)
        combined_eci = ω * trad_eci + (1 - ω) * digital_eci
        
        agent = CountryAgent(
            i, pos, row.country,
            trad_caps, digital_caps,
            row.trad_lr, row.digital_lr,
            row.init_gdp,
            trad_eci, digital_eci, combined_eci
        )
        
        add_agent!(agent, model)
    end
    
    return model
end

"""
    init_single_country(; kwargs...)

Single-country model for simple experiments.
"""
function init_single_country(;
    country::String = "KAZ",
    init_gdp::Float64 = 26374.84,
    trad_caps::Vector{Float64} = [0.475, 0.614, 0.420],
    digital_caps::Vector{Float64} = [0.035, 0.176, 0.179],
    trad_lr::Float64 = 0.03,
    digital_lr::Float64 = 0.06,
    seed::Int = 42,
    kwargs...
)
    Random.seed!(seed)
    
    space = GridSpace((10, 10))
    
    properties = Dict{Symbol, Any}(
        :λ => get(kwargs, :λ, 1.0),
        :ω => get(kwargs, :ω, 0.6),
        :σ_eps => get(kwargs, :σ_eps, 0.01),
        :α_trad => get(kwargs, :α_trad, 0.01),
        :α_digital => get(kwargs, :α_digital, 0.05),
        :σ_xi => get(kwargs, :σ_xi, 0.002),
        :oil_shock_step => get(kwargs, :oil_shock_step, -1),
        :oil_shock_gdp => get(kwargs, :oil_shock_gdp, 0.10),
        :oil_shock_digital => get(kwargs, :oil_shock_digital, -0.05),
        :digital_shock_step => get(kwargs, :digital_shock_step, -1),
        :digital_shock_boost => get(kwargs, :digital_shock_boost, 0.20),
        :t => 0
    )
    
    model = StandardABM(CountryAgent, space; properties = properties, rng = Random.MersenneTwister(seed))
    
    trad_eci = mean(trad_caps)
    digital_eci = mean(digital_caps)
    ω_val = properties[:ω]
    combined_eci = ω_val * trad_eci + (1 - ω_val) * digital_eci
    
    agent = CountryAgent(
        1, (5, 5), country,
        copy(trad_caps), copy(digital_caps),
        trad_lr, digital_lr,
        init_gdp,
        trad_eci, digital_eci, combined_eci
    )
    
    add_agent!(agent, model)
    
    return model
end

# ==============================================================================
# AGENT STEP: Core dynamics
# ==============================================================================

"""
    agent_step!(agent, model)

Per-agent update each tick:
1. Compute combined complexity
2. GDP growth from complexity
3. Traditional capability growth (logistic)
4. Digital capability growth (logistic)
5. Update ECI metrics
"""
function agent_step!(agent, model)
    λ = model.λ
    ω = model.ω
    α_trad = model.α_trad
    α_digital = model.α_digital
    σ_eps = model.σ_eps
    σ_xi = model.σ_xi
    
    # ─── 1. COMBINED COMPLEXITY ───────────────────────────────────────────
    c_trad = mean(agent.trad_caps)
    c_digital = mean(agent.digital_caps)
    C = ω * c_trad + (1 - ω) * c_digital
    
    # ─── 2. GDP GROWTH ────────────────────────────────────────────────────
    # g = λ · C · learning_rate + ε
    ε = σ_eps * randn()
    g = λ * C * agent.trad_learning_rate + ε
    g = clamp(g, -0.20, 0.30)
    agent.gdp *= (1.0 + g)
    
    # ─── 3. TRADITIONAL CAPABILITIES (logistic growth) ────────────────────
    # c(t+1) = c(t) + α · lr · c(t) · (1 − c(t)) + ξ
    # The (1−c) term ensures capabilities saturate near 1.0
    # and grow fastest at intermediate levels (S-curve)
    for k in eachindex(agent.trad_caps)
        c = agent.trad_caps[k]
        ξ = σ_xi * randn()
        new_cap = c + α_trad * agent.trad_learning_rate * c * (1.0 - c) + ξ
        agent.trad_caps[k] = clamp(new_cap, 0.0, 1.0)
    end
    
    # ─── 4. DIGITAL CAPABILITIES (logistic growth, faster) ────────────────
    # Same logistic form but α_digital >> α_trad
    for k in eachindex(agent.digital_caps)
        c = agent.digital_caps[k]
        ξ = σ_xi * randn()
        new_cap = c + α_digital * agent.digital_learning_rate * c * (1.0 - c) + ξ
        agent.digital_caps[k] = clamp(new_cap, 0.0, 1.0)
    end
    
    # ─── 5. UPDATE ECI METRICS ────────────────────────────────────────────
    agent.trad_eci = mean(agent.trad_caps)
    agent.digital_eci = mean(agent.digital_caps)
    agent.combined_eci = ω * agent.trad_eci + (1 - ω) * agent.digital_eci
end

# ==============================================================================
# MODEL STEP: Shocks
# ==============================================================================

function model_step!(model)
    t = model.t
    
    # OIL SHOCK: GDP boost + Dutch disease (digital erosion)
    if t == model.oil_shock_step
        for ag in allagents(model)
            ag.gdp *= (1.0 + model.oil_shock_gdp)
            for k in eachindex(ag.digital_caps)
                ag.digital_caps[k] *= (1.0 + model.oil_shock_digital)
                ag.digital_caps[k] = clamp(ag.digital_caps[k], 0.0, 1.0)
            end
        end
        println("Step $t: OIL SHOCK (GDP +$(model.oil_shock_gdp*100)%, Digital $(model.oil_shock_digital*100)%)")
    end
    
    # DIGITAL ACCELERATION SHOCK (e.g. COVID-19)
    if t == model.digital_shock_step
        for ag in allagents(model)
            for k in eachindex(ag.digital_caps)
                ag.digital_caps[k] *= (1.0 + model.digital_shock_boost)
                ag.digital_caps[k] = clamp(ag.digital_caps[k], 0.0, 1.0)
            end
        end
        println("Step $t: DIGITAL SHOCK (Digital caps +$(model.digital_shock_boost*100)%)")
    end
    
    model.t = t + 1
end

# ==============================================================================
# SIMULATION RUNNER
# ==============================================================================

function run_simulation(model; steps::Int = 50)
    records = DataFrame(
        step = Int[],
        country = String[],
        gdp = Float64[],
        trad_eci = Float64[],
        digital_eci = Float64[],
        combined_eci = Float64[],
        trad_cap1 = Float64[],
        trad_cap2 = Float64[],
        trad_cap3 = Float64[],
        digital_cap1 = Float64[],
        digital_cap2 = Float64[],
        digital_cap3 = Float64[]
    )
    
    for t in 0:steps
        for ag in allagents(model)
            push!(records, (
                t, ag.country, ag.gdp,
                ag.trad_eci, ag.digital_eci, ag.combined_eci,
                ag.trad_caps[1], ag.trad_caps[2], ag.trad_caps[3],
                ag.digital_caps[1], ag.digital_caps[2], ag.digital_caps[3]
            ))
        end
        
        if t < steps
            for ag in allagents(model)
                agent_step!(ag, model)
            end
            model_step!(model)
        end
    end
    
    return records
end

# ==============================================================================
# METRICS
# ==============================================================================

function compute_cagr(df::DataFrame, country::String; col::Symbol = :gdp)
    country_df = filter(row -> row.country == country, df)
    y0 = first(country_df[!, col])
    yT = last(country_df[!, col])
    T = maximum(country_df.step)
    (y0 <= 0 || yT <= 0 || T == 0) && return NaN
    return (yT / y0)^(1 / T) - 1
end

function compute_eci_change(df::DataFrame, country::String; eci_type::Symbol = :trad_eci)
    country_df = filter(row -> row.country == country, df)
    return last(country_df[!, eci_type]) - first(country_df[!, eci_type])
end

function summary_stats(df::DataFrame)
    countries = unique(df.country)
    
    println("\n" * "="^70)
    println("SIMULATION SUMMARY")
    println("="^70)
    
    for c in countries
        println("\n--- $c ---")
        
        gdp_cagr = compute_cagr(df, c; col = :gdp)
        trad_change = compute_eci_change(df, c; eci_type = :trad_eci)
        digital_change = compute_eci_change(df, c; eci_type = :digital_eci)
        combined_change = compute_eci_change(df, c; eci_type = :combined_eci)
        
        country_df = filter(row -> row.country == c, df)
        gdp_0 = first(country_df.gdp)
        gdp_T = last(country_df.gdp)
        
        println("  GDP:      $(round(gdp_0, digits=1)) → $(round(gdp_T, digits=1)) (CAGR: $(round(gdp_cagr*100, digits=2))%)")
        println("  Trad ECI: Δ = $(round(trad_change, digits=4))")
        println("  Digital ECI: Δ = $(round(digital_change, digits=4))")
        println("  Combined ECI: Δ = $(round(combined_change, digits=4))")
    end
    
    println("\n" * "="^70)
end

# ==============================================================================
# EXPORTS
# ==============================================================================

export CountryAgent
export init_model_v3, init_single_country
export agent_step!, model_step!
export run_simulation
export compute_cagr, compute_eci_change, summary_stats

end # module ModelV3
