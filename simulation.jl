# -----------------------------------------------------------------------------
# Author: Johannes Konrad
# Affiliation: Institute for Digital Communications, Friedrich-Alexander-Universität Erlangen-Nürnberg]
# Email: johannes.konrad@fau.de
#
# This code is associated with the manuscript:
# "On the Role of Relay Cells in Information Propagation Speed for Dictyostelium Chemotaxis: An Agent-Based Modeling Study" (in preparation)
#
# The code implements the methods and algorithms described in the manuscript.
# Please refer to the paper for a detailed explanation of the methodology, 
# experimental design, and results.
# -----------------------------------------------------------------------------
# =============================================================================
# Imports
# =============================================================================
using Agents, CairoMakie, Random
using Catalyst, JumpProcesses
using Distributions
import ProgressMeter
using Printf
using StaticArrays
using CSV
using DataFrames

println("Threads available: ", Threads.nthreads())

# =============================================================================
# CLI-Argumente
# =============================================================================
const outputAdd = length(ARGS) > 0 ? ARGS[1] : "new"
const N_AGENTS  = parse(Int, ARGS[2])

# =============================================================================
# Simulationskonstanten
# =============================================================================

# --- Grid ---
const WIDTH         = 500
const HEIGHT        = 200
const SPECIES_COUNT = 2
const GRID_SIZE     = (WIDTH, HEIGHT, SPECIES_COUNT)
const dx            = 3.0   # µm
const DX            = 3.0   # µm pro Simulations-Unit (Alias, für Lesbarkeit)

# --- Spezies-Indizes (Layer im Grid) ---
const cAMP_idx    = 1
const PDEe_idx    = 2
const SPECIES_MAP = Dict{Symbol, Int}(:cAMPe => cAMP_idx, :PDEe => PDEe_idx)

# --- Zeit ---
const STEPS    = 1_00_000
const dt       = 0.002
const STEPSIZE = 10

# --- Physikalische Konstanten ---
const NA = 6.023e23
const V  = 3.672e-14
const mu = 1e-6

# --- Diffusion & extrazellulärer Zerfall ---
const DIFFUSION_RATES = [350.0, 40.0]
const DECAY_RATE      = 1e5
const DECAY_PDE       = 0.01 / 60

# --- Reaktionsparameter (PDE-Induktion via CAR1) ---
const k15_ind = 1.08
const K       = 4000.0

# --- Agenten-Platzierung ---
const DESIRED_DISTANCE_UM = 40.0
const MIN_BORDER_DIST     = 50

# --- Ausgabe ---
const createMovie = false

# --- Nominale Anfangswerte (werden bei Agent-Erzeugung perturbiert) ---
const NOMINAL_U0 = Dict(
    :ACA         => 0.0,
    :PKA         => 0.0,
    :ERK2        => 0.0,
    :RegA        => 0.0,
    :cAMPi       => 0.0,
    :cAMPe       => 700.0,
    :CAR1        => 0.0,
    :PDEe        => 0.0,
    :cAMPtracked => 0.0,
)

# =============================================================================
# Reaktionsnetzwerk (Modell einer einzelnen Zelle)
# =============================================================================
const rn = @reaction_network begin
    k1, CAR1 --> ACA + CAR1
    k2, ACA + PKA --> PKA
    k3, cAMPi --> PKA + cAMPi
    k4, PKA --> 0
    k5, CAR1 --> ERK2 + CAR1
    k6, PKA + ERK2 --> PKA
    k7, 0 --> RegA
    k8, ERK2 + RegA --> ERK2
    k9, ACA --> cAMPi + ACA
    k10, RegA + cAMPi --> RegA
    k11, ACA --> cAMPe + ACA + cAMPtracked
    # k12, cAMPe --> 0     # nicht relevant
    k13, cAMPe --> CAR1 + cAMPe
    k14, CAR1 --> 0
    # --- PDEe ---
    k15_basal, 0 --> PDEe
    (k15_ind * (K^2 * CAR1^2) / (K^2 + CAR1^2)^2), 0 --> PDEe
    # --- Umgebung ---
    decay_rate, cAMPe + PDEe --> PDEe
    decay_pde, PDEe --> 0
end

# =============================================================================
# Agent-Definition
# =============================================================================
@agent struct CellAgent(GridAgent{2})
    name::String
    network::ReactionSystem
    integrator::JumpProcesses.SSAIntegrator
    camp_u_idx::Int
    camptrack_u_idx::Int
    pde_u_idx::Int
    tracker::Array{Int}
end

# =============================================================================
# Agent-Step (Gillespie-Integration pro Zelle)
# =============================================================================

# Leerer agent_step! - die eigentliche Logik läuft in model_step! (step_agent_dummy!),
# damit Reihenfolge und Thread-Kontrolle explizit bleiben.
@inline function step_agent!(agent, model)
end

@inbounds function step_agent_dummy!(agent::CellAgent, model)
    h, w = agent.pos
    int  = agent.integrator

    # I. Synchronisation: Grid -> Agent
    int.u[agent.camp_u_idx]      = model.data[h, w, SPECIES_MAP[:cAMPe]]
    int.u[agent.pde_u_idx]       = model.data[h, w, SPECIES_MAP[:PDEe]]
    int.u[agent.camptrack_u_idx] = 0

    # II. Raten neu aggregieren (u wurde manuell verändert)
    reset_aggregated_jumps!(int)

    # III. SSA-Schritt um dt vorwärts
    step!(int, dt, true)

    # IV. Tracker + Grid zurückschreiben
    agent.tracker[abmtime(model) + 1]        = int.u[agent.camptrack_u_idx]
    model.data[h, w, SPECIES_MAP[:cAMPe]]    = int.u[agent.camp_u_idx]
    model.data[h, w, SPECIES_MAP[:PDEe]]     = int.u[agent.pde_u_idx]
end

# =============================================================================
# Diffusion (Multinomial-Ziehung via sequenzielle Binomials)
# =============================================================================
@inline function draw_fluxes!(buf, n::Int, p::Float64)
    # Kategorien: N, S, O, W, Bleiben
    r  = n
    x1 = rand(Binomial(r, p));          r -= x1
    q2 = p / (1 - p)
    x2 = rand(Binomial(r, q2));         r -= x2
    q3 = p / (1 - 2p)
    x3 = rand(Binomial(r, q3));         r -= x3
    q4 = p / (1 - 3p)
    x4 = rand(Binomial(r, q4));         r -= x4

    buf[1] = x1
    buf[2] = x2
    buf[3] = x3
    buf[4] = x4
    buf[5] = r
    return nothing
end

function apply_diffusion!(data::Array{Int, 3}, rates, dt)
    height, width, species_count = size(data)
    delta = zeros(Int, height, width, species_count)

    for s in 1:species_count
        p_jump = rates[s] * dt / (dx^2)

        if p_jump > 0.24
            @warn "Stabilitätsrisiko: p_jump ($p_jump) zu hoch!"
        end

        fluxes = MVector{5, Float64}(undef)
        @inbounds for h in 1:height
            for w in 1:width
                n_total = data[h, w, s]
                n_total <= 0 && continue

                draw_fluxes!(fluxes, n_total, p_jump)

                # Abzug aus Voxel (nur die bewegten Teilchen)
                n_moving = n_total - fluxes[5]
                delta[h, w, s] -= n_moving

                # Verteilung auf Nachbarn (absorbierende Ränder)
                h < height && (delta[h + 1, w, s] += fluxes[1])   # Nord
                h > 1      && (delta[h - 1, w, s] += fluxes[2])   # Süd
                w < width  && (delta[h, w + 1, s] += fluxes[3])   # Ost
                w > 1      && (delta[h, w - 1, s] += fluxes[4])   # West
            end
        end
    end

    data .+= delta
end

# =============================================================================
# Extrazellulärer Zerfall (cAMPe + PDEe)
# =============================================================================
@inline function decay_TAU(n_src, n_kat, c_cAMP, c_PDE, dt::Float64)
    λ_camp = (c_cAMP * n_src * n_kat) * dt
    λ_pde  = (c_PDE  * n_kat)         * dt

    d_camp = (λ_camp > 0) ? rand(Poisson(λ_camp)) : 0
    d_pde  = (λ_pde  > 0) ? rand(Poisson(λ_pde))  : 0

    d_camp > n_src && (d_camp = n_src)
    d_pde  > n_kat && (d_pde  = n_kat)

    return (n_src - d_camp, n_kat - d_pde)
end

@inline @inbounds function decay_SSA(n_src, n_kat, c_cAMP, c_PDE, endtime)
    t = 0.0
    while (n_src > 0 || n_kat > 0)
        prob_PDEe = n_kat * c_PDE
        prob_cAMP = (n_src * n_kat) * c_cAMP
        total     = prob_cAMP + prob_PDEe

        total <= 0.0 && break

        t += randexp() / total
        t > endtime && break

        if rand() * total < prob_cAMP
            n_src -= 1
        else
            n_kat -= 1
        end
    end
    return (n_src, n_kat)
end

@inline function apply_decay!(data, agent_mask, src_idx::Int, kat_idx::Int, endtime::Float64)
    height = size(data, 1)
    width  = size(data, 2)

    c_cAMP = DECAY_RATE / (NA * dx^3)   # * n_src * n_kat
    c_PDE  = DECAY_PDE                  # * n_kat

    for w in 1:width
        @inbounds for h in 1:height
            agent_mask[w, h] && continue

            n_src = data[h, w, src_idx]
            n_kat = data[h, w, kat_idx]
            (n_src <= 0 || n_kat <= 0) && continue

            if n_src < 50 && n_kat < 50
                n_src, n_kat = decay_SSA(n_src, n_kat, c_cAMP, c_PDE, endtime)
            else
                n_src, n_kat = decay_TAU(n_src, n_kat, c_cAMP, c_PDE, endtime)
            end

            data[h, w, src_idx] = n_src
            data[h, w, kat_idx] = n_kat
        end
    end
end

# =============================================================================
# Agent-Mask (Voxel mit Zelle -> keine Umgebungsreaktion)
# =============================================================================
@inline function build_agent_mask!(model)
    for agent in allagents(model)
        h, w = agent.pos
        @inbounds model.agent_mask[h, w] = true
    end
end

# =============================================================================
# Agent-Erzeugung
# =============================================================================
function generate_initial_u0(pd, species_list)
    u0_dict = Dict{Symbol, Int}()

    for (spec, nominal) in NOMINAL_U0
        if spec in species_list
            d_i             = rand() * 2 - 1
            perturbed_value = nominal * (1 + pd * d_i)
            push!(u0_dict, spec => Int(round(perturbed_value)))
        else
            push!(u0_dict, spec => 0)
        end
    end

    return u0_dict
end

function addCellAgent(model::AgentBasedModel, pos; has_cAMPe=true, name="")
    TIME_SCALE = 60

    reaction_params = Dict(
        # --- Unimolekular: / TIME_SCALE ---
        :k1  => 1.08,
        :k3  => 2.5  / TIME_SCALE,
        :k4  => 1.5  / TIME_SCALE,
        :k5  => 0.6  / TIME_SCALE,
        :k9  => 0.3  / TIME_SCALE,
        :k11 => 1.0,
        :k13 => 0.71,
        :k14 => 4.5  / TIME_SCALE,

        # --- Bimolekular: / NA / V / mu / TIME_SCALE ---
        :k2  => 0.9  / NA / V / mu / TIME_SCALE,
        :k6  => 0.8  / NA / V / mu / TIME_SCALE,
        :k8  => 1.3  / NA / V / mu / TIME_SCALE,
        :k10 => 0.8  / NA / V / mu / TIME_SCALE,

        # --- Nullte Ordnung: * NA * V * mu / TIME_SCALE ---
        :k7  => 1.0  * NA * V * mu / TIME_SCALE,

        # --- PDEe-Produktion ---
        :k15_basal => 3e-4    * NA * V * mu,
        :k15_ind   => k15_ind * NA * V * mu / TIME_SCALE,
        :K         => K,

        # --- Extrazellulär (Voxel-Volumen) ---
        :decay_rate => DECAY_RATE / (NA * dx^3 * 1e-18),
        :decay_pde  => DECAY_PDE,
    )

    SPECIES_LIST = [:PKA, :ERK2, :RegA, :cAMPi, :PDEe]
    if has_cAMPe
        push!(SPECIES_LIST, :cAMPe, :CAR1, :ACA)
    end

    initial_u0 = generate_initial_u0(0.05, SPECIES_LIST)
    model.data[pos..., SPECIES_MAP[:cAMPe]] = initial_u0[:cAMPe]

    rn_species_syms = ModelingToolkit.getname.(species(rn))
    camp_rn_idx     = findfirst(==(:cAMPe),       rn_species_syms)
    pde_rn_idx      = findfirst(==(:PDEe),        rn_species_syms)
    camptrack_u_idx = findfirst(==(:cAMPtracked), rn_species_syms)

    prob             = DiscreteProblem(rn, initial_u0, (0.0, dt), reaction_params)
    jp               = JumpProblem(rn, prob, NRM(); save_positions=(false, false))
    agent_integrator = init(jp, SSAStepper())

    add_agent!(
        pos, CellAgent, model;
        network         = rn,
        integrator      = agent_integrator,
        camp_u_idx      = camp_rn_idx,
        pde_u_idx       = pde_rn_idx,
        camptrack_u_idx = camptrack_u_idx,
        tracker         = zeros(Int, STEPS),
        name            = name,
    )
end

function add_agents_unique_fast!(model)
    N_AGENTS >= 1 || throw(ArgumentError("N_AGENTS muss mindestens 1 sein."))
    WIDTH    >= 1 || throw(ArgumentError("WIDTH muss mindestens 1 sein."))
    HEIGHT   >= 1 || throw(ArgumentError("HEIGHT muss mindestens 1 sein."))

    h        = cld(HEIGHT, 2)
    
    x_center = WIDTH / 2
    x_start  = x_center - DESIRED_DISTANCE_UM / 2
    span = DESIRED_DISTANCE_UM / (N_AGENTS - 1)

    for i in 1:N_AGENTS
        t = x_start + span * (i - 1)
        x = clamp(round(Int, t), 1, WIDTH)

        pos  = (x, h)
        name = i == 1        ? "TX" :
               i == N_AGENTS ? "RX" : "R$(i - 1)"

        addCellAgent(model, pos; has_cAMPe = (i == 1), name = name)
    end
end

# =============================================================================
# PDE-Initialisierung (optional aus CSV)
# =============================================================================
function init_pde_from_csv!(data, pde_idx, width, height, n_agents, dist)
    filepath = joinpath(
        "output_pde",
        "PDE_WIDTH_$(width)_HEIGHT_$(height)_N_AGENTS_$(n_agents)_DIST_$(dist).csv",
    )

    if !isfile(filepath)
        return zeros(Int, width, height)
    end

    println("Lade PDE Initialisierung aus: $filepath")
    df = CSV.read(filepath, DataFrame)

    for row in eachrow(df)
        h = Int(row.h)
        w = Int(row.w)
        data[w, h, pde_idx] = Int(row.PDE)
    end
end

# =============================================================================
# Modell-Step & Initialisierung
# =============================================================================
function model_step!(model)
    agents_vec = collect(allagents(model))
    for i in eachindex(agents_vec)
        step_agent_dummy!(agents_vec[i], model)
    end

    apply_decay!(model.data, model.agent_mask, cAMP_idx, PDEe_idx, dt)

    # Strang-Splitting: halber Diffusions-Schritt + halber Diffusions-Schritt
    apply_diffusion!(model.data, DIFFUSION_RATES, dt / 2)
    apply_diffusion!(model.data, DIFFUSION_RATES, dt / 2)

    model.currentTime += dt
end

function initialize_model(; grid_size = (WIDTH, HEIGHT), seed = 42)
    space = GridSpace(grid_size; periodic = false)
    data  = zeros(Int, GRID_SIZE)
    mask  = falses(WIDTH, HEIGHT)

    properties = Dict(
        :data        => data,
        :currentTime => 0.0,
        :agent_mask  => mask,
    )

    model = StandardABM(
        CellAgent, space;
        agent_step! = step_agent!,
        model_step! = model_step!,
        properties  = properties,
        rng         = Xoshiro(seed),
    )

    add_agents_unique_fast!(model)
    build_agent_mask!(model)

    return model
end

# =============================================================================
# Tracking & Export
# =============================================================================
function init_full_tracking(model)
    tracked = Dict{Int, Dict{Symbol, Vector{Int}}}()
    for a in allagents(model)
        tracked[a.id] = Dict(Symbol(s) => Int[] for s in species(a.network))
    end
    return tracked
end

function write_all_molecules(times, tracked, agents)
    output_dir = "output_csv_" * outputAdd
    mkpath(output_dir)

    println("Starte händischen Export der Agenten-Daten...")
    agent_names = Dict(a.id => a.name for a in agents)

    # --- 1. Moleküle pro Agent ---
    for (agent_id, data) in tracked
        agent_name = get(agent_names, agent_id, "unknown_$(agent_id)")
        file_path  = joinpath(output_dir, "agent_$(agent_name)_molecules.csv")

        open(file_path, "w") do io
            spec_keys   = collect(keys(data))
            clean_names = [replace(string(k), "(t)" => "") for k in spec_keys]

            # Header
            print(io, "Zeit")
            for name in clean_names
                print(io, ",", name)
            end
            print(io, "\n")

            # Datenzeilen
            for (t_idx, t_val) in enumerate(times)
                @printf(io, "%.4f", t_val)
                for k in spec_keys
                    vals = data[k]
                    if t_idx <= length(vals)
                        print(io, ",", vals[t_idx])
                    else
                        print(io, ",0")
                    end
                end
                print(io, "\n")
            end
        end
        println("Agent $agent_id fertig.")
    end

    # --- 2. Tracker-Arrays ---
    tracker_file_path = joinpath(output_dir, "agent_tracker.csv")
    open(tracker_file_path, "w") do io
        for a in agents
            print(io, a.name, ",")
        end
        print(io, "\n")
        for i in 1:STEPS
            for a in agents
                print(io, a.tracker[i], ",")
            end
            print(io, "\n")
        end
    end

    println("Alle Dateien in '$output_dir' gespeichert.")
end

# =============================================================================
# Main
# =============================================================================
function mainRun()
    model   = initialize_model()
    tracked = init_full_tracking(model)
    times   = Float64[]
    p       = ProgressMeter.Progress(STEPS; desc = "Simuliere", showspeed = true)

    if createMovie
        time_obs      = Observable(model.currentTime)
        fig           = Figure(resolution = (400 * SPECIES_COUNT, 400))
        observables   = [Observable(model.data[:, :, s]) for s in 1:SPECIES_COUNT]
        agent_pos_obs = Observable([Point2f(a.pos) for a in allagents(model)])

        for s in 1:SPECIES_COUNT
            title_obs = @lift("Species $s | Zeit: $(round($time_obs, digits=2))s")
            ax        = Axis(fig[1, s], title = title_obs, aspect = DataAspect())
            hm        = heatmap!(ax, observables[s], colormap = :viridis)
            scatter!(ax, agent_pos_obs; color = :white, markersize = 8,
                     strokewidth = 1, strokecolor = :black)
            Colorbar(fig[2, s], hm; vertical = false, label = "Intensity S$s")
        end

        record(fig, "multi_species_simulation_th.mp4",
               1:STEPS ÷ STEPSIZE;
               framerate = 1 ÷ (dt * STEPSIZE)) do i
            step!(model, STEPSIZE)

            time_obs[]      = model.currentTime
            agent_pos_obs[] = [Point2f(a.pos) for a in allagents(model)]
            for s in 1:SPECIES_COUNT
                observables[s][] = model.data[:, :, s]
            end

            for a in allagents(model)
                u  = a.integrator.u
                sp = species(a.network)
                for (i, s) in enumerate(sp)
                    push!(tracked[a.id][Symbol(s)], u[i])
                end
            end

            push!(times, model.currentTime)
            ProgressMeter.next!(p; step = STEPSIZE)
        end
    else
        @inbounds for i in 1:STEPS ÷ STEPSIZE
            step!(model, STEPSIZE)

            for a in allagents(model)
                u  = a.integrator.u
                sp = species(a.network)
                for (i, s) in enumerate(sp)
                    push!(tracked[a.id][Symbol(s)], u[i])
                end
            end

            push!(times, model.currentTime)
            ProgressMeter.next!(p; step = STEPSIZE)
        end
    end

    println("Starte Export")
    write_all_molecules(times, tracked, allagents(model))
end

# =============================================================================
# Entry Point
# =============================================================================
mainRun()
