# -----------------------------------------------------------------------------
# Author: Johannes Konrad
# Affiliation: Institute for Digital Communications, Friedrich-Alexander-Universität Erlangen-Nürnberg]
# Email: johannes.konrad@fau.de
#
# This code is associated with the manuscript:
# "The Effect of Cell Density and Enzymatic Clearance on Information Propagation Speed in a Cell Culture" (in preparation)
#
# The code implements the methods and algorithms described in the manuscript.
# Please refer to the paper for a detailed explanation of the methodology, 
# experimental design, and results.
# -----------------------------------------------------------------------------
using Agents, CairoMakie, Random
using Catalyst, JumpProcesses
using Distributions
import ProgressMeter
using Printf
using StaticArrays
using CSV
using DataFrames

println("Threads available: ", Threads.nthreads())
# Konstanten für die Simulation

const WIDTH = 300
const HEIGHT = 200
const SPECIES_COUNT = 2  # Anzahl der Spezies (Layer)
const dx = 3.0 # micro m
const cAMP_idx = 1
const PDEe_idx = 2;

const DIFFUSION_RATES = [350.0, 40.0] 

const DECAY_RATE = 1e5;
const DECAY_PDE = 0.01 / 60
const GRID_SIZE = (WIDTH, HEIGHT, SPECIES_COUNT)
const STEPS = 100_000 #00_000
const dt = 0.002

const createMovie = false 
const N_AGENTS = 4
const AGENT_DIST = 10
const MIN_BORDER_DIST = 50
const STEPSIZE = 10

const NA = 6.023e23
const V = 3.672e-14
const mu = 1e-6

const outputAdd = length(ARGS) > 0 ? ARGS[1] : "new"

const SPECIES_MAP = Dict{Symbol, Int}(:cAMPe => cAMP_idx, :PDEe => PDEe_idx)

const NOMINAL_U0 = Dict(
    :ACA   => 0.0,
    :PKA   => 0.0,
    :ERK2  => 0.0,
    :RegA  => 0.0,
    :cAMPi => 0.0,
    :cAMPe => 700.0,
    :CAR1  => 0.0,
    :PDEe => 0.0,
    :cAMPtracked => 0.0
)

# 1. Definiere das interne Reaktionsnetzwerk (Modell des Agents)
const rn = @reaction_network begin
    k1, CAR1 --> ACA + CAR1
    k2, ACA + PKA --> PKA
    k3, cAMPi --> PKA + cAMPi
    k4, PKA --> 0
    k5, CAR1 --> ERK2 + CAR1
    k6, PKA + ERK2 -->  PKA
    k7, 0 --> RegA
    k8, ERK2 + RegA --> ERK2
    k9, ACA --> cAMPi + ACA
    k10, RegA + cAMPi --> RegA
    k11, ACA --> cAMPe + ACA + cAMPtracked
    #k12, cAMPe --> 0    Nicht relevant
    k13, cAMPe --> CAR1 + cAMPe
    k14, CAR1 --> 0
    ## NEW ##
    k15_basal, 0 --> PDEe
    (k15_ind * (K^2 * CAR1^2) / (K^2 + CAR1^2)^2), 0 --> PDEe
    #envoirement
    decay_rate, cAMPe + PDEe --> PDEe
    decay_pde, PDEe --> 0
end

# Einziger Agenten-Typ
@agent struct CellAgent(GridAgent{2})
    name::String
    network::ReactionSystem
    integrator::JumpProcesses.SSAIntegrator
    camp_u_idx::Int
    camptrack_u_idx::Int
    pde_u_idx::Int
    tracker::Array{Int}
end

@inline function step_agent!(agent, model)
end

@inbounds function step_agent_dummy!(agent::CellAgent, model)
    # 2. Molekülzahlen aus dem Grid für diesen Voxel (pos) extrahieren
    # Annahme: grid_data ist ein 3D-Array [H, W, Species]
    h, w = agent.pos
    int = agent.integrator
    rn = agent.network

    # I. Synchronisation: Grid -> Agent (interne Molekülzahlen updaten)
    # Wir mappen die Grid-Werte auf die entsprechenden Indizes im Catalyst-System
    #for (i, spec) in enumerate(species(rn))
    #    spec_sym = ModelingToolkit.getname(spec)
    #    if haskey(SPECIES_MAP, spec_sym)
    #        # Direkte Modifikation des Zustandsvektors im Integrator
    #        int.u[i] = model.data[h, w, SPECIES_MAP[spec_sym]]
    #    end
    #end
    int.u[agent.camp_u_idx] = model.data[h, w, SPECIES_MAP[:cAMPe]]
    int.u[agent.pde_u_idx] = model.data[h, w, SPECIES_MAP[:PDEe]]
    int.u[agent.camptrack_u_idx] = 0

    # II. Raten neu berechnen
    # Da wir 'u' manuell geändert haben, müssen die Sprungwahrscheinlichkeiten 
    # für den nächsten Gillespie-Schritt aktualisiert werden [web:47].
    reset_aggregated_jumps!(int)

    # III. Simulation für dt ausführen
    # 'step!' führt die Simulation exakt um model.dt weiter
    step!(int, dt, true)

    # 5. Ergebnisse zurück ins Grid schreiben
    # sol.u[end] enthält die Molekülzahlen nach Ablauf von dt
    #CAMP_RN_IDX = findfirst(s -> ModelingToolkit.getname(s) == :cAMPe, species(rn))
    #PDE_RN_IDX = findfirst(s -> ModelingToolkit.getname(s) == :PDEe, species(rn))
    #println(int.u[CAMP_RN_IDX])
    # Dann in step_agent!:
    agent.tracker[abmtime(model) + 1] = int.u[agent.camptrack_u_idx]
    model.data[h, w, SPECIES_MAP[:cAMPe]] = int.u[agent.camp_u_idx]
    model.data[h, w, SPECIES_MAP[:PDEe]]  = int.u[agent.pde_u_idx]

end

@inline function draw_fluxes!(buf, n::Int, p::Float64)
    # Kategorien: N, S, O, W, Stay
    r = n

    x1 = rand(Binomial(r, p))
    r -= x1

    q2 = p / (1 - p)
    x2 = rand(Binomial(r, q2))
    r -= x2

    q3 = p / (1 - 2p)
    x3 = rand(Binomial(r, q3))
    r -= x3

    q4 = p / (1 - 3p)
    x4 = rand(Binomial(r, q4))
    r -= x4

    buf[1] = x1
    buf[2] = x2
    buf[3] = x3
    buf[4] = x4
    buf[5] = r
    return nothing
end

function apply_diffusion!(data::Array{Int, 3}, rates, dt)
    height, width, species_count = size(data)
    # delta speichert die Netto-Änderungen pro Voxel
    delta = zeros(Int, height, width, species_count)
    
    for s in 1:species_count

        p_jump = rates[s] * dt / (dx^2)
        
        # Stabilitätscheck
        if p_jump > 0.24
            @warn "Stabilitätsrisiko: p_jump ($p_jump) zu hoch!"
        end

        # Wahrscheinlichkeitsvektor: [Nord, Süd, Ost, West, Bleiben]
        #p_vec = [p_jump, p_jump, p_jump, p_jump, 1.0 - 4.0 * p_jump]

        fluxes = MVector{5, Float64}(undef)
        @inbounds for h in 1:height
            for w in 1:width
                n_total = data[h, w, s]
                n_total <= 0 && continue
                draw_fluxes!(fluxes, n_total, p_jump, )
                # Ziehe alle Flüsse gleichzeitig (eliminiert Richtungs-Bias)
                #fluxes = rand(Multinomial(n_total, p_vec))
                
                # 1. Abzug vom aktuellen Voxel
                n_moving = n_total - fluxes[5]
                delta[h, w, s] -= n_moving #n_moving

                # 2. Verteilung auf Nachbarn mit absorbierenden Randbedingungen
                # Teilchen, die über den Rand springen, werden abgezogen (Schritt 1),
                # aber hier nicht wieder hinzugefügt.
                
                # Nord
                h < height && (delta[h + 1, w, s] += fluxes[1])
                # Süd
                h > 1      && (delta[h - 1, w, s] += fluxes[2])
                # Ost
                w < width  && (delta[h, w + 1, s] += fluxes[3])
                # West
                w > 1      && (delta[h, w - 1, s] += fluxes[4])
            end
        end
    end
    
    # Änderungen auf das Hauptgitter anwenden
    data .+= delta
end

@inline function build_agent_mask!(model)
    for agent in allagents(model)
        h, w = agent.pos          # Agents.jl-Konvention: pos = (row, col)
        @inbounds model.agent_mask[h, w] = true
    end
end

@inline function decay_TAU(n_src, n_kat, c_cAMP, c_PDE, dt::Float64)
            # Erwartete Events im Intervall dt
            λ_camp = (c_cAMP * n_src * n_kat) * dt
            λ_pde  = (c_PDE  * n_kat)         * dt

            # Ziehe Anzahl Reaktionen (stochastisch)
            d_camp = (λ_camp > 0) ? rand(Poisson(λ_camp)) : 0
            d_pde  = (λ_pde  > 0) ? rand(Poisson(λ_pde))  : 0

            # Nie mehr verbrauchen als da ist
            if d_camp > n_src; d_camp = n_src; end
            if d_pde  > n_kat; d_pde  = n_kat; end

            n_src = n_src - d_camp
            n_kat = n_kat - d_pde
            return (n_src, n_kat)
end

@inline @inbounds function decay_SSA(n_src, n_kat, c_cAMP, c_PDE, endtime)
 t = 0.0
            while (n_src > 0 || n_kat > 0)
                # Propensitäten
                # prob_cAMP = n_src * n_kat * c_cAMP
                # prob_PDEe = n_kat * c_PDE
                prob_PDEe = n_kat * c_PDE
                prob_cAMP = (n_src * n_kat) * c_cAMP
                total = prob_cAMP + prob_PDEe

                # total kann nur 0 werden, wenn n_kat==0 (dann würde while bald enden),
                # aber zur Sicherheit:
                total <= 0.0 && break

                # Gillespie-Zeitschritt
                t += randexp() / total
                t > endtime && break

                # Ereignis auswählen
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

    # Vorfaktoren: vermeidet teure Divisionen im Hot-Loop
    c_cAMP = DECAY_RATE / (NA * dx^3)   # multipliziert mit n_src*n_kat
    c_PDE  = DECAY_PDE                             # multipliziert mit n_kat

    for w in 1:width
        @inbounds for h in 1:height
            
            agent_mask[w, h] && continue

            n_src = data[h, w, src_idx]
            n_kat = data[h, w, kat_idx]

            # Keine Reaktion möglich
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

function generate_initial_u0(pd, species_list)

    u0_dict::Dict{Symbol, Int} = Dict();

    for (spec, nominal) in NOMINAL_U0
        if spec in species_list
            d_i = rand() * 2 - 1 
            perturbed_value = nominal * (1 + pd * d_i)
            
            # SSA benötigt Ganzzahlen
            val = Int(round(perturbed_value))
            push!(u0_dict, spec => val)
        else
            push!(u0_dict, spec => 0)
        end
    end

    return u0_dict
end

function addCellAgent(k15_ind, K, model::AgentBasedModel, pos; has_cAMPe=true, name="")
    # Der berechnete Skalierungsfaktor (nA * V * 10^-6)
    TIME_SCALE = 60

    # Nominalwerte nach dem Text
    reaction_params = Dict(
        # Unimolekular: nur / TIME_SCALE
        :k1  => 1.08,               # war 1.08, falscher Wert
        :k3  => 2.5  / TIME_SCALE,               # kein mu!
        :k4  => 1.5  / TIME_SCALE,               # kein mu!
        :k5  => 0.6  / TIME_SCALE,               # kein mu!
        :k9  => 0.3  / TIME_SCALE,
        :k11 => 1.0,               # war 1.0 ohne /TS
        :k13 => 0.71,               # war 0.71 — 32× falsch!
        :k14 => 4.5  / TIME_SCALE,

        # Bimolekular: / NA / V / mu / TIME_SCALE
        :k2  => 0.9  / NA / V / mu / TIME_SCALE,
        :k6  => 0.8  / NA / V / mu / TIME_SCALE, # war /mu/NA/V — Reihenfolge egal, Wert gleich
        :k8  => 1.3  / NA / V / mu / TIME_SCALE,
        :k10 => 0.8  / NA / V / mu / TIME_SCALE,

        # Nullte Ordnung: * NA * V * mu / TIME_SCALE
        :k7  => 1.0  * NA * V * mu / TIME_SCALE,

        # PDEe — nullte Ordnung (Produktion)
        :k15_basal => 3e-4 * NA * V * mu,        # µM/s → Moleküle/s, kein TIME_SCALE nötig wenn schon in s
        :k15_ind   => k15_ind * NA * V * mu / TIME_SCALE,

        # K bleibt in Molekülzahlen
        :K => K,

        # Extrazelluläre Reaktionen — Voxelvolumen, nicht Zellvolumen
        :decay_rate => DECAY_RATE / (NA * dx^3 * 1e-18),
        :decay_pde  => DECAY_PDE
    )

    tspan = (0.0, dt)

    SPECIES_LIST = [:PKA, :ERK2, :RegA, :cAMPi, :PDEe]
    if has_cAMPe
        push!(SPECIES_LIST, :cAMPe, :CAR1, :ACA)
    end

    initial_u0 = generate_initial_u0(0.05, SPECIES_LIST)

    val_cAMPe = initial_u0[:cAMPe]

    model.data[pos..., SPECIES_MAP[:cAMPe]] = val_cAMPe
    rn_species_syms = ModelingToolkit.getname.(species(rn))  # einmal

    camp_rn_idx = findfirst(==( :cAMPe ), rn_species_syms)
    pde_rn_idx  = findfirst(==( :PDEe  ), rn_species_syms)
    camptrack_u_idx = findfirst(==(:cAMPtracked), rn_species_syms)

    prob = DiscreteProblem(rn, initial_u0, (0.0, dt), reaction_params)

    jp = JumpProblem(rn, prob, NRM(); save_positions=(false,false))

    agent_integrator = init(jp, SSAStepper())

    add_agent!(pos, CellAgent, model; network=rn, integrator = agent_integrator, camp_u_idx=camp_rn_idx, pde_u_idx=pde_rn_idx, camptrack_u_idx=camptrack_u_idx, tracker=zeros(Int, STEPS), name=name);
end

function model_step!(model)
    agents_vec = collect(allagents(model))
    #t1 = Threads.@spawn begin
    for i in eachindex(agents_vec)
        step_agent_dummy!(agents_vec[i], model)
        #end
    end

    #t2 = Threads.@spawn begin
        apply_decay!(model.data, model.agent_mask, cAMP_idx, PDEe_idx, dt)
    #end

    #wait(t1)
    #wait(t2)
    apply_diffusion!(model.data, DIFFUSION_RATES, dt / 2)
    apply_diffusion!(model.data, DIFFUSION_RATES, dt / 2)
    model.currentTime += dt;
end

function add_agents_unique_fast!(
    k15_ind,
    K,
    model
)

    N_AGENTS >= 1 || throw(ArgumentError("N_AGENTS muss mindestens 1 sein."))
    AGENT_DIST >= 1 || throw(ArgumentError("AGENT_DIST muss mindestens 1 sein."))
    WIDTH >= 1 || throw(ArgumentError("WIDTH muss mindestens 1 sein."))
    HEIGHT >= 1 || throw(ArgumentError("HEIGHT muss mindestens 1 sein."))

    # Alle Zellen in einer horizontalen Reihe bei h = const, vertikal mittig
    min_border_dist = MIN_BORDER_DIST;
    h = cld(HEIGHT, 2)

    # Abstand zwischen linker und rechter äußerster Zelle
    total_span = (N_AGENTS - 1) * AGENT_DIST

    # Linke Startposition möglichst mittig im Grid
    # (bei nicht exakt darstellbarer Zentrierung auf Integer-Koordinaten wird nach rechts gerundet)
    x_left = ceil(Int, (WIDTH + 1 - total_span) / 2)
    x_right = x_left + total_span

    # Fehler, wenn äußerste Zellen außerhalb des Grids liegen
    if x_left < 1 || x_right > WIDTH
        throw(ArgumentError(
            "Die äußersten Zellen lieWIDTH, HEIGHT, N_AGENTS, AGENT_DIST = parse_args()gen außerhalb des Grids: " *
            "x_left=$x_left, x_right=$x_right, erlaubt ist 1:$WIDTH."
        ))
    end

    # Warnung, wenn Mindestabstand zum Rand unterschritten wird
    left_margin = x_left - 1
    right_margin = WIDTH - x_right
    if left_margin < min_border_dist || right_margin < min_border_dist
      @warn "Die äußersten Zellen unterschreiten den Mindestabstand zum Rand." left_margin right_margin min_border_dist x_left x_right h
    end

    for i in 1:N_AGENTS
        x = x_left + (i - 1) * AGENT_DIST
        pos = (x, h)

        name = if i == 1
            "TX"
        elseif i == N_AGENTS
            "RX"
        else
            "R$(i - 1)"
        end

        addCellAgent(
            k15_ind,
            K,
            model,
            pos;
            has_cAMPe = (i == 1),
            name = name,
        )
    end
end

function initialize_model(k15_ind, K; grid_size = (WIDTH, HEIGHT), seed = 42)
    space = GridSpace(grid_size; periodic = false)
    
    # Das Grid wird als Eigenschaft des Modells gespeichert
    data = zeros(Int, GRID_SIZE)

    # PDEe überall auf 22 Moleküle setzen
    
function init_pde_from_csv!(data, PDEe_idx, WIDTH, HEIGHT, N_AGENTS, DIST)

    filepath = joinpath(
        "output_pde",
        "PDE_WIDTH_$(WIDTH)_HEIGHT_$(HEIGHT)_N_AGENTS_$(N_AGENTS)_DIST_$(DIST).csv"
    )

    println("Lade PDE Initialisierung aus: $filepath")

    df = CSV.read(filepath, DataFrame)

    for row in eachrow(df)
        h = Int(row.h)
        w = Int(row.w)

        data[w, h, PDEe_idx] = Int(row.PDE)
    end

end
#init_pde_from_csv!(data, PDEe_idx, WIDTH, HEIGHT, N_AGENTS, AGENT_DIST)

    mask = falses(WIDTH, HEIGHT)

    properties = Dict(
        :data => data,
        :currentTime => 0.0,
        :agent_mask => mask
    )
    
    
    model = StandardABM(
        CellAgent, 
        space; 
        agent_step! = step_agent!, 
        model_step! = model_step!,
        properties = properties,
        rng = Xoshiro(seed)
    )

    add_agents_unique_fast!(k15_ind, K, model)

    build_agent_mask!(model) 

    return model
end


function write_all_molecules(times, tracked, agents)
    output_dir = "output_csv_" * outputAdd
    mkpath(output_dir)

    println("Starte händischen Export der Agenten-Daten...")
    agent_names = Dict(a.id => a.name for a in agents)

    for (agent_id, data) in tracked
        # 1. Datei öffnen
        agent_name = get(agent_names, agent_id, "unknown_$(agent_id)")

        # Datei öffnen
        file_path = joinpath(output_dir, "agent_$(agent_name)_molecules.csv")
        
        open(file_path, "w") do io
            # 2. Header (Spaltennamen) vorbereiten
            # Wir holen alle Spezies-Keys und bereinigen sie
            spec_keys = collect(keys(data))
            clean_names = [replace(string(k), "(t)" => "") for k in spec_keys]
            
            # Header schreiben: Zeit, Spec1, Spec2...
            print(io, "Zeit")
            for name in clean_names
                print(io, ",", name)
            end
            print(io, "\n")

            # 3. Daten Zeile für Zeile schreiben
            for (t_idx, t_val) in enumerate(times)
                # Zeitwert schreiben (formatiert auf 4 Nachkommastellen)
                @printf(io, "%.4f", t_val)
                
                # Für jede Spezies den Wert an diesem Zeitindex schreiben
                for k in spec_keys
                    vals = data[k]
                    if t_idx <= length(vals)
                        print(io, ",", vals[t_idx])
                    else
                        print(io, ",0") # Falls Daten fehlen, schreibe 0
                    end
                end
                print(io, "\n")
            end
        end
        println("Agent $agent_id fertig.")
    end

    # =========================
    # 2. Tracker-Array speichern
    # =========================
    tracker_file_path = joinpath(output_dir, "agent_tracker.csv")
    open(tracker_file_path, "w") do io
            for a in agents
              print(io, a.name, ",")
            end
            print(io, "\n")
            for i=1:STEPS
              for a in agents
                print(io, a.tracker[i], ",")
              end
              print(io, "\n")
            end
    end


    println("Alle Dateien in '$output_dir' gespeichert.")
end

function init_full_tracking(model)
    tracked = Dict{Int, Dict{Symbol, Vector{Int}}}()

    for a in allagents(model)
        tracked[a.id] = Dict(
            Symbol(s) => Int[] for s in species(a.network)
        )
    end

    return tracked
end

function mainRun(k15_ind, K)

    model = initialize_model(k15_ind, K);

    total = STEPS;

    p = ProgressMeter.Progress(total; desc = "Simuliere", showspeed = true)

    tracked = init_full_tracking(model)
    times = Float64[]

    if createMovie


        time_obs = Observable(model.currentTime)

        # Figure vorbereiten
        fig = Figure(resolution = (400 * SPECIES_COUNT, 400))
        # Listen für Observables und Plots
        observables = [Observable(model.data[:, :, s]) for s in 1:SPECIES_COUNT]

        agent_pos_obs = Observable([Point2f(a.pos) for a in allagents(model)])

        # Dynamische Erstellung der Achsen und Colorbars
        for s in 1:SPECIES_COUNT
            # 2. Titel dynamisch mit @lift an das Zeit-Observable binden
            title_obs = @lift("Species $s | Zeit: $(round($time_obs, digits=2))s")
            
            ax = Axis(fig[1, s], title = title_obs, aspect = DataAspect())
            hm = heatmap!(ax, observables[s], colormap = :viridis)

            scatter!(ax, agent_pos_obs, color = :white, markersize = 8, strokewidth = 1, strokecolor = :black)

            Colorbar(fig[2, s], hm, vertical = false, label = "Intensity S$s")
        end

    

        # Video aufnehmen
        record(fig, "multi_species_simulation_th.mp4", 1:STEPS ÷ STEPSIZE; framerate = 1 ÷ (dt * STEPSIZE)) do i
        #for i in 1:STEPS
            step!(model, STEPSIZE)
            # Alle Spezies-Ebenen gleichzeitig aktualisieren
            #println(sum(model.data))
            time_obs[] = model.currentTime
            agent_pos_obs[] = [Point2f(a.pos) for a in allagents(model)]
            for s in 1:SPECIES_COUNT
                observables[s][] = model.data[:, :, s]
            end

            # 🔴 ALLE MOLEKÜLE AUS ALLEN AGENTEN
            for a in allagents(model)
                u = a.integrator.u
                sp = species(a.network)

                for (i, s) in enumerate(sp)
                    push!(tracked[a.id][Symbol(s)], u[i])
                end
            end

            push!(times, model.currentTime)

            ProgressMeter.next!(p; step=STEPSIZE)
        end
    else
        @inbounds for i in 1:STEPS ÷ STEPSIZE
            step!(model, STEPSIZE)
            for a in allagents(model)
                u = a.integrator.u
                sp = species(a.network)

                for (i, s) in enumerate(sp)
                    push!(tracked[a.id][Symbol(s)], u[i])
                end
            end

            push!(times, model.currentTime)

            ProgressMeter.next!(p; step=STEPSIZE)
        end
    end 
    println("Starte")
    write_all_molecules(times, tracked, allagents(model))

end
tmp = ["", "1.08", "4000"]
if length(ARGS) == 3
    tmp = ARGS
end

const k15_ind = parse(Float64, tmp[2])
const K = parse(Float64, tmp[3])

println("k15_ind = $k15_ind, K = $K")

mainRun(k15_ind, K)
