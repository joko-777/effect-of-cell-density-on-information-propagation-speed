using CSV
using DataFrames
using Statistics
using Random
using NearestNeighbors
using SpecialFunctions
using Plots
# ----------------------------- 
# ⚙️ SETTINGS
# ----------------------------- 
base_dir = "results"
group_id = ARGS[1]  # "2", "3" oder "4"

file_A = "agent_tracker.csv"
file_B = "agent_RX_molecules.csv"

species_A = "TX"
species_B = "CAR1"

# KSG Parameter
k_neighbors = 20

# Subsampling Parameter
max_n = 5000
rng_seed = 42

# Minimale Varianz für gültige Berechnung
min_variance = 1e-4

# Zeitschritte überspringen (für schnellere Berechnung)
time_step = 1  # Jeden time_step-ten Zeitpunkt berechnen

# Subsampling beim Einlesen
STEP_IN_FILE_A = occursin("tracker", file_A) ? 10 : 1
STEP_IN_FILE_B = occursin("tracker", file_B) ? 10 : 1

mkpath("eval/cmiksg$(group_id)")
# ----------------------------- 
# 📂 Daten laden
# ----------------------------- 
function load_runs(base_dir, group_id, file; STEPSIZE=1)
    folder_regex = Regex("output_csv_N$(group_id)_round\\d+")
    dfs = DataFrame[]
    
    for entry in readdir(base_dir)
        if occursin(folder_regex, entry)
            path = joinpath(base_dir, entry, file)
            if isfile(path)
                df = CSV.read(path, DataFrame)
                df = df[1:STEPSIZE:end, :]
                push!(dfs, df)
            end
        end
    end
    return dfs
end

# ----------------------------- 
# 🔧 KSG-Algorithmus
# ----------------------------- 

function ksg_mi(x::Vector{Float64}, y::Vector{Float64}; k::Int=3, max_n::Int=5000, rng_seed::Int=42)
    N = length(x)
    
    if N != length(y) || N <= k + 1
        return 0
    end
    
    # Varianz-Check
    if std(x) < min_variance || std(y) < min_variance
        return 0
    end
    
    if N > max_n
        rng = MersenneTwister(rng_seed)
        idx = randperm(rng, N)[1:max_n]
        x, y = x[idx], y[idx]
        N = max_n
    end
    
    x = (x .- mean(x)) ./ std(x)
    y = (y .- mean(y)) ./ std(y)
    
    joint_points = hcat(x, y)'
    joint_tree = KDTree(joint_points, Chebyshev())
    
    x_points = reshape(x, 1, :)
    y_points = reshape(y, 1, :)
    x_tree = KDTree(x_points, Chebyshev())
    y_tree = KDTree(y_points, Chebyshev())
    
    psi_sum = 0.0
    
    for i in 1:N
        _, dists = knn(joint_tree, joint_points[:, i], k + 1)
        ε = max(dists[end], 1e-10)
        
        n_x = max(length(inrange(x_tree, x_points[:, i], ε, false)) - 1, 1)
        n_y = max(length(inrange(y_tree, y_points[:, i], ε, false)) - 1, 1)
        
        psi_sum += digamma(n_x + 1) + digamma(n_y + 1)
    end
    
    return max(0.0, digamma(k) - psi_sum / N + digamma(N))
end

function ksg_cmi(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}; 
                 k::Int=3, max_n::Int=5000, rng_seed::Int=42)
    N = length(x)
    
    if N != length(y) || N != length(z) || N <= k + 1
        return 0
    end
    
    # Varianz-Check
    if std(x) < min_variance || std(y) < min_variance || std(z) < min_variance
        return 0
    end
    
    if N > max_n
        rng = MersenneTwister(rng_seed)
        idx = randperm(rng, N)[1:max_n]
        x, y, z = x[idx], y[idx], z[idx]
        N = max_n
    end
    
    x = (x .- mean(x)) ./ std(x)
    y = (y .- mean(y)) ./ std(y)
    z = (z .- mean(z)) ./ std(z)
    
    xyz_points = hcat(x, y, z)'
    xz_points = hcat(x, z)'
    yz_points = hcat(y, z)'
    z_points = reshape(z, 1, :)
    
    xyz_tree = KDTree(xyz_points, Chebyshev())
    xz_tree = KDTree(xz_points, Chebyshev())
    yz_tree = KDTree(yz_points, Chebyshev())
    z_tree = KDTree(z_points, Chebyshev())
    
    psi_sum = 0.0
    
    for i in 1:N
        _, dists = knn(xyz_tree, xyz_points[:, i], k + 1)
        ε = max(dists[end], 1e-10)
        
        n_xz = max(length(inrange(xz_tree, xz_points[:, i], ε, false)) - 1, 1)
        n_yz = max(length(inrange(yz_tree, yz_points[:, i], ε, false)) - 1, 1)
        n_z  = max(length(inrange(z_tree, z_points[:, i], ε, false)) - 1, 1)
        
        psi_sum += digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1)
    end
    
    return max(0.0, digamma(k) + psi_sum / N)
end

function extract_species_timeseries(df::DataFrame, species::String)
    if species in names(df)
        return Float64.(df[!, species])
    else
        error("Spezies '$species' nicht gefunden. Verfügbar: $(names(df))")
    end
end

# ----------------------------- 
# 🚀 HAUPTPROGRAMM
# ----------------------------- 

function main(delta, n_runs, runs_A, runs_B)
    
    # Extrahiere alle Zeitreihen
    println("\n📊 Extrahiere Zeitreihen...")
    tx_series = [extract_species_timeseries(runs_A[i], species_A) for i in 1:n_runs]
    car1_series = [extract_species_timeseries(runs_B[i], species_B) for i in 1:n_runs]
    
    # Bestimme gemeinsame Zeitlänge
    max_time = minimum(min(length(tx_series[i]), length(car1_series[i])) for i in 1:n_runs)
    println("Gemeinsame Zeitlänge: $max_time")
    println("Anzahl Runs (= Samples pro Zeitpunkt): $n_runs")
    
    if n_runs <= k_neighbors + 1
        println("⚠️  Warnung: Wenige Runs ($n_runs). KSG benötigt mindestens k+2 = $(k_neighbors+2) Samples.")
    end
    
    # Zeitpunkte für Berechnung
    time_points = 1:time_step:(max_time - delta)
    n_times = length(time_points)
    println("Zeitpunkte zu berechnen: $n_times")
    
    # Arrays für Ergebnisse
    cmi_values = Vector{Float64}(undef, n_times)
    mi_values = Vector{Float64}(undef, n_times)
    
    # Berechnung
    println("\n🔄 Berechne CMI für jeden Zeitpunkt...")
    
    for idx in 1:n_times
        t = time_points[idx]
        
        # Sammle Samples über alle Runs für diesen Zeitpunkt
        X = Float64[]  # TX(t)
        Y = Float64[]  # CAR1(t + delta)
        Z = Float64[]  # CAR1(t)
        
        for i in 1:n_runs
            push!(X, tx_series[i][t])
            push!(Y, car1_series[i][t + delta])
            push!(Z, car1_series[i][t])
        end
        
        # Berechne CMI und MI
        cmi_values[idx] = ksg_cmi(X, Y, Z; k=k_neighbors, max_n=max_n)
        mi_values[idx] = ksg_mi(X, Y; k=k_neighbors, max_n=max_n)
    end

    println("Delta ", delta, " is ready!")
    
    # Ergebnisse speichern
    results = DataFrame(
        t = collect(time_points),
        CMI = cmi_values,
        MI = mi_values,
        info_by_history = mi_values .- cmi_values
    )
    
    outfile = "eval/cmiksg" * group_id * "/cmi_over_time_group$(group_id)_delta$(delta).csv"
    CSV.write(outfile, results)
    println("\n💾 Ergebnisse gespeichert: $outfile")
    
    # Zusammenfassung
    println("\n" * "=" ^ 60)
    println("ZUSAMMENFASSUNG")
    println("=" ^ 60)
    println("Delta: $delta")
    println("Anzahl Runs: $n_runs")
    return results
end

# Ausführen
function deltaRun()

    println("=" ^ 60)
    println("CMI über Zeit: I(CAR1(t+δ), TX(t) | CAR1(t))")
    println("Festes δ, Aggregation über Runs")
    println("=" ^ 60)
    
    # Daten laden
    println("\n📂 Lade Daten...")
    runs_A = load_runs(base_dir, group_id, file_A; STEPSIZE=STEP_IN_FILE_A)
    runs_B = load_runs(base_dir, group_id, file_B; STEPSIZE=STEP_IN_FILE_B)
    
    n_runs = min(length(runs_A), length(runs_B))
    println("Runs geladen: $n_runs")
    
    if n_runs == 0
        error("Keine Daten gefunden!")
    end

    delta_range = 0:50:3000

    max_cmi = Vector{Float64}(undef, length(delta_range))

    println("\n🔄 Berechne max(CMI) für jedes δ ...")

    t = collect(enumerate(delta_range))
   Threads.@threads for (idx, d) in t
        results = main(d, n_runs, runs_A, runs_B)

        cmi = results.CMI
        valid_cmi = filter(!isnan, cmi)

        if !isempty(valid_cmi)
            max_cmi[idx] = maximum(valid_cmi)
        else
            max_cmi[idx] = NaN
        end

        println("  δ = $d → max CMI = $(round(max_cmi[idx], digits=4))")
    end

    # Ergebnisse speichern
    delta_results = DataFrame(
        delta = collect(delta_range),
        max_CMI = max_cmi
    )
    CSV.write("eval/cmiksg" * group_id * "/max_cmi_over_delta.csv", delta_results)

 
end

deltaRun()

