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
using Catalyst
using ModelingToolkit
using DifferentialEquations
using GLMakie
GLMakie.activate!()
using JSON3
using Printf
using Symbolics

const TAU = 1.0

@inline scale_rate(k::Real) = TAU * float(k)      # 1/s -> 1/tu
@inline scale_diff(D::Real) = TAU * float(D)      # µm^2/s -> µm^2/tu
@inline scale_time(T::Real) = float(T) / TAU      # s -> tu

# --------------------------
# 1) Initial Reaction conditions (not time-scaled)
# --------------------------
const U0_PAIRS_DEFAULT = [
    # Zelle 1
    :CAR1  => 0.0,
    :ACA   => 0.0,
    :PKA   => 0.0,
    :cAMPi => 0.0,
    :ERK2  => 0.0,
    :RegA  => 0.0,
    :cAMPe => 100.0,
    # Zelle 2
    :CAR12 => 0.0,
    :ACA2  => 0.0,
    :PKA2  => 0.0,
    :cAMPi2=> 0.0,
    :ERK22 => 0.0,
    :RegA2 => 0.0,
    :cAMPe2=> 00.0,
]


const P_PAIRS_PHYS = [
    :k1  => 1.08, #4.47,   # min^-1
    :k2  => 0.9   / 60.0,   # µM^-1 min^-1
    :k3  => 2.5   / 60.0,   # min^-1
    :k4  => 1.5   / 60.0,   # min^-1
    :k5  => 0.6   / 60.0,   # min^-1
    :k6  => 0.8   / 60.0,   # µM^-1 min^-1
    :k7  => 1.0   / 60.0,   # µM min^-1  (im Paper speziell erwähnt)
    :k8  => 1.3   / 60.0,   # µM^-1 min^-1
    :k9  => 0.3   / 60.0,   # min^-1
    :k10 => 0.8   / 60.0,   # µM^-1 min^-1
    :k11 => 1.00, #5.62,   # min^-1
    :k13 => 0.71, #2.51,   # min^-1
    :k14 => 4.5   / 60.0,   # min^-1
]

# Simulation rates (scaled by TAU). This is what you pass into the ODE.
const P_PAIRS_DEFAULT = [ s => scale_rate(v) for (s, v) in P_PAIRS_PHYS ]

# --------------------------
# 3) ODE solver defaults (times are in tu)
# --------------------------
# Physical horizon for DDE/ODE (seconds):
const TSPAN_PHYS = (0.0, 4020.0)

# Scaled horizon:
const TSPAN = (scale_time(TSPAN_PHYS[1]), scale_time(TSPAN_PHYS[2]))

const SOLVER_DEFAULT = Rodas5P()
const RELTOL_DEFAULT = 1e-6
const ABSTOL_DEFAULT = 1e-9

# Physical save step (seconds) and scaled save step (tu)
const SAVEAT_PHYS    = 5.0
const SAVEAT_DEFAULT = scale_time(SAVEAT_PHYS)


#------------ load kernel ----------------- #
function load_fit_json(filename::AbstractString; as_namedtuple::Bool=true)
    d = JSON3.read(read(filename, String))  # JSON3.Object
    # convert JSON3.Object -> Dict{String,Any} recursively
    dict = JSON3.copy(d)

    if as_namedtuple
        # convert keys to Symbols for dot-access
        return (; (Symbol(k) => v for (k,v) in dict)...)
    else
        return dict
    end
end

println("Load kernel...")
kern = load_fit_json("fitparams_geht.json")
println("Kernel loaded!\nInitalize Reactions...")

# ------------ Reactions network ---------------- #
rn = @reaction_network begin
    # ---------- Cell 1 ----------
    k1,  CAR1  --> ACA  + CAR1
    k2,  ACA   + PKA   --> PKA
    k3,  cAMPi --> PKA + cAMPi
    k4,  PKA   --> 0
    k5,  CAR1  --> ERK2 + CAR1
    k6,  PKA   + ERK2 --> PKA
    k7,  0     --> RegA
    k8,  ERK2  + RegA --> ERK2
    k9,  ACA   --> cAMPi + ACA
    k10, RegA  + cAMPi --> RegA
    k11, ACA   --> cAMPe + ACA
    k13, cAMPe --> CAR1 + cAMPe
    k14, CAR1  --> 0


    # ---------- Cell 2 ----------
    k1,  CAR12  --> ACA2  + CAR12
    k2,  ACA2   + PKA2   --> PKA2
    k3,  cAMPi2 --> PKA2 + cAMPi2
    k4,  PKA2   --> 0
    k5,  CAR12  --> ERK22 + CAR12
    k6,  PKA2   + ERK22 --> PKA2
    k7,  0      --> RegA2
    k8,  ERK22  + RegA2 --> ERK22
    k9,  ACA2   --> cAMPi2 + ACA2
    k10, RegA2  + cAMPi2 --> RegA2
    k11, ACA2   --> cAMPe2 + ACA2
    k13, cAMPe2 --> CAR12 + cAMPe2
    k14, CAR12  --> 0

end

println("Reaction Network initalized!")
base = convert(ODESystem, rn)
println("Reactions simplfied!")
# ----------------------------
# 2) Direkte Impulsantwort-Kopplung über Faltung (KERNELS)
# ----------------------------
"Parameter (Symbol=>Wert) -> Vector{Float64} in der Reihenfolge parameters(sys)"
function p_pairs_to_vec(ppairs, sys)
    d = Dict(ppairs)
    ps = parameters(sys)
    pvec = zeros(Float64, length(ps))
    for (i, par) in enumerate(ps)
        s = Symbol(ModelingToolkit.getname(par))
        pvec[i] = get(d, s, 0.0)
    end
    return pvec
end

function u0_pairs_to_vec(u0pairs, sts)
    d = Dict(u0pairs)
    u0 = zeros(Float64, length(sts))
    for (i, v) in enumerate(sts)
        s = Symbol(ModelingToolkit.getname(v))
        u0[i] = get(d, s, 0.0)
    end
    return u0
end

function createODE(kern, base, p0_pairs, u0_pairs; solver=SOLVER_DEFAULT, reltol=RELTOL_DEFAULT, abstol=ABSTOL_DEFAULT, saveat=SAVEAT_DEFAULT)
    println("Start creating ODEs...")
    t = ModelingToolkit.get_iv(base)
    D = Differential(t)
    @unpack cAMPe, cAMPe2 = base
    # ---------- Fit-Parameter: self (Σexp) ----------
    @assert hasproperty(kern, :alpha_self) && hasproperty(kern, :mu_self)
    αS = collect(kern.alpha_self)
    μS = collect(kern.mu_self)
    NS = length(αS)
    @assert length(μS) == NS
    @assert all(μS .> 0)

    # ---------- Fit-Parameter: cross (Σ Erlang, order n_cross) ----------
    @assert hasproperty(kern, :n_cross) && hasproperty(kern, :A_cross) && hasproperty(kern, :mu_cross)
    nC = Int(kern.n_cross)
    ACr = collect(kern.A_cross)   # length mC
    μCr = collect(kern.mu_cross)   # length mC
    mC = length(ACr)

    @assert nC >= 1
    @assert length(μCr) == mC
    @assert all(μCr .> 0)
    @assert all(ACr .>= 0)
    
    function createSelf(αSelf, μSelf, z)
        l = [];
        for i in eachindex(αSelf)
        push!(l, αSelf[i] * (-μSelf[i] * z + z))
        end
        return sum(l)
    end

    function createCross(ACr, μCr, source; w)
        totalL = [];
        finalList = [];
        for s in eachindex(ACr)
        offset = (s - 1) * nC;
        l = [D(w[1 + offset]) ~ μCr[s]*(source - w[1 + offset])]
        for i=2:nC
            eq = D(w[i + offset]) ~ μCr[s] * (w[i-1 + offset] - w[i + offset])
            push!(l, eq)
        end
        push!(finalList, ACr[s] * w[nC + offset])
        append!(totalL, l);
        end
        
        eq = sum(finalList) 
        return totalL, eq
    end
    #base = structural_simplify(base)
    eqs = equations(base)

    rhs_cAMPe = only(
        e.rhs
        for e in eqs
        if isequal(e.lhs, D(cAMPe))
    )

    rhs_cAMPe2 = only(
        e.rhs
        for e in eqs
        if isequal(e.lhs, D(cAMPe2))
    )

    baseOthers = collect(
        e
        for e in eqs
        if !isequal(e.lhs, D(cAMPe)) && !isequal(e.lhs, D(cAMPe2))
    )

    selfPart1 = createSelf(αS, μS, cAMPe);
    selfPart2 = createSelf(αS, μS, cAMPe2);

    @variables wTX(t)[1:nC * mC]
    @variables wRX(t)[1:nC * mC]
    l1, crossPart2 = createCross(ACr, μCr, cAMPe; w=wTX);
    l2, crossPart1 = createCross(ACr, μCr, cAMPe2; w=wRX);

    #stim(t) = 10 * exp(-(t-10)^2)   # glatter Step!

    cAMPe1Eq = D(cAMPe) ~ rhs_cAMPe + selfPart1 + crossPart1 #+ stim(t)
    cAMPe2Eq = D(cAMPe2) ~ rhs_cAMPe2 + selfPart2 + crossPart2

    coupleEquations = [cAMPe1Eq, cAMPe2Eq]
    append!(coupleEquations, l1)
    append!(coupleEquations, l2)
    append!(coupleEquations, baseOthers)
    sts = unique(vcat(unknowns(base), vec(wTX), vec(wRX)))
    ps  = parameters(base)
    @named full = ODESystem(coupleEquations, t, sts, ps)
    #full = compose(base, diff_sys)
    full = structural_simplify(full)

    # ------- calc steady state -------------
    #sts = unknowns(base)
    #u0_guess = [s => 0.1 for s in sts]
    #u0_guess = map(u0_guess) do (s,v)
    #    if isequal(s, base.cAMPe) || isequal(s, base.cAMPe2)
    #        s => 0.0
    #    else
    #        s => v
    #    end
    #end
    #base = structural_simplify(base)
    #ssprob = SteadyStateProblem(base, u0_guess, p_pairs_to_vec(p0_pairs, base))
    #println("Solve steady...")
    #sssol  = solve(ssprob)
    
    #sts_full = unknowns(full)
    #u0_map = Dict(v => sssol[v] for v in unknowns(base))
    #u0_full = [v => get(u0_map, v, 0.0) for v in sts_full]
    p_full = p_pairs_to_vec(p0_pairs, full)
    u_full = u0_pairs_to_vec(u0_pairs, unknowns(full))
    problem = ODEProblem(full, u_full, TSPAN, p_full)
    println("Start solving...")
    sol  = solve(problem, Tsit5(); reltol=1e-6, abstol=1e-9, saveat=0.1)
    println("Solved!")
    return sol, full
end 

# ------------ Figure ---------------- #

# --- Initialbedingungen (Pairs) ---
u0_default = copy(U0_PAIRS_DEFAULT)         # NICHT direkt U0_PAIRS_DEFAULT mutieren
u0_obs     = Observable(copy(u0_default))   # wird über IC-Apply Button gesetzt

# --- Reaction-Parameter Startwerte (Pairs) ---
p0 = copy(P_PAIRS_DEFAULT)                  # NICHT direkt P_PAIRS_DEFAULT mutieren

# --- Welche Variablen plotten? ---
vars1 = [:CAR1, :ACA, :PKA, :cAMPi, :ERK2, :RegA, :cAMPe]
vars2 = [:CAR12, :ACA2, :PKA2, :cAMPi2, :ERK22, :RegA2, :cAMPe2]

sol0, full0 = createODE(kern, base, P_PAIRS_DEFAULT, U0_PAIRS_DEFAULT)
sol_obs = Observable(sol0)
full_obs = Observable(full0)

fig = Figure(size=(1700, 950))

ax1 = Axis(fig[1, 1], xlabel="t", ylabel="amount", title="Zelle 1 (DDE, live)")
ax2 = Axis(fig[2, 1], xlabel="t", ylabel="amount", title="Zelle 2 (DDE, live)")

pts1 = [Observable(Point2f[]) for _ in vars1]
pts2 = [Observable(Point2f[]) for _ in vars2]

for (k, sym) in enumerate(vars1)
    lines!(ax1, pts1[k], label=string(sym))
end
axislegend(ax1, position=:rt)

for (k, sym) in enumerate(vars2)
    lines!(ax2, pts2[k], label=string(sym))
end
axislegend(ax2, position=:rt)

sym_of(v) = Symbol(ModelingToolkit.getname(v))

function build_idxmap(sys)
    sts = unknowns(sys)
    return Dict(sym_of(v) => i for (i, v) in enumerate(sts))
end

function update_plot!(sol, sys, vars1, vars2, pts1, pts2, ax1, ax2)
    idxmap = build_idxmap(sys)
    t = sol.t

    # Cell 1
    for (k, sym) in enumerate(vars1)
        if haskey(idxmap, sym)
            i = idxmap[sym]
            u = log10.(max.(sol[i,:], 1))
            u = sol[i,:]
            pts1[k][] = Point2f.(t, u)
        else
            # variable got eliminated by simplify -> show empty
            pts1[k][] = Point2f[]
        end
    end

    # Cell 2
    for (k, sym) in enumerate(vars2)
        if haskey(idxmap, sym)
            i = idxmap[sym]
            u = log10.(max.(sol[i, :], 1))
            u = sol[i,:]
            pts2[k][] = Point2f.(t, u)
        else
            pts2[k][] = Point2f[]
        end
    end

    autolimits!(ax1)
    autolimits!(ax2)
    return nothing
end

update_plot!(sol0, full0, vars1, vars2, pts1, pts2, ax1, ax2)

# --- Whenever sol changes -> reindex using CURRENT system (full_obs[]) ---
on(sol_obs) do sol
    update_plot!(sol, full_obs[], vars1, vars2, pts1, pts2, ax1, ax2)
end

function solve_now!()
    sol, sys = createODE(kern, base, p_obs[], u0_obs[])
    full_obs[] = sys          # must update system first (ordering may change!)
    sol_obs[]  = sol          # triggers plot update via on(sol_obs)
    return nothing
end



# ------------------------------------------------------------
# 2) Reaction SliderGrid (k1..k17) → löst nur solve aus
# ------------------------------------------------------------

logrange = -4.0:0.05:2.0  # 1e-4 .. 1e2
log10_start(x, r) = (x <= 0 ? first(r) : clamp(log10(x), first(r), last(r)))

lbl = Dict(
    :k1  => "k1:  CAR1 → ACA + CAR1",
    :k2  => "k2:  ACA + PKA → PKA",
    :k3  => "k3:  cAMPi → PKA + cAMPi",
    :k4  => "k4:  PKA → 0",
    :k5  => "k5:  CAR1 → ERK2 + CAR1",
    :k6  => "k6:  PKA + ERK2 → PKA",
    :k7  => "k7:  0 → RegA",
    :k8  => "k8:  ERK2 + RegA → ERK2",
    :k9  => "k9:  ACA → cAMPi + ACA",
    :k10 => "k10: RegA + cAMPi → RegA",
    :k11 => "k11: ACA → cAMPe + ACA",
    :k13 => "k13: cAMPe → CAR1 + cAMPe",
    :k14 => "k14: CAR1 → 0"
)

slider_row = fig[3, 1] = GridLayout()
# ---- LEFT: Reaction sliders ----
Label(slider_row[1, 1], "Reaction parameters", tellwidth=false)

sg_reac = SliderGrid(
    slider_row[2, 1],
    (label=lbl[:k1],  range=logrange, startvalue=log10_start(p0[1].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k2],  range=logrange, startvalue=log10_start(p0[2].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k3],  range=logrange, startvalue=log10_start(p0[3].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k4],  range=logrange, startvalue=log10_start(p0[4].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k5],  range=logrange, startvalue=log10_start(p0[5].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k6],  range=logrange, startvalue=log10_start(p0[6].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k7],  range=logrange, startvalue=log10_start(p0[7].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k8],  range=logrange, startvalue=log10_start(p0[8].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k9],  range=logrange, startvalue=log10_start(p0[9].second,  logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k10], range=logrange, startvalue=log10_start(p0[10].second, logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k11], range=logrange, startvalue=log10_start(p0[11].second, logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k13], range=logrange, startvalue=log10_start(p0[12].second, logrange), format=x->@sprintf("%.3g", 10.0^x)),
    (label=lbl[:k14], range=logrange, startvalue=log10_start(p0[13].second, logrange), format=x->@sprintf("%.3g", 10.0^x)),
    tellheight=true
)

colsize!(slider_row, 1, Relative(1))
rowsize!(slider_row, 2, Auto())

apply_ode_btn = Button(slider_row[3, 1], label="Apply ODE (recompute)", width=220, height=34)

s_reac = [s.value for s in sg_reac.sliders]

# Reaction-Params Observable (Pairs)
p_obs = @lift([
    :k1 => 10.0^($(s_reac[1])),
    :k2 => 10.0^($(s_reac[2])),
    :k3 => 10.0^($(s_reac[3])),
    :k4 => 10.0^($(s_reac[4])),
    :k5 => 10.0^($(s_reac[5])),
    :k6 => 10.0^($(s_reac[6])),
    :k7 => 10.0^($(s_reac[7])),
    :k8 => 10.0^($(s_reac[8])),
    :k9 => 10.0^($(s_reac[9])),
    :k10 => 10.0^($(s_reac[10])),
    :k11 => 10.0^($(s_reac[11])),
    :k13 => 10.0^($(s_reac[12])),
    :k14 => 10.0^($(s_reac[13])),
])


# ------------------------------------------------------------
# 4) IC Editor + Buttons (Apply ICs löst solve aus)
# ------------------------------------------------------------
right = fig[:, 2] = GridLayout()

Label(right[1, 1], "Initialbedingungen (je Variable)", tellwidth=false)

ic_syms = [pair.first for pair in u0_default]
ic_vals = Dict(pair.first => pair.second for pair in u0_default)

ic_grid = right[2, 1] = GridLayout()
ic_boxes = Dict{Symbol, Any}()

function float_or_nothing(s::AbstractString)
    try
        parse(Float64, strip(s))
        return true
    catch
        return false
    end
end

for (i, sym) in enumerate(ic_syms)
    Label(ic_grid[i, 1], string(sym), tellwidth=false)
    tb = Textbox(
        ic_grid[i, 2];
        stored_string = string(ic_vals[sym]),
        placeholder = "Float64",
        width = 140,
        height = 28,
        validator = float_or_nothing,
        defocus_on_submit = true,
    )
    ic_boxes[sym] = tb
end

btn_row = length(ic_syms) + 1
apply_ic_btn = Button(ic_grid[btn_row, 1], label="Apply ICs", width=120, height=32)
reset_ic_btn = Button(ic_grid[btn_row, 2], label="Reset", width=120, height=32)

function textbox_value(tb)
    # manche Makie-Versionen haben displayed_string, manche nicht
    if hasproperty(tb, :displayed_string)
        return tb.displayed_string[]
    else
        return tb.stored_string[]
    end
end

function read_u0_from_boxes()
    new_u0 = Pair{Symbol, Float64}[]
    for sym in ic_syms
        s = textbox_value(ic_boxes[sym])
        val = parse(Float64, strip(s))
        push!(new_u0, sym => val)
    end
    return new_u0
end


on(apply_ode_btn.clicks) do _
    solve_now!()
end

# IC Apply -> update u0_obs -> solve
on(apply_ic_btn.clicks) do _
   try
        newpairs = read_u0_from_boxes()
        u0_obs[] = newpairs
        solve_now!()
    catch err
        @warn "Konnte Initialbedingungen nicht parsen. Prüfe Eingaben." exception=(err, catch_backtrace())
    end
end

on(reset_ic_btn.clicks) do _
    u0_obs[] = copy(u0_default)
    for (sym, val) in u0_default
        ic_boxes[sym].stored_string[] = string(val)
    end
    solve_now!()
end

colsize!(fig.layout, 1, Relative(0.72))
colsize!(fig.layout, 2, Relative(0.28))

screen = GLMakie.Screen(resolution = (1700, 950), scalefactor = 1.0)

screen2 = display(screen, fig)

wait(screen2)

