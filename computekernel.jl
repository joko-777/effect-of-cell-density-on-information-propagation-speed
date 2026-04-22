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
using Printf
using DelayDiffEq
using LsqFit
using Dates
using ProgressBars
import CairoMakie
using JSON3


const TSPAN=(0.0, 4000)

# --------------------------
# 4) Diffusion / PDE-Kernel defaults
# --------------------------
# Geometry (NOT time-scaled)
const DIST_UM0   = 30.0   # µm
const DX0        = 2.0     # µm
const MARGIN_UM0 = 300.0   # µm

# Physical diffusion/decay parameters
const DC   = 450.0    # µm^2/s  (cAMP)
const DP   = 70.0     # µm^2/s  (PDE)
const MU_P = 2e-5     # 1/s     (PDE decay)
const MU_C = 0      # 1/s     (extra cAMP decay, optional)

# Source/coupling (physical units as you used them)
const K_BASAL = 3e-4  # PDE source strength (a.u./s)  (small)
const K0      = 0.03  # coupling (1/(a.u.*s))         (small)
const IMPULSE_MASS0 = 1.0   # a.u. (NOT time-scaled)

const DT_FACTOR=0.22

const MAX_TIME=4000
const MAX_STEADY_TIME=10000

const N_ERLANG_APPROX     = 8   # Erlang order n
const M_ERLANG_APPROX     = 7   # number of Erlang terms m
const N_SELF_EXP_APPROX   = 4   # number of exponentials for self-kernel

# μ has units 1/s physically -> in simulation it's 1/tu, so scale it.
const ERLANG_MU_MIN  = 1e-12

const MAX_PLOT_TIME=30
# -----------------------------
# 2D helper: Laplacian (Dirichlet boundaries u=0)
# -----------------------------

@inline function factorial_float(n::Int)
    n < 0 && throw(ArgumentError("n must be >= 0"))
    prod = 1.0
    @inbounds for k in 2:n
        prod *= k
    end
    return prod
end

 @inline function eval_exp_sum(t::AbstractVector, α::AbstractVector, μ::AbstractVector)
        y = zeros(Float64, length(t))
        @inbounds for j in eachindex(α)
            @. y += α[j] * exp(-μ[j] * t)
        end
        return y
end

@inline function eval_erlang_kernel(t::AbstractVector; A::Real, μ::Real, n::Int)
        # K(t)=A*(μ^n/(n-1)!)*t^(n-1)*exp(-μ t)
        fac = (μ^n) / factorial_float2(n - 1)
        y = similar(t, Float64)
        @. y = A * fac * (t^(n - 1)) * exp(-μ * t)   # @. macht daraus elementweise .^
        return y
    end


@inline function laplacian_dirichlet!(Lu::Matrix{Float64}, u::Matrix{Float64}, dx::Float64)
    Nx, Ny = size(u)
    invdx2 = 1.0 / (dx*dx)

    @inbounds for j in 2:Ny-1, i in 2:Nx-1
        Lu[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]) * invdx2
    end

    @inbounds begin
        for i in 1:Nx
            Lu[i,1]  = 0.0
            Lu[i,Ny] = 0.0
        end
        for j in 1:Ny
            Lu[1,j]  = 0.0
            Lu[Nx,j] = 0.0
        end
    end
    return Lu
end

@inline function idx_from_xy(x::Float64, y::Float64, dx::Float64, Nx::Int, Ny::Int)
    i = Int(round(x/dx)) + 1
    j = Int(round(y/dx)) + 1
    i = clamp(i, 1, Nx)
    j = clamp(j, 1, Ny)
    return i, j
end

@inline function add_point_source!(field::Matrix{Float64}, i::Int, j::Int, rate::Float64, dt::Float64)
    field[i,j] += dt * rate
end

# Konvergenzmaß
@inline function rel_change_norm(new::Matrix{Float64}, old::Matrix{Float64})
    Nx, Ny = size(new)
    num = 0.0
    den = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        a = new[i,j]
        b = old[i,j]
        num += abs(a - b)
        den += abs(b)
    end
    return num, den
end

# -----------------------------
# PDE: steady state for p(x) with two point sources
# -----------------------------
function solve_pde_steady_2d(Nx::Int, Ny::Int, dx::Float64,
                            ig::Int, jg::Int, ir::Int, jr::Int,
                            Dp::Float64, μp::Float64,
                            k_basal::Float64,
                            dt::Float64, tmax::Float64;
                            check_every::Int=50,
                            tol_rel::Float64=1e-8,
                            tol_abs::Float64=1e-12,
                            verbose::Bool=false)

    p = zeros(Float64, Nx, Ny)
    p_old = similar(p)
    Lp = zeros(Float64, Nx, Ny)

    steps = Int(ceil(tmax/dt))

    for n in 1:steps
        copyto!(p_old, p)
        laplacian_dirichlet!(Lp, p, dx)

        @inbounds for j in 2:Ny-1, i in 2:Nx-1
            p[i,j] += dt * (Dp*Lp[i,j] - μp*p[i,j])
            if p[i,j] < 0.0
                p[i,j] = 0.0
            end
        end

        # basal sources at generator and receiver
        add_point_source!(p, ig, jg, k_basal, dt)
        add_point_source!(p, ir, jr, k_basal, dt)

        if (n % check_every) == 0
            num, den = rel_change_norm(p, p_old)
            rel = num / max(den, tol_abs)
            abschg = num / (Nx*Ny)

            if verbose
                @printf("PDE step %d/%d  t=%.1f s  rel=%.3e  abs=%.3e\n",
                        n, steps, (n-1)*dt, rel, abschg)
            end

            if rel < tol_rel || abschg < tol_abs
                if verbose
                    @printf("PDE steady reached at t=%.1f s\n", (n-1)*dt)
                end
                break
            end
        end
    end

    return p
end

# -----------------------------
# Kernel: impulse response for c(x,t) given p_ss(x)
# returns:
#  - h(t): receiver response / impulse_mass
#  - u(t): self response / impulse_mass
# -----------------------------
function compute_camp_kernels_2d(p_ss::Matrix{Float64}, Nx::Int, Ny::Int, dx::Float64,
                                ig::Int, jg::Int, ir::Int, jr::Int,
                                Dc::Float64, k0::Float64, μc::Float64,
                                dt::Float64, tmax::Float64,
                                impulse_mass::Float64;
                                verbose::Bool=false)

    c = zeros(Float64, Nx, Ny)
    Lc = zeros(Float64, Nx, Ny)

    # impulse at generator cell at t=0
    c[ig, jg] += impulse_mass

    steps = Int(ceil(tmax/dt))
    times = collect(0.0:dt:(steps-1)*dt)

    h = zeros(Float64, steps)  # receiver
    u = zeros(Float64, steps)  # self

    if verbose
        @printf("Kernel compute: steps=%d dt=%.4g tmax=%.1f\n", steps, dt, tmax)
    end

    @inbounds for n in ProgressBar(1:steps)
        # measure first (consistent with your original script)
        h[n] = c[ir, jr] / impulse_mass
        u[n] = c[ig, jg] / impulse_mass

        # advance one step
        laplacian_dirichlet!(Lc, c, dx)

        @inbounds for j in 2:Ny-1, i in 2:Nx-1
            sink = (k0 * p_ss[i,j] + μc)
            c[i,j] += dt * (Dc*Lc[i,j] - sink*c[i,j])
            if c[i,j] < 0.0
                c[i,j] = 0.0
            end
        end
        # boundaries remain 0 (Dirichlet) by construction
    end

    return times, h, u
end

function fit_sum_of_exponentials(t::AbstractVector,
                                K::AbstractVector,
                                n::Int)

    @assert length(t) == length(K)

    # ---- Initial guess (VERY important for exponentials) ----
    # mu roughly spread across timescale
    tspan = maximum(t) - minimum(t)
    μ0 = range(1/tspan, stop=10/tspan, length=n)

    # split amplitude roughly
    α0 = fill(maximum(K)/n, n)

    p0 = vcat(α0, μ0)

    # ---- model ----
    model(t, p) = begin
        α = @view p[1:n]
        μ = @view p[n+1:2n]

        y = zero(t)
        s = similar(t)

        fill!(s, 0.0)

        @inbounds for i in 1:n
            @. s += α[i] * exp(-μ[i] * t)
        end
        return s
    end

    # ---- enforce positivity (important physically) ----
    lower = vcat(fill(0.0, n), fill(0.0, n))
    upper = fill(Inf, 2n)

    fit = curve_fit(model, t, K, p0; lower=lower, upper=upper)

    p = coef(fit)

    α = p[1:n]
    μ = p[n+1:2n]

    return α, μ, fit
end


"""
fit_erlang_sum(t, K; n, m, μ0=nothing, A0=nothing, μmin=1e-8, drop_t0=true)

Fits:
    K(t) ≈ Σ_{j=1..m} A_j * (μ_j^n/(n-1)!) * t^(n-1) * exp(-μ_j t)

Constraints:
    A_j >= 0, μ_j >= μmin (enforced via μ_j = exp(θ_j) and θ_j >= log(μmin))

Returns:
    A::Vector{Float64}, μ::Vector{Float64}, fit::LsqFitResult, Khat::Vector{Float64}
"""
function fit_erlang_sum(t::AbstractVector, K::AbstractVector,
                        n::Int, m::Int;
                        μ0=nothing, A0=nothing,
                        μmin::Real=1e-8,
                        drop_t0::Bool=true)

    n >= 1 || throw(ArgumentError("n must be >= 1"))
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    length(t) == length(K) || throw(ArgumentError("t and K must have same length"))

    tt = Float64.(t)
    KK = Float64.(K)

    # valid points
    idx = isfinite.(tt) .& isfinite.(KK) .& (tt .>= 0)
    if drop_t0
        idx .&= (tt .> 0)
    end
    idx .&= (KK .> 0)   # ignore zeros/negatives in fit (optional)

    tfit = tt[idx]
    yfit = KK[idx]
    isempty(tfit) && throw(ArgumentError("No valid data points to fit (need t>0 and K>0)."))

    # ---- initial guesses ----
    # Guess μs spread on a log scale over the time span
    if μ0 === nothing
        tmin = minimum(tfit)
        tmax = maximum(tfit)
        tspan = max(tmax - tmin, 1e-12)

        # characteristic rates ~ 1/tspan .. 1/tmin (rough)
        μlo = max(1 / tspan, Float64(μmin) * 10)
        μhi = max(1 / max(tmin, 1e-6), μlo * 10)

        # log-spaced μ
        θlo = log(μlo)
        θhi = log(μhi)
        θ0 = collect(range(θlo, θhi; length=m))
    else
        μ0v = Float64.(μ0)
        length(μ0v) == m || throw(ArgumentError("μ0 must have length m"))
        θ0 = log.(max.(μ0v, Float64(μmin)))
    end

    # Guess amplitudes roughly by splitting max(K)
    if A0 === nothing
        A0v = fill(maximum(yfit) / m, m)
    else
        A0v = Float64.(A0)
        length(A0v) == m || throw(ArgumentError("A0 must have length m"))
        A0v .= max.(A0v, 0.0)
    end

    # parameter vector p = [A1..Am, θ1..θm]
    p0 = vcat(A0v, θ0)

    fac_den = factorial_float(n - 1)

    # model evaluation
    function model(tvec, p)
        A = @view p[1:m]
        θ = @view p[m+1:2m]
        μ = exp.(θ)

        # common factor t^(n-1)
        tn1 = tvec .^ (n - 1)

        y = zeros(Float64, length(tvec))
        @inbounds for j in 1:m
            fac = (μ[j]^n) / fac_den
            @. y += A[j] * fac * tn1 * exp(-μ[j] * tvec)
        end
        return y
    end

    # Bounds: A >= 0, θ >= log(μmin)
    lower = vcat(fill(0.0, m), fill(log(Float64(μmin)), m))
    upper = fill(Inf, 2m)

    fit = curve_fit(model, tfit, yfit, p0; lower=lower, upper=upper)
    p = coef(fit)

    A = p[1:m]
    μ = exp.(p[m+1:2m])

    # fitted curve aligned to original t
    Khat = fill(NaN, length(tt))
    Khat[idx] .= model(tfit, p)

    return A, μ, fit, Khat
end

# -----------------------------
# Public API: compute h(t), u(t) from (dist, k_basal, k0)
# -----------------------------
"""
Berechnet 2D-Impulseantwort-Kernel für cAMPe:

1) PDE-Steady State p_ss(x) aus Diffusion + Abbau + 2 Punktquellen (beide Zellen mit Stärke k_basal)
2) cAMP-Impuls bei Generator → misst:
   - h(t): c am Receiver / impulse_mass
   - u(t): c am Generator / impulse_mass (Self-Response)

Abhängigkeiten:
- dist_um: Zellabstand
- k_basal: PDE-Quellenstärke an den Zellen
- k0: Reaktionsrate PDE + cAMPe -> PDE (cAMP sink k0 * p * c)
"""
function camp_kernels_2d(;dist_um::Real = DIST_UM0,
                        k_basal::Real = K_BASAL,
                        k0::Real = K0,
                        Dc::Real = DC,
                        Dp::Real = DP,
                        μp::Real = MU_P,
                        μc::Real = MU_C,
                        margin_um::Real = MARGIN_UM0,
                        dx::Real = DX0,
                        dt_factor::Real = DT_FACTOR,
                        t_pde_max::Real = MAX_STEADY_TIME,
                        t_kernel_max::Real = MAX_TIME,
                        impulse_mass::Real = IMPULSE_MASS0,
                        verbose::Bool = false)

    dist_um = float(dist_um)
    margin_um = float(margin_um)
    dx = float(dx)

    # Domain
    Lx = dist_um + 2*margin_um
    Ly = 2*margin_um
    Nx = Int(round(Lx/dx)) + 1
    Ny = Int(round(Ly/dx)) + 1

    # place points centered in y
    xg = margin_um
    yg = Ly/2
    xr = margin_um + dist_um
    yr = Ly/2

    ig, jg = idx_from_xy(xg, yg, dx, Nx, Ny)
    ir, jr = idx_from_xy(xr, yr, dx, Nx, Ny)

    # Stable dt for 2D FTCS: dt <= dx^2/(4*Dmax)
    Dmax = max(float(Dc), float(Dp))
    dt = dt_factor * dx^2 / (4*Dmax)

    if verbose
        @printf("Domain: Nx=%d Ny=%d dx=%.3g Lx=%.1f Ly=%.1f dist=%.1f\n", Nx, Ny, dx, Lx, Ly, dist_um)
        @printf("idx gen=(%d,%d) rec=(%d,%d) dt=%.4g\n", ig, jg, ir, jr, dt)
    end

    # 1) PDE steady
    p_ss = solve_pde_steady_2d(Nx, Ny, dx,
        ig, jg, ir, jr,
        float(Dp), float(μp),
        float(k_basal),
        dt, float(t_pde_max);
        verbose=verbose
    )

    # 2) kernels
    times, h, u = compute_camp_kernels_2d(p_ss, Nx, Ny, dx,
        ig, jg, ir, jr,
        float(Dc), float(k0), float(μc),
        dt, float(t_kernel_max), float(impulse_mass);
        verbose=verbose
    )

    # cross: Erlang fit (order nE)
    nE = N_ERLANG_APPROX
    mE = M_ERLANG_APPROX
    println("Fitt cross...")
    A_erlang, μ_erlang, fitH, hhat = fit_erlang_sum(times, h, nE, mE; μmin=ERLANG_MU_MIN)

    # self: sum of exponentials
    println("Fit self...")
    αSelf, μSelf, fitU = fit_sum_of_exponentials(times, u, N_SELF_EXP_APPROX)

    @info "cross erlang fit" nE A_erlang μ_erlang
    @info "self  fit" αSelf μSelf

    return (
        times = times,
        h = h,
        u = u,
        dt = dt, dx = dx, Nx = Nx, Ny = Ny,
        dist_um = dist_um, margin_um = margin_um,
        Dc = float(Dc), Dp = float(Dp),
        μp = float(μp), μc = float(μc),
        k0 = float(k0), k_basal = float(k_basal),

        # --- cross (Erlang kernel) ---
        n_cross = nE,
        A_cross = A_erlang,
        μ_cross = μ_erlang,

        # optional diagnostics / fitted curve
        fitH = fitH,
        h_fit = hhat,

        # --- self (sum of exponentials) ---
        αSelf = αSelf,
        μSelf = μSelf,
        fitU = fitU,
    )
end

#--------------- PLOTTING --------------
function update_kernel_plot_and_save!(kern; overwrite=true, logplot=false, basename="kernel", onlycrossplot=true)
        try
            figK = CairoMakie.Figure(size=(900, 450))
            axK  = CairoMakie.Axis(figK[1, 1],
                xlabel="t [s]",
                ylabel=(logplot ? "log(kernel)" : "kernel"),
                title="Kernel (PDE + Fits)"
            )

            t = kern.times
            h = kern.h
            u = kern.u

            # --- self fit: Σexp ---/DT
            u_fit = (hasproperty(kern, :αSelf) && hasproperty(kern, :μSelf)) ?
                eval_exp_sum(t, kern.αSelf, kern.μSelf) : nothing

            # --- cross fit: Erlang ---
            h_fit = nothing
            if hasproperty(kern, :h_fit)
                h_fit = kern.h_fit
            elseif hasproperty(kern, :A_cross) && hasproperty(kern, :μ_cross) && hasproperty(kern, :n_cross)
                h_fit = eval_erlang_kernel(t; A=kern.A_cross, μ=kern.μ_cross, n=Int(kern.n_cross))
            end

            if logplot
                CairoMakie.lines!(axK, t, log.(clamp.(h, 1e-300, Inf)), label="h (cross) PDE")
                #CairoMakie.lines!(axK, t, log.(clamp.(u, 1e-300, Inf)), label="u (self) PDE")

                if h_fit !== nothing
                    CairoMakie.lines!(axK, t, log.(clamp.(h_fit, 1e-300, Inf)),
                                      linestyle=:dash, label="h_fit (cross) Erlang")
                end
                if u_fit !== nothing && !onlycrossplot
                    CairoMakie.lines!(axK, t, log.(clamp.(u_fit, 1e-300, Inf)),
                                      linestyle=:dash, label="u_fit (self) Σexp")
                end
            else
                CairoMakie.lines!(axK, t, h, label="h (cross) PDE")
               # CairoMakie.lines!(axK, t, u, label="u (self) PDE")

                if h_fit !== nothing
                    CairoMakie.lines!(axK, t, h_fit, linestyle=:dash, label="h_fit (cross) Erlang")
                end
                if u_fit !== nothing && !onlycrossplot
                    CairoMakie.lines!(axK, t, u_fit, linestyle=:dash, label="u_fit (self) Σexp")
                end
            end

            CairoMakie.axislegend(axK, position=:rt)
            if logplot
                CairoMakie.ylims!(axK, -30, 0)
                CairoMakie.xlims!(axK, 0, MAX_PLOT_TIME)
            else
                #CairoMakie.autolimits!(axK)
                CairoMakie.xlims!(axK, 0, MAX_PLOT_TIME)
            end

            filename = overwrite ? "$(basename).png" :
                "$(basename)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")).png"

            CairoMakie.save(filename, figK)
            @info "Kernel plot saved" file=joinpath(pwd(), filename)

        catch err
            @error "update_kernel_plot_and_save! failed" exception=(err, catch_backtrace())
        end
        return nothing
    end



"""
Save parameters to JSON (best for reloading).
Works with NamedTuple/struct; stores a plain Dict so it round-trips cleanly.
"""
function save_fit_json(filename::AbstractString, kern)
    d = Dict(
        "n_cross"   => getproperty(kern, :n_cross),
        "A_cross"   => getproperty(kern, :A_cross),
        "mu_cross"  => getproperty(kern, :μ_cross),

        "alpha_self"=> getproperty(kern, :αSelf),
        "mu_self"   => getproperty(kern, :μSelf),

    )

    JSON3.write(filename, d)
    return filename
end

kern = camp_kernels_2d();

update_kernel_plot_and_save!(kern)
update_kernel_plot_and_save!(kern; basename="kernelBoth", onlycrossplot=false)

save_fit_json("fitparams.json", kern)
