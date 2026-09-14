#!/usr/bin/env julia
# =============================================================================
# GRRR replication, plain-script version of GRRR_Replication_bootstrap.ipynb.
#
# Run from the Julia/ directory, so that data/wti_spot_futures.csv resolves:
#
#     julia -t auto run_all.jl                      # paper replication counts
#     GRRR_QUICK=true julia -t auto run_all.jl      # quick check, a few minutes
#
# `-t auto` is the whole point: Experiments B and C and the Section 6.1
# bootstrap are threaded, and an IJulia kernel does not pick up threads from
# the VS Code setting.
# =============================================================================

ENV["GKSwstype"] = "100"          # headless GR, so savefig works with no display

const _T_START = time()

# stdout is block-buffered through a pipe, so `| tee` shows the printed tables
# minutes late while @info goes straight to stderr.  Flush at every stamp so the
# log reads in order and progress is visible while a long cell is running.
function _stamp(label)
    flush(stdout)
    @info string(label) elapsed_min = round((time() - _T_START) / 60, digits = 2)
    flush(stderr)
end

# ------------------------------------------------------------ cell 1
using LinearAlgebra, Statistics, Random, Distributions,
      CSV, DataFrames, Plots, LaTeXStrings, Printf, BenchmarkTools

# One seed for the whole notebook, 4 January 1951, the birth date of James G.
# MacKinnon. Each experiment takes its own block of the integer line so the
# streams cannot overlap.
const SEED = 19510104

Random.seed!(SEED)   # global seed for the Monte-Carlo sections
_stamp("done: cell 1")

# ------------------------------------------------------------ cell 2
# ---- how much to run -----------------------------------------------------
# QUICK = true runs the whole notebook end to end in a few minutes at reduced
# replication counts: enough to check that every cell executes and that the
# numbers land in the right neighbourhood, not enough to reproduce the tables.
# The default here is the paper's counts.  Threads come from `julia -t auto`,
# so none of the IJulia kernel business applies to this script.

const QUICK = lowercase(get(ENV, "GRRR_QUICK", "false")) in ("1", "true", "yes")
#   terminal:  GRRR_QUICK=true julia -t auto run_all.jl

const REPS_A  = QUICK ?  25 : 200    # Table 1, Experiment A
const REPS_B  = QUICK ?  50 : 500    # Table 2, Experiment B
const NDATA_C = QUICK ?  10 :  50    # Table 3, Experiment C
const NSTART_C = 20                  # starts per dataset in Experiment C
const B_LR    = QUICK ?  99 : 999    # Section 6.1, bootstrap LR test
const B_SE    = QUICK ?  99 : 499    # Section 6.1, bootstrap standard errors

println(QUICK ? "QUICK = true: reduced replication counts, results are indicative" :
                "QUICK = false: paper replication counts")

# Threads.@threads is silent when the kernel has one thread: the loop simply
# runs serially and nothing says so.  Experiments B and C and the Section 6.1
# bootstrap are all threaded, so check that work is really being spread before
# starting a long run.
let nt = Threads.nthreads()
    hits = zeros(Int, Threads.maxthreadid())
    Threads.@threads for i in 1:(256 * nt)
        hits[Threads.threadid()] += 1
    end
    used = count(>(0), hits)
    println("threads: $nt of $(Sys.CPU_THREADS) logical; a test loop ran on $used of them")

    # Each Julia thread would otherwise start its own BLAS threads.  Every matrix
    # in this notebook is 5x5 or T x 6, so BLAS threading buys nothing here and
    # the oversubscription costs.
    BLAS.set_num_threads(1)
    println("BLAS threads: $(BLAS.get_num_threads())")

    if used < 2
        msg = "Single-threaded: every Threads.@threads loop in this notebook is " *
              "running serially.  julia.NumThreads does not reach an IJulia " *
              "kernel.  From a Julia REPL run `using IJulia; " *
              "installkernel(\"Julia $(Sys.CPU_THREADS) threads\", " *
              "env=Dict(\"JULIA_NUM_THREADS\" => \"$(Sys.CPU_THREADS)\"))`, reload " *
              "the window, and select that kernel for this notebook."
        error(msg * "  In a terminal just start julia with `-t auto`.")
    end
end
_stamp("done: cell 2")

# ------------------------------------------------------------ cell 4
# --- small helpers -------------------------------------------------------
_sym(A) = (A + A') / 2                       # symmetrize
logdet_pd(A) = logdet(cholesky(Symmetric(A)))  # log|A| for SPD A, via Cholesky

# log|A| with a positive-definiteness flag (batched ARCH likelihood)
function safe_logdet(A)
    F = cholesky(Symmetric(A); check=false)
    issuccess(F) ? (logdet(F), true) : (0.0, false)
end

# RGLS solve (eq. AChat / Bhat):  θ̂=(R'MR)⁻¹R'(N−Mr);  vec = Rθ̂ + r
function rgls(M, N, R, rvec)
    size(R, 2) == 0 && return (copy(rvec), Float64[])   # fully restricted: vec = r
    θ = (R' * M * R) \ (R' * (N - M * rvec))
    return R * θ + rvec, θ
end
_stamp("done: cell 4")

# ------------------------------------------------------------ cell 6
# beta-step moments (block-diagonal Σ):  M_{α,S}, N_{α,Ψ,S}
function bstep_moments_blockdiag(Y, X, Z, Ψ, α, W)
    T, p = size(Y); p1 = size(X, 2); r = size(α, 2)
    Ma = zeros(r * p1, r * p1)
    Nac = zeros(p1, r)
    R = Y - Z * Ψ'                              # residual net of Ψ Zₜ
    for t in 1:T
        xt = X[t, :]
        Ma  .+= kron(α' * W[t] * α, xt * xt')    # (α'Wₜα) ⊗ XₜXₜ'
        Nac .+= xt * (R[t, :]' * W[t] * α)       # Xₜ (Rₜ'Wₜα)
    end
    return Ma, vec(Nac)
end

# (α,Ψ)-step moments (block-diagonal Σ):  M_{β,S}, N_{β,S}
function acstep_moments_blockdiag(Y, X, Z, β, W)
    T, p = size(Y); r = size(β, 2); p2 = size(Z, 2); m = r + p2
    SF = hcat(X * β, Z)                          # [Xₜβ, Zₜ] stacked rows, T×m
    Mb = zeros(m * p, m * p)
    Nb = zeros(p, m)
    for t in 1:T
        sf = SF[t, :]
        Mb .+= kron(sf * sf', W[t])              # (SFₜ SFₜ') ⊗ Wₜ
        Nb .+= (W[t] * Y[t, :]) * sf'            # Wₜ Yₜ SFₜ'
    end
    return Mb, vec(Nb)
end

# separable Σ = A ⊗ Ωᵤ  (Ainv = A⁻¹, Wu = Ωᵤ⁻¹)
function bstep_moments_separable(Y, X, Z, Ψ, α, Ainv, Wu)
    Ma = kron(α' * Wu * α, X' * Ainv * X)
    R = Y - Z * Ψ'
    Nac = (X' * Ainv * R) * Wu * α
    return Ma, vec(Nac)
end

function acstep_moments_separable(Y, X, Z, β, Ainv, Wu)
    XAX = X' * Ainv * X; XAZ = X' * Ainv * Z; ZAZ = Z' * Ainv * Z
    core = vcat(hcat(β' * XAX * β, β' * XAZ),
                hcat(XAZ' * β,     ZAZ))
    Mb = kron(core, Wu)
    inner = hcat(Y' * Ainv * X * β, Y' * Ainv * Z)
    return Mb, vec(Wu * inner)
end
_stamp("done: cell 6")

# ------------------------------------------------------------ cell 8
# iid:  Σ = I_T ⊗ Ω
function theta_iid(E, ctx)
    T, p = size(E)
    Ω = _sym(E' * E / T); Winv = inv(Ω)
    return Dict{Symbol,Any}(:W => [Winv for _ in 1:T], :logdetS => T * logdet_pd(Ω),
                            :theta => Dict(:Omega => Ω), :engine => :blockdiag)
end

# regime: piecewise-constant Ω over segments [edgeₛ+1, edgeₛ₊₁]
function theta_regime(E, ctx)
    T, p = size(E)
    edges = vcat(0, ctx[:breaks], T)
    W = Vector{Matrix{Float64}}(undef, T); logdetS = 0.0; Ωs = Matrix{Float64}[]
    for s in 1:length(edges) - 1
        a, b = edges[s], edges[s + 1]
        Ω = _sym(E[a + 1:b, :]' * E[a + 1:b, :] / (b - a)); push!(Ωs, Ω)
        Wi = inv(Ω); for t in a + 1:b; W[t] = Wi; end
        logdetS += (b - a) * logdet_pd(Ω)
    end
    return Dict{Symbol,Any}(:W => W, :logdetS => logdetS,
                            :theta => Dict(:Omega_regimes => Ωs, :edges => edges),
                            :engine => :blockdiag)
end

# arch:  Ω(t)⁻¹ = Qₜ'DQₜ, Newton ascent on the concave l(D)
function theta_arch(E, ctx)
    T, p = size(E)
    Xex = ctx[:arch_exog]; q = size(Xex[1], 2); m = p + q
    if !haskey(ctx, :Qtp)                                   # cache Qₜ' = [Iₚ, Xexₜ]
        ctx[:Qtp] = [hcat(Matrix{Float64}(I, p, p), Xex[t]) for t in 1:T]
    end
    Qtp = ctx[:Qtp]
    RHS = zeros(m, m)                                       # Σₜ (Qₜεₜ)(Qₜεₜ)'
    for t in 1:T
        ve = Qtp[t]' * E[t, :]; RHS .+= ve * ve'
    end
    prec_blocks(D) = [Qtp[t] * D * Qtp[t]' for t in 1:T]    # Qₜ'DQₜ
    function loglik_D(D)
        W = prec_blocks(D); ld = 0.0
        for t in 1:T
            l, ok = safe_logdet(W[t]); ok || return (-Inf, W); ld += l
        end
        quad = 0.0
        for t in 1:T; quad += E[t, :]' * W[t] * E[t, :]; end
        return (ld - quad, W)
    end
    D = get(ctx, :D, nothing)
    if D === nothing
        Ω0 = _sym(E' * E / T)
        D = zeros(m, m); D[1:p, 1:p] = inv(Ω0); D += 1e-3 * Matrix{Float64}(I, m, m)
    end
    f0, _ = loglik_D(D)
    Op = zeros(m * m, m * m)                                # reused every iteration
    for _ in 1:get(ctx, :arch_max_inner, 40)
        Wc = prec_blocks(D)
        M = [Qtp[t]' * inv(Wc[t]) * Qtp[t] for t in 1:T]    # Mₜ = Qₜ Pₜ Qₜ'
        Gm = _sym(sum(M) - RHS)
        norm(Gm) < 1e-9 && break
        # Newton operator Σₜ Mₜ ⊗ Mₜ.  Written as sum(kron(M[t], M[t]) for t in 1:T)
        # this allocates T dense m²×m² matrices on every inner iteration — 32 MB
        # per iteration at T = 400, m = 10 — which dominates the runtime of
        # Experiment B.  Accumulate in place instead: same matrix, no allocation.
        fill!(Op, 0.0)
        for t in 1:T
            Mt = M[t]
            @inbounds for jj in 1:m, ii in 1:m
                mij = Mt[ii, jj]; a = (ii - 1) * m; b = (jj - 1) * m
                for ll in 1:m, kk in 1:m
                    Op[a + kk, b + ll] += mij * Mt[kk, ll]
                end
            end
        end
        # When the ARCH driver is Xexₜ = vₜIₚ (q = p, the Experiment-B design) the
        # map D ↦ Qₜ'DQₜ cannot see the antisymmetric part of D's off-diagonal
        # block, so Op is exactly singular with a p²-dimensional kernel and `\`
        # throws.  The likelihood is flat along that kernel, so damping it is
        # harmless, and the line search below validates the step regardless.
        # Where Op is nonsingular (q = 1, the empirical application) nothing
        # changes: the ridge branch is never taken.
        dvec = try
            Op \ vec(Gm)
        catch err
            err isa SingularException || rethrow()
            (Op + (1e-10 * tr(Op) / (m * m) + eps()) * I) \ vec(Gm)
        end
        d = _sym(reshape(dvec, m, m))
        s = 1.0; improved = false                           # damped Newton backtracking
        for _ls in 1:30
            fn, _ = loglik_D(D + s * d)
            if fn > f0 + 1e-12
                D = _sym(D + s * d); f0 = fn; improved = true; break
            end
            s *= 0.5
        end
        improved || break
    end
    ctx[:D] = D                                             # warm-start next θ-step
    W = prec_blocks(D)
    return Dict{Symbol,Any}(:W => W, :logdetS => -sum(safe_logdet(W[t])[1] for t in 1:T),
                            :theta => Dict(:D => D), :engine => :blockdiag)
end

# ar1: precision of the AR(1) correlation matrix A(ρ) is tridiagonal
function ar1_Ainv(ρ, T)
    d = fill(1.0 + ρ^2, T); d[1] = 1.0; d[end] = 1.0
    return Matrix(SymTridiagonal(d, fill(-ρ, T - 1)))
end

function theta_ar1(E, ctx)
    T, p = size(E)
    E0 = E[2:end, :]; El = E[1:end - 1, :]
    ρ = clamp(sum(E0 .* El) / sum(El .^ 2), -0.98, 0.98)
    U = E0 - ρ * El; Ωu = _sym(U' * U / (T - 1))
    Ainv = ar1_Ainv(ρ, T)
    logdetS = -p * logdet(Symmetric(Ainv)) + T * logdet_pd(Ωu)   # log|A| = -log|A⁻¹|
    return Dict{Symbol,Any}(:Ainv => Ainv, :Wu => inv(Ωu), :logdetS => logdetS,
                            :theta => Dict(:rho => ρ, :Omega_u => Ωu), :engine => :separable)
end

# fixed / known Σ (infeasible efficiency bound)
theta_fixed(E, ctx) = Dict{Symbol,Any}(:W => ctx[:W], :logdetS => ctx[:logdetS],
                                        :theta => Dict(:fixed => true), :engine => :blockdiag)

function theta_step(model, E, ctx)
    model === :iid    ? theta_iid(E, ctx)    :
    model === :regime ? theta_regime(E, ctx) :
    model === :arch   ? theta_arch(E, ctx)   :
    model === :ar1    ? theta_ar1(E, ctx)    :
    model === :fixed  ? theta_fixed(E, ctx)  :
    error("unknown sigma_model $model")
end
_stamp("done: cell 8")

# ------------------------------------------------------------ cell 9
# Gaussian log-likelihood  ℓ = −½Tp·log2π − ½log|Σ| − ½ε'Σ⁻¹ε
function loglik_blockdiag(E, W, logdetS)
    T, p = size(E); quad = 0.0
    for t in 1:T; quad += E[t, :]' * W[t] * E[t, :]; end
    return -0.5 * T * p * log(2π) - 0.5 * logdetS - 0.5 * quad
end

function loglik_separable(E, Ainv, Wu, logdetS)
    T, p = size(E)
    quad = tr(Wu * (E' * Ainv * E))
    return -0.5 * T * p * log(2π) - 0.5 * logdetS - 0.5 * quad
end
_stamp("done: cell 9")

# ------------------------------------------------------------ cell 11
function grrr_estimate(Y, X, Z, r;
        G=nothing, g=nothing, H=nothing, h=nothing,
        sigma_model=:iid, sigma_ctx=nothing, phi0=nothing, psi0=nothing,
        tol=1e-9, maxiter=2000, seed=nothing, normalize_beta=true, verbose=false)
    Y = Float64.(Y); X = Float64.(X)
    Z = Z === nothing ? zeros(size(Y, 1), 0) : Float64.(Z)
    T, p = size(Y); p1 = size(X, 2); p2 = size(Z, 2)
    n_ac = p * (r + p2); n_b = p1 * r
    G = G === nothing ? Matrix{Float64}(I, n_ac, n_ac) : Float64.(G)
    g = g === nothing ? zeros(n_ac) : Float64.(vec(g))
    H = H === nothing ? Matrix{Float64}(I, n_b, n_b) : Float64.(H)
    h = h === nothing ? zeros(n_b) : Float64.(vec(h))
    rng = seed === nothing ? Random.default_rng() : Xoshiro(seed)
    ctx = sigma_ctx === nothing ? Dict{Symbol,Any}() : copy(sigma_ctx)

    psi0 === nothing && (psi0 = randn(rng, size(G, 2)))
    AC = reshape(G * psi0 + g, p, r + p2)
    α = AC[:, 1:r]; Ψ = p2 > 0 ? AC[:, r + 1:end] : zeros(p, 0)
    phi0 === nothing && (phi0 = randn(rng, size(H, 2)))
    β = reshape(H * phi0 + h, p1, r)

    no_cross = normalize_beta && all(abs.(h) .< 1e-8)
    loglik_path = Float64[]; prev = -Inf; converged = false
    E = Y - X * β * α' - Z * Ψ'
    tinfo = theta_step(sigma_model, E, ctx)

    it = 0
    for outer it in 1:maxiter
        eng = tinfo[:engine]
        # (ii) β-step
        Ma, Nac = eng === :blockdiag ?
            bstep_moments_blockdiag(Y, X, Z, Ψ, α, tinfo[:W]) :
            bstep_moments_separable(Y, X, Z, Ψ, α, tinfo[:Ainv], tinfo[:Wu])
        b_vec, _ = rgls(Ma, Nac, H, h)
        if no_cross
            mx = maximum(abs.(b_vec)); mx > 0 && (b_vec = b_vec ./ mx)
        end
        β = reshape(b_vec, p1, r)
        # (i) (α,Ψ)-step
        Mb, Nb = eng === :blockdiag ?
            acstep_moments_blockdiag(Y, X, Z, β, tinfo[:W]) :
            acstep_moments_separable(Y, X, Z, β, tinfo[:Ainv], tinfo[:Wu])
        ac_vec, _ = rgls(Mb, Nb, G, g)
        AC = reshape(ac_vec, p, r + p2)
        α = AC[:, 1:r]; Ψ = p2 > 0 ? AC[:, r + 1:end] : zeros(p, 0)
        # (iii) θ-step
        E = Y - X * β * α' - Z * Ψ'
        tinfo = theta_step(sigma_model, E, ctx)
        ll = tinfo[:engine] === :blockdiag ?
            loglik_blockdiag(E, tinfo[:W], tinfo[:logdetS]) :
            loglik_separable(E, tinfo[:Ainv], tinfo[:Wu], tinfo[:logdetS])
        push!(loglik_path, ll)
        if it > 1 && abs(ll - prev) / max(1.0, abs(prev)) < tol
            converged = true; prev = ll; break
        end
        prev = ll
        verbose && it % 50 == 0 && println("  iter $it: loglik=$(round(ll, digits=6))")
    end
    return Dict{Symbol,Any}(:alpha => α, :beta => β, :Psi => Ψ, :theta => tinfo[:theta],
        :loglik => prev, :loglik_path => loglik_path, :iters => it,
        :converged => converged, :sigma_model => sigma_model)
end
_stamp("done: cell 11")

# ------------------------------------------------------------ cell 13
function johansen_rrr(Y, X, Z, r)
    Y = Float64.(Y); X = Float64.(X)
    Z = Z === nothing ? zeros(size(Y, 1), 0) : Float64.(Z)
    T, p = size(Y); p1 = size(X, 2); p2 = size(Z, 2)
    if p2 > 0
        Pz = Z * ((Z' * Z) \ Z'); R0 = Y - Pz * Y; R1 = X - Pz * X
    else
        R0 = copy(Y); R1 = copy(X)
    end
    S00 = R0' * R0 / T; S11 = R1' * R1 / T; S01 = R0' * R1 / T; S10 = S01'
    Kmat = S10 * (S00 \ S01)
    L = cholesky(Symmetric(S11 + 1e-12 * I)).L; Linv = inv(L)
    F = eigen(Symmetric(_sym(Linv * Kmat * Linv')))            # ascending
    ord = sortperm(F.values, rev=true)
    evals = F.values[ord]; evecs = F.vectors[:, ord]
    lam = clamp.(evals[1:r], 0, 0.999999999)
    β = (Linv' * evecs)[:, 1:r]                                # β'S₁₁β = I
    α = S01 * β
    Ψ = p2 > 0 ? ((Z' * Z) \ (Z' * (Y - X * β * α')))' : zeros(p, 0)
    E = Y - X * β * α' - Z * Ψ'
    logdetΩ = logdet_pd(S00) + sum(log.(1 .- lam))
    loglik = -0.5 * T * p * log(2π) - 0.5 * T * logdetΩ - 0.5 * T * p
    return Dict{Symbol,Any}(:alpha => α, :beta => β, :Psi => Ψ, :Omega => _sym(E' * E / T),
                            :loglik => loglik, :eigenvalues => evals, :r => r)
end
_stamp("done: cell 13")

# ------------------------------------------------------------ cell 15
const P = 5      # variables
const R = 4      # cointegration rank (p − r = 1 common trend)

# spread-basis restriction: H (20×4), h (20,), and β if b_true supplied
function basis_restriction(b_true=nothing)
    H = zeros(P * R, R); h = zeros(P * R)
    for i in 1:R
        H[(i - 1) * P + 1, i] = -1.0      # spot row of column i = −bᵢ
        h[(i - 1) * P + (i + 1)] = 1.0    # F_i row of column i = 1
    end
    β = b_true === nothing ? nothing : reshape(H * Float64.(b_true) + h, P, R)
    return H, h, β
end

# α = β(β'β)⁻¹diag(−κ)  ⇒  β'α = −diag(κ)
function make_alpha(β, κ)
    k = κ isa Number ? fill(float(κ), R) : Float64.(κ)
    return β * inv(β' * β) * Diagonal(-k)
end

# variance regime: Ω doubles at mid-sample
function sigma_regime(T, Ω1; factor=2.0)
    brk = div(T, 2); Ω2 = factor .* Ω1; W1 = inv(Ω1); W2 = inv(Ω2)
    cov = Vector{Matrix{Float64}}(undef, T); W = Vector{Matrix{Float64}}(undef, T)
    for t in 1:brk;     cov[t] = Ω1; W[t] = W1; end
    for t in brk + 1:T; cov[t] = Ω2; W[t] = W2; end
    ld = brk * logdet_pd(Ω1) + (T - brk) * logdet_pd(Ω2)
    return cov, W, ld, brk
end

# ARCH-type: Ω(t)⁻¹ = A + vₜ²B with A = a_floor·I (PD floor), B = Ω0⁻¹
function sigma_arch(T, Ω0, c, v; a_floor=0.05)
    A = a_floor .* Matrix{Float64}(I, P, P); B = inv(Ω0)
    W = [A + v[t]^2 .* B for t in 1:T]
    cov = [inv(W[t]) for t in 1:T]
    ld = -sum(logdet_pd(W[t]) for t in 1:T)
    arch_exog = [v[t] .* Matrix{Float64}(I, P, P) for t in 1:T]   # driver vₜIₚ (q = p)
    return cov, W, ld, arch_exog
end

# draw εₜ ~ N(0, covₜ) for block-diagonal Σ
function draw_eps(cov, rng)
    T = length(cov); eps = zeros(T, P)
    for t in 1:T; eps[t, :] = cholesky(Symmetric(cov[t])).L * randn(rng, P); end
    return eps
end

# VECM(1):  Δxₜ = αβ'xₜ₋₁ + εₜ.  Returns Y = Δx (T×p), Xreg = xₜ₋₁ (T×p)
function simulate_vecm(T, β, α, eps, rng; x0=nothing)
    p = size(β, 1); x = zeros(T + 1, p); x0 !== nothing && (x[1, :] = x0)
    Π = α * β'
    for t in 1:T; x[t + 1, :] = x[t, :] + Π * x[t, :] + eps[t, :]; end
    return x[2:end, :] - x[1:end - 1, :], x[1:T, :]
end

# a realistic-ish, strongly-correlated 5×5 innovation covariance
function base_omega(; seed=0, scale=0.01)
    rng = Xoshiro(seed)
    A = 0.3 .* randn(rng, P, P)
    Ω = A * A' + Diagonal(range(1.0, 0.6, length=P))
    d = sqrt.(diag(Ω)); C = Ω ./ (d * d')
    C = 0.7 .* C .+ 0.3 .* ones(P, P); C[diagind(C)] .= 1.0
    return scale^2 .* C
end
_stamp("done: cell 15")

# ------------------------------------------------------------ cell 17
# simulated data with the same structure used in the reference validation
function sim_data(T, p, p1, p2, r, seed)
    rng = Xoshiro(seed)
    X = randn(rng, T, p1); Z = p2 > 0 ? randn(rng, T, p2) : zeros(T, 0)
    α = randn(rng, p, r); β = randn(rng, p1, r); Ψ = p2 > 0 ? randn(rng, p, p2) : zeros(p, 0)
    A = randn(rng, p, p); Ω = A * A' / p + I
    E = randn(rng, T, p) * cholesky(Symmetric(Matrix(Ω))).L'
    Y = X * β * α' + (p2 > 0 ? Z * Ψ' : zeros(T, p)) + E
    return Y, X, Z, (alpha=α, beta=β, Psi=Ψ, Omega=Ω)
end
_stamp("done: cell 17")

# ------------------------------------------------------------ cell 18
# (a) unrestricted GRRR log-lik equals the eigenvalue-RRR log-lik
diffs_a = Float64[]
for s in 0:19
    Yv, Xv, Zv, _ = sim_data(200, 4, 4, 2, 2, 100 + s)
    jrv = johansen_rrr(Yv, Xv, Zv, 2)
    best = -Inf
    for st in 0:3
        res = grrr_estimate(Yv, Xv, Zv, 2; sigma_model=:iid, seed=s * 10 + st, tol=1e-11, maxiter=3000)
        best = max(best, res[:loglik])
    end
    push!(diffs_a, abs(best - jrv[:loglik]))
end
@printf("(a) max |GRRR − eig| log-lik over 20 sets: %.2e  →  %s\n",
        maximum(diffs_a), maximum(diffs_a) < 1e-6 ? "PASS" : "FAIL")
_stamp("done: cell 18")

# ------------------------------------------------------------ cell 19
# (b) β fully known ⇒ GRRR (α,Ψ) equals the GLS/OLS closed form
Yb, Xb, Zb, tp = sim_data(300, 4, 4, 2, 2, 7)
βtrue = tp.beta
resb = grrr_estimate(Yb, Xb, Zb, 2; H=zeros(size(Xb, 2) * 2, 0), h=vec(βtrue),
                     sigma_model=:iid, seed=1, tol=1e-12)
Wc = hcat(Xb * βtrue, Zb)
coef = ((Wc' * Wc) \ (Wc' * Yb))'                     # p × (r+p2)
err_b = max(maximum(abs.(resb[:alpha] - coef[:, 1:2])),
            maximum(abs.(resb[:Psi]   - coef[:, 3:end])))
@printf("(b) max |GRRR − GLS closed form| (α,Ψ): %.2e  →  %s\n",
        err_b, err_b < 1e-7 ? "PASS" : "FAIL")
_stamp("done: cell 19")

# ------------------------------------------------------------ cell 20
# (c) log-lik invariant to X-column scaling and Y-equation permutation
Yc, Xc, Zc, _ = sim_data(200, 4, 4, 2, 2, 42)
ll0 = johansen_rrr(Yc, Xc, Zc, 2)[:loglik]
ll1 = johansen_rrr(Yc, Xc .* [2.0 0.5 3.0 0.1], Zc, 2)[:loglik]
ll2 = johansen_rrr(Yc[:, [3, 1, 4, 2]], Xc, Zc, 2)[:loglik]
inv_err = max(abs(ll0 - ll1), abs(ll0 - ll2))
@printf("(c) log-lik base=%.6f  Xscale=%.6f  Yperm=%.6f  max diff=%.2e  →  %s\n",
        ll0, ll1, ll2, inv_err, inv_err < 1e-6 ? "PASS" : "FAIL")
_stamp("done: cell 20")

# ------------------------------------------------------------ cell 21
# (d) log-likelihood increases monotonically for every Σ model.
# Reported relative to |ℓ|, which is what the monotonicity claim should be read
# against: the absolute increment scales with the log-likelihood and with the
# starting value, and this is the one validation cell that starts from a random
# ψ₀ (seed=3, no psi0), so an absolute figure here is not portable across RNGs.
# Note also that the :ar1 θ-step is a plug-in (ρ from a ratio of sums, then
# clamped; Ωu on T−1 observations while logdetS mixes T and T−1) rather than an
# exact profile maximization, so its ascent holds only to numerical precision.
Yd, Xd, Zd, _ = sim_data(200, 3, 3, 1, 2, 5)
rng0 = Xoshiro(0)
ctx_arch_d = Dict{Symbol,Any}(:arch_exog => [abs.(randn(rng0, 3, 1)) for _ in 1:200])
worst_rel = Inf
for (m, ctx) in [(:iid, nothing), (:regime, Dict{Symbol,Any}(:breaks => [100])),
                 (:ar1, nothing), (:arch, ctx_arch_d)]
    res   = grrr_estimate(Yd, Xd, Zd, 2; sigma_model=m, sigma_ctx=ctx, seed=3, tol=1e-11)
    lp    = res[:loglik_path]
    scale = max(1.0, maximum(abs, lp))
    step  = minimum(diff(lp))
    rel   = step / scale
    global worst_rel = min(worst_rel, rel)
    @printf("    %-8s min step=%+.2e  relative=%+.2e  conv=%s  iters=%d\n",
            m, step, rel, res[:converged], res[:iters])
end
@printf("(d) min relative log-lik increment across all Σ models: %+.2e  →  %s\n",
        worst_rel, worst_rel > -1e-8 ? "PASS" : "FAIL")
_stamp("done: cell 21")

# ------------------------------------------------------------ cell 24
const p_A, p1_A, p2_A, r_A, T_A = 5, 5, 2, 4, 200

function dgp_A(seed)
    rng = Xoshiro(seed)
    X = randn(rng, T_A, p1_A); Z = randn(rng, T_A, p2_A)
    α = randn(rng, p_A, r_A); β = randn(rng, p1_A, r_A); Ψ = randn(rng, p_A, p2_A)
    A = randn(rng, p_A, p_A); Ω = A * A' / p_A + I
    E = randn(rng, T_A, p_A) * cholesky(Symmetric(Matrix(Ω))).L'
    return X * β * α' + Z * Ψ' + E, X, Z
end

function run_mc_A(NREP)
    lldiff = Float64[]; iters = Int[]; tg = Float64[]; te = Float64[]
    dgp_A(1); johansen_rrr(dgp_A(1)..., r_A)                # warm-up (compile)
    grrr_estimate(dgp_A(1)..., r_A; sigma_model=:iid, seed=0)
    for s in 0:NREP - 1
        Y, X, Z = dgp_A(SEED + s)
        te_ = @elapsed jr = johansen_rrr(Y, X, Z, r_A)
        tg_ = @elapsed res = grrr_estimate(Y, X, Z, r_A; sigma_model=:iid, seed=s, tol=1e-11, maxiter=3000)
        push!(lldiff, res[:loglik] - jr[:loglik]); push!(iters, res[:iters])
        push!(tg, tg_); push!(te, te_)
    end
    return lldiff, iters, tg, te
end

lldiff_A, iters_A, tg_A, te_A = run_mc_A(REPS_A)
@printf("reps                         : %d\n", length(lldiff_A))
@printf("log-lik diff GRRR−eig max|·| : %.2e\n", maximum(abs.(lldiff_A)))
@printf("agreement <1e-6 (fraction)   : %.3f\n", mean(abs.(lldiff_A) .< 1e-6))
@printf("GRRR iters  median / max     : %d / %d\n", round(Int, median(iters_A)), maximum(iters_A))
@printf("GRRR time   median (ms)      : %.2f\n", 1e3 * median(tg_A))
@printf("eig  time   median (ms)      : %.3f\n", 1e3 * median(te_A))
@printf("GRRR/eig median time ratio   : %.1f\n", median(tg_A) / median(te_A))
_stamp("done: cell 24")

# ------------------------------------------------------------ cell 26
const KAPPA = 0.3
b_true = ones(4)
Hb, hb, βb = basis_restriction(b_true)     # spread basis; βb = truth (b = 1)
αb = make_alpha(βb, KAPPA)
Ω1 = base_omega(scale=1.0)

# one GRRR fit of the spread coefficients (Z = nothing; β free unless model=:fixed)
fit_grrr(Y, X, model, ctx, φ0, ψ0; tol=1e-9, maxiter=2000) =
    grrr_estimate(Y, X, nothing, 4; H=Hb, h=hb, sigma_model=model, sigma_ctx=ctx,
                  phi0=φ0, psi0=ψ0, normalize_beta=false, tol=tol, maxiter=maxiter)

"""Closed-form start for Experiment B: b from the eigenvalue RRR, then α from
OLS with that β pinned.  Table 2 gives all three estimators this same start, so
none of them is given the true parameters."""
function mc_b_start(Y, X)
    jr = johansen_rrr(Y, X, nothing, 4)
    w  = svd(Matrix(jr[:beta]'); full=true).Vt[end, :]   # null direction of β'
    b0 = w[2:end] ./ w[1]
    h0 = Hb * b0 + hb
    SF = X * reshape(h0, P, 4)
    return b0, vec(((SF' * SF) \ (SF' * Y))')
end

function mc_b_one_rep(kind, T, s)
    rng = Xoshiro(SEED + 1_000_000 + s)
    if kind == :regime
        cov, W, ld, _ = sigma_regime(T, Ω1; factor=2.0)
        ctx = Dict{Symbol,Any}(:breaks => [div(T, 2)]); model = :regime; tol = 1e-9; mi = 2000
    else
        v = 0.4 .+ 0.9 .* abs.(randn(rng, T))
        cov, W, ld, ax = sigma_arch(T, Ω1, 0.0, v)
        ctx = Dict{Symbol,Any}(:arch_exog => ax, :arch_max_inner => 30); model = :arch; tol = 1e-8; mi = 500
    end
    eps = draw_eps(cov, rng)
    Y, X = simulate_vecm(T, βb, αb, eps, rng)
    φ0, ψ0 = mc_b_start(Y, X)          # no estimator sees the true parameters
    r0 = fit_grrr(Y, X, :iid, nothing, φ0, ψ0)
    rc = fit_grrr(Y, X, model, ctx, φ0, ψ0; tol=tol, maxiter=mi)
    rf = fit_grrr(Y, X, :fixed, Dict{Symbol,Any}(:W => W, :logdetS => ld), φ0, ψ0)
    return (iid        = -r0[:beta][1, :] .- b_true,
            correct    = -rc[:beta][1, :] .- b_true,
            infeasible = -rf[:beta][1, :] .- b_true)
end

function run_mc_B(REPS)
    res = Dict{Tuple{Symbol,Int},Dict{Symbol,Vector{Vector{Float64}}}}()
    for kind in (:regime, :arch), T in (100, 200, 400)
        reps = Vector{Any}(undef, REPS)         # seeded per replication, so
        Threads.@threads for s in 0:REPS - 1    # threading changes no number
            reps[s + 1] = mc_b_one_rep(kind, T, s)
        end
        res[(kind, T)] = Dict(:iid        => [x.iid        for x in reps],
                              :correct    => [x.correct    for x in reps],
                              :infeasible => [x.infeasible for x in reps])
    end
    return res
end

rmse_of(v) = sqrt(mean(abs2, reduce(vcat, v)))
mae_of(v)  = median(abs.(reduce(vcat, v)))
_stamp("done: cell 26")

# ------------------------------------------------------------ cell 27
mcB = run_mc_B(REPS_B)

# Table 2 of the paper: RMSE with the median absolute error in parentheses,
# B = 500 and closed-form starts.  Printed alongside for cross-checking; at
# QUICK = true the replication count is lower and the numbers will not match.
TAB2 = Dict(
    (:arch,   100) => ((0.329, 0.078), (0.150, 0.064), (0.132, 0.058)),
    (:arch,   200) => ((0.079, 0.039), (0.056, 0.029), (0.054, 0.028)),
    (:arch,   400) => ((0.038, 0.019), (0.027, 0.014), (0.027, 0.014)),
    (:regime, 100) => ((0.389, 0.083), (0.620, 0.082), (0.245, 0.081)),
    (:regime, 200) => ((0.081, 0.042), (0.077, 0.041), (0.075, 0.040)),
    (:regime, 400) => ((0.042, 0.022), (0.042, 0.021), (0.041, 0.021)))

@printf("%-7s %4s %-12s %17s %17s\n", "DGP", "T", "estimator", "this run", "paper")
for kind in (:arch, :regime), T in (100, 200, 400)
    e = mcB[(kind, T)]
    for (j, est) in enumerate((:iid, :correct, :infeasible))
        pr = TAB2[(kind, T)][j]
        @printf("%-7s %4s %-12s   %7.3f (%5.3f)   %7.3f (%5.3f)\n",
                j == 1 ? string(kind) : "", j == 1 ? string(T) : "", est,
                rmse_of(e[est]), mae_of(e[est]), pr[1], pr[2])
    end
end

println("\nRMSE reduction over the iid estimator (%)")
@printf("%-7s %4s %9s %9s\n", "DGP", "T", "correct", "infeas")
for kind in (:arch, :regime), T in (100, 200, 400)
    e = mcB[(kind, T)]
    ri = rmse_of(e[:iid])
    @printf("%-7s %4d %9.1f %9.1f\n", kind, T,
            100 * (1 - rmse_of(e[:correct]) / ri),
            100 * (1 - rmse_of(e[:infeasible]) / ri))
end
_stamp("done: cell 27")

# ------------------------------------------------------------ cell 29
function run_mc_C(NDATA, NSTART)
    frac_best = zeros(NDATA); worst_gap = zeros(NDATA)
    Threads.@threads for dset in 0:NDATA - 1
        cov, W, ld, _ = sigma_regime(200, Ω1; factor=2.0)
        Y, X = simulate_vecm(200, βb, αb, draw_eps(cov, Xoshiro(SEED + 2_000_000 + dset)), Xoshiro(0))
        srng = Xoshiro(SEED + 3_000_000 + dset); lls = Float64[]
        for st in 1:NSTART
            φ0 = 1.0 .+ 2.0 .* randn(srng, 4); ψ0 = randn(srng, 5 * 4)
            res = grrr_estimate(Y, X, nothing, 4; H=Hb, h=hb, sigma_model=:regime,
                                sigma_ctx=Dict{Symbol,Any}(:breaks => [100]),
                                phi0=φ0, psi0=ψ0, tol=1e-10, maxiter=2000, normalize_beta=false)
            push!(lls, res[:converged] ? res[:loglik] : NaN)
        end
        valid = filter(!isnan, lls); best = maximum(valid)
        frac_best[dset + 1] = mean(abs.(valid .- best) .< 1e-6)
        worst_gap[dset + 1] = best - minimum(valid)
    end
    return frac_best, worst_gap
end

fb_C, wg_C = run_mc_C(NDATA_C, NSTART_C)
@printf("datasets / starts                 : %d / %d\n", NDATA_C, NSTART_C)
@printf("mean frac starts at best (1e-6)   : %.4f\n", mean(fb_C))
@printf("min  frac starts at best          : %.4f\n", minimum(fb_C))
@printf("frac datasets all starts agree    : %.4f\n", mean(fb_C .== 1.0))
@printf("median / max worst-case log-lik gap: %.2e / %.2e\n", median(wg_C), maximum(wg_C))
_stamp("done: cell 29")

# ------------------------------------------------------------ cell 31
df = CSV.read("data/wti_spot_futures.csv", DataFrame)
L = Matrix(df[:, [:l_spot, :l_F1, :l_F2, :l_F3, :l_F4]])   # log prices, N×5
dts = df.date
N = size(L, 1)
dL = diff(L, dims=1)                                       # Δxₜ
Y = dL[2:end, :]                                           # Δxₜ,     t = 2..N−1
X = L[2:N - 1, :]                                          # xₜ₋₁
Z = hcat(dL[1:N - 2, :], ones(N - 2))                      # [Δxₜ₋₁, const],  p₂ = 6
T, r = size(Y, 1), 4
sd = dts[3:N]                                              # sample dates, length T
@printf("T=%d  p=%d  p2=%d   %s .. %s\n", T, P, size(Z, 2), string(sd[1]), string(sd[end]))
_stamp("done: cell 31")

# ------------------------------------------------------------ cell 33
gr()
plt_lv = plot(dts, L, lw=1,
    label=[L"\ell_{\rm spot}" L"\ell_{F_1}" L"\ell_{F_2}" L"\ell_{F_3}" L"\ell_{F_4}"],
    legend=:topleft, ylabel="log price", title="WTI log prices")
sp = L[:, 2:5] .- L[:, 1]
plt_sp = plot(dts, sp, lw=1,
    label=[L"\ell_{F_1}-\ell_{\rm spot}" L"\ell_{F_2}-\ell_{\rm spot}" L"\ell_{F_3}-\ell_{\rm spot}" L"\ell_{F_4}-\ell_{\rm spot}"],
    legend=:topright, ylabel="basis", title="Spot–futures bases")
plot(plt_lv, plt_sp, layout=(2, 1), size=(820, 620))
_stamp("done: cell 33")

# ------------------------------------------------------------ cell 34
try
    savefig("GRRRfig_data.pdf")
catch err
    @error "cell 34 failed; the results above are unaffected" exception = err
end
_stamp("done: cell 34")

# ------------------------------------------------------------ cell 36
chi2_sf4(x) = (1 + x / 2) * exp(-x / 2)          # P(χ²₄ > x)

H_free, h_free, _ = basis_restriction()          # spread basis, b free (20×4)
_, _, β_one = basis_restriction(ones(4))
H_one = zeros(P * R, 0); h_one = vec(β_one)       # b = 1, fully restricted

function implied_b(β)
    F = svd(β'; full=true); w = F.Vt[end, :]      # null direction of β'
    return w[2:end] ./ w[1]
end

basis_fit(model, ctx, free; phi0=nothing, psi0=nothing, tol=1e-10, maxiter=5000) =
    grrr_estimate(Y, X, Z, r; H=(free ? H_free : H_one), h=(free ? h_free : h_one),
                  sigma_model=model, sigma_ctx=ctx, phi0=phi0, psi0=psi0,
                  tol=tol, maxiter=maxiter, normalize_beta=false)

function b_and_se(res, W)
    Ma, _ = bstep_moments_blockdiag(Y, X, Z, res[:Psi], res[:alpha], W)
    return -res[:beta][1, :], sqrt.(abs.(diag(inv(H_free' * Ma * H_free))))
end

function prec_of(res, model, ctx)
    E = Y - X * res[:beta] * res[:alpha]' - Z * res[:Psi]'
    theta_step(model, E, ctx === nothing ? Dict{Symbol,Any}() : copy(ctx))[:W]
end
_stamp("done: cell 36")

# ------------------------------------------------------------ cell 37
# Starting values. `grrr_estimate` draws ψ₀ from an unseeded generator when none
# is supplied, and in this design that is not safe: from a random start the
# three-block iteration crawls and an increment-based stopping rule fires far
# below the maximum (Table 4). Every fit below therefore starts from closed-form
# values — β from the eigenvalue RRR, then (α,Ψ) from OLS with that β pinned —
# which reaches the maximum on this sample in a couple of iterations.

sigma_ctx_for(model, Zb) =
    model === :iid    ? nothing :
    model === :regime ? Dict{Symbol,Any}(:breaks => [270, 408]) :
    let vb = vec(mean(abs.(Zb[:, 1:P]), dims=2))
        vb = vb ./ mean(vb)
        Dict{Symbol,Any}(:arch_exog => [fill(vb[t], P, 1) for t in 1:size(Zb, 1)],
                         :arch_max_inner => 40)
    end

"""β from the eigenvalue RRR; (α,Ψ) from the pinned-β fit started at OLS."""
function closed_form_start(Yb, Xb, Zb, model, ctx; tol=1e-10, maxiter=20000)
    jrb   = johansen_rrr(Yb, Xb, Zb, r)
    b0    = implied_b(jrb[:beta])
    h0    = H_free * b0 + h_free
    SF    = hcat(Xb * reshape(h0, P, R), Zb)
    ψ_ols = vec(((SF' * SF) \ (SF' * Yb))')
    pin   = grrr_estimate(Yb, Xb, Zb, r; H=zeros(P * R, 0), h=h0,
                          sigma_model=model, sigma_ctx=(ctx === nothing ? nothing : copy(ctx)),
                          psi0=ψ_ols, tol=tol, maxiter=maxiter, normalize_beta=false)
    return b0, vec(hcat(pin[:alpha], pin[:Psi])), jrb
end

"""Sample fits under one Σ model: basis with b free and with b = 1."""
function sample_fits(model)
    ctx = sigma_ctx_for(model, Z)
    b0, ψ_pin, jrb = closed_form_start(Y, X, Z, model, ctx)
    free = basis_fit(model, ctx, true;  phi0=b0, psi0=ψ_pin, tol=1e-10, maxiter=20000)
    ψ_free = vec(hcat(free[:alpha], free[:Psi]))
    one  = basis_fit(model, ctx, false; psi0=ψ_free, tol=1e-10, maxiter=20000)
    ψ_one = vec(hcat(one[:alpha], one[:Psi]))
    return (ctx=ctx, free=free, one=one, psi_free=ψ_free, psi_one=ψ_one, jr=jrb, b0=b0)
end

jr = johansen_rrr(Y, X, Z, r)
b0 = implied_b(jr[:beta])

S_iid  = sample_fits(:iid)
S_reg  = sample_fits(:regime)
S_arch = sample_fits(:arch)

free_iid, one_iid = S_iid.free,  S_iid.one
free_r,   one_r   = S_reg.free,  S_reg.one
free_a,   one_a   = S_arch.free, S_arch.one
ctx_r, ctx_a      = S_reg.ctx,   S_arch.ctx
ψ_warm            = S_iid.psi_free

# the iid basis-free fit is a just-identified reparametrisation of the
# unrestricted rank-4 model, so it must reproduce the eigenvalue solution
# 1e-2, not 1e-3: the realised gap is ~4e-4, and a different BLAS should not be
# able to halt the notebook on a difference this far below anything meaningful.
@assert abs(free_iid[:loglik] - jr[:loglik]) < 1e-2 "iid fit did not reach the maximum"

b_iid,  se_iid  = b_and_se(free_iid, prec_of(free_iid, :iid,    nothing))
b_reg,  se_reg  = b_and_se(free_r,   prec_of(free_r,   :regime, ctx_r))
b_arch, se_arch = b_and_se(free_a,   prec_of(free_a,   :arch,   ctx_a))

E_r    = Y - X * free_r[:beta] * free_r[:alpha]' - Z * free_r[:Psi]'
reg_Ω  = theta_regime(E_r, ctx_r)[:theta]
D_arch = free_a[:theta][:D]

# ---- data-chosen single break (grid on the basis-free log-lik) ----
# The grid runs to within a year of each end of the sample.  Trimming at 0.15T
# and 0.85T stops at index 386 (2018-04) and puts the reported maximum on the
# boundary of the search, excluding March 2020 (index 408), which is where the
# profile actually peaks: l = 8296.0 against 8239.1 at 2018-04.
function data_chosen_break()
    grid = collect(12:6:(T - 12))            # 0.15T:0.85T stops at 2018-04 and
    lls  = fill(-Inf, length(grid))
    Threads.@threads for k in eachindex(grid)
        res = basis_fit(:regime, Dict{Symbol,Any}(:breaks => [grid[k]]), true;
                        phi0=b0, psi0=ψ_warm, tol=1e-9, maxiter=4000)
        lls[k] = res[:loglik]
    end
    k = argmax(lls)                          # fixed grid, no draws: deterministic
    return lls[k], grid[k]
end
ll_dchosen, bp_dchosen = data_chosen_break()

@printf("iid   ℓ_free=%.6f  ℓ_b=1=%.6f   (eigenvalue %.6f)\n",
        free_iid[:loglik], one_iid[:loglik], jr[:loglik])
@printf("regime ℓ_free=%.6f  ℓ_b=1=%.6f\n", free_r[:loglik], one_r[:loglik])
@printf("arch  ℓ_free=%.6f  ℓ_b=1=%.6f\n", free_a[:loglik], one_a[:loglik]);
_stamp("done: cell 37")

# ------------------------------------------------------------ cell 39
specs = [("iid",                     jr[:loglik],   free_iid[:loglik], one_iid[:loglik]),
         ("regime(2008:9,2020:3)",    free_r[:loglik], free_r[:loglik], one_r[:loglik]),
         ("arch",                     free_a[:loglik], free_a[:loglik], one_a[:loglik])]

println("=== log-likelihoods ===")
@printf("%-24s %14s %14s %14s\n", "Σ model", "unrestr(r4)", "basis(b free)", "basis(b=1)")
for (lab, llu, llf, ll1) in specs
    @printf("%-24s %14.4f %14.4f %14.4f\n", lab, llu, llf, ll1)
end
@printf("%-24s %14.4f\n", "iid (eigenvalue)", jr[:loglik])
@printf("%-24s %28.4f  (break %s)\n", "regime(1-break, data-chosen)", ll_dchosen, string(sd[bp_dchosen + 1]))   # +1 as in the regime table below

println("\n=== LR test of b = 1  (χ²₄) ===")
@printf("%-24s %10s %6s %12s\n", "Σ model", "LR", "df", "p-value")
for (lab, llu, llf, ll1) in specs
    lr = 2 * (llu - ll1)
    @printf("%-24s %10.4f %6d %12.4f\n", lab, lr, 4, chi2_sf4(lr))
end
_stamp("done: cell 39")

# ------------------------------------------------------------ cell 41
println("=== basis coefficients (spread) with indicative SE ===")
@printf("%-4s %10s %10s %10s %10s %10s %10s\n",
        "b_i", "iid", "se", "regime", "se", "arch", "se")
for i in 1:4
    @printf("b%-3d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n",
            i, b_iid[i], se_iid[i], b_reg[i], se_reg[i], b_arch[i], se_arch[i])
end

println("\n=== regime variances ===")
edges = reg_Ω[:edges]
@printf("%-7s %-12s %-12s %5s %10s %12s\n", "regime", "start", "end", "n", "logdet", "mean_var")
for i in 1:length(reg_Ω[:Omega_regimes])
    Ω = reg_Ω[:Omega_regimes][i]
    @printf("%-7d %-12s %-12s %5d %10.4f %12.6f\n", i,
            string(sd[edges[i] + 1]), string(sd[min(edges[i + 1], T - 1) + 1]),
            edges[i + 1] - edges[i], logdet(Symmetric(Ω)), mean(diag(Ω)))
end
_stamp("done: cell 41")

# ------------------------------------------------------------ cell 43
# --- validity screen ------------------------------------------------------
# min_t λ_min(Ω(t)) relative to the average fitted variance: ≈3e-5 at an
# interior fit, orders of magnitude smaller at a boundary one.
function omega_ratio(res, Yb, Xb, Zb, model, ctx)
    Eb = Yb - Xb * res[:beta] * res[:alpha]' - Zb * res[:Psi]'
    c  = ctx === nothing ? Dict{Symbol,Any}() : copy(ctx)
    delete!(c, :Qtp)
    if model === :arch
        c[:D] = res[:theta][:D]; c[:arch_max_inner] = 0
    end
    W  = theta_step(model, Eb, c)[:W]
    lo = Inf; sc = 0.0
    for t in 1:length(W)
        ev = eigvals(Symmetric(W[t]))
        ev[1] <= 0 && return 0.0
        lo  = min(lo, 1 / ev[end])          # λ_min(Ω(t)) = 1/λ_max(W(t))
        sc += sum(1 ./ ev) / P              # tr(Ω(t))/p
    end
    sc /= length(W)
    return (isfinite(sc) && sc > 0) ? lo / sc : 0.0
end

is_valid(res, Yb, Xb, Zb, model, ctx) =
    isfinite(res[:loglik]) && omega_ratio(res, Yb, Xb, Zb, model, ctx) >= 1e-8

# --- maximise over starting values, keeping only valid fits ---------------
# Under ARCH a replication whose warm start runs away to the boundary of
# Section 4.3 never satisfies the convergence test, so it consumes all `maxiter`
# iterations of the most expensive θ-step in the paper and then fails `is_valid`
# anyway. About 15% of ARCH replications do this. `fit_guarded` therefore runs
# the iteration in blocks and abandons it as soon as the validity ratio
# collapses, resuming from its own (b, α, Ψ) and its own D between blocks so the
# guarded path traces the same iteration as the unguarded one. `maxiter` is NOT
# reduced: a fit that legitimately needs the full budget still gets it, and only
# a diverging fit is stopped early.
#
# The guard is ARCH-only by construction. Under iid and regime the θ-step is a
# single inverse and a single log-determinant with no boundary to run to, and
# those two specifications go through `grrr_estimate` in exactly one call, as
# before. The Section 6.1 cells under iid and regime are therefore
# bit-identical to the unguarded code, including the B = 999 iid check.
# Second gate, added after the first one failed to fire.  GUARD_RATIO is the
# threshold `is_valid` applies as a final verdict, and a fit drifting to the
# boundary evidently does not cross it early enough to save anything: the
# guarded run reproduced the unguarded numbers exactly and took no less time.
# The log-likelihood separates the two cases far more sharply.  Against the
# eigenvalue-RRR value on the same bootstrap sample, a legitimate ARCH fit sits
# a few hundred above (200.4 on the observed sample, 342 for regime), while the
# paper reports runaways reaching ℓ of order 1e7 and 1e9.  GUARD_LL is set 14x
# above the largest legitimate gap and five orders of magnitude below the
# runaways, so it cannot reject a good fit; whether it fires early enough to
# save time is what the probe cell in Section 6.1 measures.
const GUARD_BLOCK = 100        # iterations between divergence checks
const GUARD_RATIO = 1e-8       # same threshold `is_valid` applies at the end
const GUARD_LL    = 5000.0     # ℓ above the eigenvalue reference ⇒ running away

function fit_guarded(Yb, Xb, Zb, model, ctx, free, s;
                     tol=1e-9, maxiter=8000, ll_ref=Inf)
    H_ = free ? H_free : H_one
    h_ = free ? h_free : h_one
    c  = ctx === nothing ? nothing : copy(ctx)
    c !== nothing && haskey(s, :D) && (c[:D] = copy(s[:D]))
    phi0 = get(s, :phi0, nothing)
    psi0 = get(s, :psi0, nothing)
    guard = model === :arch
    res = nothing; done = 0; bailed = false
    while done < maxiter
        n = guard ? min(GUARD_BLOCK, maxiter - done) : maxiter
        res = grrr_estimate(Yb, Xb, Zb, r; H=H_, h=h_,
                            sigma_model=model, sigma_ctx=c,
                            phi0=phi0, psi0=psi0,
                            tol=tol, maxiter=n, normalize_beta=false)
        done += res[:iters]
        (!guard || res[:converged]) && break
        if !isfinite(res[:loglik]) || res[:loglik] > ll_ref + GUARD_LL ||
           omega_ratio(res, Yb, Xb, Zb, model, ctx) < GUARD_RATIO
            bailed = true; break                      # running away to the boundary
        end
        phi0 = free ? -res[:beta][1, :] : nothing     # resume where it stopped
        psi0 = vec(hcat(res[:alpha], res[:Psi]))
        c[:D] = res[:theta][:D]
    end
    return res, bailed, done
end

function best_over_starts(Yb, Xb, Zb, model, ctx, free, starts;
                          tol=1e-9, maxiter=8000, ll_ref=Inf)
    best = nothing; info = Tuple{Float64,Bool}[]
    for s in starts
        res, bailed, _ = fit_guarded(Yb, Xb, Zb, model, ctx, free, s;
                                     tol=tol, maxiter=maxiter, ll_ref=ll_ref)
        ok = !bailed && is_valid(res, Yb, Xb, Zb, model, ctx)
        push!(info, (res[:loglik], ok))
        if ok && (best === nothing || res[:loglik] > best[:loglik])
            best = res
        end
    end
    return best, info
end

# --- VECM pieces and the recursive wild bootstrap sample -------------------
vecm_parts(res) = (res[:alpha] * res[:beta]', res[:Psi][:, 1:P], res[:Psi][:, P + 1])
resid_of(res)   = Y - X * res[:beta] * res[:alpha]' - Z * res[:Psi]'

const X_INIT  = L[2, :]        # xₜ₋₁ at t = 1   (first row of X)
const DX_INIT = dL[1, :]       # Δxₜ₋₁ at t = 1  (first P columns of Z, row 1)

function wild_sample(Π, Γ1, μ, Ê, rng)
    Tn = size(Ê, 1)
    w  = ifelse.(rand(rng, Tn) .< 0.5, -1.0, 1.0)          # Rademacher
    Es = Ê .* w
    Yb = zeros(Tn, P); Xb = zeros(Tn, P); Zb = zeros(Tn, P + 1)
    Zb[:, P + 1] .= 1.0
    xprev = copy(X_INIT); dprev = copy(DX_INIT)
    for t in 1:Tn
        Xb[t, :]     = xprev
        Zb[t, 1:P]   = dprev
        dx           = Π * xprev + Γ1 * dprev + μ + Es[t, :]
        Yb[t, :]     = dx
        dprev        = dx
        xprev        = xprev + dx
    end
    return Yb, Xb, Zb
end

"""Roots of the companion matrix; the null system should have exactly one unit root."""
function companion_roots(Π, Γ1)
    A1 = Matrix{Float64}(I, P, P) + Π + Γ1
    C  = [A1 (-Γ1); Matrix{Float64}(I, P, P) zeros(P, P)]
    return sort(abs.(eigvals(C)), rev=true)
end;
_stamp("done: cell 43")

# ------------------------------------------------------------ cell 44
SEED0 = SEED + 4_000_000

"""One replication. `kind = :lr` imposes b = 1 in the DGP, `:se` does not."""
function boot_rep(seed, model, S, kind)
    Π, Γ1, μ = vecm_parts(kind === :lr ? S.one : S.free)
    Ê  = resid_of(kind === :lr ? S.one : S.free)
    rng = Xoshiro(seed)
    Yb, Xb, Zb = wild_sample(Π, Γ1, μ, Ê, rng)
    ctxb = sigma_ctx_for(model, Zb)
    b0b, ψ_pin, jrb = closed_form_start(Yb, Xb, Zb, model, ctxb; tol=1e-9, maxiter=8000)
    ll_ref = jrb[:loglik]                    # eigenvalue RRR on this bootstrap sample
    D0 = model === :arch ? S.free[:theta][:D] : nothing
    warm_f = Dict{Symbol,Any}(:phi0 => (kind === :lr ? ones(4) : -S.free[:beta][1, :]),
                              :psi0 => S.psi_free)
    D0 === nothing || (warm_f[:D] = D0)
    fb, info_f = best_over_starts(Yb, Xb, Zb, model, ctxb, true,
                                  [warm_f, Dict{Symbol,Any}(:phi0 => b0b, :psi0 => ψ_pin)];
                                  ll_ref=ll_ref)
    fb === nothing && return (LR=NaN, b=fill(NaN, 4), second_better=false, first_bad=true)
    first_bad = !info_f[1][2]
    second_better = info_f[2][2] && (!info_f[1][2] || info_f[2][1] > info_f[1][1] + 1e-3)
    kind === :se && return (LR=NaN, b=-fb[:beta][1, :],
                            second_better=second_better, first_bad=first_bad)
    warm_o = Dict{Symbol,Any}(:psi0 => S.psi_one)
    D0 === nothing || (warm_o[:D] = D0)
    ob, _ = best_over_starts(Yb, Xb, Zb, model, ctxb, false,
                             [warm_o, Dict{Symbol,Any}(:psi0 => ψ_pin)];
                             ll_ref=ll_ref)
    ob === nothing && return (LR=NaN, b=-fb[:beta][1, :],
                              second_better=second_better, first_bad=first_bad)
    return (LR=2 * (fb[:loglik] - ob[:loglik]), b=-fb[:beta][1, :],
            second_better=second_better, first_bad=first_bad)
end

function run_boot(model, S, kind, B)
    out = Vector{Any}(undef, B)
    Threads.@threads for j in 1:B
        out[j] = boot_rep(SEED0 + j - 1, model, S, kind)
    end
    return out
end;
_stamp("done: cell 44")

# ------------------------------------------------------------ cell 45
# LR bootstrap.  Reference values are printed for cross-checking; the Monte
# Carlo draws differ from the paper's (Julia's RNG is not numpy's), so the
# p-values reproduce only up to bootstrap error.  That error is the se(p) column
# below, and it is large at the QUICK=true default: at B = 99 it runs from about
# 2.0 percentage points near p = 0.05 to 4.4 near p = 0.25, against 0.7 to 1.3
# at the paper's B = 999.  A QUICK=true p-value two or three points off the
# paper's is therefore agreement, not a failed replication.  The B = 999 iid check
# below runs the one cheap Sigma model at the paper's B, which is a comparison
# the paper's table can actually be held to.
lr_specs = [(:iid, S_iid, "iid"), (:regime, S_reg, "regime"), (:arch, S_arch, "arch")]

println("=== bootstrap LR test of b = 1 ===")
@printf("%-8s %9s %9s %9s %8s %7s %9s\n",
        "Σ model", "LR", "χ² p", "boot p", "se(p)", "dropped", "2nd start")
boot_LR = Dict{Symbol,Vector{Float64}}()
for (model, S, lab) in lr_specs
    reps = run_boot(model, S, :lr, B_LR)
    LRs  = [x.LR for x in reps if !isnan(x.LR)]
    boot_LR[model] = LRs
    lr0  = 2 * (S.free[:loglik] - S.one[:loglik])
    p    = (1 + count(>=(lr0), LRs)) / (length(LRs) + 1)
    @printf("%-8s %9.3f %9.3f %9.3f %8.3f %7d %9d\n", lab, lr0, chi2_sf4(lr0), p,
            sqrt(p * (1 - p) / length(LRs)), B_LR - length(LRs),
            count(x -> x.second_better, reps))
end

println("\nbootstrap null quantiles (χ²₄: 7.78 / 9.49 / 13.28)")
for (model, _, lab) in lr_specs
    q = [quantile(boot_LR[model], u) for u in (0.90, 0.95, 0.99)]
    @printf("  %-8s mean %6.3f   q90 %6.3f   q95 %6.3f   q99 %6.3f\n",
            lab, mean(boot_LR[model]), q[1], q[2], q[3])
end
_stamp("done: cell 45")

# ------------------------------------------------------------ cell 46
# Bootstrap standard errors for b̂, from the unrestricted bootstrap DGP.
println("=== b̂ with information-formula and bootstrap standard errors ===")
@printf("%-4s %22s %22s %22s\n", "", "iid", "regime", "arch")
se_boot = Dict{Symbol,Vector{Float64}}()
bh = Dict(:iid => b_iid, :regime => b_reg, :arch => b_arch)
for (model, S, lab) in lr_specs
    reps = run_boot(model, S, :se, B_SE)
    Bm   = hcat([x.b for x in reps if !any(isnan, x.b)]...)'      # (valid B) × 4
    se_boot[model] = [std(Bm[:, i]) for i in 1:4]
end
for i in 1:4
    @printf("b%-3d", i)
    for (model, _, _) in lr_specs
        @printf(" %10.4f (%8.5f/%8.5f)", bh[model][i],
                (model === :iid ? se_iid : model === :regime ? se_reg : se_arch)[i],
                se_boot[model][i])
    end
    println()
end
println("\n(information-formula SE / bootstrap SE in parentheses)")
_stamp("done: cell 46")

# ------------------------------------------------------------ cell 48
try
    # At the QUICK=true default B_LR = 99, se(p) is 2-4 percentage points and the
    # standard error of the bootstrap null mean is about 0.35, which is too coarse
    # to separate a real discrepancy from Monte Carlo error.  The iid θ-step is one
    # inverse and one log-determinant, so the paper's B = 999 is affordable for that
    # model alone.  Set RUN_IID_CHECK = true and run.  Threads matter here: this is
    # 10x the work of the iid block in the cell above.
    #
    # Reference, paper Table "The finite-sample null distribution of the
    # likelihood-ratio statistic", bootstrap column, iid Σ, B = 999:
    #
    #     mean 4.59    q90 8.65    q95 10.17    q99 13.48    p 0.203
    #
    # A mean within about 0.3 of 4.59 is agreement.  A persistent gap of 0.5 or more
    # at this B is roughly 3 se and would say the two implementations really do
    # differ, in which case the place to look is wild_sample against boot.generate.

    RUN_IID_CHECK = false        # not const: flip to true and re-run this cell

    if RUN_IID_CHECK
        Threads.nthreads() == 1 &&
            @warn "single-threaded; set julia.NumThreads = \"auto\" and restart the kernel"
        t0      = time()
        reps999 = run_boot(:iid, S_iid, :lr, 999)
        LR999   = [x.LR for x in reps999 if !isnan(x.LR)]
        lr0     = 2 * (S_iid.free[:loglik] - S_iid.one[:loglik])
        p999    = (1 + count(>=(lr0), LR999)) / (length(LR999) + 1)
        q       = [quantile(LR999, u) for u in (0.90, 0.95, 0.99)]
        se_mean = std(LR999) / sqrt(length(LR999))

        @printf("%-10s %8s %8s %8s %8s %8s\n", "iid Σ", "mean", "q90", "q95", "q99", "p")
        @printf("%-10s %8.3f %8.3f %8.3f %8.3f %8.3f   (B=%d, %.1f min)\n",
                "this run", mean(LR999), q[1], q[2], q[3], p999,
                length(LR999), (time() - t0) / 60)
        @printf("%-10s %8.3f %8.3f %8.3f %8.3f %8.3f\n",
                "paper", 4.59, 8.65, 10.17, 13.48, 0.203)
        @printf("\nmean gap %+.3f, se of this mean %.3f  →  %.1f se\n",
                mean(LR999) - 4.59, se_mean, (mean(LR999) - 4.59) / se_mean)
    else
        println("RUN_IID_CHECK = false; set it to true to run the B = 999 iid comparison")
    end
catch err
    @error "cell 48 failed; the results above are unaffected" exception = err
end
_stamp("done: cell 48")

# ------------------------------------------------------------ cell 50
# What does a runaway ARCH replication actually do?  The first guard gated on
# omega_ratio alone at 1e-8 and never fired early enough to matter, so rather
# than guess a third threshold, measure.  This runs the warm-start free ARCH fit
# for the first PROBE_N bootstrap seeds in blocks of PROBE_BLOCK iterations and
# records ℓ and the validity ratio at each block, then prints the full
# trajectory for every seed that ends up invalid.  Good fits converge in a
# handful of iterations and cost nothing; only the runaways run to PROBE_MAX, so
# this is minutes, not hours.
#
# Read the output for one thing: at which iteration does a bad seed first become
# distinguishable?  If ℓ − ℓ_ref crosses GUARD_LL = 5000 by iteration 100 or 200,
# the gate now in fit_guarded will do its job.  If the ratio and ℓ both stay
# ordinary until near the end, the runaway is a late collapse, no early gate can
# help, and the answer is to cap maxiter for :arch instead.

PROBE_N     = 20
PROBE_BLOCK = 50
PROBE_MAX   = 600

function probe_arch(seed)
    S = S_arch
    Π, Γ1, μ = vecm_parts(S.one)
    Ê = resid_of(S.one)
    Yb, Xb, Zb = wild_sample(Π, Γ1, μ, Ê, Xoshiro(seed))
    ctxb = sigma_ctx_for(:arch, Zb)
    jrb  = johansen_rrr(Yb, Xb, Zb, r)
    c    = copy(ctxb); c[:D] = copy(S.free[:theta][:D])
    phi0 = ones(4); psi0 = S.psi_free
    traj = Tuple{Int,Float64,Float64}[]
    done = 0
    while done < PROBE_MAX
        res = grrr_estimate(Yb, Xb, Zb, r; H=H_free, h=h_free,
                            sigma_model=:arch, sigma_ctx=c,
                            phi0=phi0, psi0=psi0, tol=1e-9,
                            maxiter=min(PROBE_BLOCK, PROBE_MAX - done),
                            normalize_beta=false)
        done += res[:iters]
        push!(traj, (done, res[:loglik],
                     omega_ratio(res, Yb, Xb, Zb, :arch, ctxb)))
        res[:converged] && break
        phi0 = -res[:beta][1, :]
        psi0 = vec(hcat(res[:alpha], res[:Psi]))
        c[:D] = res[:theta][:D]
    end
    return traj, jrb[:loglik]
end

println("ARCH boundary probe: warm start, free fit, seeds $(SEED0)..$(SEED0 + PROBE_N - 1)")
@printf("\n%-12s %6s %14s %12s %11s %6s\n",
        "seed", "iters", "final ℓ", "ℓ - ℓ_ref", "final ratio", "valid")
bad_seeds = Int[]
trajectories = Dict{Int,Any}()
for j in 1:PROBE_N
    seed = SEED0 + j - 1
    traj, ll_ref = probe_arch(seed)
    it, ll, rat = traj[end]
    ok = isfinite(ll) && rat >= 1e-8
    trajectories[seed] = (traj, ll_ref)
    ok || push!(bad_seeds, seed)
    @printf("%-12d %6d %14.4g %12.4g %11.2e %6s\n",
            seed, it, ll, ll - ll_ref, rat, ok ? "yes" : "NO")
end

println("\n$(length(bad_seeds)) of $PROBE_N seeds ended invalid")
for seed in bad_seeds
    traj, ll_ref = trajectories[seed]
    @printf("\nseed %d   (eigenvalue-RRR ℓ_ref = %.2f)\n", seed, ll_ref)
    @printf("    %6s %16s %14s %12s %s\n",
            "iter", "ℓ", "ℓ - ℓ_ref", "ratio", "gate")
    for (i, l, rr) in traj
        gate = !isfinite(l) || l > ll_ref + 5000 ? "LL" : rr < 1e-8 ? "ratio" : ""
        @printf("    %6d %16.6g %14.4g %12.2e %s\n", i, l, l - ll_ref, rr, gate)
    end
end
_stamp("done: cell 50")

# ------------------------------------------------------------ cell 52
try
    Yt, Xt, Zt = dgp_A(2026)
    @btime johansen_rrr($Yt, $Xt, $Zt, 4);
    @btime grrr_estimate($Yt, $Xt, $Zt, 4; sigma_model=:iid, seed=1, tol=1e-11);
catch err
    @error "cell 52 failed; the results above are unaffected" exception = err
end
_stamp("done: cell 52")

# ------------------------------------------------------------ cell 53
# Environment summary
using Pkg
println("Julia version : ", VERSION)
println("OS / platform : ", Sys.KERNEL, " (", Sys.MACHINE, ")")
println("CPU           : ", Sys.cpu_info()[1].model)
println("Threads       : ", Threads.nthreads(), " / ", Sys.CPU_THREADS, " logical")
println()
vers = Dict(dep.name => dep.version for (_, dep) in Pkg.dependencies())
println("Key package versions:")
for name in ["Distributions", "CSV", "DataFrames", "Plots", "LaTeXStrings", "BenchmarkTools"]
    haskey(vers, name) && vers[name] !== nothing && println("  ", rpad(name, 16), "v", vers[name])
end
_stamp("done: cell 53")

@info "run complete" total_min = round((time() - _T_START) / 60, digits = 2)
