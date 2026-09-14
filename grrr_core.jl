# GRRR estimator core, extracted verbatim from the replication notebook.
# Nothing here depends on p or r; the four-contract application sets those.


using LinearAlgebra, Statistics, Random, Distributions,
      CSV, DataFrames, Printf

# One seed for the whole notebook, 4 January 1951, the birth date of James G.
# MacKinnon. Each experiment takes its own block of the integer line so the
# streams cannot overlap.
const SEED = 19510104

Random.seed!(SEED)   # global seed for the Monte-Carlo sections

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
    if !haskey(ctx, :arch_S)                                # vec(d) = S vech(d), symmetric d
        S = zeros(m * m, m * (m + 1) ÷ 2)
        k = 0
        for j in 1:m, i in j:m
            k += 1
            S[i + (j - 1) * m, k] = 1.0
            S[j + (i - 1) * m, k] = 1.0
        end
        ctx[:arch_S] = S
    end
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
    Gs = zeros(m * m, T)                                    # reused every iteration
    for _ in 1:get(ctx, :arch_max_inner, 40)
        Wc = prec_blocks(D)
        M = [Qtp[t]' * inv(Wc[t]) * Qtp[t] for t in 1:T]    # Mₜ = Qₜ Pₜ Qₜ'
        Gm = _sym(sum(M) - RHS)
        norm(Gm) < 1e-9 && break
        # Newton operator Σₜ Mₜ ⊗ Mₜ, as one matrix product rather than T
        # Kronecker products or a quadruple loop.
        #
        # Put vec(Mₜ) in column t of Gs. Julia stores columns first, so the row
        # of Gs is the pair (a, c) with a running fastest, and
        #
        #     (Gs Gs')[(a,c), (b,d)] = Σₜ Mₜ[a,c] Mₜ[b,d].
        #
        # The operator needs (Mₜ ⊗ Mₜ)[(a,b), (c,d)] = Mₜ[a,c] Mₜ[b,d], which is
        # the same numbers with the middle pair of indices transposed. So one
        # gemm and one permutation replace T m⁴ scalar operations per inner
        # iteration. That loop was making the Julia ARCH θ-step about thirty
        # times slower than the numpy one, which does exactly this through
        # einsum. `test_dgp.jl` checks the identity against the literal
        # Kronecker sum.
        for t in 1:T
            Gs[:, t] = vec(M[t])
        end
        Op = reshape(permutedims(reshape(Gs * Gs', m, m, m, m), (1, 3, 2, 4)),
                     m * m, m * m)
        # Op is symmetric positive semidefinite, and it is singular whenever D
        # is not identified. For a driver of the form Xexₜ = vₜA with A of full
        # column rank q, the perturbation D₁₂ ↦ D₁₂ + AK with K skew leaves
        # every Qₜ'DQₜ unchanged, so the likelihood is exactly flat along a
        # subspace of dimension q(q−1)/2. Taking q < p shrinks that subspace; it
        # does not remove it, and only q = 1 is free of it. Ω(t) and the mean
        # parameters are identified regardless.
        #
        # The gradient has no component along the flat subspace, so the Newton
        # system is consistent and its minimum-norm solution is the Newton step
        # that moves D only in identified directions. Solve it that way always,
        # by eigendecomposition with a relative cutoff, rather than by LU: with
        # an exactly singular Op the LU throws, and with a numerically singular
        # one it returns a step with arbitrary components along the flat
        # directions.
        #
        # The system is posed on symmetric matrices, vec(d) = S vech(d), so the
        # step is symmetric by construction and the recorded null space is the
        # one that matters: on all m×m matrices it has dimension q², of which
        # q(q+1)/2 are antisymmetric directions a symmetric D never visits.
        # `test_dgp.jl` checks the recorded dimension against q(q−1)/2.
        S = ctx[:arch_S]
        F = eigen(Symmetric(S' * Op * S))
        λmax = maximum(F.values)
        λmax > 0 || break
        keep = F.values .> 1e-10 * λmax
        Vk = F.vectors[:, keep]
        x = Vk * ((Vk' * (S' * vec(Gm))) ./ F.values[keep])
        d = reshape(S * x, m, m)
        ctx[:arch_nullity] = count(!, keep)
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
                            :theta => Dict(:D => D, :nullity => get(ctx, :arch_nullity, 0)),
                            :engine => :blockdiag)
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


# ---------------------------------------------------------------------------
# archchol: globally proper ARCH-type covariance, Ω(t)⁻¹ = Qₜ'DQₜ with D = LL'.
#
# Qₜ has full column rank p, so D > 0 gives a positive definite Ω(t) at EVERY
# driver value and not merely at the observed dates. The unrestricted maximiser
# does not have that property on the crude-oil data: it has one eigenvalue of
# −1.4e−4 against a largest of 2.9e6, and Q'D̂Q is indefinite for v ∈ [6.63, 7.01],
# a window the observed sample straddles. Parametrising D through a square factor
# L confines the fit to the cone.
#
# The θ-step therefore maximises over L instead of solving the first-order
# condition for D. L is identified only up to an orthogonal factor, which is
# immaterial: D is what enters the likelihood. Projected gradient ascent with
# backtracking, warm-started at the PSD projection of the unrestricted maximiser,
# which is within 0.006 log-likelihood units of the answer, so few steps are
# needed and no optimisation package is required.
# ---------------------------------------------------------------------------

"""PSD projection of a symmetric matrix, returned as a square root factor."""
function psd_sqrt(D::AbstractMatrix)
    F = eigen(Symmetric((D + D') / 2))
    return F.vectors * Diagonal(sqrt.(max.(F.values, 0.0)))
end

"""Maximise `f` over `L` by L-BFGS with an Armijo backtracking line search.

Written in minimisation form on φ = −f so the signs stay straight: s = ΔL,
y = Δ(∇φ) = −Δg, and a curvature pair is kept only when s'y > 0, which keeps the
implicit inverse Hessian positive definite.

Steepest ascent is not adequate for this θ-step. D has eigenvalues spanning zero
to 3e6, and on the crude-oil data projected gradient ascent runs out at 3000
iterations still 1.9e−3 below the maximum with a gradient norm of 5e−2. L-BFGS
builds the curvature it needs from the last `mem` steps and reaches 1.8e−5 in
185 objective evaluations instead of 6008. `fg` returns (f, ∇f)."""
function lbfgs_max(fg, L0; maxiter=400, gtol=1e-7, ftol=1e-13, mem=20)
    L = copy(L0)
    f, g = fg(L)
    isfinite(f) || return (L, f, g)
    S = Matrix{Float64}[]; Yv = Matrix{Float64}[]; ρ = Float64[]
    prev = f
    for _ in 1:maxiter
        norm(g) < gtol && break
        q = -g                                        # ∇φ
        k = length(S); α = zeros(k)
        for i in k:-1:1
            α[i] = ρ[i] * dot(S[i], q); q = q - α[i] * Yv[i]
        end
        q = q * (k == 0 ? 1.0 / max(norm(g), 1.0) :
                          dot(S[k], Yv[k]) / dot(Yv[k], Yv[k]))
        for i in 1:k
            β = ρ[i] * dot(Yv[i], q); q = q + (α[i] - β) * S[i]
        end
        d = -q                                        # ascent direction for f
        slope = dot(d, g)
        if slope <= 0                                 # not an ascent step
            d = copy(g); slope = dot(d, g)
        end
        s = 1.0; ok = false; Ln = L; fn = f; gn = g
        for _ls in 1:40
            Ln = L + s * d
            fn, gn = fg(Ln)
            if isfinite(fn) && fn >= f + 1e-4 * s * slope
                ok = true; break
            end
            s *= 0.5
        end
        ok || break
        sV = Ln - L; yV = -(gn - g); sy = dot(sV, yV)
        L, f, g = Ln, fn, gn
        if sy > 1e-12 * norm(sV) * norm(yV)
            push!(S, sV); push!(Yv, yV); push!(ρ, 1.0 / sy)
            if length(S) > mem
                popfirst!(S); popfirst!(Yv); popfirst!(ρ)
            end
        end
        abs(f - prev) <= ftol * max(1.0, abs(f)) && break
        prev = f
    end
    return (L, f, g)
end

function theta_arch_chol(E, ctx)
    T, p = size(E)
    Xex = ctx[:arch_exog]; q = size(Xex[1], 2); m = p + q
    if !haskey(ctx, :Qtp)
        ctx[:Qtp] = [hcat(Matrix{Float64}(I, p, p), Xex[t]) for t in 1:T]
    end
    Qtp = ctx[:Qtp]
    RHS = zeros(m, m)
    for t in 1:T
        ve = Qtp[t]' * E[t, :]; RHS .+= ve * ve'
    end

    # Scratch for `fg`, allocated once per θ-step and captured by the closure.
    # These must not be module-level: the bootstrap calls this from several
    # threads at once, and shared buffers would be a data race.
    QD = zeros(p, m); Wb = zeros(p, p); Wc = zeros(p, p)
    Sb = zeros(p, m); Gb = zeros(m, m); Gzero = zeros(m, m)

    function fg(L)                                   # objective in L and dℓ/dL
        D = L * L'
        ld = 0.0; quad = 0.0
        fill!(Gb, 0.0)
        @inbounds for t in 1:T
            Qt = Qtp[t]
            mul!(QD, Qt, D)                          # Qₜ'D
            mul!(Wb, QD, Qt')                        # Wₜ = Qₜ'DQₜ
            copyto!(Wc, Wb)                          # cholesky! destroys it
            F = cholesky!(Symmetric(Wc); check=false)
            issuccess(F) || return (-Inf, Gzero)
            ld += logdet(F)
            e = view(E, t, :)
            quad += dot(e, Wb, e)                    # e'Wₜe, no temporaries
            copyto!(Sb, Qt)
            ldiv!(F, Sb)                             # Sb = Wₜ⁻¹Qₜ'
            mul!(Gb, Qt', Sb, 1.0, 1.0)              # G += Qₜ(Qₜ'DQₜ)⁻¹Qₜ'
        end
        Gb .-= RHS
        Gs = (Gb + Gb') / 2                          # dℓ/dD
        return (ld - quad, 2.0 * (Gs * L))           # dℓ/dL = 2 (dℓ/dD) L
    end

    L = haskey(ctx, :L) ? copy(ctx[:L]) :
        psd_sqrt(get(ctx, :D, nothing) === nothing ?
                 begin
                     D0 = zeros(m, m); D0[1:p, 1:p] = inv(_sym(E' * E / T))
                     D0 + 1e-3 * Matrix{Float64}(I, m, m)
                 end : ctx[:D])

    isfinite(fg(L)[1]) || (L = Matrix{Float64}(I, m, m))
    L, _, _ = lbfgs_max(fg, L; maxiter=get(ctx, :arch_chol_maxiter, 400))

    D = L * L'
    ctx[:L] = L; ctx[:D] = D                         # warm-start next θ-step
    W = [Qtp[t] * D * Qtp[t]' for t in 1:T]
    return Dict{Symbol,Any}(:W => W,
                            :logdetS => -sum(safe_logdet(W[t])[1] for t in 1:T),
                            :theta => Dict(:D => D, :L => L), :engine => :blockdiag)
end

function theta_step(model, E, ctx)
    model === :iid    ? theta_iid(E, ctx)    :
    model === :regime ? theta_regime(E, ctx) :
    model === :arch   ? theta_arch(E, ctx)   :
    model === :archchol ? theta_arch_chol(E, ctx) :
    model === :ar1    ? theta_ar1(E, ctx)    :
    model === :fixed  ? theta_fixed(E, ctx)  :
    error("unknown sigma_model $model")
end

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
    # No ridge on S₁₁. On the crude-oil data S₁₁ has an eigenvalue of 2.7e-7, and
    # a 1e-12 ridge, harmless as a starting value, moved the benchmark
    # log-likelihood by 4e-4 and made the exact solution look like a mismatch.
    L = cholesky(Symmetric(S11)).L; Linv = inv(L)
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