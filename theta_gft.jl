# The log-precision covariance model, added to the estimator core.
#
#   Ω(t)⁻¹ = Λ(t) C Λ(t),   Λ(t) = diag(exp(h(t)/2)),   h_i(t) = a_i + Σ_j b_ij V_ijt
#
# with C a correlation matrix carried by the generalized Fisher transformation
# of Archakov and Hansen (2021): C = γ⁻¹(g), where γ(C) = vecl(log C) is a
# bijection from correlation matrices onto R^{p(p−1)/2}. Every (a, b, g) gives a
# positive definite Ω(t) at every driver value, so the model is well defined with
# no cone, no boundary and no constrained optimisation. b = 0 nests the iid case.
#
# The precision rather than the covariance carries the parametrisation, because
# the two mean blocks of the GRRR are GLS steps that need Ω(t)⁻¹, and log|Σ|
# comes out of the same quantities without an inversion.
#
# For p = 4 and q = 1 there are 14 free parameters, against 15 for D = LL'.
#
# The one delicate piece is dℓ/dg. C = exp(A) with A = Off(g) + diag(x), and x is
# fixed implicitly by diag(C) = 1, so the constraint has to be differentiated too.
# `grad_g` does that in closed form; `grad_g_fd` is the finite-difference version
# kept as the thing it is checked against, and `check_gft_gradient` runs the
# check. Run that once on any machine before trusting the fits.

# ---------------------------------------------------------------------------
# the transformation
# ---------------------------------------------------------------------------

"""Strictly lower-triangular elements of A, row by row: the same order as the
Python `vecl`, so the two implementations index g identically."""
function vecl(A::AbstractMatrix)
    p = size(A, 1)
    out = Float64[]
    sizehint!(out, p * (p - 1) ÷ 2)
    for i in 2:p, j in 1:(i - 1)
        push!(out, A[i, j])
    end
    return out
end

"""Symmetric matrix with zero diagonal whose vecl is g."""
function unvecl(g::AbstractVector, p::Int)
    A = zeros(p, p)
    k = 0
    for i in 2:p, j in 1:(i - 1)
        k += 1
        A[i, j] = g[k]; A[j, i] = g[k]
    end
    return A
end

"""exp of a symmetric matrix, returning the eigendecomposition it came from,
because the gradient needs it.

The finiteness check is not defensive programming for its own sake. LAPACK's
symmetric eigensolver is not required to behave on non-finite input and can take
the process down with it rather than raising; a bootstrap explores far worse
parameter values than a single fit ever does, so this path is reached in
practice. Raising here turns a crash into an infeasible point, which is what the
caller already knows how to handle."""
function sym_expm(A::AbstractMatrix)
    all(isfinite, A) ||
        throw(DomainError(:sym_expm, "non-finite matrix passed to the eigensolver"))
    F = eigen(Symmetric(_sym(A)))
    λ = F.values; U = F.vectors
    return (U * Diagonal(exp.(λ)) * U', λ, U)
end

"""γ = vecl(log C) for a positive definite correlation matrix C."""
function gamma_of_C(C::AbstractMatrix)
    F = eigen(Symmetric(_sym(C)))
    minimum(F.values) > 0 || error("C is not positive definite")
    return vecl(F.vectors * Diagonal(log.(F.values)) * F.vectors')
end

"""Inverse transformation. Returns (C, x, λ, U), where x is the diagonal of
log C and (λ, U) its eigendecomposition, both reused by the caller.

x0 warm-starts the fixed point, which matters inside an optimiser: from a nearby
start it converges in a few steps."""
function C_of_gamma(g::AbstractVector, p::Int; x0=nothing, tol=1e-12, maxiter=200)
    all(isfinite, g) || throw(DomainError(:C_of_gamma, "non-finite gamma"))
    Off = unvecl(g, p)
    x = x0 === nothing ? zeros(p) : copy(collect(float.(x0)))
    all(isfinite, x) || (x = zeros(p))          # a poisoned warm start is not fatal
    C = Matrix{Float64}(I, p, p); λ = zeros(p); U = Matrix{Float64}(I, p, p)
    xlast = copy(x)
    for _ in 1:maxiter
        C, λ, U = sym_expm(Off + Diagonal(x))
        d = diag(C)
        # exp of a symmetric matrix has a positive diagonal in exact arithmetic;
        # if overflow has made one non-positive or non-finite, the next log would
        # poison x, so stop at the last good iterate instead
        if !(all(isfinite, d) && all(>(0.0), d))
            x = xlast
            break
        end
        all(abs.(d .- 1.0) .< tol) && break
        xlast = copy(x)
        x = x .- log.(d)
    end
    C, λ, U = sym_expm(Off + Diagonal(x))
    dg = diag(C)
    (all(isfinite, dg) && all(>(0.0), dg)) ||
        throw(DomainError(:C_of_gamma, "the fixed point did not reach a valid C"))
    s = 1.0 ./ sqrt.(dg)                # one exact rescaling to a unit diagonal
    C = _sym((s * s') .* C)
    return (C, x, λ, U)
end

# ---------------------------------------------------------------------------
# objective and gradient
# ---------------------------------------------------------------------------

# b is stored row by row, matching numpy's C-order `reshape(p, q)` and
# `ravel()`. Julia reshapes column first, so the transpose is not cosmetic: with
# q > 1 the two implementations would silently disagree without it. The
# empirical model has q = 1, where the orders coincide, which is exactly why
# this is worth pinning down here rather than discovering later.
unpack_gft(z, p, q) = (z[1:p],
                       permutedims(reshape(z[p + 1:p + p * q], q, p)),
                       z[p + p * q + 1:end])

vec_rowmajor(A) = vec(permutedims(A))

"""ℓ and its pieces at z = (a, b, g). ℓ here is −log|Σ| − ε'Σ⁻¹ε, the same
convention `theta_arch_chol` uses, so the caller can maximise it directly.
Returns `nothing` if C comes back indefinite."""
function gft_pieces(z, p, q, Xex, E, x0)
    all(isfinite, z) || return nothing
    a, b, g = unpack_gft(z, p, q)
    local C, x, λ, U
    try
        C, x, λ, U = C_of_gamma(g, p; x0=x0)
    catch err
        err isa DomainError || err isa ArgumentError ||
            err isa LinearAlgebra.LAPACKException ||
            err isa LinearAlgebra.SingularException || rethrow()
        return nothing                  # an infeasible point, not a failure
    end
    T = size(E, 1)

    h = Matrix{Float64}(undef, T, p)
    @inbounds for t in 1:T, i in 1:p
        s = a[i]
        for j in 1:q
            s += Xex[t][i, j] * b[i, j]
        end
        h[t, i] = s
    end

    u = exp.(0.5 .* h) .* E                              # T x p
    Cu = u * C
    S = u' * u
    ldC, ok = safe_logdet(C)
    ok || return nothing

    f = sum(h) + T * ldC - sum(u .* Cu)
    isfinite(f) || return nothing

    r = 1.0 .- u .* Cu                                   # ∂f/∂h
    ga = vec(sum(r, dims=1))
    gb = zeros(p, q)
    @inbounds for t in 1:T, i in 1:p, j in 1:q
        gb[i, j] += Xex[t][i, j] * r[t, i]
    end
    G_C = T * inv(Symmetric(C)) - S                      # ∂f/∂C
    return (f=f, ga=ga, gb=gb, G_C=Matrix(G_C), C=C, x=x, h=h, ldC=ldC, λ=λ, U=U)
end

"""∂f/∂g exactly, from the Frechet derivative of the matrix exponential.

C = exp(A), A = Off(g) + diag(x), with x fixed implicitly by diag(C) = 1. For
symmetric A = U diag(λ) U' the derivative is Φ(H) = U (F ∘ (U'HU)) U' with F the
divided differences of exp, and Φ is self-adjoint. Differentiating the
constraint gives dx; eliminating it through the adjoint leaves

    ∂f/∂g_k = 2 (M − U K U')_ij,   M = Φ(G_C),
    K = F ∘ (U' diag(y) U),        J' y = diag(M),
    J_lm = Σ_ab F_ab U_la U_lb U_ma U_mb.

The eigendecomposition is already in hand, so this is O(p⁴) rather than p(p−1)
fixed-point solves."""
function grad_g(p, G_C, λ, U)
    e = exp.(λ)
    F = Matrix{Float64}(undef, p, p)
    @inbounds for i in 1:p, j in 1:p
        dl = λ[i] - λ[j]
        F[i, j] = abs(dl) < 1e-10 ? exp((λ[i] + λ[j]) / 2) : (e[i] - e[j]) / dl
    end
    Φ(H) = U * (F .* (U' * H * U)) * U'
    M = Φ(_sym(G_C))

    J = Matrix{Float64}(undef, p, p)
    @inbounds for l in 1:p, m in 1:p
        s = 0.0
        for a in 1:p, b in 1:p
            s += U[l, a] * U[m, a] * F[a, b] * U[l, b] * U[m, b]
        end
        J[l, m] = s
    end
    y = J' \ diag(M)
    K = F .* (U' * Diagonal(y) * U)
    Gmat = 2.0 .* (M - U * K * U')
    return vecl(Gmat)
end

"""∂f/∂g by central differences on C alone, the reference `grad_g` is checked
against. Costs 2p(p−1)/2 fixed-point solves, so it is for testing only."""
function grad_g_fd(g, p, G_C, x; eps=1e-6)
    out = similar(collect(float.(g)))
    for k in eachindex(g)
        gp = collect(float.(g)); gp[k] += eps
        gm = collect(float.(g)); gm[k] -= eps
        Cp, _, _, _ = C_of_gamma(gp, p; x0=x)
        Cm, _, _, _ = C_of_gamma(gm, p; x0=x)
        out[k] = sum(G_C .* (Cp .- Cm)) / (2 * eps)
    end
    return out
end

# ---------------------------------------------------------------------------
# L-BFGS on a vector
# ---------------------------------------------------------------------------
# The core's `lbfgs_max` stores its history as Matrix{Float64}, since the
# Cholesky model optimises over a matrix L. This is the same algorithm with
# vector storage. The core is left untouched on purpose: it is the estimator the
# earlier results were computed with, byte for byte.

function lbfgs_max_vec(fg, z0::Vector{Float64};
                       maxiter=500, gtol=1e-8, ftol=1e-14, mem=20)
    z = copy(z0)
    f, g = fg(z)
    isfinite(f) || return (z, f, g)
    S = Vector{Float64}[]; Yv = Vector{Float64}[]; ρ = Float64[]
    prev = f
    for _ in 1:maxiter
        norm(g) < gtol && break
        q = -g
        k = length(S); α = zeros(k)
        for i in k:-1:1
            α[i] = ρ[i] * dot(S[i], q); q = q - α[i] * Yv[i]
        end
        q = q * (k == 0 ? 1.0 / max(norm(g), 1.0) :
                          dot(S[k], Yv[k]) / dot(Yv[k], Yv[k]))
        for i in 1:k
            β = ρ[i] * dot(Yv[i], q); q = q + (α[i] - β) * S[i]
        end
        d = -q
        slope = dot(d, g)
        if slope <= 0
            d = copy(g); slope = dot(d, g)
        end
        s = 1.0; ok = false; zn = z; fn = f; gn = g
        for _ls in 1:40
            zn = z + s * d
            fn, gn = fg(zn)
            if isfinite(fn) && fn >= f + 1e-4 * s * slope
                ok = true; break
            end
            s *= 0.5
        end
        ok || break
        sV = zn - z; yV = -(gn - g); sy = dot(sV, yV)
        z, f, g = zn, fn, gn
        if sy > 1e-12 * norm(sV) * norm(yV)
            push!(S, sV); push!(Yv, yV); push!(ρ, 1.0 / sy)
            if length(S) > mem
                popfirst!(S); popfirst!(Yv); popfirst!(ρ)
            end
        end
        abs(f - prev) <= ftol * max(1.0, abs(f)) && break
        prev = f
    end
    return (z, f, g)
end

# ---------------------------------------------------------------------------
# the θ-step
# ---------------------------------------------------------------------------

"""iid fit as the starting value: a_i = log W_ii, b = 0, C the correlation
matrix of the iid precision."""
function gft_start(E, p, q)
    T = size(E, 1)
    W = inv(_sym(E' * E / T))
    d = sqrt.(diag(W))
    C = _sym(W ./ (d * d'))
    for i in 1:p
        C[i, i] = 1.0
    end
    return vcat(2.0 .* log.(d), zeros(p * q), gamma_of_C(C))
end

function theta_arch_gft(E, ctx)
    T, p = size(E)
    Xex = ctx[:arch_exog]; q = size(Xex[1], 2)
    nz = p + p * q + p * (p - 1) ÷ 2
    z0 = get(ctx, :gft_z, nothing)
    if z0 === nothing || length(z0) != nz
        z0 = gft_start(E, p, q)
    else
        z0 = copy(z0)
    end
    xstate = Ref{Any}(get(ctx, :gft_x, nothing))

    function fg(z)
        out = gft_pieces(z, p, q, Xex, E, xstate[])
        out === nothing && return (-Inf, zeros(length(z)))
        xstate[] = out.x
        gg = grad_g(p, out.G_C, out.λ, out.U)
        return (out.f, vcat(out.ga, vec_rowmajor(out.gb), gg))
    end

    z, f, _ = lbfgs_max_vec(fg, collect(float.(z0));
                            maxiter=get(ctx, :gft_maxiter, 800))
    isfinite(f) || (z = collect(float.(z0)))

    a, b, g = unpack_gft(z, p, q)
    local C, x
    try
        C, x, _, _ = C_of_gamma(g, p; x0=xstate[])
    catch
        z = collect(float.(z0))         # fall back to the start, which was valid
        a, b, g = unpack_gft(z, p, q)
        C, x, _, _ = C_of_gamma(g, p)
    end
    h = Matrix{Float64}(undef, T, p)
    @inbounds for t in 1:T, i in 1:p
        s = a[i]
        for j in 1:q
            s += Xex[t][i, j] * b[i, j]
        end
        h[t, i] = s
    end
    λt = exp.(0.5 .* h)
    W = [Diagonal(λt[t, :]) * C * Diagonal(λt[t, :]) for t in 1:T]
    ldC = logdet_pd(C)
    logdetS = -(sum(h) + T * ldC)

    ctx[:gft_z] = z; ctx[:gft_x] = x
    return Dict{Symbol,Any}(:W => W, :logdetS => logdetS, :engine => :blockdiag,
                            :theta => Dict(:a => a, :b => b, :g => g,
                                           :C => C, :z => z))
end

# ---------------------------------------------------------------------------
# gradient check
# ---------------------------------------------------------------------------

"""Analytic dℓ/dg against central differences, at a random point of the
parameter space. The analytic version eliminates an implicit constraint, so it
is the one piece of this file that is easy to get subtly wrong; run this once on
any machine before trusting the fits. Returns the largest relative discrepancy,
which should be around 1e-7 with the default step."""
function check_gft_gradient(; p=4, q=1, T=200, seed=20240101, eps=1e-6)
    rng = MersenneTwister(seed)
    E = randn(rng, T, p)
    Xex = [reshape(fill(0.5 + abs(randn(rng)), p), p, q) for _ in 1:T]
    z = gft_start(E, p, q) .+ 0.05 .* randn(rng, p + p * q + p * (p - 1) ÷ 2)
    out = gft_pieces(z, p, q, Xex, E, nothing)
    out === nothing && error("gft_pieces failed at the test point")
    g = z[p + p * q + 1:end]
    ga = grad_g(p, out.G_C, out.λ, out.U)
    gf = grad_g_fd(g, p, out.G_C, out.x; eps=eps)
    rel = maximum(abs.(ga .- gf) ./ max.(abs.(gf), 1.0))
    @printf("gft gradient check: max relative error %.3e over %d components\n",
            rel, length(ga))
    return rel
end

# Extend the dispatch chain. As in theta_scale.jl, this redefines `theta_step`
# rather than editing the core.
function theta_step(model, E, ctx)
    model === :iid      ? theta_iid(E, ctx)       :
    model === :regime   ? theta_regime(E, ctx)    :
    model === :arch     ? theta_arch(E, ctx)      :
    model === :archchol ? theta_arch_chol(E, ctx) :
    model === :scale    ? theta_scale(E, ctx)     :
    model === :archgft  ? theta_arch_gft(E, ctx)  :
    model === :ar1      ? theta_ar1(E, ctx)       :
    model === :fixed    ? theta_fixed(E, ctx)     :
    error("unknown sigma_model $model")
end
