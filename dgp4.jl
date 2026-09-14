# The Section 5 simulation design, in Julia.
#
#   Δxₜ = αβ'xₜ₋₁ + εₜ,   p = 4,  r = 3,  one common stochastic trend
#
# β is the spread basis, whose ith column places −bᵢ on the front contract and
# +1 on F_{i+1}, so the true coefficient is bᵢ = 1. The loadings are
# α = β(β'β)⁻¹diag(−κ) with κ = 0.3, giving an error-correction root 0.7 in
# every relation.
#
# Include after emp4.jl: P, R and the basis restriction come from there, so the
# simulation and the empirical application cannot drift apart in the one place
# where a mismatch would be invisible.

# The innovation covariance. Written out rather than regenerated, because it is a
# constant of the design and not a random draw: the Python implementation builds
# it from its own generator, and a Julia generator would give a different matrix
# and therefore a different experiment. Unit variances, since bᵢ is invariant to
# the scale of Ω. Condition number 3.499, smallest pairwise correlation 0.2379.
const OMEGA_DESIGN = [
    1.0                  0.34742707616427654  0.280046266696199    0.2378508426423297;
    0.34742707616427654  1.0                  0.24843639090036893  0.23984648613462053;
    0.280046266696199    0.24843639090036893  1.0                  0.4514359948098515;
    0.2378508426423297   0.23984648613462053  0.4514359948098515   1.0]

# The ARCH-type design. The driver has q = 3 columns, so the precision moves
# along a three-dimensional subspace as the driver moves. κ and the spread of
# the driver together set how far the weights move, which is what the
# experiment is about: with these values the ratio of the ninetieth to the tenth
# percentile of |Ω(t)|^{-1/p} is about 7.6, against exactly 2 for the two-level
# regime design.
#
# D is not fully identified under this design, and an earlier version of this
# comment claimed that q < p made it so. It does not. For any driver of the form
# vₜA with A of full column rank q, the perturbation D₁₂ ↦ D₁₂ + AK with K skew
# leaves every Ω(t)⁻¹ = Qₜ'DQₜ unchanged, so the likelihood is exactly flat
# along a subspace of dimension q(q−1)/2: six under the proportional design
# q = p, three here, and zero only for the scalar driver q = 1 of the empirical
# application. Ω(t), the likelihood and b̂ are identified in every case, which is
# why the experiment, whose object is b̂, is unaffected. `theta_arch` solves its
# Newton system in the minimum-norm sense and records the dimension of the flat
# subspace, and `test_dgp.jl` checks that dimension against q(q−1)/2.
const Q_ARCH = 3
const KAPPA_ARCH = 2.0
const V_LOC, V_SCALE = 0.2, 1.8

"""The scalar driver vₜ of the ARCH-type design, |N(0,1)| shifted so it is
bounded away from zero. Kept here rather than in the experiment scripts so that
the design lives in one place."""
arch_driver(T, rng) = V_LOC .+ V_SCALE .* abs.(randn(rng, T))

"""β at a given b, from the basis restriction of emp4.jl.

vec(β) = Hb + h, and `reshape` fills columns first, which is the order H and h
are built in. This is the one place where a Julia/numpy ordering slip would be
silent, so `test_dgp.jl` compares β entry by entry against the Python matrix."""
beta_of(b) = reshape(H_free * collect(Float64, b) + h_free, P, R)

"""α = β(β'β)⁻¹diag(−κ), so β'α = −diag(κ) and each relation has
error-correction root 1 − κᵢ with one common trend."""
function make_alpha(β, κ)
    k = κ isa Number ? fill(Float64(κ), R) : collect(Float64, κ)
    length(k) == R || error("κ must be a scalar or have length r = $R")
    return β * ((β' * β) \ Diagonal(-k))
end

"""Variance regime: Ω doubles at mid-sample.

Returns (cov, W, logdetΣ, break index). The break is at T ÷ 2, so the first
T ÷ 2 observations carry Ω and the rest carry `factor`Ω, which is the segment
convention `theta_regime` uses for `breaks = [T ÷ 2]`."""
function sigma_regime(T, Ω1; factor=2.0)
    brk = T ÷ 2
    Ω2 = factor * Ω1
    W1 = inv(Ω1); W2 = inv(Ω2)
    cov = [t <= brk ? Ω1 : Ω2 for t in 1:T]
    W = [t <= brk ? W1 : W2 for t in 1:T]
    ld = brk * logdet_pd(Ω1) + (T - brk) * logdet_pd(Ω2)
    return cov, W, ld, brk
end

"""ARCH-type covariance in the paper's class, with a reduced-rank driver.

Ω(t)⁻¹ = Qₜ'DQₜ with Qₜ' = [Iₚ, vₜU] and D = diag(Ω₀⁻¹, κI_q), so

    Ω(t)⁻¹ = Ω₀⁻¹ + κvₜ²UΛ⁻¹U',

where U holds the q leading eigenvectors of Ω₀, the directions carrying most of
the innovation variance, and Λ the matching eigenvalues. Along the jth of them
the precision moves from 1/λⱼ to (1 + κvₜ²)/λⱼ, so κ is scale free and the
time-varying part has rank q.

The sign of each eigenvector is a convention of the eigensolver and differs
between LAPACK builds. It cancels in UΛ⁻¹U' and therefore in Ω(t), and the
model is invariant under Xex ↦ Xex S for orthogonal S, so a sign flip moves D
but not the log-likelihood and not b̂. `test_dgp.jl` compares UΛ⁻¹U' rather than
the factor for that reason.

Returns (cov, W, logdetΣ, arch_exog), the last as a T-vector of p×q matrices,
which is the shape `theta_arch` reads."""
function sigma_arch(T, Ω0, v; q=Q_ARCH, κ=KAPPA_ARCH)
    1 <= q <= P || error("q must satisfy 1 <= q <= p")
    Winv0 = inv(Ω0)
    F = eigen(Symmetric(_sym(Ω0)))                      # ascending
    ord = sortperm(F.values, rev=true)[1:q]
    Uw = F.vectors[:, ord] ./ sqrt.(F.values[ord])'     # p×q, UΛ^{-1/2}
    A = Uw * Uw'
    W = [_sym(Winv0 + κ * v[t]^2 * A) for t in 1:T]
    cov = [inv(W[t]) for t in 1:T]
    ld = -sum(logdet_pd(W[t]) for t in 1:T)
    arch_exog = [v[t] * Uw for t in 1:T]
    return cov, W, ld, arch_exog
end

"""Innovations with a time-varying covariance, one Cholesky factor per period."""
function draw_eps(cov, rng)
    T = length(cov)
    E = zeros(T, P)
    for t in 1:T
        E[t, :] = cholesky(Symmetric(cov[t])).L * randn(rng, P)
    end
    return E
end

"""Δxₜ = αβ'xₜ₋₁ + εₜ. Returns (Δx, xₜ₋₁), both T×p."""
function simulate_vecm(T, β, α, E; x0=nothing)
    Π = α * β'
    x = zeros(T + 1, P)
    x0 === nothing || (x[1, :] = x0)
    for t in 1:T
        x[t + 1, :] = x[t, :] + Π * x[t, :] + E[t, :]
    end
    return x[2:end, :] - x[1:end - 1, :], x[1:end - 1, :]
end

"""The spread of the weights, which is what separates the two designs.

The ratio of the ninetieth to the tenth percentile of |Ω(t)|^{-1/p}: exactly 2
under the regime design, where each observation gets one of two weights, and
about 7.6 under the ARCH design."""
function weight_spread(W)
    p = size(W[1], 1)
    w = [exp(logdet_pd(W[t]) / p) for t in 1:length(W)]
    return quantile(w, 0.90) / quantile(w, 0.10)
end
