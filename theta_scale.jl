# The scale covariance model, added to the estimator core.
#
#   Ω(t) = g_t Ω,   log g_t = −b (u_t − ū)
#
# with u_t the volatility driver, either the normalised mean absolute return or
# its logarithm. This is the log-precision specification of Section 6 with the p
# loadings restricted to a common value, so it moves the level of volatility and
# leaves the correlation structure fixed.
#
# The θ-step is nearly closed form. Given the scale path, the concentrated
# maximiser is a weighted sample covariance,
#
#   Ω̂(b) = T⁻¹ Σ_t g_t⁻¹ e_t e_t',
#
# and the quadratic form then collapses to Tp exactly, so the profile
# log-likelihood is
#
#   ℓ_p(b) = −½[ Tp log 2π + p Σ_t log g_t + T log|Ω̂(b)| + Tp ].
#
# Centring u so that the geometric mean of g is one kills the middle term, and
# what is left is a monotone function of a determinant. The step is therefore a
# one-dimensional search rather than an inner Newton or quasi-Newton loop, which
# makes this the cheapest of the covariance models by a wide margin.

"""Golden-section minimiser on a bracket, for the one-dimensional profile.

The profile is smooth and unimodal over the bracket used here; golden section is
chosen over anything cleverer because it needs no derivatives and cannot
overshoot, and the θ-step is called hundreds of thousands of times in the
bootstrap."""
function golden_min(f, a, b; tol=1e-10, maxit=300)
    invφ = (sqrt(5.0) - 1) / 2
    c = b - invφ * (b - a); d = a + invφ * (b - a)
    fc = f(c); fd = f(d)
    for _ in 1:maxit
        b - a < tol && break
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invφ * (b - a); fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + invφ * (b - a); fd = f(d)
        end
    end
    return (a + b) / 2
end

function theta_scale(E, ctx)
    T, p = size(E)
    u = ctx[:scale_driver]::Vector{Float64}
    uc = u .- mean(u)                       # geometric mean of g is then one
    lo = get(ctx, :scale_lo, -5.0)
    hi = get(ctx, :scale_hi, 5.0)

    function negprof(b)
        gv = exp.(-b .* uc)
        Ω = _sym((E ./ gv)' * E / T)
        ld, ok = safe_logdet(Ω)
        return ok ? T * ld : 1e12
    end

    b = golden_min(negprof, lo, hi; tol=get(ctx, :scale_tol, 1e-10))
    gv = exp.(-b .* uc)
    Ω = _sym((E ./ gv)' * E / T)
    Wi = inv(Ω)
    W = [Wi ./ gv[t] for t in 1:T]
    logdetS = p * sum(log.(gv)) + T * logdet_pd(Ω)
    return Dict{Symbol,Any}(:W => W, :logdetS => logdetS,
                            :theta => Dict(:Omega => Ω, :b => b),
                            :engine => :blockdiag)
end

# Extend the dispatch chain of the core. Redefining `theta_step` here rather
# than editing the core keeps the extracted estimator byte-identical to the one
# the earlier results were computed with.
function theta_step(model, E, ctx)
    model === :iid      ? theta_iid(E, ctx)       :
    model === :regime   ? theta_regime(E, ctx)    :
    model === :arch     ? theta_arch(E, ctx)      :
    model === :archchol ? theta_arch_chol(E, ctx) :
    model === :scale    ? theta_scale(E, ctx)     :
    model === :ar1      ? theta_ar1(E, ctx)       :
    model === :fixed    ? theta_fixed(E, ctx)     :
    error("unknown sigma_model $model")
end
