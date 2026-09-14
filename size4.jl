# The recursive null experiment of Section 6.2: Table empSize and Figure disc.
#
# Each sample is a genuine draw from the fitted observation-driven model. The
# volatility proxy is rebuilt from the simulated sample's own past returns and
# the innovations are drawn from the covariance that proxy implies:
#
#   v*ₜ  = mean_i |Δx*_{i,t−1}| / m̂
#   ε*ₜ  ~ N(0, Ω*(t)),   Ω*(t) determined by v*ₜ and θ̂
#   Δx*ₜ = Π x*ₜ₋₁ + Γ₁ Δx*ₜ₋₁ + Ψ_d z_{d,t} + ε*ₜ
#
# The mean parameters come from the restricted fit, so the null b = 1 holds in
# the DGP, and θ comes from the free fit, so the covariance is not distorted by
# a restriction the covariance parameters have nothing to do with.
#
# The specifications behave differently under this recursion, and the difference
# is structural rather than numerical.
#
#   quadratic, D = LL'  Ω(t)⁻¹ = Qₜ'DQₜ grows like v² along 1ₚ, so the
#                       conditional standard deviation in that direction falls
#                       like 1/v and the proxy is pulled back towards its mean.
#                       The recursion contracts. What the unrestricted D fails
#                       is propriety, not stability, and D = LL' repairs that.
#
#   log-precision       log Ω(t) is linear in the level of v, so a large draw
#                       feeds a larger variance. The map is locally stable but
#                       has an escape region and, untruncated, some paths run
#                       away. The DGP therefore truncates v* at the observed
#                       maximum; see CAPPED_SPECS below.
#
#   scale               the same log-linear form with one common coefficient
#                       instead of p of them, so the escape region exists but is
#                       entered less often. Truncated in the same way.
#
#   iid and regime      no proxy at all, so the recursion is not recursive and
#                       no path can run away.
#
# With u = log v neither log-linear specification has an escape region, and the
# two are simulated untruncated. Runaway paths, where they can occur, are
# counted rather than hidden.
#
# Include after grrr_core.jl, theta_scale.jl, theta_gft.jl, emp4.jl and boot4.jl.

const SIZE_SPECS = [(:iid,      "iid"),
                    (:regime,   "regime (2008:9, 2020:3)"),
                    (:archchol, "quadratic, D = LL'"),
                    (:scalelog, "scale, u = log v"),
                    (:gftlog,   "log-precision, u = log v"),
                    (:scale,    "scale, u = v (capped)"),
                    (:gft,      "log-precision, u = v (capped)")]

# Python's numbers at M = 400, for comparison rather than for authority. A
# difference of a percentage point at M = 400 is one standard error: the
# simulation standard error of an estimated rejection rate of 0.10 is 1.5
# points, and of 0.05 it is 1.1 points.
const REF_SIZE = Dict(
    :iid      => (0.1125, 0.0750, 0.0275, 400, 0),
    :regime   => (0.1575, 0.0875, 0.0250, 400, 0),
    :archchol => (0.1300, 0.0675, 0.0225, 400, 0),
    :scalelog => (0.1300, 0.0650, 0.0225, 400, 0),
    :gftlog   => (0.1050, 0.0575, 0.0225, 400, 0),
    :scale    => (0.1281, 0.0739, 0.0246, 400, 197),
    :gft      => (0.1261, 0.0762, 0.0235, 400, 59))

# The two specifications whose proxy enters in levels have an escape region:
# log Ω(t) is linear in v, so a large draw feeds a larger variance, and some
# paths run away. Their DGP truncates the simulated proxy at the observed
# maximum, which bounds the conditional variance along every path without
# touching the fit or the estimation of any sample. The other three recursive
# specifications complete without it and are simulated untruncated. Files from
# a truncated run carry the tag `_cap`, so the untruncated runs are kept.
const CAPPED_SPECS = (:scale, :gft)
capped(spec) = spec in CAPPED_SPECS

size_path(spec, M) = joinpath(boot_resultdir(),
                              "size_$(spec)_M$(M)$(capped(spec) ? "_cap" : "").txt")

"""Everything the null DGP needs, assembled once per specification.

`θ` comes from the free fit and the mean parameters from the restricted one, so
the null holds in the DGP while the covariance is the one the data actually
support. `Wfixed` is non-nothing exactly for iid and regime, where the precision
does not depend on the simulated path. `vcap` is the observed maximum of the
proxy for the two capped specifications and `Inf` otherwise."""
function size_source(spec, Y, X, Z, free, one)
    Π, Γ₁, dpath = vecm_parts(one, Z)
    T = size(Y, 1)
    kind = (spec === :scalelog || spec === :gftlog) ? :log : :level
    a = vec(mean(abs.(Z[:, 1:P]), dims=2))
    mhat = mean(a)
    vobs = a ./ mhat
    shift = kind === :log ? mean(log.(vobs)) : 0.0
    uobs = kind === :log ? log.(vobs) .- shift : vobs
    return (spec=spec, θ=free[:theta], Π=Π, Γ₁=Γ₁, dpath=dpath,
            Zdet=Z[:, P + 1:end], mhat=mhat, ubar=mean(uobs),
            kind=kind, shift=shift, T=T,
            Wfixed=fixed_precision(spec, free[:theta], T, BRK),
            vcap=capped(spec) ? maximum(vobs) : Inf,
            ψ_free=psi_of(free), ψ_one=psi_of(one))
end

"""One draw of the recursive experiment.

Returns the row written to disk: LR, the index of the winning restricted start,
the index of the winning free start, the spread across the restricted starts,
and a status code (0 usable, 1 runaway, 2 the fit threw).

Three restricted starts and three free starts. A restricted fit trapped at a low
optimum inflates LR* and would show up as size distortion that is really an
optimiser artifact, which is the one thing this experiment must not do.
Recording which start won says whether the extra starts were needed at all."""
function size_rep(src, x_init, dx_init, seed; tol=1e-8, maxiter=3000)
    rng = Xoshiro(seed)
    out = simulate_null(src.spec, src.θ, src.Π, src.Γ₁, src.dpath, src.Zdet,
                        src.mhat, src.ubar, x_init, dx_init, src.T, rng;
                        kind=src.kind, shift=src.shift, Wfixed=src.Wfixed,
                        vcap=src.vcap)
    out === nothing && return [NaN, NaN, NaN, NaN, 1.0]
    Yj, Xj, Zj = out
    try
        b0, ψ_pin, _ = closed_form_start(Yj, Xj, Zj, src.spec)
        f(free; kw...) = fit_basis(Yj, Xj, Zj, src.spec, free;
                                   tol=tol, maxiter=maxiter, kw...)
        ones_ = [f(false; psi0=src.ψ_one),
                 f(false; psi0=ψ_pin),
                 f(false; psi0=src.ψ_free)]
        lls = [r[:loglik] for r in ones_]
        o = ones_[argmax(lls)]
        frees = [f(true; phi0=ones(R), psi0=src.ψ_free),
                 f(true; phi0=b0, psi0=ψ_pin),
                 f(true; phi0=ones(R), psi0=psi_of(o))]
        fr = frees[argmax([r[:loglik] for r in frees])]
        return [2 * (fr[:loglik] - o[:loglik]),
                Float64(argmax(lls)),
                Float64(argmax([r[:loglik] for r in frees])),
                maximum(lls) - minimum(lls),
                0.0]
    catch err
        err isa InterruptException && rethrow()
        return [NaN, NaN, NaN, NaN, 2.0]
    end
end

"""The recursive experiment for one specification, resumed from and appended to
disk, one line per draw."""
function size_one(spec, label, Y, X, Z, free, one, x_init, dx_init;
                  M=400, seed0=19510104, verbose=true, save=true,
                  tol=1e-8, maxiter=3000)
    src = size_source(spec, Y, X, Z, free, one)
    path = size_path(spec, M)
    t0 = time()
    rows = mc_run(path, 1:M, 5,
                  j -> size_rep(src, x_init, dx_init, seed0 + j;
                                tol=tol, maxiter=maxiter);
                  verbose=verbose, save=save, label="size $label")
    secs = time() - t0
    return size_summary(spec, label, rows, secs; vcap=src.vcap)
end

function size_summary(spec, label, rows, secs; vcap=Inf)
    js = sort(collect(keys(rows)))
    st = [rows[j][5] for j in js]
    LR = [rows[j][1] for j in js if rows[j][5] == 0.0 && isfinite(rows[j][1])]
    pv = ccdf.(Chisq(R), LR)
    won = [rows[j][2] for j in js if rows[j][5] == 0.0]
    spread = [rows[j][4] for j in js if rows[j][5] == 0.0]
    # A specification whose paths all ran away leaves nothing to summarise, and
    # `mean` of an empty vector is an error rather than a missing value. Report
    # NaN and let the runaway count carry the information.
    m(v) = isempty(v) ? NaN : mean(v)
    q(v, u) = isempty(v) ? NaN : quantile(v, u)
    return (spec=spec, label=label, M=length(js), n_used=length(LR),
            n_improper=count(==(1.0), st), n_error=count(==(2.0), st),
            size10=m(pv .<= 0.10), size05=m(pv .<= 0.05), size01=m(pv .<= 0.01),
            q90=q(LR, 0.90), q95=q(LR, 0.95), q99=q(LR, 0.99),
            won_closed_form=m(won .== 2.0),
            spread_median=q(spread, 0.5), spread_max=isempty(spread) ? NaN :
                                                     maximum(spread),
            vcap=vcap, secs=secs)
end

function report_size(res)
    println("\n" * "=" ^ 96)
    println("RECURSIVE NULL EXPERIMENT  (Table empSize, Figure disc)")
    println("=" ^ 96)
    # @printf needs its format as one string literal, not a concatenation.
    @printf("nominal levels 10 / 5 / 1 per cent; chi2(%d) critical values %.2f / %.2f / %.2f\n\n",
            R, quantile(Chisq(R), 0.90), quantile(Chisq(R), 0.95),
            quantile(Chisq(R), 0.99))
    @printf("%-30s%22s%22s%9s%9s\n",
            "specification", "this run  10 / 5 / 1", "Python  10 / 5 / 1",
            "runaway", "usable")
    for r in res
        # Python's numbers are from the untruncated recursion, so they are not
        # comparable for a capped specification and are left blank.
        ref = isfinite(r.vcap) ? nothing : get(REF_SIZE, r.spec, nothing)
        @printf("%-30s%22s%22s%5d/%3d%9d\n", r.label,
                @sprintf("%6.1f%7.1f%7.1f", 100 * r.size10, 100 * r.size05, 100 * r.size01),
                ref === nothing ? "" :
                    @sprintf("%6.1f%7.1f%7.1f", 100 * ref[1], 100 * ref[2], 100 * ref[3]),
                r.n_improper, ref === nothing ? 0 : ref[5], r.n_used)
    end
    for r in res
        isfinite(r.vcap) || continue
        @printf("  %s: simulated proxy truncated at the observed maximum %.3f (mean-one units)\n",
                r.label, r.vcap)
    end
    println("\nLR quantiles of the recursive null distribution")
    @printf("%-30s%10s%10s%10s%10s\n", "specification", "90th", "95th", "99th",
            "time (s)")
    for r in res
        @printf("%-30s%10.2f%10.2f%10.2f%10.0f\n",
                r.label, r.q90, r.q95, r.q99, r.secs)
    end
    println("\nrestricted fit: how often the closed-form start won, and the " *
            "spread across the three starts")
    @printf("%-30s%12s%14s%14s\n", "specification", "closed form",
            "median spread", "max spread")
    for r in res
        @printf("%-30s%11.1f%%%14.4f%14.3f\n", r.label,
                100 * r.won_closed_form, r.spread_median, r.spread_max)
    end
    nerr = sum(r.n_error for r in res)
    nerr > 0 && @printf("\n%d draw(s) threw during estimation and were dropped.\n", nerr)
    println("\nThe LR* values are on disk one per line in results4/size_<spec>_M<M>.txt,")
    println("which is what Figure disc is drawn from.")
end
