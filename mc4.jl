# Section 5 of the paper: Experiments B, C and D.
#
#   B   efficiency of b̂ᵢ, modelling Σ(θ) against ignoring it        Table simB
#   C   multi-start robustness on simulated data                     Table simC
#   D   starting values and the conditioning of Ω                    Table simD
#
# Include after grrr_core.jl, theta_scale.jl, theta_gft.jl, emp4.jl, boot4.jl
# and dgp4.jl. Experiment D needs the observed sample, which is why emp4.jl is
# in that list; B and C are pure simulation.
#
# The two implementations cannot draw the same random numbers. numpy's default
# generator is PCG64 with a ziggurat normal and Julia's is Xoshiro256++, so the
# streams differ and no seed makes them agree. What the two can be held to is
# agreement to within simulation error, which at 500 replications is about three
# per cent on a root mean squared error. The one comparison that is exact is in
# `test_dgp.jl`: with the innovations supplied rather than drawn, the three
# estimators must return the same b̂ to the last digit either implementation can
# claim. Run that test before this file.
#
# Replication j depends only on its own seed, so a resumed run reproduces
# exactly what an uninterrupted one would have produced, and the order in which
# threads finish does not matter.

const KAPPA_B = 0.3                     # error-correction root 1 − κ = 0.7
const TS_B = [100, 200, 400]
const ESTIMATORS = (:iid, :correct, :infeasible)

# The Python numbers at 500 replications, as (RMSE, median absolute error). They
# are for comparison and not for authority: the two runs draw different random
# numbers, so agreement is expected only to within simulation error, which at
# 500 replications is about three per cent of a root mean squared error.
const REF_B = Dict(
    (:regime, 100, :iid)        => (0.201507, 0.080171),
    (:regime, 100, :correct)    => (0.192268, 0.076097),
    (:regime, 100, :infeasible) => (0.219678, 0.075424),
    (:regime, 200, :iid)        => (0.076335, 0.041152),
    (:regime, 200, :correct)    => (0.071867, 0.037693),
    (:regime, 200, :infeasible) => (0.071963, 0.038724),
    (:regime, 400, :iid)        => (0.039590, 0.020263),
    (:regime, 400, :correct)    => (0.037621, 0.019176),
    (:regime, 400, :infeasible) => (0.037537, 0.018739),
    (:arch,   100, :iid)        => (0.194948, 0.081615),
    (:arch,   100, :correct)    => (0.155261, 0.065571),
    (:arch,   100, :infeasible) => (0.134837, 0.060143),
    (:arch,   200, :iid)        => (0.085929, 0.040257),
    (:arch,   200, :correct)    => (0.066121, 0.031289),
    (:arch,   200, :infeasible) => (0.064891, 0.029977),
    (:arch,   400, :iid)        => (0.040808, 0.019972),
    (:arch,   400, :correct)    => (0.030907, 0.014613),
    (:arch,   400, :infeasible) => (0.030285, 0.014212))

# Python, Experiment C at 50 data sets and 20 starts, and Experiment D.
const REF_C = (mean_frac=1.0, min_frac=1.0, all_agree=1.0,
               median_gap=1.559442353027407e-9, max_gap=2.668366505531594e-8)
const REF_D = (data_frac=0.0, data_max_gap=95.494, data_median_gap=75.903,
               data_iters_median=4000, data_hit_maxiter=18,
               data_gap_closed_form=0.0, data_iters_closed_form=2,
               data_gap_vs_eig=3.763069435080979e-4,
               sim_frac=0.0, sim_max_gap=113.476, sim_iters_median=4000,
               orth_frac=0.9166666666666666, orth_max_gap=516.613,
               orth_iters_median=16,
               condition_number=64906.2, min_off_corr=0.9454)

# Section 5 is pure simulation and does not depend on the numeraire, so its
# results live in results4 whatever NUMERAIRE says; only the empirical section
# and the recursive size experiment are keyed by it.
mc_resultdir() = (d = joinpath(@__DIR__, "results4"); mkpath(d); d)

# `mc_load` and `mc_run`, the generic resume-and-append machinery this file uses,
# live in boot4.jl next to the bootstrap's own version of them.

# ---------------------------------------------------------------------------
# Experiment B: what is gained by modelling Σ(θ)
# ---------------------------------------------------------------------------
# Three estimators of the spread coefficients: an iid-Σ GRRR that ignores the
# covariance dynamics, a GRRR with the correctly specified Σ(θ), and an
# infeasible GLS estimator that is handed the true Σ. The first supplies the
# starting values for the other two, so the comparison is not confounded by
# where each fit began.

"""One replication of Experiment B. Returns the row written to disk:

    e_iid(3) e_correct(3) e_infeasible(3) σ_min(α̂) ‖α̂₁‖ nonconv(3) iters
    winning start(3) spread across starts(3) nullity

Every estimator is fitted from three starting values and the highest likelihood
is kept. This is the protocol the rest of the paper uses, and Experiment B needs
it for the same reason: on a draw where α̂ is nearly rank r − 1 a single start
can return a point that is not the maximiser. Refitting the six worst draws of a
twenty-thousand replication run from twenty dispersed starts moved the largest
absolute error from 322 to 3.7 while raising the log-likelihood by up to 2.2, so
the single-start estimate was not the maximum likelihood estimate at all.

The three starts use no knowledge a practitioner would not have. For the iid
fit they are b = 1 with the loadings from the closed-form regression under
b = 1, the eigenvalue reduced rank regression, and b = 1 with the loadings of
that regression; for the two other fits the iid fit's own estimate replaces the
first of these. An earlier version started the iid fit at b = 1 with the true
loadings, which is a start at the truth and was replaced; away from T = 100 the
three starts agree on every draw, so the change is invisible there.

σ_min(α̂) is stored because it explains what remains of the tail. When α̂ is close
to rank r − 1 the last cointegrating direction is weakly identified and b̂ can be
far from one whatever covariance is assumed and whatever start is used, so a
handful of draws dominate any sum of squares. ‖α̂₁‖, the norm of the front
contract's row of loadings, is stored to test the rival explanation."""
function mcB_rep(kind::Symbol, T::Int, s::Int, β, α, Ω1, b_true)
    rng = Xoshiro(20000 + s)
    local covs, W, ld, ctx, model, tol_c, mx_c
    if kind === :regime
        covs, W, ld, _ = sigma_regime(T, Ω1; factor=2.0)
        ctx = Dict{Symbol,Any}(:breaks => [T ÷ 2])
        model = :regime
        tol_c, mx_c = 1e-9, 2000
    else
        v = arch_driver(T, rng)
        covs, W, ld, ax = sigma_arch(T, Ω1, v)
        ctx = Dict{Symbol,Any}(:arch_exog => ax, :arch_max_inner => 30)
        model = :arch
        # 500 was not enough: one draw in 500 at T = 100 stopped at the cap and
        # the reported estimate was a truncated iterate rather than a maximiser.
        tol_c, mx_c = 1e-8, 4000
    end
    E = draw_eps(covs, rng)
    Y, X = simulate_vecm(T, β, α, E)

    one_fit(mdl, cx, phi0, psi0; tol=1e-9, maxiter=2000) =
        grrr_estimate(Y, X, nothing, R; H=H_free, h=h_free, sigma_model=mdl,
                      sigma_ctx=cx, phi0=phi0, psi0=psi0,
                      tol=tol, maxiter=maxiter, normalize_beta=false)

    # the eigenvalue reduced rank regression, and the b it implies. The null
    # direction of β' is proportional to (1, b₁, b₂, b₃) under the spread basis,
    # so b falls out of it; `full=true` is needed because the thin SVD of a 3×4
    # matrix omits that direction.
    jr = johansen_rrr(Y, X, nothing, R)
    Fj = svd(Matrix(jr[:beta]'); full=true)
    w = Fj.Vt[end, :]
    b_rrr = w[2:end] ./ w[1]
    ψ_rrr = vec(jr[:alpha])

    # b = 1 with the loadings from the closed-form regression of ΔX on the
    # b = 1 spreads: the restricted-value start, built from the data alone.
    S1 = X * beta_of(ones(R))
    ψ_one = vec((Y' * S1) / (S1' * S1))
    r0s = [one_fit(:iid, nothing, ones(R), ψ_one),
           one_fit(:iid, nothing, b_rrr, ψ_rrr),
           one_fit(:iid, nothing, ones(R), ψ_rrr)]
    r0, w_i, sp_i = best_fit(r0s)

    starts = ((-r0[:beta][1, :], vec(r0[:alpha])),
              (b_rrr, ψ_rrr),
              (ones(R), vec(r0[:alpha])))
    rcs = [one_fit(model, ctx, p, q; tol=tol_c, maxiter=mx_c) for (p, q) in starts]
    rc, w_c, sp_c = best_fit(rcs)
    cf = Dict{Symbol,Any}(:W => W, :logdetS => ld)
    rfs = [one_fit(:fixed, cf, p, q) for (p, q) in starts]
    rf, w_f, sp_f = best_fit(rfs)

    e_i = -r0[:beta][1, :] .- b_true
    e_c = -rc[:beta][1, :] .- b_true
    e_f = -rf[:beta][1, :] .- b_true
    smin = svdvals(rc[:alpha])[end]
    afront = norm(rc[:alpha][1, :])
    return vcat(e_i, e_c, e_f, smin, afront,
                Float64(!r0[:converged]), Float64(!rc[:converged]),
                Float64(!rf[:converged]), Float64(rc[:iters]),
                Float64(w_i), Float64(w_c), Float64(w_f), sp_i, sp_c, sp_f,
                Float64(get(rc[:theta], :nullity, -1)))
end

"""The fit with the highest log-likelihood, which start it came from, and the
spread across the starts. A large spread is the signal that one start alone
would have reported something other than the maximum."""
function best_fit(cands)
    ll = [isfinite(r[:loglik]) ? r[:loglik] : -Inf for r in cands]
    k = argmax(ll)
    return cands[k], k, maximum(ll) - minimum(ll)
end

"""Spearman rank correlation. No package: `sortperm` twice gives the ranks, and
ties have probability zero among these statistics."""
function spearman(x, y)
    n = length(x)
    rx = zeros(n); ry = zeros(n)
    rx[sortperm(x)] = 1:n
    ry[sortperm(y)] = 1:n
    return cor(rx, ry)
end

"""The summary of one cell of Experiment B, as a named tuple per estimator plus
the diagnostics the paper quotes about the tail."""
function mcB_cell(rows)
    js = sort(collect(keys(rows)))
    M = length(js)
    A = permutedims(reduce(hcat, [rows[j] for j in js]))   # M × 21
    smin = A[:, 10]; afront = A[:, 11]
    out = Dict{Symbol,Any}()
    for (k, nm) in enumerate(ESTIMATORS)
        e = A[:, 3k - 2:3k]
        worst = sortperm(vec(maximum(abs.(e), dims=2)), rev=true)
        keep = setdiff(1:M, worst[1:min(3, M)])
        ss = vec(sum(e .^ 2, dims=2))
        dec = sortperm(smin)[1:max(1, M ÷ 10)]          # lowest decile of σ_min
        keep5 = setdiff(1:M, worst[1:min(5, M)])
        out[nm] = (rmse=sqrt(mean(e .^ 2)),
                   mae=median(abs.(e)),
                   bias=mean(e),
                   rmse_excl3=sqrt(mean(e[keep, :] .^ 2)),
                   rmse_excl5=sqrt(mean(e[keep5, :] .^ 2)),
                   nonconv=Int(sum(A[:, 11 + k])),
                   smin_worst3=median(smin[worst[1:min(3, M)]]),
                   smin_median=median(smin),
                   ss_top5=sum(ss[worst[1:min(5, M)]]) / sum(ss),
                   ss_low_decile=sum(ss[dec]) / sum(ss),
                   rho_afront=spearman(afront, vec(maximum(abs.(e), dims=2))),
                   # How often a start other than the first won by a margin
                   # that means anything. Without the margin this counts ties:
                   # away from T = 100 the three starts reach the same optimum
                   # and differ only in the last bits, so `argmax` picks among
                   # them at random and the bare count reads about a half.
                   other_start_won=mean((A[:, 15 + k] .!= 1.0) .&
                                        (A[:, 18 + k] .> 1e-6)),
                   spread_median=median(A[:, 18 + k]),
                   spread_max=maximum(A[:, 18 + k]))
    end
    out[:M] = M
    out[:iters_median] = median(A[:, 15])
    # the dimension of the flat subspace theta_arch found, which is q(q−1)/2 = 3
    # under the ARCH design and irrelevant (recorded as −1) under the regime one
    out[:nullity_min] = minimum(A[:, 22]); out[:nullity_max] = maximum(A[:, 22])
    return out
end

function experiment_B(; reps=500, save=true, verbose=true)
    β = beta_of(ones(R))
    α = make_alpha(β, KAPPA_B)
    Ω1 = OMEGA_DESIGN
    b_true = ones(R)
    cells = Dict{Tuple{Symbol,Int},Any}()
    for kind in (:regime, :arch), T in TS_B
        # `ms` in the name: the multi-start protocol. Files written by the
        # earlier single-start version are left alone rather than resumed from,
        # because they are not draws from the same procedure.
        path = joinpath(mc_resultdir(), "mcB_$(kind)_T$(T)_M$(reps)_ms3.txt")
        t0 = time()
        rows = mc_run(path, 1:reps, 22,
                      j -> mcB_rep(kind, T, j, β, α, Ω1, b_true);
                      verbose=verbose, save=save, label="B $kind T=$T")
        cells[(kind, T)] = mcB_cell(rows)
        verbose && @printf("  B  %-6s T=%3d  %5.0f s\n", string(kind), T, time() - t0)
        flush(stdout)
    end
    return cells
end

function report_B(cells)
    println("\n" * "=" ^ 78)
    println("EXPERIMENT B: efficiency of b̂ᵢ  (Table simB)")
    println("=" ^ 78)
    println("Median absolute error, with the root mean squared error in brackets.")
    println("The median is the headline because the RMSE is not estimable at T = 100:")
    println("b̂ has tails heavy enough there that the sample second moment is carried")
    println("by single draws and does not settle as the replication count grows. At")
    println("T = 200 and T = 400 both statistics are stable and tell the same story.")
    for (lab, kind) in (("ARCH-type covariance", :arch),
                        ("regime covariance", :regime))
        @printf("\n  %s\n", lab)
        @printf("  %-14s %14s %14s %14s\n", "estimator", "T=100", "T=200", "T=400")
        for nm in ESTIMATORS
            @printf("  %-14s", string(nm))
            for T in TS_B
                c = cells[(kind, T)][nm]
                @printf(" %8.4f(%.4f)", c.mae, c.rmse)
            end
            println()
        end
        for (tag, key) in (("MAE gain, correct   %", :correct),
                           ("MAE gain, infeas.   %", :infeasible))
            @printf("  %-22s", tag)
            for T in TS_B
                base = cells[(kind, T)][:iid].mae
                @printf(" %8.1f", 100 * (1 - cells[(kind, T)][key].mae / base))
            end
            println()
        end
        # what the paper's sentence is about: how much of the gain that knowing
        # Σ would buy the feasible estimator actually recovers
        @printf("  %-22s", "share of it recovered %")
        for T in TS_B
            base = cells[(kind, T)][:iid].mae
            gc = 1 - cells[(kind, T)][:correct].mae / base
            gf = 1 - cells[(kind, T)][:infeasible].mae / base
            @printf(" %8.0f", 100 * gc / gf)
        end
        println()
        for (tag, key) in (("RMSE gain, correct  %", :correct),
                           ("RMSE gain, infeas.  %", :infeasible))
            @printf("  %-22s", tag)
            for T in TS_B
                base = cells[(kind, T)][:iid].rmse
                @printf(" %8.1f", 100 * (1 - cells[(kind, T)][key].rmse / base))
            end
            println()
        end
    end

    # The paper's paragraph about the one entry that does not follow the
    # textbook ordering is about the regime design at T = 100, where the
    # infeasible estimator's RMSE exceeds the iid one's. The explanation offered
    # there is that a handful of draws with a near rank-deficient α̂ carry the
    # sum of squares, and the rival explanation, a small loading on the front
    # contract, is ruled out by a rank correlation of about zero. Both numbers
    # are printed here so the sentence can be checked rather than trusted.
    println("\n  the tail: where the squared error comes from")
    @printf("  %-6s %-4s %-11s %8s %8s %8s %8s %8s %8s %8s\n",
            "design", "T", "estimator", "rmse", "excl 3", "excl 5",
            "5 lgst%", "dec%", "ρ(α̂₁)", "nonconv")
    for kind in (:regime, :arch), T in TS_B, nm in ESTIMATORS
        c = cells[(kind, T)][nm]
        @printf("  %-6s %-4d %-11s %8.4f %8.4f %8.4f %8.1f %8.1f %8.3f %8d\n",
                string(kind), T, string(nm), c.rmse, c.rmse_excl3, c.rmse_excl5,
                100 * c.ss_top5, 100 * c.ss_low_decile, c.rho_afront, c.nonconv)
    end
    println("\n  σ_min(α̂), which is what makes the third direction weakly identified")
    for kind in (:regime, :arch), T in TS_B
        c = cells[(kind, T)][:infeasible]
        # @printf needs its format as one string literal, not a concatenation:
        # given `"a" * "b"` the macro reads the first argument as an IO stream.
        @printf("    %-6s T=%3d  median %6.4f over all draws, %6.4f over the 3 largest errors\n",
                string(kind), T, c.smin_median, c.smin_worst3)
    end
    # Whether the extra starts were needed. A single start returning something
    # other than the maximum is what inflated the earlier T = 100 tail.
    println("\n  multi-start: how often a start other than the first won by more than")
    println("  1e-6, and the spread across the three starts in log-likelihood units.")
    println("  A max spread of zero means the three starts reached the same optimum on")
    println("  every draw, which is what happens away from T = 100.")
    @printf("  %-6s %-4s %-11s %12s %14s %14s\n",
            "design", "T", "estimator", "other won %", "median spread", "max spread")
    for kind in (:regime, :arch), T in TS_B, nm in ESTIMATORS
        c = cells[(kind, T)][nm]
        @printf("  %-6s %-4d %-11s %12.1f %14.2e %14.3f\n",
                string(kind), T, string(nm), 100 * c.other_start_won,
                c.spread_median, c.spread_max)
    end

    @printf("\n  replications per cell: %d;  median iterations %.0f\n",
            cells[(:arch, TS_B[1])][:M], cells[(:arch, TS_B[1])][:iters_median])
    # The flat subspace of D under the ARCH design, as theta_arch found it on
    # every replication: q(q−1)/2 = 3 for q = 3, and it should never vary.
    @printf("  flat subspace of D under the ARCH design, dimension over all draws: T=100 %g..%g, T=200 %g..%g, T=400 %g..%g   (predicted q(q−1)/2 = %d)\n",
            cells[(:arch, 100)][:nullity_min], cells[(:arch, 100)][:nullity_max],
            cells[(:arch, 200)][:nullity_min], cells[(:arch, 200)][:nullity_max],
            cells[(:arch, 400)][:nullity_min], cells[(:arch, 400)][:nullity_max],
            Q_ARCH * (Q_ARCH - 1) ÷ 2)

    println("\n  Python at 500 replications, for comparison. The two runs draw" *
            " different random\n  numbers, so a difference of a few per cent in" *
            " the third digit is simulation\n  error rather than disagreement.")
    @printf("  %-6s %-4s %-11s %9s %9s %9s %9s\n",
            "design", "T", "estimator", "rmse", "python", "mae", "python")
    for kind in (:regime, :arch), T in TS_B, nm in ESTIMATORS
        c = cells[(kind, T)][nm]
        ref = REF_B[(kind, T, nm)]
        @printf("  %-6s %-4d %-11s %9.4f %9.4f %9.4f %9.4f\n",
                string(kind), T, string(nm), c.rmse, ref[1], c.mae, ref[2])
    end
end

# ---------------------------------------------------------------------------
# Experiment C: multi-start robustness on simulated data
# ---------------------------------------------------------------------------
# One design from Experiment B, the regime covariance at T = 200, estimated from
# NSTART dispersed random starts on each of NDATA simulated data sets. This is
# the simulated counterpart of the multi-start protocol applied to the observed
# sample, and the contrast with Experiment D is the conditioning of Ω: well
# conditioned here, 6.5e4 there.

const T_C, NSTART_C = 200, 20

"""One data set of Experiment C: NSTART log-likelihoods, their convergence flags
and their iteration counts, flattened into one row."""
function mcC_rep(dset::Int, β, α, Ω1)
    rng = Xoshiro(30000 + dset)
    covs, _, _, _ = sigma_regime(T_C, Ω1; factor=2.0)
    E = draw_eps(covs, rng)
    Y, X = simulate_vecm(T_C, β, α, E)
    srng = Xoshiro(77 + dset)
    ll = zeros(NSTART_C); cv = zeros(NSTART_C); it = zeros(NSTART_C)
    for st in 1:NSTART_C
        phi0 = 1.0 .+ 2.0 .* randn(srng, R)
        psi0 = randn(srng, P * R)
        r = grrr_estimate(Y, X, nothing, R; H=H_free, h=h_free,
                          sigma_model=:regime,
                          sigma_ctx=Dict{Symbol,Any}(:breaks => [T_C ÷ 2]),
                          phi0=phi0, psi0=psi0, tol=1e-10, maxiter=2000,
                          normalize_beta=false)
        ll[st] = r[:loglik]; cv[st] = Float64(r[:converged]); it[st] = r[:iters]
    end
    return vcat(ll, cv, it)
end

function experiment_C(; ndata=50, save=true, verbose=true)
    β = beta_of(ones(R))
    α = make_alpha(β, KAPPA_B)
    path = joinpath(mc_resultdir(), "mcC_T$(T_C)_N$(ndata).txt")
    t0 = time()
    rows = mc_run(path, 1:ndata, 3 * NSTART_C,
                  j -> mcC_rep(j, β, α, OMEGA_DESIGN);
                  verbose=verbose, save=save, label="C")
    verbose && @printf("  C  %d data sets  %5.0f s\n", ndata, time() - t0)
    return rows
end

function report_C(rows)
    js = sort(collect(keys(rows)))
    frac = Float64[]; gap = Float64[]; allgap = Float64[]
    nconv = 0; its = Float64[]
    for j in js
        r = rows[j]
        ll = r[1:NSTART_C]; cv = r[NSTART_C + 1:2NSTART_C]
        append!(its, r[2NSTART_C + 1:3NSTART_C])
        nconv += count(==(0.0), cv)
        best_all = maximum(ll)
        append!(allgap, best_all .- ll)
        valid = ll[cv .== 1.0]
        isempty(valid) && continue
        best = maximum(valid)
        push!(frac, mean(abs.(valid .- best) .< 1e-6))
        push!(gap, best - minimum(valid))
    end
    println("\n" * "=" ^ 78)
    println("EXPERIMENT C: multi-start robustness, simulated data  (Table simC)")
    println("=" ^ 78)
    @printf("%-44s%12s%14s\n", "", "this run", "Python")
    @printf("  %-42s%12d%14d\n", "data sets", length(js), 50)
    @printf("  %-42s%12d%14d\n", "starts per data set", NSTART_C, 20)
    @printf("  %-42s%12.4f%14.4f\n", "mean fraction of starts at the best (1e-6)",
            mean(frac), REF_C.mean_frac)
    @printf("  %-42s%12.4f%14.4f\n", "minimum fraction over data sets",
            minimum(frac), REF_C.min_frac)
    @printf("  %-42s%12.4f%14.4f\n", "data sets where every start agrees",
            mean(frac .== 1.0), REF_C.all_agree)
    @printf("  %-42s%12.3g%14.3g\n", "median worst-case log-likelihood gap",
            median(gap), REF_C.median_gap)
    @printf("  %-42s%12.3g%14.3g\n", "maximum worst-case gap",
            maximum(gap), REF_C.max_gap)
    @printf("\n  total starts %d;  non-converged (dropped) %d\n",
            length(js) * NSTART_C, nconv)
    @printf("  including non-converged: fraction within 1e-6 of best = %.4f\n",
            mean(allgap .< 1e-6))
    @printf("  max gap over all starts %.4f;  starts with gap > 1e-3: %d\n",
            maximum(allgap), count(>(1e-3), allgap))
    @printf("  iterations: median %.0f, 95th %.0f, max %.0f (maxiter 2000)\n",
            median(its), quantile(its, 0.95), maximum(its))
end

# ---------------------------------------------------------------------------
# Experiment D: starting values and the conditioning of Ω
# ---------------------------------------------------------------------------
# Three arms, all under iid Σ:
#
#   data      the observed sample, with dispersed random starts and the
#             closed-form start
#   sim       samples simulated from the fitted model, so the near-singular
#             fitted Ω is retained
#   sim-orth  the same, with the innovation covariance replaced by a
#             well-conditioned matrix of the same scale
#
# The contrast between the last two isolates the role of the conditioning, and
# the contrast with Experiment C is the point: dispersed starts are adequate on
# a well-conditioned problem and not on this one.

const NSTART_D, MAXIT_D = 10, 4000

"""Rebuild a sample from given innovations along the fitted recursion."""
function replay(Π, Γ₁, dpath, E, Zdet, x_init, dx_init)
    T = size(E, 1)
    Yb = zeros(T, P); Xb = zeros(T, P)
    Zb = hcat(zeros(T, P), Zdet)
    xprev = copy(x_init); dprev = copy(dx_init)
    for t in 1:T
        Xb[t, :] = xprev
        Zb[t, 1:P] = dprev
        dx = Π * xprev + Γ₁ * dprev + dpath[t, :] + E[t, :]
        Yb[t, :] = dx
        dprev = dx
        xprev = xprev + dx
    end
    return Yb, Xb, Zb
end

function multistart(Yb, Xb, Zb, label; nstart=NSTART_D, seed=0, verbose=true)
    npar = P * (R + size(Zb, 2))
    srng = Xoshiro(seed)
    starts = [(1.0 .+ 0.5 .* randn(srng, R), randn(srng, npar)) for _ in 1:nstart]
    ll = zeros(nstart); it = zeros(Int, nstart)
    Threads.@threads for k in 1:nstart
        r = fit_basis(Yb, Xb, Zb, :iid, true; phi0=starts[k][1], psi0=starts[k][2],
                      tol=1e-10, maxiter=MAXIT_D)
        ll[k] = r[:loglik]; it[k] = r[:iters]
    end
    b0, ψ_pin, jr = closed_form_start(Yb, Xb, Zb, :iid)
    rw = fit_basis(Yb, Xb, Zb, :iid, true; phi0=b0, psi0=ψ_pin,
                   tol=1e-10, maxiter=MAXIT_D)
    best = max(maximum(ll), rw[:loglik], jr[:loglik])
    gaps = best .- ll
    res = (label=label, nstart=nstart,
           frac_at_best=mean(gaps .< 1e-6),
           max_gap=maximum(gaps), median_gap=median(gaps),
           iters_median=median(it), iters_max=maximum(it),
           n_hit_maxiter=count(>=(MAXIT_D), it),
           gap_closed_form=best - rw[:loglik],
           iters_closed_form=rw[:iters],
           gap_vs_eigenvalue=best - jr[:loglik])
    if verbose
        @printf("\n  === %s ===\n", label)
        for k in keys(res)
            k === :label || @printf("    %-22s %s\n", string(k), getfield(res, k))
        end
        flush(stdout)
    end
    return res
end

function experiment_D(Y, X, Z, x_init, dx_init; ndata=6, verbose=true)
    free, _ = sample_fits(Y, X, Z, :iid; verbose=false)
    E = resid(Y, X, Z, free)
    Ω = _sym(E' * E / size(E, 1))
    ev = eigvals(Symmetric(Ω))
    d = sqrt.(diag(Ω))
    C = Ω ./ (d * d')
    off = [C[i, j] for i in 1:P for j in 1:P if i != j]
    println("\n" * "=" ^ 78)
    println("EXPERIMENT D: starting values and the conditioning of Ω  (Table simD)")
    println("=" ^ 78)
    @printf("  fitted Ω: condition number %.1f; smallest off-diagonal correlation %.4f\n",
            ev[end] / ev[1], minimum(off))

    out = Any[multistart(Y, X, Z, "data (observed WTI sample)";
                         nstart=30, seed=11, verbose=verbose)]
    Π, Γ₁, dpath = vecm_parts(free, Z)
    Zdet = Z[:, P + 1:end]
    T = size(Y, 1)
    for (tag, Om) in (("sim (fitted Ω)", Ω),
                      ("sim-orth (well-conditioned Ω)",
                       Matrix{Float64}(I, P, P) * mean(diag(Ω))))
        L = cholesky(Symmetric(Om)).L
        agg = Any[]
        for k in 1:ndata
            rng = Xoshiro(4400 + k)
            Es = Matrix((L * randn(rng, P, T))')
            Yb, Xb, Zb = replay(Π, Γ₁, dpath, Es, Zdet, x_init, dx_init)
            push!(agg, multistart(Yb, Xb, Zb, "$tag #$k";
                                  nstart=NSTART_D, seed=900 + k, verbose=verbose))
        end
        push!(out, (label=tag * " [aggregate]",
                    frac_at_best=mean(a.frac_at_best for a in agg),
                    max_gap=maximum(a.max_gap for a in agg),
                    median_gap=median([a.median_gap for a in agg]),
                    iters_median=median([a.iters_median for a in agg]),
                    n_hit_maxiter=sum(a.n_hit_maxiter for a in agg),
                    nstart=sum(a.nstart for a in agg),
                    gap_closed_form=maximum(a.gap_closed_form for a in agg),
                    iters_closed_form=maximum(a.iters_closed_form for a in agg)))
        @printf("\n  >>> %s: mean fraction at best %.3f, max gap %.3f\n",
                tag, out[end].frac_at_best, out[end].max_gap)
        flush(stdout)
    end

    # Every entry of Table simD, one column per line. `median_gap` is the
    # median over starts of the shortfall from the best value found; for the
    # simulated columns it is the median over samples of that median.
    println("\n  Table simD, this run")
    @printf("  %-34s%14s%14s%14s\n", "", "observed", "sim, fitted", "sim, orth")
    @printf("  %-34s%14.3f%14.3f%14.3f\n", "fraction of starts at best",
            out[1].frac_at_best, out[2].frac_at_best, out[3].frac_at_best)
    @printf("  %-34s%14.0f%14.0f%14.0f\n", "median iterations",
            out[1].iters_median, out[2].iters_median, out[3].iters_median)
    @printf("  %-34s%14.1f%14.1f%14.1f\n", "largest shortfall in loglik",
            out[1].max_gap, out[2].max_gap, out[3].max_gap)
    @printf("  %-34s%14.1f%14.1f%14.1f\n", "median shortfall in loglik",
            out[1].median_gap, out[2].median_gap, out[3].median_gap)
    @printf("  %-34s%9d/%4d%9d/%4d%9d/%4d\n", "starts hitting maxiter / starts",
            out[1].n_hit_maxiter, out[1].nstart, out[2].n_hit_maxiter,
            out[2].nstart, out[3].n_hit_maxiter, out[3].nstart)
    @printf("  %-34s%14.4f%14.4f%14.4f\n", "closed-form start: shortfall",
            out[1].gap_closed_form, out[2].gap_closed_form, out[3].gap_closed_form)
    @printf("  %-34s%14d%14d%14d\n", "closed-form start: iterations (max)",
            out[1].iters_closed_form, out[2].iters_closed_form,
            out[3].iters_closed_form)
    flush(stdout)

    println("\n  against Python (Table simD)")
    @printf("  %-46s%12s%12s\n", "", "this run", "Python")
    @printf("  %-46s%12.3f%12.3f\n", "data: fraction of random starts at best",
            out[1].frac_at_best, REF_D.data_frac)
    @printf("  %-46s%12.3f%12.3f\n", "data: largest gap from a random start",
            out[1].max_gap, REF_D.data_max_gap)
    @printf("  %-46s%12d%12d\n", "data: random starts hitting maxiter",
            out[1].n_hit_maxiter, REF_D.data_hit_maxiter)
    @printf("  %-46s%12d%12d\n", "data: iterations from the closed-form start",
            out[1].iters_closed_form, REF_D.data_iters_closed_form)
    @printf("  %-46s%12.3f%12.3f\n", "sim: mean fraction at best",
            out[2].frac_at_best, REF_D.sim_frac)
    @printf("  %-46s%12.3f%12.3f\n", "sim-orth: mean fraction at best",
            out[3].frac_at_best, REF_D.orth_frac)
    @printf("  %-46s%12.0f%12.0f\n", "sim-orth: median iterations",
            out[3].iters_median, REF_D.orth_iters_median)
    @printf("  %-46s%12.1f%12.1f\n", "fitted Ω: condition number",
            ev[end] / ev[1], REF_D.condition_number)
    @printf("  %-46s%12.4f%12.4f\n", "fitted Ω: smallest off-diagonal correlation",
            minimum(off), REF_D.min_off_corr)
    flush(stdout)
    return out, (condition_number=ev[end] / ev[1], min_off_corr=minimum(off))
end
