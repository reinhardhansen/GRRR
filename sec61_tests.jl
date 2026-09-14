# Every number Section 6.1 reports, on the four-contract system.
#
#   cd .../GRRR/Julia4
#   julia -t auto sec61_tests.jl         # about two minutes on eight threads
#   GRID=0 julia -t auto sec61_tests.jl  # skip the seasonal grid of Section 6.1
#   B=99  julia -t auto sec61_tests.jl   # a quick pass of the rank bootstrap
#
# Six blocks, in the order the section reports them:
#
#   1  augmented Dickey-Fuller tests on the four log price levels and on the
#      three spreads log F_{i+1} - log F1, constant, lag by AIC, MacKinnon (1994)
#      p-values. The lag search and the p-value function are ports of statsmodels
#      `adfuller(x, autolag='AIC')`, written out here so that nothing in this
#      package depends on a Python library.
#   2  Johansen trace statistics for r <= 0, ..., 3, computed from the
#      eigenvalues of `johansen_rrr` on the same (Y, X, Z) the GRRR fits use, and
#      compared with the Osterwald-Lenum quantiles for an unrestricted constant.
#   3  the recursive wild bootstrap rank test of Cavaliere, Rahbek and Taylor
#      (2014): the VECM with rank r0 imposed, Rademacher weights, B = 999.
#   4  lag length k = 1..5 on a common sample of 454 observations, by OLS on the
#      VECM in levels and differences, with log-likelihood, AIC, BIC, Hannan-Quinn
#      and Hosking's multivariate Ljung-Box statistic at ten lags.
#   5  the seasonal block at k = 2, six specifications, and the likelihood ratio
#      of fourier2 against none.
#   6  the seasonal grid: does the seasonal block move the test of
#      b1 = b2 = b3 = 1? Two covariance specifications, regime and scale with the
#      proxy in levels, across five seasonal blocks.
#
# Values from the Python implementation of the same six blocks, to compare
# against. They were produced by v30_sec61.py and v31_seasonal_grid.py.
#
#   ADF, statistic / p / lags
#     log F1            -1.6497  0.4572  4      log F2   -1.5757  0.4958  4
#     log F3            -1.6864  0.4381  3      log F4   -1.6347  0.4650  3
#     log F2 - log F1   -7.7695  9.0e-12 2      log F3 - log F1  -6.9687  8.8e-10  2
#     log F4 - log F1   -8.4126  2.1e-13 0
#     d log F1         -11.6115  0.0000  3      d log F2 -11.2398 0.0000  3
#     d log F3         -12.5641  0.0000  2      d log F4 -12.5071 0.0000  2
#
#   trace on the GRRR design, T = 458
#     eigenvalues 0.38902118 0.19153314 0.09148355 0.00355689
#     r <= 0  368.6049     r <= 1  142.9515     r <= 2  45.5735     r <= 3  1.6320
#     The paper quotes 45.6 against 15.49 and 1.6 against 3.84, which are these.
#     statsmodels' `coint_johansen` on the raw levels with no seasonal terms gives
#     369.47, 146.57, 45.95, 1.72 instead. Those are a different design, not a
#     different implementation of this one.
#
#   recursive wild bootstrap, B = 999
#     r <= 0  p 0.0010  95th pct 66.58        r <= 1  p 0.0010  95th pct 34.73
#     r <= 2  p 0.0010  95th pct 20.51        r <= 3  p 0.5290  95th pct  8.13
#     The paper reports p = 0.001 for the first three nulls and 0.529 for the
#     fourth. The fourth will differ here in the second decimal and the
#     percentiles in the first: numpy's PCG64 and Julia's Xoshiro256++ cannot be
#     made to draw the same signs, so the only p-values that come out identically
#     are the first three, which sit at 1/(B+1), the smallest value a bootstrap
#     p-value can take.
#
#   lag length, 454 rows
#     k  loglik      AIC        BIC        HQ         LB(10) p
#     1  5965.4019  -11838.8   -11649.4   -11764.2   0.001
#     2  6012.8090  -11901.6   -11646.3   -11801.0   0.495
#     3  6024.9716  -11893.9   -11572.7   -11767.4   0.396
#     4  6034.2958  -11880.6   -11493.5   -11728.1   0.501
#     5  6046.2954  -11872.6   -11419.6   -11694.1   0.841
#     AIC and Hannan-Quinn pick k = 2, BIC picks k = 1, and the residuals of
#     k = 1 fail the Ljung-Box test that k = 2 passes.
#
#   seasonal terms at k = 2, 454 rows
#     terms     cols     loglik        AIC        BIC
#     none         0  5993.9693  -11895.9  -11706.5
#     quarter      3  6015.7491  -11915.5  -11676.6
#     fourier1     2  6002.4485  -11896.9  -11674.5
#     fourier2     4  6012.8090  -11901.6  -11646.3
#     fourier3     6  6024.6684  -11909.3  -11621.1
#     month       11  6043.4053  -11906.8  -11536.2
#     LR of fourier2 against none, 16 df: 37.679 on these 454 rows, p 0.0017, and
#     35.030 on the 458 rows the GRRR fits use, p 0.0039. The paper's 35.1 is the
#     second of the two, so both are printed below.
#
#   seasonal grid, chi-squared(3) p-value of b1 = b2 = b3 = 1
#                             none  quarter fourier2 fourier3   month   spread
#     regime                0.0142   0.0088   0.0089   0.0095   0.0119   0.0054
#     scale, proxy levels   0.4942   0.4881   0.5145   0.5409   0.4985   0.0528
#     smallest gap between the two rows 0.4793. The seasonal block moves the
#     p-value by half a percentage point under the regime covariance and by five
#     under the scale covariance, and neither spread comes close to the half-unit
#     gap between the two rows. The seasonal specification is not what decides the
#     test; the covariance specification is.
#
# Nothing in the fits is new. The loader, the spread basis, the closed-form start
# and `sample_fits` all come from emp4.jl, `johansen_rrr` from grrr_core.jl, and
# the bootstrap recursion from `generate` in boot4.jl, at the tolerances the
# sample fits in run_emp4.jl use.

using LinearAlgebra, Statistics, Random, Distributions, Printf, CSV, DataFrames, Dates

BLAS.set_num_threads(1)     # the threading here is over bootstrap replications

include("grrr_core.jl")
include("theta_scale.jl")
include("theta_gft.jl")     # must follow theta_scale.jl: it redefines theta_step
include("emp4.jl")
include("boot4.jl")         # for `generate`, the recursive bootstrap recursion

const NAMES = ["F1", "F2", "F3", "F4"]
const B_RANK = parse(Int, get(ENV, "B", "999"))
const RUN_GRID = get(ENV, "GRID", "1") != "0"

# The two sample starts. Python counts rows from zero, so its `start = 5` is row
# 6 here and its `start = 1`, the first row a k = 2 VECM can use, is row 2.
const ROW_COMMON = 6        # k = 1..5 all fit on these rows, 454 of them
const ROW_FULL   = 2        # the rows load_emp4 uses, 458 of them

# Osterwald-Lenum 5 per cent and neighbouring quantiles of the trace statistic
# with an unrestricted constant, for p - r = 4, 3, 2, 1. Copied from the table
# statsmodels' `coint_johansen` reads at det_order = 0, which is
# statsmodels/tsa/coint_tables.py, `c_sjt(n, 0)`. Columns are 90, 95, 99 per cent.
const TRACE_CV = [44.4929 47.8545 54.6815;      # r <= 0
                  27.0669 29.7961 35.4628;      # r <= 1
                  13.4294 15.4943 19.9349;      # r <= 2
                   2.7055  3.8415  6.6349]      # r <= 3

# Keyed by the numeraire exactly as the bootstrap files are, so that a run on
# the spot system cannot overwrite the front contract's table.
sec61_resultdir() = (d = joinpath(@__DIR__, NUMERAIRE === :F1 ? "results4" :
                                            "results4_$(NUMERAIRE)");
                     mkpath(d); d)

const OUTPATH = joinpath(sec61_resultdir(), "sec61_tests.txt")
const FH = open(OUTPATH, "w")

"""One line to the screen and the same line to the result file."""
function say(s::AbstractString)
    println(stdout, s)
    println(FH, s)
    flush(stdout)
    flush(FH)
    return nothing
end

# ---------------------------------------------------------------------------
# 1. augmented Dickey-Fuller
# ---------------------------------------------------------------------------

# MacKinnon (1994), "Approximate Asymptotic Distribution Functions for Unit-Root
# and Cointegration Tests", Journal of Business and Economic Statistics 12,
# 167-176. These are the response-surface coefficients for the constant-only
# regression with N = 1 series, copied verbatim from statsmodels'
# tsa/adfvalues.py: `tau_star_c`, `tau_min_c`, `tau_max_c`, and the first rows of
# `tau_c_smallp` and `tau_c_largep`, each already multiplied by the scaling
# vector that file applies. The p-value is the standard normal distribution
# function evaluated at a cubic in the test statistic.
const TAU_STAR_C = -1.61
const TAU_MIN_C = -18.83
const TAU_MAX_C = 2.74
const TAU_C_SMALLP = [2.1659, 1.4412, 3.8269 * 1e-2]
const TAU_C_LARGEP = [1.7339, 9.3202 * 1e-1, -1.2745 * 1e-1, -1.0368 * 1e-2]

"""MacKinnon's approximate p-value for an ADF statistic, constant only, N = 1."""
function mackinnonp_c(t::Real)
    t > TAU_MAX_C && return 1.0
    t < TAU_MIN_C && return 0.0
    c = t <= TAU_STAR_C ? TAU_C_SMALLP : TAU_C_LARGEP
    v = 0.0
    for i in length(c):-1:1        # Horner, lowest-order coefficient first
        v = v * t + c[i]
    end
    return cdf(Normal(), v)
end

"""Least squares, returning the coefficients and the residuals."""
function ols_fit(Xd, y)
    b = Xd \ y
    return b, y - Xd * b
end

"""The ADF regressor matrix [x_{t-1}, dx_{t-1}, ..., dx_{t-nlag}, 1] and dx_t.

The sample is the longest one the lag order allows, so it shrinks by one row for
each extra lag. That is what statsmodels does on the refit; the lag search below
uses one fixed sample instead."""
function adf_design(x::Vector{Float64}, nlag::Int)
    dx = diff(x)
    nd = length(dx)
    n = nd - nlag
    y = dx[nd - n + 1:nd]
    cols = Vector{Vector{Float64}}()
    push!(cols, x[length(x) - n:length(x) - 1])
    for i in 1:nlag
        push!(cols, dx[nd - n - i + 1:nd - i])
    end
    push!(cols, ones(n))
    return hcat(cols...), y
end

"""ADF statistic, MacKinnon p-value, chosen lag and sample size.

The protocol is statsmodels' `adfuller(x, autolag='AIC')` line by line. The
maximum lag is Schwert's ceil(12 (n/100)^(1/4)); every lag from 0 to that maximum
is fitted on the one sample the longest of them allows, so the information
criteria are comparable; and the winner is then refitted on the longest sample
that lag alone allows, which is where the statistic comes from. The AIC is the
Gaussian one statsmodels uses, -2 loglik + 2 (number of regressors)."""
function adf_aic(xin::AbstractVector{<:Real})
    x = Vector{Float64}(xin)
    n0 = length(x)
    maxlag = min(div(n0, 2) - 1 - 1, ceil(Int, 12.0 * (n0 / 100.0)^0.25))
    Xfull, yc = adf_design(x, maxlag)
    ncom = length(yc)
    nc = size(Xfull, 2)
    best = Inf
    bestlag = 0
    for lag in 0:maxlag
        idx = vcat(collect(1:lag + 1), nc)      # level, lagged diffs, constant
        _, e = ols_fit(Xfull[:, idx], yc)
        ssr = dot(e, e)
        ll = -0.5 * ncom * (log(2π) + log(ssr / ncom) + 1.0)
        aic = -2 * ll + 2 * length(idx)
        if aic < best
            best = aic
            bestlag = lag
        end
    end
    Xd, y = adf_design(x, bestlag)
    b, e = ols_fit(Xd, y)
    n, kk = size(Xd)
    V = (dot(e, e) / (n - kk)) * inv(Xd' * Xd)
    t = b[1] / sqrt(V[1, 1])
    return t, mackinnonp_c(t), bestlag, n
end

# ---------------------------------------------------------------------------
# 2. trace statistics from the GRRR design
# ---------------------------------------------------------------------------

"""Trace statistics for r <= 0, ..., p-1 from the eigenvalue problem.

`johansen_rrr` at r = p returns all p eigenvalues of the reduced-rank problem,
which is all this needs. Entry i of the result is the statistic for r <= i-1."""
function trace_stats(Yb, Xb, Zb)
    ev = johansen_rrr(Yb, Xb, Zb, P)[:eigenvalues][1:P]
    lam = clamp.(Float64.(ev), 0.0, 1 - 1e-12)
    return [-size(Yb, 1) * sum(log.(1 .- lam[i:P])) for i in 1:P]
end

# ---------------------------------------------------------------------------
# 3. the recursive wild bootstrap rank test
# ---------------------------------------------------------------------------

"""(Pi, Psi) of the VECM with rank r0 imposed.

At r0 = 0 there is no levels term at all and Psi is the OLS coefficient on Z; at
r0 > 0 the reduced-rank fit supplies both."""
function vecm_at_rank(Y, X, Z, r0)
    if r0 == 0
        return zeros(P, P), Matrix(((Z' * Z) \ (Z' * Y))')
    end
    jr = johansen_rrr(Y, X, Z, r0)
    return jr[:alpha] * jr[:beta]', jr[:Psi]
end

"""Bootstrap p-value and 95th percentile for the null r <= r0.

The recursion is `generate` from boot4.jl, which is the recursion the Section 6
bootstrap uses: the same Pi and Gamma_1, the same precomputed deterministic path,
and the observed constant-and-seasonal columns of Z carried over unchanged,
because the bootstrap sample inherits the dates of the observed one.

Seeding. The Python script draws all B samples from one generator per null,
`default_rng(90210 + r0)`. A threaded loop cannot do that and still give the same
answer whatever the thread count, so each replication gets its own generator
instead, exactly as `run_bootstrap` in boot4.jl does: replication b of null r0 is
`Xoshiro(90210 + 1000 r0 + b)`. The base is the Python seed and the spacing of
1000 keeps the four nulls from sharing a stream."""
function boot_rank(Y, X, Z, r0, obs_r, B, x_init, dx_init)
    Pi, Psi = vecm_at_rank(Y, X, Z, r0)
    G1 = Psi[:, 1:P]
    dpath = Z[:, P + 1:end] * Psi[:, P + 1:end]'
    E = Y - X * Pi' - Z * Psi'
    Zdet = Z[:, P + 1:end]
    seed_base = 90210 + 1000 * r0

    st = fill(NaN, B)
    Threads.@threads for b in 1:B
        rng = Xoshiro(seed_base + b)
        Yb, Xb, Zb = generate(Pi, G1, dpath, E, Zdet, x_init, dx_init, rng;
                              scheme=:rademacher)
        st[b] = trace_stats(Yb, Xb, Zb)[r0 + 1]
    end
    p = (1.0 + count(>=(obs_r), st)) / (B + 1.0)
    return p, quantile(st, 0.95)
end

# ---------------------------------------------------------------------------
# 4 and 5. lag length, seasonal terms
# ---------------------------------------------------------------------------

"""Seasonal regressors for the Psi block, for every kind the section reports.

`seasonal_block` in emp4.jl knows only the Fourier terms, because they are the
only ones the paper's fits use. This covers the five alternatives as well:
"none", "quarter" (dummies for quarters 2, 3, 4), "month" (dummies for months 2
to 12) and "fourier1", "fourier2", "fourier3". `months` is 1..12 per row."""
function seasonal_columns(months::Vector{Int}, kind::AbstractString)
    n = length(months)
    kind == "none" && return zeros(n, 0)
    if kind == "quarter"
        q = div.(months .- 1, 3)
        return hcat([Float64.(q .== k) for k in 1:3]...)
    end
    if kind == "month"
        return hcat([Float64.(months .== k) for k in 2:12]...)
    end
    if startswith(kind, "fourier")
        h = parse(Int, kind[8:end])
        return seasonal_block(months, h)
    end
    error("unknown seasonal block $kind")
end

"""OLS fit of the VECM in levels and differences, on rows `row0` to N-1.

    dx_t = A x_t + sum_{i=1}^{k-1} G_i dx_{t-i} + constant + seasonals + e_t

Returns the Gaussian log-likelihood at the OLS residual covariance, the three
information criteria, and the residuals. The parameter count is the mean
parameters plus the p(p+1)/2 free elements of Omega. `row0` fixes the sample, so
every k in a comparison has to be given the same one."""
function vecm_ols(L, dL, months, N, k::Int, row0::Int, kind::AbstractString)
    rows = row0:N - 1
    Yd = dL[rows, :]
    cols = Any[L[rows, :]]
    for i in 1:k - 1
        push!(cols, dL[rows .- i, :])
    end
    push!(cols, ones(length(rows), 1))
    Sm = seasonal_columns(months[rows], kind)
    size(Sm, 2) > 0 && push!(cols, Sm)
    Xd = hcat(cols...)
    Bc = Xd \ Yd
    E = Yd - Xd * Bc
    Tn = size(Yd, 1)
    Om = E' * E / Tn
    ll = -0.5 * Tn * (P * log(2π) + logabsdet(Om)[1] + P)
    kk = size(Xd, 2) * P + div(P * (P + 1), 2)
    return ll, -2 * ll + 2 * kk, -2 * ll + log(Tn) * kk,
           -2 * ll + 2 * log(log(Tn)) * kk, E
end

"""Hosking's multivariate portmanteau statistic at `nlag` lags."""
function hosking_q(E, nlag::Int)
    Tn = size(E, 1)
    C0 = E' * E / Tn
    C0i = inv(C0)
    Q = 0.0
    for lag in 1:nlag
        Cl = E[lag + 1:end, :]' * E[1:Tn - lag, :] / Tn
        Q += tr(Cl' * C0i * Cl * C0i) / (Tn - lag)
    end
    return Q * Tn * Tn
end

# ---------------------------------------------------------------------------
# 6. the seasonal grid
# ---------------------------------------------------------------------------

"""`load_emp4` with the seasonal block supplied by name rather than by order.

Line for line the loader in emp4.jl, which hard-codes the Fourier terms. Keeping
it here rather than widening that function keeps the empirical files untouched:
the seasonal alternatives are a robustness check of Section 6.1 and nothing in
Section 6 proper ever builds them."""
function load_seasonal(path::String, kind::AbstractString;
                       numeraire::Symbol=NUMERAIRE)
    df = CSV.read(path, DataFrame)
    L = Matrix(df[:, numeraire_columns(numeraire)])
    dts = df.date
    N = size(L, 1)
    dL = diff(L, dims=1)
    Y = dL[2:end, :]                                   # dx_t,  t = 2..N-1
    X = L[2:N - 1, :]                                  # x_{t-1}
    months = [month(d isa Date ? d : Date(string(d))) for d in dts[3:N]]
    S = seasonal_columns(months, kind)
    Z = hcat(dL[1:N - 2, :], ones(N - 2), S)           # [dx_{t-1}, const, seas]
    return Y, X, Z, dts[3:N]
end

# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------

const DATAPATH = joinpath(@__DIR__, "data", "wti_spot_futures.csv")

df = CSV.read(DATAPATH, DataFrame)
Lfull = Matrix(df[:, numeraire_columns(NUMERAIRE)])
dates_all = [d isa Date ? d : Date(string(d)) for d in df.date]
months_all = [month(d) for d in dates_all]
Nobs = size(Lfull, 1)
dLfull = diff(Lfull, dims=1)

say(@sprintf("four contracts, %s to %s, N = %d, numeraire %s",
             Dates.format(dates_all[1], "yyyy-mm"),
             Dates.format(dates_all[end], "yyyy-mm"), Nobs, NUMERAIRE))
say(@sprintf("threads: %d;  BLAS threads: %d", Threads.nthreads(),
             BLAS.get_num_threads()))
say("")

# ---------------------------------------------------------------------------
# 1. unit roots
# ---------------------------------------------------------------------------

say("augmented Dickey-Fuller, MacKinnon p-values, lag by AIC")
say(@sprintf("%-16s%9s%9s%6s", "series", "ADF", "p", "lags"))
for j in 1:P
    s, p, lg = adf_aic(Lfull[:, j])
    say(@sprintf("%-16s%9.2f%9.4f%6d", "log " * NAMES[j], s, p, lg))
end
for i in 1:R
    s, p, lg = adf_aic(Lfull[:, i + 1] - Lfull[:, 1])
    say(@sprintf("%-16s%9.2f%9.4f%6d",
                 "log " * NAMES[i + 1] * " - log F1", s, p, lg))
end
say("for reference, differences:")
for j in 1:P
    s, p, lg = adf_aic(dLfull[:, j])
    say(@sprintf("%-16s%9.2f%9.4f%6d", "d log " * NAMES[j], s, p, lg))
end

# The three spread p-values print as 0.0000 at four decimals, so the exponents
# are worth having: Python gives 9.0e-12, 8.8e-10 and 2.1e-13.
say("spread p-values in full:")
for i in 1:R
    s, p = adf_aic(Lfull[:, i + 1] - Lfull[:, 1])
    say(@sprintf("    log %s - log F1   stat %9.4f   p %.3e",
                 NAMES[i + 1], s, p))
end

# ---------------------------------------------------------------------------
# 2. rank by the trace test
# ---------------------------------------------------------------------------

Y, X, Z, sd = load_emp4(DATAPATH; h=2)
T = size(Y, 1)
x_init = X[1, :]
dx_init = Z[1, 1:P]

say("")
say(@sprintf("Johansen trace test on the GRRR design, k = 2, T = %d, %s .. %s",
             T, string(sd[1]), string(sd[end])))
say(@sprintf("  Z has %d columns: four lagged differences, a constant, four Fourier terms",
             size(Z, 2)))
say("  Osterwald-Lenum quantiles, statsmodels coint_tables c_sjt(p - r, 0)")
ev = johansen_rrr(Y, X, Z, P)[:eigenvalues][1:P]
say(@sprintf("  eigenvalues: %s",
             join([@sprintf("%.8f", v) for v in ev], " ")))
obs = trace_stats(Y, X, Z)
for i in 1:P
    say(@sprintf("    r <= %d: trace %8.2f   90%% %7.2f   95%% %7.2f   99%% %7.2f   %s",
                 i - 1, obs[i], TRACE_CV[i, 1], TRACE_CV[i, 2], TRACE_CV[i, 3],
                 obs[i] > TRACE_CV[i, 2] ? "reject" : "do not reject"))
end

# ---------------------------------------------------------------------------
# 3. rank by the recursive wild bootstrap
# ---------------------------------------------------------------------------

say("")
say(@sprintf("recursive wild bootstrap rank test, B = %d, Rademacher weights",
             B_RANK))
say("  Cavaliere, Rahbek and Taylor (2014); the null is imposed on the DGP")
say(@sprintf("%8s%10s%10s%11s", "H0", "trace", "boot p", "boot 95%"))
t0 = time()
for r0 in 0:P - 1
    p, q95 = boot_rank(Y, X, Z, r0, obs[r0 + 1], B_RANK, x_init, dx_init)
    say(@sprintf("%8s%10.2f%10.4f%11.2f   (%.0f s)",
                 "r <= " * string(r0), obs[r0 + 1], p, q95, time() - t0))
end

# ---------------------------------------------------------------------------
# 4. lag length
# ---------------------------------------------------------------------------

say("")
say(@sprintf("lag length on a common sample of %d observations, fourier2 seasonals",
             Nobs - ROW_COMMON))
say(@sprintf("%3s%13s%12s%12s%12s%11s", "k", "loglik", "AIC", "BIC", "HQ",
             "LB(10) p"))
for k in 1:5
    ll, aic, bic, hq, E = vecm_ols(Lfull, dLfull, months_all, Nobs, k,
                                   ROW_COMMON, "fourier2")
    Q = hosking_q(E, 10)
    dfq = max(P * P * (10 - (k - 1)), 1)
    say(@sprintf("%3d%13.2f%12.1f%12.1f%12.1f%11.3f",
                 k, ll, aic, bic, hq, ccdf(Chisq(dfq), Q)))
end

# ---------------------------------------------------------------------------
# 5. seasonal terms
# ---------------------------------------------------------------------------

say("")
say(@sprintf("seasonal terms at k = 2, the same %d observations", Nobs - ROW_COMMON))
say(@sprintf("%-12s%6s%13s%12s%12s", "terms", "cols", "loglik", "AIC", "BIC"))
ll_seas = Dict{String,Float64}()
for kind in ["none", "quarter", "fourier1", "fourier2", "fourier3", "month"]
    ll, aic, bic = vecm_ols(Lfull, dLfull, months_all, Nobs, 2,
                            ROW_COMMON, kind)
    ll_seas[kind] = ll
    ncol = size(seasonal_columns(collect(1:12), kind), 2)
    say(@sprintf("%-12s%6d%13.2f%12.1f%12.1f", kind, ncol, ll, aic, bic))
end

# The likelihood ratio of fourier2 against none, on 4 seasonal columns in each of
# the p = 4 equations. It is reported on both samples: the common sample the
# table above uses, and the sample the GRRR fits use, which is the one the paper
# quotes at 35.1.
dfs = 4 * P
lr_common = 2 * (ll_seas["fourier2"] - ll_seas["none"])
ll_f_full = vecm_ols(Lfull, dLfull, months_all, Nobs, 2, ROW_FULL, "fourier2")[1]
ll_n_full = vecm_ols(Lfull, dLfull, months_all, Nobs, 2, ROW_FULL, "none")[1]
lr_full = 2 * (ll_f_full - ll_n_full)
say(@sprintf("LR fourier2 against none, %d df: %7.3f  p %.4f   (%d rows, as above)",
             dfs, lr_common, ccdf(Chisq(dfs), lr_common), Nobs - ROW_COMMON))
say(@sprintf("LR fourier2 against none, %d df: %7.3f  p %.4f   (%d rows, GRRR sample)",
             dfs, lr_full, ccdf(Chisq(dfs), lr_full), Nobs - ROW_FULL))

# ---------------------------------------------------------------------------
# 6. the seasonal grid
# ---------------------------------------------------------------------------

const GRID_SEAS = ["none", "quarter", "fourier2", "fourier3", "month"]
const GRID_SPECS = [(:regime, "regime (2008:9, 2020:3)"),
                    (:scale, "scale, proxy in levels")]

if RUN_GRID
    say("")
    say("seasonal grid: does the seasonal block move the test of b1 = b2 = b3 = 1?")
    say("  the scale fits take a few minutes; GRID=0 skips this block")
    pgrid = Dict{Tuple{Symbol,String},Float64}()
    tg = time()
    for kind in GRID_SEAS
        Yg, Xg, Zg = load_seasonal(DATAPATH, kind)
        for (spec, _) in GRID_SPECS
            free, one = sample_fits(Yg, Xg, Zg, spec; verbose=false)
            lr = 2 * (free[:loglik] - one[:loglik])
            pgrid[(spec, kind)] = ccdf(Chisq(R), lr)
            say(@sprintf("  %-9s%-7s Z %2d  free %11.4f  b=1 %11.4f  LR %7.3f  p %.4f  %4.0fs",
                         kind, string(spec), size(Zg, 2), free[:loglik],
                         one[:loglik], lr, pgrid[(spec, kind)], time() - tg))
        end
    end

    say("")
    say("chi-squared(3) p-value of b1 = b2 = b3 = 1")
    say(@sprintf("%-26s%s%9s", "Sigma specification",
                 join([@sprintf("%10s", s) for s in GRID_SEAS], ""), "spread"))
    for (spec, lab) in GRID_SPECS
        row = [pgrid[(spec, s)] for s in GRID_SEAS]
        say(@sprintf("%-26s%s%9.4f", lab,
                     join([@sprintf("%10.4f", v) for v in row], ""),
                     maximum(row) - minimum(row)))
    end
    gaps = [abs(pgrid[(:scale, s)] - pgrid[(:regime, s)]) for s in GRID_SEAS]
    say(@sprintf("smallest gap between the two rows: %.4f", minimum(gaps)))
    say(@sprintf("regime rejects at 5%% in %d of %d blocks; scale in %d of %d",
                 count(<(0.05), [pgrid[(:regime, s)] for s in GRID_SEAS]),
                 length(GRID_SEAS),
                 count(<(0.05), [pgrid[(:scale, s)] for s in GRID_SEAS]),
                 length(GRID_SEAS)))
else
    say("")
    say("seasonal grid skipped (GRID=0)")
end

say("")
say(@sprintf("finished at %s", Dates.format(now(), "HH:MM:SS")))
close(FH)
@printf("\nthe same output is on disk at %s\n", OUTPATH)
