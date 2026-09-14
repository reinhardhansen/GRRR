# Break-date sensitivity of the regime covariance specification.
#
#   cd .../GRRR/Julia4
#   julia break_grid.jl              # 169 refits, about a quarter of an hour
#
# The paper fixes two variance breaks at the Y rows BRK = [270, 408], September
# 2008 and March 2020. Both dates come from the narrative rather than from an
# estimator, so the question this file answers is whether the rejection of
# b1 = b2 = b3 = 1 under the regime covariance survives moving them. Each break
# is shifted on its own by s1, s2 = -6..+6 months, 169 combinations, and every
# combination is refitted from scratch: the basis restriction free, then b = 1
# imposed, and LR = 2(l_free - l_one) against chi-squared on R = 3 degrees of
# freedom.
#
# Values from the Python implementation of the same grid, to compare against:
#
#     baseline LR at s1 = s2 = 0        11.593
#     min     9.775 at (s1, s2) = (+6, -1)
#     max    15.965 at (s1, s2) = (-4, +2)
#     median 11.75,   largest p 0.0206,   no combination with p > 0.05
#
# The break indices mean the same thing in the two implementations, which was
# checked rather than assumed. Python's theta step builds edges = [0] + breaks +
# [T] and takes segment rows a:b, zero based and half open; `theta_regime` in
# grrr_core.jl builds edges = vcat(0, ctx[:breaks], T) and takes rows a+1:b, one
# based and inclusive. Those are the same partition of the sample, so the
# integers 270 and 408 carry over unchanged. The date reported for a break is
# the first observation of the segment it opens, sd[k + 1] here against
# dates[k] there, and at s1 = s2 = 0 both give 2008-09 and 2020-03.
#
# Nothing in the fit itself is new: the loader, the closed-form start and the
# spread basis all come from emp4.jl, at the tolerances the sample fits in
# run_emp4.jl use. Those Julia defaults are not quite the Python ones, tol 1e-10
# and maxiter 20000 for `fit_basis` there against 8000 here, tol 1e-9 and
# maxiter 8000 for the pinned start there against 1e-10 and 20000 here, and the
# port check in run_emp4.jl is what says the difference does not reach the
# printed digits: the regime log-likelihoods agree to 5e-05.

using LinearAlgebra, Statistics, Random, Distributions, Printf, CSV, DataFrames, Dates

BLAS.set_num_threads(1)     # one fit at a time, so no thread is waiting on BLAS

include("grrr_core.jl")
include("theta_scale.jl")
include("theta_gft.jl")     # must follow theta_scale.jl: it redefines theta_step
include("emp4.jl")

const SHIFTS = -6:6         # months, applied to each break separately

# Keyed by the numeraire exactly as the bootstrap files are, so that a run on
# the spot system cannot overwrite the front contract's table.
grid_resultdir() = (d = joinpath(@__DIR__, NUMERAIRE === :F1 ? "results4" :
                                           "results4_$(NUMERAIRE)");
                    mkpath(d); d)

"""A regime specification whose breaks are not the paper's.

`BRK` is a constant and `sigma_ctx_for(:regime, Zb)` reads it, so a shifted grid
cannot travel through the `:regime` symbol. It travels through this type
instead: two small methods here, no edit to any existing file, and the fits
below are then literally the functions the empirical fits use."""
struct RegimeBreaks
    breaks::Vector{Int}
end

sigma_model_for(::RegimeBreaks) = :regime
sigma_ctx_for(spec::RegimeBreaks, Zb) = Dict{Symbol,Any}(:breaks => copy(spec.breaks))

"""The date a break opens, as yyyy-mm.

A break index is a segment edge, so the first observation of the new segment is
Y row k + 1 in Julia's one-based indexing, and `sd` carries one date per Y row."""
break_date(sd, k) = (d = sd[k + 1];
                     Dates.format(d isa Date ? d : Date(string(d)), "yyyy-mm"))

"""LR for b1 = b2 = b3 = 1 with the variance breaks at `breaks`.

The protocol is the one the Python grid uses: the closed-form start, the b = 1
fit from the pinned psi, and the free fit from two starts, the closed-form b0
and (1, 1, 1), both from that same pinned psi, keeping whichever start reaches
the higher log-likelihood. It is leaner than `sample_fits`, which also warms at
the iid fit and at the free fit; at the paper's break dates the two protocols
agree, LR 11.593 either way."""
function lr_at(Y, X, Z, breaks)
    spec = RegimeBreaks(breaks)
    b0, ψ_pin, _ = closed_form_start(Y, X, Z, spec)
    one = fit_basis(Y, X, Z, spec, false; psi0=ψ_pin)
    frees = [fit_basis(Y, X, Z, spec, true; phi0=b0, psi0=ψ_pin),
             fit_basis(Y, X, Z, spec, true; phi0=ones(R), psi0=ψ_pin)]
    free = argmax_loglik(frees)
    nonconv = count(r -> !r[:converged], vcat(frees, [one]))
    return 2 * (free[:loglik] - one[:loglik]), nonconv
end

# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------

Y, X, Z, sd = load_emp4(joinpath(@__DIR__, "data", "wti_spot_futures.csv"); h=2)
T = size(Y, 1)

BRK[1] + first(SHIFTS) > 0 && BRK[2] + last(SHIFTS) < T ||
    error("the shift range moves a break outside the sample")

@printf("T=%d  p=%d  r=%d  p2=%d   %s .. %s\n",
        T, P, R, size(Z, 2), string(sd[1]), string(sd[end]))
@printf("breaks as the paper sets them: Y rows %d and %d, %s and %s\n",
        BRK[1], BRK[2], break_date(sd, BRK[1]), break_date(sd, BRK[2]))
@printf("grid: %d x %d shifts of %d..+%d months, %d refits of the regime fit\n\n",
        length(SHIFTS), length(SHIFTS), first(SHIFTS), last(SHIFTS),
        length(SHIFTS)^2)
flush(stdout)

outpath = joinpath(grid_resultdir(), "break_grid_13x13.txt")
fh = open(outpath, "w")
println(fh, "# s1 s2 break1 break2 LR p  (regime covariance, both break dates shifted)")

shifts = Tuple{Int,Int}[]
lrs = Float64[]
ps = Float64[]
t0 = time()
for s1 in SHIFTS, s2 in SHIFTS
    k1, k2 = BRK[1] + s1, BRK[2] + s2
    d1, d2 = break_date(sd, k1), break_date(sd, k2)
    lr, nonconv = lr_at(Y, X, Z, [k1, k2])
    p = ccdf(Chisq(R), lr)
    nonconv > 0 && @printf("    warning: %d of 3 fits hit maxiter at (%+d,%+d)\n", nonconv, s1, s2)
    push!(shifts, (s1, s2)); push!(lrs, lr); push!(ps, p)
    @printf("%+d %+d %s %s LR %7.3f p %.4f  %4.0fs\n", s1, s2, d1, d2, lr, p,
            time() - t0)
    flush(stdout)
    @printf(fh, "%+d %+d %s %s %.3f %.4f\n", s1, s2, d1, d2, lr, p)
    flush(fh)
    lr < 0 && @printf("    warning: negative LR, the b = 1 fit found the higher maximum\n")
end

# ---------------------------------------------------------------------------
# summary, to the screen and to the same file
# ---------------------------------------------------------------------------

imin, imax = argmin(lrs), argmax(lrs)
ibase = findfirst(==((0, 0)), shifts)
for io in (stdout, fh)
    @printf(io, "ALL %d: min %.3f at (%+d,%+d)  max %.3f at (%+d,%+d)  median %.3f  largest p %.4f  n(p>0.05) %d  baseline %.3f\n",
            length(lrs), lrs[imin], shifts[imin][1], shifts[imin][2],
            lrs[imax], shifts[imax][1], shifts[imax][2], median(lrs),
            maximum(ps), count(>(0.05), ps), lrs[ibase])
    @printf(io, "Python reference: min 9.77 max 15.97 median 11.75 max p 0.0206\n")
end
close(fh)

@printf("\nthe table is on disk at %s\n", outpath)
