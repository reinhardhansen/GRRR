# Paired comparison of LR* between the two implementations.
#
#   julia -t auto lrstar_gen.jl          # then: python lrstar_py.py
#
# The size experiment gives Julia a thinner upper tail of LR* than Python, in
# the same direction for every specification. That could be sampling: the two
# runs draw different samples, and at four hundred draws the rejection rate at
# the one per cent level has a standard error of half a point. It could also be
# real, because Python warm-starts each θ-step at the sample fit's θ and Julia
# starts every θ-step cold, which on a hard draw could leave the free fit short
# of its maximum and so shorten LR*.
#
# Sampling noise is removable. This script writes the simulated samples to disk
# and records what Julia makes of them; `lrstar_py.py` reads the same samples
# and records what Python makes of them. On identical data the comparison is
# exact and needs no statistics: the higher log-likelihood is the better answer,
# because it is the same likelihood evaluated on the same numbers.
#
# The draws are the ones the size experiment itself used, seed 19510104 + j, so
# a difference found here is a difference in Table empSize.
#
#   N=60  SPECS=iid,scalelog,archchol  julia -t auto lrstar_gen.jl
#
# `iid` is the control. Its θ-step is closed form, Ω̂ = E'E/T, with no optimiser
# and nothing to warm-start, so the two implementations must agree there to
# rounding. If they do and the others do not, the warm start is the cause. If
# iid disagrees too, the cause is somewhere else and this says where to look.

using LinearAlgebra, Statistics, Random, Distributions, Printf, CSV, DataFrames, Dates

BLAS.set_num_threads(1)

include("grrr_core.jl")
include("theta_scale.jl")
include("theta_gft.jl")
include("emp4.jl")
include("boot4.jl")
include("size4.jl")

const NSAMP = parse(Int, get(ENV, "N", "60"))
const WANT = Symbol.(split(get(ENV, "SPECS", "iid,scalelog,archchol"), ','))
const SEED0 = 19510104
const OUTDIR = (d = joinpath(@__DIR__, "lrstar"); mkpath(d); d)

@printf("threads: %d;  %d samples per specification;  writing to %s\n\n",
        Threads.nthreads(), NSAMP, OUTDIR)

Y, X, Z, sd = load_emp4(joinpath(@__DIR__, "data", "wti_spot_futures.csv"); h=2)
x_init = X[1, :]
dx_init = Z[1, 1:P]

"""One simulated sample as plain text: a header of T, p and the width of Z,
then Y, X and Z, each written row by row, one number per line. No package on
either side needs to read it."""
function write_sample(path, Yj, Xj, Zj)
    open(path, "w") do fh
        @printf(fh, "%d %d %d\n", size(Yj, 1), size(Yj, 2), size(Zj, 2))
        for A in (Yj, Xj, Zj)
            for i in 1:size(A, 1), j in 1:size(A, 2)
                println(fh, A[i, j])
            end
        end
    end
end

for spec in WANT
    spec in first.(SIZE_SPECS) ||
        error("unknown specification $spec; choose from " *
              join(string.(first.(SIZE_SPECS)), ", "))
    label = SIZE_SPECS[findfirst(s -> s[1] === spec, SIZE_SPECS)][2]
    @printf("=== %s ===\n", label); flush(stdout)
    free, one = sample_fits(Y, X, Z, spec; verbose=false)
    @printf("  observed sample: l_free %.4f  l_b=1 %.4f  LR %.3f\n",
            free[:loglik], one[:loglik], 2 * (free[:loglik] - one[:loglik]))
    src = size_source(spec, Y, X, Z, free, one)

    # Phase one, sequential and cheap: draw the samples and write them out. The
    # runaway skip has to be sequential because it decides which j survive, and
    # simulating a path costs nothing next to fitting it.
    t0 = time()
    keep = Tuple{Int,Matrix{Float64},Matrix{Float64},Matrix{Float64}}[]
    j = 0
    while length(keep) < NSAMP && j < 20 * NSAMP
        j += 1
        rng = Xoshiro(SEED0 + j)
        out = simulate_null(src.spec, src.θ, src.Π, src.Γ₁, src.dpath, src.Zdet,
                            src.mhat, src.ubar, x_init, dx_init, src.T, rng;
                            kind=src.kind, shift=src.shift, Wfixed=src.Wfixed)
        out === nothing && continue          # a runaway path, as in the experiment
        Yj, Xj, Zj = out
        write_sample(joinpath(OUTDIR, "$(spec)_$(j).txt"), Yj, Xj, Zj)
        push!(keep, (j, Yj, Xj, Zj))
    end
    @printf("  %d samples written from %d draws (%.0f s); fitting\n",
            length(keep), j, time() - t0); flush(stdout)

    # Phase two, threaded: the fits, which are all of the cost. Each sample is
    # independent and its own starts are built inside, so nothing is shared.
    recs = Vector{Any}(undef, length(keep))
    done = Threads.Atomic{Int}(0)
    Threads.@threads for k in 1:length(keep)
        jj, Yj, Xj, Zj = keep[k]
        b0, ψ_pin, _ = closed_form_start(Yj, Xj, Zj, spec)
        f(fr; kw...) = fit_basis(Yj, Xj, Zj, spec, fr; tol=1e-8, maxiter=3000, kw...)
        ones_ = [f(false; psi0=src.ψ_one),
                 f(false; psi0=ψ_pin),
                 f(false; psi0=src.ψ_free)]
        lo = [r[:loglik] for r in ones_]
        o = ones_[argmax(lo)]
        frees = [f(true; phi0=ones(R), psi0=src.ψ_free),
                 f(true; phi0=b0, psi0=ψ_pin),
                 f(true; phi0=ones(R), psi0=psi_of(o))]
        lf = [r[:loglik] for r in frees]
        recs[k] = (jj, maximum(lo), maximum(lf),
                   2 * (maximum(lf) - maximum(lo)),
                   argmax(lo), argmax(lf),
                   maximum(lo) - minimum(lo), maximum(lf) - minimum(lf))
        n = Threads.atomic_add!(done, 1) + 1
        n % 10 == 0 && (@printf("    %d/%d  (%.0f s)\n", n, length(keep),
                                time() - t0); flush(stdout))
    end
    recs = sort(collect(recs); by=r -> r[1])

    path = joinpath(OUTDIR, "$(spec)_julia.txt")
    open(path, "w") do fh
        println(fh, "# j ll_one ll_free LR won_one won_free spread_one spread_free")
        for r in recs
            println(fh, join(r, " "))
        end
    end
    LR = [r[4] for r in recs]
    @printf("  %d samples, %d draws tried, %.0f s total\n",
            length(recs), j, time() - t0)
    @printf("  LR*: median %.3f, 90th %.3f, max %.3f;  above chi2(%d) 99%% (%.2f): %d\n\n",
            median(LR), quantile(LR, 0.90), maximum(LR), R,
            quantile(Chisq(R), 0.99), count(>=(quantile(Chisq(R), 0.99)), LR))
    flush(stdout)
end

println("done. Now run, in the Python4 folder:")
println("    python lrstar_py.py")
