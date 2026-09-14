# Julia replication of the Section 5 Monte Carlo and the Section 6.2 recursive
# null experiment, four-contract system.
#
#   cd .../GRRR/Julia4
#   julia test_dgp.jl                        # 30 seconds, run this first
#   julia -t auto run_sim4.jl                # QUICK pass, a few minutes
#   FULL=1 julia -t auto run_sim4.jl         # the paper's counts
#
# Selection, for running one piece at a time or resuming after an interrupt:
#
#   ONLY=B          FULL=1 julia -t auto run_sim4.jl     # Table simB only
#   ONLY=size       FULL=1 julia -t auto run_sim4.jl     # Table empSize, Figure disc
#   ONLY=size SPECS=iid,regime,scale,scalelog  FULL=1 julia -t auto run_sim4.jl
#
# ONLY takes any comma-separated subset of B, C, D, size; the default is all
# four. SPECS restricts the size experiment to named specifications, which is
# how to run the two cheap ones first and leave the log-precision pair for
# overnight.
#
# Every replication is written to disk as it completes and a rerun picks up
# where the last one stopped, so an interrupted run costs only the replications
# that were in flight.

using LinearAlgebra, Statistics, Random, Distributions, Printf, CSV, DataFrames, Dates

BLAS.set_num_threads(1)          # each worker thread gets one BLAS thread

include("grrr_core.jl")
include("theta_scale.jl")
include("theta_gft.jl")     # must follow theta_scale.jl: it redefines theta_step
include("emp4.jl")
include("boot4.jl")
include("dgp4.jl")          # must follow emp4.jl: P, R and the basis come from it
include("mc4.jl")
include("size4.jl")

const QUICK = get(ENV, "FULL", "0") == "0"
# REPS overrides the replication count of Experiment B. The median absolute
# error at 500 replications carries about eight per cent of sampling error, so a
# gain quoted to the nearest per cent wants more than that; the ARCH θ-step is
# now fast enough to afford it. Results are keyed by count, so a run at one
# count does not resume from another.
const REPS_B = parse(Int, get(ENV, "REPS", QUICK ? "40" : "500"))
const NDATA_C = QUICK ? 6 : 50
const NDATA_D = QUICK ? 2 : 6
const M_SIZE = QUICK ? 40 : 400

const WANT = Set(Symbol.(split(get(ENV, "ONLY", "B,C,D,size"), ',')))
const WANT_SPECS = haskey(ENV, "SPECS") ?
    Set(Symbol.(split(ENV["SPECS"], ','))) : nothing

@printf("threads: %d;  BLAS threads: %d;  %s\n",
        Threads.nthreads(), BLAS.get_num_threads(),
        QUICK ? "QUICK (set FULL=1 for the paper's counts)" : "FULL")
@printf("running: %s\n\n", join(sort(string.(collect(WANT))), ", "))

isfile(joinpath(@__DIR__, "design_reference.txt")) ||
    println("!! design_reference.txt is missing, so test_dgp.jl cannot check the\n" *
            "!! port against Python. The experiments below will still run.\n")

t_start = time()

# ---------------------------------------------------------------------------
# Section 5: Experiments B, C and D
# ---------------------------------------------------------------------------

if :B in WANT
    println("=" ^ 78)
    @printf("EXPERIMENT B, %d replications per cell\n", REPS_B)
    println("=" ^ 78)
    cells = experiment_B(; reps=REPS_B)
    report_B(cells)
    flush(stdout)
end

if :C in WANT
    println("\n" * "=" ^ 78)
    @printf("EXPERIMENT C, %d data sets x %d starts\n", NDATA_C, NSTART_C)
    println("=" ^ 78)
    rows_c = experiment_C(; ndata=NDATA_C)
    report_C(rows_c)
    flush(stdout)
end

# ---------------------------------------------------------------------------
# the observed sample, needed by Experiment D and by the size experiment
# ---------------------------------------------------------------------------

need_data = (:D in WANT) || (:size in WANT)
if need_data
    Y, X, Z, sd = load_emp4(joinpath(@__DIR__, "data", "wti_spot_futures.csv"); h=2)
    x_init = X[1, :]
    dx_init = Z[1, 1:P]
    @printf("\nT=%d  p=%d  r=%d  p2=%d   %s .. %s\n",
            size(Y, 1), P, R, size(Z, 2), string(sd[1]), string(sd[end]))
end

if :D in WANT
    experiment_D(Y, X, Z, x_init, dx_init; ndata=NDATA_D, verbose=QUICK)
    flush(stdout)
end

# ---------------------------------------------------------------------------
# Section 6.2: the recursive null experiment
# ---------------------------------------------------------------------------

if :size in WANT
    specs = WANT_SPECS === nothing ? SIZE_SPECS :
            [(s, l) for (s, l) in SIZE_SPECS if s in WANT_SPECS]
    isempty(specs) && error("SPECS matched none of " *
                            join(string.(first.(SIZE_SPECS)), ", "))
    println("\n" * "=" ^ 78)
    @printf("RECURSIVE NULL EXPERIMENT, M = %d draws per specification\n", M_SIZE)
    println("=" ^ 78)
    println("The two cheap specifications come first, then the quadratic form, then")
    println("the log-precision pair, which is most of the time. Each draw is saved as")
    println("it finishes, so stopping here and resuming later costs one draw.\n")
    res = Any[]
    for (spec, label) in specs
        @printf("  %s\n", label); flush(stdout)
        free, one = sample_fits(Y, X, Z, spec; verbose=false)
        @printf("    l_free %12.4f   l_b=1 %12.4f   LR %7.3f\n",
                free[:loglik], one[:loglik],
                2 * (free[:loglik] - one[:loglik])); flush(stdout)
        push!(res, size_one(spec, label, Y, X, Z, free, one, x_init, dx_init;
                            M=M_SIZE, verbose=true))
        r = res[end]
        # @printf needs its format as one string literal, not a concatenation.
        @printf("    -> 10/5/1 %% = %.1f / %.1f / %.1f   (%d usable of %d, %d runaway)   %.0f s\n\n",
                100 * r.size10, 100 * r.size05, 100 * r.size01, r.n_used, r.M,
                r.n_improper, r.secs)
        flush(stdout)
    end
    report_size(res)
end

@printf("\nfinished at %s, %.1f minutes\n",
        Dates.format(now(), "HH:MM:SS"), (time() - t_start) / 60)
if QUICK
    println("\nThis was the QUICK pass, at a fraction of the replication counts.")
    println("The rejection rates and root mean squared errors above carry a large")
    println("simulation error and will not match the Python column closely. What")
    println("the pass does check is that every code path runs. Rerun with FULL=1")
    println("for the paper's numbers.")
end
