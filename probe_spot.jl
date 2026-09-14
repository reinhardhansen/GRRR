# Front contract or spot price as the numeraire: the same eight fits on both.
#
#   julia -t auto probe_spot.jl                # sample fits only, a few minutes
#   B=999 julia -t auto probe_spot.jl          # add the iid and regime LR bootstraps
#   B=999 BOOT=iid,regime,archchol julia -t auto probe_spot.jl
#
# Section 2.2 of the paper writes the cost-of-carry basis as F_it − S_t and then
# lets the front contract stand in for the spot, because the two are the same
# series in all but name. This script asks what changes if the spot is used
# instead: Y = (S, F2, F3, F4) rather than (F1, F2, F3, F4), so that relation i
# is F_{i+1} − b_i S, the basis exactly as 2.2 defines it.
#
# Both systems are loaded explicitly, so the NUMERAIRE environment variable is
# irrelevant here and nothing is written to disk. The bootstrap, if asked for,
# runs with save=false for the same reason: a probe should leave no files that
# a later full run could mistake for its own.

using LinearAlgebra, Statistics, Random, Distributions, Printf, CSV, DataFrames, Dates

BLAS.set_num_threads(1)

include("grrr_core.jl")
include("theta_scale.jl")
include("theta_gft.jl")
include("emp4.jl")
include("boot4.jl")

const B_PROBE = parse(Int, get(ENV, "B", "0"))
const BOOT_SPECS = Symbol.(split(get(ENV, "BOOT", "iid,regime"), ','))
const DATA = joinpath(@__DIR__, "data", "wti_spot_futures.csv")

const SPECS = [(:iid, "iid"), (:regime, "regime (2008:9, 2020:3)"),
               (:arch, "quadratic, D unrestricted"),
               (:archchol, "quadratic, D = LL'"),
               (:scalelog, "scale, driver in logs"),
               (:gftlog, "log-precision, driver in logs"),
               (:scale, "scale, driver in levels"),
               (:gft, "log-precision, driver in levels")]

@printf("threads: %d\n\n", Threads.nthreads())

# --- the two series themselves ----------------------------------------------
df = CSV.read(DATA, DataFrame)
ls, l1 = Float64.(df.l_spot), Float64.(df.l_F1)
d = ls .- l1
rs, r1 = diff(ls), diff(l1)
println("spot against front contract, levels and monthly log returns")
@printf("  log spot − log F1: sd %.4f, median gap %.3f%% of price, equal to the cent in %d of %d months\n",
        std(d), 100 * median(abs.(expm1.(d))), count(abs.(d) .< 1e-9), length(d))
@printf("  returns: correlation %.4f; the difference is %.1f%% the size of the F1 return\n\n",
        cor(rs, r1), 100 * std(rs .- r1) / std(r1))

# --- the eight fits under each numeraire -------------------------------------
sys = Dict{Symbol,Any}()
for nm in (:F1, :spot)
    Y, X, Z, sd = load_emp4(DATA; h=2, numeraire=nm)
    sys[nm] = (Y=Y, X=X, Z=Z, x_init=X[1, :], dx_init=Z[1, 1:P], sd=sd)
end
@printf("T = %d in both systems, %s .. %s\n\n", size(sys[:F1].Y, 1),
        string(sys[:F1].sd[1]), string(sys[:F1].sd[end]))

fits = Dict{Tuple{Symbol,Symbol},Any}()
println("=" ^ 100)
println("SAMPLE FITS: front contract as numeraire  |  spot as numeraire")
println("=" ^ 100)
@printf("%-30s %10s %10s %8s %7s   | %10s %10s %8s %7s\n", "specification",
        "l free", "l b=1", "LR", "chi2 p", "l free", "l b=1", "LR", "chi2 p")
for (spec, label) in SPECS
    row = String[]
    for nm in (:F1, :spot)
        s = sys[nm]
        try
            free, one = sample_fits(s.Y, s.X, s.Z, spec; verbose=false)
            fits[(nm, spec)] = (free=free, one=one)
            LR = 2 * (free[:loglik] - one[:loglik])
            push!(row, @sprintf("%10.4f %10.4f %8.3f %7.4f", free[:loglik],
                                one[:loglik], LR, ccdf(Chisq(R), LR)))
        catch err
            fits[(nm, spec)] = nothing
            push!(row, @sprintf("%-38s", "FAILED: " * first(sprint(showerror, err), 28)))
        end
    end
    @printf("%-30s %s   | %s\n", label, row[1], row[2]); flush(stdout)
end

println("\nthe point estimates b̂ = (b₁, b₂, b₃), null is (1, 1, 1)")
@printf("%-30s %30s   | %30s\n", "specification", "front contract", "spot")
for (spec, label) in SPECS
    row = String[]
    for nm in (:F1, :spot)
        f = fits[(nm, spec)]
        push!(row, f === nothing ? @sprintf("%30s", "") :
              join([@sprintf("%9.4f", x) for x in b_of(f.free)], " "))
    end
    @printf("%-30s %30s   | %30s\n", label, row[1], row[2])
end

# --- the bootstrap, if asked for ---------------------------------------------
if B_PROBE > 0
    println("\n" * "=" ^ 100)
    @printf("LR BOOTSTRAP, B = %d, Rademacher weights, nothing saved\n", B_PROBE)
    println("=" ^ 100)
    @printf("%-30s %8s %9s %9s   | %8s %9s %9s\n", "specification",
            "LR", "boot p", "95th", "LR", "boot p", "95th")
    for (spec, label) in SPECS
        spec in BOOT_SPECS || continue
        row = String[]
        for nm in (:F1, :spot)
            f = fits[(nm, spec)]
            if f === nothing
                push!(row, @sprintf("%-30s", "no sample fit")); continue
            end
            s = sys[nm]
            LR0 = 2 * (f.free[:loglik] - f.one[:loglik])
            t0 = time()
            LR, _ = run_bootstrap(spec, s.Y, s.X, s.Z, f.free, f.one, s.x_init, s.dx_init;
                                  B=B_PROBE, kind=:lr, verbose=false, save=false)
            push!(row, @sprintf("%8.3f %9.4f %9.2f  (%.0f s)", LR0,
                                boot_pvalue(LR, LR0), quantile(LR, 0.95), time() - t0))
        end
        @printf("%-30s %s   | %s\n", label, row[1], row[2]); flush(stdout)
    end
end

println("\ndone. Nothing was written to results4 or results4_spot.")
