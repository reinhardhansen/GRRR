# Parse every file without executing anything. Two seconds, and it catches the
# one class of error that is expensive to discover halfway through a long run.
#
#   julia check_syntax.jl
#
# `Meta.parseall` builds the syntax tree and reports errors as :error nodes
# instead of throwing, so one pass reports every bad file rather than stopping
# at the first.

const FILES = ["grrr_core.jl", "theta_scale.jl", "theta_gft.jl", "emp4.jl",
               "boot4.jl", "dgp4.jl", "mc4.jl", "size4.jl",
               "run_emp4.jl", "run_sim4.jl", "lrstar_gen.jl", "probe_spot.jl",
               "test_gft.jl", "test_dgp.jl"]

bad = 0
for f in FILES
    path = joinpath(@__DIR__, f)
    if !isfile(path)
        println("MISSING  $f")
        global bad += 1
        continue
    end
    ex = Meta.parseall(read(path, String); filename=f)
    errs = filter(a -> a isa Expr && (a.head === :error || a.head === :incomplete),
                  ex.args)
    if isempty(errs)
        println("ok       $f")
    else
        println("SYNTAX   $f")
        for e in errs
            println("           ", e)
        end
        global bad += length(errs)
    end
end

println()
if bad == 0
    println("all files parse. Next:  julia test_gft.jl && julia test_dgp.jl")
    println("                  then: julia -t auto run_emp4.jl")
    println("                    and julia -t auto run_sim4.jl")
else
    println("$bad problem(s); fix these before running anything.")
    exit(1)
end
