# The P value discrepancy plot of Figure disc, drawn from the recursive null
# experiment of size4.jl.
#
#   cd .../GRRR/Julia4
#   julia fig_disc.jl                # a second or two: it reads and draws
#
# For each design, p_j = P(chi2_3 > LR_j) over the M samples simulated under the
# null as in Table empSize, and the figure plots Fhat(x) - x against x, where
# Fhat(x) = (1/M) sum_j 1(p_j <= x). A correctly sized test lies on zero and
# positive values are over-rejection. The grey curves are plus and minus two
# simulation standard errors at the smallest M. The construction is the one
# Davidson and MacKinnon (1998, Manchester School) recommend.
#
# Nothing about the model is computed here. The LR values are on disk, one line
# per draw, written by size4.jl as
#
#     j  LR  won_one  won_free  spread  status
#
# with status 0 for a usable draw, 1 for a path that ran away and 2 for a fit
# that threw, and only the usable draws with a finite LR are used. The reader
# below is `mc_load` from boot4.jl at nfield = 5, copied rather than included:
# this file reads text, and including size4.jl would pull in the estimator.
#
# The file name carries M, so whichever M is on disk is used and the file is
# named in the printout. For the two specifications whose volatility proxy
# enters in levels the name also carries the tag `_cap`, as `size_path` in
# size4.jl builds it. Those two are read from the truncated run when it is
# there, which is what Table empSize reports, and from the untruncated run
# otherwise, with a line saying so.
#
# The figure draws the three curves the text discusses, in the order, with the
# labels and with the line styles of mk_fig_disc4.py in the Python folder, so
# that the two figures differ only in the plotting library. The check printed on
# the way covers all seven designs of Table empSize, whose rejection frequencies
# are
#
#   specification                      0.10   0.05   0.01
#   iid                               0.118  0.055  0.013
#   regime (2008:9, 2020:3)           0.140  0.068  0.023
#   quadratic, D = LL'                0.138  0.078  0.008
#   scale, u = log v                  0.118  0.060  0.008
#   log-precision, u = log v          0.115  0.048  0.020
#   scale, u = v (capped)             0.105  0.050  0.005
#   log-precision, u = v (capped)     0.118  0.053  0.010
#
# The five uncapped rows come out of the files in results4/ exactly as they
# stand. The two capped rows need the truncated run. The untruncated files kept
# beside them lose 190 of 400 draws for scale and 42 of 400 for log-precision,
# and their survivors give 0.1095 / 0.0524 / 0.0048 on M = 210 and
# 0.1145 / 0.0531 / 0.0112 on M = 358, which is the survivor comparison the text
# makes rather than the table's row.
#
# The grid is 200 points from x = 0.002 to x = 0.20, which is the grid the
# Python script uses and the range the figure shows: the levels anyone reads a
# test at are inside it, and at M = 400 the left end is already as fine as the
# experiment can resolve, one draw being 0.0025.

ENV["GKSwstype"] = "100"        # GR writes the PDF with no window to draw into

using Plots, Distributions, Printf
using LaTeXStrings       # for the axis labels; Pkg.add("LaTeXStrings") once

const DF = 3                    # restrictions in b1 = b2 = b3 = 1
const RESDIR = joinpath(@__DIR__, "results4")
const OUTPATH = joinpath(@__DIR__, "GRRRfig_disc.pdf")
const CAPPED = (:scale, :gft)   # the two whose proxy enters in levels
const XS = range(0.002, 0.20, length=200)     # the grid mk_fig_disc4.py uses

# The seven designs of Table empSize, in the order the table lists them, each
# with the rejection frequencies the table reports at 10, 5 and 1 per cent.
const TABLE_SPECS = [
    (:iid,      "iid",                           (0.118, 0.055, 0.013)),
    (:regime,   "regime (2008:9, 2020:3)",       (0.140, 0.068, 0.023)),
    (:archchol, "quadratic, D = LL'",            (0.138, 0.078, 0.008)),
    (:scalelog, "scale, u = log v",              (0.118, 0.060, 0.008)),
    (:gftlog,   "log-precision, u = log v",      (0.115, 0.048, 0.020)),
    (:scale,    "scale, u = v (capped)",         (0.105, 0.050, 0.005)),
    (:gft,      "log-precision, u = v (capped)", (0.118, 0.053, 0.010))]

# The three curves the figure draws, with the labels, the line styles and the
# line widths of mk_fig_disc4.py. The ARCH-type entry is the quadratic form of
# Table empLR, estimated under D = LL' as everywhere else.
const PLOT_SPECS = [(:iid,      "iid Σ",       :solid, 1.6),
                    (:regime,   "regime Σ",    :dash,  1.4),
                    (:archchol, "ARCH-type Σ", :dot,   1.8)]

frac(p, x) = count(<=(x), p) / length(p)

"""Replications on disk as j => row, which is `mc_load` of boot4.jl at
nfield = 5. A line torn by an interrupt is skipped rather than fatal."""
function size_rows(path)
    out = Dict{Int,Vector{Float64}}()
    isfile(path) || return out
    for ln in readlines(path)
        f = split(strip(ln))
        length(f) == 6 || continue
        try
            out[parse(Int, f[1])] = [parse(Float64, f[1 + i]) for i in 1:5]
        catch
            continue
        end
    end
    return out
end

"""The result file for one specification, or `nothing` if there is none.

The name carries M, so take whichever M is on disk, the largest if there are
several. For the two capped specifications the truncated file is preferred and
the untruncated one is the fallback."""
function size_file(spec)
    isdir(RESDIR) || return nothing
    pre = "size_$(spec)_M"
    want = spec in CAPPED
    best = nothing
    bestkey = (-1, -1)
    for f in readdir(RESDIR)
        startswith(f, pre) && endswith(f, ".txt") || continue
        body = f[length(pre)+1:end-4]
        tag = endswith(body, "_cap")
        M = tryparse(Int, tag ? body[1:end-4] : body)
        M === nothing && continue
        key = (tag == want ? 1 : 0, M)
        if key > bestkey
            best = f
            bestkey = key
        end
    end
    return best === nothing ? nothing : joinpath(RESDIR, best)
end

"""The chi-squared p values of one design, with the file they came from and the
draw counts, or `nothing` when the file is not there."""
function design_pvalues(spec)
    path = size_file(spec)
    if path === nothing
        @printf("  no result file for %s in %s: looked for size_%s_M<M>.txt\n",
                spec, RESDIR, spec)
        println("  skipping this design; to produce it run",
                "  ONLY=size FULL=1 julia -t auto run_sim4.jl")
        return nothing
    end
    rows = size_rows(path)
    js = sort(collect(keys(rows)))
    st = [rows[j][5] for j in js]
    LR = [rows[j][1] for j in js if rows[j][5] == 0.0 && isfinite(rows[j][1])]
    return (p=ccdf.(Chisq(DF), LR), file=basename(path), nrow=length(js),
            naway=count(==(1.0), st), nerr=count(==(2.0), st))
end

# ---------------------------------------------------------------------------
# the seven designs of Table empSize
# ---------------------------------------------------------------------------

println("reading the Julia size experiment from ", RESDIR)
@printf("chi2(%d) critical values %.2f / %.2f / %.2f\n\n",
        DF, quantile(Chisq(DF), 0.90), quantile(Chisq(DF), 0.95),
        quantile(Chisq(DF), 0.99))

# Four decimals on this run's frequencies, because M = 400 makes them multiples
# of 0.0025 and the table's three decimals hide which way a half rounded.
@printf("%-32s%-26s%6s%9s%9s%9s   %s\n", "specification", "file", "M",
        "0.10", "0.05", "0.01", "Table empSize 10 / 5 / 1")

have = Dict{Symbol,NamedTuple}()
for (spec, label, ref) in TABLE_SPECS
    d = design_pvalues(spec)
    d === nothing && continue
    if isempty(d.p)
        @printf("%-32s%-26s   no usable draw on disk\n", label, d.file)
        continue
    end
    have[spec] = d
    @printf("%-32s%-26s%6d%9.4f%9.4f%9.4f%9.3f%7.3f%7.3f\n", label, d.file,
            length(d.p), frac(d.p, 0.10), frac(d.p, 0.05), frac(d.p, 0.01),
            ref[1], ref[2], ref[3])
    if spec in CAPPED && !endswith(d.file, "_cap.txt")
        @printf("    untruncated run: %d of %d paths ran away and are dropped, so this\n",
                d.naway, d.nrow)
        println("    row is the survivor comparison of the text, not the table's row")
    end
    d.nerr > 0 && @printf("    %d draw(s) threw during estimation and are dropped\n",
                          d.nerr)
end

# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------

# Sizes in pixels at the GR default of 100 per inch, so (640, 360) is the
# 6.4 by 3.6 inches of the Python figure. `framestyle = :axes` keeps the left
# and bottom lines only, as the Python does by hiding the top and right spines,
# and the grid is off because matplotlib draws none.
plt = plot(size=(640, 360), framestyle=:axes, grid=false,
           xlabel=L"nominal level $x$", ylabel=L"\hat{F}(x) - x",
           xlims=(0.0, 0.20), xticks=0.0:0.05:0.20,
           legend=:topleft, legendfontsize=9,
           guidefontsize=10, tickfontsize=9,
           foreground_color_legend=nothing, background_color_legend=nothing)

Ms = Int[]
for (spec, lab, ls, lw) in PLOT_SPECS
    haskey(have, spec) || continue
    p = have[spec].p
    push!(Ms, length(p))
    disc = [frac(p, x) - x for x in XS]
    leg = @sprintf("%s  (M = %d)", lab, length(p))
    plot!(plt, XS, disc, linecolor=:black, linestyle=ls, linewidth=lw,
          label=leg)
end

if isempty(Ms)
    println("\nno result file for any of the three curves, so no figure written")
else
    # The band is two simulation standard errors of Fhat(x) at the smallest M
    # among the curves drawn, which is the widest of the three bands.
    band = [2 * sqrt(x * (1 - x) / minimum(Ms)) for x in XS]
    plot!(plt, XS, band, linecolor="gray60", linewidth=0.8, label="")
    plot!(plt, XS, -band, linecolor="gray60", linewidth=0.8, label="")
    hline!(plt, [0.0], linecolor=:black, linewidth=0.6, label="")
    savefig(plt, OUTPATH)

    # The numbers the text quotes about the figure, computed rather than read
    # off the picture.
    println("\ndiscrepancy Fhat(x) - x, and where each curve first clears two")
    println("simulation standard errors")
    @printf("%-16s%10s%9s%9s%9s%12s\n", "curve", "x = 0.01", "0.05", "0.10",
            "0.20", "clears at")
    for (spec, lab, _, _) in PLOT_SPECS
        haskey(have, spec) || continue
        p = have[spec].p
        disc = [frac(p, x) - x for x in XS]
        b = [2 * sqrt(x * (1 - x) / length(p)) for x in XS]
        k = findfirst(i -> disc[i] > b[i], eachindex(disc))
        v = [disc[argmin(abs.(XS .- a))] for a in (0.01, 0.05, 0.10, 0.20)]
        @printf("%-16s%10.3f%9.3f%9.3f%9.3f%12.3f\n", lab, v[1], v[2], v[3],
                v[4], k === nothing ? NaN : XS[k])
    end
    println("\nwrote ", OUTPATH)
end
