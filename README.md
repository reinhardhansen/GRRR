# Generalized Reduced Rank Regression: replication code

Replication material for P. R. Hansen, *Generalized Reduced Rank Regression*. Julia code and the assembled crude-oil data set.
Install the packages once:

```
julia -e 'import Pkg; Pkg.add(["Distributions", "CSV", "DataFrames", "Plots", "LaTeXStrings"])'
```

`LinearAlgebra`, `Statistics`, `Random`, `Printf` and `Dates` are standard
library. The earlier notebook-based replication, which covers the
estimator-validation checks of Table 1, is in `notebook/`.

## Julia replication, four-contract system

**Julia is the primary implementation.** Python is the independent second
implementation the numbers are checked against, not the other way round. Where
the two disagree, this is the one to trust, and the disagreement is a bug report
against the Python side.

```
cd GRRR                                  # the repository root
julia check_syntax.jl                    # 2 seconds, parses every file
julia test_gft.jl                        # 1 second, the log-precision theta-step
julia test_dgp.jl                        # 30 seconds, the simulation design
julia -t auto run_emp4.jl                # QUICK pass, a few minutes
julia -t auto run_sim4.jl                # QUICK pass, a few minutes
FULL=1 julia -t auto run_emp4.jl         # Section 6: the paper's B = 999, 1999
FULL=1 julia -t auto run_sim4.jl         # Section 5 and the size experiment
```

Run the three checks first. `check_syntax.jl` catches the errors that are
expensive to find halfway through a long run; the two test files check the new
mathematics against Python before the estimator is asked to use it.

## What it covers

Everything the paper reports. `run_emp4.jl` fits the four-contract system under
**all eight covariance specifications**, bootstraps **all seven rows of
Table empLR and all four columns of Table empBeta**, and compares every
log-likelihood against a recorded baseline. `run_sim4.jl` runs **Experiments B,
C and D of Section 5** and **the recursive null experiment of Section 6.2**,
which fills Table empSize and Figure disc. The figure itself is drawn
afterwards by `fig_disc.jl`, which only reads the files that experiment leaves
in `results4/`.

The recorded baselines came from the Python implementation, which is the second
implementation and not the authority: a disagreement is reported as a finding
and the run carries on, because results are saved as they are produced and
stopping costs nothing a resumed run will not recover.

| file | contents |
|---|---|
| `grrr_core.jl` | the estimator, extracted from the replication notebook. Three changes since: the ARCH θ-step builds its Newton operator `Σ Mₜ⊗Mₜ` as a single matrix product instead of a quadruple loop (the same matrix, about thirty times faster; `test_dgp.jl` checks the two agree); the same step solves its Newton system in the minimum-norm sense, see below; and `johansen_rrr` no longer adds a `1e-12` ridge to `S₁₁`. On the crude-oil data `S₁₁` has an eigenvalue of `2.7e-7`, and the ridge moved the benchmark log-likelihood from 6054.5765 to 6054.5761, which made the exact just-identified fit look like a mismatch. As a starting value the ridge was harmless, so no result changes. |
| `theta_scale.jl` | the scale covariance model, and a redefinition of `theta_step` that adds it to the dispatch chain. |
| `theta_gft.jl` | the log-precision model: the generalized Fisher transformation, its fixed-point inverse, and the exact gradient through the Frechet derivative of the matrix exponential. Redefines `theta_step` again, so it must be included after `theta_scale.jl`. |
| `emp4.jl` | the four-contract data, the Fourier seasonal block, the spread basis with F1 as numeraire, and the multi-start sample fits. |
| `boot4.jl` | the recursive wild bootstrap with a seasonal deterministic path, the machinery for the recursive null experiment, and the resume-and-append persistence every long run uses. |
| `dgp4.jl` | the Section 5 simulation design: the spread basis at a given `b`, the regime and ARCH-type covariance paths, and the VECM recursion. Include after `emp4.jl`, which is where `P`, `R` and the basis restriction come from. |
| `mc4.jl` | Experiments B, C and D of Section 5. Experiment B fits every estimator from three starting values built from the data alone and keeps the best; an earlier version started the iid fit at the true loadings, and its files carry the tags `_ms` and `_ms2` against `_ms3` for the current protocol. |
| `size4.jl` | the recursive null experiment for all seven specifications. For the two specifications whose proxy enters in levels (`scale`, `gft`) the DGP truncates the simulated proxy at its observed maximum, so that no path can run away; their result files carry the tag `_cap`, and the untruncated files are kept beside them. |
| `fig_disc.jl` | Figure disc, the P value discrepancy plot of the chi-squared test of `b₁ = b₂ = b₃ = 1`. It reads the LR values `size4.jl` leaves in `results4/`, draws the three curves the text discusses with the labels and line styles of the Python version of the figure, writes `GRRRfig_disc.pdf` beside the sources, and prints the rejection frequencies of all seven designs beside the Table empSize row they should reproduce. Nothing is estimated, so it runs in a second; a design whose file is missing is named and skipped. Needs `Plots` and `LaTeXStrings`, installed once with `julia -e 'import Pkg; Pkg.add(["Plots", "LaTeXStrings"])'`. |
| `test_gft.jl` | the log-precision theta-step against Python, objective and every gradient block. |
| `test_dgp.jl` | the simulation design and all three Experiment B estimators against Python, on innovations neither implementation drew. |
| `run_emp4.jl` | the Section 6 driver. |
| `run_sim4.jl` | the Section 5 and size-experiment driver. |
| `break_grid.jl` | the break-date sensitivity check for the regime specification, described below. |
| `sec61_tests.jl` | every number Section 6.1 reports: unit roots, cointegration rank by the trace test and by the recursive wild bootstrap, lag length, seasonal terms, and the seasonal grid. Described below. |

## Running one piece at a time

`run_sim4.jl` takes two environment variables, because the size experiment is
hours and the rest is minutes:

```
ONLY=B,C,D   FULL=1 julia -t auto run_sim4.jl              # Section 5, ~15 min
ONLY=size    FULL=1 julia -t auto run_sim4.jl              # the rest
ONLY=size SPECS=iid,regime,scale,scalelog  FULL=1 julia -t auto run_sim4.jl
```

`ONLY` takes any comma-separated subset of `B`, `C`, `D`, `size`. `SPECS`
restricts the size experiment to named specifications, which is how to get the
five cheap ones done in ten minutes and leave the log-precision pair for
overnight. Every replication is written to disk as it finishes and a rerun
resumes from what is there, so an interrupted run costs one replication.

Rough wall-clock on eight threads, at the paper's counts: Experiments B, C and
D about fifteen minutes together; the size experiment about ten minutes for
`iid`, `regime`, `scale` and `scalelog`, then about an hour each for
`archchol`, `gft` and `gftlog`.

## Are the break dates doing the work?

The two variance breaks of the regime specification, 2008:9 and 2020:3, are read
off the narrative rather than estimated, so `break_grid.jl` asks what the
rejection of `b₁ = b₂ = b₃ = 1` does when they move. It shifts each break
independently by up to six months, refits both the free and the `b = 1` fit at
all 169 pairs of dates, and prints one line per pair: the two shifts, the two
break dates, the LR and its chi-squared p-value on three degrees of freedom.

```
julia break_grid.jl                      # 169 refits, about a quarter of an hour
```

One fit runs at a time, so `-t auto` buys nothing here. The table is written to
`results4/break_grid_13x13.txt`, summary line included. Across the whole grid
the LR runs from 9.775 at a shift of `(+6, -1)` months to 15.965 at `(-4, +2)`,
with a median of 11.75 against 11.593 at the paper's own dates, and the largest
p-value anywhere on the grid is 0.0206. Nothing here turns on the exact months.

## The specification tests of Section 6.1

`sec61_tests.jl` produces every number the section reports, in the order it
reports them, on the screen and in `results4/sec61_tests.txt`.

```
julia -t auto sec61_tests.jl             # about two minutes on eight threads
GRID=0 julia -t auto sec61_tests.jl      # without the seasonal grid
B=99 julia -t auto sec61_tests.jl        # a quick pass of the rank bootstrap
```

Six blocks. Augmented Dickey-Fuller tests with a constant and the lag chosen by
AIC: the four log price levels do not reject a unit root, `p` between 0.44 and
0.50, and the three spreads reject at any level, `p` below 1e-9. The lag search
and the MacKinnon (1994) p-value function are ported from statsmodels and written
out in the file, coefficients included, so nothing in this package calls a Python
library. Johansen trace statistics for `r <= 0, ..., 3`, computed from the
eigenvalues of `johansen_rrr` on the same `(Y, X, Z)` the GRRR fits use, against
the Osterwald-Lenum quantiles: 45.57 against 15.49 at `r <= 2` and 1.63 against
3.84 at `r <= 3`, so `r = 3`. The recursive wild bootstrap rank test of Cavaliere,
Rahbek and Taylor (2014) at `B = 999`, which agrees: `p = 0.001` for the first
three nulls and about 0.53 for the fourth. Lag length on a common sample of 454
observations, where AIC and Hannan-Quinn pick `k = 2`, BIC picks `k = 1`, and the
Ljung-Box test of the `k = 1` residuals fails at `p = 0.001` where `k = 2` passes
at 0.50. The seasonal block at `k = 2` for six specifications, with the
likelihood ratio of `fourier2` against none at 35.0 on 16 degrees of freedom.

The last block is the one that answers the obvious objection. It reruns the test
of `b₁ = b₂ = b₃ = 1` for the two covariance specifications the paper quotes,
regime and scale with the proxy in levels, across five seasonal blocks: none,
quarter, `fourier2`, `fourier3` and monthly dummies. The seasonal block moves the
p-value by 0.005 under the regime covariance and by 0.053 under the scale
covariance. The regime specification rejects at five per cent in all five, the
scale specification accepts in all five, and the gap between the two is never
smaller than 0.479. The seasonal specification is not what decides the test.

The bootstrap block is threaded and seeded per replication, so its answer does
not depend on the thread count. It cannot match Python replication for
replication, for the reason given below, and the file says so where it prints.

## What the two implementations can and cannot be held to

For the empirical section, everything. The sample fits are deterministic, the
bootstrap is seeded, and the log-likelihoods agree to 5e-05.

For Section 5 and the size experiment, not everything, and the reason is not a
defect. numpy's default generator is PCG64 with a ziggurat normal and Julia's is
Xoshiro256++, so no seed makes the two draw the same numbers, and the Monte
Carlo tables can only agree to within simulation error. At 500 replications that
is about three per cent on a root mean squared error; at 400 draws it is 1.5
percentage points on an estimated rejection rate of ten per cent. Both drivers
print the Python column beside their own so the comparison is there to read.

What can be compared exactly is everything that is not random, and `test_dgp.jl`
does compare it: the design constants, both covariance paths, the VECM
recursion, and the b-hat all three estimators return when the innovations come
from a file rather than from a generator. If that passes and a table still
differs, the difference is simulation error.

`design_reference.txt` holds those reference values. It is regenerated by
`mk_design_reference.py` in the Python4 folder. The innovation covariance of the
design is written out in `dgp4.jl` rather than regenerated, because it is a
constant of the experiment and not a draw: rebuilding it from a Julia generator
would give a different matrix and therefore a different experiment.

## The log-precision model

`theta_gft.jl` carries the correlation matrix through the transformation of
Archakov and Hansen (2021): `C = γ⁻¹(g)` where `γ(C) = vecl(log C)` is a
bijection onto `R^{p(p−1)/2}`. Every parameter value gives a positive definite
`Ω(t)` at every driver value, so there is no cone, no boundary and no
constrained optimisation.

The delicate part is `dℓ/dg`. `C = exp(A)` with `A = Off(g) + diag(x)`, and `x`
is fixed implicitly by `diag(C) = 1`, so the constraint has to be differentiated
and eliminated. `test_gft.jl` checks the result three ways: against central
differences, against the Python implementation's objective and gradient at a
common point, and on the round-trip `γ⁻¹(γ(C)) = C`.

That test runs at **q = 2** deliberately. The `b` block is stored row by row in
numpy and column by column in Julia; at `q = 1`, which is what the empirical
model uses, the two orders coincide, so a `q = 1` test would pass with the
indexing wrong. `gft_reference.txt` is regenerated by `mk_gft_reference.py` in
the Python4 folder.

## The ARCH-type design, and what it does not identify

The Section 5 ARCH design uses a driver `vₜA` with `A` of rank `q = 3`. Under
any such design `D` is identified only up to a flat subspace: `D₁₂ ↦ D₁₂ + AK`
with `K` skew leaves every `Ω(t)⁻¹ = Qₜ'DQₜ` unchanged, so the likelihood is
exactly constant along `q(q−1)/2` directions, six under the proportional design
`q = p`, three here, and none only for the scalar driver `q = 1` of the empirical
application. An earlier version of this README and of the paper said that
taking `q < p` identified `D`. It does not; a referee caught it. `Ω(t)`, the
likelihood and `b̂` are identified in every case, which is why Experiment B,
whose object is `b̂`, was never affected.

`theta_arch` now solves its Newton system in the minimum-norm sense on the
space of symmetric matrices, so the step has no component along the flat
directions, and records the dimension it found. `test_dgp.jl` computes that
dimension from the map itself and from the fit and checks both against
`q(q−1)/2`, and checks that the fit is invariant to the sign convention of the
eigensolver, which is the one thing about the design that is genuinely
arbitrary. Experiment B logs the dimension on every replication.

## Reference values it checks against

| Σ | ℓ free | ℓ under b = 1 |
|---|---|---|
| iid | 6054.5765 | 6050.8966 |
| regime | 6402.0529 | 6396.2563 |
| quadratic, D unrestricted | 6209.4137 | 6204.4114 |
| quadratic, D = LL' | 6209.4137 | 6204.4114 |
| scale, driver in logs | 6211.1146 | 6209.3834 |
| log-precision, driver in logs | 6268.3376 | 6267.0312 |
| scale, driver in levels | 6406.4339 | 6405.2891 |
| log-precision, driver in levels | 6430.6578 | 6429.7169 |

Bootstrap p-values at B = 999: iid 0.076, regime 0.013, quadratic 0.096, scale
in logs 0.340, log-precision in logs 0.512, scale in levels 0.489,
log-precision in levels 0.568. The QUICK pass uses B = 99 and will not match
these closely; its simulation standard error is about 0.03.

Standard errors run at B = 1999, not 499. The bootstrap distribution of `b̂` has
a standard deviation about 1.3 times what its interquartile range would imply
under normality, and at B = 499 two runs of the identical procedure differed by
13 per cent in the third digit.

## Threads

`julia -t auto` uses every core, and `BLAS.set_num_threads(1)` at the top of
each driver gives each worker thread a single BLAS thread, which is what made
the port scale. Each θ-step allocates its scratch inside the step rather than at
module level, so the threads do not share buffers.
