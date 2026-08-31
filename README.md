# Generalized Reduced Rank Regression — replication code

Replication material for P. R. Hansen, *Generalized Reduced Rank Regression*
(revised 2026).

The GRRR estimates a multivariate regression that combines reduced rank, affine
restrictions on the regression parameters, and a general parametrized covariance
`Σ(θ)`, by maximizing a Gaussian quasi-likelihood in a three-block switching
algorithm. This repository contains a self-contained Julia implementation, the
assembled crude-oil data, and the code that produces most of the paper's tables.

## What is here, and what is not

Reproduced by the code in this repository:

| Paper | Contents |
|---|---|
| Section 4 | the four estimator-validation checks |
| Tables 1–3 | Experiments A, B and C |
| Figure 1 | WTI log prices and the spot–futures bases |
| Tables 5, 7, 8 | log-likelihoods, basis coefficients, LR tests, and the bootstrap |

Not yet posted, and to be added here:

- Table 4, the multi-start study at the empirical design
- Table 6 and Figure 2, the finite-sample null distribution
- the Johansen trace statistics and the wild-bootstrap rank test of Section 6.1
- the positive-definiteness diagnostic and the break-date sensitivity study of
  Section 6.2

## Requirements

Julia 1.11 or later. Install the packages once:

```julia
import Pkg
Pkg.add(["Distributions", "CSV", "DataFrames", "Plots", "LaTeXStrings", "BenchmarkTools"])
```

`LinearAlgebra`, `Statistics`, `Random` and `Printf` are standard library.

## Running it

From this directory, so that `data/wti_spot_futures.csv` resolves:

```bash
GRRR_QUICK=true julia -t auto run_all.jl 2>&1 | tee quick.log   # reduced counts
julia -t auto run_all.jl 2>&1 | tee run.log                     # paper counts
```

`-t auto` matters. Experiments B and C, the break-date profile and the bootstrap
are all threaded, and `Threads.@threads` runs a loop serially and silently when
the process has one thread. The script checks this at startup, reports how many
threads actually did work, and refuses to start a full run single-threaded.

The same code is in `GRRR_Replication_bootstrap.ipynb`. A Jupyter kernel does
**not** inherit `julia.NumThreads` from VS Code; install a threaded kernel once:

```julia
using IJulia
installkernel("Julia 8 threads", env=Dict("JULIA_NUM_THREADS" => "8"))
```

## Runtime

Everything through the empirical estimates takes about ten minutes at
`GRRR_QUICK=true` on eight threads. The bootstrap is the expensive part, and the
ARCH-type specification dominates it: at `B = 99` that one arm has run for over
an hour on eight threads, so the full `B = 999` in the paper is a matter of many
hours. Run it detached:

```bash
nohup julia -t auto run_all.jl > run.log 2>&1 &
tail -f run.log
```

## Reproducibility

Every stream is seeded from one constant, `SEED = 19510104`, in separate blocks
so that no two streams overlap.

Quantities that do not depend on random draws reproduce the paper exactly: the
maximized log-likelihoods, the likelihood-ratio statistics, the basis
coefficients, the regime variances, and the data-chosen break date. Monte Carlo
quantities reproduce only up to sampling error, because the paper's tables were
produced by a numpy implementation whose generator differs from Julia's.

The T = 100 rows of Table 2 are dominated by a few extreme replications and
should not be compared closely: in the ARCH design one replication carries 64%
of the iid estimator's squared error, and in the regime design the sign of the
efficiency gain changes with the seed. The median absolute error is stable and
is the right column to check.

## Data

`data/wti_spot_futures.csv` holds monthly West Texas Intermediate log prices from
March 1986 to April 2024, T = 458. The spot series is `MCOILWTICO` from FRED; the
one- through four-month futures are `RCLC1`–`RCLC4` from the U.S. Energy
Information Administration, which discontinued the series in April 2024.
`data/build_and_diagnose.py` documents how the file was assembled from the raw
downloads in `data/raw/`.

## License

MIT, see `LICENSE`.
