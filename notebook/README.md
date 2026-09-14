# Earlier notebook-based replication

`GRRR_Replication_bootstrap.ipynb`, with `run_all.jl` as its plain-script
version and `setup_julia.jl` for the packages, is the earlier replication on
the five-variable system that included the spot price. It is kept for the
estimator-validation checks of Table 1 of the paper, which do not depend on the
empirical design. Everything else in the paper is produced by the scripts in
the repository root; see the README there.

```
cd notebook
GRRR_QUICK=true julia -t auto run_all.jl     # reduced counts, a few minutes
julia -t auto run_all.jl                     # full counts
```

`data/wti_spot_futures.csv` here is the same file as in `../data/`.
