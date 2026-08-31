# Run once, from this directory:   julia setup_julia.jl
#
# Creates a pinned environment for the replication notebook (Project.toml and
# Manifest.toml land next to this file) and registers a Jupyter kernel that
# always activates it. Use this if the existing "Julia 1.12" kernel fails to
# start: that kernelspec holds a hard-coded path to a particular julia binary,
# which breaks whenever juliaup moves or removes that version.

using Pkg

Pkg.activate(@__DIR__)

Pkg.add([
    "Distributions",     # DGP draws
    "CSV",               # data/wti_spot_futures.csv
    "DataFrames",
    "Plots",             # Figure 1
    "LaTeXStrings",
    "BenchmarkTools",    # timing section
])

Pkg.add("IJulia")
using IJulia
IJulia.installkernel("GRRR", "--project=$(@__DIR__)")

Pkg.status()
println("""

Done. Reload the VS Code window (Cmd-Shift-P, "Developer: Reload Window"),
then pick the kernel named "GRRR" for GRRR_Replication_bootstrap.ipynb.

NOTE ON DURABILITY. installkernel writes the *versioned* julia binary into
kernel.json, so this kernel will break again the next time juliaup rolls the
release channel forward. To make it permanent, repoint it at the juliaup
launcher, which always follows the default channel:

    python3 - <<'EOF'
    import json, pathlib
    for kj in (pathlib.Path.home()/"Library/Jupyter/kernels").glob("*/kernel.json"):
        k = json.loads(kj.read_text())
        if "julia" in k.get("language", ""):
            k["argv"][0] = str(pathlib.Path.home()/".juliaup/bin/julia")
            kj.write_text(json.dumps(k, indent=1))
            print("repointed", kj)
    EOF

For the bootstrap cells, start Julia with threads if you run it outside VS Code:
    julia -t auto
In VS Code, set julia.NumThreads in settings, or export JULIA_NUM_THREADS=auto
before launching. Without threads the ARCH bootstrap at B_LR = 999 takes hours;
set B_LR = B_SE = 99 in the cell above it for a first pass.
""")
