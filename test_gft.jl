# Numerical test of the log-precision θ-step against the Python implementation.
#
#   julia test_gft.jl
#
# Needs no data, no packages beyond the standard library, and takes a second.
# Run it before run_emp4.jl: if the port is wrong, this says which piece, where
# run_emp4.jl would only say that a log-likelihood is off.
#
# Four checks:
#   1. γ and γ⁻¹ invert each other on a random correlation matrix.
#   2. C_of_gamma returns a genuine correlation matrix: unit diagonal, positive
#      definite.
#   3. the analytic dℓ/dg agrees with central differences.
#   4. the objective and all three gradient blocks agree with the values the
#      Python implementation produced at the same point, read from
#      gft_reference.txt.
#
# The fourth is the one that matters. It uses q = 2, because the b block is
# stored row by row in numpy and column by column in Julia and at q = 1 the two
# orders coincide, so a q = 1 test would pass with the indexing wrong.

using LinearAlgebra, Statistics, Random, Printf

_sym(A) = (A + A') / 2
logdet_pd(A) = logdet(cholesky(Symmetric(A)))
function safe_logdet(A)
    F = cholesky(Symmetric(A); check=false)
    issuccess(F) ? (logdet(F), true) : (0.0, false)
end
# theta_gft.jl closes with a redefinition of theta_step that names the core's
# other θ-steps. They are not defined here and this file never calls it, but the
# method body still has to resolve, so give them a stub each.
for f in (:theta_iid, :theta_regime, :theta_arch, :theta_arch_chol,
          :theta_scale, :theta_ar1, :theta_fixed)
    @eval $f(E, ctx) = error("stub: test_gft.jl does not run the estimator")
end

include("theta_gft.jl")

const TOL_ROUNDTRIP = 1e-10
const TOL_GRAD      = 1e-6
const TOL_PYTHON    = 1e-8

fails = 0
report(ok, name, detail) = begin
    @printf("%-46s %s   %s\n", name, ok ? "ok  " : "FAIL", detail)
    ok || (global fails += 1)
end

println("=" ^ 78)
println("LOG-PRECISION θ-STEP, JULIA AGAINST PYTHON")
println("=" ^ 78)

# --- 1. the transformation round-trips -------------------------------------
rng = MersenneTwister(20260903)
let p = 4
    A = randn(rng, p, p)
    S = A * A' + p * I
    d = sqrt.(diag(S))
    C0 = _sym(S ./ (d * d'))
    for i in 1:p
        C0[i, i] = 1.0
    end
    g = gamma_of_C(C0)
    C1, x, _, _ = C_of_gamma(g, p)
    e = maximum(abs.(C1 .- C0))
    report(e < TOL_ROUNDTRIP, "1. γ⁻¹(γ(C)) = C", @sprintf("max |ΔC| = %.2e", e))

    ud = maximum(abs.(diag(C1) .- 1.0))
    pd = minimum(eigvals(Symmetric(C1)))
    report(ud < 1e-12 && pd > 0, "2. γ⁻¹ returns a correlation matrix",
           @sprintf("max |diag−1| = %.2e, min eig = %.3f", ud, pd))
end

# --- 3. analytic gradient against finite differences ------------------------
let
    rel = check_gft_gradient(; p=4, q=1, T=200, seed=20260903)
    report(rel < TOL_GRAD, "3. dℓ/dg analytic vs finite differences",
           @sprintf("max relative error = %.2e", rel))
end

# --- 4. against the Python reference ----------------------------------------
const REFFILE = joinpath(@__DIR__, "gft_reference.txt")
if !isfile(REFFILE)
    println("\ngft_reference.txt is missing; regenerate it with")
    println("    python mk_gft_reference.py        (in the Python4 folder)")
    global fails += 1
else
    lines = readlines(REFFILE)
    hdr = parse.(Int, split(strip(lines[1])))
    T, p, q = hdr[1], hdr[2], hdr[3]
    vals = parse.(Float64, lines[2:end])
    pos = Ref(0)
    nextvals(n) = (pos[] += n; vals[pos[] - n + 1:pos[]])

    # every array was written row by row, so read into the transposed shape and
    # permute rather than reshaping directly, Julia filling columns first
    E = permutedims(reshape(nextvals(T * p), p, T))
    Xflat = nextvals(T * p * q)
    Xex = Vector{Matrix{Float64}}(undef, T)
    # index arithmetic rather than a running counter: a `for` loop in a script
    # is a hard scope, and a counter from the enclosing scope would need `global`
    for t in 1:T
        M = Matrix{Float64}(undef, p, q)
        for i in 1:p, j in 1:q
            M[i, j] = Xflat[(t - 1) * p * q + (i - 1) * q + j]
        end
        Xex[t] = M
    end
    nz = p + p * q + p * (p - 1) ÷ 2
    z      = nextvals(nz)
    f_ref  = nextvals(1)[1]
    ga_ref = nextvals(p)
    gb_ref = nextvals(p * q)          # row major
    gg_ref = nextvals(p * (p - 1) ÷ 2)
    C_ref  = permutedims(reshape(nextvals(p * p), p, p))
    pos[] == length(vals) ||
        error("gft_reference.txt has $(length(vals)) numbers, $(pos[]) were read")

    out = gft_pieces(z, p, q, Xex, E, nothing)
    out === nothing && error("gft_pieces returned nothing at the reference point")
    gg = grad_g(p, out.G_C, out.λ, out.U)

    ef = abs(out.f - f_ref) / max(abs(f_ref), 1.0)
    report(ef < TOL_PYTHON, "4a. objective ℓ",
           @sprintf("julia %.10f, python %.10f, rel %.2e", out.f, f_ref, ef))

    ea = maximum(abs.(out.ga .- ga_ref)) / max(maximum(abs.(ga_ref)), 1.0)
    report(ea < TOL_PYTHON, "4b. dℓ/da", @sprintf("max rel %.2e", ea))

    eb = maximum(abs.(vec_rowmajor(out.gb) .- gb_ref)) / max(maximum(abs.(gb_ref)), 1.0)
    report(eb < TOL_PYTHON, "4c. dℓ/db  (q = 2, so the b ordering is tested)",
           @sprintf("max rel %.2e", eb))

    eg = maximum(abs.(gg .- gg_ref)) / max(maximum(abs.(gg_ref)), 1.0)
    report(eg < TOL_PYTHON, "4d. dℓ/dg", @sprintf("max rel %.2e", eg))

    ec = maximum(abs.(out.C .- C_ref))
    report(ec < TOL_PYTHON, "4e. the correlation matrix C",
           @sprintf("max |ΔC| = %.2e", ec))
end

println()
if fails == 0
    println("all checks passed. The log-precision θ-step matches Python at the")
    println("level of the objective and every gradient block, so a disagreement")
    println("in run_emp4.jl would be in the estimator around it, not in here.")
else
    println("$fails check(s) failed. Send me the output above; do not trust the")
    println("log-precision rows until this passes.")
    exit(1)
end
