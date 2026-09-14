# The Section 5 simulation design and estimators, against the Python
# implementation.
#
#   julia test_dgp.jl
#
# Run it before mc4.jl. If the port is wrong this says which piece; a Monte Carlo
# would only say that a table is plausible, six hours later.
#
# The two implementations cannot draw the same random numbers, so the Monte Carlo
# distributions can only be compared to within simulation error. Everything that
# is not random can be compared exactly, and this file does:
#
#   1. the design constants: Ω, the basis restriction, β and α
#   2. the ARCH design at a supplied driver: UΛ⁻¹U', Ω(t)⁻¹ and log|Σ|
#   3. the regime design: Ω(t)⁻¹ and log|Σ|
#   4. the VECM recursion, from supplied innovations
#   5. the three estimators of Experiment B on the resulting sample, started
#      exactly as the experiment starts them
#   6. invariance of the fit to the sign convention of the eigensolver
#
# Check 5 is the one that matters. It is the whole chain, and it is exact: the
# innovations come from the file rather than from a generator, so a difference
# between the two implementations has nowhere to hide.

using LinearAlgebra, Statistics, Random, Distributions, CSV, DataFrames, Printf

include("grrr_core.jl")
include("emp4.jl")
include("dgp4.jl")

const TOL_EXACT = 1e-12     # design constants: the same numbers, not close ones
const TOL_NUM   = 1e-10     # linear algebra on the same inputs
const TOL_FIT   = 1e-4      # two optimisers stopping at the same maximum

fails = 0
report(ok, name, detail) = begin
    @printf("%-52s %s   %s\n", name, ok ? "ok  " : "FAIL", detail)
    ok || (global fails += 1)
end

println("=" ^ 88)
println("SECTION 5 SIMULATION DESIGN, JULIA AGAINST PYTHON")
println("=" ^ 88)

const REFFILE = joinpath(@__DIR__, "design_reference.txt")
if !isfile(REFFILE)
    println("\ndesign_reference.txt is missing; regenerate it with")
    println("    python mk_design_reference.py        (in the Python4 folder)")
    exit(1)
end

lines = readlines(REFFILE)
hdr = parse.(Int, split(strip(lines[1])))
T_ref, p_ref, q_ref, r_ref = hdr[1], hdr[2], hdr[3], hdr[4]
vals = parse.(Float64, lines[2:end])
pos = Ref(0)
nextvals(n) = (pos[] += n; vals[pos[] - n + 1:pos[]])
# every array was written row by row, so read into the transposed shape and
# permute rather than reshaping directly, Julia filling columns first
nextmat(nr, nc) = permutedims(reshape(nextvals(nr * nc), nc, nr))

(p_ref == P && r_ref == R) ||
    error("the reference file is for p = $p_ref, r = $r_ref, this build is p = $P, r = $R")
q_ref == Q_ARCH ||
    error("the reference file uses q = $q_ref, dgp4.jl uses q = $Q_ARCH")

Om_ref  = nextmat(P, P)
H_ref   = nextmat(P * R, R)
h_ref   = nextvals(P * R)
be_ref  = nextmat(P, R)
al_ref  = nextmat(P, R)
v_ref   = nextvals(T_ref)
UwUw_ref = nextmat(P, P)
Wa_ref  = [nextmat(P, P) for _ in 1:3]
lda_ref = nextvals(1)[1]
Wr_ref  = [nextmat(P, P) for _ in 1:3]
ldr_ref = nextvals(1)[1]
eps_ref = nextmat(T_ref, P)
Y_ref   = nextmat(T_ref, P)
X_ref   = nextmat(T_ref, P)
b_arch  = [nextvals(R) for _ in 1:3]
l_arch  = nextvals(3)
epsr_ref = nextmat(T_ref, P)
Yr_ref  = nextmat(T_ref, P)
Xr_ref  = nextmat(T_ref, P)
b_reg   = [nextvals(R) for _ in 1:3]
l_reg   = nextvals(3)
pos[] == length(vals) ||
    error("design_reference.txt has $(length(vals)) numbers, $(pos[]) were read")

const TS_REF = [1, T_ref ÷ 2 + 1, T_ref]     # the periods Python wrote out

# --- 1. the design constants ------------------------------------------------
e = maximum(abs.(OMEGA_DESIGN .- Om_ref))
report(e < TOL_EXACT, "1a. the innovation covariance Ω",
       @sprintf("max |ΔΩ| = %.2e", e))

e = max(maximum(abs.(H_free .- H_ref)), maximum(abs.(h_free .- h_ref)))
report(e < TOL_EXACT, "1b. the basis restriction H, h", @sprintf("max |Δ| = %.2e", e))

β = beta_of(ones(R))
α = make_alpha(β, 0.3)
eb = maximum(abs.(β .- be_ref))
ea = maximum(abs.(α .- al_ref))
report(max(eb, ea) < TOL_NUM, "1c. β and α = β(β'β)⁻¹diag(−κ)",
       @sprintf("max |Δβ| = %.2e, max |Δα| = %.2e", eb, ea))

# β'α = −diag(κ) is the property α is built for, so check it rather than only
# the matrix: a transposed inverse would still look like a plausible α.
e = maximum(abs.(β' * α .+ 0.3 * Matrix{Float64}(I, R, R)))
report(e < TOL_NUM, "1d. β'α = −diag(κ)", @sprintf("max |Δ| = %.2e", e))

# --- 2. the ARCH design at the supplied driver ------------------------------
covs_a, Wa, lda, ax = sigma_arch(T_ref, OMEGA_DESIGN, v_ref)

F = eigen(Symmetric(OMEGA_DESIGN))
ord = sortperm(F.values, rev=true)[1:Q_ARCH]
Uw = F.vectors[:, ord] ./ sqrt.(F.values[ord])'
e = maximum(abs.(Uw * Uw' .- UwUw_ref))
report(e < TOL_NUM, "2a. UΛ⁻¹U', the sign-free part of the design",
       @sprintf("max |Δ| = %.2e", e))

e = maximum(maximum(abs.(Wa[TS_REF[k]] .- Wa_ref[k])) for k in 1:3)
report(e < TOL_NUM, "2b. Ω(t)⁻¹ at t = 1, T/2, T",
       @sprintf("max |ΔW| = %.2e", e))

e = abs(lda - lda_ref) / max(abs(lda_ref), 1.0)
report(e < TOL_NUM, "2c. log|Σ| under the ARCH design",
       @sprintf("julia %.8f, python %.8f, rel %.2e", lda, lda_ref, e))

# What the design does and does not identify. For a driver vₜA with A of full
# column rank q, the perturbation D₁₂ ↦ D₁₂ + AK with K skew leaves every
# Qₜ'DQₜ unchanged, so the map from symmetric D to the precision path has a null
# space of dimension exactly q(q−1)/2. An earlier version of this test asserted
# that q < p made D identified; it does not, and this check replaces that claim
# with the number. The dimension is computed from the map itself, on the
# reference driver, and again from what `theta_arch` records during the fit in
# check 5, so the two must agree with each other and with the algebra.
let m = P + Q_ARCH, Tk = min(40, T_ref)
    Qs = [vcat(Matrix{Float64}(I, P, P), v_ref[t] * Uw') for t in 1:Tk]  # Qₜ, m×p
    cols = Vector{Float64}[]
    for j in 1:m, i in j:m
        Eij = zeros(m, m); Eij[i, j] = 1.0; Eij[j, i] = 1.0
        push!(cols, reduce(vcat, [vec(Qs[t]' * Eij * Qs[t]) for t in 1:Tk]))
    end
    sv = svdvals(reduce(hcat, cols))
    nullity = count(<(1e-9 * sv[1]), sv)
    want = Q_ARCH * (Q_ARCH - 1) ÷ 2
    report(nullity == want, "2d. flat subspace of D has dimension q(q−1)/2",
           @sprintf("null space %d of %d symmetric entries, predicted %d", nullity,
                    m * (m + 1) ÷ 2, want))
end

# --- 3. the regime design ---------------------------------------------------
covs_r, Wr, ldr, brk = sigma_regime(T_ref, OMEGA_DESIGN; factor=2.0)
e = maximum(maximum(abs.(Wr[TS_REF[k]] .- Wr_ref[k])) for k in 1:3)
report(e < TOL_NUM, "3a. Ω(t)⁻¹ under the regime design",
       @sprintf("max |ΔW| = %.2e, break at %d", e, brk))
e = abs(ldr - ldr_ref) / max(abs(ldr_ref), 1.0)
report(e < TOL_NUM, "3b. log|Σ| under the regime design",
       @sprintf("julia %.8f, python %.8f, rel %.2e", ldr, ldr_ref, e))

# --- 4. the VECM recursion --------------------------------------------------
Yj, Xj = simulate_vecm(T_ref, β, α, eps_ref)
ey = maximum(abs.(Yj .- Y_ref)); ex = maximum(abs.(Xj .- X_ref))
report(max(ey, ex) < TOL_NUM, "4a. Δxₜ = αβ'xₜ₋₁ + εₜ, ARCH innovations",
       @sprintf("max |ΔY| = %.2e, max |ΔX| = %.2e", ey, ex))

Yr, Xr = simulate_vecm(T_ref, β, α, epsr_ref)
ey = maximum(abs.(Yr .- Yr_ref)); ex = maximum(abs.(Xr .- Xr_ref))
report(max(ey, ex) < TOL_NUM, "4b. the same, regime innovations",
       @sprintf("max |ΔY| = %.2e, max |ΔX| = %.2e", ey, ex))

# --- 5. the three estimators of Experiment B --------------------------------
"""The three estimators, started the way `mcB_rep` starts its first fit: an
iid fit from b = 1 with the closed-form loadings under b = 1, then its answer as
the start for the correct and the infeasible one. The reference values are the
optima, which the other starts of `mcB_rep` reach as well at this sample size."""
function three(Yd, Xd, model, ctx, W, ld; tol=1e-9, maxiter=2000)
    f(mdl, cx, phi0, psi0; tl=1e-9, mx=2000) =
        grrr_estimate(Yd, Xd, nothing, R; H=H_free, h=h_free, sigma_model=mdl,
                      sigma_ctx=cx, phi0=phi0, psi0=psi0, tol=tl, maxiter=mx,
                      normalize_beta=false)
    S1 = Xd * beta_of(ones(R))
    r0 = f(:iid, nothing, ones(R), vec((Yd' * S1) / (S1' * S1)))
    phi0 = -r0[:beta][1, :]; psi0 = vec(r0[:alpha])
    rc = f(model, ctx, phi0, psi0; tl=tol, mx=maxiter)
    rf = f(:fixed, Dict{Symbol,Any}(:W => W, :logdetS => ld), phi0, psi0)
    return ([-r0[:beta][1, :], -rc[:beta][1, :], -rf[:beta][1, :]],
            [r0[:loglik], rc[:loglik], rf[:loglik]],
            get(rc[:theta], :nullity, -1))
end

ctx_a = Dict{Symbol,Any}(:arch_exog => ax, :arch_max_inner => 30)
bj_a, lj_a, nul_a = three(Yj, Xj, :arch, ctx_a, Wa, lda; tol=1e-8, maxiter=4000)
bj_r, lj_r, _ = three(Yr, Xr, :regime,
                      Dict{Symbol,Any}(:breaks => [T_ref ÷ 2]), Wr, ldr)

report(nul_a == Q_ARCH * (Q_ARCH - 1) ÷ 2,
       "2e. theta_arch records the same flat dimension",
       @sprintf("recorded %d, predicted %d", nul_a, Q_ARCH * (Q_ARCH - 1) ÷ 2))

let names = ("iid Σ", "correct Σ(θ)", "infeasible, Σ known")
    for k in 1:3
        d = maximum(abs.(bj_a[k] .- b_arch[k]))
        report(d < TOL_FIT, "5a$('a' + k - 1). ARCH design, b̂ from $(names[k])",
               @sprintf("max |Δb̂| = %.2e, Δℓ = %.2e", d, lj_a[k] - l_arch[k]))
    end
    for k in 1:3
        d = maximum(abs.(bj_r[k] .- b_reg[k]))
        report(d < TOL_FIT, "5b$('a' + k - 1). regime design, b̂ from $(names[k])",
               @sprintf("max |Δb̂| = %.2e, Δℓ = %.2e", d, lj_r[k] - l_reg[k]))
    end
end

# --- 5c. the Newton operator of the ARCH θ-step -----------------------------
# theta_arch builds Σₜ Mₜ ⊗ Mₜ as one matrix product and a permutation rather
# than as T Kronecker products. The two are the same matrix; this checks that
# on random symmetric Mₜ, because a wrong permutation would still produce a
# plausible-looking symmetric operator and a fit that merely converges slowly.
let m = 7, Tk = 40
    rk = MersenneTwister(20260909)
    Mt = [(A = randn(rk, m, m); (A + A') / 2) for _ in 1:Tk]
    ref = sum(kron(Mt[t], Mt[t]) for t in 1:Tk)
    Gs = zeros(m * m, Tk)
    for t in 1:Tk
        Gs[:, t] = vec(Mt[t])
    end
    fast = reshape(permutedims(reshape(Gs * Gs', m, m, m, m), (1, 3, 2, 4)),
                   m * m, m * m)
    d = maximum(abs.(fast .- ref)) / maximum(abs.(ref))
    report(d < 1e-12, "5c. Newton operator Σ Mₜ⊗Mₜ, gemm form vs Kronecker",
           @sprintf("max relative difference = %.2e", d))
end

# --- 6. invariance to the eigensolver's sign convention ---------------------
# The model is invariant under Xex ↦ Xex S for orthogonal S, so flipping the
# sign of a column of the driver must move D and leave ℓ and b̂ alone. This is
# why check 2a compares UΛ⁻¹U' and not the factor itself: LAPACK builds disagree
# about the signs and the experiment does not care.
S = Diagonal([k == 1 ? -1.0 : 1.0 for k in 1:Q_ARCH])
ctx_flip = Dict{Symbol,Any}(:arch_exog => [ax[t] * S for t in 1:T_ref],
                            :arch_max_inner => 30)
bf, lf, _ = three(Yj, Xj, :arch, ctx_flip, Wa, lda; tol=1e-8, maxiter=4000)
e = maximum(abs.(bf[2] .- bj_a[2]))
report(e < TOL_FIT, "6. b̂ invariant to a sign flip in the driver",
       @sprintf("max |Δb̂| = %.2e, Δℓ = %.2e", e, lf[2] - lj_a[2]))

println()
if fails == 0
    println("all checks passed. The design constants, both covariance paths, the")
    println("recursion and all three estimators agree with Python on data neither")
    println("implementation drew, so a difference in the Monte Carlo tables is")
    println("simulation error and not a port error.")
else
    println("$fails check(s) failed. Send me the output above; do not trust the")
    println("Section 5 tables until this passes.")
    exit(1)
end
