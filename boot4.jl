# Recursive wild bootstrap and the recursive null experiment, four contracts.
#
# The only structural change from the five-variable code is that Ψ now carries a
# seasonal block as well as the constant, so the deterministic part of the
# recursion depends on the calendar month:
#
#   Δx*ₜ = Π x*ₜ₋₁ + Γ₁ Δx*ₜ₋₁ + Ψ_d z_{d,t} + ε*ₜ
#
# The seasonal terms are fixed by the date and every bootstrap sample inherits
# the dates of the observed one, so z_{d,t} is carried over unchanged and only
# the lagged-difference columns of Z* are regenerated. Writing the deterministic
# part as a precomputed T×p path keeps the recursion the same shape as before.

"""(Π, Γ₁, dpath) from a fitted basis model, with dpath the T×p deterministic path."""
function vecm_parts(res, Z)
    Π = res[:alpha] * res[:beta]'
    Γ₁ = res[:Psi][:, 1:P]
    dpath = Z[:, P + 1:end] * res[:Psi][:, P + 1:end]'
    return Π, Γ₁, dpath
end

resid(Y, X, Z, res) = Y - X * res[:beta] * res[:alpha]' - Z * res[:Psi]'

"""Roots of the companion matrix in levels, xₜ = (I+Π+Γ₁)xₜ₋₁ − Γ₁xₜ₋₂."""
function companion_roots(Π, Γ₁)
    C = [I + Π + Γ₁  -Γ₁; Matrix{Float64}(I, P, P)  zeros(P, P)]
    return sort(abs.(eigvals(C)), rev=true)
end

function draw_weights(rng, T, scheme::Symbol)
    scheme === :rademacher && return 2.0 .* rand(rng, Bool, T) .- 1.0
    scheme === :normal && return randn(rng, T)
    error("unknown weight scheme $scheme")
end

"""One recursive wild bootstrap sample. `Zdet` is the constant-and-seasonal
block of the observed Z, reused unchanged."""
function generate(Π, Γ₁, dpath, E, Zdet, x_init, dx_init, rng; scheme::Symbol=:rademacher)
    T = size(E, 1)
    Es = E .* draw_weights(rng, T, scheme)
    Yb = zeros(T, P); Xb = zeros(T, P)
    Zb = hcat(zeros(T, P), Zdet)
    xprev = copy(x_init); dprev = copy(dx_init)
    for t in 1:T
        Xb[t, :] = xprev
        Zb[t, 1:P] = dprev
        dx = Π * xprev + Γ₁ * dprev + dpath[t, :] + Es[t, :]
        Yb[t, :] = dx
        dprev = dx
        xprev = xprev + dx
    end
    return Yb, Xb, Zb
end

"""One bootstrap replication: LR* under the null DGP, or b̂* under the free one.

Three starting values for the free fit, the third being the restricted optimum
itself. Because the two mean blocks are exact conditional maximisations and the
Σ-step then starts at the restricted θ, the free fit cannot end below the
restricted one, so LR* ≥ 0 holds by construction rather than by luck."""
function boot_rep(spec, src, Zdet, x_init, dx_init, seed;
                  kind::Symbol=:lr, scheme::Symbol=:rademacher,
                  warm=nothing, tol=1e-9, maxiter=3000)
    rng = Xoshiro(seed)
    Yb, Xb, Zb = generate(src.Π, src.Γ₁, src.dpath, src.E, Zdet, x_init, dx_init,
                          rng; scheme=scheme)
    b0b, ψ_pin, _ = closed_form_start(Yb, Xb, Zb, spec; tol=1e-9, maxiter=8000)

    if kind === :se
        f = argmax_loglik([
            fit_basis(Yb, Xb, Zb, spec, true; phi0=warm.b, psi0=warm.ψ_free,
                      tol=tol, maxiter=maxiter),
            fit_basis(Yb, Xb, Zb, spec, true; phi0=b0b, psi0=ψ_pin,
                      tol=tol, maxiter=maxiter)])
        return (LR=NaN, b=b_of(f))
    end

    o = argmax_loglik([
        fit_basis(Yb, Xb, Zb, spec, false; psi0=warm.ψ_one, tol=tol, maxiter=maxiter),
        fit_basis(Yb, Xb, Zb, spec, false; psi0=ψ_pin, tol=tol, maxiter=maxiter)])
    f = argmax_loglik([
        fit_basis(Yb, Xb, Zb, spec, true; phi0=warm.b, psi0=warm.ψ_free,
                  tol=tol, maxiter=maxiter),
        fit_basis(Yb, Xb, Zb, spec, true; phi0=b0b, psi0=ψ_pin,
                  tol=tol, maxiter=maxiter),
        fit_basis(Yb, Xb, Zb, spec, true; phi0=ones(R), psi0=psi_of(o),
                  tol=tol, maxiter=maxiter)])
    return (LR=2 * (f[:loglik] - o[:loglik]), b=b_of(f))
end

# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
# Every replication is written to disk as it completes. A bootstrap at B = 999
# is hours of work and the reasons it stops are mundane: a dropped connection, a
# restart, a machine that went to sleep. Nothing here needs a package: one line
# per replication, "j LR b1 b2 b3", read back with `split` and `parse`.
#
# Replication j depends only on seed0 + j, so a resumed run reproduces exactly
# what an uninterrupted one would have produced. Order of completion does not
# matter and the file may be written by any thread, so the append is locked.

# Results are keyed by the numeraire. A run on the spot system must never
# resume from replications computed on the front-contract system, and the two
# would otherwise share every file name.
boot_resultdir() = (d = joinpath(@__DIR__, NUMERAIRE === :F1 ? "results4" :
                                            "results4_$(NUMERAIRE)");
                    mkpath(d); d)

boot_path(spec, kind, scheme, B) =
    joinpath(boot_resultdir(),
             string(spec, "_", kind,
                    scheme === :rademacher ? "" : string("_", scheme),
                    "_B", B, ".txt"))

"""Replications already on disk, as j => (LR, b). Silently returns an empty
dictionary if the file is absent or unreadable, so a corrupt partial file costs
recomputation rather than a crash."""
function boot_load(path, R)
    out = Dict{Int,Tuple{Float64,Vector{Float64}}}()
    isfile(path) || return out
    for ln in readlines(path)
        f = split(strip(ln))
        length(f) == 2 + R || continue
        try
            j = parse(Int, f[1])
            out[j] = (parse(Float64, f[2]),
                      [parse(Float64, f[2 + i]) for i in 1:R])
        catch
            continue                       # skip a line torn by an interrupt
        end
    end
    return out
end

"""Replications already on disk for an experiment whose row is a plain vector of
`nfield` numbers after the index. The generic form of `boot_load`, used by the
Section 5 Monte Carlo and by the recursive null experiment."""
function mc_load(path, nfield)
    out = Dict{Int,Vector{Float64}}()
    isfile(path) || return out
    for ln in readlines(path)
        f = split(strip(ln))
        length(f) == 1 + nfield || continue
        try
            out[parse(Int, f[1])] = [parse(Float64, f[1 + i]) for i in 1:nfield]
        catch
            continue                       # skip a line torn by an interrupt
        end
    end
    return out
end

"""Run `f(j)` for every j not already on disk, threaded, appending each result
as it completes. `f` returns the row of numbers to store. Replication j depends
only on its own seed, so a resumed run reproduces exactly what an uninterrupted
one would have produced and the order threads finish in does not matter."""
function mc_run(path, js, nfield, f; verbose=true, save=true, label="")
    have = save ? mc_load(path, nfield) : Dict{Int,Vector{Float64}}()
    todo = [j for j in js if !haskey(have, j)]
    if verbose && !isempty(have)
        @printf("    %s resuming: %d of %d already on disk\n",
                label, length(have), length(js))
        flush(stdout)
    end
    isempty(todo) && return have

    fh = save ? open(path, "a") : nothing
    lk = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    try
        Threads.@threads for j in todo
            row = f(j)
            n = Threads.atomic_add!(done, 1) + 1
            lock(lk) do
                have[j] = row
                if save
                    print(fh, j)
                    for v in row
                        print(fh, " ", v)
                    end
                    println(fh)
                    n % 25 == 0 && flush(fh)
                end
            end
            if verbose && n % 20 == 0
                @printf("    %s %4d/%d\n", label, n, length(todo)); flush(stdout)
            end
        end
    finally
        if fh !== nothing
            flush(fh)
            close(fh)
        end
    end
    return have
end

"""Run B replications, threaded, resuming from and appending to disk.
Returns the LR* vector and the b̂* matrix."""
function run_bootstrap(spec, Y, X, Z, free, one, x_init, dx_init;
                       B=999, kind::Symbol=:lr, scheme::Symbol=:rademacher,
                       seed0=19510104, verbose=true, save=true)
    Πo, Γo, deto = vecm_parts(one, Z)
    Πf, Γf, detf = vecm_parts(free, Z)
    src = kind === :lr ?
        (Π=Πo, Γ₁=Γo, dpath=deto, E=resid(Y, X, Z, one)) :
        (Π=Πf, Γ₁=Γf, dpath=detf, E=resid(Y, X, Z, free))
    Zdet = Z[:, P + 1:end]
    warm = (b=b_of(free), ψ_free=psi_of(free), ψ_one=psi_of(one))

    LR = fill(NaN, B); bs = fill(NaN, B, R)
    path = boot_path(spec, kind, scheme, B)
    have = save ? boot_load(path, R) : Dict{Int,Tuple{Float64,Vector{Float64}}}()
    for (j, v) in have
        1 <= j <= B || continue
        LR[j] = v[1]; bs[j, :] = v[2]
    end
    todo = [j for j in 1:B if !haskey(have, j)]
    if verbose && !isempty(have)
        @printf("    resuming: %d of %d already on disk\n", length(have), B)
        flush(stdout)
    end
    isempty(todo) && return LR, bs

    fh = save ? open(path, "a") : nothing
    lk = ReentrantLock()
    done = Threads.Atomic{Int}(0)
    try
        Threads.@threads for j in todo
            out = boot_rep(spec, src, Zdet, x_init, dx_init, seed0 + j;
                           kind=kind, scheme=scheme, warm=warm)
            LR[j] = out.LR; bs[j, :] = out.b
            n = Threads.atomic_add!(done, 1) + 1
            if save
                lock(lk) do
                    print(fh, j, " ", out.LR)
                    for v in out.b
                        print(fh, " ", v)
                    end
                    println(fh)
                    if n % 25 == 0
                        flush(fh)
                    end
                end
            end
            if verbose && n % 50 == 0
                @printf("    %4d/%d\n", n, length(todo)); flush(stdout)
            end
        end
    finally
        if fh !== nothing
            flush(fh)
            close(fh)
        end
    end
    return LR, bs
end

boot_pvalue(LR, LR0) = (1 + count(>=(LR0), LR)) / (length(LR) + 1)

# ---------------------------------------------------------------------------
# the recursive null experiment
# ---------------------------------------------------------------------------

"""Ω(t)⁻¹ implied by a fitted θ at proxy value u.

Three forms. The scale model divides a fixed precision by the scale factor. The
log-precision model exponentiates a linear function of the proxy and wraps it
round the fixed correlation matrix of the generalized Fisher transformation. The
quadratic form is Q(u)'DQ(u). The two models with no proxy, iid and regime,
have a precision that does not depend on the simulated path at all, so they go
through `fixed_precision` instead and never reach here.

The θ-step centres the scale proxy by its own sample mean, so `ubar` is applied
here for that model and not for the others, which read the proxy as given."""
function precision_at(spec, θ, u, ubar)
    if spec === :scale || spec === :scalelog
        return inv(θ[:Omega]) ./ exp(-θ[:b] * (u - ubar))
    end
    if spec === :gft || spec === :gftlog
        b = θ[:b]
        size(b, 2) == 1 ||
            error("precision_at expects a scalar proxy, q = 1, got q = $(size(b, 2))")
        λ = exp.(0.5 .* (θ[:a] .+ vec(b) .* u))
        return λ .* θ[:C] .* λ'
    end
    Q = vcat(Matrix{Float64}(I, P, P), fill(u, 1, P))
    return Q' * θ[:D] * Q
end

"""Ω(t)⁻¹ for the two specifications with no proxy.

Under iid Σ the precision is constant; under the regime specification it is
piecewise constant. Neither depends on the simulated sample's own past, so the
recursive experiment is not recursive for these two and no path can run away.
Returns `nothing` for every other specification, which is the signal to build
the precision from the path instead."""
function fixed_precision(spec, θ, T, breaks)
    spec === :iid && return [inv(_sym(θ[:Omega])) for _ in 1:T]
    if spec === :regime
        out = Vector{Matrix{Float64}}(undef, T)
        edges = vcat(0, breaks, T)
        for (j, Ω) in enumerate(θ[:Omega_regimes])
            Wi = inv(_sym(Ω))
            for t in edges[j] + 1:edges[j + 1]
                out[t] = Wi
            end
        end
        return out
    end
    return nothing
end

"""One draw from the fitted model, with the volatility proxy rebuilt from the
draw's own past. Returns `nothing` if the path leaves the finite range, which is
what the log-precision and scale specifications do when the proxy enters in
levels.

`vcap` truncates the proxy at a ceiling, in the same mean-one units as `v`.
With `vcap` finite the conditional variance is bounded along every path and no
path can run away. The ceiling is a property of the DGP only: the fit to the
observed sample and the estimation of each simulated sample are untouched, and
if the ceiling is the observed maximum of the proxy it is never active inside
the sample. `Inf` reproduces the untruncated recursion."""
function simulate_null(spec, θ, Π, Γ₁, dpath, Zdet, mhat, ubar, x_init, dx_init, T, rng;
                       kind::Symbol=:level, shift=0.0, Wfixed=nothing, big=1e6,
                       vcap=Inf)
    Yb = zeros(T, P); Xb = zeros(T, P)
    Zb = hcat(zeros(T, P), Zdet)
    xprev = copy(x_init); dprev = copy(dx_init)
    for t in 1:T
        Xb[t, :] = xprev
        Zb[t, 1:P] = dprev
        W = if Wfixed !== nothing
            Wfixed[t]
        else
            v = min(mean(abs.(dprev)) / mhat, vcap)
            v <= 0 && return nothing
            precision_at(spec, θ, kind === :log ? log(v) - shift : v, ubar)
        end
        F = eigen(Symmetric(_sym(W)))
        (F.values[1] <= 0 || !all(isfinite, F.values)) && return nothing
        ε = F.vectors * ((F.vectors' * randn(rng, P)) ./ sqrt.(F.values))
        dx = Π * xprev + Γ₁ * dprev + dpath[t, :] + ε
        (!all(isfinite, dx) || maximum(abs.(dx)) > big) && return nothing
        Yb[t, :] = dx
        dprev = dx
        xprev = xprev + dx
    end
    return Yb, Xb, Zb
end
