using Dates

# The four-contract empirical system, in Julia.
#
#   Y = (numeraire, log F2, log F3, log F4),  monthly, 1986-03 to 2024-04, T = 458
#   r = 3, k = 2 lags, unrestricted constant, four Fourier seasonal terms
#   relation i is  log F_{i+1} − b_i × numeraire,  and the null is b₁ = b₂ = b₃ = 1
#
# The numeraire is the front contract F1 or the FRED Cushing spot price. The two
# are the same series in all but name: their monthly log returns correlate at
# 0.9986 and the median gap is 0.13 per cent of the price. A system with both
# carries a near-identity among its cointegrating relations, so exactly one of
# them is used. Which one is set by NUMERAIRE=F1 (the default) or NUMERAIRE=spot
# in the environment, or by the `numeraire` keyword of `load_emp4`.
#
# The spot puts Section 6 in the terms Section 2.2 uses, the cost-of-carry basis
# F_it − S_t. The front contract is an exchange-traded settlement where the spot
# is a posted assessment, and the spot-based bases are a little noisier and more
# persistent. Both can be run; the paper reports one and the other is the check.

const P = 4                       # variables
const R = 3                       # cointegration rank, p − 1, one common trend
const BRK = [270, 408]            # 2008-09 and 2020-03, as Y-row indices

const NUMERAIRE = Symbol(get(ENV, "NUMERAIRE", "F1"))
NUMERAIRE in (:F1, :spot) ||
    error("NUMERAIRE must be F1 or spot, got $(NUMERAIRE)")

numeraire_columns(numeraire::Symbol) =
    numeraire === :spot ? [:l_spot, :l_F2, :l_F3, :l_F4] :
    numeraire === :F1   ? [:l_F1, :l_F2, :l_F3, :l_F4] :
    error("unknown numeraire $numeraire; use :F1 or :spot")

"""Fourier seasonal terms at the annual and semi-annual frequencies.

Ψ is unrestricted in the GRRR, so seasonal terms are simply extra columns of Z
and change nothing in the estimator."""
function seasonal_block(months::Vector{Int}, h::Int)
    n = length(months)
    S = zeros(n, 2h)
    for j in 1:h
        S[:, 2j - 1] = cos.(2π * j .* months ./ 12)
        S[:, 2j]     = sin.(2π * j .* months ./ 12)
    end
    return S
end

"""Load the four log prices and build (Y, X, Z) with the seasonal block.

`numeraire` is :F1 or :spot and defaults to the NUMERAIRE environment variable.
Everything downstream, the break dates, the seasonal block, the basis
restriction, is the same either way: only the first column of Y changes."""
function load_emp4(path::String; h::Int=2, numeraire::Symbol=NUMERAIRE)
    df = CSV.read(path, DataFrame)
    L = Matrix(df[:, numeraire_columns(numeraire)])
    dts = df.date
    N = size(L, 1)
    dL = diff(L, dims=1)
    Y = dL[2:end, :]                                   # Δxₜ,   t = 2..N−1
    X = L[2:N - 1, :]                                  # xₜ₋₁
    months = [month(d isa Date ? d : Date(string(d))) for d in dts[3:N]]
    S = seasonal_block(months, h)
    Z = hcat(dL[1:N - 2, :], ones(N - 2), S)           # [Δxₜ₋₁, const, seasonals]
    return Y, X, Z, dts[3:N]
end

# ---------------------------------------------------------------------------
# the spread basis, with the front contract in the numeraire role
#   column i of β has −bᵢ on F1 and +1 on F_{i+1}
# ---------------------------------------------------------------------------

function basis_restriction()
    H = zeros(P * R, R); h = zeros(P * R)
    for i in 1:R
        H[P * (i - 1) + 1, i] = -1.0        # F1 row of column i
        h[P * (i - 1) + i + 1] = 1.0        # F_{i+1} row of column i
    end
    return H, h
end

const H_free, h_free = basis_restriction()
const H_one = zeros(P * R, 0)
const h_one = H_free * ones(R) + h_free     # β with every bᵢ = 1

"""The volatility driver: lagged mean absolute log return across the strip.

`kind` is :level for the driver itself, normalised to mean one, or :log for its
logarithm. The distinction is immaterial for the fit and decisive for the
recursive experiment of Section 6.2, where a driver in levels makes the implied
volatility process explosive."""
function driver(Zb; kind::Symbol=:level)
    v = vec(mean(abs.(Zb[:, 1:P]), dims=2))
    v = v ./ mean(v)
    return kind === :log ? log.(v) .- mean(log.(v)) : v
end

"""sigma_ctx for one specification. `spec` is one of
:iid, :regime, :arch, :archchol, :scale, :scalelog, :gft, :gftlog.

The `log` variants differ only in the driver: the normalised mean absolute
return in levels, or its logarithm, centred. That choice decides whether the
recursion is stable, which is the point Section 6 makes about the two."""
function sigma_ctx_for(spec::Symbol, Zb)
    spec === :iid && return nothing
    spec === :regime && return Dict{Symbol,Any}(:breaks => copy(BRK))
    if spec === :scale || spec === :scalelog
        u = driver(Zb; kind=(spec === :scalelog ? :log : :level))
        return Dict{Symbol,Any}(:scale_driver => u)
    end
    if spec in (:arch, :archchol, :gft, :gftlog)
        vb = driver(Zb; kind=(spec === :gftlog ? :log : :level))
        return Dict{Symbol,Any}(:arch_exog => [fill(vb[t], P, 1) for t in 1:size(Zb, 1)],
                                :arch_max_inner => 40,
                                :arch_chol_maxiter => 2000,
                                :gft_maxiter => 800)
    end
    error("unknown specification $spec")
end

sigma_model_for(spec::Symbol) =
    spec === :scalelog ? :scale :
    spec === :gft      ? :archgft :
    spec === :gftlog   ? :archgft : spec

# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------

"""β from the eigenvalue RRR, then (α,Ψ) from the pinned-β fit started at OLS.

Every fit in this file starts here rather than from a random point. Table 8 of
the paper shows why: from a dispersed start not one of thirty runs reaches the
maximum on this sample, while this start reaches it in two iterations."""
function closed_form_start(Yb, Xb, Zb, spec; tol=1e-10, maxiter=20000)
    jrb = johansen_rrr(Yb, Xb, Zb, R)
    # The null direction of β'. The basis parametrisation implies it is
    # proportional to (1, b₁, b₂, b₃), so b falls out of it directly.
    # `full=true` is needed: the thin SVD of a 3×4 matrix omits it.
    F = svd(Matrix(jrb[:beta]'); full=true)
    w = F.Vt[end, :]
    b0 = w[2:end] ./ w[1]
    h0 = H_free * b0 + h_free
    SF = hcat(Xb * reshape(h0, P, R), Zb)
    ψ_ols = vec(((SF' * SF) \ (SF' * Yb))')
    ctx = sigma_ctx_for(spec, Zb)
    pin = grrr_estimate(Yb, Xb, Zb, R; H=zeros(P * R, 0), h=h0,
                        sigma_model=sigma_model_for(spec),
                        sigma_ctx=(ctx === nothing ? nothing : copy(ctx)),
                        psi0=ψ_ols, tol=tol, maxiter=maxiter, normalize_beta=false)
    return b0, vec(hcat(pin[:alpha], pin[:Psi])), jrb
end

fit_basis(Yb, Xb, Zb, spec, free; phi0=nothing, psi0=nothing,
          tol=1e-10, maxiter=8000) =
    grrr_estimate(Yb, Xb, Zb, R;
                  H=(free ? H_free : H_one), h=(free ? h_free : h_one),
                  sigma_model=sigma_model_for(spec),
                  sigma_ctx=sigma_ctx_for(spec, Zb),
                  phi0=phi0, psi0=psi0, tol=tol, maxiter=maxiter,
                  normalize_beta=false)

psi_of(res) = vec(hcat(res[:alpha], res[:Psi]))
b_of(res) = -res[:beta][1, :]

"""Free and restricted fits under one specification, from several starts.

Each candidate is printed rather than silently maximised over. On the
four-contract system every start reaches the same optimum, which is itself the
result worth seeing; on the five-variable system that included the spot series
the restricted ARCH-type fit had two local maxima, at 8196.4452 and 8194.9555,
and which one a single start found depended on the linear-algebra library."""
function sample_fits(Y, X, Z, spec; verbose=true)
    b0, ψ_pin, _ = closed_form_start(Y, X, Z, spec)
    f_i = fit_basis(Y, X, Z, :iid, true;  phi0=b0, psi0=ψ_pin)
    o_i = fit_basis(Y, X, Z, :iid, false; psi0=ψ_pin)

    cands_f = [("closed form", fit_basis(Y, X, Z, spec, true; phi0=b0, psi0=ψ_pin))]
    spec === :iid || push!(cands_f,
        ("warm at iid fit", fit_basis(Y, X, Z, spec, true;
                                      phi0=b_of(f_i), psi0=psi_of(f_i))))
    for (nm, r) in cands_f
        verbose && @printf("    %-8s %-20s %14.4f\n", "b free", nm, r[:loglik])
    end
    free = argmax_loglik([r for (_, r) in cands_f])

    cands_o = [("closed form", fit_basis(Y, X, Z, spec, false; psi0=ψ_pin)),
               ("warm at free fit", fit_basis(Y, X, Z, spec, false; psi0=psi_of(free)))]
    spec === :iid || push!(cands_o,
        ("warm at iid fit", fit_basis(Y, X, Z, spec, false; psi0=psi_of(o_i))))
    for (nm, r) in cands_o
        verbose && @printf("    %-8s %-20s %14.4f\n", "b = 1", nm, r[:loglik])
    end
    one = argmax_loglik([r for (_, r) in cands_o])

    @assert(one[:loglik] <= free[:loglik] + 1e-8,
            "restricted fit above unrestricted under $spec; starting values failed")
    return free, one
end

argmax_loglik(v) = v[argmax([r[:loglik] for r in v])]
