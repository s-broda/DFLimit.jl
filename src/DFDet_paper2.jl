# The statistics of the paper on general deterministic terms that were scripts (paper/envelope.jl and paper/lmtest.jl of
# the repository df-deterministics), as functions of the general engine: the point-optimal statistic and the Gaussian
# power envelope for any design, the LM statistic, and the companions of the GLS-detrended t-ratio (MZ_alpha, the
# coefficient statistic with GLS detrending, and MSB, the square root of the denominator).  Evaluated in the module DFDet
# (DFLimit.DFDet), whose file stays that of df-deterministics.

# ---------------------------------------------------------------------------------------------------------------------
# The point-optimal statistic and the power envelope
# ---------------------------------------------------------------------------------------------------------------------
"""
    prep_pt(D, cbar) -> Prep

The prepared design of the limit of the point-optimal statistic against cbar, {S(abar) - abar S(1)}/s^2 (S(a) the sum of
squared residuals of the regression of the data quasi-differenced at a on the trend functions transformed in the same
way): with delta the trend functions that vanish at the origin and do not jump, psi = delta' - cbar delta, Xi = int psi
psi', eta = int psi (dJ - cbar J dr), and psi_0, Xi_0, eta_0 the same at cbar = 0, the limit is Q = cbar^2 int J^2 - cbar
J(1)^2 - eta' Xi^{-1} eta + eta_0' Xi_0^{-1} eta_0 = cbar^2 V_g + ell' A ell, V_g the denominator of the GLS-detrended
statistic and A = C - cbar^2 B_g (the construction of `prep_gls` with another numerator; for a linear trend A is
(1 - cbar) J_g(1)^2).  The test rejects for small Q.
"""
function prep_pt(D::Determ, cbar::Float64)
    (cbar < 0 && isfinite(cbar)) || throw(ArgumentError("the point-optimal statistic needs cbar < 0 (a finite number)"))
    K = nreg(D); np = npc(D); deg = size(D.C[1], 1) - 1; br = copy(D.br)
    rows = [permutedims(dval(D, 0.0))]
    for j in 1:np-1
        lam = br[j+1]
        dl = [evalpoly(lam - poff(D, j + 1), view(D.C[j+1], :, i)) - evalpoly(lam - poff(D, j), view(D.C[j], :, i)) for i in 1:K]
        maximum(abs, dl; init = 0.0) > 1e-12 && push!(rows, permutedims(dl))
    end
    Mr = reduce(vcat, rows)
    Z = nullspace(Mr; atol = 1e-10 * max(1.0, maximum(abs, Mr))); k = size(Z, 2)
    offs = [poff(D, j) for j in 1:np]
    if k == 0                                                   # only the path at one: Q = cbar^2 int J^2 - cbar J(1)^2
        A = fill(-cbar, 1, 1)
        return Prep(D, 0, 0, 0, 1, deg, br, [zeros(deg + 1, 0) for _ in 1:np], Float64[], A, zeros(1, 1), ones(1, 1), [-cbar], 1, 1, 0,
                    zeros(0, 0), zeros(1), offs, cbar)
    end
    Cd = [D.C[j] * Z for j in 1:np]; Cd1 = [dcoef(Cd[j]) for j in 1:np]; Cd2 = [dcoef(Cd1[j]) for j in 1:np]
    ev(Cs, j, r) = [evalpoly(r - poff(D, j), view(Cs[j], :, i)) for i in 1:k]
    Xi = gram([Cd1[j] .- cbar .* Cd[j] for j in 1:np], br, D.loc); Xi0 = gram(Cd1, br, D.loc); Gd = gram(Cd, br, D.loc)
    psi1 = ev(Cd1, np, 1.0) .- cbar .* ev(Cd, np, 1.0); dot1 = ev(Cd1, np, 1.0)
    jl = Float64[]; vs = Vector{Float64}[]
    for j in 1:np-1
        lam = br[j+1]; s1 = ev(Cd1, j + 1, lam) .- ev(Cd1, j, lam)
        maximum(abs, s1; init = 0.0) > 1e-12 && (push!(jl, lam); push!(vs, s1))
    end
    nj = length(jl)
    F = [hcat(Cd[j], Cd2[j]) for j in 1:np]; E = eigen(Symmetric(gram(F, br, D.loc)))
    keep = E.values .> 1e-11 * maximum(E.values); U = E.vectors[:, keep]; lv = E.values[keep]; p = length(lv)
    g = [F[j] * U * Diagonal(1 ./ sqrt.(lv)) for j in 1:np]
    Phi = U * Diagonal(sqrt.(lv)); Phid = Phi[1:k, :]; Phi2 = Phi[k+1:2k, :]
    N = 1 + nj + p; iz = (2 + nj):N
    H = zeros(k, N); H0 = zeros(k, N); H[:, 1] .= psi1; H0[:, 1] .= dot1
    for i in 1:nj; H[:, 1+i] .= -vs[i]; H0[:, 1+i] .= -vs[i]; end
    H[:, iz] .= -(Phi2 .- cbar^2 .* Phid); H0[:, iz] .= -Phi2
    C = -(H' * (Xi \ H)) .+ H0' * (Xi0 \ H0); C[1, 1] -= cbar
    Bg = prep_gls(D, cbar).B; C = C .- cbar^2 .* Bg; C = (C .+ C') ./ 2
    Ae = eigen(Symmetric(C)); keepA = abs.(Ae.values) .> 1e-10 * maximum(abs, Ae.values)
    Dv = Ae.values[keepA]; L = Ae.vectors[:, keepA]
    Prep(D, k, p, nj, N, size(F[1], 1) - 1, br, g, jl, C, Bg, L, Dv, length(Dv), count(>(0), Dv), count(<(0), Dv), Gd, zeros(N), offs, cbar)
end
"""the characteristic function of Q under c on the grid of the Gil-Pelaez rule (u = sinh(v), N steps to umax): Q = U + 1/2 +
cbar^2 V, so that E exp(iuQ) = e^{iu/2} M(iu, iu cbar^2), the square root of the determinant continued along u from the origin"""
struct QGrid; u::Vector{Float64}; w::Vector{Float64}; lM::Vector{ComplexF64}; end
"""
the distance of the nearest singularity of phi(u) = E e^{iuQ} from the real line: Q is a quadratic form, whose
transform M(s, cbar^2 s) is singular on the real line only, at the first zeros s_+ > 0 and s_- < 0 of its determinant
(scanned by factors of 1.1 from 1e-10, then bisected), the smaller of s_+ and |s_-|
"""
function qedge(Mo::Model, cbar::Float64, c::Float64)
    W = MWork(Mo); Bf = fixbuf(Mo.r)
    bad(s) = (r = logmgf_point(Mo, c, complex(s), complex(cbar^2 * s), W, Bf); !(isfinite(real(r[2])) && abs(imag(r[2])) < 1.0))
    e = Inf
    for sg in (1.0, -1.0)
        s = 1e-10
        while s < 1e4 && !bad(sg * 1.1 * s); s *= 1.1; end
        s < 1e4 || continue
        lo = s; hi = 1.1 * s
        for _ in 1:40; m = sqrt(lo * hi); bad(sg * m) ? (hi = m) : (lo = m); end
        e = min(e, lo)
    end
    e
end
"""
the smallest ratio of the distance d of the singularity of phi (`qedge`) to the step h = asinh(umax)/N of the grid of
`qgrid` near the origin.  The error of the rule of Gil-Pelaez on the grid is about exp(-2 pi d/h): with a linear trend at
cbar = -13.5, d/h = 14 at c = 3 (off by 5e-10), 1.6 at c = 4 (9e-6), 0.2 at c = 5 (F from -0.1 to 3.4); with an
intercept at cbar = -7, 3.3 at c = 3 (3e-10), 0.7 at c = 4 (3e-3); Perron's model B at cbar = -17.8, 7.5 at c = 4
(4e-11), 0.75 at c = 5 (2e-3).  Below the unit root d/h is 3.7 to 33 for the critical values of the envelope (c = 0,
cbar from -10 to -30) and above 3000 for its power.  Above the unit root the law of Q widens like e^{2c} and d shrinks;
below PT_EDGE the functions refuse (a larger N, which shrinks h, extends them).
"""
const PT_EDGE = 3.3
function qgrid(P::Prep, cbar::Float64, c::Float64; N::Int = 60000, umax::Float64 = 4000.0)
    # (c = +-Inf did not return, and umax = Inf gives an infinite step)
    isfinite(c) || throw(ArgumentError("the point-optimal statistic: c must be a finite real scalar"))
    (umax > 0 && isfinite(umax)) || throw(ArgumentError("the point-optimal statistic: umax must be a positive finite number"))
    Mo = model(P); u, w = gpgrid(N, umax)
    d = blas1f(() -> qedge(Mo, cbar, c), dense_la(Mo)); h = asinh(umax) / N
    d / h >= PT_EDGE || throw(ArgumentError("the point-optimal statistic at c = $c, cbar = $cbar: the transform of Q has a singularity at $(round(d, sigdigits = 3)) from the real line, which the grid of the inversion (step $(round(h, sigdigits = 3)) near the origin) does not resolve (far above the unit root); it needs N >= $(ceil(Int, PT_EDGE * asinh(umax) / d))"))
    L0 = Vector{ComplexF64}(undef, N + 1); LD = Vector{ComplexF64}(undef, N + 1)
    Threads.@threads :static for ch in collect(Iterators.partition(1:N+1, cld(N + 1, Threads.nthreads())))
        W = MWork(Mo); Bf = fixbuf(Mo.r)
        for k in ch; L0[k], LD[k] = logmgf_point(Mo, c, complex(0.0, u[k]), complex(0.0, cbar^2 * u[k]), W, Bf); end
    end
    lM = Vector{ComplexF64}(undef, N + 1); ph = imag(LD[1])
    for k in 1:N+1
        k > 1 && (ph += angle(exp(im * (imag(LD[k]) - imag(LD[k-1])))))
        lM[k] = L0[k] - complex(real(LD[k]), ph) / 2 + im * u[k] / 2
    end
    QGrid(u, w, lM)
end
"""
P(Q <= q) and the density at q (the inversion formula of Gil-Pelaez on the grid G).  |q| at most 0.15/h, h = asinh(umax)/N
the step of the grid in v (1000 with the default grid): beyond, the steps of u near the origin do not resolve e^{-iuq}
(with a linear trend at c = 0, F came out as 1 - 4e-11 at q = 1e3, 1 - 8e-8 at 1e4, 3.0 at 1e5 and 24 at 1e6), and the
point is refused.
"""
function ptF(G::QGrid, q::Float64)
    isnan(q) && return (NaN, NaN)
    isinf(q) && return (q > 0 ? 1.0 : 0.0, 0.0)
    abs(q) * 2 * G.w[1] <= 0.15 || throw(ArgumentError("the point-optimal statistic: q = $q lies beyond the range of the inversion (|q| at most $(round(0.15 / (2 * G.w[1]), sigdigits = 3)) = 0.15 N/asinh(umax); a larger N extends it)"))
    F = 0.0; f = 0.0
    @inbounds for k in eachindex(G.u)
        z = exp(G.lM[k] - im * G.u[k] * q); F += imag(z) / G.u[k] * G.w[k]; f += real(z) * G.w[k]
    end
    0.5 - F / pi, f / pi
end
"""the p-quantile of Q on the grid G (Newton safeguarded by bisection on [1e-3, 400])"""
function ptquantile(G::QGrid, p::Real; q0::Real = 6.0)
    lo = 1e-3; hi = 400.0; q = Float64(q0)
    for _ in 1:100
        F, f = ptF(G, q); F > p ? (hi = q) : (lo = q)
        abs(F - p) < 1e-12 && break
        qn = q - (F - p) / f; q = (lo < qn < hi) ? qn : (lo + hi) / 2
    end
    q
end
"""
    pt_method(method, N, umax) -> (:inversion or :grid, N, umax)

The computation of the point-optimal functions: method = :auto (the default) and :inversion the single inversion of Bao
and Kan with the branch of the transform from the pivots and the tail completed by integration by parts (`pt_cdf1`,
src/DFKan_det.jl); method = :grid, or N or umax given, the inversion of Gil-Pelaez on the fixed grid of N steps up to umax
(60000 and 4000 by default; `qgrid`, `ptF`), the computation of before, with its refusals.
"""
function pt_method(method::Symbol, N, umax, c::Real = 0.0)
    method in (:auto, :inversion, :grid) || throw(ArgumentError("the point-optimal statistic: method must be :auto, :inversion (the single inversion of Bao and Kan) or :grid (the fixed grid of Gil-Pelaez)"))
    grid = method === :grid || (method === :auto && (N !== nothing || umax !== nothing || !(c <= PT_INV_CMAX)))
    (method === :inversion && (N !== nothing || umax !== nothing)) && throw(ArgumentError("the point-optimal statistic: N and umax belong to method = :grid"))
    grid ? (:grid, N === nothing ? 60000 : Int(N), umax === nothing ? 4000.0 : Float64(umax)) : (:inversion, 0, 0.0)
end
"""
the largest c at which the single inversion is the default of the point-optimal functions (method = :auto): compared in
November 2026 with the contour integral of the general engine on the point-optimal statistic's quadratic form (an
independent evaluation, the paths around the roots) for an intercept at cbar = -7, a linear trend at -13.5, Perron's
model B at -17.8 and a quadratic trend at -20, c from -30 to 7, the 1 to 99% points: within 3.3e-12, where the grid is off
by up to 6e-8 or refuses above the unit root (at c = 4, 5 with an intercept and a linear trend).  Beyond c = 4 the
evaluation can slow down by orders of magnitude (11 s a value for Perron's model B at c = 5, 7 s for a linear trend at c =
7, against 1 to 65 ms up to c = 4), and the fixed grid of before, with its refusal, is kept there.
"""
const PT_INV_CMAX = 4.0
"""the transform of the point-optimal statistic at c for the single inversion (the checks of `qgrid`)"""
function pt_transform(D::Determ, cbar::Real, c::Real)
    isfinite(c) || throw(ArgumentError("the point-optimal statistic: c must be a finite real scalar"))
    ptcf(prep_pt(D, Float64(cbar)), Float64(c))
end
"""the values of f(F, q) at the points qs (F the transform of the point-optimal statistic), with the error estimates and
the warning where they exceed the tolerance"""
function pt_values(f, fn::String, F::DetCF, cbar::Real, qs::AbstractVector, tol::Float64, err::Bool)
    R = [f(F, Float64(cbar), Float64(q), tol) for q in qs]
    P = first.(R); E = last.(R)
    bad = count(e -> !(e <= tol), E)
    bad > 0 && @warn "$fn: the error estimate of the single inversion exceeds the tolerance $tol at $bad of $(length(E)) points (largest $(maximum(E))); method = :grid gives the inversion on the fixed grid"
    err ? (P, E) : P
end
"""
    pt_cdf(D, qs, cbar, c = 0; method = :auto, tol = 1e-12, err = false, N, umax) -> P

The limit of P(P_T <= q) at the points qs, P_T the point-optimal statistic of Elliott, Rothenberg and Stock against cbar
(`prep_pt`) for the design D, under phi = 1 + c/n, zero start: one inversion of the characteristic function of the
quadratic form Q.  By default the single inversion of Bao and Kan ("Exact Finite-Sample Distribution and Moments of the
Least Squares Estimator in First-Order Autoregressions: O(1) Algorithms"): adaptive quadrature over decades with the
branch of the square root of the determinant from the pivots of the precision matrix at each point, and the tail
completed by integration by parts (`pt_cdf1`), for every c and q; `tol` is its absolute tolerance, `err = true` returns
`(P, errors)`, and a warning names the points at which the estimate exceeds the tolerance; up to c = PT_INV_CMAX (4),
beyond which the grid of before computes.  `method = :grid` (or N, umax given) is the inversion of Gil-Pelaez on the fixed
grid of before, refused where the grid does not resolve the
transform (`PT_EDGE`: far above the unit root, with a linear trend at cbar = -13.5 from about c = 3.7) and at points q
beyond its range (`ptF`: |q| > 1000 with the default grid).
"""
function pt_cdf(D::Determ, qs::AbstractVector, cbar::Real, c::Real = 0.0; method::Symbol = :auto, N = nothing, umax = nothing, tol::Real = BK.TOL, err::Bool = false)
    m, Ng, ug = pt_method(method, N, umax, c)
    if m === :grid
        G = qgrid(prep_pt(D, Float64(cbar)), Float64(cbar), Float64(c); N = Ng, umax = ug)
        v = [ptF(G, Float64(q))[1] for q in qs]
        return err ? (v, fill(NaN, length(v))) : v
    end
    pt_values(pt_cdf1, "pt_cdf", pt_transform(D, cbar, c), cbar, qs, Float64(tol), err)
end
"""the limiting density of the point-optimal statistic at qs (arguments as in `pt_cdf`; the single inversion by default,
the density of a quadratic form by one integral of the real part of its transform, `pt_pdf1`)"""
function pt_pdf(D::Determ, qs::AbstractVector, cbar::Real, c::Real = 0.0; method::Symbol = :auto, N = nothing, umax = nothing, tol::Real = BK.TOL, err::Bool = false)
    m, Ng, ug = pt_method(method, N, umax, c)
    if m === :grid
        G = qgrid(prep_pt(D, Float64(cbar)), Float64(cbar), Float64(c); N = Ng, umax = ug)
        v = [ptF(G, Float64(q))[2] for q in qs]
        return err ? (v, fill(NaN, length(v))) : v
    end
    pt_values(pt_pdf1, "pt_pdf", pt_transform(D, cbar, c), cbar, qs, Float64(tol), err)
end
"""
    pt_quantile(D, p, cbar; method = :auto, tol = 1e-12, N, umax) -> q

The p-quantile at c = 0 of the limit of the point-optimal statistic against cbar: the critical value of the level-p test
(it rejects for small values).  By default regula falsi on the single inversion of Bao and Kan, with the transform values
along the ray kept across the search (`pt_quantile1`); method = :grid Newton's method on the fixed grid (`ptquantile`).
"""
function pt_quantile(D::Determ, p::Real, cbar::Real; method::Symbol = :auto, N = nothing, umax = nothing, tol::Real = BK.TOL)
    0 < p < 1 || throw(ArgumentError("pt_quantile: p must lie in (0, 1)"))
    m, Ng, ug = pt_method(method, N, umax)
    m === :grid && return ptquantile(qgrid(prep_pt(D, Float64(cbar)), Float64(cbar), 0.0; N = Ng, umax = ug), p)
    pt_quantile1(pt_transform(D, cbar, 0.0), Float64(cbar), Float64(p), Float64(tol))[1]
end
"""
    pt_power(D, cbar, c; level = 0.05, method = :auto, tol = 1e-12, N, umax) -> (b, power)

The critical value b of the level-`level` point-optimal test against cbar and its limiting power at c (the single inversion
by default, as `pt_cdf` and `pt_quantile`).
"""
function pt_power(D::Determ, cbar::Real, c::Real; level::Real = 0.05, method::Symbol = :auto, N = nothing, umax = nothing, tol::Real = BK.TOL)
    0 < level < 1 || throw(ArgumentError("pt_power: the level must lie in (0, 1)"))
    m, Ng, ug = pt_method(method, N, umax, c)
    P = prep_pt(D, Float64(cbar))
    if m === :grid
        b = ptquantile(qgrid(P, Float64(cbar), 0.0; N = Ng, umax = ug), level)
        return b, ptF(qgrid(P, Float64(cbar), Float64(c); N = Ng, umax = ug), b)[1]
    end
    isfinite(c) || throw(ArgumentError("the point-optimal statistic: c must be a finite real scalar"))
    b = pt_quantile1(ptcf(P, 0.0), Float64(cbar), Float64(level), Float64(tol))[1]
    b, pt_cdf1(ptcf(P, Float64(c)), Float64(cbar), b, Float64(tol))[1]
end
"""
    pt_envelope(D, cs; level = 0.05, method = :auto, ...) -> P

The Gaussian power envelope for the design D at each c in cs: the power at c of the point-optimal test against cbar = c
(level for c >= 0).  For no deterministic terms, an intercept and a linear trend it is `power_envelope` (to 1e-8).  By
default by the single inversion of Bao and Kan (`pt_power`); method = :grid (or N, umax) the fixed grid of before.
"""
function pt_envelope(D::Determ, cs::AbstractVector; level::Real = 0.05, kw...)
    0 < level < 1 || throw(ArgumentError("pt_envelope: the level must lie in (0, 1)"))
    # (NaN at NaN, where it gave the level; 1 at -Inf, the limit)
    [isnan(c) ? NaN : c == -Inf ? 1.0 : Float64(c) < 0 ? pt_power(D, Float64(c), Float64(c); level, kw...)[2] : Float64(level) for c in cs]
end

# ---------------------------------------------------------------------------------------------------------------------
# The LM statistic (Schmidt and Phillips; with breaks, Amsler and Lee, Lee and Strazicich)
# ---------------------------------------------------------------------------------------------------------------------
"""
    prep_lm(D) -> (P, a1)

The prepared design of the denominator V = int Sbar^2 of the LM statistic (the construction of `prep_gls`, basis of
delta, delta', delta''; the numerator is a placeholder, the law of V does not use it), and the row a1 with J_0(1) =
a1' ell.  The trend coefficients are estimated from the first differences, S_t = y_t - psi - Z_t delta, and the
statistic is the t-ratio of S_{t-1} in the regression of the differences on the differenced trend functions and S_{t-1}:
with delta the trend functions that vanish at the origin and do not jump, Xi_0 = int delta' delta'' and eta_0 = int
delta' dJ, the detrended path is J_0 = J - delta' Xi_0^{-1} eta_0, the second regression projects it off the
derivatives, Sbar = J_0 - delta'' Xi_0^{-1} int delta' J_0, and the limit is {J_0(1)^2 - 1}/2/sqrt(V) for every c.  With a
linear trend among the trend functions J_0(1) = 0 (a1 = 0), and the statistic is -1/(2 sqrt V).
"""
function prep_lm(D::Determ)
    K = nreg(D); np = npc(D); br = copy(D.br)
    rows = [permutedims(dval(D, 0.0))]
    for j in 1:np-1
        lam = br[j+1]
        dl = [evalpoly(lam - poff(D, j + 1), view(D.C[j+1], :, i)) - evalpoly(lam - poff(D, j), view(D.C[j], :, i)) for i in 1:K]
        maximum(abs, dl; init = 0.0) > 1e-12 && push!(rows, permutedims(dl))
    end
    Mr = reduce(vcat, rows); Z = nullspace(Mr; atol = 1e-10 * max(1.0, maximum(abs, Mr))); k = size(Z, 2)
    k > 0 || throw(ArgumentError("prep_lm: no trend function that vanishes at the origin and does not jump"))
    offs = [poff(D, j) for j in 1:np]
    Cd = [D.C[j] * Z for j in 1:np]; Cd1 = [dcoef(Cd[j]) for j in 1:np]; Cd2 = [dcoef(Cd1[j]) for j in 1:np]
    ev(Cs, j, r) = [evalpoly(r - poff(D, j), view(Cs[j], :, i)) for i in 1:k]
    jl = Float64[]; vs = Vector{Float64}[]
    for j in 1:np-1
        lam = br[j+1]; s1 = ev(Cd1, j + 1, lam) .- ev(Cd1, j, lam)
        maximum(abs, s1; init = 0.0) > 1e-12 && (push!(jl, lam); push!(vs, s1))
    end
    nj = length(jl)
    F = [hcat(Cd[j], Cd1[j], Cd2[j]) for j in 1:np]; E = eigen(Symmetric(gram(F, br, D.loc)))
    keep = E.values .> 1e-11 * maximum(E.values); U = E.vectors[:, keep]; lv = E.values[keep]; p = length(lv)
    g = [F[j] * U * Diagonal(1 ./ sqrt.(lv)) for j in 1:np]
    Phi = U * Diagonal(sqrt.(lv)); Pd = Phi[1:k, :]; P1 = Phi[k+1:2k, :]; P2 = Phi[2k+1:3k, :]
    N = 1 + nj + p; iz = (2 + nj):N
    H0 = zeros(k, N); H0[:, 1] .= ev(Cd1, np, 1.0)
    for i in 1:nj; H0[:, 1+i] .= -vs[i]; end
    H0[:, iz] .= -P2
    Xi0 = P1 * P1'; S = zeros(p, N); S[:, iz] .= Matrix(I, p, p)
    Bb = Xi0 \ H0                                              # the coefficients of delta: b = Bb ell
    Ba = Xi0 \ (P1 * S .- (P1 * Pd') * Bb)                     # those of delta': a = Ba ell
    W = vcat(Bb, Ba); T = vcat(Pd, P1)                         # the fitted function is (delta, delta')' W ell = g' T' W ell
    B = -(S' * T' * W .+ W' * T * S) .+ W' * (T * T') * W; B = (B .+ B') ./ 2
    a1 = zeros(N); a1[1] = 1; a1 .-= Bb' * ev(Cd, np, 1.0)     # J_0(1) = a1' ell
    A = zeros(N, N); A[1, 1] = 0.5                             # placeholder
    # (the last field :lm: V is not the norm of an orthogonal projection of J, and `tstar` takes the estimate of the edge
    # t_* from the quadratic form itself, `eigtop_q`)
    P = Prep(D, k, p, nj, N, size(F[1], 1) - 1, br, g, jl, A, B, reshape([1.0; zeros(N - 1)], N, 1), [0.5], 1, 1, 0, Pd * Pd', zeros(N), offs, NaN, :lm)
    P, a1
end
"""
    span_has_constant(D; tol = 1e-6) -> Bool

Whether the constant function lies in the span of the regressors of D.  Every design is piecewise polynomial (`Determ`:
on piece j the coefficients C[j] of the powers of r, or of r - br[j], of the K regressors), so the question is decided on
values: the regressors at 2(deg + 1) Chebyshev points inside each piece (more than determine each piece's polynomials),
the least squares fit of the constant 1 by them (pivoted QR) and its largest residual, against tol.  For polynomial
trends, breaks in level and slope, dummies and the designs of `perron`, `trend` and `ltrend` the residual is zero to
rounding (1e-15); for a function design (`fdesign`) and a design from a matrix (`design_from_matrix`) the check is made on
their piecewise polynomial interpolants, which reproduce the regressors to 1e-14 (1e-8 for the matrix fits), so a span
that contains the constant through a combination (sin^2 and cos^2, say) gives a residual at that level, and tol = 1e-6
separates it from a span without the constant, whose residual is of order one unless a combination of the regressors is
constant to within 1e-6 on [0, 1] (which the check then accepts).  (The coefficients themselves are not compared: the
high-order local Taylor coefficients of a function design carry rounding of up to 1e-2 that cancels in the values.)
"""
function span_has_constant(D::Determ; tol::Float64 = 1e-6)
    K = nreg(D); K == 0 && return false
    np = npc(D); m = 2 * size(D.C[1], 1)
    X = zeros(np * m, K)
    for j in 1:np, k in 1:m
        a = D.br[j]; b = D.br[j+1]
        r = (a + b) / 2 + (b - a) / 2 * cos(pi * (k - 0.5) / m)
        for i in 1:K; X[(j-1)*m+k, i] = evalpoly(r - poff(D, j), view(D.C[j], :, i)); end
    end
    # (with one BLAS thread, as the rest of the engine: BLAS with its default threads leaves OpenBLAS's workers spinning,
    # which slowed the computation that followed by 1.5 times in line_refusal)
    blas1() do
        beta = qr(X, ColumnNorm()) \ ones(np * m)
        maximum(abs, X * beta .- 1) <= tol
    end
end
const LM_CONST_MSG = "the LM statistic: the limit computed here holds only for trend functions whose span contains the constant (the LM regression estimates the level from the first observation, and the limit of the statistic is otherwise a different one); the design"
"""the LM statistic's prepared design, with the checks that the span of the design contains the constant
(`span_has_constant`) and that J_0(1) vanishes (a linear trend among the trend functions)"""
function lmprep(D::Determ)
    span_has_constant(D) || throw(ArgumentError("$LM_CONST_MSG $(D.name) does not contain it"))
    P, a1 = prep_lm(D)
    maximum(abs, a1) < 1e-9 || throw(ArgumentError("the LM statistic: the law of -1/(2 sqrt V) needs a linear trend among the trend functions (J_0(1) does not vanish)"))
    P
end
lmV(CP::CPath, v::Real) = 1 - vsum(CP, Float64(v), true)
"""
the largest c of the LM statistic's functions.  Above it the inversion of V moves between contours (hyperbolas at 30 to
75 degrees through 0.3 to 0.7 t_*) by more than 1e-8 with a linear trend: by 5e-9 at c = 10, 2e-7 at c = 12, 3e-5 at
c = 15, 0.1 at c = 20, and a finer step does not remove it, whereas the V of least squares detrending, on the same
contours and by the same sum (`vsum`), holds to 3e-15 at c = 15.  The loss lies in the transform of the LM statistic's
V far above the unit root (not identified further); there the functions refuse.
"""
const LM_CMAX = 10.0
"""the outer contour of the LM statistic's V: the path of the law of V alone (`v_own`), which above the unit root replaces
the vertical line of the path of the t-ratio by a hyperbola (on the line P(V <= y) was off by 6e-3 at c = 5 with a linear
trend, and F exceeded 1); c at most LM_CMAX"""
function lmpath(P::Prep, c::Real)
    isfinite(c) || throw(ArgumentError("the LM statistic: c must be a finite real scalar"))
    Float64(c) <= LM_CMAX || throw(ArgumentError("the LM statistic: c must be at most $LM_CMAX (above, its distribution is not computed accurately: off by 2e-7 at c = 12 and by 3e-5 at c = 15)"))
    v_own(cpath(model(P), Float64(c)); build = cpath)
end
"""
    lm_cdf(D, xs, c = 0) -> F

The limit of P(tau_LM <= x) for the LM statistic with the trend functions of D (a linear trend among them), under phi = 1
+ c/n: tau_LM => -1/(2 sqrt V) (`prep_lm`), so that F(x) = P(V <= 1/(4 x^2)) for x < 0 and 1 for x >= 0, one inversion of
the transform of V on the path of V alone (`lmpath`).  c at most LM_CMAX = 10 (as for `lm_quantile` and `lm_power`).
The limit holds only for trend functions whose span contains the constant (the LM regression estimates the level from
the first observation): for any other design, e^r alone or r alone, say, the three functions raise an ArgumentError
(`span_has_constant`, decided exactly on the piecewise polynomial form of the design; for `fdesign` and
`design_from_matrix` on their interpolants, to 1e-6).
"""
function lm_cdf(D::Determ, xs::AbstractVector, c::Real = 0.0)
    CP = lmpath(lmprep(D), c)
    [isnan(x) ? NaN : x >= 0 ? 1.0 : x == -Inf ? 0.0 : lmV(CP, 1 / (4 * Float64(x)^2)) for x in xs]
end
"""
    lm_quantile(D, p; c = 0) -> q

The p-quantile of the limit of the LM statistic (the critical value of the level-p test, which rejects for small values),
from the quantile of V by bisection in log V.
"""
function lm_quantile(D::Determ, p::Real; c::Real = 0.0)
    0 < p < 1 || throw(ArgumentError("lm_quantile: p must lie in (0, 1)"))
    CP = lmpath(lmprep(D), c); lo, hi = 1e-6, 2.0
    while lmV(CP, hi) < p; hi *= 2; end
    for _ in 1:70; m = sqrt(lo * hi); lmV(CP, m) > p ? (hi = m) : (lo = m); end
    -0.5 / sqrt(sqrt(lo * hi))
end
"""
    lm_power(D, cs; level = 0.05) -> (q, power)

The critical value q of the level-`level` LM test and its limiting power at each c in cs.
"""
function lm_power(D::Determ, cs::AbstractVector; level::Real = 0.05)
    P = lmprep(D); q = lm_quantile(D, level)
    q, [lmV(lmpath(P, c), 1 / (4q^2)) for c in cs]
end

# ---------------------------------------------------------------------------------------------------------------------
# The companions of the GLS-detrended t-ratio: MZ_alpha and MSB
# ---------------------------------------------------------------------------------------------------------------------
"""
    mza_cdf(D, ys, c = 0; gls = -13.5, kw...) -> F

The limit of P(MZ_alpha <= y) for the modified coefficient statistic of Ng and Perron with the data GLS-detrended at
cbar = gls: its limit is that of the coefficient statistic of the GLS-detrended regression, (U - c0 V)/V (`coef_cdf`).
"""
mza_cdf(D::Determ, ys::AbstractVector, c::Real = 0.0; gls::Real = -13.5, kw...) = coef_cdf(D, ys, c; gls = Float64(gls), kw...)
"""the p-quantile of the limit of MZ_alpha (arguments as in `mza_cdf`; y0 the start of the search)"""
mza_quantile(D::Determ, p::Real, c::Real = 0.0; gls::Real = -13.5, y0::Real = -20.0, kw...) = coef_quantile(D, p, c; gls = Float64(gls), y0, kw...)
"""
    msb_cdf(D, xs, c = 0; gls = -13.5) -> F

The limit of P(MSB <= x) for the statistic MSB of Ng and Perron (Stock's modified Sargan-Bhargava statistic) with the data
GLS-detrended at cbar = gls: MSB => sqrt(V_g), so that F(x) = P(V_g <= x^2) (`v_cdf`).
"""
msb_cdf(D::Determ, xs::AbstractVector, c::Real = 0.0; gls::Real = -13.5, kw...) =
    [x <= 0 ? (isnan(x) ? NaN : 0.0) : v_cdf(D, [Float64(x)^2], c; gls = Float64(gls), kw...)[1] for x in xs]
"""the p-quantile of the limit of MSB (by bisection on its distribution function; it rejects for small values)"""
function msb_quantile(D::Determ, p::Real, c::Real = 0.0; gls::Real = -13.5)
    0 < p < 1 || throw(ArgumentError("msb_quantile: p must lie in (0, 1)"))
    lo, hi = 1e-4, 1.0
    while v_cdf(D, [hi], c; gls = Float64(gls))[1] < p; hi *= 2; end
    for _ in 1:60; m = (lo + hi) / 2; v_cdf(D, [m], c; gls = Float64(gls))[1] > p ? (hi = m) : (lo = m); end
    sqrt((lo + hi) / 2)
end
