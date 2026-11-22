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
function qgrid(P::Prep, cbar::Float64, c::Float64; N::Int = 60000, umax::Float64 = 4000.0)
    Mo = model(P); u, w = gpgrid(N, umax)
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
"""P(Q <= q) and the density at q (the inversion formula of Gil-Pelaez on the grid G)"""
function ptF(G::QGrid, q::Float64)
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
    pt_cdf(D, qs, cbar, c = 0; N = 60000, umax = 4000) -> P

The limit of P(P_T <= q) at the points qs, P_T the point-optimal statistic of Elliott, Rothenberg and Stock against cbar
(`prep_pt`) for the design D, under phi = 1 + c/n, zero start, by the inversion formula of Gil-Pelaez (one inversion of
the characteristic function of the quadratic form Q).
"""
function pt_cdf(D::Determ, qs::AbstractVector, cbar::Real, c::Real = 0.0; N::Int = 60000, umax::Real = 4000.0)
    G = qgrid(prep_pt(D, Float64(cbar)), Float64(cbar), Float64(c); N, umax = Float64(umax))
    [ptF(G, Float64(q))[1] for q in qs]
end
"""the limiting density of the point-optimal statistic at qs (arguments as in `pt_cdf`)"""
function pt_pdf(D::Determ, qs::AbstractVector, cbar::Real, c::Real = 0.0; N::Int = 60000, umax::Real = 4000.0)
    G = qgrid(prep_pt(D, Float64(cbar)), Float64(cbar), Float64(c); N, umax = Float64(umax))
    [ptF(G, Float64(q))[2] for q in qs]
end
"""
    pt_quantile(D, p, cbar; N = 60000, umax = 4000) -> q

The p-quantile at c = 0 of the limit of the point-optimal statistic against cbar: the critical value of the level-p test
(it rejects for small values).
"""
pt_quantile(D::Determ, p::Real, cbar::Real; N::Int = 60000, umax::Real = 4000.0) =
    ptquantile(qgrid(prep_pt(D, Float64(cbar)), Float64(cbar), 0.0; N, umax = Float64(umax)), p)
"""
    pt_power(D, cbar, c; level = 0.05) -> (b, power)

The critical value b of the level-`level` point-optimal test against cbar and its limiting power at c.
"""
function pt_power(D::Determ, cbar::Real, c::Real; level::Real = 0.05, N::Int = 60000, umax::Real = 4000.0)
    P = prep_pt(D, Float64(cbar))
    b = ptquantile(qgrid(P, Float64(cbar), 0.0; N, umax = Float64(umax)), level)
    b, ptF(qgrid(P, Float64(cbar), Float64(c); N, umax = Float64(umax)), b)[1]
end
"""
    pt_envelope(D, cs; level = 0.05) -> P

The Gaussian power envelope for the design D at each c in cs: the power at c of the point-optimal test against cbar = c
(level for c >= 0).  For no deterministic terms, an intercept and a linear trend it is `power_envelope` (to 1e-8).
"""
pt_envelope(D::Determ, cs::AbstractVector; level::Real = 0.05, kw...) =
    [Float64(c) < 0 ? pt_power(D, Float64(c), Float64(c); level, kw...)[2] : Float64(level) for c in cs]

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
"""the LM statistic's prepared design, with the check that J_0(1) vanishes (a linear trend among the trend functions)"""
function lmprep(D::Determ)
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
    Float64(c) <= LM_CMAX || throw(ArgumentError("the LM statistic: c must be at most $LM_CMAX (above, its distribution is not computed accurately: off by 2e-7 at c = 12 and by 3e-5 at c = 15)"))
    v_own(cpath(model(P), Float64(c)); build = cpath)
end
"""
    lm_cdf(D, xs, c = 0) -> F

The limit of P(tau_LM <= x) for the LM statistic with the trend functions of D (a linear trend among them), under phi = 1
+ c/n: tau_LM => -1/(2 sqrt V) (`prep_lm`), so that F(x) = P(V <= 1/(4 x^2)) for x < 0 and 1 for x >= 0, one inversion of
the transform of V on the path of V alone (`lmpath`).  c at most LM_CMAX = 10 (as for `lm_quantile` and `lm_power`).
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
