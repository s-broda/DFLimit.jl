"""
    DFDet

A research prototype: the exact limiting distribution of the Dickey-Fuller t-ratio for arbitrary piecewise polynomial
deterministic regressors (polynomial trends of any order, level and slope breaks at known dates), by the inversion
formula on paths around the singularities of the joint moment generating function (the contour integral
representation of Bao, Broda and Kan), generalized from a determinant of degree at most two in s to any degree.

The model.  J is the Ornstein-Uhlenbeck process dJ = c J dr + dW on [0,1], J(0) = 0.  The regressors d(r) in R^K are
piecewise polynomial with break points 0 < lambda_1 < ... < lambda_q < 1, G = int d d', m = int d J, and

    V = int J^2 - m' G^{-1} m,   U = (J(1)^2 - 1)/2 - m' G^{-1} w,   w = int d dJ = d(1) J(1) - int d' J - sum_i Delta_i J(lambda_i),

with Delta_i the jump of d at lambda_i.  The Gaussian vector ell = (J(1), J(lambda_i) at the jumps, z) with z = int g J for
an L2-orthonormal basis g of span{d, d'} makes U + 1/2 = ell' A ell and V = int J^2 + ell' B ell; the basis removes the
linear dependences between m and n = int d' J, so that the covariance Sigma of ell is nonsingular and the rank r of A is
the number of roots in s.  Girsanov's theorem with theta = sqrt(c^2 - 2t), gamma = (c + theta)/2 gives

    M(s,t) = E exp(sU + tV) = e^{-s/2} e^{-gamma} det(I - 2 (s A + t B + gamma E11) Sigma(theta))^{-1/2},

Sigma(theta) the covariance of ell under the Ornstein-Uhlenbeck law with parameter -theta started at zero.  The
function computes Sigma as int_0^1 F(u) F(u)' du with F the transfer functions of the functionals
(ell_k = int_0^1 F_k(u) dW(u)), which are in closed form on each piece in the functions e_k(z) = int_0^1 x^k e^{-zx} dx
(series or a stable recurrence, no cancellation for small theta), and the integral over u by Gauss-Legendre panels graded
toward the right end of each piece (the boundary layer e^{-theta (b - u)}).
"""
module DFDet

using LinearAlgebra, SpecialFunctions, Printf, Random
import StaticArrays
using StaticArrays: SMatrix, SVector, MMatrix, MVector

export Determ, trend, perron, Prep, prep, prep_gls, Model, model, Qf, sigma, mgfparts, tparts, logMtilde, rootpath, Dval, logmgf, tstar, linepath, rootinfo,
       CPath, cpath, contour_eval, dist_cdf, dist_pdf, dist_quantile, simulate_tratio,
       coef_cdf, coef_pdf, coef_quantile, coef_cdf_gp, dist_moments, local_power, fdesign, ltrend, design_from_matrix, regmat, simulate_gls,
       detrend_first, simulate_first,
       XSpec, contour_eval_specs, lin_cdf, lin_pdf, lin_quantile, v_cdf, v_pdf, power_envelope, adfmax_cdf, adfmax_pdf, adfmax_quantile,
       region_prob, empty_caches!

# =====================================================================================================================
# Gauss-Legendre rules on [0, 1]
# =====================================================================================================================
function gl(n::Int)
    b = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), b))
    x = (E.values .+ 1) ./ 2; w = E.vectors[1, :] .^ 2
    x, w
end
const GLR = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const GLLOCK = ReentrantLock()
glrule(n::Int) = lock(GLLOCK) do; get!(() -> gl(n), GLR, n); end

# =====================================================================================================================
# The deterministic terms
# =====================================================================================================================
"""
    Determ(br, C, name)

Piecewise polynomial regressors on [0, 1]: `br` the piece boundaries 0 = br[1] < ... < br[end] = 1, `C[j]` the
(deg+1) x K matrix of the coefficients of r^k (k = 0..deg, in the global variable r) of the K regressors on piece j.
A regressor is left-continuous at a break: piece j covers br[j] < r <= br[j+1] (r = 0 belongs to piece 1), so that a
level-shift dummy 1{r > lambda} is 0 at r = lambda, as 1{t > T_B} in the regression.  Determ(br, C, name, true): the
coefficients are those of the local variable r - br[j] on piece j (the form `fdesign` builds, well conditioned at high
degree on short pieces).  The last field is the statistic: :onestep, the t-ratio of y_{t-1} in the regression of y_t on
d(t/n) and y_{t-1} (the default), or :first, the statistic that detrends first (`detrend_first`).
"""
struct Determ
    br::Vector{Float64}
    C::Vector{Matrix{Float64}}
    name::String
    loc::Bool
    detrend::Symbol             # :onestep (y_t on d(t/n) and y_{t-1}) or :first (detrend first, `detrend_first`)
end
Determ(br::Vector{Float64}, C::Vector{Matrix{Float64}}, name::String) = Determ(br, C, name, false, :onestep)
Determ(br::Vector{Float64}, C::Vector{Matrix{Float64}}, name::String, loc::Bool) = Determ(br, C, name, loc, :onestep)
nreg(D::Determ) = size(D.C[1], 2)
npc(D::Determ) = length(D.br) - 1
"""the origin of the variable of the coefficients of piece j: 0 (global r) or br[j] (local)"""
poff(D::Determ, j::Int) = D.loc ? D.br[j] : 0.0
function piece(D::Determ, r::Real)
    j = 1
    while j < npc(D) && r > D.br[j+1]; j += 1; end
    j
end
dval(D::Determ, r::Real) = (j = piece(D, r); [evalpoly(r - poff(D, j), view(D.C[j], :, i)) for i in 1:nreg(D)])
dkey(D::Determ) = hash((D.br, D.C, D.loc, D.detrend))

"""trend(p): d = (1, r, ..., r^p); p = -1 no deterministic terms (case DK), 0 intercept (DU), 1 linear trend (DT)."""
function trend(p::Int)
    p >= -1 || error("trend: p >= -1")
    K = p + 1; deg = max(p, 0)
    C = zeros(deg + 1, K)
    for i in 1:K; C[i, i] = 1.0; end
    name = p == -1 ? "no deterministic terms (DK)" : p == 0 ? "intercept (DU)" : p == 1 ? "linear trend (DT)" :
           p == 2 ? "quadratic trend" : p == 3 ? "cubic trend" : "polynomial trend of order $p"
    Determ([0.0, 1.0], [C], name)
end

"""perron(model, lambda): Perron's model A (1, r, 1{r > lambda}), B (1, r, (r - lambda)+) or C (1, r, 1{r > lambda},
(r - lambda)+), the break at the known fraction lambda."""
function perron(model::Symbol, lam::Real)
    0 < lam < 1 || error("perron: 0 < lambda < 1")
    if model == :A
        C1 = [1.0 0 0; 0 1 0]; C2 = [1.0 0 1; 0 1 0]
    elseif model == :B
        C1 = [1.0 0 0; 0 1 0]; C2 = [1.0 0 -lam; 0 1 1]
    elseif model == :C
        C1 = [1.0 0 0 0; 0 1 0 0]; C2 = [1.0 0 1 -lam; 0 1 0 1]
    else
        error("perron: model A, B or C")
    end
    Determ([0.0, Float64(lam), 1.0], [C1, C2], "Perron model $model, lambda = $lam")
end

# ---------- trend functions beyond piecewise polynomials: Chebyshev interpolants, local Taylor coefficients ----------
"""Chebyshev coefficients (degree N - 1) of the interpolant of the columns of f on [a, b] at the N points of the first kind
(interior points only, so that f is never evaluated at a break)"""
function chebcoef(f, a::Float64, b::Float64, N::Int, K::Int)
    x = [cos(pi * (j + 0.5) / N) for j in 0:N-1]
    V = reduce(hcat, [Float64.(f((a + b) / 2 + (b - a) / 2 * xj)) for xj in x])     # K x N
    size(V, 1) == K || error("fdesign: the function returns $(size(V, 1)) values, expected $K")
    Cc = zeros(N, K)
    for k in 0:N-1, i in 1:K
        acc = 0.0
        for j in 1:N; acc += V[i, j] * cos(k * pi * (j - 0.5) / N); end
        Cc[k+1, i] = 2acc / N
    end
    Cc[1, :] ./= 2
    Cc
end
"""the derivative of a Chebyshev series (coefficients c, variable x in [-1, 1])"""
function chebder(c::AbstractVector{Float64})
    n = length(c) - 1
    n == 0 && return [0.0]
    d = zeros(n + 1)
    for k in n:-1:1
        d[k] = (k + 1 <= n ? d[k+2] : 0.0) + 2k * c[k+1]
    end
    d[1] /= 2
    d[1:n]
end
"""Clenshaw evaluation of a Chebyshev series at x"""
function chebval(c::AbstractVector{Float64}, x::Float64)
    b1 = 0.0; b2 = 0.0
    for k in length(c):-1:2
        b1, b2 = 2x * b1 - b2 + c[k], b1
    end
    x * b1 - b2 + c[1]
end
"""the Taylor coefficients g^{(k)}(r0)/k!, k = 0..n, of the Chebyshev series c on [a, b] at r0 (the derivative series and
Clenshaw, scaled by (2/(b - a))^k / k!)"""
function cheb_taylor(c::Vector{Float64}, a::Float64, b::Float64, r0::Float64, n::Int)
    x0 = (2r0 - a - b) / (b - a); sc = 2 / (b - a)
    out = zeros(n + 1); cd = copy(c); f = 1.0
    for k in 0:n
        out[k+1] = chebval(cd, x0) * f
        cd = chebder(cd); f *= sc / (k + 1)
    end
    out
end

"""
    fdesign(f, breaks; K, name, tol = 1e-14, maxdeg = 96, sub = nothing) -> Determ

Deterministic terms given by a function: f(r) returns the vector of the K regressors at r, smooth on each piece between
the breaks (0 < breaks < 1; jumps and kinks only there; left-continuous at a break, as in `Determ`).  On each piece every
regressor is replaced by its Chebyshev interpolant (first kind) of the smallest degree 2^m - 1 >= 15 at which the
coefficients beyond the last that exceeds tol times the largest have fallen below it (an error beyond maxdeg).  For the
covariance the transfer-function formula needs Taylor coefficients, whose sum over the whole piece is ill conditioned at
high degree (the monomials of sin(4 pi r) on [0, 1] reach 3e4); the piece is therefore cut into `sub` computational
sub-pieces (default ceil(deg/4)) and the interpolant is re-expanded about the left end of each (`cheb_taylor`), where
the terms g^(k) h^k / k! stay of order one.  The design carries these local coefficients (Determ(..., true)).
"""
function fdesign(f, breaks::AbstractVector = Float64[]; K::Int = length(f(0.5)), name::String = "function design", tol::Float64 = 1e-14,
                 maxdeg::Int = 96, sub = nothing)
    bk = [0.0; sort(Float64.(collect(breaks))); 1.0]
    all(diff(bk) .> 0) || error("fdesign: breaks must lie strictly inside (0, 1) and be distinct")
    pieces = Tuple{Float64,Float64,Vector{Vector{Float64}},Int}[]
    for j in 1:length(bk)-1
        a = bk[j]; b = bk[j+1]
        N = 16; Cc = zeros(0, 0); deg = 0
        while true
            Cc = chebcoef(f, a, b, N, K)
            mx = maximum(abs, Cc)
            last = mx == 0 ? 0 : maximum(k for k in 1:N if any(abs.(Cc[k, :]) .> tol * mx))
            if last <= N - 6 || N > maxdeg
                deg = last - 1
                break
            end
            N *= 2
        end
        N > maxdeg && deg > maxdeg - 6 && error("fdesign: on piece [$a, $b] the Chebyshev coefficients do not fall below $tol by degree $maxdeg; the regressors must be smooth between the breaks")
        push!(pieces, (a, b, [Cc[1:deg+1, i] for i in 1:K], deg))
    end
    D = maximum(p[4] for p in pieces)
    br = Float64[]; Cs = Matrix{Float64}[]
    for (a, b, cs, deg) in pieces
        m = sub === nothing ? max(1, ceil(Int, deg / 4)) : Int(sub)
        for q in 0:m-1
            r0 = a + (b - a) * q / m
            push!(br, r0)
            Cm = zeros(D + 1, K)
            for i in 1:K
                Cm[1:deg+1, i] .= cheb_taylor(cs[i], a, b, r0, deg)
            end
            push!(Cs, Cm)
        end
    end
    push!(br, 1.0)
    Determ(br, Cs, name, true)
end

"""
    ltrend(p) -> Determ

The polynomial trend of order p in the orthonormal (shifted Legendre) basis sqrt(2k + 1) P_k(2r - 1), k = 0..p: the same
span as trend(p), hence the same statistic, with G = I instead of a Hilbert matrix (condition 5e11 at p = 8, where the
eigenvalue threshold of `prep` would drop directions of the monomial basis).
"""
function ltrend(p::Int)
    function leg(r)
        x = 2r - 1; P = zeros(p + 1); P[1] = 1.0
        p >= 1 && (P[2] = x)
        for k in 1:p-1; P[k+2] = ((2k + 1) * x * P[k+1] - k * P[k]) / (k + 1); end
        [sqrt(2k + 1) * P[k+1] for k in 0:p]
    end
    fdesign(leg; K = p + 1, name = "polynomial trend of order $p (Legendre basis)")
end

"""
    perron(model, lams::Vector) -> Determ

Perron's models with several breaks at the fractions lams: A (1, r, 1{r > lambda_i}), B (1, r, (r - lambda_i)+), C (both
for every break).
"""
function perron(model::Symbol, lams::AbstractVector)
    lam = sort(Float64.(collect(lams))); q = length(lam)
    all(0 .< lam .< 1) && all(diff(lam) .> 0) || error("perron: 0 < lambda_1 < ... < lambda_q < 1")
    model in (:A, :B, :C) || error("perron: model A, B or C")
    K = 2 + (model == :C ? 2q : q)
    C = Matrix{Float64}[]
    for j in 1:q+1                                     # piece j: after the first j - 1 breaks
        Cj = zeros(2, K); Cj[1, 1] = 1; Cj[2, 2] = 1
        col = 3
        for i in 1:q
            on = i <= j - 1
            if model in (:A, :C)
                on && (Cj[1, col] = 1.0); col += 1
            end
            if model in (:B, :C)
                on && (Cj[1, col] = -lam[i]; Cj[2, col] = 1.0); col += 1
            end
        end
        push!(C, Cj)
    end
    Determ([0.0; lam; 1.0], C, "Perron model $model, lambda = $(join(lam, ", "))")
end

"""
    design_from_matrix(X; breaks = nothing, name, tol = 1e-8, maxdeg = 40, minlen = 10) -> Determ

The design of the limit from the n x K matrix X of the deterministic regressors used in a regression (column k sampled at
t = 1..n, any scaling).  The known breaks, if given, are the indices T_B after which the regressors change (a regime of
t <= T_B and one of t > T_B, as 1{t > T_B}), at lambda = T_B/n.  Without them the breaks are detected: on each column
scaled to unit maximum the sixth differences of a function smooth on the scale of the sample are of order n^{-6}, while a
jump or a kink at T leaves a cluster of differences of order one or 1/n around T; each cluster gives the largest T
(after which the regime changes) at which polynomials of degree at most 8 fit 30 observations on either side exactly
(for a kink of a continuous regressor both T - 1 and T fit; T is the correct one).  On each piece every column is then
fitted by least squares in Chebyshev polynomials of increasing degree up to maxdeg until the largest residual is below
tol (exact for polynomial and dummy columns, a smooth interpolant for others), and the design is the function design of
these fits (`fdesign`).  Columns that are not piecewise smooth at the sample's resolution (a piece shorter than minlen
observations, as a pulse dummy makes, or no fit within maxdeg, as seasonal dummies give) are refused with an error: they
do not affect the limit and should be left out.  Columns linearly dependent on the others are dropped.
"""
function design_from_matrix(X::AbstractMatrix; breaks = nothing, name::String = "design from a matrix", tol::Float64 = 1e-8,
                            maxdeg::Int = 40, minlen::Int = 10)
    n, K = size(X)
    Xs = Float64.(X)
    for k in 1:K
        m = maximum(abs, view(Xs, :, k)); m > 0 && (Xs[:, k] ./= m)
    end
    # dependent columns
    F = qr(Xs, ColumnNorm())
    rk = count(abs.(diag(F.R)) .> 1e-9 * abs(F.R[1, 1]))
    keep = sort(F.p[1:rk])
    rk < K && @warn "design_from_matrix: columns $(setdiff(1:K, keep)) are linearly dependent on the others and are dropped"
    Xs = Xs[:, keep]; K = rk
    pfit(y, deg) = (V = [cos(j * acos(clamp(x, -1, 1))) for x in range(-1, 1, length = length(y)), j in 0:deg]; c = V \ y; (c, maximum(abs, V * c - y)))
    if breaks === nothing
        q = 6; T = Int[]
        flagged = falses(n)
        for k in 1:K
            d = Xs[:, k]
            for _ in 1:q; d = diff(d); end
            for t in eachindex(d); abs(d[t]) > 1e-7 && (flagged[t] = true); end
        end
        idx = findall(flagged); cl = Vector{Int}[]
        for t in idx
            if isempty(cl) || t - cl[end][end] > q
                push!(cl, [t])
            else
                push!(cl[end], t)
            end
        end
        length(cl) > max(10, n ÷ 50) && throw(ArgumentError("design_from_matrix: $(length(cl)) clusters of jumps or kinks: some column is not piecewise smooth at the sample's resolution (seasonal dummies?); such columns do not affect the limit and should be left out"))
        w = 30
        for c in cl
            best = 0
            for Tc in max(c[1] - 1, 1):min(c[end] + q + 1, n - 1)
                lo = max(1, Tc - w + 1); hi = min(n, Tc + w)
                (Tc - lo + 1 < 3 || hi - Tc < 3) && continue
                ok = all(1:K) do k
                    min(pfit(Xs[lo:Tc, k], min(8, Tc - lo))[2], 1.0) <= 1e-9 && min(pfit(Xs[Tc+1:hi, k], min(8, hi - Tc - 1))[2], 1.0) <= 1e-9
                end
                ok && (best = Tc)
            end
            best == 0 && throw(ArgumentError("design_from_matrix: no break position fits the cluster of changes at t = $(c[1])..$(c[end]) (a pulse, or a column that is not piecewise smooth?); such columns do not affect the limit and should be left out"))
            push!(T, best)
        end
        breaks = T
    end
    TB = sort(Int.(collect(breaks)))
    bk = [0; TB; n]
    any(diff(bk) .< minlen) && throw(ArgumentError("design_from_matrix: a piece shorter than $minlen observations (breaks at $(TB)): a pulse or a feature at the sample's resolution; such columns do not affect the limit and should be left out"))
    coefs = Vector{Vector{Vector{Float64}}}()
    for j in 1:length(bk)-1
        t1 = bk[j] + 1; t2 = bk[j+1]
        cj = Vector{Float64}[]
        for k in 1:K
            y = Xs[t1:t2, k]; got = false
            for deg in 0:min(maxdeg, t2 - t1)
                c, res = pfit(y, deg)
                if res <= tol
                    push!(cj, c); got = true; break
                end
            end
            got || throw(ArgumentError("design_from_matrix: column $(keep[k]) is not smooth on observations $t1..$t2 at the sample's resolution (no Chebyshev fit of degree <= $maxdeg within $tol); such columns do not affect the limit and should be left out"))
        end
        push!(coefs, cj)
    end
    lams = TB ./ n
    # the fitted polynomials in the variable of piece j: t in [t1, t2] maps to x in [-1, 1]; t = n r
    function f(r)
        j = 1; while j <= length(TB) && r > lams[j]; j += 1; end
        t1 = bk[j] + 1; t2 = bk[j+1]
        x = t2 == t1 ? 0.0 : (2 * (n * r) - t1 - t2) / (t2 - t1)
        [chebval(coefs[j][k], x) for k in 1:K]
    end
    fdesign(f, lams; K, name)
end

"""
    detrend_first(D) -> Determ

The design D for the statistic that detrends first (Perron and Vogelsang): y_t is regressed on d(t/n), t = 1..n, and the
t-ratio is that of ytilde_{t-1} in the regression of Delta ytilde_t on ytilde_{t-1}, t = 2..n, ytilde the residual, with a
dummy for the single observation after each jump of d.  The one-step regression of y_t on d(t/n) and y_{t-1} is
invariant to the coefficients of the trend only when the span of d is closed under differentiation (polynomial trends,
Perron's models A and C); for a break in the slope alone (model B, the trend joined at the break) it is not, and the test
that detrends first is the valid one.  Its limit is U_d/sqrt(V) with the same V and U_d = U - m' G^{-1} int d' Jtilde (d'
the piecewise derivative, Jtilde = J - d' G^{-1} m the projection residual), so that U_d + 1/2 = ell' A_d ell with A_d equal
to A except in the block of z, which is sym((Pm Phi_n)(Pm Phi_d)) instead of sym(Pm Phi_n) (`prep`).  For a continuous d,
U_d + 1/2 = (Jtilde(1)^2 - Jtilde(0)^2)/2, of rank two; for a span closed under differentiation A_d = A.  Zero start only;
the distribution function, the density, the quantiles, local power, a local null value c0 and the coefficient statistic
are supported (and validated by `scripts/check_detrend_first.jl`); moments and the fixed and the stationary start are
refused, and GLS detrending detrends first already.  `simulate_tratio` simulates the two-step regression for such a design.
"""
detrend_first(D::Determ) = Determ(D.br, D.C, D.name * " (detrended first)", D.loc, :first)

"""whether the span of the regressors of D is closed under (piecewise) differentiation: the residuals of d' after projection
on span d vanish (the Gram matrix R of `kappa_span`, relative to int d' d'^T)"""
function closed_under_derivative(D::Determ)
    K = nreg(D); np = npc(D)
    K == 0 && return true
    F = [hcat(D.C[j], dcoef(D.C[j])) for j in 1:np]
    Gf = gram(F, D.br, D.loc)
    G = Gf[1:K, 1:K]; Cx = Gf[1:K, K+1:2K]; E = Gf[K+1:2K, K+1:2K]
    R = E - transpose(Cx) * (G \ Cx)
    maximum(abs, R) <= 1e-9 * max(1.0, maximum(abs, E))
end

# derivative of the coefficient matrix (rows: powers), same size, last row zero
function dcoef(C::Matrix{Float64})
    Dm = zeros(size(C))
    for k in 1:size(C, 1)-1, i in 1:size(C, 2)
        Dm[k, i] = k * C[k+1, i]
    end
    Dm
end
# Gram matrix of the columns of F[j] (piecewise polynomials) on [0, 1]
function gram(F::Vector{Matrix{Float64}}, br::Vector{Float64}, loc::Bool = false)
    n = size(F[1], 2); deg = size(F[1], 1) - 1
    G = zeros(n, n)
    if loc
        # local coefficients: Gauss-Legendre with deg + 2 nodes per piece (exact for the products), on the values (the
        # sum of the products of the monomial coefficients would cancel quadratically at high degree)
        z, w = glrule(deg + 2)
        for j in 1:length(F)
            L = br[j+1] - br[j]
            V = [evalpoly(L * z[q], view(F[j], :, a)) for q in eachindex(z), a in 1:n]
            G .+= transpose(V) * (V .* (L .* w))
        end
        return (G .+ transpose(G)) ./ 2
    end
    for j in 1:length(F), a in 1:n, b in 1:n, k in 0:deg, l in 0:deg
        G[a, b] += F[j][k+1, a] * F[j][l+1, b] * (br[j+1]^(k + l + 1) - br[j]^(k + l + 1)) / (k + l + 1)
    end
    G
end

"""
    Prep

The prepared design: the orthonormal basis g of span{d, d'} (coefficients g[j], (deg+1) x p, on piece j), the jump
points jl, the matrices A, B of U + 1/2 = ell' A ell and V - int J^2 = ell' B ell for ell = (J(1), J(jl...), z), and the
nonzero eigenvalues Dv (with eigenvectors L) of A: r = length(Dv) roots, rp positive and rm negative at real t
(Sylvester's law of inertia).
"""
struct Prep
    D::Determ
    K::Int; p::Int; nj::Int; N::Int; deg::Int
    br::Vector{Float64}
    g::Vector{Matrix{Float64}}
    jl::Vector{Float64}
    A::Matrix{Float64}; B::Matrix{Float64}
    L::Matrix{Float64}; Dv::Vector{Float64}
    r::Int; rp::Int; rm::Int
    G::Matrix{Float64}
    b::Vector{Float64}          # the coefficient of J(0) in U + 1/2 = ell' A ell - J(0)^2/2 + J(0) b' ell (z block: Pm d(0))
    off::Vector{Float64}        # the origins of the coefficients of the pieces (0, or br[j] for local coefficients)
    cbar::Float64               # GLS detrending at abar = 1 + cbar/n (`prep_gls`); NaN: least squares
    detrend::Symbol             # :onestep or :first (the statistic that detrends first: A is A_d, see `detrend_first`);
                                # :lm the denominator of the LM statistic (`prep_lm`, A a placeholder)
end
# (the constructor without the last field: the one-step regression)
Prep(D::Determ, K::Int, p::Int, nj::Int, N::Int, deg::Int, br, g, jl, A, B, L, Dv, r::Int, rp::Int, rm::Int, G, b, off, cbar::Float64) =
    Prep(D, K, p, nj, N, deg, br, g, jl, A, B, L, Dv, r, rp, rm, G, b, off, cbar, :onestep)

function prep(D::Determ)
    D.detrend in (:onestep, :first) || throw(ArgumentError("the statistic of the design must be :onestep or :first"))
    K = nreg(D); np = npc(D); deg = size(D.C[1], 1) - 1; br = copy(D.br)
    jl = Float64[]; Dl = Vector{Float64}[]
    for j in 1:np-1
        lam = br[j+1]
        dl = [evalpoly(lam - poff(D, j + 1), view(D.C[j+1], :, i)) - evalpoly(lam - poff(D, j), view(D.C[j], :, i)) for i in 1:K]
        if maximum(abs, dl; init = 0.0) > 1e-12
            push!(jl, lam); push!(Dl, dl)
        end
    end
    nj = length(jl)
    if K == 0
        g = [zeros(deg + 1, 0) for _ in 1:np]; p = 0
        G = zeros(0, 0); Pm = zeros(0, 0); Phid = zeros(0, 0); Phin = zeros(0, 0)
    else
        F = [hcat(D.C[j], dcoef(D.C[j])) for j in 1:np]
        Gf = gram(F, br, D.loc)
        E = eigen(Symmetric(Gf))
        keep = E.values .> 1e-11 * maximum(E.values)
        U = E.vectors[:, keep]; lam = E.values[keep]; p = length(lam)
        g = [F[j] * U * Diagonal(1 ./ sqrt.(lam)) for j in 1:np]
        Phi = U * Diagonal(sqrt.(lam))
        Phid = Phi[1:K, :]; Phin = Phi[K+1:2K, :]
        G = Gf[1:K, 1:K]
        Pm = Phid' / G                                    # p x K: m' G^{-1} = z' Pm
        # the basis reproduces d and d' (f = Phi g on the span)
        maximum(abs, Phid * Phid' - G) < 1e-10 * maximum(abs, G) || error("prep: basis does not reproduce G")
    end
    N = 1 + nj + p
    A = zeros(N, N); B = zeros(N, N); bv = zeros(N)
    A[1, 1] = 0.5
    if K > 0
        iz = (2 + nj):N
        bv[iz] .= Pm * dval(D, 0.0)
        d1 = dval(D, 1.0)
        v = -0.5 * (Pm * d1)
        A[iz, 1] .= v; A[1, iz] .= v
        # the one-step regression: z' Pm Phi_n z from m' G^{-1} int d' J; detrending first subtracts m' G^{-1} int d' Jtilde,
        # Jtilde = J - d' G^{-1} m, which leaves z' (Pm Phi_n)(Pm Phi_d) z (`detrend_first`)
        M = D.detrend == :first ? (Pm * Phin) * (Pm * Phid) : Pm * Phin
        A[iz, iz] .= 0.5 .* (M .+ M')
        for i in 1:nj
            v = 0.5 * (Pm * Dl[i])
            A[iz, 1+i] .= v; A[1+i, iz] .= v
        end
        B[iz, iz] .= -(Pm * Phid)
        B .= (B .+ B') ./ 2
    end
    Ae = eigen(Symmetric(A))
    keep = abs.(Ae.values) .> 1e-10 * maximum(abs, Ae.values)
    Dv = Ae.values[keep]; L = Ae.vectors[:, keep]
    Prep(D, K, p, nj, N, deg, br, g, jl, A, B, L, Dv, length(Dv), count(>(0), Dv), count(<(0), Dv), G, bv, [poff(D, j) for j in 1:np], NaN,
         D.detrend)
end

# ---------- GLS detrending (Elliott, Rothenberg and Stock) for a general design ----------
"""
    prep_gls(D, cbar) -> Prep

The design D with the data GLS-detrended at abar = 1 + cbar/n (cbar < 0): the first observation as it is, y_t - abar
y_{t-1} for t >= 2, the same for the regressors, beta^ by least squares, ytilde = y - beta^' d, and the t-ratio of the
regression of Delta ytilde_t on ytilde_{t-1} without deterministic terms.  The quasi-differenced regressors are O(1) at
t = 1 (value d(0)) and at each jump (value Delta_i) and O(1/n) elsewhere, so only the directions b with d(0)' b = 0 and
Delta_i' b = 0 are estimated at the rate that matters.  With Z a basis of that space, delta = Z' d (continuous, delta(0) =
0), psi = delta' - cbar delta, Psi = int psi psi', eta = int psi (dJ - cbar J dr):

    J_g = J - delta' Psi^{-1} eta,   U_g = (J_g(1)^2 - 1)/2,   V_g = int J_g^2,
    eta = psi(1) J(1) - sum_i varsigma_i J(lambda_i) - int (delta'' - cbar^2 delta) J dr,

varsigma_i the jump of delta' at lambda_i.  With g an orthonormal basis of span{delta, delta''}, delta = Phi_d g,
delta'' = Phi_2 g, ell = (J(1), J(lambda_i) where varsigma_i != 0, z = int g J): eta = H ell,
H = [psi(1), -varsigma_i, -(Phi_2 - cbar^2 Phi_d)], J_g(1) = a' ell with a = e_1 - H' Psi^{-1} delta(1), and

    U_g + 1/2 = ell' (a a'/2) ell,   V_g - int J^2 = ell' B_g ell,
    B_g = -(H' Psi^{-1} Phi_d S + S' Phi_d' Psi^{-1} H) + H' Psi^{-1} G_d Psi^{-1} H   (S the z block, G_d = int delta delta').

A_g has rank one: one positive root, no negative group (the inversion of the design without deterministic terms).  The
Prep carries cbar (NaN for least squares detrending).  No fixed or stationary start.
"""
function prep_gls(D::Determ, cbar::Real)
    cbar = Float64(cbar)
    (cbar < 0 && isfinite(cbar)) || throw(ArgumentError("GLS detrending needs cbar < 0 (a finite number)"))
    D.detrend == :first && throw(ArgumentError(FIRST_GLS_MSG))
    K = nreg(D); np = npc(D); deg = size(D.C[1], 1) - 1; br = copy(D.br)
    # the directions b with d(0)' b = 0 and Delta_i' b = 0
    rows = [permutedims(dval(D, 0.0))]
    for j in 1:np-1
        lam = br[j+1]
        dl = [evalpoly(lam - poff(D, j + 1), view(D.C[j+1], :, i)) - evalpoly(lam - poff(D, j), view(D.C[j], :, i)) for i in 1:K]
        maximum(abs, dl; init = 0.0) > 1e-12 && push!(rows, permutedims(dl))
    end
    Mr = reduce(vcat, rows)
    Z = nullspace(Mr; atol = 1e-10 * max(1.0, maximum(abs, Mr)))
    k = size(Z, 2)
    if k == 0
        A = fill(0.5, 1, 1); B = zeros(1, 1)
        return Prep(D, 0, 0, 0, 1, deg, br, [zeros(deg + 1, 0) for _ in 1:np], Float64[], A, B, ones(1, 1), [0.5], 1, 1, 0,
                    zeros(0, 0), zeros(1), [poff(D, j) for j in 1:np], cbar)
    end
    Cd = [D.C[j] * Z for j in 1:np]                         # delta on the pieces
    Cd1 = [dcoef(Cd[j]) for j in 1:np]; Cd2 = [dcoef(Cd1[j]) for j in 1:np]
    ev(Cs, j, r) = [evalpoly(r - poff(D, j), view(Cs[j], :, i)) for i in 1:k]
    # psi = delta' - cbar delta; Psi = int psi psi'
    Psi = gram([Cd1[j] .- cbar .* Cd[j] for j in 1:np], br, D.loc)
    Gd = gram(Cd, br, D.loc)
    psi1 = ev(Cd1, np, 1.0) .- cbar .* ev(Cd, np, 1.0); delta1 = ev(Cd, np, 1.0)
    # the jumps of delta' (slope breaks); delta itself is continuous
    jl = Float64[]; vs = Vector{Float64}[]
    for j in 1:np-1
        lam = br[j+1]
        dd = ev(Cd, j + 1, lam) .- ev(Cd, j, lam)
        maximum(abs, dd; init = 0.0) < 1e-9 * max(1.0, maximum(abs, ev(Cd, j, lam))) || error("prep_gls: delta not continuous at $lam")
        s1 = ev(Cd1, j + 1, lam) .- ev(Cd1, j, lam)
        maximum(abs, s1; init = 0.0) > 1e-12 && (push!(jl, lam); push!(vs, s1))
    end
    nj = length(jl)
    # orthonormal basis of span{delta, delta''}
    F = [hcat(Cd[j], Cd2[j]) for j in 1:np]
    Gf = gram(F, br, D.loc)
    E = eigen(Symmetric(Gf))
    keep = E.values .> 1e-11 * maximum(E.values)
    U = E.vectors[:, keep]; lv = E.values[keep]; p = length(lv)
    g = [F[j] * U * Diagonal(1 ./ sqrt.(lv)) for j in 1:np]
    Phi = U * Diagonal(sqrt.(lv)); Phid = Phi[1:k, :]; Phi2 = Phi[k+1:2k, :]
    maximum(abs, Phid * Phid' - Gd) < 1e-10 * maximum(abs, Gd) || error("prep_gls: basis does not reproduce G_d")
    N = 1 + nj + p
    H = zeros(k, N); H[:, 1] .= psi1
    for i in 1:nj; H[:, 1+i] .= -vs[i]; end
    iz = (2 + nj):N
    H[:, iz] .= -(Phi2 .- cbar^2 .* Phid)
    PiH = Psi \ H
    av = zeros(N); av[1] = 1; av .-= H' * (Psi \ delta1)
    A = av * av' ./ 2
    S = zeros(p, N); S[:, iz] .= Matrix(I, p, p)
    B = -(H' * (Psi \ (Phid * S)) .+ (S' * Phid') * PiH) .+ PiH' * Gd * PiH
    B = (B .+ B') ./ 2
    Ae = eigen(Symmetric(A))
    keepA = abs.(Ae.values) .> 1e-10 * maximum(abs, Ae.values)
    Dv = Ae.values[keepA]; L = Ae.vectors[:, keepA]
    Prep(D, k, p, nj, N, size(F[1], 1) - 1, br, g, jl, A, B, L, Dv, length(Dv), count(>(0), Dv), count(<(0), Dv), Gd, zeros(N),
         [poff(D, j) for j in 1:np], cbar)
end

"""the covariance Cov(J(r), J(u)) = e^{c(r+u)} min(r,u) e_0(2c min(r,u)) of the Ornstein-Uhlenbeck process started at zero;
far from the unit root, where e_0(2c min) overflows (-2c min(r,u) > 700), in the form (e^{c|r-u|} - e^{c(r+u)})/(-2c)"""
function oukernel(c::Float64, r::Float64, u::Float64)
    mn = min(r, u)
    -2c * mn > 700 && return (exp(c * abs(r - u)) - exp(c * (r + u))) / (-2c)
    exp(c * (r + u)) * mn * real(e0(complex(2c * mn)))
end
"""the same with the stationary start's further term e^{c(r+u)} v0"""
function oukernel(c::Float64, r::Float64, u::Float64, v0::Float64)
    mn = min(r, u)
    -2c * mn > 700 && return (exp(c * abs(r - u)) - exp(c * (r + u))) / (-2c) + exp(c * (r + u)) * v0
    exp(c * (r + u)) * (mn * real(e0(complex(2c * mn))) + v0)
end

"""
    eigtop_q(P, c; m = 48) -> (lambda1, lambda2)

The two largest eigenvalues of the operator of the quadratic form V = int J^2 + ell' B ell for any prepared design (with
GLS detrending V_g is not the norm of an orthogonal projection of J, so `eigtop` does not apply): Nystrom nodes as in
`eigtop`, the vector X = (J at the nodes, J(1), J(lambda_i)) with its covariance C (zero start), ell = Lambda X (z by the
quadrature of int g J), V = X' (W + Lambda' B Lambda) X, and the eigenvalues of C^{1/2} (W + Lambda' B Lambda) C^{1/2}.
"""
function eigtop_q(P::Prep, c::Real; m::Int = 48)
    z, w = glrule(m)
    rs = Float64[]; ws = Float64[]; pj = Int[]
    for j in 1:length(P.br)-1
        a = P.br[j]; b = P.br[j+1]
        for kk in 0:7, q in 1:m
            aa = a + (b - a) * kk / 8; push!(rs, aa + (b - a) / 8 * z[q]); push!(ws, (b - a) / 8 * w[q]); push!(pj, j)
        end
    end
    n = length(rs); c = Float64(c)
    pts = [rs; 1.0; P.jl]; M = length(pts)
    Kc(r, u) = oukernel(c, r, u)
    C = [Kc(pts[i], pts[j]) for i in 1:M, j in 1:M]
    Lam = zeros(P.N, M); Lam[1, n+1] = 1
    for i in 1:P.nj; Lam[1+i, n+1+i] = 1; end
    for i in 1:P.p, q in 1:n
        Lam[1+P.nj+i, q] = ws[q] * evalpoly(rs[q] - P.off[pj[q]], view(P.g[pj[q]], :, i))
    end
    Q = Lam' * P.B * Lam
    for q in 1:n; Q[q, q] += ws[q]; end
    Ce = eigen(Symmetric(C)); sq = Ce.vectors * Diagonal(sqrt.(max.(Ce.values, 0.0))) * Ce.vectors'
    ev = eigvals(Symmetric(sq * Symmetric(Q) * sq))
    ev[end], ev[end-1]
end

# =====================================================================================================================
# The covariance Sigma(theta) of ell under the Ornstein-Uhlenbeck law with parameter -theta
# =====================================================================================================================
"""e_k(z) = int_0^1 x^k e^{-zx} dx, k = 0..D.  For |z| < 2 the power series (terms below 1e-18 at the end); for |z| >=
max(D, 2) the upward recurrence e_k = (k e_{k-1} - e^{-z})/z (stable for |z| >= k); in between the upward recurrence for k
<= |z| and, for k > |z|, the downward recurrence e_{k-1} = (z e_k + e^{-z})/k (stable for k >= |z|) from e_D = e^{-z}
sum_m z^m D!/(D + m + 1)! (the series in 1 - x, whose terms decrease from the first for |z| < D + 2).  (The power series up to
|z| = D, used before the speed stage, sums terms of size e^{|z|}/sqrt(|z|) to a result of order 1: at D = 26 and |z| = 20
it lost eight digits, and with the Legendre trend of order 8 (D = 8) the distribution function was off by 1.6e-12.)"""
function ek!(e::AbstractVector{ComplexF64}, z::ComplexF64, D::Int)
    # (the reciprocals 1/m from a table and one inversion of z in place of the divisions: they took half the time of
    # `sigma!`, which every evaluation of the transform calls)
    a2 = abs2(z)
    if a2 < 4
        @inbounds for k in 0:D; e[k+1] = INVN[k+1]; end
        term = one(ComplexF64)
        for n in 1:300
            term *= -z * INVN[n]
            if n + D + 1 <= length(INVN)
                @inbounds for k in 0:D; e[k+1] = muladd(term, INVN[n+k+1], e[k+1]); end
            else
                for k in 0:D; e[k+1] += term / (n + k + 1); end
            end
            abs2(term) < 1e-36 && break
        end
    else
        em = exp(-z); iz = inv(z)
        K1 = a2 < D^2 ? floor(Int, sqrt(a2)) : D
        e[1] = (1 - em) * iz
        @inbounds for k in 1:K1
            e[k+1] = (k * e[k] - em) * iz
        end
        if K1 < D
            term = complex(1 / (D + 1)); acc = term
            for m in 1:5000
                term *= z / (D + m + 1)
                acc += term
                abs2(term) < 1e-36 * abs2(acc) && break
            end
            e[D+1] = em * acc
            @inbounds for k in D:-1:K1+2
                e[k] = (z * e[k+1] + em) / k
            end
        end
    end
    e
end
const INVN = [1 / m for m in 1:1024]

"""the Taylor coefficients at s of the polynomials in the columns of C (coefficients of r^k): a[k+1, i] =
g_i^{(k)}(s)/k!, by repeated synthetic division"""
function taylor!(a::Matrix{Float64}, C::Matrix{Float64}, s::Float64)
    copyto!(a, C)
    D = size(C, 1) - 1
    @inbounds for i in 1:size(C, 2), k in 0:D-1, j in D-1:-1:k
        a[j+1, i] += s * a[j+2, i]
    end
    a
end

struct SigWork
    e::Vector{ComplexF64}
    a::Matrix{Float64}
    F::Vector{ComplexF64}
    Rb::Matrix{ComplexF64}
    Ib::Matrix{ComplexF64}
    z::Vector{Float64}; w::Vector{Float64}
    h::Vector{ComplexF64}       # F(0): the functionals of ell applied to the path e^{-theta r} (the fixed start)
end
SigWork(P::Prep; nq::Int = 24) = (r = glrule(nq);
    SigWork(zeros(ComplexF64, P.deg + 1), zeros(P.deg + 1, max(P.p, 1)), zeros(ComplexF64, P.N),
            zeros(ComplexF64, P.N, length(P.br) - 1), zeros(ComplexF64, max(P.p, 1), length(P.br) - 1), r[1], r[2],
            zeros(ComplexF64, P.N)))

"""
    sigma!(S, P, theta, W)

Sigma(theta) = int_0^1 F(u) F(u)' du (bilinear, no conjugation), F the transfer functions of ell: e^{-theta (1 - u)} for
J(1), 1{u < lambda} e^{-theta (lambda - u)} for J(lambda), int_u^1 g(r) e^{-theta (r - u)} dr for z.  On piece j = [a, b]
and with y = b - u, F(u) = sum_k g^{(k)}(u)/k! y^{k+1} e_k(theta y) + e^{-theta y} F(b) (exact), and F(b) by a backward
recursion over the pieces.  The integral over y by Gauss-Legendre panels (nq nodes) graded from y = 0: first length
1/|theta|, doubling, at most 24/|theta| while |e^{-2 theta y}| > e^{-80} (the oscillation when Re theta is small).
"""
function sigma!(S::Matrix{ComplexF64}, P::Prep, θ::ComplexF64, W::SigWork)
    N = P.N; p = P.p; nj = P.nj; np = length(P.br) - 1; D = P.deg; br = P.br
    fill!(S, 0)
    e = W.e; a = W.a; F = W.F; Rb = W.Rb; Ib = W.Ib
    @inbounds for m in 1:np
        p == 0 && break
        am = br[m]; Lm = br[m+1] - am
        ek!(e, θ * Lm, D)
        taylor!(a, P.g[m], am - P.off[m])
        for i in 1:p
            acc = zero(ComplexF64); Lk = Lm
            for k in 0:D
                acc += a[k+1, i] * Lk * e[k+1]; Lk *= Lm
            end
            Ib[i, m] = acc
        end
    end
    @inbounds for j in np:-1:1
        b = br[j+1]
        Rb[1, j] = exp(-θ * (1 - b))
        for i in 1:nj
            lam = P.jl[i]
            Rb[1+i, j] = lam >= b - 1e-14 ? exp(-θ * (lam - b)) : zero(ComplexF64)
        end
        if j == np
            for i in 1:p; Rb[1+nj+i, j] = 0; end
        else
            el = exp(-θ * (br[j+2] - br[j+1]))
            for i in 1:p; Rb[1+nj+i, j] = Ib[i, j+1] + el * Rb[1+nj+i, j+1]; end
        end
    end
    at = abs(θ); rt = real(θ)
    h0 = at > 0 ? 1 / at : Inf
    @inbounds for j in 1:np
        b = br[j+1]; Lj = b - br[j]
        y0 = 0.0
        while y0 < Lj * (1 - 1e-14)
            if at * Lj <= 2
                y1 = Lj
            else
                wdt = max(y0, h0)
                2 * rt * y0 < 80 && (wdt = min(wdt, 24h0))
                y1 = min(y0 + wdt, Lj)
                (Lj - y1) < 0.5wdt && (y1 = Lj)
            end
            for q in eachindex(W.z)
                y = y0 + (y1 - y0) * W.z[q]; wq = (y1 - y0) * W.w[q]
                ey = exp(-θ * y)
                for i in 1:1+nj; F[i] = ey * Rb[i, j]; end
                if p > 0
                    ek!(e, θ * y, D)
                    taylor!(a, P.g[j], b - y - P.off[j])
                    for i in 1:p
                        acc = zero(ComplexF64); yk = y
                        for k in 0:D
                            acc += a[k+1, i] * yk * e[k+1]; yk *= y
                        end
                        F[1+nj+i] = acc + ey * Rb[1+nj+i, j]
                    end
                end
                for jj in 1:N
                    fj = wq * F[jj]
                    for ii in 1:jj
                        S[ii, jj] += fj * F[ii]
                    end
                end
            end
            y0 = y1
        end
    end
    @inbounds for jj in 1:N, ii in 1:jj-1
        S[jj, ii] = S[ii, jj]
    end
    # Var J(1) in closed form
    S[1, 1] = e0(2θ)
    # h = F(0): e^{-theta}, e^{-theta lambda_i}, int_0^1 g(r) e^{-theta r} dr = Ib(piece 1) + e^{-theta L_1} F_z(b_1)
    h = W.h
    h[1] = exp(-θ)
    @inbounds for i in 1:nj; h[1+i] = exp(-θ * P.jl[i]); end
    if p > 0
        el = exp(-θ * (br[2] - br[1]))
        @inbounds for i in 1:p; h[1+nj+i] = Ib[i, 1] + el * Rb[1+nj+i, 1]; end
    end
    S
end
sigma(P::Prep, θ::Number; nq::Int = 24) = sigma!(zeros(ComplexF64, P.N, P.N), P, complex(float(θ)), SigWork(P; nq))

e0(z::ComplexF64) = abs(z) < 1e-3 ? 1 - z / 2 + z * z / 6 - z^3 / 24 : (1 - exp(-z)) / z

"""1 - (c + theta) e_0(2 theta) = 1 - 2 gamma Var J(1), without the cancellation of theta - c for c > 0"""
function m011(c::Float64, t::ComplexF64, θ::ComplexF64)
    abs(θ) < 0.5 && return 1 - (c + θ) * e0(2θ)
    δ = c > 0 ? -2t / (θ + c) : θ - c
    (δ + (c + θ) * exp(-2θ)) / (2θ)
end

# =====================================================================================================================
# The moment generating function and its roots in s
# =====================================================================================================================
struct MWork
    S::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
    sw::SigWork
    Sx::Matrix{ComplexF64}      # the extended covariance of the stationary start (otherwise S)
    sd::Any                     # the static matrices of the design (`SData`), or nothing: the dense code
end
MWork(S, M, sw, Sx) = MWork(S, M, sw, Sx, nothing)
MWork(P::Prep; nq::Int = 24) = (S = zeros(ComplexF64, P.N, P.N);
    MWork(S, zeros(ComplexF64, P.N, P.N), SigWork(P; nq), S, sdata(P.A, P.B, P.L, P.Dv, zeros(P.N))))

"""the matrix I - 2 (s A + t B + gamma E11) Sigma into W.M (its (1,1) entry without cancellation); Sigma in W.S.
The (1,1) entry is m011 - 2 ((s A + t B) Sigma)_11, formed from these two terms alone: above the unit root it is of the
order e^{-2c} where the transform matters (the moments, the inversion of Gil-Pelaez), and the earlier form 1 - 2 ((s A +
t B) Sigma)_11 - 2 gamma Sigma_11 + (m011 - (1 - 2 gamma Sigma_11)) carried an absolute rounding error of 1e-16, which
put the moments off by 6e-4 relative at c = 15 and by 0.95 at c = 20.  (At s = 0 the product term is -2 t (B Sigma)_11,
zero where the first row of B vanishes, as for the polynomial trends and Perron's models: there m011 came through before
as well, and the roots and log D_V of the contour integral change by rounding only.)"""
function buildM!(W::MWork, P::Prep, c::Float64, s::ComplexF64, t::ComplexF64, θ::ComplexF64)
    S = W.S; M = W.M; N = P.N
    γ = (c + θ) / 2
    Q = s .* P.A .+ t .* P.B
    mul!(M, Q, S, -2, false)
    q11 = M[1, 1]
    for i in 1:N; M[i, i] += 1; end
    for j in 1:N; M[1, j] -= 2γ * S[1, j]; end
    M[1, 1] = m011(c, t, θ) + q11
    M
end

"""
    mgfparts(P, c, t[, W]) -> (theta, logDV, mu)

At one point t: theta = sqrt(c^2 - 2t) (principal), log D_V(t) = c + theta + log det M0 with M0 = I - 2 (t B + gamma
E11) Sigma (D_V = M_V(t)^{-2}, an entire function of t; the imaginary part is not reduced), and the r eigenvalues mu_k of
H = 2 Dv L' Sigma M0^{-1} L, so that det(I - 2 (sA + tB + gamma E11) Sigma) = det M0 prod_k (1 - s mu_k): the roots are
s_k = 1/mu_k.
"""
function mgfparts(P::Prep, c::Float64, t::ComplexF64, W::MWork = MWork(P))
    W.sd === nothing || return mgfparts_s(P, c, t, W, W.sd)::Tuple{ComplexF64,ComplexF64,Vector{ComplexF64}}
    θ = sqrt(complex(c * c - 2t))
    sigma!(W.S, P, θ, W.sw)
    M0 = buildM!(W, P, c, zero(ComplexF64), t, θ)
    F = lu(M0)
    ld = logdet(F)
    X = F \ complex.(P.L)
    H = 2 .* (Diagonal(P.Dv) * (transpose(P.L) * (W.S * X)))
    μ = eigvals(H)
    θ, c + θ + ld, μ
end
mgfparts(P::Prep, c::Real, t::Number, W::MWork = MWork(P)) = mgfparts(P, Float64(c), ComplexF64(t), W)

"""
    tparts(P, c, t) -> NamedTuple

Everything at one point t: theta, Sigma (N x N), A, B, the roots s_k = 1/mu_k (the first P.rp positive at real t when t
is real), log D_V(t) (D_V = M_V(t)^{-2}, imaginary part not reduced) and the matrix M0 = I - 2 (t B + gamma E11) Sigma.
Along a path, `rootpath` labels the roots by continuity and continues log P(t); then

    e^{s/2} M(s,t) = exp(logMtilde(roots, P.rp, logP, s)).
"""
function tparts(P::Prep, c::Real, t::Number)
    W = MWork(P)
    θ, ld, μ = mgfparts(P, Float64(c), ComplexF64(t), W)
    M0 = copy(buildM!(W, P, Float64(c), zero(ComplexF64), ComplexF64(t), θ))
    rts = 1 ./ μ
    if isreal(t) || abs(imag(t)) == 0
        rts = complex.([sort(real.(rts[real.(rts) .> 0])); sort(real.(rts[real.(rts) .< 0]))])
    end
    (theta = θ, Sigma = copy(W.S), A = P.A, B = P.B, roots = rts, logDV = ld, M0 = M0)
end

"""log of e^{s/2} M(s,t) = P(t) prod_{k <= rp} (s_k - s)^{-1/2} prod_{k > rp} (s - s_k)^{-1/2} (principal logarithms)"""
function logMtilde(rts::AbstractVector{ComplexF64}, rp::Int, logP::ComplexF64, s::Number)
    acc = logP
    for j in 1:rp; acc -= log(rts[j] - s) / 2; end
    for j in rp+1:length(rts); acc -= log(s - rts[j]) / 2; end
    acc
end

"""
    Dval(P, c, s, t) -> e^{c + theta} det(I - 2 (s A + t B + gamma E11) Sigma(theta))

so that M(s,t) = e^{-s/2} Dval^{-1/2}; in cases DU and DT this is D(s,t) of the paper (DFExact.Dfun), in case DK
e^{c} D(s,t).
"""
function Dval(P::Prep, c::Float64, s::ComplexF64, t::ComplexF64, W::MWork = MWork(P))
    θ = sqrt(complex(c * c - 2t))
    sigma!(W.S, P, θ, W.sw)
    exp(c + θ) * det(buildM!(W, P, c, s, t, θ))
end
Dval(P::Prep, c::Real, s::Number, t::Number, W::MWork = MWork(P)) = Dval(P, Float64(c), ComplexF64(s), ComplexF64(t), W)

"""log M(s,t) at real (s,t) on the tube (the determinant positive there)"""
function logmgf(P::Prep, c::Real, s::Real, t::Real)
    D = Dval(P, c, s, t)
    abs(imag(D)) < 1e-8 * abs(D) && real(D) > 0 || error("logmgf: (s,t) = ($s,$t) not on the tube (D = $D)")
    -s / 2 - log(real(D)) / 2
end

# =====================================================================================================================
# The initial observation: zero, fixed and stationary start
# =====================================================================================================================
"""
    fixed_shift(P) -> (yb, kappa_raw, residual)

The shift of the fixed start.  For every design there is a vector y with 2 A y = b (b the coefficient of J(0) in U), so
that U + 1/2 + kappa x^2 = (ell + x y)' A (ell + x y) with x = J(0) and kappa = 1/2 + y' A y, and kappa is exactly 0 or
1/2: 0 when the span of the regressors contains a continuous function phi with phi(0) != 0 whose derivative (piecewise)
lies in the span again (then y = -ell(phi)/phi(0), the functionals applied to the path phi; with an intercept phi = 1),
and 1/2 otherwise.  This is proved, and confirmed numerically in `scripts/check_kappa.jl`.  Here y = A^+ b / 2 from the
eigen-decomposition of A; `residual` is |b - 2 A y| / max(|b|, 1) (zero in exact arithmetic) and `kappa_raw` the computed
1/2 + y' A y before it is set to the exact value.
"""
function fixed_shift(P::Prep)
    cb = transpose(P.L) * P.b
    res = maximum(abs, P.b - P.L * cb; init = 0.0) / max(1.0, maximum(abs, P.b; init = 0.0))
    yb = P.L * (cb ./ P.Dv) ./ 2
    yb, 0.5 + dot(yb, P.A * yb), res
end

"""
    kappa_span(D) -> (kappa, alpha)

The value of kappa that the span criterion predicts, computed from the regressors alone (no Gaussian vector, no A): the
coefficient vectors alpha for which phi = alpha' d is continuous (alpha' Delta_i = 0 at every jump) and phi' = alpha' d'
lies in the span of d (alpha' R alpha = 0, R = int d' d'^T - int d' d^T G^{-1} int d d'^T the Gram matrix of the residuals
of d' after projection on span d) form a subspace S; kappa = 0 when d(0) is not orthogonal to S (some such phi has
phi(0) != 0) and 1/2 otherwise.  alpha is the element of S with the largest phi(0) = alpha' d(0) for |alpha| = 1 (empty
for kappa = 1/2).  A check of `fixed_shift`, not used by the engine.
"""
function kappa_span(D::Determ)
    K = nreg(D); np = npc(D)
    K == 0 && return 0.5, Float64[]
    F = [hcat(D.C[j], dcoef(D.C[j])) for j in 1:np]
    Gf = gram(F, D.br, D.loc)
    G = Gf[1:K, 1:K]; Cx = Gf[1:K, K+1:2K]; E = Gf[K+1:2K, K+1:2K]
    R = E - transpose(Cx) * (G \ Cx); R = (R + transpose(R)) ./ 2
    rows = [R]
    for j in 1:np-1
        lam = D.br[j+1]
        dl = [evalpoly(lam - poff(D, j + 1), view(D.C[j+1], :, i)) - evalpoly(lam - poff(D, j), view(D.C[j], :, i)) for i in 1:K]
        maximum(abs, dl; init = 0.0) > 1e-12 && push!(rows, permutedims(dl))
    end
    Mr = reduce(vcat, rows)
    S = nullspace(Mr; atol = 1e-9 * max(1.0, maximum(abs, Mr)))
    size(S, 2) == 0 && return 0.5, Float64[]
    v = transpose(S) * dval(D, 0.0)
    norm(v) > 1e-8 * max(1.0, norm(dval(D, 0.0))) || return 0.5, Float64[]
    0.0, S * v ./ norm(v)
end

"""
    shift_functionals(P, alpha) -> ell(phi)

The functionals ell = (J(1), J(lambda_i) at the jumps, z = int g J) applied to the deterministic path phi = alpha' d (for
the span criterion: with phi continuous and phi' in the span, 2 A ell(phi) + phi(0) b = 0).  int g phi by Gauss-Legendre
on the pieces (exact for the polynomial pieces).
"""
function shift_functionals(P::Prep, alpha::AbstractVector)
    D = P.D
    ph(r) = dot(alpha, dval(D, r))
    out = zeros(P.N)
    out[1] = ph(1.0)
    for i in 1:P.nj; out[1+i] = ph(P.jl[i]); end
    z, w = glrule(P.deg + 2)
    for j in 1:length(P.br)-1
        a = P.br[j]; L = P.br[j+1] - a
        for q in eachindex(z)
            r = a + L * z[q]
            phv = sum(alpha[k] * evalpoly(r - P.off[j], view(D.C[j], :, k)) for k in eachindex(alpha))
            for i in 1:P.p
                out[1+P.nj+i] += L * w[q] * phv * evalpoly(r - P.off[j], view(P.g[j], :, i))
            end
        end
    end
    out
end

"""
    Model

A design with a start for the Ornstein-Uhlenbeck process: :zero (J(0) = 0), :fixed (J(0) = g0) or :stationary (J(0) ~
N(0, -1/(2c)) independent of W, c < 0).  With J(0) = x the functionals gain two terms,

    U + 1/2 = ell' A ell - x^2/2 + x b' ell,   b = Pm d(0) on the z block,

and V is unchanged (from w = d(1) J(1) - d(0) J(0) - int d' J - sum_i Delta_i J(lambda_i)).

- :stationary: ell_ext = (ell, J(0)) is Gaussian with mean zero under the tilt, U + 1/2 = ell_ext' A_ext ell_ext with
  A_ext = [A b/2; b'/2 -1/2], B_ext = [B 0; 0 0], Sigma_ext = [Sigma + v h h', v h; v h', v], v = -1/(2c), and Girsanov's
  exponent gamma (J(1)^2 - J(0)^2 - 1) puts -gamma on the last diagonal entry:
  M = e^{-s/2} e^{-gamma} det(I - 2 (s A_ext + t B_ext + gamma (E_11 - E_NN)) Sigma_ext)^{-1/2}, a determinant of degree
  rank(A_ext) in s.  The fields N, A, B, L, Dv, r, rp, rm are the extended ones.
- :fixed: b lies in the range of A for every design; with 2 A yb = b and zeta = ell + x yb, U + 1/2 + x^2 kappa =
  zeta' A zeta, kappa = 1/2 + yb' A yb, so that Mt = e^{a s} M(s,t), a = 1/2 + g0^2 kappa, is a bounded function times
  algebraic factors on the paths: log Mt = log Mt_0 + g0^2 (cf(t) + E1(s,t)) with E1 bounded at infinity (see
  `fixparts`).  kappa is exactly 0 or 1/2 (`fixed_shift`): 0 when the span of d contains a continuous phi with phi(0) != 0
  and phi' in the span (the constant, for instance: a = 1/2), 1/2 otherwise (no deterministic terms: a = (1 + g0^2)/2).
"""
struct Model
    P::Prep
    start::Symbol
    g0::Float64
    N::Int
    A::Matrix{Float64}; B::Matrix{Float64}
    L::Matrix{Float64}; Dv::Vector{Float64}
    r::Int; rp::Int; rm::Int
    a::Float64
    yb::Vector{Float64}
    kappa::Float64
end

const FIRST_START_MSG = "the statistic that detrends first (detrend_first) is implemented for the zero start only; start = :fixed (gamma0 != 0) and start = :stationary are not supported with it"
const FIRST_GLS_MSG = "GLS detrending (gls = cbar) detrends first already: use the design itself, not detrend_first(D), with gls"
const FIRST_MOM_MSG = "the moments of the statistic that detrends first (detrend_first) are not implemented (not validated)"
const GLS_START_MSG = "GLS detrending (gls = cbar) is implemented for the zero start only: the derivation of the GLS limit assumes u_0 = 0, and start = :fixed (gamma0 != 0) and start = :stationary are not supported with it"
"""model(P, start = :zero, g0 = 0.0): the design P with its start (:zero, :fixed with J(0) = g0, :stationary)"""
function model(P::Prep, start::Symbol = :zero, g0::Real = 0.0)
    g0 = Float64(g0)
    isnan(P.cbar) || start == :zero || (start == :fixed && g0 == 0) || throw(ArgumentError(GLS_START_MSG))
    P.detrend == :first && !(start == :zero || (start == :fixed && g0 == 0)) && throw(ArgumentError(FIRST_START_MSG))
    if start == :zero || (start == :fixed && g0 == 0)
        return Model(P, :zero, 0.0, P.N, P.A, P.B, P.L, P.Dv, P.r, P.rp, P.rm, 0.5, zeros(P.N), 0.5)
    elseif start == :fixed
        # (b lies in the range of A and kappa is exactly 0 or 1/2 for every design: `fixed_shift`; the two checks below are
        # internal assertions, which only a numerical failure of the decomposition of A could trigger)
        yb, kraw, res = fixed_shift(P)
        res <= 1e-8 || error("model: internal error: b is not in the range of A to 1e-8 (residual $res), which contradicts the shift identity of the fixed start; the eigen-decomposition of A has failed for this design")
        kappa = abs(kraw) < abs(kraw - 0.5) ? 0.0 : 0.5
        abs(kraw - kappa) <= 1e-6 || error("model: internal error: kappa = $kraw is not 0 or 1/2 to 1e-6, which contradicts the shift identity of the fixed start; the eigen-decomposition of A has failed for this design")
        return Model(P, :fixed, g0, P.N, P.A, P.B, P.L, P.Dv, P.r, P.rp, P.rm, 0.5 + g0^2 * kappa, yb, kappa)
    elseif start == :stationary
        n = P.N; N = n + 1
        A = zeros(N, N); B = zeros(N, N)
        A[1:n, 1:n] .= P.A; A[1:n, N] .= P.b ./ 2; A[N, 1:n] .= P.b ./ 2; A[N, N] = -0.5
        B[1:n, 1:n] .= P.B
        Ae = eigen(Symmetric(A))
        keep = abs.(Ae.values) .> 1e-10 * maximum(abs, Ae.values)
        Dv = Ae.values[keep]; L = Ae.vectors[:, keep]
        return Model(P, :stationary, 0.0, N, A, B, L, Dv, length(Dv), count(>(0), Dv), count(<(0), Dv), 0.5, zeros(N), 0.5)
    end
    throw(ArgumentError("start must be :zero, :fixed or :stationary"))
end
model(D::Determ, args...) = model(prep(D), args...)

"""the data of the fixed start at one t: log Mt = log Mt_0 + g0^2 (cf + E1(s)), E1(s) = s [p1' (I - sH)^{-1} w1 +
p2' (I - sH)^{-1} w2] (r-vectors, H the r x r matrix whose eigenvalues are the 1/s_k)"""
struct FixT
    cf::ComplexF64
    H::Matrix{ComplexF64}
    w1::Vector{ComplexF64}; p1::Vector{ComplexF64}
    w2::Vector{ComplexF64}; p2::Vector{ComplexF64}
    mt::Vector{ComplexF64}      # the mean m_t of X = L'(ell + g0 y) under the tilt, per unit g0 (for `fixbound`)
end
FixT(cf, H, w1, p1, w2, p2) = FixT(cf, H, w1, p1, w2, p2, ComplexF64[])

"""the stationary covariance Sigma_ext = [S + v h h', v h; v h', v] (bilinear) into Sx"""
function statsigma!(Sx::Matrix{ComplexF64}, S::Matrix{ComplexF64}, h::Vector{ComplexF64}, v::Float64)
    n = size(S, 1); N = n + 1
    @inbounds for j in 1:n
        for i in 1:n; Sx[i, j] = S[i, j] + v * h[i] * h[j]; end
        Sx[N, j] = v * h[j]; Sx[j, N] = v * h[j]
    end
    Sx[N, N] = v
    Sx
end

"""I - 2 (s A_ext + t B_ext + gamma (E_11 - E_NN)) Sigma_ext into W.M (stationary start); Sigma_ext in W.Sx"""
function buildMx!(W, Mo::Model, c::Float64, s::ComplexF64, t::ComplexF64, θ::ComplexF64)
    S = W.Sx; M = W.M; N = Mo.N
    γ = (c + θ) / 2
    Q = s .* Mo.A .+ t .* Mo.B
    Q[1, 1] += γ; Q[N, N] -= γ
    mul!(M, Q, S, -2, false)
    for i in 1:N; M[i, i] += 1; end
    M
end

MWork(Mo::Model; nq::Int = 24) = (S = zeros(ComplexF64, Mo.P.N, Mo.P.N);
    MWork(S, zeros(ComplexF64, Mo.N, Mo.N), SigWork(Mo.P; nq), Mo.start == :stationary ? zeros(ComplexF64, Mo.N, Mo.N) : S,
          sdata(Mo.A, Mo.B, Mo.L, Mo.Dv, Mo.yb)))

# =====================================================================================================================
# Static arrays for the small matrices of the transform
# =====================================================================================================================
# The N x N matrix M0 = I - 2 (s A + t B + gamma E11) Sigma, its LU factors and solves, and the r x r matrix whose
# eigenvalues are the reciprocal roots, as static arrays of sizes known to the compiler (N <= STATIC_MAX), with an LU and an
# eigenvalue routine of their own: no BLAS or LAPACK call, no allocation but the results.  Larger designs keep the dense
# code.

@inline cabs1(z::ComplexF64) = abs(real(z)) + abs(imag(z))

"""LU factorization with partial pivoting (pivots by |.|^2) of a static complex matrix: the unit lower factor below the
diagonal and U on and above it in F, the row permutation p (A[p, :] = L U) and its parity"""
struct SLU{N,NN}
    F::SMatrix{N,N,ComplexF64,NN}
    p::SVector{N,Int}
    odd::Bool
end
function slu(A::SMatrix{N,N,ComplexF64,NN}) where {N,NN}
    M = MMatrix(A); p = MVector{N,Int}(ntuple(identity, Val(N))); odd = false
    @inbounds for k in 1:N
        piv = k; mx = abs2(M[k, k])
        for i in k+1:N
            a = abs2(M[i, k]); a > mx && (mx = a; piv = i)
        end
        if piv != k
            for j in 1:N; M[k, j], M[piv, j] = M[piv, j], M[k, j]; end
            p[k], p[piv] = p[piv], p[k]; odd = !odd
        end
        ip = inv(M[k, k])
        for i in k+1:N
            l = M[i, k] * ip; M[i, k] = l
            for j in k+1:N; M[i, j] = muladd(-l, M[k, j], M[i, j]); end
        end
    end
    SLU{N,NN}(SMatrix(M), SVector(p), odd)
end
"""log det A from its LU factors, as LinearAlgebra's logdet of an LU (log|det| plus the logarithm of the phase)"""
function slogdet(F::SLU{N}) where {N}
    d = 0.0; s = complex(F.odd ? -1.0 : 1.0)
    @inbounds for i in 1:N
        u = F.F[i, i]; a = abs(u); d += log(a); s *= u / a
    end
    d + log(s)
end
"""A \\ b"""
function ssolve(F::SLU{N}, b::SVector{N,T}) where {N,T}
    x = MVector{N,ComplexF64}(undef)
    @inbounds for i in 1:N; x[i] = b[F.p[i]]; end
    @inbounds for i in 2:N, j in 1:i-1; x[i] = muladd(-F.F[i, j], x[j], x[i]); end
    @inbounds for i in N:-1:1
        for j in i+1:N; x[i] = muladd(-F.F[i, j], x[j], x[i]); end
        x[i] /= F.F[i, i]
    end
    SVector(x)
end
"""A \\ B, column by column"""
ssolve(F::SLU{N}, B::SMatrix{N,R,T}) where {N,R,T} = hcat(ntuple(j -> ssolve(F, B[:, j]), Val(R))...)
"""transpose(A) \\ b (A = P' L U, so A' = U' L' P)"""
function ssolvet(F::SLU{N}, b::SVector{N,T}) where {N,T}
    y = MVector{N,ComplexF64}(b)
    @inbounds for i in 1:N
        for j in 1:i-1; y[i] = muladd(-F.F[j, i], y[j], y[i]); end
        y[i] /= F.F[i, i]
    end
    @inbounds for i in N-1:-1:1, j in i+1:N; y[i] = muladd(-F.F[j, i], y[j], y[i]); end
    x = MVector{N,ComplexF64}(undef)
    @inbounds for i in 1:N; x[F.p[i]] = y[i]; end
    SVector(x)
end

"""the Givens rotation (c real, s complex) with c x + s y = r, -conj(s) x + c y = 0"""
@inline function givens_c(x::ComplexF64, y::ComplexF64)
    y == 0 && return 1.0, zero(ComplexF64)
    ax = abs(x)
    ax == 0 && return 0.0, conj(y) / abs(y)
    nrm = hypot(ax, abs(y))
    ax / nrm, (x / ax) * conj(y) / nrm
end

"""
    seigvals(H) -> SVector

The eigenvalues of a small static complex matrix without LAPACK: the closed form for sizes 1 and 2 (the quadratic formula
in its stable form), otherwise balancing by powers of 2 (as LAPACK's zgebal, scaling only), the reduction to Hessenberg form
by Givens rotations and the single-shift QR algorithm with Wilkinson shifts, deflation at |h_{k,k-1}| <= eps (|h_kk| +
|h_{k-1,k-1}|) and exceptional shifts every 10 iterations (as zlahqr).  nothing when it does not converge in 30 n
iterations (the caller then takes LAPACK's eigvals).
"""
function seigvals(H::SMatrix{1,1,ComplexF64})
    SVector(H[1, 1])
end
function seigvals(H::SMatrix{2,2,ComplexF64})
    a = H[1, 1]; b = H[1, 2]; c = H[2, 1]; d = H[2, 2]
    tr = a + d; dt = a * d - b * c
    hd = (a - d) / 2
    disc = sqrt(muladd(hd, hd, b * c))
    # the larger root by the sum of terms of the same direction, the other from the product
    m = tr / 2; q = real(conj(m) * disc) >= 0 ? m + disc : m - disc
    q == 0 && return SVector(m, m)
    SVector(q, dt / q)
end
function seigvals(H0::SMatrix{R,R,ComplexF64}) where {R}
    H = MMatrix(H0); ep = eps(Float64)
    # balancing (EISPACK balanc, radix 2, scaling only)
    @inbounds for _ in 1:20
        noconv = false
        for i in 1:R
            c = 0.0; r = 0.0
            for j in 1:R
                j == i && continue
                c += cabs1(H[j, i]); r += cabs1(H[i, j])
            end
            (c == 0 || r == 0) && continue
            g = r / 2; f = 1.0; s = c + r
            while c < g; f *= 2; c *= 4; end
            g = r * 2
            while c >= g; f /= 2; c /= 4; end
            if (c + r) / f < 0.95 * s
                noconv = true
                for j in 1:R; H[i, j] /= f; H[j, i] *= f; end
            end
        end
        noconv || break
    end
    # Hessenberg form by Givens rotations (similarity)
    @inbounds for k in 1:R-2, i in R:-1:k+2
        cg, sg = givens_c(H[i-1, k], H[i, k])
        sg == 0 && cg == 1 && continue
        for j in k:R
            x = H[i-1, j]; y = H[i, j]
            H[i-1, j] = cg * x + sg * y; H[i, j] = -conj(sg) * x + cg * y
        end
        for j in 1:R
            x = H[j, i-1]; y = H[j, i]
            H[j, i-1] = cg * x + conj(sg) * y; H[j, i] = -sg * x + cg * y
        end
        H[i, k] = 0
    end
    ev = MVector{R,ComplexF64}(undef)
    ihi = R; its = 0; total = 0
    @inbounds while ihi >= 1
        if ihi == 1
            ev[1] = H[1, 1]; break
        end
        l = 1
        for k in ihi:-1:2
            if cabs1(H[k, k-1]) <= ep * (cabs1(H[k-1, k-1]) + cabs1(H[k, k]))
                H[k, k-1] = 0; l = k; break
            end
        end
        if l == ihi
            ev[ihi] = H[ihi, ihi]; ihi -= 1; its = 0; continue
        end
        if l == ihi - 1
            # a 2 x 2 block: its eigenvalues in closed form
            e2 = seigvals(SMatrix{2,2,ComplexF64}(H[l, l], H[ihi, l], H[l, ihi], H[ihi, ihi]))
            ev[l] = e2[1]; ev[ihi] = e2[2]; ihi -= 2; its = 0; continue
        end
        its += 1; total += 1
        total > 30R && return nothing
        if its % 10 == 0
            μ = H[ihi, ihi] + 0.75 * abs(real(H[ihi, ihi-1]))
        else
            a = H[ihi-1, ihi-1]; b = H[ihi-1, ihi]; c = H[ihi, ihi-1]; d = H[ihi, ihi]
            hd = (a - d) / 2; disc = sqrt(muladd(hd, hd, b * c))
            m1 = (a + d) / 2 + disc; m2 = (a + d) / 2 - disc
            μ = abs2(m1 - d) <= abs2(m2 - d) ? m1 : m2
        end
        # one implicit single-shift QR step on the block l..ihi
        x = H[l, l] - μ; y = H[l+1, l]
        for k in l:ihi-1
            cg, sg = givens_c(x, y)
            for j in max(l, k - 1):ihi
                u = H[k, j]; v = H[k+1, j]
                H[k, j] = cg * u + sg * v; H[k+1, j] = -conj(sg) * u + cg * v
            end
            for j in l:min(k + 2, ihi)
                u = H[j, k]; v = H[j, k+1]
                H[j, k] = cg * u + conj(sg) * v; H[j, k+1] = -sg * u + cg * v
            end
            if k < ihi - 1
                x = H[k+1, k]; y = H[k+2, k]
            end
        end
        for k in l+2:ihi; H[k, k-2] = 0; end
    end
    SVector(ev)
end

"""the largest N (the size of Sigma, or of the extended covariance of the stationary start) of the static code"""
const STATIC_MAX = 12
"""whether `MWork` uses the static code for designs with N <= STATIC_MAX (false: the dense code with LAPACK)"""
const USE_STATIC = Ref(true)
"""the matrices of a design (or of a model with its start) as static arrays: A, B (N x N), L (N x r), Dv, yb"""
struct SData{N,R,NN,NR}
    A::SMatrix{N,N,Float64,NN}; B::SMatrix{N,N,Float64,NN}
    L::SMatrix{N,R,Float64,NR}; Dv::SVector{R,Float64}; yb::SVector{N,Float64}
end
function sdata(A::Matrix{Float64}, B::Matrix{Float64}, L::Matrix{Float64}, Dv::Vector{Float64}, yb::Vector{Float64})
    N = size(A, 1); R = size(L, 2)
    (USE_STATIC[] && 1 <= N <= STATIC_MAX && R >= 1) || return nothing
    SData{N,R,N * N,N * R}(SMatrix{N,N,Float64,N * N}(A), SMatrix{N,N,Float64,N * N}(B), SMatrix{N,R,Float64,N * R}(L),
                           SVector{R,Float64}(Dv), SVector{N,Float64}(yb))
end
"""the eigenvalues of H by `seigvals`, LAPACK's where it does not converge"""
@inline function eigs_s(H::SMatrix{R,R,ComplexF64}) where {R}
    e = seigvals(H)
    e === nothing ? eigvals(Matrix(H)) : Vector{ComplexF64}(e)
end
"""M0 = I - 2 (s A + t B + gamma E11) S with the (1,1) entry without cancellation (`buildM!`)"""
@inline function buildM_s(sd::SData{N}, S::SMatrix{N,N,ComplexF64}, c::Float64, s::ComplexF64, t::ComplexF64, θ::ComplexF64) where {N}
    γ = (c + θ) / 2
    M = MMatrix{N,N,ComplexF64}(-2 * ((s * sd.A + t * sd.B) * S))
    @inbounds q11 = M[1, 1]
    @inbounds for i in 1:N; M[i, i] += 1; end
    @inbounds for j in 1:N; M[1, j] -= 2γ * S[1, j]; end
    @inbounds M[1, 1] = m011(c, t, θ) + q11
    SMatrix(M)
end
"""the stationary start: I - 2 (s A + t B + gamma (E11 - ENN)) Sx (`buildMx!`)"""
@inline function buildMx_s(sd::SData{N}, Sx::SMatrix{N,N,ComplexF64}, c::Float64, s::ComplexF64, t::ComplexF64, θ::ComplexF64) where {N}
    γ = (c + θ) / 2
    Q = MMatrix{N,N,ComplexF64}(s * sd.A + t * sd.B)
    @inbounds Q[1, 1] += γ; @inbounds Q[N, N] -= γ
    M = MMatrix{N,N,ComplexF64}(-2 * (SMatrix(Q) * Sx))
    @inbounds for i in 1:N; M[i, i] += 1; end
    SMatrix(M)
end
"""`mgfparts` of the zero start with static arrays"""
function mgfparts_s(P::Prep, c::Float64, t::ComplexF64, W::MWork, sd::SData{N,R}) where {N,R}
    θ = sqrt(complex(c * c - 2t))
    sigma!(W.S, P, θ, W.sw)
    S = SMatrix{N,N,ComplexF64}(W.S)
    F = slu(buildM_s(sd, S, c, zero(ComplexF64), t, θ))
    X = ssolve(F, SMatrix{N,R,ComplexF64}(sd.L))
    H = 2 .* (sd.Dv .* (transpose(sd.L) * (S * X)))
    θ, c + θ + slogdet(F), eigs_s(H)
end
"""`mgfparts` of the stationary start with static arrays"""
function mgfparts_stat_s(Mo::Model, c::Float64, t::ComplexF64, W::MWork, sd::SData{N,R}) where {N,R}
    θ = sqrt(complex(c * c - 2t))
    sigma!(W.S, Mo.P, θ, W.sw)
    statsigma!(W.Sx, W.S, W.sw.h, -1 / (2c))
    Sx = SMatrix{N,N,ComplexF64}(W.Sx)
    F = slu(buildMx_s(sd, Sx, c, zero(ComplexF64), t, θ))
    X = ssolve(F, SMatrix{N,R,ComplexF64}(sd.L))
    H = 2 .* (sd.Dv .* (transpose(sd.L) * (Sx * X)))
    θ, c + θ + slogdet(F), eigs_s(H), nothing
end
@inline tdot(x::SVector{N}, y::SVector{N}) where {N} = sum(x .* y)
"""`fixparts` with static arrays"""
function fixparts_s(Mo::Model, c::Float64, t::ComplexF64, W::MWork, sd::SData{N,R}) where {N,R}
    P = Mo.P
    θ = sqrt(complex(c * c - 2t)); γ = (c + θ) / 2
    sigma!(W.S, P, θ, W.sw)
    S = SMatrix{N,N,ComplexF64}(W.S); h = SVector{N,ComplexF64}(W.sw.h)
    F = slu(buildM_s(sd, S, c, zero(ComplexF64), t, θ))
    L = sd.L; Lt = transpose(L); Dv = sd.Dv; yb = sd.yb
    X = ssolve(F, SMatrix{N,R,ComplexF64}(L))
    H = 2 .* (Dv .* (Lt * (S * X)))
    Qm = MMatrix{N,N,ComplexF64}(t * sd.B); @inbounds Qm[1, 1] += γ; Q0 = SMatrix(Qm)
    mt = h + yb
    v0 = Q0 * (h - yb); qt = -2 * (Q0 * yb)
    a0 = ssolve(F, v0); aq = ssolve(F, qt)
    Sq = S * qt
    cf = -γ + tdot(yb, Q0 * yb) + tdot(mt, a0) + tdot(qt, S * aq) / 2
    p1 = Lt * ssolvet(F, mt)
    w1 = 2 .* Dv .* (Lt * (S * a0)) .+ Dv .* (Lt * mt)
    p2 = Lt * ssolvet(F, Sq)
    w2 = Dv .* (Lt * (S * aq))
    mm = Lt * (ssolvet(F, h) + yb)
    θ, c + θ + slogdet(F), eigs_s(H), FixT(cf, Matrix(H), Vector(w1), Vector(p1), Vector(w2), Vector(p2), Vector(mm))
end
"""`logmgf_spart` with static arrays"""
function logmgf_spart_s(Mo::Model, c::Float64, s::ComplexF64, t::ComplexF64, θ::ComplexF64, fx, W::MWork, Bf::Matrix{ComplexF64},
                        sd::SData{N}) where {N}
    if Mo.start == :stationary
        ld = slogdet(slu(buildMx_s(sd, SMatrix{N,N,ComplexF64}(W.Sx), c, s, t, θ)))
        return -s / 2 - (c + θ) / 2, ld
    end
    extra = Mo.start == :fixed ? Mo.g0^2 * (fx.cf + fixE1(fx, s, Bf) - Mo.kappa * s) : zero(ComplexF64)
    ld = slogdet(slu(buildM_s(sd, SMatrix{N,N,ComplexF64}(W.S), c, s, t, θ)))
    -s / 2 - (c + θ) / 2 + extra, ld
end

"""
    fixparts(Mo, c, t, W) -> (theta, logDV, mu, FixT)

The zero-start parts at t (as `mgfparts`) and the data of the fixed start.  With Q0 = t B + gamma E11, M0 = I - 2 Q0
Sigma, mt = h + yb, v0 = Q0 (h - yb), qt = -2 Q0 yb and K(s) = I - 2 (s A + Q0) Sigma, the Gaussian integral over zeta ~
N(x mt, Sigma) gives

    log Mt = log Mt_0 + x^2 [ -gamma + yb' Q0 yb + mt' K^{-1} (v0 + s A mt) + qt' Sigma K^{-1} qt / 2 ],

and Woodbury's identity with A = L Dv L', K^{-1} = M0^{-1} + 2s M0^{-1} L (I - sH)^{-1} Dv L' Sigma M0^{-1},
H = 2 Dv L' Sigma M0^{-1} L, turns the bracket into cf + s [p1' (I - sH)^{-1} w1 + p2' (I - sH)^{-1} w2], bounded as
|s| -> infinity: p1 = L' M0^{-T} mt, w1 = 2 Dv L' Sigma M0^{-1} v0 + Dv L' mt, p2 = L' M0^{-T} Sigma qt,
w2 = Dv L' Sigma M0^{-1} qt, cf = -gamma + yb' Q0 yb + mt' M0^{-1} v0 + qt' Sigma M0^{-1} qt / 2.  Then
Qf(s,t) = cf + E1(s) - kappa s is the exponent of the brief, log M_{g0} = log M_0 + g0^2 Qf.
"""
function fixparts(Mo::Model, c::Float64, t::ComplexF64, W::MWork)
    W.sd === nothing || return fixparts_s(Mo, c, t, W, W.sd)::Tuple{ComplexF64,ComplexF64,Vector{ComplexF64},FixT}
    P = Mo.P
    θ = sqrt(complex(c * c - 2t)); γ = (c + θ) / 2
    sigma!(W.S, P, θ, W.sw)
    M0 = buildM!(W, P, c, zero(ComplexF64), t, θ)
    F = lu(M0)
    ld = logdet(F)
    S = W.S; h = W.sw.h; L = P.L; Lt = transpose(P.L)
    X = F \ complex.(L)
    H = 2 .* (Diagonal(P.Dv) * (Lt * (S * X)))
    μ = eigvals(H)
    yb = Mo.yb
    Q0 = t .* P.B; Q0[1, 1] += γ
    mt = h .+ yb
    v0 = Q0 * (h .- yb); qt = -2 .* (Q0 * yb)
    a0 = F \ v0; aq = F \ qt
    Sq = S * qt
    cf = -γ + transpose(yb) * (Q0 * yb) + transpose(mt) * a0 + transpose(qt) * (S * aq) / 2
    p1 = Lt * (transpose(F) \ mt)
    w1 = 2 .* P.Dv .* (Lt * (S * a0)) .+ P.Dv .* (Lt * mt)
    p2 = Lt * (transpose(F) \ Sq)
    w2 = P.Dv .* (Lt * (S * aq))
    mm = Lt * ((transpose(F) \ h) .+ yb)
    θ, c + θ + ld, μ, FixT(cf, H, w1, p1, w2, p2, mm)
end

"""work space of `fixE1`: the r x (r + 2) augmented matrix"""
fixbuf(r::Int) = zeros(ComplexF64, r, r + 2)
"""E1(s) = s [p1' (I - sH)^{-1} w1 + p2' (I - sH)^{-1} w2] by Gaussian elimination with partial pivoting on (I - sH) x =
s w (|s| <= 1) or (I/s - H) x = w (|s| > 1), so that nothing grows at large |s|"""
function fixE1(fx::FixT, s::ComplexF64, Bf::Matrix{ComplexF64})
    r = length(fx.p1)
    r == 0 && return zero(ComplexF64)
    big = abs2(s) > 1
    d = big ? 1 / s : one(ComplexF64); f = big ? one(ComplexF64) : s
    @inbounds for j in 1:r
        for i in 1:r
            Bf[i, j] = (big ? -fx.H[i, j] : -s * fx.H[i, j])
        end
        Bf[j, j] += d
        Bf[j, r+1] = f * fx.w1[j]; Bf[j, r+2] = f * fx.w2[j]
    end
    # (pivots by |.|^2 and multiplications by the inverse pivot, kept in the diagonal: the hypot of abs and the guarded
    # complex division took most of the time of an evaluation, which `place` and `nsplit` make thousands of times per node)
    @inbounds for k in 1:r
        piv = k; mx = abs2(Bf[k, k])
        for i in k+1:r
            a2 = abs2(Bf[i, k]); a2 > mx && (mx = a2; piv = i)
        end
        if piv != k
            for j in k:r+2; Bf[k, j], Bf[piv, j] = Bf[piv, j], Bf[k, j]; end
        end
        ip = inv(Bf[k, k]); Bf[k, k] = ip
        for i in k+1:r
            l = Bf[i, k] * ip
            for j in k+1:r+2; Bf[i, j] = muladd(-l, Bf[k, j], Bf[i, j]); end
        end
    end
    acc = zero(ComplexF64)
    @inbounds for col in (r + 1, r + 2)
        pv = col == r + 1 ? fx.p1 : fx.p2
        for k in r:-1:1
            x = Bf[k, col]
            for j in k+1:r; x = muladd(-Bf[k, j], Bf[j, col], x); end
            Bf[k, col] = x * Bf[k, k]
            acc = muladd(pv[k], Bf[k, col], acc)
        end
    end
    acc
end

"""
    Qf(Mo, c, s, t) -> the fixed-start exponent: log M_{g0}(s,t) = log M_0(s,t) + g0^2 Qf(s,t)
"""
function Qf(Mo::Model, c::Real, s::Number, t::Number)
    W = MWork(Mo)
    _, _, _, fx = fixparts(Mo, Float64(c), ComplexF64(t), W)
    fx.cf + fixE1(fx, ComplexF64(s), fixbuf(Mo.r)) - Mo.kappa * s
end

"""the brief's formula for Qf, directly in N x N matrices (a check): -s/2 - gamma + h'Qh + s b'h + (1/2) (2Qh + sb)'
Sigma (I - 2 Q Sigma)^{-1} (2Qh + sb), Q = sA + tB + gamma E11"""
function Qf_direct(P::Prep, c::Real, s::Number, t::Number)
    c = Float64(c); s = ComplexF64(s); t = ComplexF64(t)
    W = MWork(P)
    θ = sqrt(complex(c * c - 2t)); γ = (c + θ) / 2
    sigma!(W.S, P, θ, W.sw)
    S = W.S; h = copy(W.sw.h)
    Q = s .* P.A .+ t .* P.B; Q[1, 1] += γ
    K = copy(buildM!(W, P, c, s, t, θ))
    v = 2 .* (Q * h) .+ s .* P.b
    -s / 2 - γ + transpose(h) * (Q * h) + s * dot(P.b, h) + transpose(v) * (S * (K \ v)) / 2
end

"""
    mgfparts(Mo::Model, c, t, W) -> (theta, logDV, mu, fx)

As mgfparts(P, ...) for the model's start: the stationary start with the extended matrices (D_V = e^{c + theta} det M0,
M0 = I - 2 (t B_ext + gamma (E_11 - E_NN)) Sigma_ext; the r eigenvalues of H = 2 Dv L' Sigma_ext M0^{-1} L), the fixed
start by `fixparts` (fx its data; nothing otherwise).
"""
function mgfparts(Mo::Model, c::Float64, t::ComplexF64, W::MWork)
    Mo.start == :zero && return (mgfparts(Mo.P, c, t, W)..., nothing)
    Mo.start == :fixed && return fixparts(Mo, c, t, W)
    W.sd === nothing || return mgfparts_stat_s(Mo, c, t, W, W.sd)::Tuple{ComplexF64,ComplexF64,Vector{ComplexF64},Nothing}
    P = Mo.P
    θ = sqrt(complex(c * c - 2t))
    sigma!(W.S, P, θ, W.sw)
    statsigma!(W.Sx, W.S, W.sw.h, -1 / (2c))
    M0 = buildMx!(W, Mo, c, zero(ComplexF64), t, θ)
    F = lu(M0)
    ld = logdet(F)
    X = F \ complex.(Mo.L)
    H = 2 .* (Diagonal(Mo.Dv) * (transpose(Mo.L) * (W.Sx * X)))
    θ, c + θ + ld, eigvals(H), nothing
end

"""Dval(Mo, c, s, t): e^{c + theta} det(...) of the model's start, so that M(s,t) = e^{-s/2} Dval^{-1/2} (zero, stationary)
and M_{g0} = e^{-s/2} Dval^{-1/2} e^{g0^2 Qf} (fixed, Dval of the zero start)"""
function Dval(Mo::Model, c::Real, s::Number, t::Number)
    Mo.start == :stationary || return Dval(Mo.P, c, s, t)
    c = Float64(c); s = ComplexF64(s); t = ComplexF64(t)
    W = MWork(Mo)
    θ = sqrt(complex(c * c - 2t))
    sigma!(W.S, Mo.P, θ, W.sw)
    statsigma!(W.Sx, W.S, W.sw.h, -1 / (2c))
    exp(c + θ) * det(buildMx!(W, Mo, c, s, t, θ))
end

"""
    eigtop(P, c; m = 48) -> (lambda1, lambda2)

The two largest eigenvalues of the covariance operator of the residual Jt = J - d' G^{-1} m on L2[0,1] (V = sum_j
lambda_j Z_j^2), by the Nystrom method: Gauss-Legendre with m nodes on each of 8 equal panels of every piece, the kernel
Cov(J(r), J(u)) = e^{c(r+u)} min(r,u) e_0(2c min(r,u)) projected on the complement of span d in the discrete inner
product.  The kink of the kernel on the diagonal limits the accuracy to about 1e-6 relative at the default m.  With
stat = true the stationary start, whose kernel has the further term e^{c(r+u)} (-1/(2c)).
"""
function eigtop(P::Prep, c::Real; m::Int = 48, stat::Bool = false)
    z, w = glrule(m)
    rs = Float64[]; ws = Float64[]
    for j in 1:length(P.br)-1
        a = P.br[j]; b = P.br[j+1]
        for k in 0:7, q in 1:m
            aa = a + (b - a) * k / 8; push!(rs, aa + (b - a) / 8 * z[q]); push!(ws, (b - a) / 8 * w[q])
        end
    end
    n = length(rs); c = Float64(c)
    v0 = stat ? -1 / (2c) : 0.0
    Kc(r, u) = oukernel(c, r, u, v0)
    sw = sqrt.(ws)
    Kw = [sw[i] * Kc(rs[i], rs[k]) * sw[k] for i in 1:n, k in 1:n]
    if P.K > 0
        X = reduce(vcat, [permutedims(dval(P.D, r)) for r in rs]) .* sw
        Q = Matrix(qr(X).Q)
        Kw = Kw - Q * (Q' * Kw); Kw = Kw - (Kw * Q) * Q'
    end
    ev = eigvals(Symmetric((Kw + Kw') / 2))
    ev[end], ev[end-1]
end

"""
    blas1(f)

f() with one BLAS thread, the number restored afterwards (nested and concurrent calls share one count); `tstar` and
`prep_cached` run in it, and with the dense code of the transform (`dense_la`) also `cpath` and the loops of
`dist_moments` and `coef_cdf_gp`.  The eigenvalue problems of the estimate of t_* (of size 128 and more, LAPACK's by
nature) and the factorizations of the dense code ran on OpenBLAS's threads, which then spin for some 0.1 s waiting for work
and took the cores from the threads of the evaluation that follows: the first `contour_eval` after a new path took twice
its time (40 ms for 19 ms with the linear trend at c = 0).  The problems are too small to gain from BLAS threads.  The
static code (N <= STATIC_MAX) calls no BLAS, and its paths and moments take the same time with or without one thread.
"""
function blas1(f)
    lock(BLAS_LOCK) do
        if BLAS_DEPTH[] == 0
            BLAS_SAVED[] = BLAS.get_num_threads()
            BLAS_SAVED[] > 1 && BLAS.set_num_threads(1)
        end
        BLAS_DEPTH[] += 1
    end
    try
        f()
    finally
        lock(BLAS_LOCK) do
            BLAS_DEPTH[] -= 1
            BLAS_DEPTH[] == 0 && BLAS_SAVED[] > 1 && BLAS.set_num_threads(BLAS_SAVED[])
        end
    end
end
const BLAS_LOCK = ReentrantLock(); const BLAS_DEPTH = Ref(0); const BLAS_SAVED = Ref(1)
"""f() with one BLAS thread when on (`blas1`), as it is otherwise"""
blas1f(f, on::Bool) = on ? blas1(f) : f()
"""whether the transform of the model takes the dense code (LAPACK and BLAS on its small matrices) rather than the static
one: the path and the loops of `dist_moments` and `coef_cdf_gp` then run with one BLAS thread (with the static code they
call no BLAS, and the times are the same with or without; t_* and the prepared design always run with one)"""
dense_la(Mo::Model) = !(USE_STATIC[] && Mo.N <= STATIC_MAX)

"""the Gauss-Legendre nodes per eighth of a piece of the Nystrom estimate of t_* in `tstar` (48 before the speed stage, 8
pieces of 48 = 384 points per piece and an eigenvalue problem of that size, 7 ms for the linear trend and 22 ms for model C;
the estimate only has to lie within 2% of t_*, which bisection on D_V then finds to 1e-13; far from the unit root, where
it does not, the scan from the branch point finds t_*).  GLS detrending and the LM statistic (`eigtop_q`) keep 48: their
t_* lies below the branch point far from the unit root (4818 at c = -100 in case DT, cbar = -13.5; 139 for the LM
statistic with a linear trend at c = -20), where the scan does not apply."""
const NYSTROM_M = Ref(16)
"""
    tstar(P, c) -> (tstar, multiplicity)

t_* = sup{t: E e^{tV} < inf} = 1/(2 lambda_1), the first zero of D_V(t) = M_V(t)^{-2} on the positive real axis.  The
Nystrom estimate of eigtop locates it; D_V is then sampled on 81 points of +-2% around the estimate: at the first sign
change bisection, and where there is none (a double zero: lambda_1 = lambda_2, as in model C at lambda = 1/2, where V
is the sum of two independent copies) the minimum of D_V by golden section.  (Scanning D_V for a sign change alone
misses a double zero.)  Far from the unit root, where the estimate is off by more than 2%, D_V is scanned from the branch
point c^2/2 (see below).
"""
tstar(P::Prep, c::Real) = tstar(model(P), c)
tstar(Mo::Model, c::Real) = blas1(() -> tstar_(Mo, c))
function tstar_(Mo::Model, c::Real)
    c = Float64(c)
    Mo.start == :stationary && c >= 0 && throw(ArgumentError("the stationary start needs c < 0"))
    W = MWork(Mo)
    f(t) = real(exp(mgfparts(Mo, c, complex(t), W)[2]))
    # (eigtop projects J orthogonally off the span of the design; the V of GLS detrending and of the LM statistic is no such
    # norm, and eigtop_q takes its quadratic form: with eigtop the LM statistic's t_* came out up to 2.3 times too large,
    # above the unit root beyond twice the edge, where the path then failed)
    l1, l2 = isnan(Mo.P.cbar) && Mo.P.detrend != :lm ? eigtop(Mo.P, c; m = NYSTROM_M[], stat = Mo.start == :stationary) : eigtop_q(Mo.P, c)
    te = 1 / (2l1)
    grid = range(0.98te, 1.02te; length = 81)
    vals = f.(grid)
    function bisect(a, b)
        for _ in 1:100
            mid = (a + b) / 2
            f(mid) > 0 ? (a = mid) : (b = mid)
            b - a < 1e-13 * b && break
        end
        (a + b) / 2
    end
    k = findfirst(<=(0), vals)
    k !== nothing && k > 1 && return bisect(grid[k-1], grid[k]), abs(l1 - l2) < 1e-4 * l1 ? 2 : 1
    # Far from the unit root the estimate is off by more than 2% (the kernel of the covariance has the width 1/|c|, which the
    # Nystrom rule does not resolve: with 48 nodes per eighth t_* came out at 19624 for 20005 in case DK at c = -200, with 16
    # at 4716 for 5005 at c = -100; the distribution functions do not change, the path takes tau = 1 there).  t_* then lies
    # beyond the branch point c^2/2 (theta = i omega imaginary, t = (c^2 + omega^2)/2, omega of order pi times the number of
    # regressors), and D_V is scanned in omega from the branch point for its first sign change (at most 4000 steps of
    # 0.05), or for a double zero (a local minimum below 1e-6 times the largest value before it).
    a = grid[1]; b = grid[end]; kmin = 0
    if c < 0 && f(c * c / 2 + 5e-7) > 0
        tw(ω) = (c * c + ω * ω) / 2
        fprev = f(tw(1e-3)); fmax = abs(fprev); fpp = Inf; found = false
        for j in 1:4000
            ω = 0.05j; fj = f(tw(ω))
            # (a sign change is a simple zero; the estimate's l1 and l2 are too rough here to tell a double one)
            fj <= 0 && return bisect(tw(ω - 0.05), tw(ω)), 1
            if fprev < fpp && fprev <= fj && fprev < 1e-6 * fmax
                a = tw(max(ω - 0.1, 1e-3)); b = tw(ω); found = true; break
            end
            fmax = max(fmax, fj); fpp = fprev; fprev = fj
        end
        found || (a = grid[1]; b = grid[end])
        found && @goto golden
    end
    k = argmin(vals)
    a = grid[max(k - 1, 1)]; b = grid[min(k + 1, end)]
    @label golden
    g = (sqrt(5) - 1) / 2
    x1 = b - g * (b - a); x2 = a + g * (b - a); f1 = f(x1); f2 = f(x2)
    for _ in 1:80
        if f1 < f2
            b = x2; x2 = x1; f2 = f1; x1 = b - g * (b - a); f1 = f(x1)
        else
            a = x1; x1 = x2; f1 = f2; x2 = a + g * (b - a); f2 = f(x2)
        end
    end
    (a + b) / 2, 2
end

# =====================================================================================================================
# Roots and branches along a path in t
# =====================================================================================================================
# assignment of the new roots to the previous labels: the permutation of least total distance (exhaustive for r <= 6)
function permutations_of(n::Int)
    n == 1 && return [[1]]
    out = Vector{Int}[]
    for p in permutations_of(n - 1), k in 1:n
        q = copy(p); insert!(q, k, n); push!(out, q)
    end
    out
end
const PERMS = [permutations_of(n) for n in 1:6]
function match!(new::AbstractVector{ComplexF64}, prev::AbstractVector{ComplexF64})
    r = length(new)
    if r <= 6
        best = Inf; bp = PERMS[r][1]
        for p in PERMS[r]
            d = 0.0
            for k in 1:r; d += abs(new[p[k]] - prev[k]); end
            d < best && (best = d; bp = p)
        end
        new .= new[bp]
    else
        left = collect(1:r); out = similar(new)
        for k in 1:r
            j = argmin([abs(new[l] - prev[k]) for l in left]); out[k] = new[left[j]]; deleteat!(left, j)
        end
        new .= out
    end
    new
end

"""
    rootpath(P, c, ts) -> (roots, logP, logDV)

Along the path ts (ts[1] real, between 0 and t_*): the roots s_k(t) (r x length(ts)), labelled by continuity, the
first P.rp the group that is positive at ts[1]; log P(t) with P(t)^{-2} = D_V(t) prod_+ mu_k prod_- (-mu_k), so that

    e^{s/2} M(s,t) = P(t) prod_+ (s_k - s)^{-1/2} prod_- (s - s_k)^{-1/2}   (principal branches),

its imaginary part continued along the path (the path must be fine enough that it changes by less than pi per step).
"""
rootpath(P::Prep, c::Float64, ts::Vector{ComplexF64}) = rootpath(model(P), c, ts)[1:3]
function rootpath(Mo::Model, c::Float64, ts::Vector{ComplexF64})
    nt = length(ts); r = Mo.r
    MU = Matrix{ComplexF64}(undef, r, nt); LD = Vector{ComplexF64}(undef, nt)
    FX = Vector{Union{Nothing,FixT}}(nothing, nt)
    Threads.@threads :static for ch in collect(Iterators.partition(1:nt, cld(nt, Threads.nthreads())))
        W = MWork(Mo)
        for j in ch
            _, ld, μ, fx = mgfparts(Mo, c, ts[j], W)
            MU[:, j] .= μ; LD[j] = ld; FX[j] = fx
        end
    end
    R = Matrix{ComplexF64}(undef, r, nt); lP = Vector{ComplexF64}(undef, nt)
    s1 = 1 ./ MU[:, 1]
    maximum(abs, imag.(s1)) < 1e-8 * maximum(abs, s1) || @warn "rootpath: roots not real at the start" s1
    pos = sort(real.(s1[real.(s1) .> 0])); neg = sort(real.(s1[real.(s1) .< 0]))
    (length(pos) == Mo.rp && length(neg) == Mo.rm) || error("rootpath: $(length(pos)) positive and $(length(neg)) negative roots at the start, inertia of A $(Mo.rp), $(Mo.rm)")
    R[:, 1] .= complex.([pos; neg])
    for j in 2:nt
        new = 1 ./ MU[:, j]
        match!(new, view(R, :, j - 1))
        R[:, j] .= new
    end
    Lprev = 0.0im
    for j in 1:nt
        Lr = LD[j]
        for k in 1:Mo.rp; Lr += log(1 / R[k, j]); end
        for k in Mo.rp+1:r; Lr += log(-1 / R[k, j]); end
        if j == 1
            m = round(imag(Lr) / 2pi)
            abs(imag(Lr) - 2pi * m) < 1e-7 || @warn "rootpath: D_V prod mu not positive at the start" Lr
            Lc = complex(real(Lr), 0.0)
        else
            Lc = Lr + 2pi * im * round((imag(Lprev) - imag(Lr)) / 2pi)
        end
        lP[j] = -Lc / 2; Lprev = Lc
    end
    R, lP, LD, FX
end

"""whether `cpath` continues the roots by `rootpath_sparse` (true) or on every point of the fine path (`rootpath`)"""
const SPARSE_ROOTS = Ref(true)
"""
    rootpath_sparse(Mo, c, tt, sub) -> (R, lP, FX, minpos, maxneg)

`rootpath` on the fine path tt (sub points per step) evaluated only where needed: at the nodes tt[1:sub:end] (on all
threads), and at the sub - 1 points between two nodes only when the step between them is not safe for the continuation:
when a root moves by 0.45 times its distance to the nearest other root or more (a double root counted once), or when the increment of the imaginary part
of log P (which grows like sqrt|t| along the path) differs by more than pi/4 from the increment of the step before, scaled
by the ratio of the steps in t, which is the prediction that fixes the multiple of 2 pi; the first step is always walked.
The roots, their labels and log P at the nodes are then those of `rootpath` (the labels and the multiples of 2 pi are the
same; checked bit for bit on 18 of 19 designs and starts and on 11 further paths, 8 of them vertical lines, and to 6e-13
with the double roots of model C with breaks at 0.3 and 0.7 at c = 0).  minpos and maxneg are over the points evaluated.
"""
function rootpath_sparse(Mo::Model, c::Float64, tt::Vector{ComplexF64}, sub::Int)
    nt = length(tt); r = Mo.r; rp = Mo.rp
    idx = collect(1:sub:nt); K = length(idx)
    MU = Matrix{ComplexF64}(undef, r, K); LD = Vector{ComplexF64}(undef, K); FX = Vector{Union{Nothing,FixT}}(nothing, K)
    Threads.@threads :static for ch in collect(Iterators.partition(1:K, cld(K, Threads.nthreads())))
        W = MWork(Mo)
        for k in ch
            _, ld, μ, fx = mgfparts(Mo, c, tt[idx[k]], W)
            MU[:, k] .= μ; LD[k] = ld; FX[k] = fx
        end
    end
    R = Matrix{ComplexF64}(undef, r, K); lP = Vector{ComplexF64}(undef, K)
    s1 = 1 ./ MU[:, 1]
    maximum(abs, imag.(s1)) < 1e-8 * maximum(abs, s1) || @warn "rootpath: roots not real at the start" s1
    pos = sort(real.(s1[real.(s1) .> 0])); neg = sort(real.(s1[real.(s1) .< 0]))
    (length(pos) == Mo.rp && length(neg) == Mo.rm) || error("rootpath: $(length(pos)) positive and $(length(neg)) negative roots at the start, inertia of A $(Mo.rp), $(Mo.rm)")
    R[:, 1] .= complex.([pos; neg])
    lr(ld, rts) = (L = ld; for j in 1:rp; L += log(1 / rts[j]); end; for j in rp+1:r; L += log(-1 / rts[j]); end; L)
    Lr1 = lr(LD[1], view(R, :, 1))
    m = round(imag(Lr1) / 2pi)
    abs(imag(Lr1) - 2pi * m) < 1e-7 || @warn "rootpath: D_V prod mu not positive at the start" Lr1
    L1 = complex(real(Lr1), 0.0); lP[1] = -L1 / 2
    # one step of the continuation from (Rp, Lp) to a point with its mu and log D_V, as in `rootpath`: (roots, Lc)
    function step(Rp, Lp, μ, ld)
        nw = 1 ./ μ
        match!(nw, Rp)
        Lr = lr(ld, nw)
        nw, Lr + 2pi * im * round((imag(Lp) - imag(Lr)) / 2pi)
    end
    # the walk over the nodes; sp holds mu and log D_V at the points between nodes; with dry = true the steps that are not safe
    # are only recorded (and continued with the coarse step), so that their points can be computed beforehand on all threads
    function walk!(sp::Dict{Int,Tuple{Vector{ComplexF64},ComplexF64}}, dry::Bool, W::MWork)
        Lprev = L1; dprev = NaN; flagged = Int[]
        mnp = minimum(real, view(R, 1:rp, 1)); mxn = Mo.rm == 0 ? -Inf : maximum(real, view(R, rp+1:r, 1))
        for k in 2:K
            Rp = R[:, k-1]
            # the predicted increment of the imaginary part: the previous one, scaled by the ratio of the steps in t
            dpred = isnan(dprev) ? NaN : dprev * abs(tt[idx[k]] - tt[idx[k-1]]) / abs(tt[idx[k-1]] - tt[idx[k-2]])
            nw = 1 ./ view(MU, :, k); match!(nw, Rp)
            Lr = lr(LD[k], nw)
            Lc = Lr + 2pi * im * round((imag(Lprev) + (isnan(dpred) ? 0.0 : dpred) - imag(Lr)) / 2pi)
            # safe: every root moved by less than 0.45 times its distance to the nearest other root (a double root counted
            # once), and the increment agrees with the prediction to pi/4 (so that the multiple of 2 pi is a fine path's)
            safe = !isnan(dpred) && abs(imag(Lc) - imag(Lprev) - dpred) <= pi / 4
            if safe
                for i in 1:r
                    dmin = Inf
                    for j in 1:r
                        dj = abs(Rp[j] - Rp[i])
                        (j == i || dj <= 1e-6 * abs(Rp[i])) || (dmin = min(dmin, dj))
                    end
                    abs(nw[i] - Rp[i]) < 0.45 * dmin || (safe = false; break)
                end
            end
            if !safe
                push!(flagged, k)
                if !dry
                    # walk through the points between the nodes, as `rootpath` does (its rule for the multiple of 2 pi)
                    Rw = Rp; Lw = Lprev
                    for j in idx[k-1]+1:idx[k]
                        if j < idx[k]
                            μ, ld = haskey(sp, j) ? sp[j] : (v = mgfparts(Mo, c, tt[j], W); (v[3], v[2]))
                            Rw, Lw = step(Rw, Lw, μ, ld)
                            mnp = min(mnp, minimum(real, view(Rw, 1:rp))); Mo.rm > 0 && (mxn = max(mxn, maximum(real, view(Rw, rp+1:r))))
                        else
                            Rw, Lw = step(Rw, Lw, view(MU, :, k), LD[k])
                        end
                    end
                    nw = Rw; Lc = Lw
                end
            end
            R[:, k] .= nw; lP[k] = -Lc / 2
            dprev = imag(Lc) - imag(Lprev); Lprev = Lc
            mnp = min(mnp, minimum(real, view(R, 1:rp, k))); Mo.rm > 0 && (mxn = max(mxn, maximum(real, view(R, rp+1:r, k))))
        end
        flagged, mnp, mxn
    end
    sp = Dict{Int,Tuple{Vector{ComplexF64},ComplexF64}}()
    flagged, _, _ = walk!(sp, true, MWork(Mo))
    # the points between the nodes of the steps the dry walk flagged, on all threads
    pts = [j for k in flagged for j in idx[k-1]+1:idx[k]-1]
    vals = Vector{Tuple{Vector{ComplexF64},ComplexF64}}(undef, length(pts))
    if !isempty(pts)
        Threads.@threads :static for ch in collect(Iterators.partition(eachindex(pts), cld(length(pts), Threads.nthreads())))
            Wt = MWork(Mo)
            for q in ch
                v = mgfparts(Mo, c, tt[pts[q]], Wt); vals[q] = (v[3], v[2])
            end
        end
    end
    for q in eachindex(pts); sp[pts[q]] = vals[q]; end
    _, minpos, maxneg = walk!(sp, false, MWork(Mo))
    R, lP, FX, minpos, maxneg
end

"""
    linepath(P, c, tau0; wmax = 1e4, du = 0.004) -> (ts, roots, logP)

The vertical line t = tau0 + i w, w = b sinh(u) from 0 to wmax (b = 1), on the grid of step du in u.
"""
function linepath(P::Prep, c::Real, tau0::Real; wmax::Real = 1e4, du::Real = 0.004)
    us = 0.0:du:asinh(wmax)
    ts = [complex(Float64(tau0), sinh(u)) for u in us]
    R, lP, _ = rootpath(P, Float64(c), ts)
    ts, R, lP
end

"""
    rootinfo(P, c, ts, R) -> NamedTuple

The root configuration along a path: the roots at ts[1]; the smallest Re of the positive group and the largest Re of
the negative group along the path (the splitting holds when they stay on their sides of the gap at ts[1]); the largest
|s_k| (a root that ran off to infinity would signal a drop of the degree in s).
"""
function rootinfo(P::Prep, ts, R)
    rp = P.rp
    s0 = real.(R[:, 1])
    minpos = minimum(minimum(real.(R[1:rp, j])) for j in axes(R, 2))
    maxneg = P.rm == 0 ? -Inf : maximum(maximum(real.(R[rp+1:end, j])) for j in axes(R, 2))
    (s0 = s0, gaplo = P.rm == 0 ? -Inf : maximum(s0[rp+1:end]), gaphi = minimum(s0[1:rp]),
     minpos = minpos, maxneg = maxneg, maxabs = maximum(abs, R))
end

# =====================================================================================================================
# The contour integral: the outer hyperbola in t and the U-shaped paths in s
# =====================================================================================================================
"""
    CPath

The outer contour t(u) = tau + a (cosh u - 1) + i b sinh u (as in DFExact.ctr_hyp): the nodes t_k = t(k h), k = 0..K,
t'(u_k), the roots and log P(t_k) continued along the path on `sub` points per step, and the root diagnostics.
"""
struct CPath
    c::Float64; tau::Float64; tstar::Float64; a::Float64; b::Float64; h::Float64; phi::Float64
    t::Vector{ComplexF64}; dt::Vector{ComplexF64}
    R::Matrix{ComplexF64}; lP::Vector{ComplexF64}
    minpos::Float64; maxneg::Float64
    Mo::Model                                 # the design with its start
    fx::Vector{Union{Nothing,FixT}}           # the data of the fixed start at the nodes (nothing otherwise)
    wt::Vector{Float64}                       # the weights of the outer rule (h/pi; h/(2 pi) at u = 0)
end

"""
    cpath(P, c; phi = pi/4, hfac = 1, sub = 8, rmax = 1e5, tau = nothing)

The hyperbola of DFExact: asymptotic angle phi, y = min(0.5, 0.9 (pi/2 - phi)) the half-width of the strip in u,
b = 0.9 min(tau, t_* - tau)/sin(y), a = b cot(phi), step h = hfac * 2 pi y/(37 + 37 y tan(phi)), up to |t - tau| = rmax;
tau = min(1, t_*/2) unless given.
"""
cpath(P::Prep, c::Real; kw...) = cpath(model(P), c; kw...)
"""the hyperbola at the angle phi: (tau, a, b, h, K) as in `cpath`; an error beyond PATH_MAXNODES nodes (a factor hfac of
the step that is not positive or far below one)"""
function hyperbola(ts_::Float64, phi::Float64, hfac::Float64, rmax::Float64, τ::Float64)
    y = min(0.5, 0.9 * (pi / 2 - phi))
    b = 0.9 * min(τ, ts_ - τ) / sin(y); a = b / tan(phi)
    h = hfac * 2pi * y / (37 + 37 * tan(phi) * y)
    Kf = asinh(rmax / hypot(a, b)) / h
    (h > 0 && Kf <= PATH_MAXNODES) || error("the outer contour would need $(round(Kf, sigdigits = 3)) nodes (hfac = $hfac, the angle $(round(phi * 180 / pi, sigdigits = 3)) degrees), more than PATH_MAXNODES = $PATH_MAXNODES: hfac must be a positive factor of order one")
    a, b, h, ceil(Int, Kf)
end
"""the most nodes of an outer hyperbola (`hyperbola`): the default rules need at most some 5500 (the 88-degree hyperbola),
11000 in the finer variant of `verify_eval`; the path stores its roots on 8 points per node"""
const PATH_MAXNODES = 100_000
const PHI_FIXED = (45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 78.0, 80.0, 82.0, 84.0, 85.0, 86.0, 87.0, 88.0)
"""
    fixed_phi(Mo, c, ts_, τ, hfac, rmax) -> phi

The asymptotic angle of the outer hyperbola with a fixed start: the start adds g0^2 cf(t) to log M_V(t), which far from
the unit root (|c| g0^2 large) grows along a flat hyperbola, so that the O(1) value of F comes out of large terms that
cancel.  As DFExact.ctr_phi1 in case DK: the smallest of the angles PHI_FIXED at which the largest term
|M_V(t_k) e^{g0^2 cf(t_k)} t'(u_k)/t_k| h/pi of the outer rule is at most 2 (the steepest one if none qualifies).  The
same rule (without the term of the start) chooses the angle with a zero and a stationary start below c = -60 (FAR_CMIN), where
M_V(t), close to e^{t E V} with E V of order 1/(2|c|) while |t| << c^2, grows along a flat hyperbola like
e^{Re(t)/(2|c|)}: on the 45-degree hyperbola the terms reach 1e4 at |t| = 6000 for c = -150, and the distribution function
was off by 3e-6, at c = -200 by 5e4.
"""
function fixed_phi(Mo::Model, c::Float64, ts_::Float64, τ::Float64, hfac::Float64, rmax::Float64)
    g2 = Mo.g0^2; nth = Threads.nthreads()
    Ws = [MWork(Mo) for _ in 1:nth]
    for phd in PHI_FIXED
        phi = phd * pi / 180
        a, b, h, K = hyperbola(ts_, phi, hfac, rmax, τ)
        # (the angle is taken when every term is finite and at most 2, so the nodes are checked in blocks on all threads,
        # and the next angle is tried after the first block with a term beyond)
        ok = true; k0 = 0; B = 16 * nth
        while ok && k0 <= K
            k1 = min(K, k0 + B - 1)
            bad = zeros(Bool, nth)
            Threads.@threads :static for ch in 1:nth
                W = Ws[ch]
                for k in k0+ch-1:nth:k1
                    t = complex(τ + a * (cosh(k * h) - 1), b * sinh(k * h)); dt = complex(a * sinh(k * h), b * cosh(k * h))
                    _, ld, _, fx = mgfparts(Mo, c, t, W)
                    v = exp(-real(ld) / 2 + (fx === nothing ? 0.0 : g2 * real(fx.cf))) * abs(dt / t) * h / pi
                    (isfinite(v) && v <= 2) || (bad[ch] = true; break)
                end
            end
            ok = !any(bad); k0 = k1 + 1
        end
        ok && return phi
    end
    PHI_FIXED[end] * pi / 180
end

"""the end |t - tau| of the outer contour: the larger of 1e5 and 20 c^2 (20 c^2 from c = -70.7 on).  Far from the unit root the
transform decays along the contour only once |t| exceeds c^2 (M_V(t) carries e^{-(c + theta)/2}, and c + theta stays small
while |t| << c^2), so that at |t| = 1e5 the terms are of order e^{-16} at c = -150 and the distribution function was off by
3e-6 (the density, whose kernel decays faster, by 2e-13); at 20 c^2 the factor e^{-Re(c + theta)/2} is below e^{-70} (on
the 45-degree hyperbola).  (Until November 2026 the end was 1e5 down to c = -100.)"""
default_rmax(c::Float64) = max(1e5, 20 * c * c)
"""the c below which the engine takes its rules for c far from the unit root (until November 2026: c = -100): the outer
hyperbola the angle of `fixed_phi` with a zero and a stationary start as well (with a fixed start always), and the U-shapes
the caps 20/min(A, 40/|sigma|) of their segments (`ucaps`).  On the 45-degree hyperbola the terms grow far from the unit
root, and from c = -70 to -100 the distribution function was off by 1e-11 to 1.4e-9 (cases DU, DT, SK, SU at c0 = 0 against
the line integral, case DU at c0 = 2) and the linear form of the envelope by up to 9e-10; with both rules from c = -60 on
within 5e-13 (case DK 3e-11, as below c = -100), and at 0.2 to 1.5 times the cost on that range (faster in cases DU, DT,
1.3 to 1.5 times slower in cases DK and SU, from 25 and 37 ms to 32-38 and 47-57 ms for nine points)."""
const FAR_CMIN = -60.0
"""
the largest c at which the general engine computes (`gen_cmax`; above, every function refuses with `cmax_msg`):
GEN_CMAX0 = 20 for the designs without deterministic terms, with any start; with deterministic terms GEN_CMAX = 16 with a
zero start and GEN_CMAX_FIXED = 10 with a fixed start.  Above the unit root the transform has one root s_+ of the order
2c e^{-2c}, and the reciprocal roots, the eigenvalues of a small matrix, span e^{2c}: with deterministic terms the others
are lost in double precision (the second reciprocal root of case DU is -0.0051 at c = 14, -0.0040 at c = 16, -0.0023 at
c = 17 and zero at c = 19), and the paths fail with "rootpath: ... roots at the start" from c = 17 in case DU and c = 18 in
case DT; with a fixed start the coefficient statistic was off by 3e-5 in case DU and 1e-4 in case DT at c = 12 and 14
without a warning, and NaN or 1e16 with one beyond.  Measured in November 2026 against the classical engine and the
single inversion of DFKan (the coefficient statistic at seven levels from 0.1% to 99.9% and with a fixed start g0 in
{0.5, 1, 3} at three, the moments, the quantiles): without deterministic terms the distribution function within 4e-9 up
to c = 20 (the conditioning of the statistic itself, whose spread is of the order e^{-c}: the engines differ by up to
4e-9 there; with a fixed start 5.4e-8); with an intercept, a linear or a quadratic trend or Perron's models A and C within
1.5e-8 up to c = 16, and with a fixed start (cases DU, DT) within 3.2e-6 up to c = 10.  The classical cases have the
classical engine beyond.
"""
const GEN_CMAX0 = 20.0
const GEN_CMAX = 16.0
const GEN_CMAX_FIXED = 10.0
gen_cmax(D::Determ, fixed::Bool = false) = nreg(D) == 0 ? GEN_CMAX0 : fixed ? GEN_CMAX_FIXED : GEN_CMAX
cmax_msg(what::AbstractString, c::Real, b::Real) =
    "$what at c = $c: the general engine computes up to c = $b for this design and start ($(GEN_CMAX0) without deterministic terms; with them $(GEN_CMAX), and $(GEN_CMAX_FIXED) with a fixed start; above, the roots of its transform are no longer resolved in double precision); the classical cases DK, DU, DT are available with engine = :classical or the default engine = :auto"
"""refuse c above `gen_cmax` of the design D with its start (an ArgumentError that names the statistic what)"""
function check_cmax(D::Determ, c::Real, start::Symbol, gamma0::Real, what::AbstractString)
    b = gen_cmax(D, start == :fixed && gamma0 != 0)
    (isfinite(c) && c > b) && throw(ArgumentError(cmax_msg(what, c, b)))
    nothing
end

"""
    cpath(Mo, c; phi = nothing, hfac = 1, sub = 8, rmax = default_rmax(c), tau = nothing)

As above for the design with its start; phi = nothing: pi/4, except with a fixed start, where `fixed_phi` chooses it.
"""
cpath(Mo::Model, c::Real; phi = nothing, hfac::Real = 1.0, sub::Int = 8, rmax = nothing, tau = nothing) =
    blas1f(() -> cpath_(Mo, Float64(c), phi, Float64(hfac), sub, rmax, tau), dense_la(Mo))
function cpath_(Mo::Model, c::Float64, phi, hfac::Float64, sub::Int, rmax, tau)
    rmax = rmax === nothing ? default_rmax(c) : Float64(rmax)
    ts_ = tstar(Mo, c)[1]
    τ = tau === nothing ? min(1.0, ts_ / 2) : Float64(tau)
    auto = phi === nothing
    phi === nothing && (phi = (Mo.start == :fixed || c < FAR_CMIN) ? fixed_phi(Mo, c, ts_, τ, Float64(hfac), Float64(rmax)) : pi / 4)
    phi = Float64(phi)
    local a, b, h, K, tt, R, lP, FX, minpos, maxneg
    while true
        a, b, h, K = hyperbola(ts_, phi, Float64(hfac), Float64(rmax), τ)
        ud = [k * h / sub for k in 0:sub*K]
        tt = [complex(τ + a * (cosh(u) - 1), b * sinh(u)) for u in ud]
        if SPARSE_ROOTS[]
            R, lP, FX, minpos, maxneg = rootpath_sparse(Mo, c, tt, sub)
        else
            R, lP, _, FX = rootpath(Mo, c, tt)
            minpos = minimum(real, R[1:Mo.rp, :])
            maxneg = Mo.rm == 0 ? -Inf : maximum(real, R[Mo.rp+1:end, :])
            idx = 1:sub:length(tt); R = R[:, idx]; lP = lP[idx]; FX = FX[idx]
        end
        # GLS detrending: far out on a flat hyperbola the positive root can move into Re s < 0, where the kernels of G+
        # grow like e^{-a Re s}; steepen the hyperbola until the positive group stays right of the origin (on the line
        # Re t = tau it does: Re s_+(t) >= s_+(tau) > 0)
        # (and for c > 0, where the positive group leaves Re s > 0 far out and the kernels of G+ grow like
        # e^{|sigma|^2 x^2/(4 Re t)} in the upper tail, x > 0)
        # (and for designs without a proper intercept, as d(r) = 1 + r or a level dummy 1{r <= 0.4}, whose positive roots
        # leave Re s > 0 on the 45-degree hyperbola at every c: at t = 25.6 + 26.4i the smallest real part of the
        # positive group is -83 with d(r) = 1 + r at c = 0)
        (auto && minpos <= 0 && phi < 88 * pi / 180) || break
        phi = min(phi + 5 * pi / 180, 88 * pi / 180)
    end
    idx = 1:sub:length(tt)
    if auto && minpos <= 0
        # c > 0, or GLS detrending, beyond the steepest hyperbola: the vertical line itself, where Re s_+(t) >= s_+(tau) > 0
        # (GLS detrending at cbar = -2, as the power envelope needs at c = -2, has the positive root at Re s = -8495 far out
        # on the 88-degree hyperbola, and the terms overflowed)
        return linecpath(Mo, c, ts_, τ; rmax = Float64(rmax))
    end
    wt = fill(h / pi, K + 1); wt[1] /= 2
    CPath(c, τ, ts_, a, b, h, phi, tt[idx], [complex(a * sinh(k * h), b * cosh(k * h)) for k in 0:K],
          R, lP, minpos, maxneg, Mo, FX, wt)
end

"""
    linecpath(Mo, c, ts_, tau; rmax = 1e5, nq = 24, sub = 8)

The outer integral on the vertical line t = tau + i w itself, (1/pi) int_0^inf Im{I_F(t) i/t} dw, by Gauss-Legendre
panels (nq nodes) in w, the first of length min(tau, 1)/4 and doubling up to rmax, the roots continued on sub points
between nodes.  On the line the root groups stay on their sides of the gap at the real point (|M(s,t)| <= M(Re s, tau)),
which the hyperbola does not guarantee for c > 0.  (The trapezoidal rule on the hyperbola converges geometrically; this
rule is slower to converge in w but needs no strip of analyticity.)
"""
function linecpath(Mo::Model, c::Float64, ts_::Float64, τ::Float64; rmax::Float64 = 1e5, nq::Int = 24, sub::Int = 8, cuts = nothing)
    z, w = glrule(nq)
    if cuts === nothing
        cuts = [0.0]; hh = min(τ, 1.0) / 4
        while cuts[end] < rmax; push!(cuts, cuts[end] + hh); hh *= 2; end
    end
    ws = Float64[]; wq = Float64[]
    for k in 1:length(cuts)-1, q in 1:nq
        push!(ws, cuts[k] + (cuts[k+1] - cuts[k]) * z[q]); push!(wq, (cuts[k+1] - cuts[k]) * w[q] / pi)
    end
    # the path for the continuation: 0, the nodes, and sub - 1 points between consecutive ones
    pathw = [0.0]; nodeidx = Int[]
    for k in eachindex(ws)
        prev = pathw[end]
        for j in 1:sub-1; push!(pathw, prev + (ws[k] - prev) * j / sub); end
        push!(pathw, ws[k]); push!(nodeidx, length(pathw))
    end
    tt = [complex(τ, wv) for wv in pathw]
    if SPARSE_ROOTS[]
        # (the points 1, 1 + sub, 1 + 2 sub, ... of the path are w = 0 and the nodes)
        R, lP, FX, minpos, maxneg = rootpath_sparse(Mo, c, tt, sub)
        R = R[:, 2:end]; lP = lP[2:end]; FX = FX[2:end]
    else
        R, lP, _, FX = rootpath(Mo, c, tt)
        minpos = minimum(real, R[1:Mo.rp, :])
        maxneg = Mo.rm == 0 ? -Inf : maximum(real, R[Mo.rp+1:end, :])
        R = R[:, nodeidx]; lP = lP[nodeidx]; FX = FX[nodeidx]
    end
    CPath(c, τ, ts_, 0.0, 1.0, NaN, pi / 2, tt[nodeidx], fill(complex(0.0, 1.0), length(ws)), R, lP, minpos, maxneg, Mo, FX, wq)
end

"""the number of nodes per panel of a line path built by `linecpath` with its default panels (those in the first panel)"""
linenq(CP::CPath) = count(w -> w < min(CP.tau, 1.0) / 4, imag.(CP.t))
"""the panel boundaries in w of the line path CP built by `linecpath` with its default panels (nq nodes per panel): the
first of length min(tau, 1)/4, doubling"""
function linecuts(CP::CPath, nq::Int = linenq(CP))
    np = length(CP.t) ÷ nq
    cuts = [0.0]; hh = min(CP.tau, 1.0) / 4
    for _ in 1:np; push!(cuts, cuts[end] + hh); hh *= 2; end
    cuts
end

"""
    line_refine(CP, xmax; nq = 24, amax, c0s) -> CPath

The vertical line of CP (c > 0) with its panels in w cut to the oscillation of the outer integrand at |x| <= xmax.  Far
above the unit root the root s_k(t) of the group around which a U-shape runs moves with t (in case DK, s_1(t) is close to
s_1(0) - t/c), and the v-kernel e^{-t v^2 - s_k X(v)}, X(v) = x v + A, has a stationary point in v at v_s = kappa x/2,
kappa = -Im s_k/w, where its phase changes with w at the rate Omega = kappa^2 x^2/4: the inner integral, and with it the
outer integrand, oscillates in w at that rate (at c = 5 and x = 194 in case DK, Omega = 376), with the amplitude
|M_V(t)| exp(-tau v_s^2 - |Re s_k X(v_s)|), which falls as |Re s_k| grows along the line.  Each panel of the line is cut
into parts of length at most 30/Omega (Omega the largest over the roots at the panel's ends and over |x| <= xmax with an amplitude
above e^{-40}; a Gauss-Legendre rule of 24 nodes integrates e^{i Omega w} over 30/Omega to about 1e-18); the roots
are continued along the new nodes.  Without the cut the panels doubling in w carried up to 25 oscillations, and the upper
tail at c = 5 (x = 194, F = 0.991) was off by 2.3e-4.  The number of nodes grows with xmax like xmax (about 2 xmax at c = 5).
With a local null value c0 > 0 (c0s, the values of the kernels with a slope) the kernel is e^{-(t + c0 s_k) v^2 - s_k X(v)}
and the rate is that of its stationary value (comment below), faster by about 1/|1 - c0/c| in case DK.
Returns CP itself when no panel needs cutting.  amax: the values |A| of the kernels (default the shift a of the model, the
A of the t-ratio), each with its own damping; the parabola terms of the linear form U - aV <= b have A = a + b, and their
factor e^{-s A} at the root, e^{-A s_k(t)}, oscillates in w at the rate kappa |A| (with x = 0 the cut is to that rate
alone: P(U - aV <= b) in case DK at c = 5, a = -3.5, b = 770 was off by 2.5e-6 on the doubling panels, to 1e-15 on the
cut ones).
"""
function line_refine(CP::CPath, xmax::Float64; nq::Int = linenq(CP), amax = CP.Mo.a, c0s = (0.0,))
    (isnan(CP.h) && (xmax > 0 || maximum(amax) > CP.Mo.a)) || return CP
    cuts = linecuts(CP, nq); np = length(cuts) - 1
    w = imag.(CP.t); τ = CP.tau
    # only a line with the default panels (`linecpath` without cuts) is refined: a refined line has other panels, and
    # refining it again from panels reconstructed as doubling ones cut it into millions of parts (case DK at c = 5 with
    # verified = true grew to 38 GB that way)
    z1 = glrule(nq)[1][1]
    (np * nq == length(w) && all(abs(w[(k-1)*nq+1] - (cuts[k] + (cuts[k+1] - cuts[k]) * z1)) <= 1e-9 * cuts[k+1] for k in 1:np)) || return CP
    newcuts = [0.0]; changed = false
    for k in 1:np
        a = cuts[k]; b = cuts[k+1]
        Om = 0.0
        for j in ((k - 1) * nq + 1, k * nq)
            wj = w[j]; wj > 0 || continue
            lmv = real(logMtilde(view(CP.R, :, j), CP.Mo.rp, CP.lP[j], 0.0im))      # log |M_V(t)|
            for r in view(CP.R, :, j), am in amax
                kap = abs(imag(r)) / wj; ar = abs(real(r))
                # the largest |x| <= xmax whose term is not negligible: tau v_s^2 + |Re s_k| X(v_s) - log|M_V| < 40 with
                # v_s = kap x/2 and X(v_s) = kap x^2/2 + a (the frequency and the damping both grow with x)
                num = 40 + lmv - ar * am; den = τ * kap^2 / 4 + ar * kap / 2
                num > 0 || continue
                xu = den > 0 ? min(xmax, sqrt(num / den)) : xmax
                Om = max(Om, kap^2 * xu^2 / 4 + kap * am)
            end
            # a local null value c0 != 0: X(v) = c0 v^2 + x v + a, the exponent -alpha v^2 - s_k x v - s_k a with
            # alpha = t + c0 s_k has the stationary value phi_s = (s_k x)^2/(4 alpha) - s_k a, which changes with w at the
            # rate Re dphi_s/dt (s_k' = Im s_k/w the drift of the root along the line, as above).  For c0 near c in case
            # DK, alpha ~ t (1 - c0/c) and the rate is that of c0 = 0 times 1/|1 - c0/c| (at c = 10, c0 = 9 the panels cut
            # to the rate of c0 = 0 left the distribution function off by 1.1e-3 at x = 1000 and 3.8e-3 at x = 3000).  The
            # largest |x| <= xmax at which the term is not negligible from log|M_V| + Re phi_s; the cut is to the larger
            # of this rate and that of c0 = 0.  Only c0 > 0: with c0 < 0, alpha ~ t (1 + |c0|/c) and the rate is below that
            # of c0 = 0, which the cut above covers (taken for c0 < 0 too, the stationary values at the roots of a fixed
            # start gave rates far above the oscillation: the line of case DU at c = 10, g0 = 1, c0 = -2 grew to 1e6 nodes
            # at x = 132, against 2544)
            for c0 in c0s
                (c0 > 0 && xmax > 0) || continue
                for r in view(CP.R, :, j), am in amax
                    rp = imag(r) / wj
                    al = CP.t[j] + c0 * r
                    L0 = 40 + lmv - real(r) * am
                    L0 > 0 || continue
                    Q = real(r * r / (4al))
                    xu = Q < 0 ? min(xmax, sqrt(L0 / -Q)) : xmax
                    Om = max(Om, abs(real(xu^2 * (2r * rp * al - r * r * (1 + c0 * rp)) / (4al^2))) + abs(rp) * am)
                end
            end
        end
        m = max(1, ceil(Int, min((b - a) * Om / 30, 1e6)))
        m > 1 && (changed = true)
        for i in 1:m; push!(newcuts, a + (b - a) * i / m); end
    end
    changed || return CP
    nn = (length(newcuts) - 1) * nq
    nn > LINE_MAXNODES && error("line_refine: the line would need $nn nodes for |x| <= $xmax at c = $(CP.c), more than LINE_MAXNODES = $LINE_MAXNODES (the oscillation of the outer integrand grows with x; at c = 10 the bulk needs about 60000)")
    length(newcuts) > LINE_MAXPANELS && @warn "line_refine: the line needs $(length(newcuts) - 1) panels for |x| <= $xmax at c = $(CP.c) (the oscillation of the outer integrand grows with x); this is slow"
    linecpath(CP.Mo, CP.c, CP.tstar, τ; nq, cuts = newcuts)
end
const LINECACHE = Dict{UInt64,CPath}()
const LINEORDER = UInt64[]
"""the number of panels of a refined line above which `line_refine` warns that it is slow"""
const LINE_MAXPANELS = 4000
"""the number of nodes of a refined line above which `line_refine` refuses (about 100 bytes a node and eight root
evaluations each)"""
const LINE_MAXNODES = 400_000
"""`line_refine` cached per path, xmax and the values amax beyond the shift a rounded up to powers of two, and the local
null values c0 (a number or a vector; those != 0 count where x != 0) (nothing to do unless CP is a vertical line)"""
function line_refined(CP::CPath, xs::AbstractVector{Float64}; amax = CP.Mo.a, c0 = 0.0)
    isnan(CP.h) || return CP
    c0s = sort(unique(Float64[c0...]))
    fin = [abs(x) for x in xs if isfinite(x)]
    xm = isempty(fin) ? 0.0 : maximum(fin)
    xb = xm > 0 ? 2.0^ceil(log2(xm)) : 0.0
    ab = sort(unique([2.0^ceil(log2(a)) for a in amax if isfinite(a) && a > CP.Mo.a]))
    (xb > 0 || !isempty(ab)) || return CP
    # (keyed by the contents of the path, not its address, which a freed path can pass on to a new one)
    key = isempty(ab) ? hash((CP.t, CP.c, CP.Mo.start, CP.Mo.g0, dkey(CP.Mo.P.D), CP.Mo.P.cbar, xb)) :
                        hash((CP.t, CP.c, CP.Mo.start, CP.Mo.g0, dkey(CP.Mo.P.D), CP.Mo.P.cbar, xb, ab))
    c0s = filter(>(0), c0s)                                    # (only c0 > 0 refines further, line_refine)
    (xb > 0 && !isempty(c0s)) && (key = hash((key, c0s)))
    ab = [CP.Mo.a; ab]
    hit = lock(CPLOCK) do; get(LINECACHE, key, nothing); end
    hit === nothing || return hit
    CPr = line_refine(CP, xb; amax = ab, c0s)
    lock(CPLOCK) do; cache_put!(LINECACHE, LINEORDER, key, CPr); end
    CPr
end

# ---------- the incomplete Gaussian integrals (copied from DFExact: ctr_h, ctr_rend, ctr_moments, ctr_pieces) ----------
"""h_k(z) = int_0^inf u^k exp(-u^2 - 2 z u) du, k = 0, 1, 2, for Re z >= 0 (DFExact.ctr_h)"""
@inline function ctr_h(z::ComplexF64)
    if abs2(z) < 36
        h0 = (sqrt(pi) / 2) * erfcx(z)
        h1 = (1 - 2z * h0) / 2
        return h0, h1, (h0 - 2z * h1) / 2
    end
    w = 1 / (2z); w2 = w * w
    a0 = w; a1 = w2; a2 = 2 * w2 * w
    s0 = a0; s1 = a1; s2 = a2
    j = 0
    while true
        r0 = -(2j + 1) * (2j + 2) / (j + 1) * w2
        r1 = -(2j + 2) * (2j + 3) / (j + 1) * w2
        r2 = -(2j + 3) * (2j + 4) / (j + 1) * w2
        !(abs2(r2) < 1) && break
        a0 *= r0; a1 *= r1; a2 *= r2
        s0 += a0; s1 += a1; s2 += a2
        j += 1
        (abs2(a0) < 1e-34 * abs2(s0) && abs2(a2) < 1e-34 * abs2(s2)) && break
    end
    s0, s1, s2
end
@inline function ctr_rend(a::Float64, sa::ComplexF64, beta::ComplexF64, Ea::ComplexF64)
    z = sa * a + beta / (2sa)
    if real(z) >= 0
        n = 0; g0, g1, g2 = ctr_h(z)
    else
        n = 1; h0, h1, h2 = ctr_h(-z); g0 = -h0; g1 = h1; g2 = -h2
    end
    i1 = 1 / sa; i2 = i1 * i1; i3 = i2 * i1
    n, Ea * i1 * g0, Ea * (a * i1 * g0 + i2 * g1), Ea * (a * a * i1 * g0 + 2a * i2 * g1 + i3 * g2)
end
"""int over the pieces (a, b) of v^m e^{-t v^2 - s X(v)} dv, m = 0, 1, 2, X(v) = c0 v^2 + q v + A (DFExact.ctr_moments with
the constant A of X in place of 1/2): the endpoint remainders and the full-line Gaussian as often as the flags at the
lower ends exceed those at the upper ends; alpha = t + c0 s; a semi-infinite piece needs Re alpha > 0"""
function ctr_moments(t::ComplexF64, s::ComplexF64, c0::Float64, q::Float64, A::Float64, pcs)
    alpha = c0 == 0 ? t : t + c0 * s; beta = q * s; sa = sqrt(alpha)
    M0 = zero(ComplexF64); M1 = zero(ComplexF64); M2 = zero(ComplexF64); nf = 0
    @inbounds for (a, b) in pcs
        na, R0, R1, R2 = ctr_rend(a, sa, beta, exp(-t * a * a - s * (c0 * a * a + q * a + A)))
        M0 += R0; M1 += R1; M2 += R2; nf += na
        if isfinite(b)
            nb, S0, S1, S2 = ctr_rend(b, sa, beta, exp(-t * b * b - s * (c0 * b * b + q * b + A)))
            M0 -= S0; M1 -= S1; M2 -= S2; nf -= nb
        end
    end
    if nf != 0
        G0 = sqrt(pi) / sa * exp(beta * beta / (4alpha) - s * A); b2 = beta / (2alpha)
        M0 += nf * G0; M1 -= nf * b2 * G0; M2 += nf * (b2 * b2 + 1 / (2alpha)) * G0
    end
    M0, M1, M2
end
ctr_moments(t::ComplexF64, s::ComplexF64, q::Float64, pcs) = ctr_moments(t, s, 0.0, q, 0.5, pcs)
"""`ctr_moments` with the factors that do not depend on x given, per node s of the U-shapes: sa = sqrt(t + c0 s), i1 =
1/sa and eA = e^{-A s}, the exponential at the end v = 0 (computed once per node instead of once per node and x; the
end v = 0 is the lower end of the first piece of every x)"""
@inline function ctr_moments_pre(t::ComplexF64, s::ComplexF64, c0::Float64, q::Float64, A::Float64, pcs, sa::ComplexF64, i1::ComplexF64,
                                 eA::ComplexF64)
    beta = q * s; hb = beta * i1 / 2
    i2 = i1 * i1; i3 = i2 * i1
    M0 = zero(ComplexF64); M1 = zero(ComplexF64); M2 = zero(ComplexF64); nf = 0
    @inbounds for (a, b) in pcs
        Ea = a == 0 ? eA : exp(-t * a * a - s * (c0 * a * a + q * a + A))
        z = sa * a + hb
        if real(z) >= 0
            g0, g1, g2 = ctr_h(z)
        else
            h0, h1, h2 = ctr_h(-z); g0 = -h0; g1 = h1; g2 = -h2; nf += 1
        end
        M0 += Ea * i1 * g0; M1 += Ea * (a * i1 * g0 + i2 * g1); M2 += Ea * (a * a * i1 * g0 + 2a * i2 * g1 + i3 * g2)
        if isfinite(b)
            Eb = exp(-t * b * b - s * (c0 * b * b + q * b + A))
            z = sa * b + hb
            if real(z) >= 0
                g0, g1, g2 = ctr_h(z)
            else
                h0, h1, h2 = ctr_h(-z); g0 = -h0; g1 = h1; g2 = -h2; nf -= 1
            end
            M0 -= Eb * i1 * g0; M1 -= Eb * (b * i1 * g0 + i2 * g1); M2 -= Eb * (b * b * i1 * g0 + 2b * i2 * g1 + i3 * g2)
        end
    end
    if nf != 0
        alpha = sa * sa
        G0 = sqrt(pi) * i1 * exp(beta * beta / (4alpha) - s * A); b2 = beta / (2alpha)
        M0 += nf * G0; M1 -= nf * b2 * G0; M2 += nf * (b2 * b2 + 1 / (2alpha)) * G0
    end
    M0, M1, M2
end
"""the terms of `ctr_rend` at the end v with the factor E = e^{-s X(v)} in place of e^{-t v^2 - s X(v)}: at a root v of X
the part of the kernel that Filon's rule integrates against e^{-t v^2} (`filon_roots`); beta = q s, i1 = 1/sa"""
@inline function rend_part(v::Float64, sa::ComplexF64, i1::ComplexF64, beta::ComplexF64, E::ComplexF64)
    z = sa * v + beta * i1 / 2
    if real(z) >= 0
        g0, g1, g2 = ctr_h(z)
    else
        h0, h1, h2 = ctr_h(-z); g0 = -h0; g1 = h1; g2 = -h2
    end
    i2 = i1 * i1; i3 = i2 * i1
    E * i1 * g0, E * (v * i1 * g0 + i2 * g1), E * (v * v * i1 * g0 + 2v * i2 * g1 + i3 * g2)
end
"""h_k(z), k = 0..3 (ctr_h with h_3 = h_1 - z h_2 by the recurrence 2 h_k + 2 z h_{k-1} = (k - 1) h_{k-2}, and the
asymptotic series to the term of h_3 for |z| >= 6)"""
@inline function ctr_h3(z::ComplexF64)
    if abs2(z) < 36
        h0 = (sqrt(pi) / 2) * erfcx(z)
        h1 = (1 - 2z * h0) / 2; h2 = (h0 - 2z * h1) / 2
        return h0, h1, h2, h1 - z * h2
    end
    w = 1 / (2z); w2 = w * w
    a0 = w; a1 = w2; a2 = 2 * w2 * w; a3 = 6 * w2 * w2
    s0 = a0; s1 = a1; s2 = a2; s3 = a3
    j = 0
    while true
        r0 = -(2j + 1) * (2j + 2) / (j + 1) * w2
        r1 = -(2j + 2) * (2j + 3) / (j + 1) * w2
        r2 = -(2j + 3) * (2j + 4) / (j + 1) * w2
        r3 = -(2j + 4) * (2j + 5) / (j + 1) * w2
        !(abs2(r3) < 1) && break
        a0 *= r0; a1 *= r1; a2 *= r2; a3 *= r3
        s0 += a0; s1 += a1; s2 += a2; s3 += a3
        j += 1
        (abs2(a0) < 1e-34 * abs2(s0) && abs2(a3) < 1e-34 * abs2(s3)) && break
    end
    s0, s1, s2, s3
end
@inline function ctr_rend3(a::Float64, sa::ComplexF64, beta::ComplexF64, Ea::ComplexF64)
    z = sa * a + beta / (2sa)
    if real(z) >= 0
        n = 0; g0, g1, g2, g3 = ctr_h3(z)
    else
        n = 1; h0, h1, h2, h3 = ctr_h3(-z); g0 = -h0; g1 = h1; g2 = -h2; g3 = h3
    end
    i1 = 1 / sa; i2 = i1 * i1; i3 = i2 * i1; i4 = i3 * i1
    n, Ea * i1 * g0, Ea * (a * i1 * g0 + i2 * g1), Ea * (a * a * i1 * g0 + 2a * i2 * g1 + i3 * g2),
       Ea * (a^3 * i1 * g0 + 3a * a * i2 * g1 + 3a * i3 * g2 + i4 * g3)
end
"""as ctr_moments, with the moment m = 3 as well (the density of the coefficient statistic)"""
function ctr_moments3(t::ComplexF64, s::ComplexF64, c0::Float64, q::Float64, A::Float64, pcs)
    alpha = c0 == 0 ? t : t + c0 * s; beta = q * s; sa = sqrt(alpha)
    M0 = zero(ComplexF64); M1 = zero(ComplexF64); M2 = zero(ComplexF64); M3 = zero(ComplexF64); nf = 0
    @inbounds for (a, b) in pcs
        na, R0, R1, R2, R3 = ctr_rend3(a, sa, beta, exp(-t * a * a - s * (c0 * a * a + q * a + A)))
        M0 += R0; M1 += R1; M2 += R2; M3 += R3; nf += na
        if isfinite(b)
            nb, S0, S1, S2, S3 = ctr_rend3(b, sa, beta, exp(-t * b * b - s * (c0 * b * b + q * b + A)))
            M0 -= S0; M1 -= S1; M2 -= S2; M3 -= S3; nf -= nb
        end
    end
    if nf != 0
        G0 = sqrt(pi) / sa * exp(beta * beta / (4alpha) - s * A); b2 = beta / (2alpha)
        M0 += nf * G0; M1 -= nf * b2 * G0; M2 += nf * (b2 * b2 + 1 / (2alpha)) * G0
        M3 -= nf * (b2^3 + 3b2 / (2alpha)) * G0
    end
    M0, M1, M2, M3
end
"""the pieces of {v > 0} with X(v) = c0 v^2 + x v + A > 0 (plus) and < 0 (minus), split at the positive roots of X (a
double root is no split), as DFExact.ctr_pieces"""
function xpieces(x::Float64, c0::Float64 = 0.0, A::Float64 = 0.5)
    if c0 == 0
        x < 0 || return [(0.0, Inf)], Tuple{Float64,Float64}[]
        v = -A / x
        return [(0.0, v)], [(v, Inf)]
    end
    dsc = x * x - 4c0 * A
    r = dsc <= 0 ? Float64[] : sort([y for y in ((-x - sqrt(dsc)) / (2c0), (-x + sqrt(dsc)) / (2c0)) if y > 0])
    bk = [0.0; r; Inf]
    plus = Tuple{Float64,Float64}[]; minus = Tuple{Float64,Float64}[]
    for j in 1:length(bk)-1
        isodd(j) ? push!(plus, (bk[j], bk[j+1])) : push!(minus, (bk[j], bk[j+1]))
    end
    plus, minus
end
"""the slowest decay scale of the v-kernel along the rays (DFExact.ctr_xscale with the constant A): per piece 1/min(max|X|,
|X(vertex)|) and |c0|/X'(v_j)^2 at an end v_j that is a root of X; semi-infinite pieces cut where e^{-Re(t) v^2} = e^{-18}"""
function xscale(tr::Float64, c0::Float64, q::Float64, A::Float64, pcs)
    isempty(pcs) && return 0.0
    sc = 0.0
    for (a, b) in pcs
        bb = isfinite(b) ? b : a + 6 / sqrt(tr)
        m = 0.0
        for k in 0:32
            v = a + (bb - a) * k / 32
            m = max(m, abs(c0 * v * v + q * v + A))
        end
        sp = 0.0
        if c0 != 0
            v0 = -q / (2c0)
            a < v0 < b && (m = min(m, abs(c0 * v0 * v0 + q * v0 + A)))
            for e in (a, b)
                (e > 0 && isfinite(e)) || continue
                sp = max(sp, abs(c0) / max((2c0 * e + q)^2, 1e-12))
            end
        end
        sc = max(sc, 1 / max(m, 1e-12), sp)
    end
    sc
end
active(pcs, tr::Float64) = [p for p in pcs if tr * p[1] * p[1] <= 50]

"""|z| by sqrt(abs2(z)), for the estimates of the rounding error in the inner loops (hypot, which `abs` uses, guards against
overflow and underflow and took about a third of the time of `contour_eval`; the terms there are far from both)"""
@inline fabs(z::ComplexF64) = sqrt(abs2(z))

# ---------- the U-shaped paths ----------
"""panels (lo, hi) of [st, en] (either order) graded from st: first length h0 (Inf: one panel), growing by `ratio`"""
function seg_panels!(out::Vector{Tuple{Float64,Float64}}, st::Float64, en::Float64, h0::Float64; ratio::Float64 = 2.0)
    len = abs(en - st); dir = en >= st ? 1.0 : -1.0
    len == 0 && return out
    a = 0.0; h = isfinite(h0) ? min(h0, len) : len
    while a < len * (1 - 1e-14)
        b = min(a + h, len); (len - b) < 0.5h && (b = len)
        u1 = st + dir * a; u2 = st + dir * b
        push!(out, (min(u1, u2), max(u1, u2)))
        a = b; h *= ratio
    end
    out
end
"""panels of [A, B] graded toward the focus points (position, scale): a focus point whose position lies within its own
scale of one with a smaller scale is dropped; between two kept points the interval is split at the midpoint (as in
DFExact.ctr_upath).  Points within 1e-12 (B - A) of each other or of an end are merged with it: a tolerance relative to
the length of the interval, because far above the unit root the segments of the U-shapes are tiny (the positive root
lies within about 2c e^{-2c} of the branch point -t/c0 of the coefficient statistic, 1e-16 at c = 20); with the
tolerance 1e-12 max(B - A, 1) used before, every point of a segment shorter than 1e-12 merged with its lower end, the
segment got no panel at all, and from c = 16 on the coefficient statistic was off by up to 0.2."""
function graded_panels(A::Float64, B::Float64, foc::Vector{Tuple{Float64,Float64}}; ratio::Float64 = 2.0)
    kl = Float64[]; kh = Float64[]
    for k in sortperm([f[2] for f in foc])
        pos = clamp(foc[k][1], A, B); h = foc[k][2]
        any(abs(pos - y) <= h for y in kl) && continue
        push!(kl, pos); push!(kh, h)
    end
    tol = 1e-12 * (B - A)
    pts = [(A, Inf)]
    for k in sortperm(kl)
        if abs(kl[k] - pts[end][1]) <= tol
            pts[end] = (pts[end][1], min(pts[end][2], kh[k]))
        else
            push!(pts, (kl[k], kh[k]))
        end
    end
    abs(B - pts[end][1]) <= tol ? (pts[end] = (B, pts[end][2])) : push!(pts, (B, Inf))
    out = Tuple{Float64,Float64}[]
    for k in 1:length(pts)-1
        a, ha = pts[k]; b, hb = pts[k+1]
        b - a <= 0 && continue
        if !isfinite(ha) && !isfinite(hb)
            push!(out, (a, b))
        elseif !isfinite(ha)
            seg_panels!(out, b, a, hb; ratio)
        elseif !isfinite(hb)
            seg_panels!(out, a, b, ha; ratio)
        else
            m = (a + b) / 2
            seg_panels!(out, a, m, ha; ratio); seg_panels!(out, b, m, hb; ratio)
        end
    end
    out
end

"""
    ushape!(S, W, grp, oth, sigma, side, n, ntail)

Nodes S and weights W of (1/2 pi i) int over the U-shape around the group of roots grp: side = +1 the segment Re s =
sigma with the rays to +infinity (Gamma_+), side = -1 the mirror image (Gamma_-), traversed from below to above.  d is
the horizontal distance from the segment to the nearest root of the group; the segment runs from min Im - d to max Im +
d, so that the rays pass every root of the group at a distance of at least d.  The segment is graded toward the levels
of all roots and of the origin (scales: their horizontal distances from the segment), each ray toward the roots of the
group it passes (scales: their vertical distances), up to R = max distance + max(2d, 1), with n nodes per panel; beyond
R the map r = R + R' (u/(1 - u))^2, R' = max(R, 2), with ntail nodes (the integrand decays like a power r^{-r/2 - 1} with
an expansion in 1/r, which this map turns into integer powers of 1 - u for every r).  With lm = log Mt (the fixed start,
whose roots are essential singularities) the rays lie at the heights of `ulevels` with wide = true (each root passed at
its own horizontal distance from the segment), and every panel of the segment and of the rays is cut by `nsplit` to the
rate at which log Mt varies (akern: the constant a of the kernels).
"""
function ushape!(S::Vector{ComplexF64}, W::Vector{ComplexF64}, grp::AbstractVector{ComplexF64}, oth::AbstractVector{ComplexF64},
                 σ::Float64, side::Int, n::Int, ntail::Int; extra = ComplexF64[], rs::Float64 = 0.0, hmax::Float64 = Inf,
                 lm = nothing, akern::Float64 = 0.0, levels = nothing)
    z, w = glrule(n); zt, wt = glrule(ntail)
    d = minimum(side * (real(s) - σ) for s in grp)
    d > 0 || error("ushape: segment not separated from its roots")
    ylo, yhi = levels === nothing ? ulevels(grp, σ, side, lm !== nothing) : levels
    foc = Tuple{Float64,Float64}[]
    for s in grp; push!(foc, (imag(s), max(abs(real(s) - σ), 1e-6d))); end
    for s in oth; push!(foc, (imag(s), max(abs(real(s) - σ), 1e-6d))); end
    push!(foc, (0.0, max(abs(σ), 1e-6d)))
    for s in extra; push!(foc, (imag(s), max(abs(real(s) - σ), 1e-6d))); end
    for (a0, b0) in graded_panels(ylo, yhi, foc)
        # (along the segment the kernels oscillate like e^{-i a Im s}: panels at most hmax long)
        m = isfinite(hmax) ? max(1, ceil(Int, (b0 - a0) / hmax)) : 1
        for j in 0:m-1
            a1 = a0 + (b0 - a0) * j / m; b1 = a0 + (b0 - a0) * (j + 1) / m
            ms = nsplit(lm, complex(σ, a1), complex(σ, b1), n, akern, side)
            for i in 0:ms-1
                a = a1 + (b1 - a1) * i / ms; b = a1 + (b1 - a1) * (i + 1) / ms
                for q in 1:n
                    push!(S, complex(σ, a + (b - a) * z[q])); push!(W, im * (b - a) * w[q])
                end
            end
        end
    end
    for (yc, dir) in ((yhi, 1), (ylo, -1))
        sgn = side == 1 ? dir : -dir
        rf = Tuple{Float64,Float64}[]
        for s in grp; push!(rf, (side * (real(s) - σ), max(abs(imag(s) - yc), 1e-6d))); end
        R = max(maximum(f[1] for f in rf) + max(2d, 1.0), rs)
        for (a1, b1) in graded_panels(0.0, R, rf)
            ms = nsplit(lm, complex(σ + side * a1, yc), complex(σ + side * b1, yc), n, akern, side)
            for i in 0:ms-1
                a = a1 + (b1 - a1) * i / ms; b = a1 + (b1 - a1) * (i + 1) / ms
                for q in 1:n
                    push!(S, complex(σ + side * (a + (b - a) * z[q]), yc)); push!(W, sgn * (b - a) * w[q])
                end
            end
        end
        Rt = max(R, 2.0)
        for q in 1:ntail
            u = zt[q]; yv = u / (1 - u)
            r = R + Rt * yv * yv; dr = 2Rt * yv / (1 - u)^2
            push!(S, complex(σ + side * r, yc)); push!(W, sgn * dr * wt[q])
        end
    end
    W ./= 2pi * im
    S, W
end

"""
    fixbound(fx, Dv, t) -> (B, phi)

The radius beyond which the exponent of the fixed start is within 1/2 of its limit (the third bound of the proof of the
theorem on the fixed start): with Theta_t = Lambda^{-1} H/2 (H = 2 Lambda Theta_t, the matrix whose eigenvalues are the
reciprocal roots), m_t the mean of X under the tilt, and for an angle phi with H_phi = Re(e^{-i phi} Theta_t) positive
definite, Theta_t = e^{i phi} H_phi^{1/2} (I + i C) H_phi^{1/2} with C real symmetric, and
|Q_kappa(s,t) - Q_kappa(inf,t)| <= 1/2 wherever |s| >= rho_phi max(1, |z_phi|^2), rho_phi = ||H_phi^{-1/2} Lambda^{-1}
H_phi^{-1/2}||, z_phi = H_phi^{-1/2} m_t.  (The proof carries over to the rotation: (I + i C)^{-1} has norm at most 1 for
C real symmetric, and the Neumann series of (I + i C - w e^{-i phi} Lambda~)^{-1} is the same; only the positive
definiteness of H_phi, which the proof shows for phi = 0 with Re t < t_*, is checked here.)  B the smallest over phi in
(0, pi/4, -pi/4), phi the angle that gives it; (Inf, NaN) when no H_phi is positive definite.
"""
function fixbound(fx::FixT, Dv::Vector{Float64}, t::ComplexF64)
    Θ = (fx.H ./ Dv) ./ 2                 # Lambda^{-1} H / 2
    Θ = (Θ + transpose(Θ)) / 2
    best = Inf; bphi = NaN
    for φ in (0.0, pi / 4, -pi / 4)
        Hφ = real(cis(-φ) .* Θ)
        E = eigen(Symmetric(Hφ)); minimum(E.values) > 0 || continue
        Hm = E.vectors * Diagonal(1 ./ sqrt.(E.values)) * E.vectors'
        ρ = opnorm(Hm * Diagonal(1 ./ Dv) * Hm); z = Hm * fx.mt
        B = ρ * max(1.0, sum(abs2, z))
        B < best && (best = B; bphi = φ)
    end
    best, bphi
end
"""
the rule of the U-shapes of the fixed start: :heuristic (the default: the abscissa by `place`, the wide heights of
`ulevels`), :bound (the abscissas alpha s_+-(tau) of the zero start at the real point of the contour and the heights
+-(1 + B) of `fixbound`, as in the proof of the theorem on the fixed start) or :both (the heuristic abscissa, heights at
least 1 + B).  The bound rule was tried and not adopted (`scripts/check_fixed_bound.jl`, results/fixed_bound.txt): B
grows with |t| and |c| (beyond 1e5 at every node of case DT at c = -40, 3.6e6 for a level dummy at c = 0), so that the
U-shapes carry 2 to 200 times the nodes and take 1.4 to 180 times as long; and on the hyperbola, whose real part passes
t_* far out, the segment at a fixed abscissa leaves the strip where the proof bounds the exponent on it, which then grew
(case DT, g0 = 3, c = -10: rounding estimate 1e23, against 8e-14 by the present rule).
"""
const FIXED_RULE = Ref(:heuristic)
"""the largest height 1 + B of the bound rule (a U-shape beyond it would carry millions of nodes)"""
const FIXED_HCAP = Ref(1e5)
"""the largest number of nodes of the U-shapes of a node of the outer contour (with a fixed start by the wide heights, with
the zero and the stationary start by the segments of the narrow ones): `ushape_guard` stops a call beyond it before the
outer sum, `ushapes` at a node the guard leaves to it"""
const USHAPE_MAXNODES = Ref(1_000_000)

"""the wide U-shapes of the fixed start (true, the default; false: the narrow ones, kept only so that
`scripts/check_fixed_far.jl` can show what they did)"""
const WIDE_U = Ref(true)
"""
    ulevels(grp, σ, side, wide) -> (ylo, yhi)

The heights of the rays of the U-shape around the group grp with its segment at Re s = σ.  wide = false (the zero and the
stationary start): min Im - d and max Im + d, d the horizontal distance of the segment from the nearest root, so that every
root is passed at a distance of at least d.  wide = true (the fixed start, whose roots are essential singularities of
Mt): min_k (Im s_k - d_k) and max_k (Im s_k + d_k), d_k = |Re s_k - σ| the horizontal distance of root k, so that every root
is passed at a distance of at least its own horizontal distance from the segment, as the U-shape around a single root of
DFLimit (one root per group: the same).  With roots of different sizes in a group the narrow U-shape passes the far ones
close: Perron's model C at c = -40, g0 = 1.5 has the negative roots -107.6 and -282 at t = 1, the segment of G- lies at
-0.39, the rays at the heights -+107.2 passed the root at -282 at 107 and met Re log Mt = 59 between the roots, and the
distribution function came out as 1e13.
"""
function ulevels(grp::AbstractVector{ComplexF64}, σ::Float64, side::Int, wide::Bool)
    if wide && WIDE_U[]
        return minimum(imag(s) - abs(real(s) - σ) for s in grp), maximum(imag(s) + abs(real(s) - σ) for s in grp)
    end
    d = minimum(side * (real(s) - σ) for s in grp)
    minimum(imag, grp) - d, maximum(imag, grp) + d
end

"""
    nsplit(lm, za, zb, n, A, side) -> m

The number of equal parts into which the panel from za to zb (a straight piece of a U-shape, n Gauss-Legendre nodes per
part) is cut so that the integrand is resolved, from log Mt = lm(s) at the ends and the midpoint (three evaluations, no
error estimate): with the rate omega = max(|L(mid) - L(za)|, |L(zb) - L(mid)|)/(|zb - za|/2) + A (log Mt, and the kernel
e^{-A s}), the parts are short enough that omega times their length is at most n, which keeps the oscillation within half
a radian per node on average.  A panel on which the integrand is negligible (max Re L + log|zb - za| + max(0, -A Re s)
below -40) is not cut.  lm = nothing (the zero and the stationary start, where log Mt has no exponential part): 1.
With a fixed start far from the unit root log Mt varies like -omega s near the real axis, omega the mean of U + a under
the tilt: 10.6 in case DT at c = -50, g0 = 5, where panels of the length 20/A = 40 along the segment of G- carried more
than 60 oscillations, and the distribution function was off by 9e-4.
"""
nsplit(::Nothing, za::ComplexF64, zb::ComplexF64, n::Int, A::Float64, side::Int) = 1
function nsplit(lm::L, za::ComplexF64, zb::ComplexF64, n::Int, A::Float64, side::Int) where {L}
    len = abs(zb - za)
    len > 0 || return 1
    zm = (za + zb) / 2
    La = lm(za); Lm = lm(zm); Lb = lm(zb)
    (isfinite(La) && isfinite(Lm) && isfinite(Lb)) || return 1
    kb(z) = side == 1 ? max(0.0, -A * real(z)) : 0.0
    big = max(real(La) + kb(za), real(Lm) + kb(zm), real(Lb) + kb(zb)) + log(len)
    big < -40 && return 1
    om = max(abs(Lm - La), abs(Lb - Lm)) / (len / 2) + A
    clamp(ceil(Int, min(om * len / n, 1e6)), 1, NSPLIT_MAX)
end
"""the most parts `nsplit` cuts a panel into (a panel that would need more lies where Mt is enormous: then the paths are
badly placed, and the estimate of the rounding error, hence the warning, shows it)"""
const NSPLIT_MAX = 64

"""the grid of `place`: two-ended (true, the default), or the earlier one from 2% to 98% of the admissible interval
(false), kept only so that `scripts/check_fixed_far.jl` can show what that grid did and how the warning saw it"""
const PLACE_TWO_ENDED = Ref(true)
"""
    place(logMt, grp, oth, side, lo, hi, A, σ0) -> sigma

The abscissa of the U-shape around the root group grp (side +1: the segment left of the group, rays to +infinity; -1
the mirror image) in the admissible interval (lo, hi), where Mt has essential singularities at the roots (the fixed
start).  The objective is the largest of Re log Mt + kb at the probe points of the U-shape (the corners, the segment at
the levels of the roots of both groups and of the origin, the rays above and below each root and beyond the group, and one point far
out on each ray, where Mt tends to its limit at infinity), kb = max(0, -A Re s) on G+ (the kernel (1 - e^{-A s})/s and
the v-kernel e^{-s X}, 0 < X <= A, grow to the left of the origin), plus the grading cost 0.5 sum log(1 + d/dist) of
DFExact.ctr_place toward the origin and the nearest root of the other group (d the distance of the segment from the
group).  It is minimized on a grid of distances d of the segment from the group, geometric from both ends of the
admissible interval of length D (from 1e-3 D to D/2 and from D/2 to D - 1e-3 D, 12 points each), refined by 12
golden-section steps between the neighbours of the best; the default abscissa σ0 is kept unless the minimum is lower by
more than 2 (a factor e^2 in the largest term).  The grid reaches both ends because the best abscissa can lie at either:
far from the unit root the negative root of case DU runs off to -1680 (c = -40, g0 = 1.5, t = 1), Mt grows like
e^{1.13 |s|} between the origin and it, and the U-shape around it must cross the real axis next to the origin (at -0.5,
where Mt is O(1)), at a distance from the root of the size of the root.  (A grid from 2% to 98% of D stopped at -34,
where Mt is e^{37}, and the distribution function came out wrong by 11.)
"""
function place(logMt::L, grp::AbstractVector{ComplexF64}, oth::AbstractVector{ComplexF64}, side::Int, lo::Float64, hi::Float64, A::Float64, σ0::Float64;
               xq::Float64 = 0.0, tr::Float64 = 1.0, c0p::Float64 = 0.0) where {L}
    edge = side == 1 ? minimum(real, grp) : maximum(real, grp)
    foc = [0.0]
    isempty(oth) || push!(foc, side == 1 ? maximum(real, oth) : minimum(real, oth))
    # (above the unit root with c0 > 0 the branch point -Re(t)/c0 of the v-kernels, the left end of the admissible interval of
    # G+, is a focus of the grading cost too: the kernel of the density of the coefficient statistic has a double pole there,
    # int 2 v^3 e^{-(t + c0 s) v^2} dv = (t + c0 s)^{-2}, which the objective did not see, and with a fixed start at c from 1
    # to 5 the density was off by up to 0.12 (case DU, c = 5, g0 = 0.5, at the 90% point), the distribution function by up to
    # 1e-5, without a warning; with the cost within 1.3e-8 of DFKan's single inversion on a map of c from 1 to 5, g0 in {0.5,
    # 1, 3}, cases DK, DU, DT)
    side == 1 && c0p > 0 && push!(foc, -tr / c0p)
    # (xq > 0: the growth of the v-kernels of the upper tail to the left of the origin, max_v e^{-Re(t + c0 s) v^2 - Re(s) xq v}
    # = e^{Re(s)^2 xq^2/(4 Re(t + c0 s))}, c0p the largest positive c0, which makes it grow without bound toward the branch
    # point -Re(t)/c0)
    kbx(σ) = (d = tr + c0p * σ; d > 0 ? σ^2 * xq^2 / (4d) : Inf)
    kb(s) = side == 1 ? max(0.0, -A * real(s)) + (xq > 0 && real(s) < 0 ? kbx(real(s)) : 0.0) : 0.0
    function obj(σ)
        d = minimum(side * (real(s) - σ) for s in grp)
        ylo, yhi = ulevels(grp, σ, side, true)
        far = maximum(side * (real(s) - σ) for s in grp) + d
        m = -Inf
        pr(s) = (v = relogMt(logMt, s) + kb(s); m = max(m, isfinite(v) ? v : Inf))
        for y in (ylo, yhi, clamp(0.0, ylo, yhi)); pr(complex(σ, y)); end
        for s in grp
            pr(complex(σ, imag(s))); pr(complex(real(s), yhi)); pr(complex(real(s), ylo))
        end
        for s in oth; pr(complex(σ, clamp(imag(s), ylo, yhi))); end
        for y in (ylo, yhi)
            pr(complex(σ + side * far, y)); pr(complex(σ + side * 1e3 * max(far, 1.0), y))
        end
        for f in foc
            m += 0.5 * log(1 + d / max(abs(σ - f), 1e-300))
        end
        m
    end
    D = side == 1 ? edge - lo : hi - edge
    if PLACE_TWO_ENDED[]
        m = 1e-3 * D; nh = 12
        g1 = [m * (D / 2 / m)^((k - 1) / (nh - 1)) for k in 1:nh]
        ds = sort(unique([g1; D .- g1]))
    else
        ds = [0.02D * 49^((k - 1) / 23) for k in 1:24]        # the earlier grid, 2% to 98% of D
    end
    ng = length(ds)
    vals = [obj(edge - side * dd) for dd in ds]
    kbest = argmin(vals)
    a = ds[max(kbest - 1, 1)]; b = ds[min(kbest + 1, ng)]
    g = (sqrt(5) - 1) / 2
    x1 = b - g * (b - a); x2 = a + g * (b - a); f1 = obj(edge - side * x1); f2 = obj(edge - side * x2)
    for _ in 1:12
        if f1 < f2
            b = x2; x2 = x1; f2 = f1; x1 = b - g * (b - a); f1 = obj(edge - side * x1)
        else
            a = x1; x1 = x2; f1 = f2; x2 = a + g * (b - a); f2 = obj(edge - side * x2)
        end
    end
    σb = edge - side * (f1 < f2 ? x1 : x2)
    min(f1, f2) < obj(σ0) - 2 ? σb : σ0
end

"""
    tnode_values!(Fo, fo, CP, k, xs; c0 = 0, n, ntail, alpha)

The inner integrals I_F(t_k) (distribution function) and I_f(t_k) (density) for every x in xs at the node t_k of the
outer contour, in the form of the terms Im{I_F t'/t} and Im{I_f t'} of the outer trapezoidal rule.

    I_F(t) = (1/2 pi i) [ int_{G+} Mt (W+_F + (1 - e^{-a s})/s) ds + int_{G-} Mt (W-_F - 1/s) ds ],
    I_f(t) = (1/2 pi i) [ int_{G+} Mt W+_f ds + int_{G-} Mt W-_f ds ],

Mt = e^{a s} M(s,t) (a = 1/2 except with a fixed start, see `Model`), W+-_F = int_{V+-} X'(v) e^{-t v^2 - s X(v)} dv,
W+-_f = int_{V+-} 2 v^2 e^{-t v^2 - s X(v)} dv, X(v) = c0 v^2 + x v + a, V+ and V- the sets of v > 0 where X > 0 and
X < 0.  G+ is the U-shape around the positive group at sigma+ = alpha min Re s_+ and G- the one around the negative group
at sigma- = alpha max Re s_- (none when there is no negative root: Mt is then analytic to the left and the terms of G-
vanish).  When the positive group does not lie to the right of the origin (c > 0, far from the line), sigma+ is placed in
the gap (lo, hi) between the groups, hi = min Re s_+, lo = max(max Re s_-, hi - 2 max(|hi|, 1)), at hi - alpha (hi - lo),
and sigma- at lo + alpha (min(0, hi) - lo): the kernel (1 - e^{-a s})/s of G+ is entire, and only G- needs Re s < 0 (for
-1/s).  With c0 > 0 the semi-infinite piece of V+ needs Re(t + c0 s) > 0 on G+, so sigma+ stays right of -Re(t)/c0 (its
branch point -t/c0 is a focus of the grading); with c0 != 0 the rays reach at least the decay scale of the v-kernels
(`xscale`, the largest over xs).  With a fixed start log Mt gains g0^2 (cf + E1(s)) (`fixparts`, `fixE1`).
"""
function tnode_values!(Fo::AbstractVector{Float64}, fo::AbstractVector{Float64}, CP::CPath, k::Int, xs::Vector{Float64};
                       c0::Float64 = 0.0, n::Int = 24, ntail::Int = 40, alpha::Float64 = 0.5)
    tnode_values_m!(Fo, fo, zeros(length(xs)), zeros(length(xs)), CP, k, xs, c0, n, ntail, alpha)
    Fo, fo
end
"""log Mt(s) at one node of the outer contour: the algebraic factors (`logMtilde`) and, with a fixed start (F = FixT), the
term g0^2 (cf + E1(s)); a callable struct, so that the inner loops are specialized on the start"""
struct LogMt{F}
    rts::Vector{ComplexF64}; rp::Int; lP::ComplexF64
    fx::F; g2::Float64; Bf::Matrix{ComplexF64}
end
(L::LogMt{Nothing})(s::ComplexF64) = logMtilde(L.rts, L.rp, L.lP, s)
(L::LogMt{FixT})(s::ComplexF64) = logMtilde(L.rts, L.rp, L.lP, s) + L.g2 * (L.fx.cf + fixE1(L.fx, s, L.Bf))
"""Re log Mt(s) (for `place`, which needs only the modulus: log|.| = log(|.|^2)/2 in place of the complex logarithms)"""
function relogMt(L::LogMt{F}, s::ComplexF64) where {F}
    acc = real(L.lP)
    @inbounds for j in 1:L.rp; acc -= log(abs2(L.rts[j] - s)) / 4; end
    @inbounds for j in L.rp+1:length(L.rts); acc -= log(abs2(s - L.rts[j])) / 4; end
    F === FixT ? acc + L.g2 * real(L.fx.cf + fixE1(L.fx, s, L.Bf)) : acc
end

"""the terms of node k for every x (`tnode_inner!`); AFo and Afo receive the error weights of the terms (see there)"""
function tnode_values_m!(Fo::AbstractVector{Float64}, fo::AbstractVector{Float64}, AFo::AbstractVector{Float64}, Afo::AbstractVector{Float64},
                         CP::CPath, k::Int, xs::Vector{Float64}, c0::Float64, n::Int, ntail::Int, alpha::Float64, vpow::Int = 2;
                         vr = nothing, JFo = nothing, Jfo = nothing)
    Mo = CP.Mo; fx = CP.fx[k]; rts = CP.R[:, k]
    fx === nothing ? tnode_inner!(Fo, fo, AFo, Afo, CP, k, xs, c0, n, ntail, alpha, LogMt(rts, Mo.rp, CP.lP[k], nothing, 0.0, zeros(ComplexF64, 0, 0)), vpow; vr, JFo, Jfo) :
                     tnode_inner!(Fo, fo, AFo, Afo, CP, k, xs, c0, n, ntail, alpha, LogMt(rts, Mo.rp, CP.lP[k], fx, Mo.g0^2, fixbuf(Mo.r)), vpow; vr, JFo, Jfo)
end
"""the gap between the root groups of a node and the default abscissas of its U-shapes (`ushapes`): hi = min Re s_+, lo =
max Re s_- (-Inf without negative roots), sigma+ = alpha hi and sigma- = alpha lo when the gap contains the origin, else
in the gap (lo2, hi), lo2 = max(lo, hi - 2 max(|hi|, 1)) -> (hi, lo, sp, sm)"""
function uabscissas(gp::AbstractVector{ComplexF64}, gm::AbstractVector{ComplexF64}, rm::Int, alpha::Float64)
    hi = minimum(real, gp); lo = rm > 0 ? maximum(real, gm) : -Inf
    if hi > 0 && lo < 0
        sp = alpha * hi; sm = alpha * lo
    else
        lo2 = max(lo, hi - 2 * max(abs(hi), 1.0))
        sp = hi - alpha * (hi - lo2); sm = lo2 + alpha * (min(0.0, hi) - lo2)
    end
    hi, lo, sp, sm
end
"""sigma+ moved right of the branch point -Re(t)/c0 of every c0 > 0 of c0s, to -Re(t)/c0 + alpha (hi + Re(t)/c0) (`ushapes`;
NaN where a branch point is not left of hi, where `ushapes` refuses)"""
function ubranch(sp::Float64, hi::Float64, tr::Float64, c0s, alpha::Float64)
    for c0 in c0s
        c0 > 0 || continue
        lb = -tr / c0
        lb < hi || return NaN
        sp <= lb && (sp = lb + alpha * (hi - lb))
    end
    sp
end
"""the caps (hp, hm) of the panels of the segments of G+ and G- (`ushapes`): 20/A, and below c = FAR_CMIN = -60
20/min(A, 40/|sigma|).  (Far from the unit root the segments lie far from the origin, where the kernels are damped: on the
segment Re s = sigma the v-kernel e^{-s X} oscillates at the rate |X| with the modulus e^{-|sigma X|}, and e^{-A s} has the
modulus e^{-A sigma}, so that only rates up to 40/|sigma| matter.  At c = -300 in case DU the segment of G- lies at -45300
and the cap 20/A gave it 54608 nodes.)"""
function ucaps(c::Float64, sp::Float64, sm::Float64, A::Float64)
    hseg = 20 / A
    (c < FAR_CMIN && sp > 0 ? 20 / min(A, 40 / sp) : hseg), (c < FAR_CMIN && sm < 0 ? 20 / min(A, 40 / abs(sm)) : hseg)
end
"""the nodes of the narrow U-shapes of the zero and the stationary start at the abscissas sp, sm: the segments, at most hp
or hm per panel with n nodes, a lower bound of their count (within 1% of it beyond 1e5) -> (nodes, longest segment)"""
function unarrow(gp::AbstractVector{ComplexF64}, gm::AbstractVector{ComplexF64}, sp::Float64, sm::Float64, hp::Float64, hm::Float64, n::Int)
    nest = 0.0; hseg = 0.0
    for (g, σ, side, hh) in ((gp, sp, 1, hp), (gm, sm, -1, hm))
        isempty(g) && continue
        ylo, yhi = ulevels(g, σ, side, false)
        nest += (yhi - ylo) / hh * n; hseg = max(hseg, yhi - ylo)
    end
    nest, hseg
end
ushape_msg(start::Symbol, t::ComplexF64, nest::Float64, hseg::Float64) =
    "the U-shapes of the $start start at t = $t would need some $(round(nest, sigdigits = 3)) nodes (segments to $(round(hseg, sigdigits = 3)) long): a root runs off (USHAPE_MAXNODES)"
ushape_msg_fixed(t::ComplexF64, nest::Float64, hmx::Float64) =
    "the U-shapes of the fixed start at t = $t would need some $(round(Int, nest)) nodes (heights to $(round(hmx, sigdigits = 3))): the roots run off (USHAPE_MAXNODES)"
"""
    ushape_guard(CP, c0s, A, n, alpha, tol, xq)

The guard on the nodes of the U-shapes, before any U-shape is built or any inner integral evaluated (`contour_eval`,
`contour_eval_specs`; the arguments as there and in `ushapes`).  The outer contour carries the roots at its nodes, and with
them the count of `ushapes` that its guard compares with USHAPE_MAXNODES at every node t_k: with the zero and the
stationary start the segments of the narrow U-shapes at the abscissas of `ushapes` (`unarrow`); with a fixed start the wide
heights at the abscissas of `place`, which needs log Mt at the probe points of each U-shape, so that they are placed only
where an upper bound of the count exceeds the cap (every abscissa lies in the admissible interval of `place`, and every cap
of the segments is at least 20/A), a few nodes at most of an ordinary call.  The first node beyond USHAPE_MAXNODES at which
the transform has not fallen below the tolerance tol of the outer sum, |M_V(t_k) t'_k| max(1, 1/|t_k|) >= tol, stops the
call with the error of `ushapes`.  The nodes where the transform is below tol are left to the guard of `ushapes`, since the
outer sum stops before them unless its terms there are not small: d(r) = 1 + r on the vertical line at c = 0 needs more
than 1e6 nodes from |t| = 3e4 on, where |M_V| is 1e-36, and its sum stops at |t| = 1.6e4 with F right to 1e-4 (at c = -20
it goes on to the node beyond, which `ushapes` refuses).  The design (1 + r, 1{r <= 0.4}) on the vertical line has the
positive root 86030 - 764901i at t = 1 + 2749i and needs 1e6 nodes at |t| = 6300, where |M_V| is 1e-15, at every c from
-1 to 2: it took 3 to 4 s and 0.8 GB before `ushapes` refused, and is refused here before any inner integral.
"""
function ushape_guard(CP::CPath, c0s, A::Float64, n::Int, alpha::Float64, tol::Float64, xq::Float64)
    Mo = CP.Mo; rp = Mo.rp; r = Mo.r; fixed = Mo.start == :fixed
    cap = USHAPE_MAXNODES[]; lt = log(tol)
    @inbounds for k in eachindex(CP.t)
        gp = view(CP.R, 1:rp, k); gm = view(CP.R, rp+1:r, k)
        hi, lo, sp, sm = uabscissas(gp, gm, Mo.rm, alpha)
        lo < min(hi, 0.0) || continue                    # (`ushapes` refuses such a node with its own error)
        tr = real(CP.t[k])
        if fixed
            # (an upper bound of the count: sigma+ >= L, the lower end of the interval of `place`, sigma- <= min(hi, 0))
            L = Mo.rm > 0 ? lo : hi - 2 * max(abs(hi), 1.0)
            hmx = 0.0
            for z in gp; hmx = max(hmx, abs(imag(z)) + real(z) - L); end
            for z in gm; hmx = max(hmx, abs(imag(z)) + min(hi, 0.0) - real(z)); end
            FIXED_RULE[] == :heuristic && 2 * hmx * A / 20 * n <= cap && continue
        else
            sp = ubranch(sp, hi, tr, c0s, alpha)
            isnan(sp) && continue
            hp, hm = ucaps(CP.c, sp, sm, A)
            nest, hseg = unarrow(gp, gm, sp, sm, hp, hm, n)
            nest > cap || continue
        end
        # log |M_V(t_k)| = Re log Mt(0) (with a fixed start and g0^2 Re cf(t_k): E1(0) = 0)
        fx = CP.fx[k]
        lmv = real(logMtilde(view(CP.R, :, k), rp, CP.lP[k], 0.0im))
        fx !== nothing && (lmv += Mo.g0^2 * real(fx.cf))
        lmv + log(abs(CP.dt[k]) * max(1.0, 1 / abs(CP.t[k]))) >= lt || continue
        if fixed
            fx === nothing && continue
            # (the count of `ushapes`, with its placed abscissas)
            lm = LogMt(CP.R[:, k], rp, CP.lP[k], fx, Mo.g0^2, fixbuf(r))
            sp, sm = ufixed_place(CP, lm, c0s, gp, gm, hi, lo, sp, sm, tr, alpha, A, xq)
            sp = ubranch(sp, hi, tr, c0s, alpha)
            isnan(sp) && continue
            hp, hm = ucaps(CP.c, sp, sm, A)
            hmx = 0.0
            for z in view(CP.R, :, k); hmx = max(hmx, abs(imag(z)) + abs(real(z) - (real(z) > 0 ? sp : sm))); end
            nest = 2 * hmx / min(hp, hm) * n
            nest > cap && error(ushape_msg_fixed(CP.t[k], nest, hmx))
            continue
        end
        error(ushape_msg(Mo.start, CP.t[k], nest, hseg))
    end
    nothing
end
"""the abscissas (sp, sm) of the U-shapes of the fixed start at a node with the roots gp, gm (`ushapes`): the defaults sp, sm
of `uabscissas` replaced by `place` (or, with FIXED_RULE[] = :bound, by those of the proof)"""
function ufixed_place(CP::CPath, logMt::LogMt{FixT}, c0s, gp::AbstractVector{ComplexF64}, gm::AbstractVector{ComplexF64}, hi::Float64,
                      lo::Float64, sp::Float64, sm::Float64, tr::Float64, alpha::Float64, A::Float64, xq::Float64)
    Mo = CP.Mo; rp = Mo.rp
    # the fixed start: the abscissas where the largest |Mt| on the U-shape is smallest.  The objective also counts the
    # growth e^{sigma^2 xq^2/(4 Re(t + c0 s))} of the v-kernels of the upper tail (x up to xq > 0, c0 the largest positive
    # one) on G+ to the left of the origin (case DK at c = 1, g0 = 1: without it G+ went to -0.82, and F was off by 1e31 at
    # mean + 5 sd; at c = 0.5 the abscissa left of the origin is the right one).  Until November 2026 only above the unit
    # root: in case DK near it (c = 0 and -0.5) G+ went left of the origin in the upper tail as well, and one evaluation
    # was off by 3e-7 at x = 9.7 (g0 = 1), 7e-3 at x = 12 and 1e39 at x = 20, with c0 = 2 by 55 at x = 4 and with c0 = -2
    # by 9e-5 at x = 20, where F is 1 to 1e-12 (the default route takes the general engine for these arguments); with
    # the growth counted at every c (and with c0 > 0 in it) these points are within 1e-12 of the classical engine and the
    # bulk is unchanged; cases DU, DT, whose G+ stays right of the origin, are unchanged bit for bit
    xg = xq
    # (above the unit root with c0 > 0 the placement is searched right of the branch point -Re(t)/c0, which the
    # semi-infinite piece of V+ needs, and counts the growth of the v-kernels there with Re(t + c0 s) in place of Re t:
    # cases DU at c = 2, g0 = 1.5 and DT at c = 2, g0 = 3, both with c0 = 3, were off by 1e-2 and 1e87 at mean + 4 sd)
    lbp = -Inf
    if CP.c > 0
        for c0 in c0s; c0 > 0 && (lbp = max(lbp, -tr / c0)); end
    end
    sp0 = isfinite(lbp) ? max(sp, lbp + alpha * (hi - lbp)) : sp
    c0p = maximum((c0 for c0 in c0s if c0 > 0); init = 0.0)
    rule = FIXED_RULE[]
    if rule == :bound && CP.c <= 0 && hi > 0 && lo < 0
        # the abscissas of the proof: alpha times the real roots at the real point tau of the contour (the first node),
        # where Re Q_kappa <= Q_kappa(sigma, Re t) on the segment for Re t < t_*
        sp = alpha * minimum(real, view(CP.R, 1:rp, 1))
        Mo.rm > 0 && (sm = alpha * maximum(real, view(CP.R, rp+1:Mo.r, 1)))
        sp = min(sp, alpha * hi); Mo.rm > 0 && (sm = max(sm, alpha * lo))
    elseif Mo.rm > 0
        sp = place(logMt, gp, gm, 1, max(lo, lbp), hi, A, sp0; xq = xg, tr, c0p)
        sm = place(logMt, gm, gp, -1, lo, min(hi, 0.0), A, sm)
    else
        sp = place(logMt, gp, gm, 1, max(hi - 2 * max(abs(hi), 1.0), lbp), hi, A, sp0; xq = xg, tr, c0p)
    end
    sp, sm
end
"""
    ushapes(CP, k, logMt, c0s, rsp, rsm, alpha, A, n, ntail) -> (Sp, Mp, Ep, Sm, Mm, Em)

The U-shapes of node k of the outer contour (see `tnode_values!`): the nodes Sp of G+ and Sm of G-, Mt times the weights
there (Mp, Mm) and the factors 1 + |log Mt| of the error weights (Ep, Em).  c0s: the coefficients c0 != 0 of v^2 in the
parabolas X(v) of the kernels (each puts the branch point -t/c0 among the foci of the grading, and with c0 > 0 keeps the
abscissa of G+ right of -Re(t)/c0, where the semi-infinite piece of V+ needs Re(t + c0 s) > 0); rsp, rsm the reach of the
rays (the decay scale of the v-kernels, `xscale`); A the constant of the kernels (the growth max(0, -A Re s) of G+ to the
left of the origin in `place`, the rate of `nsplit` and the cap 20/A of the segment panels).
"""
function ushapes(CP::CPath, k::Int, logMt::LogMt{F}, c0s, rsp::Float64, rsm::Float64, alpha::Float64, A::Float64, n::Int,
                 ntail::Int; xq::Float64 = 0.0) where {F}
    Mo = CP.Mo
    t = CP.t[k]; tr = real(t)
    rts = logMt.rts; rp = Mo.rp
    gp = rts[1:rp]; gm = rts[rp+1:end]
    hi, lo, sp, sm = uabscissas(gp, gm, Mo.rm, alpha)
    lo < min(hi, 0.0) || error("the root groups are not separated by a line Re s = sigma < 0 at t = $t (lo = $lo, hi = $hi)")
    exp_p = ComplexF64[]; exp_m = ComplexF64[]
    if F === FixT
        sp, sm = ufixed_place(CP, logMt, c0s, gp, gm, hi, lo, sp, sm, tr, alpha, A, xq)
        rule = FIXED_RULE[]
        if rule != :heuristic
            B, _ = fixbound(logMt.fx, Mo.Dv, t)
            hb = min(1 + B, FIXED_HCAP[])
            lvp = ulevels(gp, sp, 1, true); levp = (min(lvp[1], -hb), max(lvp[2], hb))
            if Mo.rm > 0
                lvm = ulevels(gm, sm, -1, true); levm = (min(lvm[1], -hb), max(lvm[2], hb))
            end
            if rule == :bound
                levp = (-max(hb, -minimum(imag, gp) + 1), max(hb, maximum(imag, gp) + 1))
                Mo.rm > 0 && (levm = (-max(hb, -minimum(imag, gm) + 1), max(hb, maximum(imag, gm) + 1)))
            end
        end
    end
    for c0 in c0s
        c0 == 0 && continue
        if c0 > 0
            -tr / c0 < hi || error("no abscissa for G+ right of the branch point -t/c0 at t = $t")
            push!(exp_p, -t / c0)
        else
            push!(exp_m, -t / c0)
        end
    end
    sp = ubranch(sp, hi, tr, c0s, alpha)
    Sp = ComplexF64[]; Wp = ComplexF64[]
    hp, hm = ucaps(CP.c, sp, sm, A)
    lmf = F === FixT ? logMt : nothing
    # (the guard on the size of the U-shapes, which `ushape_guard` checks before the outer sum at the nodes where the
    # transform is not negligible; here at the other nodes, which the sum reaches where its terms are not small.  The wide
    # heights of the fixed start follow a root that runs off, as the root at 31700 - 1.6e6i of d(r) = 1 + r with a fixed
    # start on the vertical line at |t| = 2.8e4, whose U-shape had 3.9e6 nodes and exhausted the memory)
    if F === FixT
        hmx = 0.0
        for g in (gp, gm), z in g; hmx = max(hmx, abs(imag(z)) + abs(real(z) - (real(z) > 0 ? sp : sm))); end
        nest = 2 * hmx / min(hp, hm) * n
        nest > USHAPE_MAXNODES[] && error(ushape_msg_fixed(t, nest, hmx))
    else
        nest, hseg = unarrow(gp, gm, sp, sm, hp, hm, n)
        nest > USHAPE_MAXNODES[] && error(ushape_msg(Mo.start, t, nest, hseg))
    end
    lvp_ = F === FixT && FIXED_RULE[] != :heuristic ? levp : nothing
    lvm_ = F === FixT && FIXED_RULE[] != :heuristic && Mo.rm > 0 ? levm : nothing
    ushape!(Sp, Wp, gp, gm, sp, 1, n, ntail; extra = exp_p, rs = rsp, hmax = hp, lm = lmf, akern = A, levels = lvp_)
    Lp = [logMt(s) for s in Sp]
    Mp = [exp(l) * w for (l, w) in zip(Lp, Wp)]
    Ep = [1 + abs(l) for l in Lp]
    Sm = ComplexF64[]; Mm = ComplexF64[]; Em = Float64[]
    if Mo.rm > 0
        Wm = ComplexF64[]
        ushape!(Sm, Wm, gm, gp, sm, -1, n, ntail; extra = exp_m, rs = rsm, hmax = hm, lm = lmf, akern = A, levels = lvm_)
        Lm = [logMt(s) for s in Sm]
        Mm = [exp(l) * w for (l, w) in zip(Lm, Wm)]
        Em = [1 + abs(l) for l in Lm]
    end
    Sp, Mp, Ep, Sm, Mm, Em
end

"""
    tnode_inner!(Fo, fo, AFo, Afo, CP, k, xs, c0, n, ntail, alpha, logMt, vpow)

The terms of node k of the outer rule for every x in xs: Fo = Im{I_F t'/t}, fo = Im{I_f t'} (see `tnode_values!`), and
their error weights AFo = |t'/t| sum_j (1 + |log Mt(s_j)|) |Mt(s_j) w_j K_F(s_j)| and Afo likewise with |t'| and K_f, the
sums over the nodes s_j of the U-shapes with weights w_j and kernels K_F, K_f: eps times the weight bounds the rounding
error of the inner sum, the factor 1 + |log Mt| that of exp(log Mt) when its logarithm is large.
"""
function tnode_inner!(Fo::AbstractVector{Float64}, fo::AbstractVector{Float64}, AFo::AbstractVector{Float64}, Afo::AbstractVector{Float64},
                      CP::CPath, k::Int, xs::Vector{Float64}, c0::Float64, n::Int, ntail::Int, alpha::Float64, logMt::LogMt{F}, vpow::Int = 2;
                      vr = nothing, JFo = nothing, Jfo = nothing) where {F}
    Mo = CP.Mo; A = Mo.a
    t = CP.t[k]; dt = CP.dt[k]; tr = real(t)
    rsp = 0.0; rsm = 0.0
    if c0 != 0
        for x in xs
            plus, minus = xpieces(x, c0, A)
            rsp = max(rsp, xscale(tr, c0, x, A, active(plus, tr)))
            rsm = max(rsm, xscale(tr, c0, x, A, active(minus, tr)))
        end
        rsp = min(rsp, 1e4); rsm = min(rsm, 1e4)
    end
    Sp, Mp, Ep, Sm, Mm, Em = ushapes(CP, k, logMt, (c0,), rsp, rsm, alpha, A, n, ntail; xq = max(0.0, maximum(xs; init = 0.0)))
    # the factors of the kernels that do not depend on x, once per node (`ctr_moments_pre`)
    sqp = [sqrt(c0 == 0 ? t : t + c0 * s) for s in Sp]; iqp = [1 / q for q in sqp]; eAp = [exp(-A * s) for s in Sp]
    psp = [abs2(s) < 1e-16 ? A - A * A * s / 2 : -expm1(-A * s) / s for s in Sp]
    sqm = [sqrt(c0 == 0 ? t : t + c0 * s) for s in Sm]; iqm = [1 / q for q in sqm]; eAm = [exp(-A * s) for s in Sm]
    for (ix, x) in enumerate(xs)
        plus, minus = xpieces(x, c0, A)
        plus = active(plus, tr); minus = active(minus, tr)
        IF = zero(ComplexF64); If = zero(ComplexF64); aF = 0.0; af = 0.0
        JF = zero(ComplexF64); Jf = zero(ComplexF64)
        v1 = vr === nothing ? 0.0 : vr[ix]
        # (with Filon's rule, `filon_roots`: the sign of the root v1 as an end of the pieces of each side)
        sgp = 0; sgm = 0
        if v1 > 0
            for (a, b) in plus; b == v1 && (sgp -= 1); a == v1 && (sgp += 1); end
            for (a, b) in minus; b == v1 && (sgm -= 1); a == v1 && (sgm += 1); end
        end
        for j in eachindex(Sp)
            s = Sp[j]; m = Mp[j]
            ps = psp[j]
            if vpow == 2
                M0, M1, M2 = isempty(plus) ? (0.0im, 0.0im, 0.0im) : ctr_moments_pre(t, s, c0, x, A, plus, sqp[j], iqp[j], eAp[j])
                tF = m * (x * M0 + 2c0 * M1 + ps); tf = m * 2M2
            else
                M0, M1, M2, M3 = isempty(plus) ? (0.0im, 0.0im, 0.0im, 0.0im) : ctr_moments3(t, s, c0, x, A, plus)
                tF = m * (x * M0 + 2c0 * M1 + ps); tf = m * 2M3
            end
            if sgp != 0
                E = sgp * exp(-s * (c0 * v1 * v1 + x * v1 + A))
                if vpow == 2
                    e0, e1, e2 = rend_part(v1, sqp[j], iqp[j], x * s, E); Jf += 2m * e2
                else
                    _, e0, e1, _, e3 = ctr_rend3(v1, sqp[j], x * s, E); Jf += 2m * e3
                end
                JF += m * (x * e0 + 2c0 * e1)
            end
            IF += tF; If += tf; aF += Ep[j] * fabs(tF); af += Ep[j] * fabs(tf)
        end
        for j in eachindex(Sm)
            s = Sm[j]; m = Mm[j]
            if vpow == 2
                M0, M1, M2 = isempty(minus) ? (0.0im, 0.0im, 0.0im) : ctr_moments_pre(t, s, c0, x, A, minus, sqm[j], iqm[j], eAm[j])
                tF = m * (x * M0 + 2c0 * M1 - 1 / s); tf = m * 2M2
            else
                M0, M1, M2, M3 = isempty(minus) ? (0.0im, 0.0im, 0.0im, 0.0im) : ctr_moments3(t, s, c0, x, A, minus)
                tF = m * (x * M0 + 2c0 * M1 - 1 / s); tf = m * 2M3
            end
            if sgm != 0
                E = sgm * exp(-s * (c0 * v1 * v1 + x * v1 + A))
                if vpow == 2
                    e0, e1, e2 = rend_part(v1, sqm[j], iqm[j], x * s, E); Jf += 2m * e2
                else
                    _, e0, e1, _, e3 = ctr_rend3(v1, sqm[j], x * s, E); Jf += 2m * e3
                end
                JF += m * (x * e0 + 2c0 * e1)
            end
            IF += tF; If += tf; aF += Em[j] * fabs(tF); af += Em[j] * fabs(tf)
        end
        if v1 > 0
            eb = exp(-t * v1 * v1)
            IF -= eb * JF; If -= eb * Jf
            JFo[ix] = JF * dt / t; Jfo[ix] = Jf * dt
        end
        Fo[ix] = imag(IF * dt / t); fo[ix] = imag(If * dt)
        AFo[ix] = aF * abs(dt / t); Afo[ix] = af * abs(dt)
    end
    # the sum of |Mt w| (for the estimate of the rounding error of the sums: the kernels are at most of order one)
    (sum(abs, Mp; init = 0.0) + sum(abs, Mm; init = 0.0)) * abs(dt) * max(1 / abs(t), 1.0)
end

# ---------- events bounded by parabolas in sqrt(V): the line U - aV <= b, the ADFmax statistic ----------
"""
    XSpec(c0, q, A; vmax = Inf, w = 1, dc0 = 0, dq = 0, dA = 0, bnd = true)

One term w P(R <= X(sqrt V), sqrt V <= vmax) of an event, with R = U + a (a the shift of Mt = e^{a s} M, `Model`) and the
parabola X(v) = c0 v^2 + q v + A; (dc0, dq, dA) are the derivatives of its coefficients in the parameter whose density is
wanted.  The contour integral of such a term is that of the t-ratio (`tnode_values!`) with two changes, both from the
same integration by parts in v.  First, the constant A of X need not be the shift a of Mt (the t-ratio and the coefficient
statistic have A = a; the line U - aV <= b has X(v) = a v^2 + a + b): the term P(R <= A under the tilt) is split at R = 0,
(1 - e^{-As})/s on G+ and -1/s on G- for A > 0, -e^{-As}/s on G- for A < 0, -1/s on G- for A = 0.  Second, with a finite
vmax the v-pieces end at vmax and the boundary term -e^{-t vmax^2} P(R <= X(vmax) under the tilt) is added (bnd = true).
The density kernel is 2 v dX/dparameter = 2 (dc0 v^3 + dq v^2 + dA v); the term of a moving vmax is left out, which is
right when the terms of the event have the same X(vmax) and weights that sum to zero there (as the two parabolas of
ADFmax, where bnd = false drops their boundary terms, which cancel exactly).
"""
struct XSpec
    c0::Float64; q::Float64; A::Float64
    vmax::Float64; w::Float64
    dc0::Float64; dq::Float64; dA::Float64
    bnd::Bool
end
XSpec(c0::Real, q::Real, A::Real; vmax::Real = Inf, w::Real = 1.0, dc0::Real = 0.0, dq::Real = 0.0, dA::Real = 0.0, bnd::Bool = true) =
    XSpec(Float64(c0), Float64(q), Float64(A), Float64(vmax), Float64(w), Float64(dc0), Float64(dq), Float64(dA), bnd)

"""the pieces of (0, vmax) on which X(v) = c0 v^2 + q v + A is positive (plus) and negative (minus), split at its positive
roots (computed without cancellation; a double root is no split), for any sign of A"""
function xpieces_gen(q::Float64, c0::Float64, A::Float64, vmax::Float64 = Inf)
    r = Float64[]
    if c0 == 0
        if q != 0
            v = -A / q; v > 0 && push!(r, v)
        end
    else
        dsc = q * q - 4c0 * A
        if dsc > 0
            qq = -(q + (q >= 0 ? 1.0 : -1.0) * sqrt(dsc)) / 2
            for y in (qq / c0, A / qq)
                (y > 0 && isfinite(y)) && push!(r, y)
            end
            sort!(r)
        end
    end
    bk = [0.0; [y for y in r if y < vmax]; vmax]
    plus = Tuple{Float64,Float64}[]; minus = Tuple{Float64,Float64}[]
    for j in 1:length(bk)-1
        a = bk[j]; b = bk[j+1]
        b > a || continue
        m = isfinite(b) ? (a + b) / 2 : a + 1.0
        X = c0 * m * m + q * m + A
        X > 0 ? push!(plus, (a, b)) : X < 0 && push!(minus, (a, b))
    end
    plus, minus
end
"""the constant terms of the kernels on G+ and G-: the split of P(R <= A under the tilt) at R = 0 (see `XSpec`)"""
@inline function psi_pm(s::ComplexF64, A::Float64)
    if A > 0
        return (abs2(s) < 1e-16 ? A - A * A * s / 2 : -expm1(-A * s) / s), -1 / s
    elseif A < 0
        return zero(ComplexF64), -exp(-A * s) / s
    end
    zero(ComplexF64), -1 / s
end

"""the terms of node k for events made of the parabola terms specs[i] (`XSpec`), i = 1..length(specs), as `tnode_inner!`"""
function tnode_specs!(Fo::AbstractVector{Float64}, fo::AbstractVector{Float64}, AFo::AbstractVector{Float64}, Afo::AbstractVector{Float64},
                      CP::CPath, k::Int, specs::Vector{Vector{XSpec}}, n::Int, ntail::Int, alpha::Float64, logMt::LogMt{F};
                      vr = nothing, JFo = nothing, Jfo = nothing) where {F}
    Mo = CP.Mo
    t = CP.t[k]; dt = CP.dt[k]; tr = real(t)
    # the pieces of every term, active on this node (e^{-Re(t) a^2} not negligible at the lower end)
    pcs = [[(sp, active.(xpieces_gen(sp.q, sp.c0, sp.A, sp.vmax), tr)...) for sp in S] for S in specs]
    c0s = Float64[]; rsp = 0.0; rsm = 0.0; rAp = 0.0; rAm = 0.0; ak = Mo.a; d3 = false
    for S in pcs, (sp, plus, minus) in S
        ak = max(ak, abs(sp.A)); d3 |= sp.dc0 != 0
        # (the kernel psi(s, A) of every term carries e^{-A s}, on G+ for A > 0 and on G- for A < 0, which decays only on
        # the scale 1/|A|: at the edge of the support (b near -a, A small) far beyond the roots, where the rays ended.
        # P(U <= b) in case DK was off by 1.9e-7 at b = -0.499, P(U - aV <= b) at a = 0.1 by 1.9e-8 at b = -0.4999 (`xscale`
        # takes the scale of the v-kernel, 1/max|X|, which is short for a convex X).  The rays reach 1/|A| for every
        # term, beyond the cap 1e4 of `xscale`: at b = -0.49999 (A = 1e-5) the cap left an error of 7e-12)
        if sp.A != 0
            sp.A > 0 ? (rAp = max(rAp, 1 / sp.A)) : (rAm = max(rAm, -1 / sp.A))
        end
        sp.c0 == 0 && continue
        sp.c0 in c0s || push!(c0s, sp.c0)
        rsp = max(rsp, xscale(tr, sp.c0, sp.q, sp.A, plus)); rsm = max(rsm, xscale(tr, sp.c0, sp.q, sp.A, minus))
    end
    rsp = max(min(rsp, 1e4), min(rAp, 1e8)); rsm = max(min(rsm, 1e4), min(rAm, 1e8))
    Sp, Mp, Ep, Sm, Mm, Em = ushapes(CP, k, logMt, c0s, rsp, rsm, alpha, ak, n, ntail;
                                     xq = max(0.0, maximum((sp.q for S in specs for sp in S); init = 0.0)))
    for (ix, S) in enumerate(pcs)
        IF = zero(ComplexF64); If = zero(ComplexF64); aF = 0.0; af = 0.0
        JF = zero(ComplexF64); Jf = zero(ComplexF64)
        v1 = vr === nothing ? 0.0 : vr[ix]
        for side in (1, -1)
            Ss = side == 1 ? Sp : Sm; Ms = side == 1 ? Mp : Mm; Es = side == 1 ? Ep : Em
            for j in eachindex(Ss)
                s = Ss[j]; m = Ms[j]
                tF = zero(ComplexF64); tf = zero(ComplexF64)
                for (sp, plus, minus) in S
                    pc = side == 1 ? plus : minus
                    c0 = sp.c0; q = sp.q
                    if v1 > 0
                        # the part of the kernel at the end v1 (the root of X; `filon_roots`) without its factor e^{-t v1^2}
                        # (X(v1) = 0): the terms of `ctr_moments` at v1, with the sign of v1 as an upper (-) or lower (+) end
                        sg = 0
                        for (a, b) in pc; b == v1 && (sg -= 1); a == v1 && (sg += 1); end
                        if sg != 0
                            sa = sqrt(c0 == 0 ? t : t + c0 * s)
                            e0, e1, e2 = rend_part(v1, sa, 1 / sa, q * s, sg * exp(-s * (c0 * v1 * v1 + q * v1 + sp.A)))
                            JF += m * sp.w * (q * e0 + 2c0 * e1); Jf += m * sp.w * 2 * (sp.dq * e2 + sp.dA * e1)
                        end
                    end
                    if isempty(pc)
                        M0 = M1 = M2 = M3 = zero(ComplexF64)
                    elseif d3
                        M0, M1, M2, M3 = ctr_moments3(t, s, c0, q, sp.A, pc)
                    else
                        M0, M1, M2 = ctr_moments(t, s, c0, q, sp.A, pc); M3 = zero(ComplexF64)
                    end
                    pp, pm = psi_pm(s, sp.A)
                    kF = q * M0 + 2c0 * M1 + (side == 1 ? pp : pm)
                    if sp.bnd && isfinite(sp.vmax)
                        vm = sp.vmax; bp, bm = psi_pm(s, c0 * vm * vm + q * vm + sp.A)
                        kF -= exp(-t * vm * vm) * (side == 1 ? bp : bm)
                    end
                    tF += sp.w * kF
                    tf += sp.w * 2 * (sp.dc0 * M3 + sp.dq * M2 + sp.dA * M1)
                end
                tF *= m; tf *= m
                IF += tF; If += tf; aF += Es[j] * fabs(tF); af += Es[j] * fabs(tf)
            end
        end
        if v1 > 0
            # (the remainder, and the part at the root for Filon's rule, `contour_sum_filon`)
            eb = exp(-t * v1 * v1)
            IF -= eb * JF; If -= eb * Jf
            JFo[ix] = JF * dt / t; Jfo[ix] = Jf * dt
        end
        Fo[ix] = imag(IF * dt / t); fo[ix] = imag(If * dt)
        AFo[ix] = aF * abs(dt / t); Afo[ix] = af * abs(dt)
    end
    nothing
end
function tnode_specs_m!(Fo, fo, AFo, Afo, CP::CPath, k::Int, specs::Vector{Vector{XSpec}}, n::Int, ntail::Int, alpha::Float64;
                        vr = nothing, JFo = nothing, Jfo = nothing)
    Mo = CP.Mo; fx = CP.fx[k]; rts = CP.R[:, k]
    fx === nothing ? tnode_specs!(Fo, fo, AFo, Afo, CP, k, specs, n, ntail, alpha, LogMt(rts, Mo.rp, CP.lP[k], nothing, 0.0, zeros(ComplexF64, 0, 0)); vr, JFo, Jfo) :
                     tnode_specs!(Fo, fo, AFo, Afo, CP, k, specs, n, ntail, alpha, LogMt(rts, Mo.rp, CP.lP[k], fx, Mo.g0^2, fixbuf(Mo.r)); vr, JFo, Jfo)
end
"""
    contour_eval_specs(CP, specs; n, ntail, alpha = 0.5, tol = 1e-17, warn = true) -> (F, f, nodes, (F = eF, f = ef))

`contour_eval` for events made of parabola terms (`XSpec`): F[i] = sum over the terms sp of specs[i] of sp.w P(R <=
X_sp(sqrt V), sqrt V <= sp.vmax) and f[i] its derivative in the parameter of the terms, by the contour integral on the
outer contour CP, with the estimates of the rounding error and the warnings of `contour_eval`.
"""
function contour_eval_specs(CP::CPath, specs::Vector{Vector{XSpec}}; n = nothing, ntail = nothing, alpha::Float64 = 0.5,
                            tol::Float64 = 1e-17, warn::Bool = true, filon::Bool = true)
    fixed = CP.Mo.start == :fixed
    n = n === nothing ? (fixed ? 48 : 24) : Int(n); ntail = ntail === nothing ? (fixed ? 64 : 40) : Int(ntail)
    # (the guard on the nodes of the U-shapes, with the c0 and the constant of the kernels of `tnode_specs!`)
    c0s = Float64[]; ak = CP.Mo.a
    for S in specs, sp in S
        ak = max(ak, abs(sp.A))
        (sp.c0 == 0 || sp.c0 in c0s) || push!(c0s, sp.c0)
    end
    ushape_guard(CP, c0s, ak, n, alpha, tol, max(0.0, maximum((sp.q for S in specs for sp in S); init = 0.0)))
    vr = filon && isnan(CP.h) ? filon_roots(CP, specs) : nothing
    if vr !== nothing
        return contour_sum_filon(CP, length(specs), (gF, gf, aF, af, jF, jf, k) -> tnode_specs_m!(gF, gf, aF, af, CP, k, specs, n, ntail, alpha; vr, JFo = jF, Jfo = jf),
                                 tol, warn, vr .^ 2)
    end
    if warn && isnan(CP.h)
        b2 = 0.0; neg = false
        for S in specs, sp in S
            plus, minus = xpieces_gen(sp.q, sp.c0, sp.A, sp.vmax)
            for (a, b) in (plus..., minus...)
                (isfinite(b) && b < sp.vmax) || continue
                b2 = max(b2, b * b); sp.c0 < 0 && (neg = true)
            end
        end
        line_edge_warn(CP, b2, neg)
    end
    contour_sum(CP, length(specs), (gF, gf, aF, af, k) -> tnode_specs_m!(gF, gf, aF, af, CP, k, specs, n, ntail, alpha), tol, warn, NaN)
end

"""
    contour_eval(CP, xs; c0 = 0, n, ntail, alpha = 0.5, tol = 1e-17, warn = true) -> (F, f, nodes, (F = eF, f = ef))

The distribution function F(x) = P(tau <= x) and the density f(x) of tau = (U - c0 V)/sqrt(V) at the points xs by the
contour integral: the trapezoidal rule in u >= 0 on the hyperbola CP (which carries the design and its start), F =
(h/pi) sum' Im{I_F(t_k) t'(u_k)/t_k} and f likewise, the outer nodes processed in chunks of 32 until a chunk whose terms
all stay below tol.  vpow = 3 replaces the density kernel 2 v^2 by 2 v^3: at x = 0 the derivative of F in c0, the
density of the coefficient statistic (`coef_pdf`).  n and ntail are the nodes per panel and on the tails of the rays: 24 and 40, with a fixed start 48
and 64 (the essential singularities).  The fourth output holds, for each x, the estimates eF and ef of the rounding error:
eps times the sums over all nodes (inner and outer, with the weights) of (1 + |log Mt|) |term|, the terms with their
kernels (`tnode_inner!`).  A warning when the largest of them exceeds WARN_TOL[] = 1e-4 (the transform is then large on
the paths and the sums cancel), when a value is not finite, or when the outer contour ends (at |t - tau| = 1e5) with
terms above 1e-10.  Measured against DFLimit with a fixed start, the actual error is 0.005 to 0.03 times the estimate
where rounding dominates (badly placed paths) and of the order of the estimate or below elsewhere
(`scripts/check_fixed_far.jl`).  The estimate covers rounding and cancellation, not the discretization error, which the
panel rules control (`nsplit`, the grading toward roots and origin).  On the vertical line the parts of the kernels at
the root of the parabola, which carry e^{-t v^2}, go to Filon's rule (`filon_roots_x`, `contour_sum_filon`; filon =
false: the line's Gauss-Legendre rule for them as well, as before).  contour_eval(P, CP, xs; ...) is the same (P is
not used).
"""
function contour_eval(CP::CPath, xs::Vector{Float64}; c0::Real = 0.0, n = nothing, ntail = nothing, alpha::Float64 = 0.5,
                      tol::Float64 = 1e-17, warn::Bool = true, vpow::Int = 2, filon::Bool = true)
    fixed = CP.Mo.start == :fixed
    n = n === nothing ? (fixed ? 48 : 24) : Int(n); ntail = ntail === nothing ? (fixed ? 64 : 40) : Int(ntail)
    c0 = Float64(c0)
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))     # (+-Inf gave NaN, NaN an InexactError)
    # (the guard counts the nodes of the present rule's U-shapes, so that it refuses what it refused before November 2026
    # whatever the rule: with the 16 nodes of the thin rule the count of the design (1 + r, 1{r <= 0.4}) stays below the
    # cap where the transform matters, and `ushapes` refused it only after 3 s)
    ushape_guard(CP, (c0,), CP.Mo.a, max(n, fixed ? 48 : 24), alpha, tol, max(0.0, maximum(xs; init = 0.0)))
    vr = filon && isnan(CP.h) ? filon_roots_x(CP, xs, c0) : nothing
    if vr !== nothing
        return contour_sum_filon(CP, length(xs), (gF, gf, aF, af, jF, jf, k) -> tnode_values_m!(gF, gf, aF, af, CP, k, xs, c0, n, ntail, alpha, vpow; vr, JFo = jF, Jfo = jf),
                                 tol, warn, vr .^ 2)
    end
    if warn && isnan(CP.h)
        b2 = 0.0
        for x in xs
            plus, minus = xpieces(x, c0, CP.Mo.a)
            for (a, b) in (plus..., minus...); isfinite(b) && (b2 = max(b2, b * b)); end
        end
        line_edge_warn(CP, b2, c0 < 0)
    end
    contour_sum(CP, length(xs), (gF, gf, aF, af, k) -> tnode_values_m!(gF, gf, aF, af, CP, k, xs, c0, n, ntail, alpha, vpow), tol, warn, c0)
end
"""
    line_edge_warn(CP, b2)

The warning of the vertical line (c > 0, or GLS detrending with a small |cbar|, beyond the steepest hyperbola) for a
parabola X(v) with a positive root b, b^2 = b2, where Filon's rule does not take the part at the root (`filon_roots`,
`filon_roots_x`: it does for the linear form and for the t-ratio and the coefficient statistic with one root; the
warning remains for ADFmax and for two roots, c0 > 0 with x < 0).  The text below describes the
line without Filon's rule (`filon = false`).  The kernels of the pieces of v that end at b carry e^{-t b^2} (X(b) = 0), so that the outer integrand oscillates in
w at the rate b^2, damped on the line only by e^{-tau b^2}, and tau = min(1, t_*/2) is tiny above the unit root (1.1e-3 at
c = 5 in case DK).  With c0 < 0 the root is large in the bulk (b close to x/|c0|: b^2 = 986 at c = 5, c0 = -3, x = 95), and
the panels in w, which `line_refine` cuts to the stationary point of the v-kernel only, carry up to hundreds of these
oscillations: in cases DK, DU, DT at c = 5, c0 = -3 the rule variants differed by up to 1e-4 and F from DFLimit's line
integral by 3e-5 to 6e-5.  Resolving them would take some 1e6 nodes; the cure would be to split off the parts carrying
e^{-t b^2} and integrate them on their own contour (as DFLimit's line integral does with its Filon parts), which is not
implemented.  With c0 < 0 the warning comes for every root that e^{-tau b^2} does not damp (tau b^2 < 40): in cases DK,
DU, DT at c = 4 to 7, c0 = -0.5 to -5, the rule variants differ by 1e-5 to 6e-4.  With GLS detrending at cbar = -2 (the power envelope at c = -2) the line is used at c = 0 with tau = 1, and
P(U - aV <= b) in the upper tail (b^2 = 3.75) differs by up to 1e-4 between the rule variants; the critical values of the
envelope lie in the lower tail and agree with DFLimit to 5e-9.  The warning comes when the panels on which b^2 times the panel's length exceeds 30 carry an outer integral
of more than 1e-6 by the estimate |M_V(t)| e^{-tau b^2} L/(pi |t|) (L the panel's length), which they then cannot resolve.
"""
function line_edge_warn(CP::CPath, b2::Float64, c0neg::Bool = false)
    b2 > 0 || return nothing
    if c0neg && CP.tau * b2 < 40
        # (with c0 < 0 the estimate below understates the part at the root: in cases DK, DU, DT at c = 4 to 7 with c0 from
        # -0.5 to -5 the rule variants differed by 1e-5 to 6e-4 also where it stayed below 1e-6; so the line warns for every
        # root that e^{-tau b^2} does not damp)
        @warn "contour_eval: on the vertical line Re t = $(round(CP.tau, sigdigits = 3)) (c = $(CP.c)$(isnan(CP.Mo.P.cbar) ? "" : ", GLS detrending"): no hyperbola keeps the positive roots right of the origin) with c0 < 0 the parabola has a root b with b^2 = $(round(b2, sigdigits = 3)), whose part of the outer integrand the line does not resolve: the result may be off by 1e-5 to 1e-3 (see `line_edge_warn`)"
        return nothing
    end
    nq = linenq(CP); np = length(CP.t) ÷ max(nq, 1)
    (nq > 0 && np * nq == length(CP.t)) || (nq = 24; np = length(CP.t) ÷ 24)
    w = imag.(CP.t); est = 0.0; worst = 0.0
    for k in 1:np
        j1 = (k - 1) * nq + 1; j2 = k * nq
        L = k == 1 ? 2w[j2] : (w[j2] - w[j1]) * 1.02          # (about the panel's length)
        b2 * L > 30 || continue
        lmv = max(real(logMtilde(view(CP.R, :, j1), CP.Mo.rp, CP.lP[j1], 0.0im)), real(logMtilde(view(CP.R, :, j2), CP.Mo.rp, CP.lP[j2], 0.0im)))
        est += exp(lmv - CP.tau * b2) * L / (pi * max(abs(CP.t[j1]), CP.tau))
        worst = max(worst, b2 * L / 30)
    end
    est > 1e-6 && @warn "contour_eval: on the vertical line Re t = $(round(CP.tau, sigdigits = 3)) (c = $(CP.c)$(isnan(CP.Mo.P.cbar) ? "" : ", GLS detrending"): no hyperbola keeps the positive roots right of the origin) the parabola has a root b with b^2 = $(round(b2, sigdigits = 3)), whose part of the outer integrand (of size $(round(est, sigdigits = 2))) oscillates up to $(round(worst, sigdigits = 2)) times faster than its panels resolve: the result may be off by more than 1e-5 (see `line_edge_warn`)"
    nothing
end
# ---------- Filon's rule on the vertical line ----------
"""
    filon_roots(CP, specs) -> vr (vector) or nothing

On the vertical line CP (Gauss-Legendre panels, `linepanels`), the events whose parts at a root of the parabola go to
Filon's rule (`contour_sum_filon`): an event of one term with q = 0 (the linear form U - aV <= b, `lin_cdf`, `lin_pdf`,
`lin_quantile`), c0 != 0, no derivative in c0, vmax = Inf: vr[i] its positive root of X(v) = c0 v^2 + A (0 without one).
The kernels of the pieces that end at the root carry e^{-t vr^2} exactly (X(vr) = 0), which oscillates on the line at the
rate vr^2, damped only by e^{-tau vr^2}: with tau = min(1, t_*/2) tiny above the unit root (1.1e-3 at c = 5 in case DK)
the doubling panels of the line do not resolve it, and P(U - aV <= b) was off by up to 4e-3 in case DK at c = 4 to 7
(against the classical code; also with GLS detrending at small |cbar|, by 2e-5 to 1e-4).  Without the term in v (q = 0)
the rest of those kernels is smooth along the line: z = sqrt(t + c0 s) vr stays in Re z >= 0, so that no reflection of
`ctr_rend` (whose Gaussian part would carry no e^{-t vr^2}) splits it.  nothing: every event as before (another event).
"""
function filon_roots(CP::CPath, specs::Vector{Vector{XSpec}})
    linepanels(CP) === nothing && return nothing
    vr = zeros(length(specs)); found = false
    for (i, S) in enumerate(specs)
        roots = Float64[]
        for sp in S
            plus, minus = xpieces_gen(sp.q, sp.c0, sp.A, sp.vmax)
            for (a, b) in (plus..., minus...); (isfinite(b) && b > 0) && push!(roots, b); end
        end
        isempty(roots) && continue
        (length(S) == 1 && S[1].q == 0 && S[1].c0 != 0 && S[1].dc0 == 0 && !isfinite(S[1].vmax) && length(unique(roots)) == 1) || return nothing
        vr[i] = roots[1]; found = true
    end
    found ? vr : nothing
end
"""
    filon_roots_x(CP, xs, c0) -> vr (vector) or nothing

`filon_roots` for the t-ratio and the coefficient statistic (`contour_eval`): vr[i] the positive root of X(v) = c0 v^2 + x_i v + a where it
has one (0 without), nothing where an X has two (c0 > 0 with x < 0) or CP is not a line of Gauss-Legendre panels.  With
x != 0 the end terms at the root are reflected where Re z < 0 (`ctr_rend`), and the Gaussian that the reflection adds
carries no e^{-t vr^2}: their split is smooth only where the reflection does not switch on a panel; at the switch
z = sqrt(t + c0 s) vr + x s/(2 sqrt(t + c0 s)) is imaginary, and the jump of the split, e^{z^2}, is negligible once |z|
is large, as on the panels where the oscillation is fast (beta L >= FILON_OMIN[]; the slower panels keep the
Gauss-Legendre rule, `contour_sum_filon`).
"""
function filon_roots_x(CP::CPath, xs::Vector{Float64}, c0::Float64)
    c0 == 0 && return nothing
    linepanels(CP) === nothing && return nothing
    vr = zeros(length(xs)); found = false
    for (i, x) in enumerate(xs)
        isfinite(x) || continue
        plus, minus = xpieces(x, c0, CP.Mo.a)
        roots = unique([b for (a, b) in (plus..., minus...) if isfinite(b) && b > 0])
        isempty(roots) && continue
        length(roots) == 1 || return nothing
        vr[i] = roots[1]; found = true
    end
    found ? vr : nothing
end
"""
    linepanels(CP) -> (nq, wa, L) or nothing

The Gauss-Legendre panels [wa[p], wa[p] + L[p]] of the line CP (`linecpath`, with its default or refined panels), each
with the nq nodes of `glrule` (24, or 32 in the variant of `verify_eval`), from the nodes themselves (the first and the
last of each panel); nothing if the nodes are not such panels.
"""
function linepanels(CP::CPath)
    isnan(CP.h) || return nothing
    for nq in (24, 32, 16, 48)
        r = linepanels(CP, nq)
        r === nothing || return r
    end
    nothing
end
function linepanels(CP::CPath, nq::Int)
    K = length(CP.t); np = K ÷ nq
    (np > 0 && np * nq == K) || return nothing
    z = glrule(nq)[1]; w = imag.(CP.t)
    wa = zeros(np); L = zeros(np)
    for p in 1:np
        j1 = (p - 1) * nq + 1; j2 = p * nq
        L[p] = (w[j2] - w[j1]) / (z[nq] - z[1]); wa[p] = w[j1] - L[p] * z[1]
        all(abs(w[j1+q-1] - (wa[p] + L[p] * z[q])) <= 1e-9 * (abs(wa[p]) + L[p]) for q in 1:nq) || return nothing
    end
    nq, wa, L
end
"""the data of Filon's rule on a panel of nq Gauss-Legendre nodes z on [0, 1]: the barycentric weights lam and the
derivatives of orders 0..nq-1 of the interpolating polynomial at x = 0 (D0) and x = 1 (D1), as rows applied to its values
(from the inverse of the Vandermonde matrix in 512-bit arithmetic)"""
struct FilonData
    z::Vector{Float64}; lam::Vector{Float64}; D0::Matrix{Float64}; D1::Matrix{Float64}
end
const FILON = Dict{Int,FilonData}()
const FILONLOCK = ReentrantLock()
function filondata(nq::Int)
    lock(FILONLOCK) do
        get!(FILON, nq) do
            z = glrule(nq)[1]
            lam = [1 / prod(z[q] - z[j] for j in 1:nq if j != q) for q in 1:nq]
            setprecision(BigFloat, 512) do
                Vi = inv([BigFloat(z[q])^j for q in 1:nq, j in 0:nq-1])
                D0 = [factorial(big(k)) * Vi[k+1, q] for k in 0:nq-1, q in 1:nq]
                D1 = [sum(factorial(big(j)) / factorial(big(j - k)) * Vi[j+1, q] for j in k:nq-1) for k in 0:nq-1, q in 1:nq]
                FilonData(z, lam, Float64.(D0), Float64.(D1))
            end
        end
    end
end
"""
    filon_int(F, g, om) -> int_0^1 p(x) e^{-i om x} dx

p the polynomial through the values g at the nodes of F.  For om <= 2000 a composite Gauss-Legendre rule (64 nodes on
each of ceil(om/20) parts, p by the barycentric formula); beyond, the expansion by parts, which ends after nq terms for a
polynomial, sum_k (p^(k)(0) - p^(k)(1) e^{-i om})/(i om)^(k+1) (its terms fall like (2 nq^2/om)^k by the inequality of
Markov, so that the rounding of the derivatives stays below that of the values)
"""
function filon_int(F::FilonData, g::AbstractVector{ComplexF64}, om::Float64)
    nq = length(F.z)
    if om <= 2000
        m = max(1, ceil(Int, om / 20)); zr, wr = glrule(64)
        acc = zero(ComplexF64)
        for i in 0:m-1, r in 1:64
            x = (i + zr[r]) / m
            num = zero(ComplexF64); den = 0.0; hit = 0
            for q in 1:nq
                d = x - F.z[q]
                d == 0 && (hit = q; break)
                c = F.lam[q] / d; num += c * g[q]; den += c
            end
            acc += (wr[r] / m) * (hit > 0 ? g[hit] : num / den) * cis(-om * x)
        end
        return acc
    end
    io = 1 / complex(0.0, om); e = cis(-om); acc = zero(ComplexF64); fac = io
    for k in 1:nq
        d0 = zero(ComplexF64); d1 = zero(ComplexF64)
        @inbounds for q in 1:nq; d0 += F.D0[k, q] * g[q]; d1 += F.D1[k, q] * g[q]; end
        acc += fac * (d0 - d1 * e); fac *= io
    end
    acc
end
"""the panels on which `contour_sum_filon` takes Filon's rule: beta L >= FILON_OMIN[] = 30 (the slower panels keep the
Gauss-Legendre rule for the whole integrand, which resolves e^{-i beta w} there: on the first panel of the line, next to
w = 0, the split parts are nearly singular, and only their sum is smooth)"""
const FILON_OMIN = Ref(30.0)
"""
    contour_sum_filon(CP, nx, node!, tol, warn, beta) -> as `contour_sum`

`contour_sum` on the vertical line CP for events whose terms at the roots are split off (`filon_roots`): node!(GF_k, Gf_k,
AF_k, Af_k, JF_k, Jf_k, k) fills the terms of the rest as before and, per event, the part J(t_k) t'/t_k (and J(t_k) t' for
the density) whose integrand is e^{-t beta} J(t).  The rest takes the Gauss-Legendre weights of the line, the parts
Filon's rule on each panel: (1/pi) Im{e^{-tau beta} L e^{-i beta w_a} int_0^1 p(x) e^{-i beta L x} dx}, p through the
values of J t'/t on the panel [w_a, w_a + L] (`filon_int`).  Nodes beyond the last chunk computed count as 0.
"""
function contour_sum_filon(CP::CPath, nx::Int, node!::NF, tol::Float64, warn::Bool, beta::Vector{Float64}; omin::Float64 = FILON_OMIN[]) where {NF}
    K1 = length(CP.t)
    GF = zeros(K1, nx); Gf = zeros(K1, nx); AF = zeros(K1, nx); Af = zeros(K1, nx)
    JF = zeros(ComplexF64, K1, nx); Jf = zeros(ComplexF64, K1, nx)
    eb = [exp(-CP.tau * b) for b in beta]
    done = 0; CH = 32; last = Inf
    while done < K1
        k1 = done + 1; k2 = min(done + CH, K1)
        Threads.@threads :dynamic for k in k1:k2
            node!(view(GF, k, :), view(Gf, k, :), view(AF, k, :), view(Af, k, :), view(JF, k, :), view(Jf, k, :), k)
        end
        done = k2
        mx = max(maximum(abs, view(GF, k1:k2, :)), maximum(abs, view(Gf, k1:k2, :)))
        for ix in 1:nx
            mx = max(mx, eb[ix] * maximum(abs, view(JF, k1:k2, ix)), eb[ix] * maximum(abs, view(Jf, k1:k2, ix)))
        end
        last = mx * maximum(view(CP.wt, k1:k2))
        mx < tol && break
    end
    w = CP.wt[1:done]
    F = [sum(w[k] * GF[k, ix] for k in 1:done) for ix in 1:nx]
    f = [sum(w[k] * Gf[k, ix] for k in 1:done) for ix in 1:nx]
    nq, wa, Ls = linepanels(CP); Fd = filondata(nq)
    for ix in 1:nx
        b = beta[ix]; b > 0 || continue
        for p in eachindex(wa)
            j1 = (p - 1) * nq + 1; j1 > done && break
            L = Ls[p]
            if b * L < omin
                for k in j1:min(j1 + nq - 1, done)
                    e = exp(-CP.t[k] * b)
                    F[ix] += CP.wt[k] * imag(e * JF[k, ix]); f[ix] += CP.wt[k] * imag(e * Jf[k, ix])
                end
                continue
            end
            ph = L * cis(-b * wa[p]) * eb[ix] / pi
            F[ix] += imag(ph * filon_int(Fd, view(JF, j1:j1+nq-1, ix), b * L))
            f[ix] += imag(ph * filon_int(Fd, view(Jf, j1:j1+nq-1, ix), b * L))
        end
    end
    eF = [eps() * sum(w[k] * AF[k, ix] for k in 1:done) for ix in 1:nx]
    ef = [eps() * sum(w[k] * Af[k, ix] for k in 1:done) for ix in 1:nx]
    if warn && (max(maximum(eF; init = 0.0), maximum(ef; init = 0.0)) > WARN_TOL[] || !all(isfinite, F) || !all(isfinite, f))
        @warn "contour_eval: the terms of the sums are large: the rounding error may reach $(round(max(maximum(eF), maximum(ef)), sigdigits = 2)); the result may be inaccurate" CP.Mo.start CP.Mo.g0 CP.c
    end
    if warn && last > 1e-10
        @warn "contour_eval: the outer contour ends with terms of size $(round(last, sigdigits = 2)) (it is cut at |t - tau| = $(default_rmax(CP.c))); the result may be inaccurate" CP.Mo.start CP.Mo.g0 CP.c
    end
    F, f, done, (F = eF, f = ef)
end
"""the outer trapezoidal rule of `contour_eval` and `contour_eval_specs`: node!(GF_k, Gf_k, AF_k, Af_k, k) fills the terms
of node k for the nx points, the nodes are processed in chunks of 32 until a chunk whose terms all stay below tol"""
function contour_sum(CP::CPath, nx::Int, node!::NF, tol::Float64, warn::Bool, c0::Float64) where {NF}
    K1 = length(CP.t)
    GF = zeros(K1, nx); Gf = zeros(K1, nx); AF = zeros(K1, nx); Af = zeros(K1, nx)
    done = 0; CH = 32; last = Inf
    while done < K1
        k1 = done + 1; k2 = min(done + CH, K1)
        Threads.@threads :dynamic for k in k1:k2
            node!(view(GF, k, :), view(Gf, k, :), view(AF, k, :), view(Af, k, :), k)
        end
        done = k2
        mx = max(maximum(abs, view(GF, k1:k2, :)), maximum(abs, view(Gf, k1:k2, :)))
        last = mx * maximum(view(CP.wt, k1:k2))
        mx < tol && break
    end
    w = CP.wt[1:done]
    F = [sum(w[k] * GF[k, ix] for k in 1:done) for ix in 1:nx]
    f = [sum(w[k] * Gf[k, ix] for k in 1:done) for ix in 1:nx]
    eF = [eps() * sum(w[k] * AF[k, ix] for k in 1:done) for ix in 1:nx]
    ef = [eps() * sum(w[k] * Af[k, ix] for k in 1:done) for ix in 1:nx]
    if warn && (max(maximum(eF; init = 0.0), maximum(ef; init = 0.0)) > WARN_TOL[] || !all(isfinite, F) || !all(isfinite, f))
        @warn "contour_eval: the terms of the sums are large: the rounding error may reach $(round(max(maximum(eF), maximum(ef)), sigdigits = 2)); the result may be inaccurate" CP.Mo.start CP.Mo.g0 CP.c c0
    end
    if warn && last > 1e-10
        @warn "contour_eval: the outer contour ends with terms of size $(round(last, sigdigits = 2)) (it is cut at |t - tau| = $(default_rmax(CP.c))); the result may be inaccurate" CP.Mo.start CP.Mo.g0 CP.c c0
    end
    F, f, done, (F = eF, f = ef)
end
contour_eval(P::Prep, CP::CPath, xs::Vector{Float64}; kw...) = contour_eval(CP, xs; kw...)

"""the estimate of the rounding error above which `contour_eval` warns: 1e-4, because where the rounding error dominates
the actual error was 0.005 to 0.03 times the estimate (`scripts/check_fixed_far.jl`, (d)), so that the warning comes before
the error can reach 1e-5 (3e-6 at most at the threshold)"""
const WARN_TOL = Ref(1e-4)
const CPCACHE = Dict{Tuple{UInt64,Symbol,Float64,Float64,Float64,Float64},CPath}()
const CPORDER = Tuple{UInt64,Symbol,Float64,Float64,Float64,Float64}[]
const CPLOCK = ReentrantLock()
const PREPCACHE = Dict{UInt64,Prep}()
const PREPORDER = UInt64[]
"""the most entries each cache of the engine keeps (the outer paths, the refined lines, the prepared designs); beyond it
the oldest entry goes.  Without a bound the caches grew with every design, start and c of a long grid."""
const CACHE_MAX = Ref(64)
"""val under key in the cache d, whose keys in the order of insertion are in ord, the oldest dropped beyond CACHE_MAX[]
(call with CPLOCK held)"""
function cache_put!(d::Dict, ord::Vector, key, val)
    haskey(d, key) || push!(ord, key)
    d[key] = val
    while length(ord) > CACHE_MAX[]
        delete!(d, popfirst!(ord))
    end
    val
end
"""empty_caches!(): drop every cached outer path, refined line and prepared design (the values do not change; the next call
builds them again)"""
function empty_caches!()
    lock(CPLOCK) do
        empty!(CPCACHE); empty!(CPORDER); empty!(LINECACHE); empty!(LINEORDER); empty!(PREPCACHE); empty!(PREPORDER)
        empty!(LMCACHE); empty!(LMORDER)                        # (the providers of the line integral, DFDet_line.jl)
    end
    nothing
end
"""the prepared design, cached by the coefficients"""
function prep_cached(D::Determ, gls = nothing)
    key = hash((dkey(D), gls === nothing ? NaN : Float64(gls)))
    lock(CPLOCK) do
        hit = get(PREPCACHE, key, nothing)
        hit === nothing ? cache_put!(PREPCACHE, PREPORDER, key, blas1(() -> gls === nothing ? prep(D) : prep_gls(D, gls))) : hit
    end
end

# ---------- the outer paths of the classical cases at the unit root, shipped as tables ----------
"""whether the outer paths of the tables in data/paths are used (`cpath_cached`); false: always computed.  The tables hold
the paths of cases DK, DU and DT at c = 0 with a zero start (`scripts/make_path_tables.jl` writes them; a test checks
that they give the values of the computed paths), which then load in some 0.1 ms instead of 1 to 3 ms."""
const USE_TABLES = Ref(true)
const TABLE_DIR = joinpath(@__DIR__, "..", "data", "paths")
table_file(p::Int) = joinpath(TABLE_DIR, ("DK", "DU", "DT")[p+2] * "_c0_zero.bin")
"""p in -1, 0, 1 when Mo is trend(p) (the monomial basis, no GLS, the one-step statistic) with a zero start and c = 0, the
cases whose outer path ships as a table; -2 otherwise"""
function table_case(Mo::Model, c::Float64)
    (c == 0 && Mo.start == :zero && isnan(Mo.P.cbar)) || return -2
    D = Mo.P.D
    (D.detrend == :onestep && !D.loc && D.br == [0.0, 1.0]) || return -2
    for p in -1:1
        D.C == trend(p).C && return p
    end
    -2
end
const TABLE_MAGIC = "DFDetPath1"
"""write the outer path CP (zero start) to file: a magic string, the sizes, the scalars and the arrays as raw Float64"""
function write_path_table(file::AbstractString, CP::CPath)
    CP.Mo.start == :zero || error("write_path_table: zero start only")
    K = length(CP.t); r = size(CP.R, 1)
    open(file, "w") do io
        write(io, TABLE_MAGIC)
        write(io, Int64(K), Int64(r), Int64(CP.Mo.rp))
        write(io, CP.c, CP.tau, CP.tstar, CP.a, CP.b, CP.h, CP.phi, CP.minpos, CP.maxneg)
        write(io, CP.t, CP.dt, CP.R, CP.lP, CP.wt)
    end
    file
end
"""the outer path stored by `write_path_table`, for the model Mo (its sizes are checked against it)"""
function read_path_table(file::AbstractString, Mo::Model)
    open(file, "r") do io
        String(read(io, length(TABLE_MAGIC))) == TABLE_MAGIC || error("read_path_table: $file is not a path table")
        K = Int(read(io, Int64)); r = Int(read(io, Int64)); rp = Int(read(io, Int64))
        (r == Mo.r && rp == Mo.rp) || error("read_path_table: $file has $r roots ($rp positive), the design $(Mo.r) ($(Mo.rp))")
        sc = [read(io, Float64) for _ in 1:9]
        t = Vector{ComplexF64}(undef, K); dt = similar(t); R = Matrix{ComplexF64}(undef, r, K); lP = similar(t); wt = Vector{Float64}(undef, K)
        read!(io, t); read!(io, dt); read!(io, R); read!(io, lP); read!(io, wt)
        CPath(sc[1], sc[2], sc[3], sc[4], sc[5], sc[6], sc[7], t, dt, R, lP, sc[8], sc[9], Mo, Vector{Union{Nothing,FixT}}(nothing, K), wt)
    end
end

"""the outer path of the design with its start at c, cached per (design, start, g0, c, phi, hfac)"""
function cpath_cached(Mo::Model, c::Real; phi = nothing, hfac::Real = 1.0)
    key = (hash((dkey(Mo.P.D), Mo.P.cbar)), Mo.start, Mo.g0, Float64(c), phi === nothing ? -1.0 : Float64(phi), Float64(hfac))
    hit = lock(CPLOCK) do; get(CPCACHE, key, nothing); end
    hit === nothing || return hit
    tp = phi === nothing && hfac == 1 ? table_case(Mo, Float64(c)) : -2
    CP = tp >= -1 && USE_TABLES[] && isfile(table_file(tp)) ? read_path_table(table_file(tp), Mo) : cpath(Mo, c; phi, hfac)
    lock(CPLOCK) do; cache_put!(CPCACHE, CPORDER, key, CP); end
    CP
end
cpath_cached(P::Prep, c::Real; kw...) = cpath_cached(model(P), c; kw...)

"""the model of D with the start: start = :zero (default), :fixed (J(0) = gamma0), :stationary (c < 0)"""
function startmodel(D::Determ, c::Real, start::Symbol, gamma0::Real, gls = nothing)
    # (the checks of check_common of the C engine: a non-finite c gave "matrix contains Infs or NaNs", a non-finite gamma0
    # NaN)
    isfinite(c) || throw(ArgumentError("c must be a finite real scalar"))
    isfinite(gamma0) || throw(ArgumentError("gamma0 must be a finite real scalar"))
    start == :stationary && c >= 0 && throw(ArgumentError("the stationary start needs c < 0"))
    check_cmax(D, c, start, gamma0, "the general engine")
    model(prep_cached(D, gls), start, gamma0)
end

# ---------- the rules of one evaluation of the inversion on paths (dist_cdf, dist_pdf, dist_quantile) ----------
"""
    PATH_RULES

The rules of one evaluation of the inversion on paths (`contour_eval` on the outer contour of `cpath`): (hfac, n, ntail,
n with a fixed start, ntail with a fixed start), hfac the factor of the step of the outer hyperbola (`hyperbola`), n the
Gauss-Legendre nodes per panel of the U-shapes and ntail those of the tails of their rays (a fixed start, whose roots are
essential singularities of Mt, has more).  On the vertical line (`linecpath`: above the unit root, where no hyperbola
keeps the positive roots right of the origin) hfac does not apply and the line keeps its 24 nodes per panel.

  * :present, the rule until November 2026: hfac 1, n 24, ntail 40 (48, 64 with a fixed start);
  * :thin, the thinner rule: hfac 1.5, n 16, ntail 32 (32, 48 with a fixed start);
  * :inner, the thin inner rule on the outer contour of the present one;
  * :outer, the inner rule of the present one on the outer contour of the thin one;
  * :fine, the thin inner rule on the present hyperbola with half its step (`rule_path`);
  * :finestep, the inner rule of the present one on the present hyperbola with the step min(1/2, 2.5/(g0 |c0|)) times
    its own where the hyperbola is flatter than 70 degrees (2 to 11 times), for a fixed start without deterministic
    terms (`rule_path`).

The rule of `rule = :auto` is `path_region`; `verified = true` takes :present (and `verify_eval` its variants).
"""
const PATH_RULES = Dict(:present => (1.0, 24, 40, 48, 64), :thin => (1.5, 16, 32, 32, 48), :inner => (1.0, 16, 32, 32, 48),
                        :outer => (1.5, 24, 40, 48, 64), :fine => (0.5, 16, 32, 32, 48), :finestep => (1.0, 24, 40, 48, 64))
const RULE_MSG = "rule must be :auto, :present, :thin, :inner, :outer, :fine or :finestep (the rules of one evaluation of the inversion on paths, PATH_RULES)"
check_rule(rule::Symbol) = (rule === :auto || haskey(PATH_RULES, rule)) || throw(ArgumentError(RULE_MSG))
"""
    path_region(Mo, c, c0) -> Symbol

The rule of one evaluation (`PATH_RULES`) for the design of Mo with its start at c and the local null value c0, by
regions of (the class of the design, the start, c, c0), measured in November 2026 against a reference with half the
outer step and twice the inner nodes (the criterion: within max(1e-9, twice the present rule's own error) in F and
max(1e-8, twice its own) of the largest density in f; see the comments below and the README).
"""
function path_region(Mo::Model, c::Float64, c0::Float64)
    fixed = Mo.start == :fixed
    rank1 = Mo.r == 1 && Mo.rm == 0
    # near and above the unit root (c > -1) the outer integrand of the upper tail needs the outer step of the present rule
    # or a finer one: with c0 < 0, where the parabola X(v) has its root b ~ x/|c0| far out, the thinner outer step lost up
    # to 1e-4 with a fixed start (case DK at c = 0, c0 = -5 by 3.9e-5, case DU at c = 2, c0 = -5 by 3.1e-6) and 1.3e-8 of
    # the density's peak with a zero start (case DK at c = 0, c0 = -5); without deterministic terms and with a fixed start
    # the present step itself was off by up to 0.86 in the upper tail (case DK at c = 0.5, g0 = 1, c0 = -20; 1.8e-3 at c0 =
    # -5), where the steps of :finestep are within 1e-11 of the classical engine (`rule_path`), and with deterministic
    # terms :fine within 1e-12; there at c0 = 0 too the thinner step lost 1.4e-8 of the peak at mean + 5.75 sd (c = 0.5)
    if c > PATH_UNITROOT_CMIN
        # (beyond the range of the sweep above the unit root, with a fixed start beyond c = 2 and beyond c = 5 with a zero
        # start: the present rule; at c = 5 with a fixed start the thin rule differed from it by up to 5.6e-6 of the
        # peak of the density, Perron's model C with two breaks)
        (c > PATH_THIN_CMAX_FIXED && fixed || c > PATH_THIN_CMAX) && return :present
        c0 < 0 && return !fixed ? :inner : rank1 ? :finestep : :fine
        (c0 == 0 && fixed && rank1) && return :inner
    end
    # (far from the unit root, below the range of the sweep, with c0 other than 0 and c: the present rule; in case DK at
    # c = -480, c0 = 2 the thin rule differed from it by 7.7e-9)
    (c < FAR_CMIN && c0 != 0 && c0 != c) && return :present
    :thin
end
"""the borders of the regions of `path_region` in c"""
const PATH_UNITROOT_CMIN = -1.0
const PATH_THIN_CMAX_FIXED = 2.0
const PATH_THIN_CMAX = 5.0
"""
    path_rule(rule, verified, Mo, c, c0, hfac, kw) -> (hfac, n, ntail, rule)

The step factor of the outer hyperbola, the inner nodes and the rule of one evaluation: those of `rule` (:auto:
`path_region`; :present with verified = true), with hfac (a number) and the keywords n and ntail of kw, where given, in
their place.  The outer contour is `rule_path`'s.
"""
function path_rule(rule::Symbol, verified::Bool, Mo::Model, c::Real, c0::Real, hfac, kw)
    check_rule(rule)
    r = verified ? :present : rule === :auto ? path_region(Mo, Float64(c), Float64(c0)) : rule
    hf, n, nt, nf, ntf = PATH_RULES[r]
    fixed = Mo.start == :fixed
    hfac === nothing || (hf = Float64(hfac); r = :given)
    n = something(get(kw, :n, nothing), fixed ? nf : n)
    nt = something(get(kw, :ntail, nothing), fixed ? ntf : nt)
    hf, Int(n), Int(nt), r
end
"""
    rule_path(Mo, c, phi, hf, r, c0) -> CPath

The outer contour of the rule r with the step factor hf (`path_rule`): `cpath_cached`, except that the rules with a finer
step keep the angle of the present rule's hyperbola (where `fixed_phi` chooses it, its criterion on the largest term
weighs the step, and a finer step took a flatter hyperbola with larger terms: case DK at c = -0.5, g0 = 2, c0 = -20 was
off by 1.7e-4 on it, against 4.5e-12 on the present one): :fine half the present step, :finestep (a fixed start without
deterministic terms) the step min(1/2, 2.5/(g0 |c0|)) times the present one where the present hyperbola is flatter than
70 degrees (at c = 0.5, g0 = 1, c0 = -20 the present step was off by 0.86, half of it by 0.31, this one by 7e-12; at
g0 = 1.5, c0 = -1 the present step by 2.7e-7; on the steep hyperbolas of g0 = 2, 88 degrees, the present step is within
1e-12 and the thin inner rule is not); on the vertical line the line itself.
"""
function rule_path(Mo::Model, c::Real, phi, hf::Float64, r::Symbol, c0::Real)
    if phi === nothing && r in (:fine, :finestep)
        CP1 = cpath_cached(Mo, c)
        isnan(CP1.h) && return CP1
        phi = CP1.phi
        r === :finestep && (hf = CP1.phi < FINESTEP_PHI ? min(0.5, 2.5 / (abs(Mo.g0) * abs(Float64(c0)))) : 1.0)
    end
    cpath_cached(Mo, c; phi, hfac = hf)
end
"""the angle below which the step of :finestep shrinks (`rule_path`)"""
const FINESTEP_PHI = 70 * pi / 180

"""
    dist_cdf(D, xs, c = 0; c0 = 0, start = :zero, gamma0 = 0, phi = nothing, hfac = nothing, rule = :auto, kw...) -> F (vector)

The distribution function of the limit (U - c0 V)/sqrt(V) of the t-ratio for phi = 1 + c0/n under phi = 1 + c/n, with
the start J(0) = 0 (start = :zero), J(0) = gamma0 (:fixed; y_0 = gamma0 sigma sqrt(n)) or J(0) ~ N(0, -1/(2c))
(:stationary, c < 0).  gls = cbar: the data GLS-detrended at abar = 1 + cbar/n (`prep_gls`; zero start only).  method =
:auto (the default), :line or :paths: the line integral with the explicit kernel where it covers the design (`line_dist`), or
the inversion on paths around the roots.  rule: the rule of one evaluation of the paths (`PATH_RULES`; :auto, the
default, by `path_region`; :present the rule until November 2026, also taken by verified = true); hfac (the factor of
the outer step) and the keywords n, ntail (the nodes of the U-shapes) override it.
"""
dist_cdf(D::Determ, xs::AbstractVector, c::Real = 0.0; start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing, phi = nothing, hfac = nothing,
         verified::Bool = false, vtol::Real = 1e-9, rule::Symbol = :auto, kw...) =
    dist_eval(D, xs, c, 1, start, gamma0, gls, phi, hfac, verified, Float64(vtol); rule, kw...)
"""
    dist_pdf(D, xs, c = 0; c0, start, gamma0, verified = false, kw...) -> f (vector)

Its density.  For a design whose numerator has rank one (no deterministic terms, GLS detrending) with c0 > 0 the density
is infinite at the tangency x = -sqrt(4 c0 a) of the parabola X(v) (a the shift of Mt: 1/2, or (1 + gamma0^2)/2 with a
fixed start), and Inf is returned there (as by DFLimit); elsewhere it is continuous.
"""
dist_pdf(D::Determ, xs::AbstractVector, c::Real = 0.0; start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing, phi = nothing, hfac = nothing,
         verified::Bool = false, vtol::Real = 1e-9, rule::Symbol = :auto, kw...) =
    dist_eval(D, xs, c, 2, start, gamma0, gls, phi, hfac, verified, Float64(vtol); rule, kw...)
"""the value of `dist_cdf` (which = 1) or `dist_pdf` (2): the non-finite points set (`nonfinite`), the finite ones by one
evaluation of the contour integral or, with verified = true, by `verify_eval`; the infinite density at a tangency"""
function dist_eval(D::Determ, xs::AbstractVector, c::Real, which::Int, start::Symbol, gamma0::Real, gls, phi, hfac, verified::Bool, vtol::Float64;
                   method::Symbol = :auto, rule::Symbol = :auto, kw...)
    v = dist_eval_(D, xs, c, which, start, gamma0, gls, phi, hfac, verified, vtol; method, rule, kw...)
    which == 1 && cdf_range_check("dist_cdf", D, c, start, gamma0, Float64(get(kw, :c0, 0.0)), "x", xs, v)
    v
end
"""
The final check of a distribution function (`dist_cdf`, `coef_cdf`): a value below -1e-8 or above 1 + 1e-8 (CDF_RANGE_TOL)
raises an error that names the call, the design, c, c0, the start, the point and the value.  Where the contour integral
holds its error is far below 1e-8, so a value outside [0, 1] by more than that is a computation that failed, never
rounding; the check costs nothing and is not a cross-check (nothing is recomputed).  The same check and tolerance in
DFLimit's classical engine (cdf_range_check of DFExact), the C engine and the MATLAB programs.
"""
const CDF_RANGE_TOL = 1e-8
function cdf_range_check(fn::String, D::Determ, c::Real, start::Symbol, gamma0::Real, c0::Float64, xname::String, xs, v)
    for k in eachindex(v)
        (v[k] < -CDF_RANGE_TOL || v[k] > 1 + CDF_RANGE_TOL) || continue
        st = start == :fixed ? "fixed start gamma0 = $gamma0" : "$start start"
        error("$fn: the design $(D.name), c = $c, c0 = $c0, $st, $xname = $(xs[k]): the computed distribution function is " *
              "$(v[k]), outside [0, 1] by more than $CDF_RANGE_TOL; the computation fails here and no value is returned")
    end
    v
end
function dist_eval_(D::Determ, xs::AbstractVector, c::Real, which::Int, start::Symbol, gamma0::Real, gls, phi, hfac, verified::Bool, vtol::Float64;
                    method::Symbol = :auto, rule::Symbol = :auto, kw...)
    check_cmax(D, c, start, gamma0, which == 1 ? "the t-ratio (dist_cdf)" : "the t-ratio (dist_pdf)")
    check_rule(rule)
    lv = line_dist(method, D, xs, c, which, start, gamma0, gls, phi, hfac, verified, vtol; rule, kw...); lv === nothing || return lv
    c0 = Float64(get(kw, :c0, 0.0))
    Mo = startmodel(D, c, start, gamma0, gls)
    hf, n, nt, rl = path_rule(rule, verified, Mo, c, c0, hfac, kw)
    CP = rule_path(Mo, c, phi, hf, rl, c0)
    nonfinite(xs, which == 1) do x
        v = verified ? verify_eval((CPv, o) -> contour_eval(line_refined(CPv, x; c0), x; o..., kw...)[which], CP, vtol) :
                       contour_eval(line_refined(CP, x; c0), x; kw..., n, ntail = nt)[which]
        which == 2 && tangency!(v, CP.Mo, x, c0)
        v
    end
end
"""
    dist_both(D, xs, c = 0; c0 = 0, start = :zero, gamma0 = 0, gls = nothing) -> (F, f)

`dist_cdf` and `dist_pdf` of the inversion on paths (method = :paths) in one evaluation: the paths return both from one
pass (`contour_eval`), the values those of the two functions bit for bit (one evaluation by the rule of `path_region`).
"""
function dist_both(D::Determ, xs::Vector{Float64}, c::Real = 0.0; c0::Real = 0.0, start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing)
    check_cmax(D, c, start, gamma0, "the t-ratio (dist_cdf)")
    c0 = Float64(c0)
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))
    Mo = startmodel(D, c, start, gamma0, gls)
    hf, n, nt, rl = path_rule(:auto, false, Mo, c, c0, nothing, pairs((;)))
    CP = rule_path(Mo, c, nothing, hf, rl, c0)
    all(isfinite, xs) || throw(ArgumentError("dist_both: finite points only"))
    F, f = contour_eval(line_refined(CP, xs; c0), xs; c0, n, ntail = nt)[1:2]
    tangency!(f, CP.Mo, xs, c0)
    cdf_range_check("dist_cdf", D, c, start, gamma0, c0, "x", xs, F)
    F, f
end
"""Inf at the points of x where the density of a rank-one design with c0 > 0 is infinite (`dist_pdf`)"""
function tangency!(v::Vector{Float64}, Mo::Model, x::Vector{Float64}, c0::Float64)
    (c0 > 0 && Mo.r == 1 && Mo.rm == 0) || return v
    xs_ = -sqrt(4c0 * Mo.a)
    for j in eachindex(x)
        abs(x[j] - xs_) <= 1e-12 * max(1.0, abs(x[j])) && (v[j] = Inf)
    end
    v
end

"""
    verify_eval(evalf, CP, vtol) -> values

The checked computation (verified = true of the user-level functions): evalf(CP', options) evaluates the values on the
outer contour CP' with the inner options (alpha, n, ntail).  Two structurally different evaluations: the default rule,
and a variant in which every node moves (the hyperbola 7.5 degrees steeper, at most 88, with 0.8 times the step, or on
the vertical line 32 nodes per panel for 24; the U-shapes at alpha = 0.3 for 0.5 with 16 and 24 more nodes per panel and
tail).  When they agree to vtol the default values are returned; otherwise a third, finer evaluation (half the step,
twice the nodes) is returned, with a warning if it and the variant still differ by more than vtol.  About twice the cost
of the default.
"""
function verify_eval(evalf, CP::CPath, vtol::Float64)
    Mo = CP.Mo; fixed = Mo.start == :fixed
    n0 = fixed ? 48 : 24; nt0 = fixed ? 64 : 40
    v1 = evalf(CP, (alpha = 0.5, n = n0, ntail = nt0))
    v2 = try
        evalf(variant_path(CP, 1), (alpha = 0.3, n = n0 + 16, ntail = nt0 + 24))
    catch e
        # (any error, as in the C engine; those of the node loops arrive wrapped by their threads)
        nofallback(e) && rethrow()
        evalf(variant_path(CP, 0), (alpha = 0.3, n = n0 + 16, ntail = nt0 + 24))     # (the groups not separated on the steeper hyperbola)
    end
    d = maximum(abs, v1 .- v2; init = 0.0)
    (d <= vtol || !all(isfinite, v1)) && return v1
    v3 = evalf(variant_path(CP, 2), (alpha = 0.5, n = 2n0, ntail = 2nt0))
    d3 = maximum(abs, v3 .- v2; init = 0.0)
    d3 <= vtol || @warn "verified: two evaluations by different rules differ by $(round(d3, sigdigits = 2)) (above $vtol)" Mo.P.D.name Mo.start CP.c
    v3
end
"""whether the error e must reach the caller instead of the fallback of `verify_eval`: an interrupt or an exhausted
memory, also when it arrives wrapped by the threads of a node loop (TaskFailedException, CompositeException, as
DFExact.pathtoolarge unwraps them)"""
nofallback(e) = e isa InterruptException || e isa OutOfMemoryError ||
    (e isa TaskFailedException && nofallback(e.task.exception)) || (e isa CompositeException && any(nofallback, e.exceptions))
"""the outer contour of a variant rule (`verify_eval`): level 1 steeper and with a shorter step, level 0 the same with the
default angle, level 2 the default angle with half the step; on the vertical line more nodes per panel"""
function variant_path(CP::CPath, level::Int)
    Mo = CP.Mo; c = CP.c
    if CP.phi == pi / 2 && isnan(CP.h)
        return linecpath(Mo, c, CP.tstar, CP.tau; nq = level == 2 ? 48 : 32)
    end
    level == 1 && return cpath_cached(Mo, c; phi = min(CP.phi + 7.5 * pi / 180, 88 * pi / 180), hfac = 0.8)
    level == 0 && return cpath_cached(Mo, c; phi = CP.phi, hfac = 0.8)
    cpath_cached(Mo, c; phi = CP.phi, hfac = 0.5)
end

"""the checks of the start value x0 (named name) and the tolerance tol of a search of fn"""
function search_args(fn::String, name::String, x0::Real, tol::Real)
    isfinite(x0) || throw(ArgumentError("$fn: $name must be a finite real scalar"))
    (tol > 0 && isfinite(tol)) || throw(ArgumentError("$fn: tol must be a positive finite number"))
    nothing
end
"""
    dist_quantile(D, p, c = 0; x0, tol = 1e-9, c0, start, gamma0, kw...) -> x

The p-quantile by Newton's method on F(x) - p with the density from the same evaluation, from x0, the step limited to 1;
where that does not converge in 30 steps (or leaves the finite numbers), the safeguarded Newton's method of `newton_safe`
(a bracket, bisection where a step leaves it).  p may be a vector (one quantile each, from the same x0).  method, rule
and hfac as in `dist_cdf` (the search by the line integral: `line_quantile`).
"""
function dist_quantile(D::Determ, p::Real, c::Real = 0.0; x0::Real = -2.0, tol::Real = 1e-9, start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing,
                       phi = nothing, hfac = nothing, method::Symbol = :auto, rule::Symbol = :auto, kw...)
    0 < p < 1 || throw(ArgumentError("dist_quantile: p must lie in (0, 1)"))
    search_args("dist_quantile", "x0", x0, tol)
    check_cmax(D, c, start, gamma0, "the quantiles of the t-ratio (dist_quantile)")
    check_rule(rule)
    lq = line_quantile(method, D, p, c; x0, tol, start, gamma0, gls, phi, hfac, kw...); lq === nothing || return lq
    c0 = Float64(get(kw, :c0, 0.0))
    Mo = startmodel(D, c, start, gamma0, gls)
    hf, n, nt, rl = path_rule(rule, false, Mo, c, c0, hfac, kw)
    CP = rule_path(Mo, c, phi, hf, rl, c0)
    x = Float64(x0)
    for it in 1:30
        F, f, _ = contour_eval(line_refined(CP, [x]; c0), [x]; kw..., n, ntail = nt)
        dx = clamp((F[1] - p) / f[1], -1.0, 1.0)
        isfinite(dx) || break
        x -= dx
        abs(dx) < tol && return x
    end
    FF(y) = (r = contour_eval(line_refined(CP, [y]; c0), [y]; kw..., n, ntail = nt); (r[1][1], r[2][1]))
    newton_safe(FF, p, x0, 1.0; tol)
end
dist_quantile(D::Determ, p::AbstractVector, c::Real = 0.0; kw...) = [dist_quantile(D, pk, c; kw...) for pk in p]

# =====================================================================================================================
# The coefficient statistic, moments, local power
# =====================================================================================================================
"""
    logmgf_point(Mo, c, s, t, W) -> log M(s,t)

The logarithm of the joint moment generating function of the model at one complex point, with the principal logarithm of
the determinant (no continuation): zero start -s/2 - (c + theta)/2 - log det(I - 2 (sA + tB + gamma E11) Sigma)/2, the
stationary start with the extended matrices, the fixed start adding g0^2 Qf(s,t) (`fixparts`, single-valued).
Returns (log M without the determinant, log det): the caller continues the branch of the determinant.
"""
logmgf_point(Mo::Model, c::Float64, s::ComplexF64, t::ComplexF64, W::MWork, Bf::Matrix{ComplexF64}) =
    logmgf_spart(Mo, c, s, t, logmgf_tpart!(Mo, c, t, W)..., W, Bf)
"""the part of `logmgf_point` that depends on t only: Sigma(theta) into W (and the extended covariance of the stationary
start into W.Sx), with a fixed start its data fx (`fixparts`) -> (theta, fx)"""
function logmgf_tpart!(Mo::Model, c::Float64, t::ComplexF64, W::MWork)
    θ = sqrt(complex(c * c - 2t))
    if Mo.start == :fixed
        _, _, _, fx = fixparts(Mo, c, t, W)
        return θ, fx
    end
    sigma!(W.S, Mo.P, θ, W.sw)
    Mo.start == :stationary && statsigma!(W.Sx, W.S, W.sw.h, -1 / (2c))
    θ, nothing
end
"""the rest of `logmgf_point` at s, with W as `logmgf_tpart!` left it for the same t"""
function logmgf_spart(Mo::Model, c::Float64, s::ComplexF64, t::ComplexF64, θ::ComplexF64, fx, W::MWork, Bf::Matrix{ComplexF64})
    W.sd === nothing || return logmgf_spart_s(Mo, c, s, t, θ, fx, W, Bf, W.sd)::Tuple{ComplexF64,ComplexF64}
    if Mo.start == :stationary
        ld = logdet(lu(buildMx!(W, Mo, c, s, t, θ)))
        return -s / 2 - (c + θ) / 2, ld
    end
    extra = Mo.start == :fixed ? Mo.g0^2 * (fx.cf + fixE1(fx, s, Bf) - Mo.kappa * s) : zero(ComplexF64)
    ld = logdet(lu(buildM!(W, Mo.P, c, s, t, θ)))
    -s / 2 - (c + θ) / 2 + extra, ld
end

"""the grid of `coef_cdf_gp`: u = sinh(v), v = k h, k = 0..N, h = asinh(umax)/N, u_0 = 1e-7, trapezoidal weights in v
(DFExact.gpgrid)"""
function gpgrid(N::Int, umax::Float64)
    h = asinh(umax) / N
    v = [k * h for k in 0:N]; u = sinh.(v); u[1] = 1e-7; w = cosh.(v) .* h; w[1] /= 2
    u, w
end

"""
    gp_edge(Mo, a, c) -> d

The distance from the origin of the nearest singularity of phi(u) = E exp(iu (U - aV)) in the complex plane (U - aV is a
quadratic form, whose transform M(s, -a s) is singular on the real line only, at the first zeros s_+ > 0 and s_- < 0 of
the entire function e^{c + theta} det(...) along s, scanned by factors of 1.1 from 1e-10 to 1e4 and bisected): the
smaller of s_+ and |s_-|.  (The phase of the determinant alone is -Im theta on the real line where -a s > c^2/2, so that
the sign change is read off the entire function.)
"""
function gp_edge(Mo::Model, a::Float64, c::Float64)
    W = MWork(Mo); Bf = fixbuf(Mo.r)
    function bad(s)
        t = complex(-a * s)
        θ, fx = logmgf_tpart!(Mo, c, t, W)
        _, ld = logmgf_spart(Mo, c, complex(s), t, θ, fx, W, Bf)
        L = ld + c + θ
        !(isfinite(real(L)) && abs(imag(L)) < 1.0)
    end
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
the smallest ratio of the distance d of the singularity of phi(u) (`gp_edge`) to the step h = asinh(umax)/N of the grid of
`coef_cdf_gp` near the origin, below which it refuses.  The error of the trapezoidal rule there is about 0.16
exp(-2 pi d/h) (case DK, against the classical engine, November 2026: 4.5e-4 at d/h = 0.98, 6.8e-5 at 1.25, 8.4e-6 at
1.57, 7.7e-7 at 1.94, below 6e-8 from 3.06 on); the law of U - aV widens like e^{2c} above the unit root and d shrinks
like 2c e^{-c}, so that with the default grid the 30% point of case DK has d/h = 3.1 at c = 10 and 0.5 at c = 12 (off by
1.1e-2), and the 1% point already d/h = 1.9 at c = 5.  The bound of the point-optimal statistic (`PT_EDGE`), for the same
rule.
"""
const GP_EDGE = 3.3
"""
    coef_cdf_grid(D, ys, c = 0; c0 = 0, start = :zero, gamma0 = 0, N = 60000, umax = 4000) -> P (vector)

`coef_cdf_gp` with method = :grid (in DFLimit the default of `coef_cdf_gp` is the single inversion of Bao and Kan,
src/DFKan_det.jl).  The distribution function of the limit (U - c0 V)/V of n(phi^ - phi0), phi0 = 1 + c0/n, by the
inversion formula of Gil-Pelaez: P(U - aV <= 0) = 1/2 - (1/pi) int_0^inf Im phi(u)/u du, a = c0 + y, phi(u) = M(iu, -i a u), with the branch of
the inverse square root of the determinant continued along u (the phase of log det unwrapped from u = 0, where the
determinant is positive), the trapezoidal rule in u = sinh(v) with N steps up to umax (as DFExact.tr_gilpelaez).  An
independent check of `coef_cdf`, which uses the contour integral.  Where a = 0 (y = -c0) the transform decays only like
u^{-r/2} and the truncation at umax shows.  Where the grid does not resolve the singularity of phi nearest the real line
(`GP_EDGE`: above the unit root, from about c = 5 in the lower tail and c = 10 in the bulk with the default N), it
refuses with an ArgumentError that names the N it would need.
"""
function coef_cdf_grid(D::Determ, ys::AbstractVector, c::Real = 0.0; c0::Real = 0.0, start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing,
                       N::Int = 60000, umax::Real = 4000.0)
    check_cmax(D, c, start, gamma0, "the coefficient statistic (coef_cdf_gp)")
    Mo = startmodel(D, c, start, gamma0, gls); c = Float64(c)
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))
    (umax > 0 && isfinite(umax)) || throw(ArgumentError("coef_cdf_gp: umax must be a positive finite number"))
    u, w = gpgrid(N, Float64(umax))
    hgp = asinh(Float64(umax)) / N
    out = Float64[]
    for y in ys
        # (1, 0, NaN at Inf, -Inf, NaN, without the sum: at Inf the transform at t = -i Inf u did not return)
        isfinite(y) || (push!(out, isnan(y) ? NaN : y > 0 ? 1.0 : 0.0); continue)
        a = Float64(c0) + Float64(y)
        d = blas1f(() -> gp_edge(Mo, a, c), dense_la(Mo))
        d / hgp >= GP_EDGE || throw(ArgumentError("coef_cdf_gp at c = $c, y = $y: the transform of U - (c0 + y) V has a singularity at $(round(d, sigdigits = 3)) from the real line, which the grid of the inversion (step $(round(hgp, sigdigits = 3)) near the origin) does not resolve (above the unit root the law widens like e^{2c}); it needs N >= $(ceil(Int, GP_EDGE * asinh(Float64(umax)) / d)), or coef_cdf (the contour integral)"))
        L0 = Vector{ComplexF64}(undef, N + 1); LD = Vector{ComplexF64}(undef, N + 1)
        # (one BLAS thread, see `blas1`)
        blas1f(dense_la(Mo)) do
            Threads.@threads :static for ch in collect(Iterators.partition(1:N+1, cld(N + 1, Threads.nthreads())))
                W = MWork(Mo); Bf = fixbuf(Mo.r)
                for k in ch
                    L0[k], LD[k] = logmgf_point(Mo, c, complex(0.0, u[k]), complex(0.0, -a * u[k]), W, Bf)
                end
            end
        end
        ph = imag(LD[1]); acc = 0.0
        for k in 1:N+1
            if k > 1
                ph += angle(exp(im * (imag(LD[k]) - imag(LD[k-1]))))
            end
            lM = L0[k] - complex(real(LD[k]), ph) / 2
            acc += imag(exp(lM)) / u[k] * w[k]
        end
        push!(out, 0.5 - acc / pi)
    end
    out
end

"""
    coef_cdf(D, ys, c = 0; c0 = 0, start = :zero, gamma0 = 0, kw...) -> P (vector)

The distribution function of the limit (U - c0 V)/V of n(phi^ - phi0), phi0 = 1 + c0/n, under phi = 1 + c/n: P[(U - c0
V)/V <= y] = P[U - (c0 + y) V <= 0], the distribution function of the t-ratio at x = 0 with the local null value c0 + y
(the contour integral; DFLimit's coef_cdf likewise).  `coef_cdf_gp` computes the same by Gil-Pelaez.
"""
function coef_cdf(D::Determ, ys::AbstractVector, c::Real = 0.0; c0::Real = 0.0, start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing,
                  phi = nothing, hfac = 1.0, verified::Bool = false, vtol::Real = 1e-9, kw...)
    check_cmax(D, c, start, gamma0, "the coefficient statistic (coef_cdf)")
    CP = cpath_cached(startmodel(D, c, start, gamma0, gls), c; phi, hfac)
    v = nonfinite(ys, true) do y
        verified && return verify_eval((CPv, o) -> [contour_eval(CPv, [0.0]; c0 = Float64(c0) + yk, o..., kw...)[1][1] for yk in y], CP, Float64(vtol))
        [contour_eval(CP, [0.0]; c0 = Float64(c0) + yk, kw...)[1][1] for yk in y]
    end
    cdf_range_check("coef_cdf", D, c, start, gamma0, Float64(c0), "y", ys, v)
end
"""coef_pdf(D, ys, c = 0; c0, start, gamma0, kw...) -> f (vector): its density, by the kernel 2 v^3 (d/dc0 of F at x = 0)"""
function coef_pdf(D::Determ, ys::AbstractVector, c::Real = 0.0; c0::Real = 0.0, start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing,
                  phi = nothing, hfac = 1.0, verified::Bool = false, vtol::Real = 1e-9, kw...)
    check_cmax(D, c, start, gamma0, "the coefficient statistic (coef_pdf)")
    CP = cpath_cached(startmodel(D, c, start, gamma0, gls), c; phi, hfac)
    nonfinite(ys, false) do y
        verified && return verify_eval((CPv, o) -> [contour_eval(CPv, [0.0]; c0 = Float64(c0) + yk, vpow = 3, o..., kw...)[2][1] for yk in y], CP, Float64(vtol))
        [contour_eval(CP, [0.0]; c0 = Float64(c0) + yk, vpow = 3, kw...)[2][1] for yk in y]
    end
end
"""
    coef_quantile(D, p, c = 0; y0 = -10, tol = 1e-9, c0, start, gamma0, kw...) -> y

The p-quantile of the limit of n(phi^ - phi0) by Newton's method on coef_cdf - p with coef_pdf from the same evaluation,
from y0, the step limited to 5.
"""
function coef_quantile(D::Determ, p::Real, c::Real = 0.0; y0::Real = -10.0, tol::Real = 1e-9, c0::Real = 0.0, start::Symbol = :zero,
                       gamma0::Real = 0.0, gls = nothing, phi = nothing, hfac = 1.0, kw...)
    0 < p < 1 || throw(ArgumentError("coef_quantile: p must lie in (0, 1)"))
    search_args("coef_quantile", "y0", y0, tol)
    check_cmax(D, c, start, gamma0, "the coefficient statistic (coef_quantile)")
    CP = cpath_cached(startmodel(D, c, start, gamma0, gls), c; phi, hfac)
    y = Float64(y0)
    for it in 1:40
        F, f, _ = contour_eval(CP, [0.0]; c0 = Float64(c0) + y, vpow = 3, kw...)
        dy = clamp((F[1] - p) / f[1], -5.0, 5.0)
        isfinite(dy) || break
        y -= dy
        abs(dy) < tol && return y
    end
    # (no convergence: the safeguarded Newton's method)
    FF(z) = (r = contour_eval(CP, [0.0]; c0 = Float64(c0) + z, vpow = 3, kw...); (r[1][1], r[2][1]))
    newton_safe(FF, p, y0, 5.0; tol)
end
coef_quantile(D::Determ, p::AbstractVector, c::Real = 0.0; kw...) = [coef_quantile(D, pk, c; kw...) for pk in p]

"""
    local_power(D, cs; level = 0.05, statistic = :tratio, c0 = 0, start = :zero, gamma0 = 0, kw...) -> power (vector)

The asymptotic power of the left-tailed level-`level` test of phi = 1 + c0/n against phi = 1 + c/n for each c in cs:
P_c(stat <= q), q the level-quantile under c = c0 (with the same start; a stationary start needs c0 < 0), stat the
t-ratio (statistic = :tratio) or n(phi^ - phi0) (:coef).
"""
function local_power(D::Determ, cs::AbstractVector; level::Real = 0.05, statistic::Symbol = :tratio, c0::Real = 0.0, start::Symbol = :zero,
                     gamma0::Real = 0.0, gls = nothing, kw...)
    0 < level < 1 || throw(ArgumentError("local_power: the level must lie in (0, 1)"))
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))
    if statistic == :tratio
        q = dist_quantile(D, level, c0; c0, start, gamma0, gls, x0 = gls === nothing ? -3.5 : -2.5, tol = 1e-11, kw...)
        return [dist_cdf(D, [q], c; c0, start, gamma0, gls, kw...)[1] for c in cs]
    elseif statistic == :coef
        q = coef_quantile(D, level, c0; c0, start, gamma0, gls, y0 = -15.0, tol = 1e-11, kw...)
        return [coef_cdf(D, [q], c; c0, start, gamma0, gls, kw...)[1] for c in cs]
    end
    throw(ArgumentError("statistic must be :tratio or :coef"))
end

# ---------- the line U - aV <= b, the law of V, the power envelope, ADFmax, regions ----------
"""
    newton_safe(FF, p, x0, step; tol = 1e-10, maxit = 100) -> x

The solution of F(x) = p for an increasing F, FF(x) -> (F(x), f(x)): a bracket grown from [x0 - step, x0 + step] by
doubling, then Newton's method kept inside the bracket (bisection where a step leaves it), to |dx| < tol.
"""
function newton_safe(FF, p::Real, x0::Real, step::Real; tol::Real = 1e-10, maxit::Int = 100)
    x = Float64(x0); lo = x - step; hi = x + step; w = Float64(step)
    for _ in 1:60
        FF(lo)[1] > p || break
        lo -= w; w *= 2
    end
    w = Float64(step)
    for _ in 1:60
        FF(hi)[1] < p || break
        hi += w; w *= 2
    end
    for _ in 1:maxit
        F, f = FF(x)
        F > p ? (hi = x) : (lo = x)
        xn = x - (F - p) / f
        (lo <= xn <= hi && isfinite(xn)) || (xn = (lo + hi) / 2)     # (a step below half an ulp lands on the end of the bracket: converged)
        abs(xn - x) < tol && return xn
        x = xn
    end
    @warn "newton_safe: no convergence" p x
    x
end

"""the cached outer contour of D with its start at c (as `dist_cdf`)"""
user_path(D::Determ, c::Real, start::Symbol, gamma0::Real, gls, phi, hfac; what::AbstractString = "the general engine") =
    (check_cmax(D, c, start, gamma0, what); cpath_cached(startmodel(D, c, start, gamma0, gls), c; phi, hfac))

"""
    lin_cdf(D, bs, a, c = 0; start = :zero, gamma0 = 0, gls = nothing, kw...) -> P (vector)

The limit of P(U - aV <= b) for each b in bs (as DFLimit's ltratio_lincdf; the point-optimal test of Elliott,
Rothenberg and Stock against cbar rejects for small U - aV), by the contour integral: with R = U + a0 (a0 the shift of Mt,
`Model`) the event is R <= X(sqrt V), X(v) = a v^2 + a0 + b, the parabola of the t-ratio with c0 = a, x = 0 and the
constant a0 + b in place of a0 (`XSpec`).  For every design and start (gls = cbar: GLS-detrended data).  a = NaN gives
P(V <= b) (`v_cdf`).
"""
function lin_cdf(D::Determ, bs::AbstractVector, a::Real, c::Real = 0.0; start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing, phi = nothing,
                 hfac = 1.0, verified::Bool = false, vtol::Real = 1e-9, kw...)
    isnan(a) && return v_cdf(D, bs, c; start, gamma0, gls, phi, hfac)
    isfinite(a) || throw(ArgumentError("lin_cdf: a must be a finite real scalar (NaN: the law of V)"))
    CP = user_path(D, c, start, gamma0, gls, phi, hfac; what = "the linear form U - aV (lin_cdf)")
    nonfinite(bs, true) do b
        specs = [[XSpec(Float64(a), 0.0, CP.Mo.a + bk; dA = 1.0)] for bk in b]
        spec_eval(CP, specs, 1, verified, Float64(vtol); kw...)
    end
end
"""the values (which = 1) or the derivatives (2) of the events specs on the outer contour CP, by one evaluation or, with
verified = true, by `verify_eval`"""
function spec_eval(CP::CPath, specs::Vector{Vector{XSpec}}, which::Int, verified::Bool, vtol::Float64; kw...)
    qm = [maximum((abs(sp.q) for sp in S); init = 0.0) for S in specs]          # (the slopes: x of the t-ratio and ADFmax)
    am = [abs(sp.A) for S in specs for sp in S]                                 # (the constants of the kernels, `line_refine`)
    c0s = [sp.c0 for S in specs for sp in S if sp.q != 0]                       # (the parabolas with a slope, `line_refine`)
    verified ? verify_eval((CPv, o) -> contour_eval_specs(line_refined(CPv, qm; amax = am, c0 = c0s), specs; o..., kw...)[which], CP, vtol) :
               contour_eval_specs(line_refined(CP, qm; amax = am, c0 = c0s), specs; kw...)[which]
end
"""lin_pdf(D, bs, a, c = 0; kw...) -> the density of U - aV at each b (arguments as `lin_cdf`; a = NaN: of V, `v_pdf`)"""
function lin_pdf(D::Determ, bs::AbstractVector, a::Real, c::Real = 0.0; start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing, phi = nothing,
                 hfac = 1.0, verified::Bool = false, vtol::Real = 1e-9, kw...)
    isnan(a) && return v_pdf(D, bs, c; start, gamma0, gls, phi, hfac)
    isfinite(a) || throw(ArgumentError("lin_pdf: a must be a finite real scalar (NaN: the law of V)"))
    CP = user_path(D, c, start, gamma0, gls, phi, hfac; what = "the linear form U - aV (lin_pdf)")
    nonfinite(bs, false) do b
        specs = [[XSpec(Float64(a), 0.0, CP.Mo.a + bk; dA = 1.0)] for bk in b]
        spec_eval(CP, specs, 2, verified, Float64(vtol); kw...)
    end
end
"""lin_quantile(D, p, a, c = 0; b0 = 0, tol = 1e-10, kw...) -> b with P(U - aV <= b) = p (`newton_safe` on `lin_cdf`, `lin_pdf`)"""
function lin_quantile(D::Determ, p::Real, a::Real, c::Real = 0.0; b0::Real = 0.0, tol::Real = 1e-10, start::Symbol = :zero, gamma0::Real = 0.0,
                      gls = nothing, phi = nothing, hfac = 1.0, kw...)
    0 < p < 1 || throw(ArgumentError("lin_quantile: p must lie in (0, 1)"))
    isfinite(a) || throw(ArgumentError("lin_quantile: a must be a finite real scalar"))
    search_args("lin_quantile", "b0", b0, tol)
    CP = user_path(D, c, start, gamma0, gls, phi, hfac; what = "the quantiles of the linear form U - aV (lin_quantile)")
    FF(b) = (r = contour_eval_specs(line_refined(CP, Float64[]; amax = [abs(CP.Mo.a + b)]), [[XSpec(Float64(a), 0.0, CP.Mo.a + b; dA = 1.0)]]; kw...); (r[1][1], r[2][1]))
    newton_safe(FF, p, b0, 1.0; tol)
end

"""
    v_cdf(D, ys, c = 0; start = :zero, gamma0 = 0, gls = nothing) -> P(V <= y) (vector)

The distribution function of V by the single inversion of M_V(t) = M(0,t) on the outer contour: P(V > y) =
(1/2 pi i) int_C M_V(t) e^{-t y} dt/t, M_V(t) = Mt(0, t) from the roots and log P(t) along the contour (the first term of
the contour integral representation, a probability for V alone).
"""
function v_cdf(D::Determ, ys::AbstractVector, c::Real = 0.0; start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing, phi = nothing, hfac = 1.0)
    CP = v_path(D, c, start, gamma0, gls, phi, hfac)
    # (0 at y <= 0, where the factor e^{-t y} grows along the path and the sum gave NaN, and at -Inf (V > 0); 1 at Inf)
    [isnan(y) ? NaN : !(y > 0) ? 0.0 : isinf(y) ? 1.0 : 1 - vsum(CP, Float64(y), true) for y in ys]
end
"""v_pdf(D, ys, c = 0; kw...) -> the density of V (the inversion (1/2 pi i) int_C M_V(t) e^{-t y} dt)"""
function v_pdf(D::Determ, ys::AbstractVector, c::Real = 0.0; start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing, phi = nothing, hfac = 1.0)
    CP = v_path(D, c, start, gamma0, gls, phi, hfac)
    [isnan(y) ? NaN : (!(y > 0) || isinf(y)) ? 0.0 : vsum(CP, Float64(y), false) for y in ys]
end
"""
    v_path(D, c, start, gamma0, gls, phi, hfac) -> the outer contour of the law of V alone

The path of the design (`cpath_cached`), except where that is the vertical line (c > 0, or GLS detrending with a small
|cbar|: no hyperbola keeps the positive roots right of the origin).  The U-shapes need the roots on their sides; V alone
does not: M_V(t) = M(0, t) is analytic off the real half-line t >= t_*, and the roots enter only through the logarithms
at s = 0, which `vsum` continues along the path.  On the line the integrand M_V(t) e^{-t y}/t oscillates in w at the
rate y, damped only by e^{-tau y} with tau = min(1, t_*/2) tiny above the unit root, and the doubling panels of
`linecpath` do not resolve it: P(V <= y) was off by 2e-3 in case DK at c = 5.  Here the hyperbola at the angle the path
would have without the steepening for the roots (pi/4, or `fixed_phi` with a fixed start and below c = -60), on which
e^{-t y} decays.  An angle given by phi is kept (and so is the line, if that is what it gives).
"""
function v_path(D::Determ, c::Real, start::Symbol, gamma0::Real, gls, phi, hfac)
    CP = user_path(D, c, start, gamma0, gls, phi, hfac; what = "the law of V (v_cdf, v_pdf)")
    phi === nothing ? v_own(CP; hfac) : CP
end
"""
    v_own(CP; hfac = 1, build = cpath_cached) -> CPath

The path of `v_path` from the path CP of the design: CP itself, or where CP is the vertical line the hyperbola of the
law of V alone (pi/4, or `fixed_phi` with a fixed start and below c = -60), built by build(Mo, c; phi, hfac).  The LM
statistic (`lmpath`) takes it with build = `cpath`: its prepared design is not that of `prep_cached`, under whose key
`cpath_cached` would keep its paths.
"""
function v_own(CP::CPath; hfac = 1.0, build = cpath_cached)
    isnan(CP.h) || return CP
    Mo = CP.Mo; cf = CP.c
    ph = (Mo.start == :fixed || cf < FAR_CMIN) ?
         blas1f(() -> fixed_phi(Mo, cf, CP.tstar, CP.tau, Float64(hfac), default_rmax(cf)), dense_la(Mo)) : pi / 4
    build(Mo, cf; phi = ph, hfac)
end
"""(1/pi) sum' Im{M_V(t_k) e^{-t_k y} t'(u_k)/t_k} (cdf = true; P(V > y)) or without the 1/t_k (the density) on the path
CP.  log M_V(t) = log P(t) - (1/2) sum_k log(+-s_k(t)) at s = 0 (`logMtilde`), each logarithm continued from node to node
(on a hyperbola above the unit root, `v_path`, a positive root can cross the negative real axis, where the principal
logarithm jumps by 2 pi i and M_V would change its sign; where no root crosses, as on every path that keeps the root
groups on their sides, the continued logarithms are the principal ones)"""
function vsum(CP::CPath, y::Float64, cdf::Bool)
    Mo = CP.Mo; acc = 0.0
    r = size(CP.R, 1); lg = zeros(ComplexF64, r)
    for k in eachindex(CP.t)
        fx = CP.fx[k]
        lM = CP.lP[k]
        for j in 1:r
            z = j <= Mo.rp ? CP.R[j, k] : -CP.R[j, k]
            l = log(z)
            k > 1 && (l += 2pi * im * round((imag(lg[j]) - imag(l)) / 2pi))
            lg[j] = l; lM -= l / 2
        end
        fx === nothing || (lM += Mo.g0^2 * fx.cf)
        t = CP.t[k]
        acc += CP.wt[k] * imag(exp(lM - t * y) * (cdf ? CP.dt[k] / t : CP.dt[k]))
    end
    acc
end

"""
    power_envelope(D, cs; level = 0.05) -> power (vector)

The asymptotic power envelope of Elliott, Rothenberg and Stock: at each c < 0 the power of the point-optimal level-`level`
test against c itself, which rejects for small U - aV: without deterministic terms and with an intercept (whose
point-optimal test GLS-demeans the data, which leaves the limit of the case without deterministic terms) a = c/2 and the
limit of no deterministic terms; with a linear trend the data GLS-detrended at cbar = c and a = c^2/(2 (c - 1)) (the
invariant statistic cbar^2 V + (1 - cbar) J(1)^2 of the detrended data).  The critical value is the level-quantile of
U - aV at c = 0 (`lin_quantile`), the envelope its power at c (`lin_cdf`); `level` for c >= 0.  D: trend(-1), trend(0)
or trend(1), the cases of DFLimit's power_envelope; for other designs the limit of the point-optimal statistic was not
derived, and they are refused.
"""
function power_envelope(D::Determ, cs::AbstractVector; level::Real = 0.05, kw...)
    K = nreg(D)
    trendcase = K == 2 && D.br == [0.0, 1.0] && size(D.C[1], 1) == 2 && D.C[1] == [1.0 0.0; 0.0 1.0]
    nodet = K == 0 || (K == 1 && D.br == [0.0, 1.0] && size(D.C[1], 1) == 1 && D.C[1] == fill(1.0, 1, 1))
    (trendcase || nodet) || throw(ArgumentError("power_envelope: the design must be trend(-1), trend(0) or trend(1) (the limit of the point-optimal statistic of other designs was not derived)"))
    0 < level < 1 || throw(ArgumentError("power_envelope: the level must lie in (0, 1)"))
    map(cs) do c
        c = Float64(c)
        isnan(c) && return NaN
        c < 0 || return Float64(level)
        c == -Inf && return 1.0
        if nodet
            a = c / 2
            b = lin_quantile(trend(-1), level, a, 0.0; b0 = 0.0, kw...)
            lin_cdf(trend(-1), [b], a, c; kw...)[1]
        else
            a = c * c / (2 * (c - 1))
            b = lin_quantile(D, level, a, 0.0; b0 = 0.0, gls = c, kw...)
            lin_cdf(D, [b], a, c; gls = c, kw...)[1]
        end
    end
end

"""the two parabola terms of {ADFmax <= x} = {|U + 1/2| <= 1/2 + x sqrt(V)} with R = U + a0: R <= x v + a0 minus R < -x v +
a0 - 1, on sqrt V <= -1/(2x) for x < 0 (the region is empty beyond), where both parabolas reach the same value, so that
their boundary terms cancel exactly and are left out"""
function adfmax_specs(x::Float64, a0::Float64)
    vm = x < 0 ? -0.5 / x : Inf
    [XSpec(0.0, x, a0; vmax = vm, w = 1.0, dq = 1.0, bnd = false), XSpec(0.0, -x, a0 - 1; vmax = vm, w = -1.0, dq = -1.0, bnd = false)]
end
"""the values of a distribution function or density at xs with the non-finite points set (0 at -Inf, 1 or 0 at Inf, NaN at
NaN) and the finite ones from fin(xs_finite)"""
function nonfinite(fin, xs::AbstractVector, cdf::Bool)
    x = Float64.(collect(xs)); out = similar(x)
    k = findall(isfinite, x)
    for j in eachindex(x)
        isfinite(x[j]) && continue
        out[j] = isnan(x[j]) ? NaN : x[j] > 0 ? (cdf ? 1.0 : 0.0) : 0.0
    end
    isempty(k) || (out[k] .= fin(x[k]))
    out
end

"""
    adfmax_cdf(D, xs, c = 0; start = :zero, gamma0 = 0, kw...) -> F (vector)

The limit of P(ADFmax <= x) for the larger of the Dickey-Fuller t-ratios of the forward regression (y_t on d(t/n) and
y_{t-1}) and of the reverse regression (y_t on d(t/n) and y_{t+1}) of Leybourne (as DFLimit's adfmax_cdf).  In the limit
both regressions have the denominator V and the numerators U and -U - 1, so that ADFmax => max(U, -U - 1)/sqrt(V) and
{ADFmax <= x} = {|U + 1/2| <= 1/2 + x sqrt(V)}: two parabolas in sqrt(V), each a term of the contour integral of the
t-ratio (`adfmax_specs`, `XSpec`).  The limit of the numerators is DFLimit's derivation for the constant and the linear
trend (cases DU, DT, SU, ST); for other designs and starts this is the same functional of the design's (U, V).
"""
adfmax_cdf(D::Determ, xs::AbstractVector, c::Real = 0.0; start::Symbol = :zero, gamma0::Real = 0.0, phi = nothing, hfac = 1.0,
           verified::Bool = false, vtol::Real = 1e-9, kw...) =
    nonfinite(xs, true) do x
        CP = user_path(D, c, start, gamma0, nothing, phi, hfac; what = "ADFmax (adfmax_cdf)")
        spec_eval(CP, [adfmax_specs(xk, CP.Mo.a) for xk in x], 1, verified, Float64(vtol); kw...)
    end
"""adfmax_pdf(D, xs, c = 0; kw...) -> f (vector): its density"""
adfmax_pdf(D::Determ, xs::AbstractVector, c::Real = 0.0; start::Symbol = :zero, gamma0::Real = 0.0, phi = nothing, hfac = 1.0,
           verified::Bool = false, vtol::Real = 1e-9, kw...) =
    nonfinite(xs, false) do x
        CP = user_path(D, c, start, gamma0, nothing, phi, hfac; what = "ADFmax (adfmax_pdf)")
        spec_eval(CP, [adfmax_specs(xk, CP.Mo.a) for xk in x], 2, verified, Float64(vtol); kw...)
    end
"""adfmax_quantile(D, p, c = 0; x0 = -2, tol = 1e-10, kw...) -> x (`newton_safe` on `adfmax_cdf`, `adfmax_pdf`)"""
function adfmax_quantile(D::Determ, p::Real, c::Real = 0.0; x0::Real = -2.0, tol::Real = 1e-10, start::Symbol = :zero, gamma0::Real = 0.0,
                         phi = nothing, hfac = 1.0, kw...)
    0 < p < 1 || throw(ArgumentError("adfmax_quantile: p must lie in (0, 1)"))
    search_args("adfmax_quantile", "x0", x0, tol)
    CP = user_path(D, c, start, gamma0, nothing, phi, hfac; what = "the quantiles of ADFmax (adfmax_quantile)")
    FF(x) = (r = contour_eval_specs(line_refined(CP, [x]), [adfmax_specs(x, CP.Mo.a)]; kw...); (r[1][1], r[2][1]))
    newton_safe(FF, p, x0, 1.0; tol)
end

"""
    region_prob(D, bnd, c = 0; kw...) -> P

The probability P{U <= g(V)} of a region of the plane below a curve (as DFLimit's ltratio_region), for the boundaries
that the contour integral delivers: bnd = (type = "line", a, b) (g(v) = b + a v, `lin_cdf`) and (type = "parabola", x,
c0 = 0) (g(v) = x sqrt(v) + c0 v, the t-ratio, `dist_cdf`).  The weighted-average-power boundary ("wap") and a general
function ("fun") are refused: the kernels of the contour integral are in closed form for boundaries that are quadratic in
sqrt(V), and a general g would need a further numerical integration over V inside the integrals over s and t (or the
conditional integral representation, which conditions on V and is not used here).
"""
function region_prob(D::Determ, bnd, c::Real = 0.0; kw...)
    fld(f) = bnd isa AbstractDict ? get(bnd, f, get(bnd, String(f), nothing)) : (haskey(bnd, f) ? bnd[f] : nothing)
    ty = lowercase(String(fld(:type)))
    if ty == "line"
        isfinite(fld(:a)) || throw(ArgumentError("region_prob: the slope a of the line must be a finite real scalar"))
        return lin_cdf(D, [Float64(fld(:b))], Float64(fld(:a)), c; kw...)[1]
    elseif ty == "parabola"
        c0 = fld(:c0); c0 = c0 === nothing ? 0.0 : Float64(c0)
        return dist_cdf(D, [Float64(fld(:x))], c; c0, kw...)[1]
    elseif ty in ("wap", "fun")
        throw(ArgumentError("region_prob: the boundary type $ty is not available by the contour integral (its kernels are in closed form for boundaries quadratic in sqrt(V) only)"))
    end
    throw(ArgumentError("region_prob: unknown boundary type $ty"))
end

"""the logarithm along a closed circle, continued from its first point (where it is real); ok: the steps are resolved
and the circle closes (DFExact.circlog)"""
function circlog(Z::Vector{ComplexF64})
    n = length(Z); lz = log.(Z)
    ok = abs(imag(lz[1])) < 1e-8
    ph = Vector{Float64}(undef, n); ph[1] = imag(lz[1])
    for k in 2:n
        d = angle(exp(im * (imag(lz[k]) - imag(lz[k-1]))))
        abs(d) < pi / 3 || (ok = false)
        ph[k] = ph[k-1] + d
    end
    ok &= abs(ph[end] + angle(Z[1] / Z[end]) - ph[1]) < 1e-6
    [complex(real(lz[k]), ph[k]) for k in 1:n], ok
end

"""
    dist_moments(D, c = 0; p = 4, c0 = 0, start = :zero, gamma0 = 0, Nu = 64, h = 0.05) -> (m, err)

E tau^k, k = 1..p, of tau = (U - c0 V)/sqrt(V): E tau^k = Gamma(k/2)^{-1} int_0^inf s^{k/2-1} m_k(s) ds, m_k(s) = d^k/du^k
M(u, -s - c0 u) at u = 0.  The derivatives are Cauchy integrals on a circle of radius r(s) (trapezoidal rule, Nu nodes)
with the logarithm of the determinant continued around the circle from its real point u = r (a circle on which it does
not close encloses a zero, and its radius is halved); r(s) = 0.4 min |s_k(-s)| (the roots at the real point t = -s), at
most 0.4 (s + t_*)/|c0| with c0 != 0 so that -s - c0 u stays left of t_*, at most 0.25 with a fixed or stationary
start, and at most 2 for c < -20 (DFExact.moments_err, whose rules these are).  The integral over s is the trapezoidal
rule in log s with step h on [min(-60, -2c - 40), max(8, log(80|c|))]; err is the difference to the rule with step 2h.
"""
function dist_moments(D::Determ, c::Real = 0.0; p::Int = 4, c0::Real = 0.0, start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing,
                      Nu::Int = 64, h::Float64 = 0.05)
    D.detrend == :first && throw(ArgumentError(FIRST_MOM_MSG))
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))
    check_cmax(D, c, start, gamma0, "the moments of the t-ratio (dist_moments)")
    Mo = startmodel(D, c, start, gamma0, gls); c = Float64(c); c0 = Float64(c0)
    tlo = min(-60.0, -2c - 40); thi = c < -20 ? max(8.0, log(80 * abs(c))) : 8.0
    # (the grid grows with c above the unit root: 2c/h points, 4e5 at c = 1e4; beyond 1e6 an error)
    (thi - tlo) / h <= 1e6 || error("dist_moments: the grid of the integral in s would have $(round((thi - tlo) / h, sigdigits = 3)) points at c = $c (more than 1e6): c is beyond the range of the moments")
    svals = exp.(collect(tlo:h:thi)); ns = length(svals)
    smax = tstar(Mo, c)[1]
    wk = [cis(2pi * k / Nu) for k in 0:Nu-1]
    acc = zeros(p, ns)
    # (one BLAS thread: the factorizations of the transform are small, and OpenBLAS's threads competed with these; see `blas1`)
    blas1f(dense_la(Mo)) do
        Threads.@threads :dynamic for i in 1:ns
            W = MWork(Mo); Bf = fixbuf(Mo.r)
            s = svals[i]
            _, _, μ, _ = mgfparts(Mo, c, complex(-s), W)
            r = 0.4 / maximum(abs, μ)
            c0 != 0 && (r = min(r, 0.4 * (s + smax) / abs(c0)))
            Mo.start != :zero && (r = min(r, 0.25))
            c < -20 && (r = min(r, 2.0))
            isnan(Mo.P.cbar) || (r = min(r, 2.0))
            ok = false; L = ComplexF64[]
            for _ in 1:8
                U = r .* wk
                L0 = Vector{ComplexF64}(undef, Nu); LD = Vector{ComplexF64}(undef, Nu)
                # (with c0 = 0 every point of the circle has t = -s: Sigma(theta) and the data of the fixed start are computed
                # once, not Nu times)
                θk, fxk = logmgf_tpart!(Mo, c, complex(-s - c0 * U[1]), W)
                # (whether the circle in t = -s - c0 u reaches the cut t >= c^2/2 of theta: then see below)
                cut = c0 != 0 && -s + abs(c0) * r > c * c / 2
                for k in 1:Nu
                    tk = complex(-s - c0 * U[k])
                    (c0 != 0 && k > 1) && ((θk, fxk) = logmgf_tpart!(Mo, c, tk, W))
                    L0[k], LD[k] = logmgf_spart(Mo, c, U[k], tk, θk, fxk, W, Bf)
                    # (where the circle reaches the cut, the logarithm continued around it is that of the entire function
                    # e^{c + theta} det(...), real and positive at the real point u = r, and not that of the determinant
                    # alone, whose phase jumps with theta across the cut: where -s - c0 r > c^2/2, as at c = 0 with c0 < 0
                    # for every s below |c0| r, theta is imaginary at the real point, the check of the real start failed on
                    # every circle, and the moments refused (case DT at c = 0, c0 = -2).  Elsewhere the logarithm of the
                    # determinant, as before, bit for bit: the moments far from the unit root with c0 = c are converged to
                    # 1e-8 only, and the rounding of the other form moves them by 3e-9.  With c0 = 0 the circle is t = -s.)
                    cut && (LD[k] += c + θk; L0[k] += (c + θk) / 2)
                end
                ld, ok = circlog(exp.(LD .- real(LD[1])))
                L = L0 .- (ld .+ real(LD[1])) ./ 2
                ok && break
                r /= 2
            end
            ok || error("dist_moments: no circle in u free of zeros of the determinant at s = $s")
            f = exp.(L)
            for j in 1:p
                Mj = factorial(j) / (Nu * r^j) * real(sum(f[k] * wk[k]^(-j) for k in 1:Nu))
                acc[j, i] = s^(j / 2) * Mj
            end
        end
    end
    m = [h * sum(acc[j, :]) / gamma(j / 2) for j in 1:p]
    err = [abs(h * sum(acc[j, :]) - 2h * sum(acc[j, 1:2:ns])) / gamma(j / 2) for j in 1:p]
    m, err
end

# =====================================================================================================================
# Simulation of the regression
# =====================================================================================================================
"""regressor matrix d(t/n), t = 1..n"""
regmat(D::Determ, n::Int) = nreg(D) == 0 ? zeros(n, 0) : permutedims(reduce(hcat, [dval(D, t / n) for t in 1:n]))

"""
    simulate_tratio(D, n, nrep; c = 0, seed = 1, start = :zero, gamma0 = 0, c0 = 0, statistic = :tratio) -> Vector{Float64}

The t-ratio for phi = 1 + c0/n (statistic = :coef: n(phi^ - 1 - c0/n)) in the least squares regression of y_t on d(t/n) and y_{t-1}, t = 1..n, with y_t = phi y_{t-1} +
e_t, phi = 1 + c/n, y_0 = 0 (start = :zero), y_0 = gamma0 sqrt(n) (:fixed) or y_0 ~ N(0, 1/(1 - phi^2)) (:stationary,
the stationary law of the AR(1), drawn before the errors), and e_t independent N(0,1): by Frisch-Waugh-Lovell with the thin QR factor of the regressor
matrix; nrep replications on all threads, each with its own random stream (seed + thread chunk).
"""
function simulate_tratio(D::Determ, n::Int, nrep::Int; c::Real = 0.0, seed::Int = 1, start::Symbol = :zero, gamma0::Real = 0.0, c0::Real = 0.0,
                         statistic::Symbol = :tratio)
    statistic in (:tratio, :coef) || throw(ArgumentError("statistic must be :tratio or :coef"))
    D.detrend == :first && return simulate_first(D, n, nrep; c, seed, start, gamma0, c0, statistic)
    coef = statistic == :coef
    start in (:zero, :fixed, :stationary) || throw(ArgumentError("start must be :zero, :fixed or :stationary"))
    start == :stationary && c >= 0 && throw(ArgumentError("the stationary start needs c < 0"))
    X = regmat(D, n); K = size(X, 2)
    Q = K == 0 ? zeros(n, 0) : Matrix(qr(X).Q)
    phi = 1 + c / n
    out = Vector{Float64}(undef, nrep)
    nch = 4 * Threads.nthreads()
    chunks = collect(Iterators.partition(1:nrep, cld(nrep, nch)))
    Threads.@threads :dynamic for ic in eachindex(chunks)
        rng = Random.Xoshiro(seed * 1_000_003 + ic)
        e = zeros(n); yl = zeros(n); dy = zeros(n); qa = zeros(K); qb = zeros(K)
        for rep in chunks[ic]
            y = start == :zero ? 0.0 : start == :fixed ? gamma0 * sqrt(n) : randn(rng) / sqrt(1 - phi^2)
            randn!(rng, e)
            for i in 1:n
                yl[i] = y; y = phi * y + e[i]
            end
            @. dy = e + (phi - 1) * yl              # Delta y_t
            if K > 0
                mul!(qa, transpose(Q), yl); mul!(qb, transpose(Q), dy)
            end
            syy = dot(yl, yl) - dot(qa, qa)
            syd = dot(yl, dy) - dot(qa, qb)
            sdd = dot(dy, dy) - dot(qb, qb)
            bhat = syd / syy
            s2 = (sdd - bhat * syd) / (n - K - 1)
            out[rep] = coef ? n * bhat - c0 : (bhat - c0 / n) / sqrt(s2 / syy)
        end
    end
    out
end

"""the last t at which d(t/n) is in the regime before each jump point of P (the regime changes after T_B, as 1{t > T_B})"""
jumpdates(P::Prep, n::Int) = [findlast(t -> t / n <= lam, 1:n) for lam in P.jl]

"""
    simulate_first(D, n, nrep; c = 0, seed = 1, start = :zero, gamma0 = 0, c0 = 0, statistic = :tratio) -> Vector{Float64}

The statistic that detrends first (Perron and Vogelsang; `detrend_first`): y_t = phi y_{t-1} + e_t, phi = 1 + c/n, y_0 as in
`simulate_tratio`, e_t independent N(0,1); the residual ytilde of the least squares regression of y_t on d(t/n), t = 1..n;
the regression of Delta ytilde_t on ytilde_{t-1}, t = 2..n, with a dummy for the single observation t = T_B + 1 after each
jump of d (which removes that observation); the t-ratio of phi = 1 + c0/n (residual variance on n - 2 - (number of
jumps) degrees of freedom), or statistic = :coef, n(phi^ - 1 - c0/n).  `simulate_tratio` calls it for a design built by
`detrend_first`.
"""
function simulate_first(D::Determ, n::Int, nrep::Int; c::Real = 0.0, seed::Int = 1, start::Symbol = :zero, gamma0::Real = 0.0,
                        c0::Real = 0.0, statistic::Symbol = :tratio)
    statistic in (:tratio, :coef) || throw(ArgumentError("statistic must be :tratio or :coef"))
    start in (:zero, :fixed, :stationary) || throw(ArgumentError("start must be :zero, :fixed or :stationary"))
    start == :stationary && c >= 0 && throw(ArgumentError("the stationary start needs c < 0"))
    coef = statistic == :coef; c0f = Float64(c0); phi = 1 + Float64(c) / n; g0f = Float64(gamma0)
    X = regmat(D, n); K = size(X, 2)
    Q = K == 0 ? zeros(n, 0) : Matrix(qr(X).Q)
    skip = falses(n)
    for T in jumpdates(prep(D), n); skip[T+1] = true; end
    nd = count(skip)
    out = Vector{Float64}(undef, nrep)
    nch = 4 * Threads.nthreads()
    chunks = collect(Iterators.partition(1:nrep, cld(nrep, nch)))
    Threads.@threads :dynamic for ic in eachindex(chunks)
        rng = Random.Xoshiro(seed * 1_000_003 + ic)
        y = zeros(n); qa = zeros(K); yt = zeros(n)
        for rep in chunks[ic]
            u = start == :zero ? 0.0 : start == :fixed ? g0f * sqrt(n) : randn(rng) / sqrt(1 - phi^2)
            for i in 1:n; u = phi * u + randn(rng); y[i] = u; end
            if K > 0
                mul!(qa, transpose(Q), y); mul!(yt, Q, qa); @. yt = y - yt
            else
                yt .= y
            end
            syy = 0.0; syd = 0.0; sdd = 0.0
            for i in 2:n
                skip[i] && continue
                l = yt[i-1]; d = yt[i] - yt[i-1]
                syy += l * l; syd += l * d; sdd += d * d
            end
            b = syd / syy
            s2 = (sdd - b * syd) / (n - 2 - nd)
            out[rep] = coef ? n * b - c0f : (b - c0f / n) / sqrt(s2 / syy)
        end
    end
    out
end

"""
    simulate_gls(D, n, nrep; cbar = -13.5, c = 0, seed = 1, c0 = 0, statistic = :tratio) -> Vector{Float64}

The GLS-detrended Dickey-Fuller t-ratio (Elliott, Rothenberg and Stock): y_t = u_t (beta = 0; the statistic does not
depend on beta), u_t = phi u_{t-1} + e_t, phi = 1 + c/n, u_0 = 0, e_t independent N(0,1); the quasi-differences at
abar = 1 + cbar/n (the first observation as it is, y_t - abar y_{t-1} for t >= 2, the same for the regressors d(t/n)),
beta^ by least squares on them, ytilde_t = y_t - beta^' d(t/n), and the t-ratio of the regression of Delta ytilde_t on
ytilde_{t-1}, t = 2..n, without deterministic terms (residual variance on n - 2 degrees of freedom): with c0 the t-ratio
of phi = 1 + c0/n, (b - c0/n)/se(b); statistic = :coef the coefficient statistic n b - c0 = n(phi^ - 1 - c0/n).
"""
function simulate_gls(D::Determ, n::Int, nrep::Int; cbar::Real = -13.5, c::Real = 0.0, seed::Int = 1, c0::Real = 0.0,
                      statistic::Symbol = :tratio)
    statistic in (:tratio, :coef) || throw(ArgumentError("statistic must be :tratio or :coef"))
    coef = statistic == :coef; c0f = Float64(c0)
    X = regmat(D, n); K = size(X, 2)
    abar = 1 + cbar / n; phi = 1 + c / n
    Xa = copy(X); Xa[2:end, :] .-= abar .* X[1:end-1, :]
    F = qr(Xa); Q = Matrix(F.Q); Rm = F.R
    out = Vector{Float64}(undef, nrep)
    nch = 4 * Threads.nthreads()
    chunks = collect(Iterators.partition(1:nrep, cld(nrep, nch)))
    Threads.@threads :dynamic for ic in eachindex(chunks)
        rng = Random.Xoshiro(seed * 1_000_003 + ic)
        e = zeros(n); y = zeros(n); ya = zeros(n); yt = zeros(n); qa = zeros(K); bet = zeros(K)
        for rep in chunks[ic]
            randn!(rng, e)
            u = 0.0
            for i in 1:n; u = phi * u + e[i]; y[i] = u; end
            ya[1] = y[1]
            for i in 2:n; ya[i] = y[i] - abar * y[i-1]; end
            mul!(qa, transpose(Q), ya); bet .= Rm \ qa
            mul!(yt, X, bet); @. yt = y - yt
            syy = 0.0; syd = 0.0; sdd = 0.0
            for i in 2:n
                l = yt[i-1]; d = yt[i] - yt[i-1]
                syy += l * l; syd += l * d; sdd += d * d
            end
            b = syd / syy
            s2 = (sdd - b * syd) / (n - 2)
            out[rep] = coef ? n * b - c0f : (b - c0f / n) / sqrt(s2 / syy)
        end
    end
    out
end

end # module
