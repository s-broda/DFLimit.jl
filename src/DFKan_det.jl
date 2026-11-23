# =====================================================================================================================
# The devices of Bao and Kan for the general engine: its determinant with the branch fixed at each point, and the single
# inversion of the half-plane statistics (evaluated in the module DFDet, after src/DFDet.jl; src/DFDet_paper2.jl and the
# interface use it)
# =====================================================================================================================
# The method is that of Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments of the Least Squares
# Estimator in First-Order Autoregressions: O(1) Algorithms": the transform at each point with the branch of its square
# root from factors whose arguments are known, and one inversion along the ray of the half-plane by adaptive quadrature
# with the tail completed by integration by parts (the generic code of the module DFLimit.DFKan, src/DFKan_ratio.jl).
#
# The general engine writes M(s, t) = e^{-s/2} [e^{c + theta} det(I - 2 Q Sigma)]^{-1/2}, Q = s A + t B + gamma E11,
# Sigma = Sigma(theta) the covariance of the N functionals under a reference law, theta = (c^2 - 2t)^{1/2}, and continued
# the phase of the determinant along a grid (qgrid, coef_cdf_gp) or along the outer path.  Here it is fixed at the point
# by the lemma on accretive matrices of Bao and Kan: det(I - 2 Q Sigma) = det(K) det(Sigma) with K = Sigma^{-1} - 2 Q the
# precision matrix of the functionals under the tilt.  Both have a positive definite Hermitian part where (Re s, Re t)
# lies in the domain of convergence, in particular for imaginary (s, t), the arguments of every inversion of Gil-Pelaez
# type, for any design, start and detrending (A and B are real symmetric, so that an imaginary (s, t) adds a
# skew-Hermitian part only).  Elimination without pivoting of such a matrix has every pivot in the open right half-plane
# (each is a Schur complement of order one), so that the logarithm of the determinant, with its continuous argument, is
# the sum of the principal logarithms of the pivots.  (Bao and Kan take a Schur complement of order two at once, whose
# determinant is off the negative axis; pivots of order one extend this to any order.)  With a stationary start the
# extended covariance Sigma_x has det Sigma_x = v det Sigma, v = -1/(2c) > 0, and the pivots of Sigma serve.

import ..DFKan, ..DFExact
const BK = DFKan

"""the sum of the principal logarithms of the pivots of Gaussian elimination without pivoting (A is overwritten)"""
function pivlogs!(A::Matrix{ComplexF64})
    n = size(A, 1); s = zero(ComplexF64)
    @inbounds for k in 1:n
        p = A[k, k]; s += log(p)
        ip = inv(p)
        for i in k+1:n
            l = A[i, k] * ip
            l == 0 && continue
            for j in k+1:n; A[i, j] = muladd(-l, A[k, j], A[i, j]); end
        end
    end
    s
end

"""
    DetCF(Mo, c)

The joint transform of the pair (U, V) of a model of the general engine (a design with a zero or stationary start, any
detrending) at c, with the branch of the square root of the determinant fixed at each point from the pivots (the method
of Bao and Kan; see the header of this file): `DFKan.logcf(F, t1, t2)` = log E exp(i t1 A + i t2 B), A = U - cV, B = V,
as for the classical cases' `DFKan.LimCF`.  Not thread-safe (one work space): one per task.
"""
struct DetCF
    Mo::Model
    c::Float64
    W::MWork
    w0::Float64
    K::Matrix{ComplexF64}
    S::Matrix{ComplexF64}
    Sl::Matrix{ComplexF64}
end
function DetCF(Mo::Model, c::Real)
    Mo.start in (:zero, :stationary) || throw(ArgumentError("DetCF: a zero or stationary start (the fixed start's transform has a further term)"))
    N = Mo.N; n = Mo.P.N
    DetCF(Mo, Float64(c), MWork(Mo), Mo.a, zeros(ComplexF64, N, N), zeros(ComplexF64, n, n), zeros(ComplexF64, N, N))
end
"""log det(I - 2 Q Sigma) at (s, t) with the continuous argument, from the pivots of K = M Sigma_x^{-1} and of Sigma
(W holds Sigma and Sigma_x for t, `logmgf_tpart!`)"""
function detlog(F::DetCF, s::ComplexF64, t::ComplexF64, θ::ComplexF64)
    Mo = F.Mo; W = F.W
    W.sd === nothing || return detlog_s(F, W.sd, s, t, θ)::ComplexF64
    stat = Mo.start == :stationary
    M = stat ? buildMx!(W, Mo, F.c, s, t, θ) : buildM!(W, Mo.P, F.c, s, t, θ)
    Sx = stat ? W.Sx : W.S
    # K^T = Sigma_x^{-1} M^T (Sigma_x is complex symmetric); the pivots of K^T without pivoting are those of K
    N = Mo.N
    @inbounds for j in 1:N, i in 1:N; F.K[i, j] = M[j, i]; end
    copyto!(F.Sl, Sx)
    ldiv!(lu!(F.Sl), F.K)
    ld = pivlogs!(F.K)
    copyto!(F.S, W.S)
    ld += pivlogs!(F.S)
    stat ? ld + log(-1 / (2 * F.c)) : ld
end
"""the sum of the principal logarithms of the pivots of a static matrix by elimination without pivoting"""
function spivlogs(A::SMatrix{N,N,ComplexF64}) where {N}
    M = MMatrix(A); s = zero(ComplexF64)
    @inbounds for k in 1:N
        p = M[k, k]; s += log(p); ip = inv(p)
        for i in k+1:N
            l = M[i, k] * ip
            for j in k+1:N; M[i, j] = muladd(-l, M[k, j], M[i, j]); end
        end
    end
    s
end
"""`detlog` with the static matrices of the design (N <= STATIC_MAX): K^T = Sigma_x^{-1} M^T by the static LU, then the
pivots of K^T and of Sigma without pivoting; no allocation"""
function detlog_s(F::DetCF, sd::SData{N}, s::ComplexF64, t::ComplexF64, θ::ComplexF64) where {N}
    Mo = F.Mo; W = F.W; stat = Mo.start == :stationary
    Sx = SMatrix{N,N,ComplexF64}(stat ? W.Sx : W.S)
    M = stat ? buildMx_s(sd, Sx, F.c, s, t, θ) : buildM_s(sd, Sx, F.c, s, t, θ)
    KT = ssolve(slu(Sx), transpose(M))
    ld = spivlogs(KT)
    if stat
        ld += spivlogs(SMatrix{N - 1,N - 1,ComplexF64}(W.S)) + log(-1 / (2 * F.c))
    else
        ld += spivlogs(Sx)
    end
    ld
end
"""log M(s, t) = log E exp(sU + tV) at one point (s, t), imaginary or with real parts in the domain of convergence"""
function logmgf(F::DetCF, s::ComplexF64, t::ComplexF64)
    θ, _ = logmgf_tpart!(F.Mo, F.c, t, F.W)
    -s / 2 - (F.c + θ) / 2 - detlog(F, s, t, θ) / 2
end
BK.logcf(F::DetCF, t1::ComplexF64, t2::ComplexF64) = logmgf(F, im * t1, im * (t2 - F.c * t1))
BK.logcf(F::DetCF, t1::Real, t2::Real) = BK.logcf(F, ComplexF64(t1), ComplexF64(t2))

"""means and variances of (A, B) from the log characteristic function next to the origin (no jets for this transform)"""
function BK.ratio_scale(F::DetCF)
    mv(a, b) = begin
        h = 1e-2
        L = BK.logcf(F, a * h, b * h)
        while abs(L) > 0.05 && h > 1e-12; h /= 4; L = BK.logcf(F, a * h, b * h); end
        imag(L) / h, -2 * real(L) / h^2
    end
    Eu, Vu = mv(1.0, 0.0); Ev, Vv = mv(0.0, 1.0)
    (Eu, Ev, sqrt(max(Vv, 0.0)), Vu)
end
"""
The tail int_S^inf psi(u) e^{-i w u} du of the single inversion for a transform without jets, psi(u) = phi(u alpha, u
beta) e^{-i b u} e^{i w u}/u (cdf) or without the 1/u (the density): the derivatives of the slowly varying factor at S
from the polynomial through 9 points of the real axis about S (the arguments stay imaginary), four terms of the
expansion -> (the integral, the first omitted term)
"""
function tail_int(F::DetCF, alpha::Float64, beta::Float64, b::Float64, w::Float64, S::Float64, cdf::Bool)
    m = 9; L = S / 16
    xk = [cos(pi * (k - 1) / (m - 1)) for k in 1:m]
    g = [(u = S + L * x; exp(BK.logcf(F, complex(alpha * u), complex(beta * u)) - im * b * u + im * w * (u - S)) / (cdf ? u : 1.0)) for x in xk]
    cf = [x^(j - 1) for x in xk, j in 1:m] \ g                 # g(S + L x) = sum_j cf[j] x^(j-1)
    acc = zero(ComplexF64); iw = inv(im * w); p = iw; KT = 4
    for k in 0:KT-1
        acc += BK.fact(k) * cf[k+1] / L^k * p; p *= iw
    end
    acc, abs(BK.fact(KT) * cf[KT+1] / L^KT * p)
end
function BK.tail_cdf(F::DetCF, alpha::Float64, beta::Float64, b::Float64, w::Float64, S::Float64, ::Val{N}) where {N}
    T, e = tail_int(F, alpha, beta, b, w, S, true)
    imag(T), e
end
"""
    half_pdf(F, alpha, beta, b; tol, w0) -> (f, err, evals)

The density of alpha A + beta B at b for the general engine's transform: (1/pi) int_0^inf Re[phi(u alpha, u beta)
e^{-i b u}] du, over [0, S] by adaptive quadrature over decades and beyond S = 400/|w| (w = w0 alpha + b) by the expansion
of `tail_int` (S doubled at most five times while its first omitted term exceeds tol/10), as `DFKan.half_cdf`.
"""
function half_pdf(F::DetCF, alpha::Float64, beta::Float64, b::Float64; tol::Float64 = BK.TOL, w0::Float64 = F.w0)
    sc = BK.ray_scale(F, alpha, beta)
    f = z -> begin
        u = z * sc
        L = BK.logcf(F, complex(alpha * u), complex(beta * u))
        y = sc * exp(real(L)) * cos(imag(L) - b * u)
        isfinite(y) ? y : 0.0
    end
    w = w0 * alpha + b
    T = 0.0; et = 0.0; S = Inf
    if abs(w) > 1e-7
        S = 400 / abs(w)
        for dbl in 0:5
            Tc, et = tail_int(F, alpha, beta, b, w, S, false); T = real(Tc)
            (et <= tol / 10 || dbl == 5) && break
            S *= 2
        end
    end
    I, e, nev = BK.decades(f, S / sc, tol, max(tol, 1e-14))
    (I + T) / pi, (e + et) / pi, nev
end

"""
    MemoCF(F)

The transform F with a memo of its values along one ray (the quantile searches of the point-optimal statistic and the
envelope evaluate the same ray at many b: the nodes of the adaptive quadrature that recur are evaluated once), and its
scale computed once.
"""
struct MemoCF{T}
    F::T
    c::Float64
    w0::Float64
    memo::Dict{Tuple{Float64,Float64},ComplexF64}
    scale::NTuple{4,Float64}
end
MemoCF(F) = MemoCF(F, F.c, F.w0, Dict{Tuple{Float64,Float64},ComplexF64}(), BK.ratio_scale(F))
function BK.logcf(M::MemoCF, t1::ComplexF64, t2::ComplexF64)
    (imag(t1) == 0 && imag(t2) == 0) || return BK.logcf(M.F, t1, t2)
    get!(() -> BK.logcf(M.F, t1, t2), M.memo, (real(t1), real(t2)))
end
BK.ratio_scale(M::MemoCF) = M.scale
BK.tail_cdf(M::MemoCF, alpha::Float64, beta::Float64, b::Float64, w::Float64, S::Float64, v::Val) = BK.tail_cdf(M.F, alpha, beta, b, w, S, v)

# ---------------------------------------------------------------------------------------------------------------------
# The point-optimal statistic and the coefficient statistic by the single inversion (src/DFDet_paper2.jl, coef_cdf_gp)
# ---------------------------------------------------------------------------------------------------------------------
"""the transform of the point-optimal statistic's quadratic form (`prep_pt`) at c"""
ptcf(P::Prep, c::Real) = DetCF(model(P), c)
"""
    pt_cdf1(F, cbar, q, tol) -> (P, err)

P(Q <= q) for the point-optimal statistic, Q = U + 1/2 + cbar^2 V = A + (cbar^2 + c) B + 1/2 (the transform F of
`ptcf`), by the single inversion of Bao and Kan with the tail completed by integration by parts: no grid, so that neither
the distance of the singularity of the transform from the real line (far above the unit root) nor the size of q limits it.
"""
function pt_cdf1(F, cbar::Float64, q::Float64, tol::Float64)
    isnan(q) && return NaN, 0.0
    isinf(q) && return (q > 0 ? 1.0 : 0.0), 0.0
    P, e, _ = BK.half_cdf(F, 1.0, cbar^2 + F.c, q - 0.5; tol, w0 = 0.5)
    clamp(P, 0.0, 1.0), e
end
"""the density of the point-optimal statistic at q by the single inversion (`half_pdf`) -> (f, err)"""
function pt_pdf1(F, cbar::Float64, q::Float64, tol::Float64)
    isfinite(q) || return (isnan(q) ? NaN : 0.0), 0.0
    f, e, _ = half_pdf(F isa MemoCF ? F.F : F, 1.0, cbar^2 + F.c, q - 0.5; tol, w0 = 0.5)
    max(f, 0.0), e
end
"""
    pt_quantile1(F, cbar, p, tol) -> (q, err of F at q)

The p-quantile of the point-optimal statistic under the transform F: the Illinois variant of regula falsi on `pt_cdf1`
to the width 1e-12 times the scale of Q, from the bracket E Q -/+ its standard deviation (grown as needed), with the
transform values along the ray memoized across the search (`MemoCF`).
"""
function pt_quantile1(F::DetCF, cbar::Float64, p::Float64, tol::Float64)
    M = MemoCF(F)
    Eu, Ev, sdv, Vu = M.scale
    bq = cbar^2 + F.c
    m = Eu + bq * Ev + 0.5; s = sqrt(Vu + bq^2 * sdv^2)
    el = Ref(0.0)
    q = BK.illinois(x -> begin P, e = pt_cdf1(M, cbar, x, tol); el[] = e; P - p end, max(m - s, 0.0), m + s, 1e-12 * s; grow = s)
    q, el[]
end
"""
    gcoef_cdf1(F, y, c0, tol) -> (P, err)

P[(U - c0 V)/V <= y] = P[A + (c - c0 - y) B <= 0] for a design of the general engine (the transform F, a zero or stationary
start, any detrending) by the single inversion of Bao and Kan (`coef_cdf_gp`).
"""
function gcoef_cdf1(F, y::Float64, c0::Float64, tol::Float64)
    isnan(y) && return NaN, 0.0
    isinf(y) && return (y > 0 ? 1.0 : 0.0), 0.0
    P, e, _ = BK.half_cdf(F, 1.0, F.c - (c0 + y), 0.0; tol)
    clamp(P, 0.0, 1.0), e
end

"""
the largest c at which the single inversion is the default of `coef_cdf_gp`: compared in November 2026 with the contour
integral (`coef_cdf`) for an intercept, linear and quadratic trends, Perron's model C with one and two breaks, a stationary
start and GLS detrending, c from -30 to 5: within 9.6e-13 (2.5e-11 with GLS detrending at cbar = -13.5, where the grid
differs from the contour integral by 1.6e-10), 3 to 58 ms a value against 120 to 160 ms for the grid; at c = 10 one value
takes 7 s, and the grid of before, with its refusal, stays the default above c = 5
"""
const GP_INV_CMAX = 5.0
"""
    coef_cdf_gp(D, ys, c = 0; c0 = 0, start = :zero, gamma0 = 0, gls = nothing, method = :auto, tol = 1e-12, err = false, N, umax) -> P

The distribution function of the limit (U - c0 V)/V of n(phi^ - phi0), phi0 = 1 + c0/n, by the inversion formula of
Gil-Pelaez, P(U - aV <= 0) = 1/2 - (1/pi) int_0^inf Im phi(u)/u du, a = c0 + y: an independent check of `coef_cdf`, which
uses the contour integral.  By default (a zero or stationary start, any detrending) the single inversion of Bao and Kan
("Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1)
Algorithms"): adaptive quadrature over decades with the branch of the square root of the determinant from the pivots at
each point (`DetCF`) and the tail completed by integration by parts (`gcoef_cdf1`); `tol` is its absolute tolerance, `err
= true` returns `(P, errors)`, and a warning names the points at which the estimate exceeds it.  `method = :grid` (or N,
umax given, and a fixed start gamma0 != 0) is the trapezoidal rule of before on the fixed grid in u = sinh(v),
`coef_cdf_grid`, with its refusal where the grid does not resolve the singularity of phi nearest the real line
(`GP_EDGE`); it is also the default above c = GP_INV_CMAX (5).
"""
function coef_cdf_gp(D::Determ, ys::AbstractVector, c::Real = 0.0; c0::Real = 0.0, start::Symbol = :zero, gamma0::Real = 0.0, gls = nothing,
                     method::Symbol = :auto, N = nothing, umax = nothing, tol::Real = BK.TOL, err::Bool = false)
    method in (:auto, :inversion, :grid) || throw(ArgumentError("coef_cdf_gp: method must be :auto, :inversion (the single inversion of Bao and Kan) or :grid"))
    fixed = start == :fixed && gamma0 != 0
    (method === :inversion && (fixed || N !== nothing || umax !== nothing)) && throw(ArgumentError("coef_cdf_gp: method = :inversion takes a zero or stationary start and no N, umax"))
    if method === :grid || N !== nothing || umax !== nothing || fixed || (method === :auto && !(c <= GP_INV_CMAX))
        v = coef_cdf_grid(D, ys, c; c0, start, gamma0, gls, N = N === nothing ? 60000 : Int(N), umax = umax === nothing ? 4000.0 : umax)
        return err ? (v, fill(NaN, length(v))) : v
    end
    check_cmax(D, c, start, gamma0, "the coefficient statistic (coef_cdf_gp)")
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))
    F = DetCF(startmodel(D, c, start, gamma0, gls), c)
    R = [gcoef_cdf1(F, Float64(y), Float64(c0), Float64(tol)) for y in ys]
    P = first.(R); E = last.(R)
    bad = count(e -> !(e <= tol), E)
    bad > 0 && @warn "coef_cdf_gp: the error estimate of the single inversion exceeds the tolerance $tol at $bad of $(length(E)) points (largest $(maximum(E))); method = :grid gives the fixed grid"
    err ? (P, E) : P
end

# ---------------------------------------------------------------------------------------------------------------------
# The outer contour
# ---------------------------------------------------------------------------------------------------------------------
"""
    contour_parts(P, c, t) -> (logP, positive roots, negative roots)

For the zero start and least squares detrending, at one point t with Im t > 0 (anywhere in the upper half-plane, also
beyond t_*): the roots of D(., t) in their groups (positive type: Im s < 0; negative type: Im s > 0) and log P(t), P(t)^{-2}
= D_V(t) prod_+ mu_k prod_- (-mu_k), as `rootpath` continues them along a path from the real point.  Least squares
detrending makes D_V(t)/D_K(t) the determinant of G^{-1} F'(I - 2tC)^{-1} F (G the Gram matrix of the regressors, C the
covariance operator of the process), a matrix with a positive definite imaginary part for Im t > 0; its eigenvalues lie
in the upper half-plane, and the continuous argument is the sum of their principal arguments (Bao and Kan's lemma in
another form); D_K(t) is the case without deterministic terms, `DFExact.argK`.
"""
function contour_parts(P::Prep, c::Float64, t::ComplexF64)
    imag(t) > 0 || throw(ArgumentError("contour_parts: Im t > 0"))
    tp = tparts(P, c, t)
    θ = tp.theta; γ = (c + θ) / 2; N = size(tp.M0, 1)
    E = Matrix{ComplexF64}(I, N, N)
    for j in 1:N; E[1, j] -= 2γ * tp.Sigma[1, j]; end
    E[1, 1] = m011(c, t, θ)
    ν = eigvals(tp.M0 / E)                                     # those of G^{-1} F'(I - 2tC)^{-1} F, and ones
    aK = DFExact.argK(c, t)[1]
    phD = aK + sum(angle, ν)
    rts = tp.roots
    pos = [s for s in rts if imag(s) < 0]; neg = [s for s in rts if imag(s) >= 0]
    ph = phD + sum(s -> angle(1 / s), pos; init = 0.0) + sum(s -> angle(-1 / s), neg; init = 0.0)
    Lr = tp.logDV + sum(s -> log(1 / s), pos; init = 0.0im) + sum(s -> log(-1 / s), neg; init = 0.0im)
    L = complex(real(Lr), imag(Lr) + 2pi * round((ph - imag(Lr)) / (2pi)))
    -L / 2, pos, neg
end

# ---------------------------------------------------------------------------------------------------------------------
# One-dimensional anchors for the contour engine
# ---------------------------------------------------------------------------------------------------------------------
"""
    RootCF(mu, a)

The transform of U alone: E exp(i u U) = e^{-i a u} prod_k (1 - i u mu_k)^{-1/2}, mu_k the reciprocal roots of D(., 0) (U +
a is a sum of r independent chi-square(1) variables with the weights mu_k/2, the chi-square representation of Bao and Kan
for any design).  Each factor is off the negative real axis: principal branches.
"""
struct RootCF
    mu::Vector{Float64}
    w0::Float64
end
function rootlcf(F::RootCF, t1)
    acc = (-im * F.w0) * t1
    for m in F.mu; acc -= BK.logp(1 - (im * m) * t1) / 2; end
    acc
end
BK.logcf(F::RootCF, t1::ComplexF64, t2::ComplexF64) = rootlcf(F, t1)
BK.logcf(F::RootCF, t1::BK.Jet{N}, t2::BK.Jet{N}) where {N} = rootlcf(F, t1)
"""
    zero_cdf(D, c = 0; start = :zero, tol = 1e-12) -> (P, err, evals)

P[U <= 0] for a design of the general engine: the distribution function at zero of the t-ratio (c0 = 0) and of the
coefficient statistic n(phi^ - 1), from the roots at t = 0 alone by a single inversion with its tail completed by
integration by parts (Bao and Kan's chi-square representation).  An exact one-dimensional value of the double integral at
x = 0, and a cheap check of it.
"""
function zero_cdf(D::Determ, c::Real = 0.0; start::Symbol = :zero, tol::Float64 = BK.TOL)
    Mo = startmodel(D, c, start, 0.0, nothing)
    μ = mgfparts(Mo, Float64(c), complex(0.0), MWork(Mo))[3]
    BK.half_cdf(RootCF(real.(μ), Mo.a), 1.0, 0.0, 0.0; tol)
end
