# =====================================================================================================================
# The general engine's determinant with the branch fixed at each point
# =====================================================================================================================
# The general engine (DFDet) writes M(s, t) = e^{-s/2} [e^{c + theta} det(I - 2 Q Sigma)]^{-1/2}, Q = s A + t B + gamma E11,
# Sigma = Sigma(theta) the covariance of the N functionals under a reference law, theta = (c^2 - 2t)^{1/2}, and continues
# the phase of the determinant along a grid (qgrid, coef_cdf_gp, cf_gls) or along the outer path (rootpath, the line
# integral's provider).  Three ways to have it at one point, after the lemma on accretive matrices of Bao and Kan:
#
#  (i)  Pivots (`DetCF`).  det(I - 2 Q Sigma) = det(K) det(Sigma) with K = Sigma^{-1} - 2 Q the precision matrix of the
#       functionals under the tilt.  Both have a positive definite Hermitian part where (Re s, Re t) lies in the domain
#       of convergence, in particular for imaginary (s, t), the arguments of every inversion of Gil-Pelaez type, for any
#       design, start and detrending (the matrices A and B are real symmetric, so that an imaginary (s, t) adds a
#       skew-Hermitian part only).  Elimination without pivoting of such a matrix has every pivot in the open right
#       half-plane (each is a Schur complement of order one), so that the continuous argument of the determinant is the
#       sum of the principal arguments of the pivots.  (Bao and Kan take a Schur complement of order two at once, whose
#       determinant is off the negative axis; pivots of order one extend this to any order.)
#  (ii) Half-planes (`contour_parts`), for the whole upper half-plane of t, beyond t_*, where (i) fails: least squares
#       detrending makes D_V(t)/D_K(t) the determinant of G^{-1} F'(I - 2tC)^{-1} F (G the Gram matrix of the regressors, C
#       the covariance operator of the process), a matrix with a positive definite imaginary part for Im t > 0; its
#       eigenvalues lie in the upper half-plane, and the continuous argument is the sum of their principal arguments.
#       The roots of positive type lie in the lower half-plane and those of negative type in the upper one.
#  (iii) The line (`line_phase`): (i) at s = 0 with the roots' reciprocals, each in its half-plane.

import ..DFDet
const G = DFDet

"""the sum of the principal arguments of the pivots of Gaussian elimination without pivoting (A is overwritten)"""
function pivargs!(A::AbstractMatrix{ComplexF64})
    n = size(A, 1); s = 0.0
    @inbounds for k in 1:n
        p = A[k, k]; s += angle(p)
        for i in k+1:n
            l = A[i, k] / p
            for j in k+1:n; A[i, j] -= l * A[k, j]; end
        end
    end
    s
end

"""
    DetCF(Mo, c)

The joint transform of the pair (U, V) of a model of the general engine (a design with its start: zero or stationary; any
detrending) at c, with the branch of the square root of the determinant fixed at each point (pivots): `logmgf(F, s, t)`
for imaginary (s, t), `logcf(F, t1, t2)` as for `LimCF`.  Not thread-safe (one work space).
"""
struct DetCF
    Mo::G.Model
    c::Float64
    W::G.MWork
    Bf::Matrix{ComplexF64}
    w0::Float64
    K::Matrix{ComplexF64}
    S::Matrix{ComplexF64}
end
function DetCF(Mo::G.Model, c::Real)
    Mo.start in (:zero, :stationary) || throw(ArgumentError("DetCF: zero and stationary starts"))
    N = Mo.N; n = Mo.P.N
    DetCF(Mo, Float64(c), G.MWork(Mo), G.fixbuf(Mo.r), Mo.a, zeros(ComplexF64, N, N), zeros(ComplexF64, n, n))
end
"""the continuous argument of det(I - 2 Q Sigma) at (s, t) from the pivots of Sigma and of the precision matrix K"""
function detphase(F::DetCF, s::ComplexF64, t::ComplexF64, θ::ComplexF64)
    Mo = F.Mo; W = F.W
    stat = Mo.start == :stationary
    M = stat ? G.buildMx!(W, Mo, F.c, s, t, θ) : G.buildM!(W, Mo.P, F.c, s, t, θ)
    Sx = stat ? W.Sx : W.S
    copyto!(F.K, M / lu(Sx))                                  # K = M Sigma^{-1}
    copyto!(F.S, W.S)                                         # (det Sigma_ext = v det Sigma, v > 0)
    pivargs!(F.K) + pivargs!(F.S)
end
"""log M(s, t) = log E exp(sU + tV) at one point (s, t), imaginary or with real parts in the domain of convergence"""
function logmgf(F::DetCF, s::ComplexF64, t::ComplexF64)
    θ, fx = G.logmgf_tpart!(F.Mo, F.c, t, F.W)
    L0, ld = G.logmgf_spart(F.Mo, F.c, s, t, θ, fx, F.W, F.Bf)
    ph = detphase(F, s, t, θ)
    L0 - complex(real(ld), imag(ld) + 2pi * round((ph - imag(ld)) / (2pi))) / 2
end
logcf(F::DetCF, t1::ComplexF64, t2::ComplexF64) = logmgf(F, im * t1, im * (t2 - F.c * t1))
logcf(F::DetCF, t1::Real, t2::Real) = logcf(F, ComplexF64(t1), ComplexF64(t2))

"""means and variances of (A, B) from the log characteristic function next to the origin (no jets for this transform)"""
function ratio_scale(F::DetCF)
    mv(a, b) = begin
        h = 1e-2
        L = logcf(F, a * h, b * h)
        while abs(L) > 0.05 && h > 1e-12; h /= 4; L = logcf(F, a * h, b * h); end
        imag(L) / h, -2 * real(L) / h^2
    end
    Eu, Vu = mv(1.0, 0.0); Ev, Vv = mv(0.0, 1.0)
    (Eu, Ev, sqrt(max(Vv, 0.0)), Vu)
end
"""the tail of `half_cdf` for a transform without jets: the derivatives of the slowly varying factor from the polynomial
through 9 points of the real axis about S (the arguments stay imaginary), four terms of the expansion"""
function tail_cdf(F::DetCF, alpha::Float64, beta::Float64, b::Float64, w::Float64, S::Float64, ::Val{N}) where {N}
    m = 9; L = S / 16
    xk = [cos(pi * (k - 1) / (m - 1)) for k in 1:m]
    g = [(u = S + L * x; exp(logcf(F, complex(alpha * u), complex(beta * u)) - im * b * u + im * w * (u - S)) / u) for x in xk]
    cf = [x^(j - 1) for x in xk, j in 1:m] \ g                 # g(S + L x) = sum_j cf[j] x^(j-1)
    acc = zero(ComplexF64); iw = inv(im * w); p = iw; KT = 4
    for k in 0:KT-1
        acc += fact(k) * cf[k+1] / L^k * p; p *= iw
    end
    imag(acc), abs(fact(KT) * cf[KT+1] / L^KT * p)
end

"""
    pt_cdf(D, q, cbar, c = 0; tol = 1e-10) -> (P, err, evals)

The limit of P(P_T <= q) for the point-optimal statistic against cbar of the design D under c (DFDet.pt_cdf), by adaptive
quadrature of the inversion formula with the branch at each point and the tail by integration by parts: no grid, so
that neither the distance of the singularity of the transform from the real line (far above the unit root) nor the size
of q limits it.
"""
function pt_cdf(D::G.Determ, q::Real, cbar::Real, c::Real = 0.0; tol::Float64 = 1e-10, F::DetCF = ptcf(D, cbar, c))
    # (Q = U + 1/2 + cbar^2 V has no constant: the phase of its transform at q is -u q, and far above the unit root, where
    # q is large, the integral is cut at 400/q and completed by integration by parts)
    half_cdf(F, 1.0, Float64(cbar)^2 + Float64(c), Float64(q) - 0.5; tol, w0 = 0.5)
end
"""the transform of the point-optimal statistic's quadratic form (DFDet.prep_pt) at c"""
ptcf(D::G.Determ, cbar::Real, c::Real) = DetCF(G.model(G.prep_pt(D, Float64(cbar))), c)

"""
    gcoef_cdf(D, y, c = 0; c0 = 0, start = :zero, gls = nothing, tol = 1e-10) -> (P, err, evals)

The coefficient statistic of a design of the general engine (DFDet.coef_cdf, DFDet.coef_cdf_gp, mza_cdf) by the single
inversion: P[U - (c0 + y) V <= 0], adaptive quadrature on [0, S] and the tail by integration by parts.
"""
function gcoef_cdf(D::G.Determ, y::Real, c::Real = 0.0; c0::Real = 0.0, start::Symbol = :zero, gls = nothing, tol::Float64 = 1e-10,
                   F::DetCF = DetCF(G.startmodel(D, c, start, 0.0, gls), c))
    half_cdf(F, 1.0, Float64(c) - (Float64(c0) + Float64(y)), 0.0; tol)
end
"""P(U - aV <= b) for a design of the general engine by the single inversion (DFDet.lin_cdf)"""
function glin_cdf(D::G.Determ, b::Real, a::Real, c::Real = 0.0; start::Symbol = :zero, gls = nothing, tol::Float64 = 1e-10,
                  F::DetCF = DetCF(G.startmodel(D, c, start, 0.0, gls), c))
    half_cdf(F, 1.0, Float64(c) - Float64(a), Float64(b); tol)
end

# ---------------------------------------------------------------------------------------------------------------------
# The line integral's provider and the outer contour
# ---------------------------------------------------------------------------------------------------------------------
"""
    line_phase(lm, t, v) -> the imaginary part of log P(t)^{-2} at the point t of the line Re t = tau

For the provider `lm` of the general engine's line integral (DFDet.LineModel) and its transform v = (log D_V, mu_+, mu_-)
at t: the argument of D_V from the pivots (at s = 0), those of mu_+ and -mu_- principal (each in the right half-plane on
the line).
"""
function line_phase(lm::G.LineModel, t::ComplexF64, v::NTuple{3,ComplexF64})
    lock(LINELOCK) do
        F, memo = get!(() -> (DetCF(lm.Mo, lm.c), Dict{ComplexF64,Float64}()), LINECF, objectid(lm))::Tuple{DetCF,Dict{ComplexF64,Float64}}
        ph = get!(memo, t) do
            θ = sqrt(complex(lm.c^2 - 2t))
            G.logmgf_tpart!(lm.Mo, lm.c, t, F.W)
            imag(θ) + detphase(F, zero(ComplexF64), t, θ)
        end
        lm.r == 1 ? ph + angle(v[2]) : ph + angle(v[2]) + angle(-v[3])
    end
end
const LINECF = Dict{UInt,Tuple{DetCF,Dict{ComplexF64,Float64}}}()
const LINELOCK = ReentrantLock()
"""switch the general engine's line integral to the branch at each point (true) or back to the continuation (false)"""
line_pointwise!(on::Bool = true) = (G.LINE_PHASE[] = on ? line_phase : nothing; on)

"""
    contour_parts(P, c, t) -> (logP, positive roots, negative roots)

For the zero start and least squares detrending, at one point t with Im t > 0 (anywhere in the upper half-plane, also
beyond t_*): the roots of D(., t) in their groups (positive type: Im s < 0; negative type: Im s > 0) and log P(t), P(t)^{-2}
= D_V(t) prod_+ mu_k prod_- (-mu_k), as DFDet.rootpath continues them along a path from the real point.
"""
function contour_parts(P::G.Prep, c::Float64, t::ComplexF64)
    imag(t) > 0 || throw(ArgumentError("contour_parts: Im t > 0"))
    tp = G.tparts(P, c, t)
    θ = tp.theta; γ = (c + θ) / 2; N = size(tp.M0, 1)
    E = Matrix{ComplexF64}(I, N, N)
    for j in 1:N; E[1, j] -= 2γ * tp.Sigma[1, j]; end
    E[1, 1] = G.m011(c, t, θ)
    ν = eigvals(tp.M0 / E)                                     # those of G^{-1} F'(I - 2tC)^{-1} F, and ones
    aK = argK(c, t)[1]
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
    for m in F.mu; acc -= logp(1 - (im * m) * t1) / 2; end
    acc
end
logcf(F::RootCF, t1::ComplexF64, t2::ComplexF64) = rootlcf(F, t1)
logcf(F::RootCF, t1::Jet{N}, t2::Jet{N}) where {N} = rootlcf(F, t1)
"""
    zero_cdf(D, c = 0; start = :zero, tol = 1e-10) -> (P, err, evals)

P[U <= 0] for a design of the general engine: the distribution function at zero of the t-ratio (c0 = 0) and of the
coefficient statistic n(phi^ - 1), from the roots at t = 0 alone by a single inversion with its tail completed by
integration by parts.  An exact one-dimensional value of the double integral at x = 0, and a cheap check of it.
"""
function zero_cdf(D::G.Determ, c::Real = 0.0; start::Symbol = :zero, tol::Float64 = 1e-10)
    Mo = G.startmodel(D, c, start, 0.0, nothing)
    μ = G.mgfparts(Mo, Float64(c), complex(0.0), G.MWork(Mo))[3]
    half_cdf(RootCF(real.(μ), Mo.a), 1.0, 0.0, 0.0; tol)
end
