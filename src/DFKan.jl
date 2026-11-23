"""
    DFKan

The numerical devices of Yong Bao and Raymond Kan, from their paper "Exact Finite-Sample Distribution and Moments of the
Least Squares Estimator in First-Order Autoregressions: O(1) Algorithms" and its MATLAB programs, for the limits of this
package.  The method of every function here is theirs; the code is a port to the limit pair (U, V) of the package:

  * the joint characteristic function of the limit pair in closed forms that are free of cancellation on both sides of
    the unit root, with the branch of the square root settled at each point from factors whose arguments are known
    (no tracking along a grid), evaluated in Taylor-mode arithmetic (`Jet`), so that derivatives of any order come out
    of the same assembly (`logcf`);
  * the single inversion for statistics that reject on a half-plane (the coefficient statistic, the linear form U - aV,
    the power envelope): adaptive quadrature over decades and the tail completed by repeated integration by parts, the
    derivatives read off a jet (`half_cdf`, `ratio_cdf`, `ratio_pdf`, `ratio_icdf`, `ratio_mom`), each value with an
    error estimate; the package's `coef_cdf`, `coef_pdf`, `coef_icdf`, `coef_mom`, `ltratio_lincdf`, `power_envelope`
    and `ltratio_mom` take these for the six classical cases (src/merged.jl, the functions `coef_cdf1` and so on here);
  * the branch of P(t), M_0(t) and the labels of the roots of the line integral at a single point t (`tparts_point`,
    file DFKan_branch.jl), and of the general engine's determinant (`DetCF`, file DFKan_det.jl);
  * the moments of the t-ratio at a local null value from one jet per node (`tratio_mom1`), and the distribution
    function of the t-ratio at zero in closed form (`ray_cdf`).

The pair is (U, V) of the package: the t-ratio is (U - c0 V)/sqrt(V) and the coefficient statistic (U - c0 V)/V.  Bao
and Kan write A = U - cV and B = V, so that E exp(i t1 A + i t2 B) = M(i t1, i (t2 - c t1)), M the joint moment
generating function of (U, V).  The case codes are those of the package; the start g0 of the package corresponds to their
parameter by par = g0 (case DK), c g0 (case DU) and c^2 g0 (case DT).
"""
module DFKan

using StaticArrays, QuadGK, SpecialFunctions, LinearAlgebra

# =====================================================================================================================
# Taylor-mode arithmetic: truncated power series in one variable (after the jets of Bao and Kan)
# =====================================================================================================================
"""
    Jet{N}

A truncated power series sum_k c[k+1] x^k, k = 0..N-1, with complex coefficients: the value of a quantity and its first
N-1 derivatives at a point, scaled as Taylor coefficients.  Sums, products, quotients, `sqrt`, `exp` and the logarithms
`logp` (principal argument of the constant coefficient) and `logc` (a prescribed argument) are exact in the retained
coefficients.  Only the constant coefficient of a logarithm or a root carries a branch.
"""
struct Jet{N} <: Number
    c::SVector{N,ComplexF64}
end
Jet{N}(x::Number) where {N} = Jet{N}(SVector{N,ComplexF64}(ntuple(k -> k == 1 ? ComplexF64(x) : zero(ComplexF64), N)))
"""the jet of the variable itself at the point x0, scaled by lam: x0 + lam x"""
jetvar(::Val{N}, x0::Number, lam::Number = 1.0) where {N} =
    Jet{N}(SVector{N,ComplexF64}(ntuple(k -> k == 1 ? ComplexF64(x0) : k == 2 ? ComplexF64(lam) : zero(ComplexF64), N)))
Base.zero(::Type{Jet{N}}) where {N} = Jet{N}(zero(SVector{N,ComplexF64}))
Base.one(::Type{Jet{N}}) where {N} = Jet{N}(1.0)
Base.promote_rule(::Type{Jet{N}}, ::Type{<:Number}) where {N} = Jet{N}
Base.convert(::Type{Jet{N}}, x::Jet{N}) where {N} = x
Base.convert(::Type{Jet{N}}, x::Number) where {N} = Jet{N}(x)

"""the constant coefficient (the value)"""
val(a::Jet) = a.c[1]
val(a::Number) = a
"""the coefficient of order k"""
coef(a::Jet, k::Int) = a.c[k+1]
"""the same jet with another constant coefficient"""
setval(a::Jet{N}, v::Number) where {N} = Jet{N}(setindex(a.c, ComplexF64(v), 1))
setval(a::Number, v::Number) = ComplexF64(v)

Base.:+(a::Jet{N}, b::Jet{N}) where {N} = Jet{N}(a.c + b.c)
Base.:-(a::Jet{N}, b::Jet{N}) where {N} = Jet{N}(a.c - b.c)
Base.:-(a::Jet{N}) where {N} = Jet{N}(-a.c)
Base.:+(a::Jet{N}, b::Number) where {N} = setval(a, a.c[1] + b)
Base.:+(b::Number, a::Jet{N}) where {N} = setval(a, a.c[1] + b)
Base.:-(a::Jet{N}, b::Number) where {N} = setval(a, a.c[1] - b)
Base.:-(b::Number, a::Jet{N}) where {N} = setval(-a, b - a.c[1])
Base.:*(a::Jet{N}, b::Number) where {N} = Jet{N}(a.c * ComplexF64(b))
Base.:*(b::Number, a::Jet{N}) where {N} = Jet{N}(a.c * ComplexF64(b))
Base.:/(a::Jet{N}, b::Number) where {N} = Jet{N}(a.c / ComplexF64(b))
function Base.:*(a::Jet{N}, b::Jet{N}) where {N}
    m = MVector{N,ComplexF64}(undef)
    @inbounds for k in 1:N
        acc = zero(ComplexF64)
        for j in 1:k; acc += a.c[j] * b.c[k-j+1]; end
        m[k] = acc
    end
    Jet{N}(SVector(m))
end
function Base.:/(a::Jet{N}, b::Jet{N}) where {N}
    m = MVector{N,ComplexF64}(undef)
    ib = inv(b.c[1])
    @inbounds for k in 1:N
        acc = a.c[k]
        for j in 2:k; acc -= b.c[j] * m[k-j+1]; end
        m[k] = acc * ib
    end
    Jet{N}(SVector(m))
end
Base.:/(a::Number, b::Jet{N}) where {N} = Jet{N}(a) / b
Base.inv(a::Jet{N}) where {N} = Jet{N}(1.0) / a
function Base.sqrt(a::Jet{N}) where {N}
    m = MVector{N,ComplexF64}(undef)
    m[1] = sqrt(a.c[1]); h = inv(2 * m[1])
    @inbounds for k in 2:N
        acc = a.c[k]
        for j in 2:k-1; acc -= m[j] * m[k-j+1]; end
        m[k] = acc * h
    end
    Jet{N}(SVector(m))
end
function Base.exp(a::Jet{N}) where {N}
    m = MVector{N,ComplexF64}(undef)
    m[1] = exp(a.c[1])
    @inbounds for k in 1:N-1
        acc = zero(ComplexF64)
        for j in 1:k; acc += j * a.c[j+1] * m[k-j+1]; end
        m[k+1] = acc / k
    end
    Jet{N}(SVector(m))
end
"""the logarithm of a jet with the constant coefficient v (the branch is the caller's)"""
function logv(a::Jet{N}, v::Number) where {N}
    m = MVector{N,ComplexF64}(undef)
    m[1] = v; ia = inv(a.c[1])
    @inbounds for k in 1:N-1
        acc = k * a.c[k+1]
        for i in 1:k-1; acc -= i * m[i+1] * a.c[k-i+1]; end
        m[k+1] = acc * ia / k
    end
    Jet{N}(SVector(m))
end
"""log with the principal argument of the constant coefficient"""
logp(a::Jet) = logv(a, log(a.c[1]))
logp(a::Number) = log(complex(a))
"""log with the prescribed (continuous) argument ph of the constant coefficient"""
logc(a::Jet, ph::Real) = logv(a, complex(log(abs(a.c[1])), ph))
logc(a::Number, ph::Real) = complex(log(abs(a)), ph)
"""sum_k cf[k] z^(k-1) by Horner's rule (z a number or a jet)"""
function jser(cf::AbstractVector{Float64}, z::T) where {T}
    y = zero(T)
    @inbounds for k in length(cf):-1:1; y = y * z + cf[k]; end
    y
end

# =====================================================================================================================
# The joint characteristic function of the limit pair: closed forms of Bao and Kan (their AR1LIM_LOGCF)
# =====================================================================================================================
const FACT = [exp(loggamma(k + 1.0)) for k in 0:170]                      # FACT[k+1] = k!
fact(k::Int) = FACT[k+1]
const KMS = 26
const SER_COS = [(-1.0)^k / fact(2k) for k in 0:KMS]                       # cos th = sum (-1)^k zeta^k/(2k)!
const SER_SIN = [(-1.0)^k / fact(2k + 1) for k in 0:KMS]                   # sin th/th
const SER_SH = [(-1.0)^k / (4.0^k * fact(2k + 1)) for k in 0:KMS]          # sin(th/2)/(th/2)
const SER_CD = [(-1.0)^(k + 1) * 2k / (4.0^k * fact(2k + 1)) for k in 1:KMS+1]   # (Sh - Ch)/zeta
const SER_A = [2 * (-1.0)^k / fact(2k + 2) for k in 0:KMS]                 # Sh^2 = 2(1 - cos th)/zeta
const SER_D1 = let KM = 40                                                 # (tan(u)/u - 1)/th^2, u = th/2, in w = zeta/4
    sn = [(-1.0)^k / fact(2k + 1) for k in 0:KM]; cs = [(-1.0)^k / fact(2k) for k in 0:KM]
    a = zeros(KM + 1)
    for i in 0:KM
        acc = sn[i+1]
        for j in 1:i; acc -= cs[j+1] * a[i-j+1]; end
        a[i+1] = acc
    end
    [a[i+1] / 4 for i in 1:KM]
end
const KMT = 40
const SER_F1 = [4j / (4.0^j * fact(2j + 1)) for j in 1:KMT+2]              # e^{vt/2} h1/vt^3 in vt^2
const SER_F2 = [(12 / fact(2j) - 4 / fact(2j - 1) - 12 / fact(2j + 1)) / 4.0^j for j in 2:KMT+3]   # e^{vt/2} h2/vt^5
const SER_CH = [1 / (4.0^k * fact(2k)) for k in 0:KMT]                     # cosh(vt/2)
const SER_SHC = [1 / (2.0^(2k + 1) * fact(2k + 1)) for k in 0:KMT]         # sinh(vt/2)/vt

"""
    LimCF(cas, c, par = 0)

The limit pair of one of the six cases at c: `cas` one of "DK", "DU", "DT" (fixed start; `par` is the parameter of Bao
and Kan: gamma_0 in case DK, varkappa = b + c gamma_0 in case DU, varkappa in case DT) or "SK", "SU", "ST" (stationary
start, c < 0).  `logcf(F, t1, t2)` is the logarithm of E exp(i t1 A + i t2 B), A = U - cV, B = V.
"""
struct LimCF
    det::Int            # 1: no deterministic terms, 2: intercept, 3: intercept and trend
    stat::Bool
    c::Float64
    par::Float64
    w0::Float64         # the slope of the dominant linear phase along the inversion paths
    ag::Vector{Float64} # the series of e^{vt} G/vt^8 of the trend cases
end
function LimCF(cas::AbstractString, c::Real, par::Real = 0.0)
    cas = uppercase(cas)
    cas in ("DK", "DU", "DT", "SK", "SU", "ST") || throw(ArgumentError("LimCF: the case must be one of DK, DU, DT, SK, SU, ST"))
    stat = cas[1] == 'S'; det = cas[2] == 'K' ? 1 : cas[2] == 'U' ? 2 : 3
    c = Float64(c); par = Float64(par)
    (isfinite(c) && isfinite(par)) || throw(ArgumentError("LimCF: c and par must be finite"))
    (stat && c >= 0) && throw(ArgumentError("LimCF: a stationary start requires c < 0"))
    ag = Float64[]
    if det == 3
        gam = (24c^2 * (c - 2) * (c^2 - 3c + 6), -24c^2 * (c - 2) * (c^2 - 3c + 6), 8 * (c^5 - 4c^4 + 6c^3 + 6c^2 - 36c + 36),
               -c^5 + 24c^3 - 120c^2 + 288c - 288, c^4 - 8c^3 + 32c^2 - 72c + 72)
        ag = [sum(2 * gam[j+1] / fact(2k - j) for j in 0:4) for k in 4:KMT+4]
    end
    LimCF(det, stat, c, stat ? 0.0 : par, (det == 1 && !stat) ? (1 + par^2) / 2 : 0.5, ag)
end
"""the limit pair of the package's case at c with the fixed start g0 (par = g0, c g0, c^2 g0 in cases DK, DU, DT)"""
function limcf(cas::AbstractString, c::Real, g0::Real = 0.0)
    cas = uppercase(cas)
    par = cas == "DK" ? g0 : cas == "DU" ? c * g0 : cas == "DT" ? c^2 * g0 : 0.0
    LimCF(cas, c, par)
end

"""vartheta - c without cancellation: for c >= 0 as -(theta^2 + c^2)/(vartheta + c), otherwise directly"""
@inline vtmc(vt, zc, c::Float64) = c >= 0 ? -zc / (vt + c) : vt - c
"""vartheta - omega (omega = c + i t1) without cancellation: on the explosive side as tau0/(vartheta + omega) where that
is the larger of the two"""
@inline function vtmo(vt, zc, c::Float64, om, t1, tau0)
    nu = vtmc(vt, zc, c) - im * t1
    if c > 0
        vpo = vt + om
        abs(val(vpo)) > abs(val(nu)) && (nu = tau0 / vpo)
    end
    nu
end
"""
The continuous argument of cos(theta) as a function of zeta = theta^2 (Re zeta <= 0): the principal argument shifted by
the multiple of 2 pi that brings it within pi/2 of -sign(Im theta) Re theta; equal to Im vartheta + Arg(1 + e^{-2
vartheta}), vartheta the principal root of -zeta (Bao and Kan).
"""
function argcos(z::ComplexF64)
    th = sqrt(z); tr = real(th); ti = imag(th)
    e = exp(-2 * abs(ti))
    cz = complex(cos(tr) * (1 + e), -sin(tr) * sign(ti) * (1 - e))
    a = angle(cz); target = -sign(ti) * tr
    a + 2pi * round((target - a) / (2pi))
end
"""
The continuous phase of the factor of the demeaned cases under the square root as the sum of the principal arguments of
three factors, each off the closed negative real axis (Bao and Kan): W = (theta/2) cot(theta/2), the limit of the
rank-one factor, and the limit of the border Schur complement.
"""
function argtau4(z0::ComplexF64, zc0::ComplexF64, D2::ComplexF64, T0::ComplexF64)
    D1 = abs(z0) < 4 ? jser(SER_D1, z0 / 4) : (2D2 - 1) / z0
    rho = 1 + zc0 * D1
    Sinf = T0 * D2 / (2rho)
    W = 1 / (2D2)
    angle(W) + angle(rho) + angle(Sinf)
end
"""B0(vartheta) of the trend cases expanded about vartheta = c (sg = 1) or -c (sg = -1), u = vartheta -/+ c"""
@inline function bpoly(u, c::Float64, sg::Int)
    c4 = c^4 - 8c^3 + 32c^2 - 72c + 72
    if sg > 0
        b0 = c * (c - 2)^2 * (c^2 - 6c + 12)^2
        b1 = 3c^6 - 40c^5 + 232c^4 - 744c^3 + 1344c^2 - 1152c + 288
        b2 = 3c^5 - 32c^4 + 152c^3 - 408c^2 + 576c - 288
    else
        b0 = -c^7
        b1 = 3c^6 - 8c^5 + 8c^4 + 24c^3 - 96c^2 + 288
        b2 = -3c^5 + 16c^4 - 40c^3 + 24c^2 + 144c - 288
    end
    ((c4 * u + b2) * u + b1) * u + b0
end

"""
    logcf(F, t1, t2) -> log E exp(i t1 A + i t2 B)

The logarithm of the joint characteristic function of A = U - cV and B = V at (t1, t2), numbers or jets (complex
arguments give the moment generating function).  With omega = c + i t1, theta^2 = 2 i t2 - 2 i c t1 - c^2 and tau_0 =
t1^2 - 2 i t2, every function of theta is even, a function of theta^2: for small |theta^2| by power series, elsewhere
through vartheta = (-theta^2)^{1/2} and e^{-vartheta} in forms in which nothing of order one is a difference of
quantities of order e^{vartheta}, on either side of the unit root.  The branch of the square root is fixed at the point:
each argument under the root is a product of factors whose continuous arguments are known (cos theta = e^{vartheta}
(1 + e^{-2 vartheta})/2; the other factors are off the closed negative real axis).  A port of AR1LIM_LOGCF of Bao and Kan.
"""
function _logcf(F::LimCF, t1::T, t2::T) where {T<:Union{ComplexF64,Jet}}
    c = F.c
    om = c + im * t1                              # omega
    zc = 2im * t2 - 2im * c * t1                  # theta^2 + c^2
    zeta = zc - c^2                               # theta^2
    tau0 = t1 * t1 - 2im * t2
    F.det == 3 && return trendlim(F, t1, t2, om, zc, tau0)
    z0 = val(zeta)
    small = abs(z0) < 4
    if F.det == 1
        if small
            C = jser(SER_COS, zeta); S = jser(SER_SIN, zeta)
            f = C - om * S
            g = 1 - om * S / C
            lC = logc(C, argcos(z0)); lg = logp(g)
            Kf = -(tau0 * S) / (2 * f)
        else
            vt = sqrt(-zeta); E2 = exp(-2 * vt)
            nu = vtmo(vt, zc, c, om, t1, tau0)
            Bk = nu + E2 * (vt + om)
            oe = 1 + E2
            lC = vt + logp(oe) - log(2.0)         # log cos theta: Im vartheta + Arg(1 + e^{-2 vartheta})
            lg = logp(Bk / (vt * oe))
            Kf = -(tau0 * (1 - E2)) / (2 * Bk)
        end
        L = -om / 2 - (lC + lg) / 2
        return F.stat ? L - logp(1 + Kf / c) / 2 : (F.par == 0 ? L : L + F.par^2 * Kf)
    end
    tau1 = 2 * (t1 * t1 + im * c^2 * t1 - 2im * c * t2)
    tau2 = im * (c - 2) * c * t1 - tau0 + c^3
    if small
        C = jser(SER_COS, zeta); S = jser(SER_SIN, zeta); Sh = jser(SER_SH, zeta); ShmCh = jser(SER_CD, zeta)
        tau10 = tau1 + 2c * zeta; tau20 = tau2 - zeta
        Nz = zero(T)
        @inbounds for k in KMS:-1:1                # N/zeta, N = 2 tau1 Sh^2 - 4 c^2 cos th + 4 tau2 sin th/th
            Nz = Nz * zeta + (2 * SER_A[k+1] * tau10 + 4 * SER_SIN[k+1] * tau20 + (-4c * SER_A[k] - 4c^2 * SER_COS[k+1] + 4 * SER_SIN[k]))
        end
        T0 = Nz / (Sh * Sh)
        lS = logp(Sh)
        Q = 2 * (tau0 * ShmCh * Sh) / Nz
        D2 = val(Sh)^2 / (2 * val(S))
    else
        vt = sqrt(-zeta); E = exp(-vt); E2 = E * E
        nu = vtmo(vt, zc, c, om, t1, tau0)
        Q0 = (2 - c) * (1 + c) * (t1 * t1) + 2im * c * (c - 1) * t2 + nu * tau2 - im * (t1 * tau0)
        vt2 = vt * vt
        Den = Q0 - 2 * (E * tau1) + E2 * (tau1 - c^2 * vt2 - tau2 * vt)
        omE = 1 - E
        T0 = -2 * Den / (vt2 * (omE * omE))
        lS = vt / 2 + logp(omE) - logp(vt)        # sin(th/2)/(th/2) = e^{vt/2}(1 - E)/vt (single-valued)
        br = (1 - vt / 2) - E * (1 + vt / 2)
        Q = (tau0 * br * omE) / Den
        D2 = val(omE) / ((1 + val(E)) * val(vt))
    end
    lT = logc(T0, argtau4(z0, val(zc), D2, val(T0)))
    L = log(2.0) - om / 2 - lS - lT / 2
    F.stat ? L - logp(1 - c * Q) / 2 : (F.par == 0 ? L : L - F.par^2 * Q)
end
const Scalar = Union{Real,Complex}
logcf(F::LimCF, t1::ComplexF64, t2::ComplexF64) = _logcf(F, t1, t2)
logcf(F::LimCF, t1::Jet{N}, t2::Jet{N}) where {N} = _logcf(F, t1, t2)
logcf(F::LimCF, t1::Scalar, t2::Scalar) = _logcf(F, ComplexF64(t1), ComplexF64(t2))
logcf(F::LimCF, t1::Jet{N}, t2::Scalar) where {N} = _logcf(F, t1, Jet{N}(t2))
logcf(F::LimCF, t1::Scalar, t2::Jet{N}) where {N} = _logcf(F, Jet{N}(t1), t2)

"""the cases with an intercept and a trend (the trend limit of Bao and Kan): the phase is that of the case without
deterministic terms corrected by the principal arguments of three factors that are off the closed negative real axis"""
function trendlim(F::LimCF, t1::T, t2::T, om::T, zc::T, tau0::T) where {T}
    c = F.c
    Kb = -4 * (c^2 - 3c + 3) * tau0 + c * (c - 2) * (c^2 - 6c + 12) * (im * t1)
    zv = c^2 - zc                                 # vartheta^2
    z0 = val(zv)
    if abs(val(zc - c^2)) < 16
        f1 = jser(SER_F1, zv); f2 = jser(SER_F2, zv); Ge = jser(F.ag, zv)
        f12 = f1 * f2
        Psi = f12 * Kb + Ge                       # e^{vartheta} D3/vartheta^8
        Lr = -om / 2 - logp(Psi) / 2 + log(2.0) / 2
        Q = (tau0 * f12) / (2 * Psi)
        g11 = val(f1) / (2 * jser(SER_CH, z0)); gll = -val(f2) / (24 * jser(SER_SHC, z0))
    else
        vt = sqrt(zv); E = exp(-vt); E2 = E * E
        if c >= 0
            vmc = -zc / (vt + c); vpc = vt + c
            Bp = bpoly(vmc, c, 1); Bm = bpoly(-vmc, c, -1)
        else
            vpc = -zc / (vt - c); vmc = vt - c
            Bp = bpoly(vpc, c, -1); Bm = bpoly(-vpc, c, 1)
        end
        B1 = (c - 3) * (c^2 - 4c + 6) * zv - 6 * (c^3 - 5c^2 + 12c - 12)
        h1 = (vt - 2) + E * (vt + 2)
        h2 = -(zv - 6 * vt + 12) + E * (zv + 6 * vt + 12)
        h12 = h1 * h2
        if c > 0                                  # the explosive side: every term of the order of D3 itself
            K1 = c * (c - 2) * (c^2 - 6c + 12)
            Pp = vt * (zv + 8 * vt + 24) + 24
            Qc = (c - 3) * vt - 3 * (c - 2)
            vQ = vmc * Qc
            D3 = 8 * (vQ * vQ) + K1 * (vmc * (8 * ((zv - 6) * E) + Pp * E2)) - (tau0 * h12) * (K1 / (vt + om) + 4 * (c^2 - 3c + 3)) +
                 8 * (zc * (B1 * E)) - vpc * (Bm * E2)
        else
            D3 = h12 * Kb + (vmc * Bp + 8 * (zc * (B1 * E)) - vpc * (Bm * E2))
        end
        Lr = -(om + vt) / 2 - (logp(D3) - 8 * logp(vt)) / 2 + log(2.0) / 2
        Q = (tau0 * h12) / (2 * D3)
        v0 = val(vt); e0 = val(E)
        g11 = val(h1) / (v0^3 * (1 + e0)); gll = -val(h2) / (12 * v0^4 * (1 - e0))
    end
    # the continuous phase from the factors
    t10 = val(t1); t20 = val(t2)
    L0 = _logcf(LimCF(1, false, c, 0.0, 0.5, F.ag), t10, t20)
    z1 = val(zc)
    tr = z1 * (g11 + 12 * gll)
    dt = (12 * z1^2 + 144 * t10^2) * g11 * gll
    sq = sqrt(tr^2 - 4dt)
    k1 = (tr + sq) / 2; k2 = (tr - sq) / 2
    rho = exp(2 * (L0 - val(Lr))) / (1 + tr + dt)
    ph = imag(L0) - (angle(1 + k1) + angle(1 + k2) + angle(rho)) / 2
    L = setval(Lr, complex(real(val(Lr)), ph))
    F.stat ? L - logp(1 + c^3 * Q) / 2 : (F.par == 0 ? L : L + F.par^2 * Q)
end

"""log M(s, t) = log E exp(sU + tV), the joint moment generating function of the package's pair, from `logcf`"""
logmgf(F::LimCF, s, t) = logcf(F, -im * s, -im * (t + F.c * s))

include("DFKan_ratio.jl")
include("DFKan_branch.jl")
include("DFKan_det.jl")

end # module
