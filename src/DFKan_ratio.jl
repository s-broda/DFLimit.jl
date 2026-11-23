# =====================================================================================================================
# The single inversion with tail completion (the engines RATIO_CDF, RATIO_PDF, RATIO_ICDF, RATIO_MOM of Bao and Kan)
# =====================================================================================================================
# For a statistic that rejects on a half-plane, {alpha A + beta B <= b}, the distribution function is one integral of the
# characteristic function along the ray (t1, t2) = u (alpha, beta) (Gil-Pelaez).  Along the ray the characteristic function
# has the dominant linear phase -w0 u (w0 = 1/2 from the constant -1/2 of U, plus g0^2/2 with a fixed start and no
# deterministic terms) and, next to beta = c alpha, where alpha A + beta B is of chi-square type, decays only like a power
# of u.  The integral is therefore taken over [0, S] by adaptive quadrature, decade by decade, and the rest by repeated
# integration by parts,
#     int_S^inf psi(u) e^{-i w u} du = e^{-i w S} sum_k psi^{(k)}(S)/(i w)^{k+1},      w = w0 + b,
# with psi the slowly varying remainder and its derivatives read off one jet at S.

"""means and variances of (A, B) from jets of the log characteristic function at the origin: (EA, EB, sd B, Var A)"""
function ratio_scale(F)
    Lt = logcf(F, Jet{3}(0.0), jetvar(Val(3), 0.0))
    Ls = logcf(F, jetvar(Val(3), 0.0), Jet{3}(0.0))
    Ev = real(coef(Lt, 1) / im); Vv = real(-2 * coef(Lt, 2))
    Eu = real(coef(Ls, 1) / im); Vu = real(-2 * coef(Ls, 2))
    (Eu, Ev, isfinite(Vv) ? sqrt(max(Vv, 0.0)) : Inf, Vu)
end
"""the scale of u on the ray u (alpha, beta): 1/sd of alpha A + beta B without the covariance (so that it never degenerates)"""
function ray_scale(F, alpha::Float64, beta::Float64)
    _, _, sdv, Vu = ratio_scale(F)
    vv = alpha^2 * Vu + beta^2 * sdv^2
    (isfinite(vv) && vv > 0) || (vv = Vu)
    1 / sqrt(vv)
end

"""int_0^b f by adaptive Gauss-Kronrod quadrature over [0, 10], [10, 100], ..., so that a power-law stretch of the
integrand is resolved on every scale; with b = Inf it stops after two consecutive negligible decades -> (I, err, evals)"""
function decades(f::FN, b::Float64, tol::Float64, rtol::Float64) where {FN}
    I = 0.0; e = 0.0; a = 0.0; small = 0; nev = 0
    k = 1
    while a < b
        hi = isinf(b) ? 10.0^k : (10.0^(k + 0.5) >= b ? b : 10.0^k)
        Ik, ek, n = quadgk_count(f, a, hi; atol = tol / 10, rtol = rtol, maxevals = 10^6)
        I += Ik; e += ek; a = hi; k += 1; nev += n
        (abs(Ik) + ek <= tol / 20) ? (small += 1) : (small = 0)
        (small >= 2 || k > 31) && break
    end
    I, e, nev
end

"""
The tail int_S^inf Im[phi(u alpha, u beta) e^{-i b u}]/u du by repeated integration by parts, the derivatives of the slowly
varying factor from a jet of order N - 1 at S -> (T, the first omitted term)
"""
function tail_cdf(F, alpha::Float64, beta::Float64, b::Float64, w::Float64, S::Float64, ::Val{N}) where {N}
    sj = jetvar(Val(N), S)
    ph = exp(logcf(F, alpha * sj, beta * sj) - (im * b) * sj)
    g = ph * exp(jetvar(Val(N), 0.0, im * w)) / sj
    acc = zero(ComplexF64); iw = inv(im * w); p = iw
    @inbounds for k in 0:N-2
        acc += fact(k) * coef(g, k) * p; p *= iw
    end
    imag(acc), abs(fact(N - 1) * coef(g, N - 1) * p)
end

"""
    half_cdf(F, alpha, beta, b; tol = 1e-10, K = 8, tail = true, w0 = F.w0) -> (P, err, evals)

P[alpha A + beta B <= b] for the pair of F (alpha != 0): 1/2 - (1/pi) int_0^inf Im[phi(u alpha, u beta) e^{-i b u}]/u du,
over [0, S] by adaptive quadrature and beyond S by the expansion of `tail_cdf` (S = 400/|w|, w = w0 alpha + b the slope of
the dominant phase; doubled, at most five times, while the first omitted term exceeds tol/10).  Next to w = 0 (the
edge of the support of U, b = -w0 with beta = c alpha) S grows like 1/|w|; at w = 0 itself the tail has no oscillation to
integrate by parts and the quadrature runs on, decade by decade.  err is the estimate of the
quadrature plus the first omitted term of the tail.
"""
function half_cdf(F, alpha::Float64, beta::Float64, b::Float64; tol::Float64 = 1e-10, K::Int = 8, tail::Bool = true, w0::Float64 = F.w0)
    sc = ray_scale(F, alpha, beta)
    f = z -> begin
        u = z * sc
        L = logcf(F, complex(alpha * u), complex(beta * u))
        y = exp(real(L)) * sin(imag(L) - b * u) / z
        isfinite(y) ? y : 0.0
    end
    w = w0 * alpha + b
    T = 0.0; et = 0.0; S = Inf
    if tail && abs(w) > 1e-7
        S = 400 / abs(w)
        for dbl in 0:5
            T, et = tail_cdf(F, alpha, beta, b, w, S, Val(K + 1))
            (et <= tol / 10 || dbl == 5) && break
            S *= 2
        end
    end
    I, e, nev = decades(f, S / sc, tol, max(tol, 1e-14))
    0.5 - (I + T) / pi, (e + et) / pi, nev
end

"""
    ratio_cdf(F, r; tol = 1e-10) -> (P, err, evals)

P[A/B <= r] = P[A - r B <= 0] (B > 0), the single inversion along t2 = -r t1.
"""
ratio_cdf(F, r::Real; kw...) = half_cdf(F, 1.0, -Float64(r), 0.0; kw...)

"""
The tail int_S^inf Im H(u) du of the density, H(u) = d/dt2 phi(t1, t2) at (u, -r u): the derivatives of H along the ray
are mixed derivatives of phi, read off jets of x -> phi(S + x, -r (S + x) + xi x) at N complex xi on a circle (the
coefficient of xi in the k-th derivative is k H^{(k-1)}(S), extracted by a discrete Fourier transform), as in RATIO_PDF
of Bao and Kan -> (T, the first omitted term)
"""
function tail_pdf(F, r::Float64, w::Float64, S::Float64, rho::Float64, ::Val{N}) where {N}
    K = N - 1
    a1 = zeros(ComplexF64, N)                                   # the coefficient of xi^1 of (d/dx)^k phi, k = 0..K
    x = jetvar(Val(N), 0.0); sj = jetvar(Val(N), S)
    for m in 0:K
        xi = rho * cis(2pi * m / N)
        ph = exp(logcf(F, sj, xi * x - r * sj))
        for k in 0:K; a1[k+1] += fact(k) * coef(ph, k) / xi; end
    end
    a1 ./= N
    H = [a1[k+1] / k for k in 1:K]                              # H^{(0..K-1)}(S)
    T = zero(ComplexF64); last = 0.0
    for k in 0:K-1
        acc = zero(ComplexF64)
        for l in 0:k; acc += binomial(k, l) * (im * w)^(k - l) * H[l+1]; end
        term = acc / (im * w)^(k + 1)
        k < K - 1 ? (T += term) : (last = abs(term))
    end
    imag(T), last
end

"""
    ratio_pdf(F, r; tol = 1e-10) -> (f, err, evals, fs)

The density of A/B at r (Geary): (1/pi) int_0^inf Im[d/dt2 phi(t1, t2) at t2 = -r t1] dt1, the derivative from a jet of
order one in t2, the tail beyond S = 400/w0 by `tail_pdf`.  The tolerance refers to the scale fs = max(1, E[B]/sd(A)) of
the density (the absolute tolerance is tol fs).
"""
function ratio_pdf(F, r::Real; tol::Float64 = 1e-10, K::Int = 8)
    r = Float64(r)
    _, Ev, sdv, Vu = ratio_scale(F)
    fs = max(1.0, Ev / sqrt(max(Vu, floatmin())))
    vv = Vu + r^2 * sdv^2; (isfinite(vv) && vv > 0) || (vv = Vu)
    sc = 1 / sqrt(vv)
    f = z -> begin
        u = z * sc
        L = logcf(F, Jet{2}(u), Jet{2}(SVector{2,ComplexF64}(-r * u, 1.0)))
        y = sc * imag(exp(coef(L, 0)) * coef(L, 1))
        isfinite(y) ? y : 0.0
    end
    w = F.w0; S = 400 / w; T = 0.0; et = 0.0
    for dbl in 0:5
        T, et = tail_pdf(F, r, w, S, 1 / Ev, Val(K + 1))
        (et <= tol / 10 || dbl == 5) && break
        S *= 2
    end
    I, e, nev = decades(f, S / sc, tol * fs, max(tol, 1e-14))
    (I + T) / pi, (e + et) / pi, nev, fs
end

"""the root of g on a bracket grown from [lo, hi] (g increasing), by the Illinois variant of regula falsi, to the width xtol"""
function illinois(g::G, lo::Float64, hi::Float64, xtol::Float64; grow::Float64 = hi - lo) where {G}
    glo = g(lo); ghi = g(hi); d = grow; k = 0
    while glo > 0 && k < 60; hi = lo; ghi = glo; lo -= d; d *= 2; glo = g(lo); k += 1; end
    while ghi < 0 && k < 120; lo = hi; glo = ghi; hi += d; d *= 2; ghi = g(hi); k += 1; end
    (isfinite(glo) && isfinite(ghi) && glo <= 0 && ghi >= 0) || throw(ErrorException("the quantile could not be bracketed (the distribution function is not finite or not monotone)"))
    side = 0; x = (lo + hi) / 2
    for _ in 1:200
        x = (lo * ghi - hi * glo) / (ghi - glo)
        (lo < x < hi) || (x = (lo + hi) / 2)
        gx = g(x)
        gx == 0 && break
        if gx < 0
            lo = x; glo = gx; side == -1 && (ghi /= 2); side = -1
        else
            hi = x; ghi = gx; side == 1 && (glo /= 2); side = 1
        end
        if hi - lo < xtol
            x = (lo * ghi - hi * glo) / (ghi - glo); break
        end
    end
    x
end

"""
    ratio_icdf(F, p; tol = 1e-10) -> r

The p-quantile of A/B: the bracket starts at E[A]/E[B] -/+ twice the scale sd(A)/E[B] + |E[A]|/E[B] of the ratio and the
root is located to 1e-11 times that scale.
"""
function ratio_icdf(F, p::Real; tol::Float64 = 1e-10)
    0 < p < 1 || throw(ArgumentError("ratio_icdf: p must lie in (0, 1)"))
    Eu, Ev, _, Vu = ratio_scale(F)
    sd0 = sqrt(Vu) / Ev + abs(Eu) / Ev
    c0 = Eu / Ev
    illinois(x -> ratio_cdf(F, x; tol)[1] - p, c0 - 2sd0, c0 + 2sd0, 1e-11 * sd0)
end

"""
    ratio_mom(F, k, x0 = 0; power = 1, tol = 1e-10) -> (m, err)

m[j] = E[(A - x0 B)^j / B^(j power)], j = 1..k (power = 1: the moments of A/B about x0, Sawa's representation; power = 1/2:
those of the t-ratio (A - x0 B)/sqrt(B)): with M(s1, s2) = E exp(s1 (A - x0 B) - s2 B),

    m_j = 1/Gamma(j power) int_0^inf s2^{j power - 1} [d^j M/ds1^j at s1 = 0] ds2,

the j-th derivative from one jet of order j in s1 at each node (no numerical differentiation), integrated in log(s2 E[B]).
"""
function ratio_mom(F, k::Int, x0::Real = 0.0; power::Float64 = 1.0, tol::Float64 = 1e-10)
    _, Ev, _, _ = ratio_scale(F)
    m = zeros(k); err = zeros(k)
    for j in 1:k
        m[j], err[j] = mom_one(F, Val(j + 1), Float64(x0), power, Ev, tol)
    end
    m, err
end
function mom_one(F, ::Val{N}, x0::Float64, power::Float64, Ev::Float64, tol::Float64) where {N}
    j = N - 1; a = j * power
    cst = fact(j) / gamma(a)
    t1 = jetvar(Val(N), 0.0, -im)
    g = y -> begin
        s2 = exp(y) / Ev
        L = logcf(F, t1, jetvar(Val(N), im * s2, im * x0))
        v = cst * real(exp(a * log(s2) + coef(L, 0)) * coef(exp(setval(L, 0.0)), j))
        isfinite(v) ? v : 0.0
    end
    I, e = quadgk(g, -60.0, -40.0, -30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 45.0; atol = tol * 1e-2, rtol = tol, maxevals = 10^6)
    I, e
end

# ---------------------------------------------------------------------------------------------------------------------
# The statistics of the classical cases by the single inversion of Bao and Kan, each value with its error estimate.  The
# callers are the package's user-level functions (src/merged.jl): coef_cdf, coef_pdf, coef_icdf, coef_mom,
# ltratio_lincdf, power_envelope (the cases DK, DU, DT with a zero or fixed start, SK, SU, ST) and ltratio_mom.
# ---------------------------------------------------------------------------------------------------------------------
"""the default absolute tolerance of each value of the single inversion (the quadrature and the first omitted term of
its tail)"""
const TOL = 1e-12
"""
the largest c at which the single inversion is the default (`method = :auto`): it was compared with the engines' values
from c = -550 to c = 20 (November 2026; within 1e-10 of their checked computations, see the README)
"""
const INV_CMAX = 20.0

"""
    coef_cdf1(F, y, c0, tol) -> (P, err)

P[(U - c0 V)/V <= y] = P[A/B <= y + c0 - c], the coefficient statistic n(phi^ - phi0), by the single inversion of Bao and
Kan (`ratio_cdf`), with its error estimate; 0, 1 and NaN at -Inf, Inf and NaN.
"""
function coef_cdf1(F::LimCF, y::Float64, c0::Float64, tol::Float64)
    isnan(y) && return NaN, 0.0
    isinf(y) && return (y > 0 ? 1.0 : 0.0), 0.0
    P, e, _ = ratio_cdf(F, y + c0 - F.c; tol)
    clamp(P, 0.0, 1.0), e
end
"""
    coef_pdf1(F, y, c0, tol) -> (f, err, ftol)

The density of the coefficient statistic at y by the single inversion of Bao and Kan (`ratio_pdf`), its error estimate
and the absolute tolerance that `tol` amounts to for it (tol times the scale E[V]/sd(U) of the density, at least tol).
"""
function coef_pdf1(F::LimCF, y::Float64, c0::Float64, tol::Float64)
    isfinite(y) || return (isnan(y) ? NaN : 0.0), 0.0, tol
    f, e, _, fs = ratio_pdf(F, y + c0 - F.c; tol)
    max(f, 0.0), e, tol * fs
end
"""
    coef_icdf1(F, p, c0, xtol, tol) -> (y, err of F at y)

The p-quantile of the coefficient statistic: the Illinois variant of regula falsi on the single inversion's distribution
function (tolerance tol), from the bracket E[A]/E[B] -/+ twice the scale sd(A)/E[B] + |E[A]|/E[B] of the ratio, to the
width xtol; with the error estimate of the distribution function at its last evaluation.
"""
function coef_icdf1(F::LimCF, p::Float64, c0::Float64, xtol::Float64, tol::Float64)
    Eu, Ev, _, Vu = ratio_scale(F)
    sd0 = sqrt(Vu) / Ev + abs(Eu) / Ev; m0 = Eu / Ev
    elast = Ref(0.0)
    g(x) = begin P, e, _ = ratio_cdf(F, x; tol); elast[] = e; P - p end
    r = illinois(g, m0 - 2sd0, m0 + 2sd0, xtol)
    r - (c0 - F.c), elast[]
end
"""the moments E[{(U - c0 V)/V}^j], j = 1..k, of the coefficient statistic, from jets (`ratio_mom`) -> (m, err)"""
coef_mom1(F::LimCF, k::Int, c0::Float64, tol::Float64) = ratio_mom(F, k, c0 - F.c; tol)
"""
    tratio_mom1(F, k, c0, tol) -> (m, err)

The moments E[t^j], j = 1..k, of the t-ratio (U - c0 V)/sqrt(V) at the local null value c0: one-dimensional integrals
whose integrand takes the j-th derivative of the moment generating function along its affine path from one jet per node
(`ratio_mom` with power 1/2), after Bao and Kan.
"""
tratio_mom1(F::LimCF, k::Int, c0::Float64, tol::Float64) = ratio_mom(F, k, c0 - F.c; power = 0.5, tol)
"""
    lin_cdf1(F, b, a, tol) -> (P, err)

P(U - aV <= b) = P[A + (c - a) B <= b] by the single inversion of Bao and Kan (`half_cdf`) with its error estimate; 0, 1
and NaN at b = -Inf, Inf, NaN.
"""
function lin_cdf1(F::LimCF, b::Float64, a::Float64, tol::Float64)
    isnan(b) && return NaN, 0.0
    isinf(b) && return (b > 0 ? 1.0 : 0.0), 0.0
    P, e, _ = half_cdf(F, 1.0, F.c - a, b; tol)
    clamp(P, 0.0, 1.0), e
end
"""
    envelope1(c, alpha, tol) -> (power, err)

The asymptotic power envelope of Elliott, Rothenberg and Stock at c < 0 without deterministic terms (and with an
intercept, the same curve): the power at c of the test that rejects for small U - (c/2) V, its critical value the
alpha-quantile of U - (c/2) V at the unit root, both by the single inversion of Bao and Kan; the critical value by the
Illinois variant of regula falsi to 1e-12 from the bracket [-0.45, 0.5] (grown in steps of one).  err is the sum of the
error estimates of the two inversions at the end.  alpha at c >= 0, 1 at c = -Inf, NaN at NaN.
"""
function envelope1(c::Float64, alpha::Float64, tol::Float64)
    isnan(c) && return NaN, 0.0
    c < 0 || return alpha, 0.0
    c == -Inf && return 1.0, 0.0
    a = c / 2
    F0 = LimCF("DK", 0.0); Fc = LimCF("DK", c)
    e0 = Ref(0.0)
    b = illinois(x -> begin P, e, _ = half_cdf(F0, 1.0, -a, x; tol); e0[] = e; P - alpha end, -0.45, 0.5, 1e-12; grow = 1.0)
    P, e, _ = half_cdf(Fc, 1.0, c - a, b; tol)
    clamp(P, 0.0, 1.0), e + e0[]
end
"""
    lin_ok(cas, c, g0, a, b) -> Bool

The region in which the single inversion is the default for the linear form P(U - aV <= b) (`lin_cdf1`): a != 0, and
-100 <= c <= 4 in case DK with a zero start, -100 <= c <= 2 in the other cases (the region compared with the engines in
November 2026: within 5.5e-11 of the classical code in case DK with a zero start, within 1e-12 of the general engine in the
others, and right where the general engine's one evaluation was off, by up to 6e-2 in case DK and 2.9e-3 in case DU with
a fixed start in the upper tail above the unit root, as its checked computation and the plain quadrature without the tail
confirm).  At a = 0 (the law of U alone) the transform decays like a power, and b near the edge -(1 + g0^2)/2 of the support
makes the start S = 400/|w| of the tail run away (1 s and an error estimate of 3e-8 at the 0.1% point); the engines have
the closed form there.  At c = 7 in case DK the classical code (the region's conditional integral) differs from the
single inversion by 2e-8 at the 99.9% point with a = 2, which no third computation settled.
"""
lin_ok(cas::AbstractString, c::Float64, g0::Float64, a::Float64, b::Float64) =
    a != 0 && -100 <= c <= ((cas == "DK" && g0 == 0) ? 4.0 : 2.0)
"""
whether the moments of the t-ratio from jets (`tratio_mom1`) are the default (`ltratio_mom` with method = :auto): c up to
INV_CMAX, any c0 and start (compared with the engines in November 2026 over the six cases, c from -480 to 20, c0 in {0,
c, -2, 2}, g0 in {0, 1}: within 1e-11 relative near the unit root; with c0 = c far from it, 1e-8 to 1.3e-7, where the
engines' circles are not converged and move towards these values under refinement; with a fixed start above the unit
root (c = 1, 2, c0 = -2) and at c = 10, 20 with c0 = c the engines' one evaluation was off by up to 0.65 relative, and
their refined rules reach these values to 3e-13)
"""
mom_ok(cas::String, c::Float64, c0::Float64, g0::Float64) = c <= INV_CMAX
"""whether the single inversion is the default for the power envelope at c (`envelope1`): c <= -1e-4 (within 2.7e-12 of the
general engine's envelope from c = -1e-4 to -500, November 2026; closer to zero the weight a = c/2 puts the critical value
at the edge of the support of U - aV), and c >= 0 or NaN (the level, NaN)"""
envelope_ok(c::Float64) = !(c < 0) || c <= -1e-4

# ---------------------------------------------------------------------------------------------------------------------
# The distribution function of the t-ratio at zero in closed form (AR1LIM_RAY of Bao and Kan)
# ---------------------------------------------------------------------------------------------------------------------
Phi(x) = erfc(-x / sqrt(2.0)) / 2
npdf(x) = exp(-x * x / 2) / sqrt(2pi)
cexp(x) = x == 0 ? 1.0 : expm1(x) / x
"""no deterministic terms, fixed start: U = [c_0 chi^2_1(delta) - (1 + g0^2)]/2"""
function ray_k(c::Float64, g0::Float64)
    c0 = cexp(2c); dl = g0^2 * exp(2c) / c0; a = sqrt((1 + g0^2) / c0)
    Phi(a - sqrt(dl)) - Phi(-a - sqrt(dl))
end
const RAY_C5 = [1 / 12, 1 / 12, 17 / 360, 7 / 360, 43 / 6720, 107 / 60480, 769 / 1814400, 163 / 1814400, 4097 / 239500800, 709 / 239500800,
                6827 / 14529715200, 15019 / 217945728000, 19661 / 2092278988800, 1139 / 951035904000]
"""an intercept: U = lambda_1 X_1 - lambda_2 X_2 - 1/2 with independent noncentral chi-square(1) variables; one integral of
the noncentral chi-square distribution function against the normal density"""
function ray_u(c::Float64, kap::Float64)
    c0 = cexp(2c); c1 = cexp(c)
    if abs(c) < 0.25
        q = evalpoly(c, RAY_C5); c2 = c^2 * q
    else
        c2 = c0 - c1^2; q = c2 / c^2
    end
    vs = sqrt(c2^2 + 4q)
    l1 = (vs + c2) / 4; l2 = q / (vs + c2)
    d1 = c == 0 ? kap^2 / 2 : kap^2 / 2 * (1 - (c - 4) * c2 / (c * vs)); d2 = kap^2 - d1
    sd1 = sqrt(max(d1, 0.0)); sd2 = sqrt(max(d2, 0.0))
    F1(y) = (r = sqrt(y); Phi(r - sd1) - Phi(-r - sd1))
    wp = Float64[-12.0, 12.0]
    sd2 < 12 && push!(wp, -sd2)
    ys = (l1 * d1 - 0.5) / l2
    if ys > 0
        for sg in (-1, 1); zz = sg * sqrt(ys) - sd2; abs(zz) < 12 && push!(wp, zz); end
    end
    quadgk(z -> F1((0.5 + l2 * (sd2 + z)^2) / l1) * npdf(z), sort(unique(wp))...; atol = 1e-14, rtol = 1e-12)[1]
end
"""
    ray_cdf(cas, c; g0 = 0) -> P[U <= 0]

The distribution function at zero of the t-ratio (c0 = 0), and of the coefficient statistic n(phi^ - 1), in closed form:
U is a quadratic form in two or three normal variables minus a constant (the chi-square representation of Bao and Kan,
their AR1LIM_RAY).  Cases DK, DU, SK, SU (the trend cases need a two-dimensional normal integral, not ported).
"""
function ray_cdf(cas::AbstractString, c::Real; g0::Real = 0.0)
    cas = uppercase(cas); c = Float64(c); g0 = Float64(g0)
    cas == "DK" && return ray_k(c, g0)
    cas == "DU" && return ray_u(c, c * g0)
    c < 0 || throw(ArgumentError("ray_cdf: a stationary start requires c < 0"))
    if cas == "SK"
        lc = sqrt(-expm1(2c)) / (-4c); a = 1 / (2lc)
        return quadgk(z -> erf(sqrt(max(a + z * z, 0.0) / 2)) * npdf(z), -12.0, 0.0, 12.0; atol = 1e-14, rtol = 1e-12)[1]
    elseif cas == "SU"
        sg = sqrt(-c / 2)
        return quadgk(z -> ray_u(c, sg * z) * npdf(z), -12.0, 0.0, 12.0; atol = 1e-12, rtol = 1e-10)[1]
    end
    throw(ArgumentError("ray_cdf: cases DK, DU, SK, SU"))
end
