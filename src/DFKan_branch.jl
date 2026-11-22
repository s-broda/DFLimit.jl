# =====================================================================================================================
# The branch of P(t), M_0(t) and the labels of the roots of the line integral at a single point t
# =====================================================================================================================
# The classical engine fixes the branch of P(t) = (-a_2(t))^{-1/2}, of M_0(t) = (-s_1 s_2)^{-1/2} and the labels of the
# roots s_1(t), s_2(t) of D(., t) by continuity along a grid from the real point (DFExact.tparts: unwrap!).  Here they are
# determined at one point, after Bao and Kan, who write each argument under a root as a product of factors whose
# arguments are known.  For Im t > 0 (the conjugate for Im t < 0), with mu = (c^2 - 2t)^{1/2} the principal root:
#
#  (a) The roots.  The reciprocal roots are the eigenvalues nu of K Sigma_t, K the real symmetric matrix of the numerator
#      and Sigma_t the covariance of its Gaussian functionals under the tilt e^{tV}; an eigenvector y gives nu = y*Ky /
#      y*Sigma_t^{-1}y, a real number over a number that has a positive real part for Re t < t_* (Sigma_t^{-1} has a
#      positive definite Hermitian part, the lemma on accretive matrices of Bao and Kan) and a negative imaginary part for
#      Im t > 0.  Hence s_1 (negative type) lies in the open left half-plane on the line and in the open upper half-plane
#      for every Im t > 0, s_2 (positive type) in the right and the lower one: the labels are read off the signs, and the
#      roots never meet off the real axis.
#  (b) M_0 = (-s_1)^{-1/2} s_2^{-1/2} with principal roots on the line, where -s_1 and s_2 lie in the right half-plane.
#  (c) D(0, t) = M_V(t)^{-2} = a_0 = (-a_2)(-s_1) s_2 is a canonical product over the positive zeros t_j of the moment
#      generating function of V, with the continuous argument sum_j Arg(1 - t/t_j).  For no deterministic terms it is
#      e^c cosh(mu) G, G = 1 - c tanh(mu)/mu, with the continuous argument Im mu + Arg(1 + e^{-2 mu}) of cosh(mu) (Bao and
#      Kan) and G in the upper (c < 0) or lower (c > 0) half-plane: tanh(mu)/mu is a sum of 2/(mu^2 + a_k), a_k > 0.  A
#      regressor projected out, or a start integrated out, changes V by a quadratic form of rank one of one sign: the
#      zeros interlace, the ratio of the two determinants maps the upper half-plane into a half-plane, and its principal
#      argument, in (0, pi) for a projection and in (-pi, 0) for the stationary start, is the continuous one.  Two
#      projections (intercept and trend) give an argument in (0, 2 pi), which the ratio determines; GLS detrending, one
#      update of each sign, an argument in (-pi, pi), the principal one.
#  Then arg(-a_2) = arg D(0, t) - Arg(-s_1) - Arg(s_2), all at the point.

import ..DFExact
const X = DFExact

"""the series of D(., t)/e^c at s = 0, -1, 1 of the cases with an intercept or a trend, computed once per (model, c)"""
struct TPoint
    model::Int
    c::Float64
    stat::Bool
    coef::NTuple{3,Vector{Float64}}
end
function TPoint(model::Int, c::Float64, stat::Bool)
    model in (2, 3) || return TPoint(model, c, stat, (Float64[], Float64[], Float64[]))
    SER = model == 2 ? X.D1_SERIES : X.D2_SERIES
    TPoint(model, c, stat, ntuple(k -> Float64[f((0.0, 1.0, -1.0)[k], c, Float64) for f in SER], 3))
end

"""an argument known to lie in (0, pi) (one projection) or (0, 2 pi) (two) from its principal value"""
@inline argpos(a::Float64, two::Bool) = two ? (a < -1e-6 ? a + 2pi : a) : (a < -pi / 2 ? a + 2pi : a)
"""an argument known to lie in (-pi, 0) from its principal value"""
@inline argneg(a::Float64) = a > pi / 2 ? a - 2pi : a

"""
    argK(c, t) -> (arg, mu, e2, G)

The continuous argument (zero at real t below t_*) of cosh(mu) - c sinh(mu)/mu = cosh(mu) G at Im t > 0: Im mu + Arg(1 +
e^{-2 mu}) + Arg G, with mu = (c^2 - 2t)^{1/2}, e2 = e^{-2 mu} and G = 1 - c tanh(mu)/mu formed without cancellation.
"""
@inline function argK(c::Float64, t::ComplexF64)
    z = c * c - 2t; mu = sqrt(z); e2 = exp(-2mu)
    if abs(mu) < 1e-3
        G = 1 - c * (1 - z / 3 + 2 * z * z / 15)
    else
        del = c > 0 ? -2t / (mu + c) : mu - c
        G = (del + (mu + c) * e2) / (mu * (1 + e2))
    end
    imag(mu) + angle(1 + e2) + angle(G), mu, e2, G
end

"""
    tparts_point(tp, t) -> (P, s1, s2, M0)

P(t), s_1(t), s_2(t) and M_0(t) of the classical line integral (DFExact.tparts) at the single point t, Re t < t_*, with
no reference to neighbouring points.
"""
function tparts_point(tp::TPoint, t::ComplexF64)
    if imag(t) < 0
        P, s1, s2, M0 = tparts_point(tp, conj(t))
        return conj(P), conj(s1), conj(s2), conj(M0)
    end
    model = tp.model; c = tp.c; stat = tp.stat
    re = imag(t) == 0                                         # the real point: every argument is zero
    zc = zero(ComplexF64)
    if model == 1 && !stat
        z = c * c - 2t
        C, _, S = X.pieces(z)
        s1 = c > 0 ? X.eta1(c, t) : (C - c * S) / S
        ph = 0.0
        if !re
            mu = sqrt(z)
            ph = imag(mu) + angle(1 - exp(-2mu)) - angle(mu)  # the continuous argument of sinh(mu)/mu
        end
        return exp(-c / 2) * abs(S)^(-0.5) * cis(-ph / 2), s1, zc, zc
    end
    if model == 4 && !stat
        lam = X.lam_gls(X.CBAR_GLS); v = ComplexF64.(X.VGLS)
        G = X.girsanov(4, c, t)
        s1 = 1 / ((1 - lam)^2 * (transpose(v) * (G.Sg * (G.N \ v))))
        ph = 0.0
        if !re
            GK = X.girs_g(c, t).N11                           # 1 - c tanh(th)/th
            ph = angle(GK) + angle(G.D / GK)                  # one update of each sign: the principal argument
        end
        return exp(-c / 2 - (G.lch + log(abs(G.D)) + im * ph) / 2 + (log(abs(s1)) + im * angle(s1)) / 2), s1, zc, zc
    end
    if !stat && c > X.CGIRS
        G = X.girsanov(model, c, t)
        disc = sqrt(G.A1 * G.A1 - 4 * G.A0 * G.A2)
        sw = sign(real(conj(G.A1) * disc)); sw == 0 && (sw = one(sw))
        rb = (G.A1 + sw * disc) / (2 * G.A0); rs = G.A2 / (G.A0 * rb)
        s1, s2 = label(rb, rs)
        ph = re ? 0.0 : angle(G.N[1, 1]) + argpos(angle(G.D / G.N[1, 1]), model == 3)
        lm = complex(log(abs(s1 * s2)), angle(-s1) + angle(s2))
        return exp(-c / 2 - (G.lch + log(abs(G.D)) + im * ph) / 2 + lm / 2), s1, s2, exp(-lm / 2)
    end
    # the quadratic D(s, t) = a2 lam^2 + a1 lam + a0, lam = -s, from its values at s = 0, -1, 1
    z = c * c - 2t
    s0, sp, sm = complex(0.0), complex(-1.0), complex(1.0)
    d0z = zc
    if model == 1
        d0 = X.Dfun(1, c, s0, t; stat = true); dp = X.Dfun(1, c, sp, t; stat = true); dm = X.Dfun(1, c, sm, t; stat = true)
    elseif abs(z) < X.Z0
        e = exp(c)
        d0 = e * evalpoly(z, tp.coef[1]); dp = e * evalpoly(z, tp.coef[2]); dm = e * evalpoly(z, tp.coef[3])
        d0z = d0
        if stat
            d0 += X.Dstat_add(model, c, s0, t); dp += X.Dstat_add(model, c, sp, t); dm += X.Dstat_add(model, c, sm, t)
        end
    else
        d0 = X.Dfun(model, c, s0, t; stat); dp = X.Dfun(model, c, sp, t; stat); dm = X.Dfun(model, c, sm, t; stat)
        d0z = stat ? d0 - X.Dstat_add(model, c, s0, t) : d0
    end
    a2 = (dp + dm) / 2 - d0; a1 = (dp - dm) / 2; a0 = d0
    disc = sqrt(a1 * a1 - 4 * a2 * a0)
    sw = sign(real(conj(a1) * disc)); sw == 0 && (sw = one(sw))
    rb = (-a1 - sw * disc) / (2a2); rs = a0 / (a2 * rb)
    s1, s2 = label(-rb, -rs)
    M0 = 1 / (sqrt(-s1) * sqrt(s2))
    ph0 = 0.0
    if !re
        aK, mu, e2, G = argK(c, t)
        DK = exp(c + mu) * (1 + e2) / 2 * G                   # e^c (cosh mu - c sinh(mu)/mu)
        ph0 = aK
        model == 1 || (ph0 += argpos(angle(d0z / DK), model == 3))
        stat && (ph0 += argneg(angle(d0 / (model == 1 ? DK : d0z))))
    end
    P = abs(a2)^(-0.5) * cis(-(ph0 - angle(-s1) - angle(s2)) / 2)
    P, s1, s2, M0
end
"""the two roots labelled by their signs: s_1 in the left half-plane and s_2 in the right one; where both real parts have the
same sign (beyond t_*), s_1 the root in the upper half-plane"""
@inline function label(ra::ComplexF64, rb::ComplexF64)
    (real(ra) < 0) != (real(rb) < 0) && return real(ra) < real(rb) ? (ra, rb) : (rb, ra)
    imag(ra) > imag(rb) ? (ra, rb) : (rb, ra)
end

"""
    tparts_pointwise(model, c, t; stat = false) -> (P, s1, s2, M0)

DFExact.tparts on the points t in any order (no real point needed), each point by `tparts_point`.
"""
function tparts_pointwise(model::Int, c::Float64, t::Vector{ComplexF64}; stat::Bool = false)
    tp = TPoint(model, c, stat)
    n = length(t)
    r = Vector{NTuple{4,ComplexF64}}(undef, n)
    X.tmap!(j -> tparts_point(tp, t[j]), r, n)
    [x[1] for x in r], [x[2] for x in r], [x[3] for x in r], [x[4] for x in r]
end
"""switch the classical engine's line integral to the branch at each point (true) or back to the tracking (false)"""
pointwise!(on::Bool = true) = (X.POINTWISE[] = on ? tparts_pointwise : nothing; on)
