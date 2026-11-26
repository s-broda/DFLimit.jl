# =====================================================================================================================
# The line integral with the explicit kernel for the designs of the general engine whose numerator has rank one or two
# =====================================================================================================================
# Where the determinant D(., t) has degree one (a positive root) or two (one root of each sign), the inversion in s is
# explicit: a scaled chi-square density, or the density of a difference of two scaled chi-square variables (a Bessel
# function K_0), and the density and the distribution function are one integral over t along the line Re t = tau of real
# integrals over v of that kernel (the line integral representation of Bao, Broda and Kan).  The classical engine (DFExact)
# has that integrator, with P(t) and the roots from its closed forms case by case (DFExact.tparts).  Here a `LineModel`
# stands for the case code: its methods of DFExact.tparts, kernel, default_tau and k0first supply P(t), s_1(t), s_2(t)
# and M_0(t) = (-s_1 s_2)^{-1/2} from the general engine's transform (`mgfparts`: log D_V(t) and the reciprocal roots
# mu_k(t) of the eigenvalue problem), and the classical integrator does the rest unchanged: its outer rules, the inner
# paths, the fixed layouts of its kernels, the general kernel for c0 != 0 and its bounds on the paths (PathTooLarge).  (The
# classical functions outer, outer_gen, value_gen, density, cdf and saddle_loc take the model untyped for this.)
#
# Coverage (what the papers state).  The line integral holds with the general determinant for polynomial trends of any
# order and for the statistic that detrends first with any continuous trend that does not vanish at the origin, the growth
# lemma of the paper on general deterministic terms supplying the bounds its proof uses; without deterministic terms it is
# the classical representation.  A design whose span is closed under differentiation has the same numerator for the
# one-step statistic as for the one that detrends first, so a continuous such design that does not vanish at the origin
# (polynomial trends, an intercept with sines and cosines) is covered.  With a stationary start the root configuration, the
# growth lemma and the inversion on paths hold with the determinant of the stationary start, which keeps the rank two for
# these designs (an intercept in the span) and has rank two without deterministic terms (the classical case SK).  Not
# covered: GLS detrending, a fixed start (its kernel is the noncentral density, a further integral with an intercept or a
# trend), designs with a jump or a numerator of another rank, the statistic that detrends first with d(0) = 0 (rank one,
# for which the papers do not state the line integral), c below the classical engine's CMIN, and a stationary start for
# c > DFExact.CSTAT.
#
# Guards.  Shared with the classical engine, because the integrator is: its bounds on the paths of the inner integrals
# (PathTooLarge, raised before anything is evaluated: method = :auto then takes the inversion on paths, :line refuses), the
# breakpoints and partitions at the edges of the support of X(v) and the infinite density at the tangency of a numerator of
# rank one with c0 > 0, and its finer rules below c = -20.  Of its own: the roots labelled by the sign of their real part
# (one root in each group, which stay apart on the line), the argument of log P at each point from the pivots of the
# precision matrix (`line_point`, after Bao and Kan), the end of the line by |P| (`line_end`), the caps on the
# first panels of the K_0 layouts by the root scale (k0first) and the line moved off tau = 1 where the fixed layouts'
# bound on s_2 fails (`line_tau`).  Above the unit root the classical engine's Girsanov forms (case codes only) are not
# needed: the general determinant is formed without the cancellation (`m011`, Sigma in closed form), within 1e-14 of the
# Girsanov forms in case DU at c = 5.

import ..DFExact
const LX = DFExact

"""
    LineModel

The provider of P(t), s_1(t), s_2(t) for the classical engine's line integral: the design with its start (Mo), c, the rank
r of the numerator (1 or 2), whether the start is stationary, t_*, the line Re t = tau (`line_tau`), its end wend
(`line_end`), and a memo of the
general engine's transform at the points of the line, t -> (log D_V(t), mu_+(t), mu_-(t)), mu_+ the reciprocal of the
positive root and mu_- of the negative one (0 for r = 1).  The memo holds at most LINE_MEMO_MAX points (emptied when full;
the values do not depend on it).
"""
struct LineModel
    Mo::Model
    c::Float64
    r::Int
    stat::Bool
    tstar::Float64
    tau::Float64
    wend::Float64
    memo::Dict{ComplexF64,NTuple{3,ComplexF64}}
    lock::ReentrantLock
end
const LINE_MEMO_MAX = 100_000

"""(log D_V(t), mu_+(t), mu_-(t)) at one point of the line (`mgfparts`): the groups by the sign of Re mu, which on the line
Re t = tau is that of Re s (the roots of each group stay on their side of the gap at the real point)"""
function line_point(lm::LineModel, t::ComplexF64, F::DetCF)
    θ, ld, μ, _ = mgfparts(lm.Mo, lm.c, t, F.W)
    ld = ComplexF64(ld)
    # the argument of log D_V at the point from the pivots of the precision matrix (`detlog`, Sigma as mgfparts left it: one
    # small elimination), after Bao and Kan; with the principal arguments of mu_+ and -mu_-, each in the right half-plane
    # on the line, log P(t)^{-2} has its continuous imaginary part at the point
    ld = complex(real(ld), imag(θ) + imag(detlog(F, zero(ComplexF64), t, θ)))
    lm.r == 1 && return (ld, ComplexF64(μ[1]), zero(ComplexF64))
    m1 = ComplexF64(μ[1]); m2 = ComplexF64(μ[2])
    real(m1) >= real(m2) ? (ld, m1, m2) : (ld, m2, m1)
end

"""the transform at the points ts: from the memo, the missing points on DFExact.NWORKERS[] tasks"""
function line_raw(lm::LineModel, ts::Vector{ComplexF64})
    n = length(ts)
    raw = Vector{NTuple{3,ComplexF64}}(undef, n)
    miss = Int[]
    lock(lm.lock) do
        for j in 1:n
            v = get(lm.memo, ts[j], nothing)
            v === nothing ? push!(miss, j) : (raw[j] = v)
        end
    end
    isempty(miss) && return raw
    nm = length(miss); nw = min(LX.NWORKERS[], nm)
    blas1f(dense_la(lm.Mo)) do
        if nw <= 1
            F = DetCF(lm.Mo, lm.c)
            for j in miss; raw[j] = line_point(lm, ts[j], F); end
        else
            @sync for w in 1:nw
                Threads.@spawn begin
                    Ft = DetCF(lm.Mo, lm.c)
                    for i in w:nw:nm; j = miss[i]; raw[j] = line_point(lm, ts[j], Ft); end
                end
            end
        end
    end
    lock(lm.lock) do
        length(lm.memo) + nm > LINE_MEMO_MAX && empty!(lm.memo)
        for j in miss; lm.memo[ts[j]] = raw[j]; end
    end
    raw
end

"""log of P(t)^{-2} = D_V(t) mu_+ (-mu_-) from the transform, before the continuation of its imaginary part"""
@inline lineLr(r::Int, v::NTuple{3,ComplexF64}) = r == 1 ? v[1] + log(v[2]) : v[1] + log(v[2]) + log(-v[3])

"""P(t), s_1(t), s_2(t) and M_0(t) on the points t of the line (t[1] real) for the classical integrator: s_1 the negative
root and s_2 the positive one (rank two), s_1 the positive root (rank one, s_2 and M_0 unused); M_0 = (-s_1)^{-1/2} s_2^{-1/2}
with the principal branches (both arguments stay in the right half-plane on the line); P from log P^{-2} with its argument
fixed at each point (`line_point`; until November 2026 continued from the real point along the line, within 1.7e-14 of
these values on the engine's grids, at the same cost)"""
function LX.tparts(lm::LineModel, c::Float64, t::Vector{ComplexF64}; stat::Bool = false)
    if !isempty(t) && imag(t[1]) != 0                              # (every grid of the classical integrator starts at the real point)
        P, s1, s2, M0 = LX.tparts(lm, c, [complex(real(t[1])); t])
        return P[2:end], s1[2:end], s2[2:end], M0[2:end]
    end
    n = length(t)
    raw = line_raw(lm, t)
    P = Vector{ComplexF64}(undef, n); s1 = similar(P); s2 = similar(P); M0 = similar(P)
    L = zero(ComplexF64)
    for j in 1:n
        v = raw[j]
        Lr = lineLr(lm.r, v)
        if j == 1
            m = round(imag(Lr) / 2pi)
            abs(imag(Lr) - 2pi * m) < 1e-7 || @warn "the line integral: D_V mu_+ (-mu_-) not positive at the real point" Lr
            Ln = complex(real(Lr), 0.0)
        else
            Ln = Lr                                                # (the argument fixed at the point, `line_point`)
        end
        L = Ln
        P[j] = exp(-L / 2)
        if lm.r == 1
            s1[j] = 1 / v[2]; s2[j] = zero(ComplexF64); M0[j] = zero(ComplexF64)
        else
            s1[j] = 1 / v[3]; s2[j] = 1 / v[2]; M0[j] = sqrt(v[2]) * sqrt(-v[3])
        end
    end
    P, s1, s2, M0
end

LX.kernel(lm::LineModel, stat::Bool) = lm.r
LX.default_tau(lm::LineModel, c::Float64, g0::Float64; stat::Bool = false) = lm.tau
"""
    line_tau(lm) -> tau

The line Re t = tau of the classical engine's rule: 1 for c <= 0 with a zero start (t_* > 1 there for every design: V is
at most int J^2), otherwise min(1, t_*/2).  The fixed layouts of the K_0 kernel's distribution function, which the
classical integrator takes at tau = 1 exactly, end the ray of the constant part over X > 1/2 where e^{-s_2 X} has fallen
for s_2(tau) >= 2.5 (DFExact.layouts_k0, as in the classical cases); where the positive root lies below LINE_S2MIN at tau = 1
the line moves to prevfloat(1.0), so that the adaptive rules, which follow the decay of the kernel, take the distribution
function.
"""
function line_tau(lm::LineModel)
    tau = (lm.c <= 0 && !lm.stat) ? 1.0 : min(1.0, lm.tstar / 2)
    if tau == 1.0 && lm.r == 2
        v = line_raw(lm, [complex(tau)])[1]
        real(1 / v[2]) < LINE_S2MIN && (tau = prevfloat(1.0))
    end
    tau
end
const LINE_S2MIN = 2.5
"""the caps on the first panels of the K_0 layouts (DFExact.k0first): 1/b and 1e-2/|s_1| at tau, b = (s_2 - s_1)/2, for
c < -20 as in the classical engine, and wherever |s_1(tau)| exceeds K0FIRST_S1 (the root scale of case DU at c = -20,
up to which the classical layouts are checked without the caps; higher-order trends reach it nearer the unit root)"""
function LX.k0first(lm::LineModel, c::Float64, stat::Bool, tau::Float64)
    _, s1, s2, _ = LX.tparts(lm, c, [complex(tau)])
    (c >= -20 && abs(real(s1[1])) <= K0FIRST_S1) && return (Inf, Inf)
    (1 / abs(real(s2[1] - s1[1]) / 2), 1e-2 / abs(real(s1[1])))
end
const K0FIRST_S1 = 450.0
LX.wmaxc(lm::LineModel, c::Float64) = lm.wend
"""
    line_end(lm, tau) -> W

The end of the outer integrals on the line Re t = tau.  The classical engine ends at w = DFExact.wmaxc(c), set by the decay
of the kernel's exponential factor alone; the transform also carries a power of t that grows with the number of trend
functions, and at c = -20 |P(tau + iW)| is 9.5e-9 in case DU, 2.2e-8 in case DT, 7e-7 for a trend of order four (whose
density was off by 6e-10 there) and 7e-6 for one of order six.  The line is therefore extended, in steps of 25% from
wmaxc(c), until |P(tau + iW)| <= LINE_PEND, the largest value the classical engine leaves at its end in its six cases
(case DT at c = -20), so that the three classical designs keep the classical end; at most LINE_WMAXFAC times wmaxc(c).
"""
function line_end(lm::LineModel, tau::Float64)
    W0 = LX.wmaxc(lm.c); W = W0
    absP(w) = (v = line_raw(lm, [complex(tau, w)])[1]; exp(-real(lineLr(lm.r, v)) / 2))
    while absP(W) > LINE_PEND && W < LINE_WMAXFAC * W0
        W *= 1.25
    end
    W
end
const LINE_PEND = 2.5e-8
const LINE_WMAXFAC = 64.0

# ---------- which designs, starts and options the line integral covers ----------
const LINE_RANK_MSG = "the line integral needs a numerator of rank one (a positive root) or two (one root of each sign)"
"""nothing where the line integral covers the design with its start and options, otherwise the reason (see the header)"""
function line_refusal(D::Determ, c::Float64, start::Symbol, gamma0::Float64, gls)
    gls === nothing || return "GLS detrending is not covered: the papers state the line integral for least squares detrending"
    (start == :fixed && gamma0 != 0) && return "a fixed start is not covered: its kernel is the noncentral density of the numerator, a further integral with an intercept or a trend (the classical engine takes its means from closed forms case by case)"
    c >= LX.CMIN || return "c below $(LX.CMIN), the range in which the classical line integral is checked"
    (start == :stationary && !(c <= LX.CSTAT)) && return "a stationary start needs c <= $(LX.CSTAT) for the classical line integral's quadrature"
    nreg(D) == 0 && return nothing                               # no deterministic terms: the classical representation
    P = prep_cached(D)
    jumps = line_jumps(D)
    d0 = dval(D, 0.0); vanish = maximum(abs, d0) <= 1e-12 * max(1.0, maximum(abs, dval(D, 0.5)))
    if D.detrend == :first
        (jumps || vanish) && return "the statistic that detrends first is covered for continuous trends that do not vanish at the origin only (the design has " *
            (jumps ? "a jump" : "d(0) = 0, a numerator of rank one for which the papers do not state the line integral") * ")"
        return nothing
    end
    (closed_under_derivative(D) && !jumps && !vanish) && return nothing
    "the one-step statistic is covered for continuous designs that do not vanish at the origin and whose span is closed under differentiation (polynomial and trigonometric trends with an intercept) only: " *
        (jumps ? "the design has a jump" : vanish ? "d(0) = 0" : "the span of the design is not closed under differentiation") *
        " (the numerator has rank $(P.r), $(P.rp) positive and $(P.rm) negative roots)"
end

"""whether d jumps at a break of D by more than 1e-9 times its size (the Chebyshev re-expansions of `fdesign` meet at
their computational breaks to about 1e-12, which `prep` already counts as jumps at high degree)"""
function line_jumps(D::Determ)
    sz = max(1.0, maximum(abs, dval(D, 0.5)), maximum(abs, dval(D, 0.0)), maximum(abs, dval(D, 1.0)))
    for j in 1:npc(D)-1
        lam = D.br[j+1]
        for i in 1:nreg(D)
            dl = evalpoly(lam - poff(D, j + 1), view(D.C[j+1], :, i)) - evalpoly(lam - poff(D, j), view(D.C[j], :, i))
            abs(dl) > 1e-9 * sz && return true
        end
    end
    false
end

const LMCACHE = Dict{Tuple{UInt64,Symbol,Float64},LineModel}()
const LMORDER = Tuple{UInt64,Symbol,Float64}[]
"""
    line_model(D, c, start, gamma0, gls) -> (LineModel or nothing, reason or nothing)

The provider of the design with its start at c, cached per (design, start, c) (at most CACHE_MAX[] of them, the oldest
dropped), or nothing and the reason where the line integral does not cover the arguments (`line_refusal`, and the rank of
the numerator with the start).
"""
function line_model(D::Determ, c::Real, start::Symbol, gamma0::Real, gls)
    c = Float64(c)
    why = line_refusal(D, c, start, Float64(gamma0), gls)
    why === nothing || return nothing, why
    key = (dkey(D), start == :fixed ? :zero : start, c)
    hit = lock(CPLOCK) do; get(LMCACHE, key, nothing); end
    hit === nothing || return hit, nothing
    Mo = startmodel(D, c, start, gamma0, gls)
    ((Mo.r == 1 && Mo.rp == 1) || (Mo.r == 2 && Mo.rp == 1 && Mo.rm == 1)) ||
        return nothing, "$LINE_RANK_MSG; with this start it has rank $(Mo.r) ($(Mo.rp) positive, $(Mo.rm) negative)"
    ts = tstar(Mo, c)[1]; stat = Mo.start == :stationary
    lm0 = LineModel(Mo, c, Mo.r, stat, ts, NaN, NaN, Dict{ComplexF64,NTuple{3,ComplexF64}}(), ReentrantLock())
    tau = line_tau(lm0)
    lm = LineModel(Mo, c, Mo.r, stat, ts, tau, line_end(lm0, tau), lm0.memo, lm0.lock)
    lock(CPLOCK) do; cache_put!(LMCACHE, LMORDER, key, lm); end
    lm, nothing
end

"""the value of the line integral at x: the distribution function (which = 1) or the density (which = 2) by the classical
integrator with the provider lm"""
function line_value(lm::LineModel, x::Float64, which::Int, c0::Float64)
    st = lm.stat ? :stationary : :zero
    which == 1 ? LX.cdf(x, lm; c = lm.c, c0, start = st) : LX.density(x, lm; c = lm.c, c0, start = st)
end

# ---------- the option `method` of dist_cdf, dist_pdf and dist_quantile ----------
const LINE_METHOD_MSG = "method must be :auto, :line (the line integral with the explicit kernel) or :paths (the inversion on paths around the roots)"
"""
    line_auto(D, c, c0) -> Bool

The rule of method = :auto where the line integral covers the arguments: the line integral at c0 = 0, except above the
unit root with deterministic terms while the outer hyperbola of the paths keeps its angle of 45 degrees, for 0 < c <= 1.5
(0 < c <= 2 with two or more regressors), where the inversion on paths is the faster; the inversion on paths at c0 != 0.
Measured on warm calls (scratch timing of November 2026, F and f at mean + (-2, -1, 0, 1, 2) sd and quantiles at 2.5%,
5% and 10%): at c0 = 0 the line integral was 1.3 to 5 times faster than the paths, for every covered design, start and c
from -50 to 2 (it takes the classical integrator's fixed layouts below the unit root); at c0 != 0 the classical
integrator's general kernel (breakpoints of X(v), adaptive pieces) makes it 2 to 5 times slower than the paths.  Since the
thinner rules of the paths (re-measured on November 26, 2026, interleaved warm calls, F and f at five points and the 5%
quantile; no deterministic terms, an intercept, the linear and the quadratic trend, the trend detrended first and an
intercept with a sine and a cosine): at c0 = 0 below the unit root the line integral is 1.03 to 4.1 times faster (median
1.8); above it, for 0 < c <= 1.5 with an intercept and 0 < c <= 2 with the other designs, the paths are 1.3 to 2.3 times
faster (the outer hyperbola has the angle of 45 degrees up to c = 1.5 for the intercept, 2 to 3 for the others, and the
paths slow down as it steepens: the line integral 1.5 times faster for the intercept at c = 2, 2 to 9 times for every
design at c = 3 except the quadratic trend); without deterministic terms the hyperbola steepens from c = 0.5 on and the
line integral stays (the paths 1.3 to 1.5 times faster per value at c = 0.25, 0.5, the line integral from c = 1); at
c0 != 0 the line integral is 4 to 10 times slower.
"""
line_auto(D::Determ, c::Float64, c0::Float64) = c0 == 0 && !(c > 0 && nreg(D) >= 1 && c <= (nreg(D) == 1 ? LINE_PATHS_CMAX1 : LINE_PATHS_CMAX))
"""
    line_vline(D, c, start, gamma0, gls, c0) -> Bool

Whether method = :auto takes the line integral, where it covers the arguments, with c0 > 0 above the unit root on the
vertical line (the outer contour of the paths where no hyperbola keeps the positive roots right of the origin): there the
parabola X(v) of a point x < 0 has two positive roots, whose parts of the outer integrand the line's rule does not resolve,
and the paths' one evaluation was off by up to 1e-3 in the lower tail, below zero or NaN (the range check refused them;
November 26, 2026: an intercept with a sine and a cosine at c = 5, c0 = 2, F(-26.7) = 1.3e-3 for 6.9e-14, F(-8.19) =
6.7e-5 for 4.7e-13, F(-17.4) = -2.3e-4; the quadratic trend at c = 8, c0 = 2; the line's warning fired), while the line
integral agrees with the paths to 2e-11 at x > 0 and with the classical engine for the linear trend.  The points beyond the
line integral's bounds go to the paths, as elsewhere.
"""
line_vline(D::Determ, c::Float64, start::Symbol, gamma0::Real, gls, c0::Float64) =
    c0 > 0 && c > 0 && start != :fixed && isnan(cpath_cached(startmodel(D, c, start, gamma0, gls), c).h)
"""the end of the range above the unit root in which method = :auto takes the paths at c0 = 0 (`line_auto`): with one
regressor (the intercept) and with two or more"""
const LINE_PATHS_CMAX1 = 1.5
const LINE_PATHS_CMAX = 2.0
"""
    line_select(method, D, c, start, gamma0, gls, phi, hfac, c0, kw) -> LineModel or nothing

The provider when `method` (:auto or :line) takes the line integral for these arguments, nothing when the inversion on
paths computes.  method = :line raises an ArgumentError with the reason where the line integral does not cover the
arguments (`line_refusal`, the rank), or where options of the paths are given (phi, hfac, n, ntail, alpha, ...); :auto then
takes the paths silently.
"""
function line_select(method::Symbol, D::Determ, c::Real, start::Symbol, gamma0::Real, gls, phi, hfac, c0::Float64, kw)
    other = String[string(k) for k in keys(kw) if k !== :c0]
    phi === nothing || push!(other, "phi"); (hfac === nothing || hfac == 1) || push!(other, "hfac")
    why = !isempty(other) ? "the options $(join(other, ", ")) belong to the inversion on paths" :
          !(isfinite(c) && isfinite(c0)) ? "c and c0 must be finite" : nothing
    # (before the provider is built; c0 > 0 on the vertical line: see `line_vline`)
    (method === :auto && why === nothing && !line_auto(D, Float64(c), c0) && !line_vline(D, Float64(c), start, gamma0, gls, c0)) && return nothing
    lm = nothing
    why === nothing && ((lm, why) = line_model(D, c, start, gamma0, gls))
    if lm === nothing
        method === :line && throw(ArgumentError("method = :line: " * why))
        return nothing
    end
    lm
end
"""the ArgumentError of method = :line at a point where the classical integrator's paths exceed the bounds of the engine"""
line_path_error(fn::String, lm::LineModel, x::Float64, c0::Float64, p) =
    ArgumentError("$fn: method = :line: $(lm.Mo.P.D.name), c = $(lm.c), c0 = $c0, x = $x: the line integral needs more nodes than the engine allows here ($(sprint(showerror, p))); method = :auto takes the inversion on paths at such points")

"""
    line_dist(method, D, xs, c, which, start, gamma0, gls, phi, hfac, verified, vtol; c0 = 0) -> values or nothing

The values of `dist_cdf` (which = 1) or `dist_pdf` (2) by the line integral with the explicit kernel (the classical
integrator with a `LineModel`), or nothing where the inversion on paths computes them (method = :paths, or :auto where
`line_select` declines).  One evaluation at each point; at a point where the classical integrator's paths would exceed the
bounds of the engine (PathTooLarge, raised before anything is evaluated) method = :auto takes the inversion on paths and
:line raises an ArgumentError.  verified = true: the values checked against one evaluation of the inversion on paths; where
the two differ by more than vtol, the checked inversion on paths (`verify_eval`) is returned, with a warning.
"""
function line_dist(method::Symbol, D::Determ, xs::AbstractVector, c::Real, which::Int, start::Symbol, gamma0::Real, gls, phi, hfac,
                   verified::Bool, vtol::Float64; rule::Symbol = :auto, kw...)
    method === :paths && return nothing
    method in (:auto, :line) || throw(ArgumentError(LINE_METHOD_MSG))
    c0 = Float64(get(kw, :c0, 0.0))
    lm = line_select(method, D, c, start, gamma0, gls, phi, hfac, c0, kw)
    lm === nothing && return nothing
    fn = which == 1 ? "dist_cdf" : "dist_pdf"
    # (the paths at the points the line integral leaves to them, by the rule of one evaluation; with verified = true, the
    # check below, by the present rule)
    function paths(xp)
        Mo = startmodel(D, c, start, gamma0, gls)
        hf, n, nt, rl = path_rule(rule, verified, Mo, c, c0, hfac, kw)
        CP = rule_path(Mo, c, phi, hf, rl, c0)
        v = contour_eval(line_refined(CP, xp; c0), xp; c0, n, ntail = nt)[which]; which == 2 && tangency!(v, CP.Mo, xp, c0); v
    end
    nonfinite(xs, which == 1) do x
        v = similar(x); fb = Int[]
        for k in eachindex(x)
            v[k] = try
                line_value(lm, x[k], which, c0)
            catch e
                p = LX.pathtoolarge(e); p === nothing && rethrow()
                method === :line && throw(line_path_error(fn, lm, x[k], c0, p))
                push!(fb, k); NaN
            end
        end
        isempty(fb) || (v[fb] = paths(x[fb]))
        if verified
            vp = paths(x)
            fin = [k for k in eachindex(x) if isfinite(v[k]) || isfinite(vp[k])]
            d = maximum(k -> abs(v[k] - vp[k]), fin; init = 0.0)
            if !(d <= vtol)
                CP = cpath_cached(startmodel(D, c, start, gamma0, gls), c; phi, hfac = something(hfac, 1.0))
                v = verify_eval((CPv, o) -> contour_eval(line_refined(CPv, x; c0), x; o..., c0)[which], CP, vtol)
                which == 2 && tangency!(v, CP.Mo, x, c0)
                @warn "verified: the line integral and the inversion on paths differ by $(round(d, sigdigits = 2)) (above $vtol); the checked inversion on paths is returned" D.name start c c0
            end
        end
        v
    end
end

"""
    line_quantile(method, D, p, c; x0, tol, start, gamma0, gls, phi, hfac, c0 = 0) -> x or nothing

The p-quantile by the line integral (as `dist_quantile`: Newton's method on F(x) - p with the density, from x0, the step
limited to 1, then `newton_safe`), or nothing where the inversion on paths computes it (`line_select`; with method = :auto
also when the classical integrator's paths would exceed the bounds of the engine during the search).
"""
function line_quantile(method::Symbol, D::Determ, p::Real, c::Real; x0::Real, tol::Real, start::Symbol, gamma0::Real, gls, phi, hfac, kw...)
    method === :paths && return nothing
    method in (:auto, :line) || throw(ArgumentError(LINE_METHOD_MSG))
    c0 = Float64(get(kw, :c0, 0.0))
    lm = line_select(method, D, c, start, gamma0, gls, phi, hfac, c0, kw)
    lm === nothing && return nothing
    FF(y) = (line_value(lm, y, 1, c0), line_value(lm, y, 2, c0))
    try
        x = Float64(x0)
        for _ in 1:30
            F, f = FF(x)
            dx = clamp((F - p) / f, -1.0, 1.0)
            isfinite(dx) || break
            x -= dx
            abs(dx) < tol && return x
        end
        return newton_safe(FF, p, x0, 1.0; tol)
    catch e
        pt = LX.pathtoolarge(e); pt === nothing && rethrow()
        method === :line && throw(ArgumentError("dist_quantile: method = :line: the line integral needs more nodes than the engine allows during the search ($(sprint(showerror, pt))); method = :auto takes the inversion on paths"))
        return nothing
    end
end
