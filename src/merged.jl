# =====================================================================================================================
# One interface over the two engines: the classical engine (DFExact: the three classical cases, the line, conditional
# and contour integrals) and the general engine (DFDet: any deterministic terms, the inversion on paths around the
# singularities).  The classical functions take, in place of the case string, a design of the general engine (trend(p),
# perron(...), fdesign(...), ltrend(p), design_from_matrix(X), detrend_first(D)), with the start by the keywords g0 (a
# fixed start J(0) = g0) and start = :stationary, and GLS detrending by gls = cbar.  Every call, with a case string or a
# design, takes the engine of `autoengine` (engine = :auto, the default), or the one forced by engine = :classical or
# :general.
# =====================================================================================================================
import .DFExact: ltratio_cdf, ltratio_pdf, ltratio_icdf, ltratio_mom, coef_cdf, coef_icdf, adfmax_cdf, adfmax_pdf, adfmax_icdf,
                 ltratio_lincdf, power_envelope, ltratio_region
const DX = DFExact

"""
A computation's design and start: the design D, the start (:zero, :fixed, :stationary), g0 (the fixed start J(0) = g0),
gls (GLS detrending at abar = 1 + cbar/n, or nothing) and the classical engine's case code for them ("DK", "DU", "DT",
"SK", "SU", "ST", "DG"), or nothing where only the general engine has the computation.
"""
struct Spec
    D::DFDet.Determ
    start::Symbol
    g0::Float64
    gls::Union{Nothing,Float64}
    case::Union{Nothing,String}
end

isdesign(D::DFDet.Determ, E::DFDet.Determ) = D.C == E.C && D.br == E.br && D.loc == E.loc && D.detrend == E.detrend
const KUT = ((-1, 'K'), (0, 'U'), (1, 'T'))
"""the classical case of a design and start, or nothing: trend(-1), trend(0), trend(1) (the monomial basis, the one-step
regression) with a zero, fixed or stationary start, and trend(1) GLS-detrended at cbar = -13.5 with a zero start (DG)"""
function classical_case(D::DFDet.Determ, start::Symbol, gls)
    for (p, l) in KUT
        isdesign(D, DFDet.trend(p)) || continue
        gls === nothing && return string(start == :stationary ? 'S' : 'D', l)
        (l == 'T' && start == :zero && Float64(gls) == DX.CBAR_GLS) && return "DG"
        return nothing
    end
    nothing
end
"""the Spec of a case string ("DG": the trend GLS-detrended at cbar = -13.5)"""
function case_spec(cas::AbstractString, g0::Real = 0.0)
    cas = uppercase(String(cas))
    cas in DX.CASES || throw(ArgumentError("the case must be one of DK, DU, DT, SK, SU, ST, DG"))
    cas == "DG" && return Spec(DFDet.trend(1), :zero, 0.0, DX.CBAR_GLS, "DG")
    p = cas[2] == 'K' ? -1 : cas[2] == 'U' ? 0 : 1
    st = cas[1] == 'S' ? :stationary : (g0 != 0 ? :fixed : :zero)
    Spec(DFDet.trend(p), st, Float64(g0), nothing, cas)
end
"""the Spec of a design: start = :zero, :fixed (with g0), :stationary, or nothing (then :fixed when g0 != 0, else :zero)"""
function design_spec(D::DFDet.Determ; g0::Real = 0.0, start = nothing, gls = nothing)
    isfinite(g0) || throw(ArgumentError("g0 must be a finite real scalar"))
    st = start === nothing ? (g0 != 0 ? :fixed : :zero) : Symbol(start)
    st in (:zero, :fixed, :stationary) || throw(ArgumentError("start must be :zero, :fixed or :stationary"))
    st == :stationary && g0 != 0 && throw(ArgumentError("g0 (a fixed start) and start = :stationary exclude each other"))
    st == :fixed && g0 == 0 && (st = :zero)
    gl = gls === nothing ? nothing : Float64(gls)
    Spec(D, st, Float64(g0), gl, classical_case(D, st, gl))
end
"""the keywords of the general engine's functions for the start and the detrending of S"""
gkw(S::Spec) = (start = S.start, gamma0 = S.g0, gls = S.gls)
"""whether the classical engine accepts the arguments (its checks of the range of c, the start and the method)"""
function classical_valid(S::Spec, c::Real, c0::Real, method::Symbol)
    S.case === nothing && return false
    try
        DX.ltratio_args(S.case, c; c0, g0 = S.g0, method)
        true
    catch e
        e isa ArgumentError || rethrow()
        false
    end
end
outshape(F::Vector{Float64}, x) = x isa Number ? F[1] : reshape(F, size(x))
xvec(x) = x isa Number ? [Float64(x)] : Float64.(vec(collect(x)))

"""
    autoengine(f, S, c, c0, x) -> :classical or :general

The choice of `engine = :auto` for a computation the classical engine accepts (a case string, or a design with the
classical case S.case, within the classical engine's checks; elsewhere the general engine computes): f = :dist (the
distribution function and the density at the point x), :icdf, :mom, :lincdf (x = a), :envelope, :adfmax, :region (x = 1
for the parabola of the t-ratio, with its c0; 0 for the other boundaries).  The rule and the measurements behind it are the
table "Which engine" of the README (dev/engine_table.jl, dev/engine_table.txt); it was re-derived on November 26, 2026,
after the thinner rules of one evaluation, from interleaved timings of the two engines in each cell (a cell moved where
the other engine was at least 1.3 times faster in every repetition) and checks of the new choice's one evaluation against
the other engine's checked computation on grids of x from the 0.01% to the 99.99% point.
"""
function autoengine(f::Symbol, S::Spec, c::Float64, c0::Float64, x::Float64)
    cas = S.case; fixed = S.start == :fixed; stat = S.start == :stationary
    # (above the range of the general engine, DFDet.gen_cmax: 20 without deterministic terms, 16 with them, 10 with them
    # and a fixed start, where it refuses, the classical engine; below, the rule)
    c > DFDet.gen_cmax(S.D, fixed) && return :classical
    if f === :dist
        # the t-ratio.  The faster engine where the two agree to 1e-10, the more accurate one where they do not; the
        # classical engine keeps through the default below c0 = 0 with a zero or stationary start below the unit root (its
        # line integral; the general engine's line integral 1.0 to 1.24 times faster, below the threshold of 1.3), the
        # conditional integral with c0 < 0 in case DU and in case DT outside -50 <= c <= -10, c0 < 0 above c = 2 and the
        # bulk at c >= 7 (1.1 to 32 times faster)
        stat && c0 != 0 && c <= -3e-3 && return :general                   # 2.4 to 140 times faster; nearer zero the general engine
                                                                           # is off by 1e-10 to 7e-8 in cases SU, ST (tails below 1e-3: see merged_dist)
        # x = 0, the points of the coefficient statistic (coef_cdf, coef_icdf: F at x = 0 with the null value c0 + y), one
        # point per call (scratch timing of November 2026: cases DK, DU, DT, c from 1 to -80, y at the 1 to 99% quantiles of
        # the statistic and c0 = -3, 2): with c0 > 0 the classical line integral, 2 to 12 times faster there (X(v) = c0 v^2 +
        # 1/2 has no root), where the rule below takes the general engine for the t-ratio; with c0 < 0 in case DK for c > -5
        # the general engine, 1.4 to 2.7 times faster (the classical line integral with its general kernel); otherwise the
        # rule of the t-ratio (with c0 < 0 the classical engine, equal to 1.8 times faster in cases DU, DT; since November 26,
        # 2026 in case DT for -50 <= c <= -10 the general engine, 1.6 to 3.1 times faster at x = 0 as well).  Up to c = 2 since
        # then, with the cell of c0 > 0 in case DT
        if x == 0 && !fixed && !stat && c0 != 0 && -100 < c <= 2 && cas in ("DK", "DU", "DT")
            c0 > 0 && return :classical
            (cas == "DK" && c > -5) && return :general
        end
        # a zero start with c0 > 0 below c = 1 (case DT below c = 2): the general engine, 2.4 to 5 times faster at c = -100,
        # 3.6 to 7 times in case DT at c = 1.5, 2 (re-measured November 26, 2026, after the thinner rules of the paths; the
        # rule took the classical engine at c = -100 itself and in case DT above c = 1)
        if !fixed && !stat && c0 > 0 && (c <= 1 || (cas == "DT" && c <= 2))
            # 1.6 to 4.8 times faster.  In case DK the density is infinite at the tangency x* = -sqrt(2 c0); just above it
            # (within 1e-3 of x*) the classical line integral is the more accurate (the general engine's density off by up
            # to 1.5e-3 relative at x* + 1e-9, the classical by 5e-10), below it the general engine (the classical density
            # off by 10% at x* - 1e-9 at c = -20, c0 = 0.5, and by orders of magnitude at c = -50, c0 = 3)
            if cas == "DK"
                xs = -sqrt(2c0)
                (xs < x <= xs + 1e-3 * max(1.0, abs(xs))) && return :classical
            end
            return :general
        end
        !fixed && !stat && c0 != 0 && c < -100 && return :general            # 2.4 to 3.3 times faster
        # a zero start with c0 < 0: in case DK near and above the unit root (-5 < c <= 2), where the classical engine takes its
        # line integral with the general kernel, the general engine, 6 to 28 times faster; in case DT for -50 <= c <= -10, 1.3
        # to 2.7 times faster (the point x = 0 of the coefficient statistic too); elsewhere the classical engine (re-measured
        # November 26, 2026: in case DK from c = -5 on, where the classical engine takes its conditional integral, it is 1.05
        # to 2.6 times faster; in case DU and nearer the unit root in case DT the gains of either are below 1.3 or vary with
        # c0)
        !fixed && !stat && c0 < 0 && ((cas == "DK" && -5 < c <= 2) || (cas == "DT" && -50 <= c <= -10)) && return :general
        # a zero start at c0 = 0 above the unit root, where the outer hyperbola of the general engine keeps its angle of 45
        # degrees: its paths (the general engine's method :auto there, `DFDet.line_auto`), in case DU for 0 < c <= 1.5 and in
        # case DT for 0 < c <= 2, 1.6 to 2.3 times faster than the classical engine (re-measured November 26, 2026)
        !fixed && !stat && c0 == 0 && ((cas == "DU" && 0 < c <= 1.5) || (cas == "DT" && 0 < c <= 2)) && return :general
        # the fixed start of case DK: 1.04 to 4.3 times faster (beyond, the classical engine is up to 1.5 times faster),
        # except with c0 < 0 near the unit root (c > -1): there the outer hyperbola of the general engine does not resolve
        # the upper tail, and its one evaluation was off by 2.7e-6 at the 99.99% point (c = 0.5, c0 = -5, g0 = 1) and 2.9e-2
        # beyond (mean + 6 sd at c0 = -10, where at mean + 4 sd it gave 1.23 and the range check refused), 6.7e-6 at c = 0
        # with g0 = 1.5 (November 2026, `scratch/rules/probe_dkfixed6.jl`); the classical engine's line integral, its
        # checked computation and the conditional integral agree to 1e-12 there.  Re-measured November 26, 2026: beyond
        # (c g0)^2/2 <= 400 the general engine is 1.4 to 6 times faster also where |g0| <= 1.5 and |c g0| <= 50 with
        # c >= -100 (the region of g0 <= 0.5 with c >= -100 among them); with |g0| = 2 or |c g0| > 50 the gains vary or the
        # classical engine is the faster.  (|g0|: the law depends on |g0| only; the rule tested g0 <= 0.5 until then)
        ag = abs(S.g0)
        fixed && cas == "DK" && c <= 1 && !(c0 < 0 && c > -1) && ((c * ag)^2 / 2 <= 400 || (c >= -100 && ag <= 1.5 && abs(c) * ag <= 50)) && return :general
        # the fixed start of cases DU, DT far from the unit root: with c0 > 0 below c = -20 (and below c = -10 where |c g0| >
        # 20), and with any c0 below c = -20 where |c g0| > 60, the classical contour integral's one evaluation was off, in
        # the density by up to 5e-6 of its peak (case DU at c = -30, g0 = 0.5, c0 = 2), 4.2e-3 (c = -75, c0 = 5) and 2.2e-2
        # (g0 = 1.5, c = -60, c0 = 5), in the distribution function by 3.9e-5 (case DU, g0 = 1.5, c = -60, c0 = -2) and
        # 6.3e-6 (g0 = 0.9, c = -100), and its checked computation in the density by up to 8.4e-6 of the peak; the general
        # engine's density there is the derivative of the distribution function on which the two engines agree (to 1e-10
        # of the peak; Richardson's central difference), its F within 1e-10 of the classical checked one.  It takes 0.23 to
        # 0.9 times the time of the classical one evaluation down to c = -45, up to 1.8 times at c = -60 and 1.2 to 5 times
        # below c = -75 (November 2026, `scratch/rules/probe_ctr_far2.jl` to `probe_ctr_far6.jl`)
        fixed && cas in ("DU", "DT") && c < -10 && ((c0 > 0 && (c < -20 || abs(c * S.g0) > 20)) || (c < -20 && abs(c * S.g0) > 60)) && return :general
        return :classical
    elseif f === :icdf
        # the quantiles at c0 = 0: the general engine's search, which evaluates F and f in one pass on one path, where it is
        # the faster (the trend cases and the fixed start: 1.7 to 4.5 times); the classical Newton search otherwise
        c0 == 0 || return :classical
        fixed && return :general
        (cas == "DT" && c <= 2) && return :general                          # (above the unit root, 0 < c <= 2, by the paths: 2.3 to 2.5 times)
        # (a stationary start: cases SU, ST from c = -0.5 on, 1.3 to 1.7 times faster, November 26, 2026; next to zero the
        # general engine's search is off, by 3.4e-8 in case SU and 2.6e-7 in case ST at c = -1e-3, within 2e-9 of the classical
        # search for c <= -0.01)
        (cas in ("SU", "ST") && c <= -0.5) && return :general
        (cas == "DU" && 0 <= c <= 1.5) && return :general                   # (above the unit root by the paths, 2.5 times; from c = -1 on the
                                                                            # gains of the general search vary from 1.3 to 1.6)
        return :classical
    elseif f === :mom
        # the general engine at c0 = 0 (1.1 to 25 times faster; equal in case DK) and with a stationary start (1.3 to 5
        # times faster); with c0 != 0 and a zero start the classical engine (2.5 to 3 times faster in cases DK, DU, equal in
        # case DT).  Far from the unit root with c0 = c and a stationary start both engines move by 1e-8 to 5e-7 relative
        # under refinement of their rules, and they differ by up to 4e-9: neither is the more accurate there
        # (with a stationary start next to zero the general engine's moments lose accuracy in cases SU, ST, by 1e-10 at
        # c = -0.1 to 4e-7 at c = -3e-3, where they fail; the classical ones hold to 1e-12 down to c = -1e-6)
        # (re-measured November 26, 2026: with c0 != 0 below c = -5 the classical engine is 1.8 to 10 times faster in cases
        # SK, ST and DT, which took the general engine; nearer the unit root the general engine is the faster there, in case
        # ST at c = c0 = -1 25 times, in case DT at c = 0, c0 = -2 2.8 times)
        if stat
            (cas == "SU" && c0 != 0) && return :classical                  # (the classical moments of case SU are cheap: 2.4 times faster with c0 != 0)
            (c0 != 0 && c <= -5) && return :classical
            return (cas == "SK" ? c <= -3e-3 : c <= -1) ? :general : :classical
        end
        c0 == 0 && return :general
        # (c0 != 0 with a zero start: the classical engine in cases DK, DU, 1.5 to 2.6 times faster, and in case DT below c =
        # -5; the general engine in case DT nearer the unit root: without c0 the circle of the general engine shares
        # Sigma(theta), with c0 != 0 it does not)
        return (cas == "DT" && c >= -5) ? :general : :classical
    elseif f === :lincdf
        # x = a.  Case DK: the classical code (repaired: the region), except the law of V (a = NaN), where the general
        # engine is 40 to 400 times faster, within 1e-14 of the classical code below c = 4 and the more accurate above
        # (against the closed form of M_V inverted on hyperbolas of its own: the general engine within 1e-13 at c = 4, 5 and
        # 4.4e-12 at c = 7, the classical inversion off by 5.9e-10, 1.6e-8, 1.8e-7).  Elsewhere the general engine is at most
        # 1.3 times faster in case DK (at a = 0, where the classical code is the closed form, 1.1 times faster).  Case DG,
        # a != 0, up to c = 4: the general engine for cbar <= -2, and for cbar <= -1 above the unit root, 1.4 to 80 times
        # faster than the classical conditional integral given R (equal at cbar = -7 and -10 below the unit root) and
        # within 1.1e-12 of it on the hyperbola (cbar <= -7), 7e-12 on the vertical line (cbar from -5 to -2, where
        # Filon's rule takes the parts at the root); at cbar = -1 below the unit root the classical code is 1.1 to 1.3 times
        # faster.  At a = 0 the classical closed form (the general engine, 1.2 to 3 times faster, is within 7.2e-14 of it).
        # Beyond c = 4, where the classical code refuses (its inner inversion needs more than 2e6 nodes), the general
        # engine (within its checked computation at c = 4.5 to 7).  Re-measured November 26, 2026: below and at the unit
        # root the classical code is now 2 to 11 times faster in case DG at every cbar (the general engine took cbar <= -2
        # there until then); above it the general engine stays for cbar <= -1
        (cas == "DK" && isnan(x)) && return :general
        (cas == "DG" && x != 0 && (c > 4 || (S.gls <= -1 && c > 0))) && return :general
        return :classical
    elseif f === :adfmax
        # a stationary start: the general engine, 1.7 to 3 times faster and within 4e-12 of the classical values (which
        # do not move under finer rules); next to zero it is off by 2e-9 (c = -1e-4); a zero start: the classical engine
        # (1.9 to 2.7 times faster)
        return (stat && c <= -0.1) ? :general : :classical
    elseif f === :envelope
        # the general engine finds the critical value to 1e-10 where the classical bisects it to 1e-9 (in case DG to 2e-9 in
        # the power, against a bisection to 1e-14: the general engine within 1.4e-12); near the unit root in case DG the
        # classical engine (from c = -0.5 on the general engine is the slower, and at c = -0.01 it exhausts the memory)
        # (case DK: within 3e-12 of the classical engine's down to c = -0.01; at c = -1e-4, where a = c/2 puts the critical
        # value at the edge of the support of U - aV, the general engine is off by 1.1e-9)
        # (re-measured November 26, 2026: in case DG the classical engine is 4.7 to 21 times faster at c = -2 to -20, and
        # its power within 2.1e-9 of the general engine's; the classical engine there, as in the C rule)
        cas == "DK" && return c <= -0.01 ? :general : :classical
        return :classical
    elseif f === :region
        # x = 1 for the parabola of the t-ratio (with its c0), 0 for the other boundaries.  The parabola at c0 = 0 below and
        # at the unit root: the general engine (its line integral), 2.5 to 27 times faster than the classical region code's
        # conditional integral (re-measured November 26, 2026, c from 0 to -50; at c = 2 the classical code is 1.7 times
        # faster, dev/engine_table.txt); with c0 != 0, above the unit root and for the line the classical code
        return (x == 1 && c0 == 0 && c <= 0) ? :general : :classical
    end
    :classical
end

# ---------- the router of the classical engine's user-level functions (DFExact.ROUTE) ----------
function router(f::Symbol, engine::Symbol, args...; kw...)
    f === :dist && return route_dist(engine, args...; kw...)
    f === :icdf && return route_icdf(engine, args...; kw...)
    f === :mom && return route_mom(engine, args...; kw...)
    f === :lincdf && return route_lincdf(engine, args...; kw...)
    f === :envelope && return route_envelope(engine, args...; kw...)
    f === :adfmax && return route_adfmax(engine, args...; kw...)
    f === :region && return route_region(engine, args...; kw...)
    f === :coef_cdf && return inv_coef_cdf(engine, args...; kw...)
    f === :coef_icdf && return inv_coef_icdf(engine, args...; kw...)
    f === :lincdf_inv && return inv_lincdf(engine, args...; kw...)
    f === :envelope_inv && return inv_envelope(engine, args...; kw...)
    f === :mom_inv && return inv_mom(engine, args...; kw...)
    f === :mom_start && return mom_start(args...; kw...)
    f === :dist2 && return route_dist2(args...; kw...)
    throw(ArgumentError("router: unknown function $f"))
end

"""
    mom_start(p, cas, c; c0 = 0, g0 = 0) -> m or nothing

The first p moments of the t-ratio of a classical case for the start of a quantile search (`ltratio_icdf`, `gen_icdf`):
from one jet per node, the method of Bao and Kan (`DFKan.tratio_mom1`, 0.2 to 1 ms for two moments, where the circles of
the engines take 10 to 150 ms), or nothing outside its region (`DFKan.mom_ok`; case DG) or where its error estimate
exceeds 1e-6 relative; the search then starts from the engines' circles.  The start only places the first step and the
bracket of the search: the quantile is the root of F(x) - p to the tolerance of the search either way.
"""
function mom_start(p::Int, cas::AbstractString, c::Real; c0::Real = 0.0, g0::Real = 0.0)
    MOM_START[] || return nothing
    cs = uppercase(String(cas))
    (inv_case(cs) && DFKan.mom_ok(cs, Float64(c), Float64(c0), Float64(g0))) || return nothing
    m, e = try
        DFKan.tratio_mom1(DFKan.limcf(cs, c, g0), max(p, 2), Float64(c0), 1e-10)
    catch err
        DFDet.nofallback(err) && rethrow()
        return nothing
    end
    sc = [max(abs(m[j]), abs(m[2])^(j / 2)) for j in eachindex(m)]
    all(j -> isfinite(m[j]) && e[j] <= 1e-6 * sc[j], 1:p) || return nothing
    m[1:p]
end
"""whether the quantile searches start from the moments from jets (`mom_start`; false: from the engines' circles, as before
November 2026, kept for checking)"""
const MOM_START = Ref(true)

# ---------- the single inversion of Bao and Kan (DFKan) for the half-plane statistics of the six classical cases ----------
# The method of Yong Bao and Raymond Kan, "Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in
# First-Order Autoregressions: O(1) Algorithms": one inversion of the closed-form joint characteristic function along the
# ray of the half-plane, adaptive quadrature over decades and the tail completed by integration by parts (src/DFKan*.jl).
"""the cases of the closed-form transform of the single inversion (case DG has none)"""
inv_case(cas::AbstractString) = uppercase(String(cas)) in ("DK", "DU", "DT", "SK", "SU", "ST")
"""
    inv_take(fn, method, engine, cas, c, kw) -> Bool

Whether the single inversion computes: with method = :inversion always (an ArgumentError with engine = :general, for a
case it does not have or with engine keywords), with method = :auto under engine = :auto in its cases, without engine
keywords, for c up to DFKan.INV_CMAX; otherwise the engines compute (method = :engine or a representation of the
engines, engine = :classical or :general).
"""
function inv_take(fn::String, method::Symbol, engine::Symbol, cas::AbstractString, c::Real, kw)
    if method === :inversion || method === :jets
        engine === :general && throw(ArgumentError("$fn: method = :$method (the devices of Bao and Kan) does not apply with engine = :general"))
        inv_case(cas) || throw(ArgumentError("$fn: method = :$method applies to the cases DK, DU, DT, SK, SU, ST"))
        isempty(kw) || throw(ArgumentError("$fn: method = :$method takes no further options ($(join(keys(kw), ", ")))"))
        return true
    end
    method === :auto && engine === :auto && isempty(kw) && inv_case(cas) && c <= DFKan.INV_CMAX
end
"""the warning of the single inversion where error estimates exceed the tolerance (one per call)"""
function inv_warn(fn::String, E::AbstractVector{Float64}, tol::Float64)
    bad = count(e -> !(e <= tol), E)
    bad == 0 && return nothing
    @warn "$fn: the error estimate of the single inversion exceeds the tolerance $tol at $bad of $(length(E)) points (largest $(maximum(E))); verified = true recomputes them by the engines' checked computation"
    nothing
end
"""f(k) for k = 1..n on the engine's tasks where n > 1 (the single inversion is thread-safe: its transform is immutable)"""
function inv_map(f, n::Int)
    out = Vector{Tuple{Float64,Float64}}(undef, n)
    n == 1 ? (out[1] = f(1)) : DX.tmap!(f, out, n)
    first.(out), last.(out)
end
function inv_coef_cdf(engine::Symbol, y, cas::AbstractString, c::Real; c0::Real, g0::Real, method::Symbol, verified::Bool, tol::Float64, err::Bool, kw...)
    DX.ltratio_args(cas, c; c0, g0)                                    # (the classical engine's checks of the arguments)
    inv_take("coef_cdf", method, engine, cas, c, kw) || return nothing
    F = DFKan.limcf(cas, c, g0); ys = xvec(y)
    P, E = inv_map(k -> DFKan.coef_cdf1(F, ys[k], Float64(c0), tol), length(ys))
    if verified
        bad = findall(e -> !(e <= tol), E)
        isempty(bad) || (P[bad] = DX.coef_cdf(ys[bad], cas, c; c0, g0, method = :engine, verified = true))
    else
        inv_warn("coef_cdf", E, tol)
    end
    err ? (outshape(P, y), outshape(E, y)) : outshape(P, y)
end
function inv_coef_icdf(engine::Symbol, p, cas::AbstractString, c::Real; tol::Float64, c0::Real, g0::Real, method::Symbol, verified::Bool, err::Bool, kw...)
    DX.ltratio_args(cas, c; c0, g0)
    inv_take("coef_icdf", method, engine, cas, c, kw) || return nothing
    (isfinite(tol) && tol > 0) || throw(ArgumentError("coef_icdf: tol must be a positive finite number"))
    F = DFKan.limcf(cas, c, g0); ps = xvec(p)
    for pk in ps; 0 < pk < 1 || throw(ArgumentError("coef_icdf: p must lie in (0, 1)")); end
    Y, E = inv_map(k -> DFKan.coef_icdf1(F, ps[k], Float64(c0), tol, DFKan.TOL), length(ps))
    if verified
        bad = findall(e -> !(e <= DFKan.TOL), E)
        isempty(bad) || (Y[bad] = DX.coef_icdf(ps[bad], cas, c; tol, c0, g0, method = :engine, verified = true))
    else
        inv_warn("coef_icdf", E, DFKan.TOL)
    end
    if err                                                             # (in y: the estimate of F divided by the density)
        for k in eachindex(Y); E[k] /= DFKan.coef_pdf1(F, Y[k], Float64(c0), DFKan.TOL)[1]; end
        return outshape(Y, p), outshape(E, p)
    end
    outshape(Y, p)
end
function inv_lincdf(engine::Symbol, b, a::Real, cas::AbstractString, c::Real; method::Symbol, verified::Bool, tol::Float64, err::Bool, kw...)
    cs = uppercase(String(cas))
    if method === :inversion
        isnan(a) && throw(ArgumentError("ltratio_lincdf: method = :inversion needs a finite a (a = NaN, the law of V, is the general engine's)"))
        cs == "DG" && throw(ArgumentError("ltratio_lincdf: method = :inversion applies to case DK (the designs of the six classical cases), not to case DG"))
    end
    (isnan(a) || cs != "DK") && method !== :inversion && return nothing
    isinf(a) && throw(ArgumentError("ltratio_lincdf: a must be a finite real scalar (NaN: the law of V)"))
    inv_lin(engine, b, Float64(a), "DK", c, 0.0; method, verified, tol, err, kw...)
end
function inv_mom(engine::Symbol, p::Int, cas::AbstractString, c::Real; c0::Float64, g0::Float64, method::Symbol, verified::Bool, tol::Float64, err::Bool, kw...)
    inv_take("ltratio_mom", method === :jets ? :jets : method, engine, cas, c, kw) || return nothing
    method === :auto && !DFKan.mom_ok(String(cas), Float64(c), c0, g0) && return nothing
    m, e = DFKan.tratio_mom1(DFKan.limcf(cas, c, g0), max(p, 2), c0, tol)
    # (the tolerance relative to max(|m_j|, m_2^{j/2}): a mean near zero, as with c0 = c far above the unit root, has no
    # relative accuracy to speak of)
    sc = [max(abs(m[j]), abs(m[2])^(j / 2)) for j in eachindex(m)]
    m = m[1:p]; e = e[1:p]
    bad = findall(j -> !(e[j] <= tol * sc[j]), 1:p)
    if !isempty(bad)
        if verified
            m = DX.ltratio_mom(p, cas, c; c0, g0, method = :engine)
        else
            @warn "ltratio_mom: the error estimate of the moments from jets exceeds the tolerance $tol (relative) for the moments $(join(bad, ", ")); verified = true recomputes them by the engines"
        end
    end
    err ? (m, e) : m
end
"""the linear form of a classical case (cas, g0) by the single inversion where `inv_take` and `DFKan.lin_ok` take it, or nothing"""
function inv_lin(engine::Symbol, b, a::Float64, cas::String, c::Real, g0::Float64; method::Symbol, verified::Bool, tol::Float64, err::Bool, kw...)
    inv_take("ltratio_lincdf", method, engine, cas, c, kw) || return nothing
    bs = xvec(b)
    # (method = :auto: the points outside the region of `DFKan.lin_ok` by the engines)
    ok = method === :inversion ? trues(length(bs)) : [DFKan.lin_ok(cas, Float64(c), g0, a, bk) for bk in bs]
    any(ok) || return nothing
    F = DFKan.limcf(cas, c, g0); ik = findall(ok)
    P = fill(NaN, length(bs)); E = fill(NaN, length(bs))
    P[ik], E[ik] = inv_map(k -> DFKan.lin_cdf1(F, bs[ik[k]], a, tol), length(ik))
    all(ok) || (P[.!ok] = lin_engine(bs[.!ok], a, cas, c, g0))
    if verified
        bad = findall(k -> ok[k] && !(E[k] <= tol), eachindex(E))
        isempty(bad) || (P[bad] = lin_engine(bs[bad], a, cas, c, g0))
    else
        inv_warn("ltratio_lincdf", E[ik], tol)
    end
    err ? (outshape(P, b), outshape(E, b)) : outshape(P, b)
end
"""the linear form by the engines (method = :engine): case DK with a zero start by the classical function, the other
classical cases by the general engine (as before the single inversion)"""
lin_engine(bs::Vector{Float64}, a::Float64, cas::String, c::Real, g0::Float64) =
    (cas == "DK" && g0 == 0) ? DX.ltratio_lincdf(bs, a, "DK", c; method = :engine) :
    DFDet.lin_cdf(case_spec(cas, g0).D, bs, a, Float64(c); gkw(case_spec(cas, g0))...)
function inv_envelope(engine::Symbol, c, cas::AbstractString; alpha::Real, method::Symbol, verified::Bool, tol::Float64, err::Bool, kw...)
    cs = uppercase(String(cas))
    method === :inversion && cs == "DG" && throw(ArgumentError("power_envelope: method = :inversion applies to case DK (DU), not to case DG"))
    cs == "DK" || return nothing
    0 < alpha < 1 || throw(ArgumentError("power_envelope: the level must lie in (0, 1)"))
    cv = xvec(c)
    inv_take("power_envelope", method, engine, "DK", 0.0, kw) || return nothing
    # (method = :auto: the points outside the region of `DFKan.envelope_ok` by the engines)
    ok = method === :inversion ? trues(length(cv)) : [DFKan.envelope_ok(ck) for ck in cv]
    any(ok) || return nothing
    ik = findall(ok); P = fill(NaN, length(cv)); E = fill(NaN, length(cv))
    P[ik], E[ik] = inv_map(k -> DFKan.envelope1(cv[ik[k]], Float64(alpha), tol), length(ik))
    all(ok) || (P[.!ok] = DX.power_envelope(cv[.!ok], "DK"; alpha, method = :engine))
    if verified
        bad = findall(k -> ok[k] && !(E[k] <= tol), eachindex(E))
        isempty(bad) || (P[bad] = DX.power_envelope(cv[bad], "DK"; alpha, method = :engine))
    else
        inv_warn("power_envelope", E[ik], tol)
    end
    err ? (outshape(P, c), outshape(E, c)) : outshape(P, c)
end
"""
    genmethod(method) -> :auto, :line or :paths

The keyword `method` where the general engine computes (engine = :general, or a design the classical engine does not
have): the general engine's own choice, which `DFDet.dist_cdf`, `dist_pdf` and `dist_quantile` take as their `method`:
:auto (the line integral with the explicit kernel where it covers the design and c0 = 0, the inversion on paths around
the roots otherwise), :line (the line integral with the explicit kernel, an ArgumentError with the reason where it does
not cover the design; :single, the classical engine's name of the line integral, is taken as :line, as by the C router)
or :paths (the inversion on paths).  The other representations of the classical engine (:double, :conditional,
:contour) are refused.  (For a design that is a classical case, with engine = :auto, method = :line is the
classical engine's line integral, which ltratio_args accepts under that name, and :paths, which it does not accept,
takes the general engine's paths.)
"""
function genmethod(method::Symbol)
    method === :single && return :line                  # (the classical name of the line integral, as in ltratio_args and the C router)
    method in (:auto, :line, :paths) && return method
    throw(ArgumentError("method = :$method names a representation of the classical engine; it does not apply with engine = :general (whose methods are :auto, :line and :paths)"))
end
"""the general engine's F (kind = 2) or f (kind = 1) at the points xs (method: `genmethod`)"""
function gen_dist(kind::Int, xs::Vector{Float64}, S::Spec, c::Real; c0::Real = 0.0, verified::Bool = false, method::Symbol = :auto, kw...)
    v = kind == 2 ? DFDet.dist_cdf(S.D, xs, Float64(c); c0 = Float64(c0), verified, method, gkw(S)..., kw...) :
                    DFDet.dist_pdf(S.D, xs, Float64(c); c0 = Float64(c0), verified, method, gkw(S)..., kw...)
    kind == 1 && (v[v .< 0] .= 0.0)
    v
end
"""F or f at x of S by the engine of each point (engine = :auto) or by the general engine (engine = :general)"""
function merged_dist(kind::Int, x, S::Spec, c::Real; c0::Real, engine::Symbol, method::Symbol, verified::Bool, kw...)
    xs = xvec(x)
    if engine === :general || S.case === nothing
        return outshape(gen_dist(kind, xs, S, c; c0, verified, method = genmethod(method), kw...), x)
    end
    eng = [isfinite(xk) ? autoengine(:dist, S, Float64(c), Float64(c0), xk) : :classical for xk in xs]
    all(==(:classical), eng) && return nothing
    F = zeros(length(xs))
    gi = findall(==(:general), eng)
    isempty(gi) || (F[gi] = gen_dist(kind, xs[gi], S, c; c0, verified))
    # the tails: where the classical engine would take its conditional integral, whose error is relative, the values of
    # the general engine below 1e-3 (absolute errors) go back to it (as the classical engine's checked computation
    # recomputes its tails)
    md = DX.ltratio_args(S.case, c; c0, g0 = S.g0, method)
    for k in gi
        F[k] < 1e-3 && DX.automethod(md[1], md[2], Float64(c), Float64(c0), S.g0, xs[k]) === :double && (eng[k] = :classical)
    end
    ci = findall(==(:classical), eng)
    isempty(ci) || (F[ci] = DX.ltratio_dist(kind, xs[ci], S.case, c; c0, g0 = S.g0, method, verified, engine = :classical, kw...))
    outshape(F, x)
end
"""
    route_dist2(x, cas, c; c0, g0, method, engine) -> (F, f) or nothing

F and f of the t-ratio at one point x in one evaluation, for the Newton steps of the classical quantile search
(`ltratio_icdf`): where the rule gives the point to the general engine (engine = :auto, method = :auto, no further
options), its paths return both from one pass (`DFDet.dist_both`), with the values of `ltratio_cdf` and `ltratio_pdf`
bit for bit; nothing elsewhere (the classical engine, the tails that go back to it, other options), and the search takes
the two evaluations.
"""
function route_dist2(x::Real, cas::AbstractString, c::Real; c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto, engine::Symbol = :auto, kw...)
    (DIST2[] && method === :auto && engine === :auto && isempty(kw) && isfinite(x)) || return nothing
    S = case_spec(cas, g0)
    autoengine(:dist, S, Float64(c), Float64(c0), Float64(x)) === :general || return nothing
    F, f = DFDet.dist_both(S.D, [Float64(x)], Float64(c); c0 = Float64(c0), gkw(S)...)
    # (the tails below 1e-3 where the classical engine would take its conditional integral go back to it, as in
    # `merged_dist`, for F and f each: the search then takes the two evaluations)
    md = DX.ltratio_args(S.case, c; c0, g0 = S.g0, method)
    ((F[1] < 1e-3 || f[1] < 1e-3) && DX.automethod(md[1], md[2], Float64(c), Float64(c0), S.g0, Float64(x)) === :double) && return nothing
    DX.cdf_range_check("ltratio_cdf", DX.cdf_args(cas, c, c0, g0), x, F)
    F[1], max(f[1], 0.0)
end
"""whether the classical quantile search takes F and f in one evaluation where the general engine computes (`route_dist2`;
false: two evaluations, as before November 2026, kept for checking)"""
const DIST2 = Ref(true)
function route_dist(engine::Symbol, kind::Int, x, cas::AbstractString, c::Real; c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto,
                    verified::Bool = false, kw...)
    S = case_spec(cas, g0)
    # (a representation of the classical engine named by method, its own keywords, or its checked computation: the
    # classical engine, as before the merge)
    engine === :auto && (method !== :auto || verified || !isempty(kw)) && return nothing
    merged_dist(kind, x, S, c; c0, engine, method, verified, kw...)
end
"""the general engine's p-quantiles of S, started at the normal approximation from the moments: from jets in the six
classical cases (`mom_start`), otherwise from the engines' moments where they have them"""
function gen_icdf(p, S::Spec, c::Real; tol::Real = 1e-10, c0::Real = 0.0, method::Symbol = :auto, kw...)
    x0 = -2.0; sd = 1.0
    # (the start from the moments of the engine the rule takes for them; a design without moments starts at -2)
    m = nothing
    # (the six classical cases: the moments from jets, `mom_start`, where they hold; otherwise the engine of the rule)
    cv = S.case !== nothing && S.case != "DG" && classical_valid(S, c, c0, :auto)
    cv && (m = mom_start(2, S.case, c; c0, g0 = S.g0))
    if m !== nothing
    elseif cv && autoengine(:mom, S, Float64(c), Float64(c0), 0.0) === :classical
        m = DX.ltratio_mom(2, S.case, c; c0, g0 = S.g0, engine = :classical)
    elseif S.D.detrend != :first
        # (where the moments fail, the start at -2; an interrupt or an exhausted memory reaches the caller)
        m = try; DFDet.dist_moments(S.D, Float64(c); p = 2, c0 = Float64(c0), gkw(S)...)[1]; catch e; DFDet.nofallback(e) && rethrow(); nothing; end
    end
    if m !== nothing
        x0 = m[1]; sd = sqrt(max(m[2] - m[1]^2, 1e-6))
    end
    one(pk) = begin
        0 < pk < 1 || throw(ArgumentError("p must lie in (0, 1)"))
        DFDet.dist_quantile(S.D, Float64(pk), Float64(c); x0 = x0 + sd * sqrt(2) * erfinv(2pk - 1), tol, c0 = Float64(c0), method, gkw(S)..., kw...)
    end
    p isa Number ? one(p) : map(one, p)
end
function route_icdf(engine::Symbol, p, cas::AbstractString, c::Real; tol::Real = 1e-10, c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto, kw...)
    S = case_spec(cas, g0)
    if engine === :auto
        # (otherwise the classical search, with each evaluation routed)
        (method === :auto && isempty(kw) && autoengine(:icdf, S, Float64(c), Float64(c0), 0.0) === :general) || return nothing
    end
    gen_icdf(p, S, c; tol, c0, method = genmethod(method), kw...)
end
gen_mom(p::Int, S::Spec, c::Real; c0::Real = 0.0, kw...) = DFDet.dist_moments(S.D, Float64(c); p, c0 = Float64(c0), gkw(S)..., kw...)[1]
function route_mom(engine::Symbol, p::Int, cas::AbstractString, c::Real; c0::Real = 0.0, g0::Real = 0.0, kw...)
    S = case_spec(cas, g0)
    if engine === :auto
        isempty(kw) || return nothing
        autoengine(:mom, S, Float64(c), Float64(c0), 0.0) === :general || return nothing
    end
    gen_mom(p, S, c; c0, kw...)
end
"""the design of the linear form of a classical case: "DK" (no deterministic terms) or "DG" (the trend GLS-detrended at cbar)"""
lin_spec(cas::AbstractString, cbar::Real) = uppercase(String(cas)) == "DG" ? Spec(DFDet.trend(1), :zero, 0.0, Float64(cbar), "DG") :
    uppercase(String(cas)) in ("DK", "DU") ? Spec(DFDet.trend(-1), :zero, 0.0, nothing, "DK") :
    throw(ArgumentError("ltratio_lincdf: the case must be DK or DG"))
function gen_lincdf(b, a::Real, S::Spec, c::Real)
    bs = xvec(b)
    v = isnan(a) ? DFDet.v_cdf(S.D, bs, Float64(c); gkw(S)...) : DFDet.lin_cdf(S.D, bs, Float64(a), Float64(c); gkw(S)...)
    outshape(v, b)
end
function route_lincdf(engine::Symbol, b, a::Real, cas::AbstractString, c::Real; cbar::Real = DX.CBAR_GLS, kw...)
    # (a = +-Inf gave 0.5 in the classical code: refused, as by the general engine and the C router; NaN is the law of V)
    isinf(a) && throw(ArgumentError("ltratio_lincdf: a must be a finite real scalar (NaN: the law of V)"))
    isfinite(cbar) || throw(ArgumentError("ltratio_lincdf: cbar must be a finite real scalar < 0"))
    S = lin_spec(cas, cbar)
    if engine === :auto
        isempty(kw) || return nothing
        autoengine(:lincdf, S, Float64(c), 0.0, Float64(a)) === :general || return nothing
    end
    gen_lincdf(b, a, S, c)
end
function gen_envelope(c, S::Spec; alpha::Real = 0.05)
    one(ck) = isnan(ck) ? NaN : Float64(ck) < 0 ? DFDet.power_envelope(S.D, [Float64(ck)]; level = alpha)[1] : Float64(alpha)
    c isa Number ? one(c) : map(one, c)
end
function route_envelope(engine::Symbol, c, cas::AbstractString; alpha::Real = 0.05, kw...)
    cas = uppercase(String(cas))
    S = cas == "DG" ? Spec(DFDet.trend(1), :zero, 0.0, nothing, "DG") : cas in ("DK", "DU") ? Spec(DFDet.trend(-1), :zero, 0.0, nothing, "DK") :
        throw(ArgumentError("power_envelope: the case must be DK (or DU) or DG"))
    0 < alpha < 1 || throw(ArgumentError("power_envelope: the level must lie in (0, 1)"))
    engine === :general && return gen_envelope(c, S; alpha)
    isempty(kw) || return nothing
    # (1 at c = -Inf, the limit, where the classical bisection found no bracket; NaN at NaN, where the classical engine
    # gave the level, as the C router; the level at c >= 0)
    one(ck) = isnan(ck) ? NaN : Float64(ck) == -Inf ? 1.0 : autoengine(:envelope, S, Float64(ck), 0.0, 0.0) === :general ? gen_envelope(ck, S; alpha) :
              DX.power_envelope(ck, cas; alpha, engine = :classical)
    c isa Number ? one(c) : map(one, c)
end
function gen_adfmax(pdf::Bool, x, S::Spec, c::Real; kw...)
    xs = xvec(x)
    v = pdf ? DFDet.adfmax_pdf(S.D, xs, Float64(c); start = S.start, gamma0 = S.g0, kw...) : DFDet.adfmax_cdf(S.D, xs, Float64(c); start = S.start, gamma0 = S.g0, kw...)
    pdf && (v[v .< 0] .= 0.0)
    outshape(v, x)
end
function route_adfmax(engine::Symbol, pdf::Bool, x, cas::AbstractString, c::Real; kw...)
    S = case_spec(cas)
    engine === :general && return gen_adfmax(pdf, x, S, c; kw...)
    isempty(kw) || return nothing
    autoengine(:adfmax, S, Float64(c), 0.0, 0.0) === :general || return nothing
    # (the tails below 1e-3 by the classical conditional integral, whose error is relative, as for the t-ratio)
    xs = xvec(x); v = gen_adfmax(pdf, xs, S, c)
    tl = findall(k -> isfinite(xs[k]) && v[k] < 1e-3, eachindex(xs))
    isempty(tl) || (v[tl] = DX.adfmax_vals(pdf, xs[tl], cas, c; engine = :classical))
    outshape(v, x)
end
"""the region below a line or a parabola at a non-finite b or x: 1, 0, NaN (the classical code gave 0, also at NaN), an
error for a slope a or a c0 that is not finite; nothing for the other boundaries and for finite ones (as the C router)"""
function region_nonfinite(bnd)
    fld(f) = bnd isa AbstractDict ? get(bnd, f, get(bnd, String(f), nothing)) : (haskey(bnd, f) ? bnd[f] : nothing)
    ty = fld(:type); ty === nothing && return nothing
    ty = lowercase(String(ty)); ty in ("line", "parabola") || return nothing
    (fld(:v0) === nothing && fld(:brks) === nothing) || return nothing
    s, v = ty == "line" ? (fld(:a), fld(:b)) : (something(fld(:c0), 0.0), fld(:x))
    (s === nothing || v === nothing) && return nothing
    isfinite(s) || throw(ArgumentError(ty == "line" ? "region_prob: the slope a of the line must be a finite real scalar" :
                                                      "region_prob: c0 of the parabola must be a finite real scalar"))
    isfinite(v) && return nothing
    isnan(v) ? NaN : v > 0 ? 1.0 : 0.0
end
"""the arguments of the rule for a region (`autoengine` with f = :region): (1.0, c0) for the parabola of the t-ratio without
v0 and brks (with its c0, finite), (0.0, 0.0) for the other boundaries"""
function region_rule_args(bnd)
    fld(f) = bnd isa AbstractDict ? get(bnd, f, get(bnd, String(f), nothing)) : (haskey(bnd, f) ? bnd[f] : nothing)
    ty = fld(:type)
    (ty === nothing || lowercase(String(ty)) != "parabola" || fld(:v0) !== nothing || fld(:brks) !== nothing) && return 0.0, 0.0
    c0 = something(fld(:c0), 0.0)
    (c0 isa Real && isfinite(c0)) ? (1.0, Float64(c0)) : (0.0, 0.0)
end
function route_region(engine::Symbol, bnd, cas::AbstractString, c::Real; kw...)
    v = region_nonfinite(bnd); v === nothing || return v
    S = Spec(DFDet.trend(-1), :zero, 0.0, nothing, "DK")
    if engine === :auto
        isempty(kw) || return nothing
        xr, c0r = region_rule_args(bnd)
        autoengine(:region, S, Float64(c), c0r, xr) === :general || return nothing
    end
    DFDet.region_prob(S.D, bnd, Float64(c); kw...)
end

# ---------- the classical functions with a design in place of the case ----------
# (method: the classical engine's representations (:auto, :single or :line, :double, :contour) where the classical engine
# computes; where the general engine computes, its own (`genmethod`): :auto, :line, :paths.  So for a design that is not a
# classical case method = :line forces the general engine's line integral with the explicit kernel and :paths its
# inversion on paths; for a design that is a classical case :line is the classical line integral and :paths the general
# engine's paths; with engine = :general :line and :paths are the general engine's for every design.)
"""the classical function f with the case of S (engine = :auto, its rule) where the classical engine accepts the arguments;
otherwise nothing (the general engine computes)"""
function classical_or_nothing(S::Spec, engine::Symbol, c::Real, c0::Real, method::Symbol)
    engine === :general && return false
    if engine === :classical
        S.case === nothing && throw(ArgumentError("engine = :classical: the classical engine has the cases DK, DU, DT, SK, SU, ST and DG only (trend(-1), trend(0), trend(1), the last GLS-detrended at cbar = -13.5)"))
        return true
    end
    classical_valid(S, c, c0, method)
end
for (f, kind) in ((:ltratio_cdf, 2), (:ltratio_pdf, 1))
    @eval function $f(x, D::DFDet.Determ, c::Real = 0; c0::Real = 0.0, g0::Real = 0.0, start = nothing, gls = nothing, method::Symbol = :auto,
                      verified::Bool = false, engine::Symbol = :auto, kw...)
        S = design_spec(D; g0, start, gls)
        classical_or_nothing(S, engine, c, c0, method) && return DX.ltratio_dist($kind, x, S.case, c; c0, g0 = S.g0, method, verified, engine, kw...)
        outshape(gen_dist($kind, xvec(x), S, c; c0, verified, method = genmethod(method), kw...), x)
    end
end
function ltratio_icdf(p, D::DFDet.Determ, c::Real = 0; tol::Real = 1e-10, c0::Real = 0.0, g0::Real = 0.0, start = nothing, gls = nothing,
                      method::Symbol = :auto, engine::Symbol = :auto, kw...)
    S = design_spec(D; g0, start, gls)
    classical_or_nothing(S, engine, c, c0, method) && return ltratio_icdf(p, S.case, c; tol, c0, g0 = S.g0, method, engine, kw...)
    gen_icdf(p, S, c; tol, c0, method = genmethod(method), kw...)
end
function ltratio_mom(p::Int, D::DFDet.Determ, c::Real = 0; c0::Real = 0.0, g0::Real = 0.0, start = nothing, gls = nothing, engine::Symbol = :auto, kw...)
    S = design_spec(D; g0, start, gls)
    classical_or_nothing(S, engine, c, c0, :auto) && S.case != "DG" && return ltratio_mom(p, S.case, c; c0, g0 = S.g0, engine, kw...)
    gen_mom(p, S, c; c0, kw...)
end
function coef_cdf(y, D::DFDet.Determ, c::Real = 0; c0::Real = 0.0, g0::Real = 0.0, start = nothing, gls = nothing, engine::Symbol = :auto, kw...)
    S = design_spec(D; g0, start, gls)
    classical_or_nothing(S, engine, c, c0, :auto) && return coef_cdf(y, S.case, c; c0, g0 = S.g0, engine, kw...)
    ys = xvec(y)
    outshape(DFDet.coef_cdf(S.D, ys, Float64(c); c0 = Float64(c0), gkw(S)..., kw...), y)
end
"""
    coef_pdf(y, cas, c = 0; c0 = 0, g0 = 0, method = :auto, engine = :auto, verified = false, tol = 1e-12, err = false) -> f
    coef_pdf(y, D::Determ, c = 0; c0, g0, start, gls, ...)

The limiting density of n(phi^ - phi0), the coefficient statistic (U - c0 V)/V of `coef_cdf` (y a number or an array).
In the cases DK, DU, DT (zero or fixed start g0) and SK, SU, ST up to c = 20 it is the single inversion of Bao and Kan
("Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1)
Algorithms"; `DFLimit.DFKan.ratio_pdf`): the density of a ratio by the formula of Geary, one integral of the derivative
of the joint characteristic function along the ray, from jets, its tail completed by integration by parts, 0.05 to 1 ms
a value.  `tol` is its tolerance relative to the scale max(1, E[V]/sd(U)) of the density, `err = true` returns `(f,
errors)`, and a warning names the points at which the estimate exceeds the tolerance.  `method = :inversion` forces it;
`method = :engine` (and engine = :general, case DG, a design beyond the classical cases, further options) is the
general engine's density, the kernel 2 v^3 of its contour integral (`DFDet.coef_pdf`).  verified = true recomputes the
values whose estimate exceeds the tolerance by the general engine's checked computation.  The classical engine has no
density of the coefficient statistic: with engine = :classical the single inversion computes.

    coef_pdf(-8.039, "DK")                          # the density at the 5% critical value of rho^
"""
function DFDet.coef_pdf(y, cas::AbstractString, c::Real = 0; c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto, engine::Symbol = :auto,
                        verified::Bool = false, tol::Real = 1e-12, err::Bool = false, kw...)
    method = DX.inv_method("coef_pdf", method, false)
    engine in (:auto, :classical, :general) || throw(ArgumentError("engine must be :auto, :classical or :general"))
    S = case_spec(cas, g0)
    engine === :general || DX.ltratio_args(cas, c; c0, g0)
    if inv_take("coef_pdf", method, engine === :classical ? :auto : engine, cas, c, kw)
        F = DFKan.limcf(cas, c, g0); ys = xvec(y); tl = Float64(tol)
        R = Vector{NTuple{3,Float64}}(undef, length(ys))
        length(ys) == 1 ? (R[1] = DFKan.coef_pdf1(F, ys[1], Float64(c0), tl)) : DX.tmap!(k -> DFKan.coef_pdf1(F, ys[k], Float64(c0), tl), R, length(ys))
        P = [r[1] for r in R]; E = [r[2] for r in R]; T = [r[3] for r in R]
        bad = findall(k -> !(E[k] <= T[k]), eachindex(E))
        if verified
            isempty(bad) || (P[bad] = DFDet.coef_pdf(S.D, ys[bad], Float64(c); c0 = Float64(c0), gkw(S)..., verified = true))
        elseif !isempty(bad)
            @warn "coef_pdf: the error estimate of the single inversion exceeds its tolerance at $(length(bad)) of $(length(E)) points (largest $(maximum(E[bad]))); verified = true recomputes them by the general engine's checked computation"
        end
        return err ? (outshape(P, y), outshape(E, y)) : outshape(P, y)
    end
    v = DFDet.coef_pdf(S.D, xvec(y), Float64(c); c0 = Float64(c0), gkw(S)..., verified, kw...)
    v[v .< 0] .= 0.0
    DX.noerr(outshape(v, y), err)
end
function DFDet.coef_pdf(y, D::DFDet.Determ, c::Real = 0; c0::Real = 0.0, g0::Real = 0.0, start = nothing, gls = nothing, engine::Symbol = :auto,
                        method::Symbol = :auto, err::Bool = false, kw...)
    S = design_spec(D; g0, start, gls)
    classical_or_nothing(S, engine, c, c0, :auto) && return DFDet.coef_pdf(y, S.case, c; c0, g0 = S.g0, engine, method, err, kw...)
    DX.inv_method("coef_pdf", method, false) === :inversion && throw(ArgumentError("coef_pdf: method = :inversion applies to the cases DK, DU, DT, SK, SU, ST"))
    v = DFDet.coef_pdf(S.D, xvec(y), Float64(c); c0 = Float64(c0), gkw(S)..., kw...)
    v[v .< 0] .= 0.0
    DX.noerr(outshape(v, y), err)
end
"""
    coef_mom(p, cas, c = 0; c0 = 0, g0 = 0, tol = 1e-10, err = false) -> m
    coef_mom(p, D::Determ, c = 0; c0, g0, start, ...)

m[j] = E[{(U - c0 V)/V}^j], j = 1..p, the moments of the limit of n(phi^ - phi0) (the coefficient statistic of
`coef_cdf`), in the cases DK, DU, DT (zero or fixed start g0) and SK, SU, ST (a design that is one of them): Sawa's
representation m_j = int_0^inf [d^j M(s1, -s2)/ds1^j at s1 = 0] s2^{j-1}/Gamma(j) ds2 with the derivative from one jet of
order j per node, the method of Bao and Kan ("Exact Finite-Sample Distribution and Moments of the Least Squares Estimator
in First-Order Autoregressions: O(1) Algorithms"; `DFLimit.DFKan.ratio_mom`), adaptive quadrature in log s2 to the
relative tolerance `tol`.  `err = true` returns `(m, errors)`.  Up to c = 20; the engines have no moments of the
coefficient statistic (an ArgumentError for case DG, other designs and engine = :general).

    coef_mom(1, "DK")[1]                            # -1.7814: the mean of n(phi^ - 1) at the unit root
"""
function coef_mom(p::Int, cas::AbstractString, c::Real = 0; c0::Real = 0.0, g0::Real = 0.0, tol::Real = 1e-10, err::Bool = false, engine::Symbol = :auto,
                  method::Symbol = :auto)
    p >= 1 || throw(ArgumentError("p must be a positive integer"))
    method in (:auto, :inversion, :jets) || throw(ArgumentError("coef_mom: method must be :auto (the jets of Bao and Kan)"))
    engine === :general && throw(ArgumentError("coef_mom: the general engine has no moments of the coefficient statistic"))
    engine in (:auto, :classical) || throw(ArgumentError("engine must be :auto, :classical or :general"))
    DX.ltratio_args(cas, c; c0, g0)
    inv_case(cas) || throw(ArgumentError("coef_mom: the moments of the coefficient statistic are available for the cases DK, DU, DT, SK, SU, ST"))
    c <= DFKan.INV_CMAX || method !== :auto || throw(ArgumentError("coef_mom: c must be at most $(DFKan.INV_CMAX) (method = :jets beyond)"))
    m, e = DFKan.coef_mom1(DFKan.limcf(cas, c, g0), p, Float64(c0), Float64(tol))
    err ? (m, e) : m
end
function coef_mom(p::Int, D::DFDet.Determ, c::Real = 0; c0::Real = 0.0, g0::Real = 0.0, start = nothing, gls = nothing, kw...)
    S = design_spec(D; g0, start, gls)
    (S.case !== nothing && inv_case(S.case)) || throw(ArgumentError("coef_mom: the moments of the coefficient statistic are available for the cases DK, DU, DT, SK, SU, ST (trend(-1), trend(0), trend(1) with a zero, fixed or stationary start)"))
    coef_mom(p, S.case, c; c0, g0 = S.g0, kw...)
end
function coef_icdf(p, D::DFDet.Determ, c::Real = 0; tol::Real = 1e-9, c0::Real = 0.0, g0::Real = 0.0, start = nothing, gls = nothing,
                   engine::Symbol = :auto, kw...)
    S = design_spec(D; g0, start, gls)
    classical_or_nothing(S, engine, c, c0, :auto) && return coef_icdf(p, S.case, c; tol, c0, g0 = S.g0, engine, kw...)
    one(pk) = DFDet.coef_quantile(S.D, Float64(pk), Float64(c); tol, c0 = Float64(c0), gkw(S)..., kw...)
    p isa Number ? one(p) : map(one, p)
end
for (f, pdf) in ((:adfmax_cdf, false), (:adfmax_pdf, true))
    @eval function $f(x, D::DFDet.Determ, c::Real = 0; start = nothing, engine::Symbol = :auto, kw...)
        S = design_spec(D; start)
        cas = S.case
        if engine !== :general && cas !== nothing && cas in ("DU", "DT", "SU", "ST")
            return DX.adfmax_vals($pdf, x, cas, c; engine, kw...)
        end
        engine === :classical && throw(ArgumentError("engine = :classical: ADFmax of the classical engine has the cases DU, DT, SU, ST only"))
        gen_adfmax($pdf, x, S, c; kw...)
    end
end
function adfmax_icdf(p, D::DFDet.Determ, c::Real = 0; tol::Float64 = 1e-10, x0::Float64 = -2.0, start = nothing, engine::Symbol = :auto, kw...)
    S = design_spec(D; start)
    if engine !== :general && S.case !== nothing && S.case in ("DU", "DT", "SU", "ST")
        return adfmax_icdf(p, S.case, c; tol, x0, engine, kw...)
    end
    one(pk) = DFDet.adfmax_quantile(S.D, Float64(pk), Float64(c); x0, tol, start = S.start, kw...)
    p isa Number ? one(p) : map(one, p)
end
function ltratio_lincdf(b, a::Real, D::DFDet.Determ, c::Real = 0; start = nothing, g0::Real = 0.0, gls = nothing, engine::Symbol = :auto,
                        method::Symbol = :auto, verified::Bool = false, tol::Real = 1e-12, err::Bool = false, kw...)
    S = design_spec(D; g0, start, gls)
    method = DX.inv_method("ltratio_lincdf", method, false)
    # the single inversion of Bao and Kan for the six classical cases with any start, within the classical checks of the
    # arguments (`inv_lin`); method = :inversion forces it
    inv = engine !== :general && S.case !== nothing && inv_case(S.case) && isfinite(a) && classical_valid(S, c, 0.0, :auto)
    method === :inversion && !inv && throw(ArgumentError("ltratio_lincdf: method = :inversion applies to the cases DK, DU, DT, SK, SU, ST (trend(-1), trend(0), trend(1)) within the classical engine's checks, a finite a and engine = :auto or :classical"))
    if inv
        isinf(a) && throw(ArgumentError("ltratio_lincdf: a must be a finite real scalar (NaN: the law of V)"))
        r = inv_lin(engine, b, Float64(a), S.case, c, S.g0; method, verified, tol = Float64(tol), err, kw...); r === nothing || return r
    end
    # (the classical linear form has the zero start only: no deterministic terms with a fixed start, case string "DK",
    # take the general engine)
    cas = S.start != :zero ? nothing : S.case == "DK" ? "DK" : (S.case == "DG") ? "DG" : nothing
    if engine !== :general && cas !== nothing
        return ltratio_lincdf(b, a, cas, c; engine, method = :engine, err, kw...)
    end
    engine === :classical && throw(ArgumentError("engine = :classical: the linear form of the classical engine has the cases DK and DG only"))
    DX.noerr(gen_lincdf(b, a, S, c), err)
end
function ltratio_region(bnd, D::DFDet.Determ, c::Real = 0; start = nothing, g0::Real = 0.0, engine::Symbol = :auto, kw...)
    S = design_spec(D; g0, start)
    # (the classical regions have the zero start only: no deterministic terms with a fixed start, whose case string is
    # also "DK", take the general engine, as in the C router)
    dk = S.case == "DK" && S.start == :zero
    engine !== :general && dk && return ltratio_region(bnd, "DK", c; engine, kw...)
    engine === :classical && throw(ArgumentError("engine = :classical: the regions of the classical engine have case DK only"))
    DFDet.region_prob(S.D, bnd, Float64(c); start = S.start, gamma0 = S.g0, kw...)
end
"""
    power_envelope(c, D::Determ; alpha = 0.05, engine = :auto)

The Gaussian power envelope for the design D: for trend(-1) and trend(0) the classical case DK, for trend(1) the case DG
(the point-optimal test with the data GLS-detrended at c), and for every other design the envelope of the general
construction (`pt_envelope`: the point-optimal statistic of the design against cbar = c, by one inversion).
"""
function power_envelope(c, D::DFDet.Determ; alpha::Real = 0.05, engine::Symbol = :auto, kw...)
    cas = isdesign(D, DFDet.trend(-1)) || isdesign(D, DFDet.trend(0)) ? "DK" : isdesign(D, DFDet.trend(1)) ? "DG" : nothing
    cas !== nothing && return power_envelope(c, cas; alpha, engine, kw...)
    engine === :classical && throw(ArgumentError("engine = :classical: the envelope of the classical engine has the cases DK (DU) and DG only"))
    cs = c isa Number ? [Float64(c)] : Float64.(vec(collect(c)))
    v = DFDet.pt_envelope(D, cs; level = alpha, kw...)
    c isa Number ? v[1] : reshape(v, size(c))
end
