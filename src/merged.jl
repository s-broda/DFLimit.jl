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
distribution function and the density at the point x), :mom, :lincdf (x = a), :envelope, :adfmax, :region.  The rule
and the measurements behind it are the table "Which engine" of the README (dev/engine_table.jl, dev/engine_table.txt).
"""
function autoengine(f::Symbol, S::Spec, c::Float64, c0::Float64, x::Float64)
    cas = S.case; fixed = S.start == :fixed; stat = S.start == :stationary
    if f === :dist
        # the t-ratio.  The faster engine where the two agree to 1e-10, the more accurate one where they do not; the
        # classical line integral keeps c0 = 0 with a zero or stationary start (1.7 to 28 times faster), its conditional
        # integral c0 < 0 with a zero start down to c = -100, and c0 < 0 above the unit root (the general engine off by 7e-3)
        # and the bulk at c >= 7 (2.7 to 34 times faster) through the default below
        stat && c0 != 0 && c <= -3e-3 && return :general                   # 2.4 to 140 times faster; nearer zero the general engine
                                                                           # is off by 1e-10 to 7e-8 in cases SU, ST (tails below 1e-3: see merged_dist)
        if !fixed && !stat && c0 > 0 && -100 < c <= 1
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
        # the fixed start of case DK: 1.04 to 4.3 times faster (beyond, the classical engine is up to 1.5 times faster)
        fixed && cas == "DK" && c <= 1 && ((c * S.g0)^2 / 2 <= 400 || (S.g0 <= 0.5 && c >= -100)) && return :general
        return :classical
    elseif f === :icdf
        # the quantiles at c0 = 0: the general engine's search, which evaluates F and f in one pass on one path, where it is
        # the faster (the trend cases and the fixed start: 1.7 to 4.5 times); the classical Newton search otherwise
        c0 == 0 || return :classical
        fixed && return :general
        (cas == "DT" && c <= 0) && return :general
        (cas == "ST" && c <= -1) && return :general                         # (next to zero, as the moments)
        (cas == "DU" && c == 0) && return :general                          # (from c = -1 on the classical search is the faster, by 1.1 to 1.3)
        return :classical
    elseif f === :mom
        # the general engine at c0 = 0 (1.1 to 25 times faster; equal in case DK) and with a stationary start (1.3 to 5
        # times faster); with c0 != 0 and a zero start the classical engine (2.5 to 3 times faster in cases DK, DU, equal in
        # case DT).  Far from the unit root with c0 = c and a stationary start both engines move by 1e-8 to 5e-7 relative
        # under refinement of their rules, and they differ by up to 4e-9: neither is the more accurate there
        # (with a stationary start next to zero the general engine's moments lose accuracy in cases SU, ST, by 1e-10 at
        # c = -0.1 to 4e-7 at c = -3e-3, where they fail; the classical ones hold to 1e-12 down to c = -1e-6)
        if stat
            (cas == "SU" && c0 != 0) && return :classical                  # (the classical moments of case SU are cheap: 2.4 times faster with c0 != 0)
            return (cas == "SK" ? c <= -3e-3 : c <= -1) ? :general : :classical
        end
        c0 == 0 && return :general
        # (c0 != 0 with a zero start: the classical engine in cases DK, DU, 1.5 to 2.6 times faster, the general in case DT,
        # 1.4 times faster: without c0 the circle of the general engine shares Sigma(theta), with c0 != 0 it does not)
        return cas == "DT" ? :general : :classical
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
        # engine (within its checked computation at c = 4.5 to 7)
        (cas == "DK" && isnan(x)) && return :general
        (cas == "DG" && x != 0 && (c > 4 || S.gls <= -2 || (S.gls <= -1 && c > 0))) && return :general
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
        cas == "DK" && return c <= -0.01 ? :general : :classical
        return c <= -1 ? :general : :classical
    end
    :classical                                                             # (the regions)
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
    throw(ArgumentError("router: unknown function $f"))
end
nomethod(method::Symbol) = method === :auto || throw(ArgumentError("method = :$method names a representation of the classical engine; it does not apply with engine = :general"))
"""the general engine's F (kind = 2) or f (kind = 1) at the points xs"""
function gen_dist(kind::Int, xs::Vector{Float64}, S::Spec, c::Real; c0::Real = 0.0, verified::Bool = false, kw...)
    v = kind == 2 ? DFDet.dist_cdf(S.D, xs, Float64(c); c0 = Float64(c0), verified, gkw(S)..., kw...) :
                    DFDet.dist_pdf(S.D, xs, Float64(c); c0 = Float64(c0), verified, gkw(S)..., kw...)
    kind == 1 && (v[v .< 0] .= 0.0)
    v
end
"""F or f at x of S by the engine of each point (engine = :auto) or by the general engine (engine = :general)"""
function merged_dist(kind::Int, x, S::Spec, c::Real; c0::Real, engine::Symbol, method::Symbol, verified::Bool, kw...)
    xs = xvec(x)
    if engine === :general || S.case === nothing
        nomethod(method)
        return outshape(gen_dist(kind, xs, S, c; c0, verified, kw...), x)
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
function route_dist(engine::Symbol, kind::Int, x, cas::AbstractString, c::Real; c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto,
                    verified::Bool = false, kw...)
    S = case_spec(cas, g0)
    # (a representation of the classical engine named by method, its own keywords, or its checked computation: the
    # classical engine, as before the merge)
    engine === :auto && (method !== :auto || verified || !isempty(kw)) && return nothing
    merged_dist(kind, x, S, c; c0, engine, method, verified, kw...)
end
"""the general engine's p-quantiles of S, started at the normal approximation from its moments where it has them"""
function gen_icdf(p, S::Spec, c::Real; tol::Real = 1e-10, c0::Real = 0.0, kw...)
    x0 = -2.0; sd = 1.0
    # (the start from the moments of the engine the rule takes for them; a design without moments starts at -2)
    m = nothing
    if S.case !== nothing && S.case != "DG" && classical_valid(S, c, c0, :auto) &&
       autoengine(:mom, S, Float64(c), Float64(c0), 0.0) === :classical
        m = DX.ltratio_mom(2, S.case, c; c0, g0 = S.g0, engine = :classical)
    elseif S.D.detrend != :first
        m = try; DFDet.dist_moments(S.D, Float64(c); p = 2, c0 = Float64(c0), gkw(S)...)[1]; catch; nothing; end
    end
    if m !== nothing
        x0 = m[1]; sd = sqrt(max(m[2] - m[1]^2, 1e-6))
    end
    one(pk) = begin
        0 < pk < 1 || throw(ArgumentError("p must lie in (0, 1)"))
        DFDet.dist_quantile(S.D, Float64(pk), Float64(c); x0 = x0 + sd * sqrt(2) * erfinv(2pk - 1), tol, c0 = Float64(c0), gkw(S)..., kw...)
    end
    p isa Number ? one(p) : map(one, p)
end
function route_icdf(engine::Symbol, p, cas::AbstractString, c::Real; tol::Real = 1e-10, c0::Real = 0.0, g0::Real = 0.0, method::Symbol = :auto, kw...)
    S = case_spec(cas, g0)
    if engine === :auto
        # (otherwise the classical search, with each evaluation routed)
        (method === :auto && isempty(kw) && autoengine(:icdf, S, Float64(c), Float64(c0), 0.0) === :general) || return nothing
    end
    nomethod(method)
    gen_icdf(p, S, c; tol, c0, kw...)
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
    S = lin_spec(cas, cbar)
    if engine === :auto
        isempty(kw) || return nothing
        autoengine(:lincdf, S, Float64(c), 0.0, Float64(a)) === :general || return nothing
    end
    gen_lincdf(b, a, S, c)
end
function gen_envelope(c, S::Spec; alpha::Real = 0.05)
    one(ck) = Float64(ck) < 0 ? DFDet.power_envelope(S.D, [Float64(ck)]; level = alpha)[1] : Float64(alpha)
    c isa Number ? one(c) : map(one, c)
end
function route_envelope(engine::Symbol, c, cas::AbstractString; alpha::Real = 0.05, kw...)
    cas = uppercase(String(cas))
    S = cas == "DG" ? Spec(DFDet.trend(1), :zero, 0.0, nothing, "DG") : cas in ("DK", "DU") ? Spec(DFDet.trend(-1), :zero, 0.0, nothing, "DK") :
        throw(ArgumentError("power_envelope: the case must be DK (or DU) or DG"))
    engine === :general && return gen_envelope(c, S; alpha)
    isempty(kw) || return nothing
    one(ck) = autoengine(:envelope, S, Float64(ck), 0.0, 0.0) === :general ? gen_envelope(ck, S; alpha) :
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
function route_region(engine::Symbol, bnd, cas::AbstractString, c::Real; kw...)
    S = Spec(DFDet.trend(-1), :zero, 0.0, nothing, "DK")
    if engine === :auto
        isempty(kw) || return nothing
        autoengine(:region, S, Float64(c), 0.0, 0.0) === :general || return nothing
    end
    DFDet.region_prob(S.D, bnd, Float64(c); kw...)
end

# ---------- the classical functions with a design in place of the case ----------
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
        nomethod(method)
        outshape(gen_dist($kind, xvec(x), S, c; c0, verified, kw...), x)
    end
end
function ltratio_icdf(p, D::DFDet.Determ, c::Real = 0; tol::Real = 1e-10, c0::Real = 0.0, g0::Real = 0.0, start = nothing, gls = nothing,
                      method::Symbol = :auto, engine::Symbol = :auto, kw...)
    S = design_spec(D; g0, start, gls)
    classical_or_nothing(S, engine, c, c0, method) && return ltratio_icdf(p, S.case, c; tol, c0, g0 = S.g0, method, engine, kw...)
    nomethod(method)
    gen_icdf(p, S, c; tol, c0, kw...)
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
function ltratio_lincdf(b, a::Real, D::DFDet.Determ, c::Real = 0; start = nothing, g0::Real = 0.0, gls = nothing, engine::Symbol = :auto, kw...)
    S = design_spec(D; g0, start, gls)
    cas = S.case == "DK" ? "DK" : (S.case == "DG") ? "DG" : nothing
    if engine !== :general && cas !== nothing
        return ltratio_lincdf(b, a, cas, c; engine, kw...)
    end
    engine === :classical && throw(ArgumentError("engine = :classical: the linear form of the classical engine has the cases DK and DG only"))
    gen_lincdf(b, a, S, c)
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
