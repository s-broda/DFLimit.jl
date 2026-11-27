# =====================================================================================================================
# Seasonal unit roots at a harmonic frequency: the first t-ratio of the regression of Hylleberg, Engle, Granger and Yoo
# for a pair of complex unit roots
# =====================================================================================================================
# At a frequency strictly between 0 and pi (pi/2 for quarterly data, each pair of harmonic frequencies of monthly data) the
# limit process is complex: Z = W_a + i W_b with independent standard Brownian motions, dZ = c Z dr + dW, Z(0) = 0, and Z~
# its residual on the deterministic terms (the real trend functions of a design of the engine with complex coefficients:
# trend(0) for seasonal dummies, trend(1) for seasonal dummies and seasonal trends).  With
#
#     R = Re int Z~* dZ,   V = int |Z~|^2,
#
# the first t-ratio is t3 = (R - c0 V)/sqrt(V) (the t-ratio of pi_3 in the quarterly numbering: the coefficient of
# y_{3,t-2}, whose limit has the shape of a Dickey-Fuller distribution).  (The second t-ratio and the F statistic of the
# same regression are not part of this release.)
#
# t3.  (R, V) is the sum of two independent copies of the lead paper's (U, V), so E exp(sR + tV) = e^{-s}/D(s,t) with the
# engine's determinant: its singularities in s are poles, and the inner integral of the inversion on paths is the sum of the
# residues at the roots s_k(t).  The density and the distribution function are single integrals over the engine's outer
# contour of closed-form functions (`t3_residues`), the kernels at the roots those of the engine (`ctr_moments`).

# ---------- t3: the sum of the residues ----------
"""
    t3_residues(CP, x, c0; tol = 1e-17) -> (F, f)

P(t3 <= x) and the density of t3 = (R - c0 V)/sqrt(V) at the finite points x on the outer contour CP of the real design.
With e^{s} E exp(sR + tV) = P(t)^2 prod_+ (s_k - s)^{-1} prod_- (s - s_k)^{-1} and its partial fractions sum_+ w_k/(s_k -
s) + sum_- w_k/(s - s_k), the inner integral of the inversion on paths is the sum of the residues,

    f(x)      = (1/2 pi i) int sum_k w_k(t) K^f(s_k(t), t) dt,
    P(t3 > x) = P{X(sqrt V) < 0} + (1/2 pi i) int sum_k w_k(t) K^F(s_k(t), t) dt,

with X(v) = c0 v^2 + x v + 1, K^f = int 2 v^2 e^{-t v^2 - s X(v)} dv and K^F = (1/s) int 2 v e^{-t v^2 - s X(v)} dv over
the part of v > 0 on which X has the sign of the root's group; the first term of the second line is a single inversion of
M_V(t) = P^2 prod_+ (1/s_k) prod_- (-1/s_k).  The outer integral is the engine's rule on CP, the nodes until a chunk of 32
whose terms all stay below tol.
"""
function t3_residues(CP::CPath, x::Vector{Float64}, c0::Float64; tol::Float64 = 1e-17)
    r = size(CP.R, 1); rp = CP.Mo.rp; nx = length(x)
    F = zeros(nx); f = zeros(nx)
    pcs = [xpieces(xv, c0, 1.0) for xv in x]
    wk = zeros(ComplexF64, r)
    K1 = length(CP.t); k = 0
    while k < K1
        mx = 0.0; kend = min(k + 32, K1)
        @inbounds for kk in k+1:kend
            t = CP.t[kk]; P2 = exp(2 * CP.lP[kk])
            MV = P2
            for j in 1:r; MV /= (j <= rp ? CP.R[j, kk] : -CP.R[j, kk]); end
            for j in 1:r
                w = P2; sj = CP.R[j, kk]
                for i in 1:r
                    i == j && continue
                    w /= (i <= rp ? (CP.R[i, kk] - sj) : (sj - CP.R[i, kk]))
                end
                wk[j] = w
            end
            for ix in 1:nx
                plus, minus = pcs[ix]
                gF = zero(ComplexF64); gf = zero(ComplexF64)
                for j in 1:r
                    pc = j <= rp ? plus : minus
                    isempty(pc) && continue
                    sj = CP.R[j, kk]
                    _, M1, M2 = ctr_moments(t, sj, c0, x[ix], 1.0, pc)
                    gf += wk[j] * 2 * M2; gF += wk[j] * 2 * M1 / sj
                end
                for (a, b) in minus
                    gF += MV * (exp(-t * a * a) - (isfinite(b) ? exp(-t * b * b) : zero(ComplexF64))) / t
                end
                tF = CP.wt[kk] * imag(gF * CP.dt[kk]); tf = CP.wt[kk] * imag(gf * CP.dt[kk])
                F[ix] += tF; f[ix] += tf
                mx = max(mx, abs(tF), abs(tf))
            end
        end
        k = kend
        mx < tol && break
    end
    1 .- F, f
end

"""the smallest distance of two roots of one group along the path CP, relative to their size (the residues divide by it)"""
function t3_gap(CP::CPath)
    r = size(CP.R, 1); rp = CP.Mo.rp; g = Inf
    for k in axes(CP.R, 2), (lo, hi) in ((1, rp), (rp + 1, r)), i in lo:hi, j in i+1:hi
        g = min(g, abs(CP.R[i, k] - CP.R[j, k]) / max(abs(CP.R[i, k]), abs(CP.R[j, k])))
    end
    g
end
"""the relative distance of two roots of a group below which `t3_values` takes the doubled transform on the paths"""
const T3_GAP = 1e-4

"""
    t3_doubled(CP) -> CPath

The transform e^{-s}/D(s,t) in the form the engine's inversion on paths integrates, P(t) prod (s_k - s)^{-1/2} ...: every
root of the design twice, log P twice, the shift a = 1; `contour_eval(t3_doubled(CP), x)` is the inversion on paths of the
same transform, independent of the residues.
"""
function t3_doubled(CP::CPath)
    Mo = CP.Mo; r = Mo.r; rp = Mo.rp
    idx = [repeat(1:rp, inner = 2); repeat(rp+1:r, inner = 2)]
    Mo2 = Model(Mo.P, :zero, 0.0, Mo.N, Mo.A, Mo.B, Mo.L, Mo.Dv, 2r, 2rp, 2 * Mo.rm, 1.0, zeros(Mo.N), 0.5)
    CPath(CP.c, CP.tau, CP.tstar, CP.a, CP.b, CP.h, CP.phi, CP.t, CP.dt, CP.R[idx, :], 2 .* CP.lP, CP.minpos, CP.maxneg, Mo2, CP.fx, CP.wt)
end

# ---------- the user-level functions ----------
"""the checks of the seasonal functions: c <= 0 and finite (above the unit root not implemented), c >= cmin"""
function seas_cargs(fn::String, c, cmin::Float64)
    c isa Real || return nothing
    isfinite(c) || throw(ArgumentError("$fn: c must be a finite real scalar"))
    c <= 0 || throw(ArgumentError("$fn: c must be at most 0 (the seasonal statistics are implemented below and at the unit root)"))
    c >= cmin || throw(ArgumentError("$fn: c must be at least $cmin (the range in which the computation is checked)"))
    nothing
end
"""
    hegy_t3_cdf(D, xs, c = 0; c0 = 0, vol = nothing, verified = false, vtol = 1e-9) -> F (vector)

The limit of P(t3 <= x) for the first t-ratio of the regression of Hylleberg, Engle, Granger and Yoo at a harmonic
frequency (strictly between 0 and pi: pi/2 for quarterly data, where t3 is the t-ratio of pi_3, the coefficient of
y_{3,t-2}; each pair of harmonic frequencies of monthly data), the one of Dickey-Fuller shape: t3 = (R - c0 V)/sqrt(V), with
R = Re int Z~* dZ and V = int |Z~|^2 of the complex Ornstein-Uhlenbeck process Z, dZ = c Z dr + dW (W = W_a + i W_b), Z(0) =
0, and Z~ its residual on the deterministic terms of the design D (the real trend functions with complex coefficients):
`trend(-1)` for a regression without seasonal terms (a constant or a linear trend leave the harmonic statistics alone),
`trend(0)` for seasonal dummies (with or without a trend), `trend(1)` for seasonal dummies and seasonal trends, and any
design of the general engine (breaks in the seasonal means, ...).  c (<= 0) and c0 in the units of the root's modulus 1 +
c/n, n the number of years; vol = volpath(...) a volatility path (as in `dist_cdf`; c may then be a vector).

The moment generating function of (R, V) is e^{-s}/D(s, t) with the general engine's determinant D, whose singularities
in s are poles: the inner integral of the inversion on paths is the sum of the residues at the roots, and the
distribution function is one integral over the engine's outer contour of closed-form functions (0.05 ms a value on a
cached contour).  Where two roots of a group come within 1e-4 (relative) of each other along the contour, the
engine's inversion on paths of the doubled transform (every root twice) computes instead.  verified = true: both, the
latter returned with a warning where they differ by more than vtol.  Zero start; least squares detrending.

    hegy_t3_cdf(trend(0), [-3.3442])       # 0.05: the 5% critical value with seasonal dummies
"""
hegy_t3_cdf(D::Determ, xs::AbstractVector, c::Union{Real,AbstractVector{<:Real}} = 0.0; kw...) = t3_values("hegy_t3_cdf", D, xs, c, 1; kw...)
"""
    hegy_t3_pdf(D, xs, c = 0; c0 = 0, vol = nothing, verified = false) -> f (vector)

The density of the limit of t3 (`hegy_t3_cdf`).
"""
hegy_t3_pdf(D::Determ, xs::AbstractVector, c::Union{Real,AbstractVector{<:Real}} = 0.0; kw...) = t3_values("hegy_t3_pdf", D, xs, c, 2; kw...)

"""the outer contour of t3: the engine's (present rule) for the design at c with a zero start, or the volatility path"""
function t3_path(fn::String, D::Determ, c, vol)
    if vol !== nothing
        VM = vol_args(fn, D, c, :zero, 0.0, nothing, vol, :auto)
        VM === nothing || return vol_path(VM)
        c = c isa Real ? c : c[1]
    end
    c isa Real || throw(ArgumentError("$fn: " * EXT_CVEC_MSG))
    seas_cargs(fn, c, VOL_CMIN)
    CP = cpath_cached(startmodel(D, c, :zero, 0.0), c)
    isnan(CP.h) && throw(ArgumentError("$fn: the outer contour of this design is the vertical line (no hyperbola keeps its positive roots right of the origin); the seasonal statistics are not implemented there"))
    CP
end
"""the values of t3: F (which = 1) or f (2)"""
function t3_values(fn::String, D::Determ, xs::AbstractVector, c, which::Int; c0::Real = 0.0, vol = nothing, verified::Bool = false, vtol::Real = 1e-9)
    isfinite(c0) || throw(ArgumentError("$fn: c0 must be a finite real scalar"))
    CP = t3_path(fn, D, c, vol)
    c0f = Float64(c0)
    v = nonfinite(xs, which == 1) do x
        if verified
            v1 = t3_gap(CP) < T3_GAP ? contour_eval(t3_doubled(CP), x; c0 = c0f)[which] : t3_residues(CP, x, c0f)[which]
            v2 = contour_eval(t3_doubled(CP), x; c0 = c0f, n = 40, ntail = 64)[which]
            d = maximum(abs, v1 .- v2; init = 0.0)
            d <= vtol && return v1
            @warn "verified: the residues and the inversion on paths of the doubled transform differ by $(round(d, sigdigits = 2)) (above $vtol); the latter is returned" D.name c c0
            return v2
        end
        t3_gap(CP) < T3_GAP ? contour_eval(t3_doubled(CP), x; c0 = c0f)[which] : t3_residues(CP, x, c0f)[which]
    end
    which == 1 && cdf_range_check(fn, D, c isa Real ? c : minimum(c), :zero, 0.0, c0f, "x", xs, v)
    v
end
"""
    hegy_t3_quantile(D, p, c = 0; x0 = -2, tol = 1e-10, c0 = 0, vol = nothing) -> x

The p-quantile of the limit of t3 (`hegy_t3_cdf`; p may be a vector), by Newton's method with the density from x0, the step
limited to 1, then the safeguarded search.

    hegy_t3_quantile(trend(0), 0.05)       # -3.3442, the 5% critical value with seasonal dummies
"""
hegy_t3_quantile(D::Determ, p::Real, c::Union{Real,AbstractVector{<:Real}} = 0.0; kw...) = t3_quantile(D, p, c; kw...)::Float64
function t3_quantile(D::Determ, p::Real, c; x0::Real = -2.0, tol::Real = 1e-10, c0::Real = 0.0, vol = nothing)
    fn = "hegy_t3_quantile"
    0 < p < 1 || throw(ArgumentError("$fn: p must lie in (0, 1)"))
    search_args(fn, "x0", x0, tol)
    isfinite(c0) || throw(ArgumentError("$fn: c0 must be a finite real scalar"))
    CP = t3_path(fn, D, c, vol)
    gap = t3_gap(CP) < T3_GAP
    FF(y) = ((F, f) = gap ? contour_eval(t3_doubled(CP), [Float64(y)]; c0 = Float64(c0))[1:2] : t3_residues(CP, [Float64(y)], Float64(c0)); (F[1], f[1]))
    x = Float64(x0)
    for _ in 1:30
        F, f = FF(x)
        dx = clamp((F - p) / f, -1.0, 1.0)
        isfinite(dx) || break
        x -= dx
        abs(dx) < tol && return x
    end
    newton_safe(FF, p, x0, 1.0; tol)
end
hegy_t3_quantile(D::Determ, p::AbstractVector, c::Union{Real,AbstractVector{<:Real}} = 0.0; kw...) = [hegy_t3_quantile(D, pk, c; kw...) for pk in p]
