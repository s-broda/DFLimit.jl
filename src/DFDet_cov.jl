# =====================================================================================================================
# A normal component: T = rho tau_{c0} + sqrt(1 - rho^2) Z, the limit of the covariate-augmented t-ratio and of the t-ratio
# of the slope in a predictive regression with a regressor local to unity; and the dispatcher of the extension keywords
# =====================================================================================================================
# tau_{c0} = (U - c0 V)/sqrt(V) is the limit of the t-ratio of the design at the local null value c0 under c, and Z is
# standard normal and independent of (U, V).  Two applications share the object:
#
#   the covariate-augmented test of Hansen (1995): rho the long-run correlation of the innovation of the series and the
#     error of the augmented regression, R the ratio of their long-run standard deviations (Hansen's rho^2 and R^2;
#     R = |rho| when the regression is well specified), c0 = c (1 - 1/(rho R)), zero at the unit root (`cadf_c0`);
#   the predictive regression y_t = mu + beta x_{t-1} + u_t, x_t = phi x_{t-1} + v_t (the companion paper's section on
#     it): rho the correlation of the innovations, beta = b sigma_u/(sigma_v n), c0 = c - b/rho (`predictive_c0`).
#
# Two computations (method = :lines and :mixture):
#
#   the inversion on two lines with the normal component in the kernel: the double inversion of the moment generating
#     function M(s,t) of (U, V), as the engine has it, against the kernels of the lead paper at changed arguments,
#         P(T > x) = (2 pi i)^{-2} int int M(s,t) (2/s) E_1(x s/rho, p) dt ds,      (sigma > 0)
#         f_T(x)   = (2 pi i)^{-2} int int M(s,t) (2/rho) E_2(x s/rho, p) dt ds,
#         p = t + c0 s - (1 - rho^2) s^2/(2 rho^2),   E_m(b, p) = int_0^inf v^m e^{-b v - p v^2} dv;
#     for sigma < 0 the first integral is -P(T <= x).  The outer integral runs over the engine's outer contour in t (its
#     roots and log P), the inner one over the line Re s = sigma between the origin and the nearest root on the side
#     given by the sign of x (so that x sigma >= 0 and the kernels do not grow), whose two ends are turned into the right
#     half-plane (a deformation that leaves the integral unchanged; on the line e^{-a s} oscillates and the integrand
#     decays like a power only);
#   the normal average of the t-ratio's distribution function: P(T <= x) = int F((x - rb z)/rho) phi(z) dz, the density
#     -(1/rb) int F((x - rb z)/rho) z phi(z) dz (by parts, so that the density of tau, infinite at one point when c0 > 0
#     and the numerator has rank one, is not needed), the trapezoidal rule in z, graded toward the kink when c0 > 0.

normcdf_(x::Float64) = erfc(-x / sqrt(2)) / 2
normpdf_(x::Float64) = exp(-x * x / 2) / sqrt(2pi)

"""
    cadf_c0(c, rho, R = abs(rho)) -> c0

The local null value of the limit of the covariate-augmented Dickey-Fuller t-ratio of Hansen (1995): c (1 - 1/(rho R)),
with rho the long-run correlation of the innovation of the series and the error of the augmented regression and R the
ratio of their long-run standard deviations (R = |rho| for the well-specified regression; Hansen's rho^2 and R^2 are the
squares); zero at the unit root whatever rho and R.  c is the local-to-unity parameter of this package (phi = 1 + c/n;
Hansen's c with the opposite sign).  With `dist_cdf(D, x, c; rho, c0 = cadf_c0(c, rho, R))` and the other t-ratio
functions the limit of the statistic, rho tau_{c0} + sqrt(1 - rho^2) Z.

    cadf_c0(-5, sqrt(0.5))                 # 5.0: -c (1 - rho^2)/rho^2 when R = rho
"""
function cadf_c0(c::Real, rho::Real, R::Real = abs(rho))
    (isfinite(rho) && -1 <= rho <= 1 && rho != 0) || throw(ArgumentError("cadf_c0: rho must lie in [-1, 1] and not be 0"))
    (isfinite(R) && R > 0) || throw(ArgumentError("cadf_c0: R must be finite and positive"))
    isfinite(c) || throw(ArgumentError("cadf_c0: c must be finite"))
    c == 0 ? 0.0 : Float64(c) * (1 - 1 / (Float64(rho) * Float64(R)))
end
"""
    predictive_c0(c, rho, b = 0) -> c0

The local null value of the limit of the t-ratio of the slope beta in the predictive regression y_t = mu + beta x_{t-1} +
u_t, x_t = phi x_{t-1} + v_t, phi = 1 + c/n, with rho the correlation of the innovations (u_t, v_t) and the local
alternative beta = b sigma_u/(sigma_v n): c0 = c - b/rho (the companion paper's theorem on the predictive regression).
With `dist_cdf(trend(0), x, c; rho, c0 = predictive_c0(c, rho, b))` and the other t-ratio functions the limit of the
statistic (an intercept in the regression: the design trend(0)), rho tau_{c0} + sqrt(1 - rho^2) Z.

    ltratio_cdf(1.645, trend(0), 0; rho = -0.95, c0 = predictive_c0(0, -0.95))   # 0.5787: the right-tailed 5% test rejects 42%
"""
function predictive_c0(c::Real, rho::Real, b::Real = 0.0)
    (isfinite(rho) && -1 <= rho <= 1 && rho != 0) || throw(ArgumentError("predictive_c0: rho must lie in [-1, 1] and not be 0"))
    (isfinite(c) && isfinite(b)) || throw(ArgumentError("predictive_c0: c and b must be finite"))
    Float64(c) - Float64(b) / Float64(rho)
end

# ---------- the inversion on two lines with the normal component in the kernel ----------
"""h_m(z) = int_0^inf u^m exp(-u^2 - 2 z u) du, m = 0, 1, 2, for any complex z (`ctr_h` for Re z >= 0, and the full-line
Gaussian integral minus the reflected function for Re z < 0)"""
@inline function hfull(z::ComplexF64)
    real(z) >= 0 && return ctr_h(z)
    h0, h1, h2 = ctr_h(-z)
    g = sqrt(pi) * exp(z * z)
    g - h0, -z * g + h1, (z * z + 0.5) * g - h2
end

"""
    lines_rule!(S, W, sigma, rts, t, c0, vp, a, xmax, rho, n, theta)

Nodes S and weights W of (1/2 pi i) int ds along the inner path: the segment of the line Re s = sigma with |Im s| <= Y,
and from its two ends the rays s = sigma +- i Y + r e^{+- i theta}, r >= 0, into the right half-plane, along which e^{-a s}
decays while the real part of p = t + c0 s - vp s^2, vp = (1 - rho^2)/(2 rho^2), keeps growing; they stay above and below
every singular point.  The segment is cut into panels graded toward the levels of the singular points (the roots of
D(., t), the origin, the two zeros in s of p), each with its horizontal distance from the line as the smallest panel
(`graded_panels`), n Gauss-Legendre nodes per panel; Y exceeds every singular point's level by twice that distance.  A panel
is split where it is longer than 4/a (the phase of e^{-a s}) or where the phase x^2 s^2/(4 rho^2 p) of the kernel turns
by more than 4 radians at x = xmax.  The rays have panels that double from the scale at their start up to 4/a, as far as
r = 40/(a cos theta).
"""
function lines_rule!(S::Vector{ComplexF64}, W::Vector{ComplexF64}, σ::Float64, rts::AbstractVector{ComplexF64}, t::ComplexF64, c0::Float64,
                     vp::Float64, a::Float64, xmax::Float64, rho::Float64, n::Int, θ::Float64)
    empty!(S); empty!(W)
    z, w = glrule(n)
    foc = Tuple{Float64,Float64}[(0.0, abs(σ))]
    Y = 4 * abs(σ); dmin = 4 * abs(σ)
    for s in rts
        d = abs(real(s) - σ)
        push!(foc, (imag(s), d)); Y = max(Y, abs(imag(s)) + 2d); dmin = min(dmin, 2d)
    end
    if vp > 0
        sq = sqrt(complex(c0 * c0) + 4vp * t)
        for s in ((c0 + sq) / (2vp), (c0 - sq) / (2vp))
            d = abs(real(s) - σ)
            push!(foc, (imag(s), d)); Y = max(Y, abs(imag(s)) + 2d); dmin = min(dmin, 2d)
        end
    end
    cap = 4 / a
    ph = xmax^2 / (4 * rho^2)
    phase(u::Float64) = (s = complex(σ, u); ph * imag(s * s / (t + c0 * s - vp * s * s)))
    for (a0, b0) in graded_panels(-Y, Y, foc)
        mid = (a0 + b0) / 2
        m = max(ceil(Int, (b0 - a0) / cap), ceil(Int, max(abs(phase(b0) - phase(mid)), abs(phase(mid) - phase(a0))) / 4))
        m = clamp(m, 1, 4096)
        for i in 0:m-1
            a1 = a0 + (b0 - a0) * i / m; b1 = a0 + (b0 - a0) * (i + 1) / m
            for q in 1:n
                push!(S, complex(σ, a1 + (b1 - a1) * z[q])); push!(W, (b1 - a1) * w[q] / (2pi))
            end
        end
    end
    rmax = 40 / (a * cos(θ))
    for dir in (1, -1)
        e = cis(dir * θ); s0 = complex(σ, dir * Y)
        r0 = 0.0; h = min(max(dmin, 1e-3), cap)
        while r0 < rmax
            for q in 1:n
                push!(S, s0 + (r0 + h * z[q]) * e); push!(W, dir * e * h * w[q] / (2pi * im))
            end
            r0 += h; h = min(2h, cap)
        end
    end
    S, W
end

"""the abscissa of the inner line on the side (+1: between the origin and the positive roots; -1: between the negative roots
and the origin) at a node with the roots rts: the fraction frac of the way to the nearest root of the side, but no farther
than the point at which the real part of p = t + c0 s - vp s^2 at Im s = 0 has fallen to half the real part of t (the
kernels need Re p > 0), and on the left no farther than -6 frac/a (the factor e^{-a s} of the moment generating function
grows to the left, while the kernel falls like |s|^{-3} at best)"""
function lines_abscissa(side::Int, rts::AbstractVector{ComplexF64}, rp::Int, tr::Float64, c0::Float64, vp::Float64, frac::Float64, a::Float64)
    r = length(rts)
    cap = vp > 0 ? (c0 + side * sqrt(c0^2 + 2vp * tr)) / (2vp) : (side * c0 < 0 ? -tr / (2c0) : side * Inf)
    if side == 1
        mp = Inf
        for k in 1:rp; mp = min(mp, real(rts[k])); end
        mp > 0 || error("the inversion on two lines: a root of the positive group has left the right half-plane at a node of the outer contour")
        return min(frac * mp, cap)
    end
    mm = -Inf
    for k in rp+1:r; mm = max(mm, real(rts[k])); end
    mm < 0 || error("the inversion on two lines: a root of the negative group has left the left half-plane at a node of the outer contour")
    max(frac * mm, cap, -6 * frac / a)
end

"""the terms of node k of the outer contour CP for the points x (the sides ipos, x >= 0, and ineg): Im{t'(u_k) I(t_k)} with
the weight of the outer rule, I the inner integral of the distribution function's kernel (GFk) and of the density's (Gfk)"""
function lines_node!(GFk::AbstractVector{Float64}, Gfk::AbstractVector{Float64}, CP::CPath, k::Int, x::Vector{Float64}, ipos::Vector{Int},
                     ineg::Vector{Int}, xmp::Float64, xmn::Float64, rho::Float64, c0::Float64, vp::Float64, a::Float64, rp::Int, n::Int,
                     frac::Float64, θ::Float64, S::Vector{ComplexF64}, W::Vector{ComplexF64}, IF::Vector{ComplexF64}, If::Vector{ComplexF64})
    t = CP.t[k]; dt = CP.dt[k]; wk = CP.wt[k]; rts = view(CP.R, :, k); lP = CP.lP[k]
    for side in (1, -1)
        idx = side == 1 ? ipos : ineg; xm = side == 1 ? xmp : xmn
        isempty(idx) && continue
        σ = lines_abscissa(side, rts, rp, real(t), c0, vp, frac, a)
        lines_rule!(S, W, σ, rts, t, c0, vp, a, xm, rho, n, θ)
        fill!(IF, 0); fill!(If, 0)
        @inbounds for j in eachindex(S)
            s = S[j]
            m = exp(logMtilde(rts, rp, lP, s) - a * s) * W[j]
            p = t + c0 * s - vp * s * s
            (real(p) > 0 || θ < pi / 4) || error("the inversion on two lines: the real part of the kernel's second argument is not positive on the inner path (node $k)")
            sp = sqrt(p); ip = 1 / p; ip32 = ip / sp
            hz = s / (2 * rho * sp)
            mF = 2m / s * ip; mf = 2m / rho * ip32
            for i in eachindex(idx)
                _, h1, h2 = hfull(x[idx[i]] * hz)
                IF[i] += mF * h1; If[i] += mf * h2
            end
        end
        @inbounds for i in eachindex(idx)
            GFk[idx[i]] = wk * imag(dt * IF[i]); Gfk[idx[i]] = wk * imag(dt * If[i])
        end
    end
    nothing
end

"""
    lines_both(CP, x, rho, c0; n = 24, tol = 1e-17, frac = 0.5, theta = 0.3 pi) -> (F, f, nodes)

P(T <= x) and the density of T = rho tau_{c0} + sqrt(1 - rho^2) Z, 0 < rho < 1, at the finite points x by the inversion
on two lines with the normal component in the kernel, on the outer contour CP of the design with its start (zero or
stationary).  At each node t_k the inner integral over s is taken on the line Re s = sigma_k (`lines_abscissa`), to the
right of the origin for x >= 0 (the kernel's integral is P(T > x)) and to the left for x < 0 (-P(T <= x)), its ends turned
into the right half-plane at the angle theta (`lines_rule!`).  The outer rule is the engine's (the weights of CP, the
terms Im{t'(u_k) I(t_k)}), the nodes in chunks of 32 on the threads until a chunk whose terms all stay below tol.  The
moment generating function enters through CP alone: e^{a s} M(s,t) = P(t) prod (s_k(t) - s)^{-1/2} prod (s - s_k(t))^{-1/2}
(`logMtilde`).
"""
function lines_both(CP::CPath, x::Vector{Float64}, rho::Float64, c0::Float64; n::Int = 24, tol::Float64 = 1e-17, frac::Float64 = 0.5,
                    theta::Float64 = 0.3pi)
    nx = length(x)
    vp = (1 - rho^2) / (2 * rho^2)
    Mo = CP.Mo; a = Mo.a; rp = Mo.rp
    K1 = length(CP.t)
    GF = zeros(K1, nx); Gf = zeros(K1, nx)
    plus = x .>= 0
    ipos = findall(plus); ineg = findall(.!plus)
    xmp = isempty(ipos) ? 0.0 : maximum(abs, view(x, ipos)); xmn = isempty(ineg) ? 0.0 : maximum(abs, view(x, ineg))
    done = 0
    while done < K1
        k1 = done + 1; k2 = min(done + 32, K1)
        Threads.@threads :dynamic for k in k1:k2
            S = ComplexF64[]; W = ComplexF64[]; IF = zeros(ComplexF64, nx); If = zeros(ComplexF64, nx)
            lines_node!(view(GF, k, :), view(Gf, k, :), CP, k, x, ipos, ineg, xmp, xmn, rho, c0, vp, a, rp, n, frac, theta, S, W, IF, If)
        end
        done = k2
        max(maximum(abs, view(GF, k1:k2, :)), maximum(abs, view(Gf, k1:k2, :))) < tol && break
    end
    F = zeros(nx); f = zeros(nx)
    for ix in 1:nx
        sF = 0.0; sf = 0.0
        for k in 1:done; sF += GF[k, ix]; sf += Gf[k, ix]; end
        F[ix] = plus[ix] ? 1 - sF : -sF
        f[ix] = sf
    end
    F, f, done
end

# ---------- the normal average ----------
"""
    mixture_both(D, x, c, rho, c0, start, gamma0; h = nothing, zmax = 8.5, n = 12, verified = false) -> (F, f)

P(rho tau_{c0} + rb Z <= x) and its density for 0 < rho < 1 at the finite points x, from the t-ratio's distribution
function (`dist_cdf` of the design with its start, method = :auto, one call for all points; with verified = true its
checked computation): P(T <= x) = int F((x - rb z)/rho) phi(z) dz, f_T(x) = -(1/rb) int F((x - rb z)/rho) z phi(z) dz.
For c0 <= 0 the trapezoidal rule with step h in z (default 0.25/(1 + rb/rho), half that for a numerator of rank one, the
design without deterministic terms with a zero or fixed start, where the rule converges more slowly: at rho^2 = 0.2 the
longer step was off by 6e-9) on |z| <= zmax.  For c0 > 0 the distribution function of tau is not smooth at q_* = -2
sqrt(c0 a), where the parabola c0 v^2 + q v + a has a double root (a the shift of the model): Gauss-Legendre panels with
n nodes, graded toward the point z at which (x - rb z)/rho = q_* from a smallest panel of 1e-8 and no longer than 4h.  F is
taken as 0 or 1 outside the bulk of tau (`tau_bulk`), where it is so to rounding: with a small rho the points (x - rb z)/rho
reach far into the tails, where the engine is not asked (at x = -424 its value was -1.2e-8, which its range check refuses).
"""
function mixture_both(D::Determ, x::Vector{Float64}, c::Float64, rho::Float64, c0::Float64, start::Symbol, gamma0::Float64;
                      h = nothing, zmax::Float64 = 8.5, n::Int = 12, verified::Bool = false, method::Symbol = :auto)
    rb = sqrt(1 - rho^2)
    Mo = startmodel(D, c, start, gamma0)
    hz = h === nothing ? (Mo.r == 1 ? 0.125 : 0.25) / (1 + rb / rho) : Float64(h)
    zs = Float64[]; ws = Float64[]; first = Int[]
    qstar = c0 > 0 ? -2 * sqrt(c0 * Mo.a) : NaN
    gz, gw = glrule(n)
    for xv in x
        push!(first, length(zs) + 1)
        zstar = (xv - rho * qstar) / rb
        if c0 > 0 && abs(zstar) < zmax
            for (a0, b0) in graded_panels(-zmax, zmax, [(zstar, 1e-8)])
                m = max(1, ceil(Int, (b0 - a0) / (4hz)))
                for i in 0:m-1
                    a1 = a0 + (b0 - a0) * i / m; b1 = a0 + (b0 - a0) * (i + 1) / m
                    for j in 1:n; push!(zs, a1 + (b1 - a1) * gz[j]); push!(ws, (b1 - a1) * gw[j]); end
                end
            end
        else
            nz = ceil(Int, zmax / hz)
            for k in -nz:nz; push!(zs, k * hz); push!(ws, hz); end
        end
    end
    push!(first, length(zs) + 1)
    q = similar(zs)
    for i in eachindex(x), k in first[i]:first[i+1]-1; q[k] = (x[i] - rb * zs[k]) / rho; end
    lo, hi = tau_bulk(c, c0, start, gamma0)
    Fq = [qk < lo ? 0.0 : 1.0 for qk in q]
    ib = findall(qk -> lo <= qk <= hi, q)
    if !isempty(ib)
        Fq[ib] = dist_cdf(D, q[ib], c; c0, start, gamma0, verified, method)
        # (where the line integral, the engine's default for these designs at c0 = 0, returns a value that is not finite,
        # the inversion on paths: case DK at c = 0 gives NaN at x = -7.8125 and below x = 1e-150 on the positive side)
        nf = [ib[k] for k in eachindex(ib) if !isfinite(Fq[ib[k]])]
        isempty(nf) || (Fq[nf] = dist_cdf(D, q[nf], c; c0, start, gamma0, verified, method = :paths))
    end
    F = zeros(length(x)); f = zeros(length(x))
    for i in eachindex(x), k in first[i]:first[i+1]-1
        w = ws[k] * normpdf_(zs[k]) * clamp(Fq[k], 0.0, 1.0)
        F[i] += w; f[i] -= w * zs[k] / rb
    end
    F, f
end

"""
    tau_bulk(c, c0, start, gamma0) -> (lo, hi)

A generous bound on the bulk of tau_{c0} = tau_c + (c - c0) sqrt(V): m -+ 40 s with m = (c - c0) sqrt(E int J^2) and
s = 1 + |c - c0| sqrt(E int J^2), E int J^2 of the process with its start (an upper bound on E V; the tails of the t-ratio's
limit decay like those of a normal law, so that F is 0 or 1 to rounding beyond).
"""
function tau_bulk(c::Float64, c0::Float64, start::Symbol, gamma0::Real)
    ej = abs(c) < 1e-6 ? 0.5 : (expm1(2c) - 2c) / (4c^2)
    start == :stationary && (ej = -1 / (2c))
    start == :fixed && (ej += gamma0^2 * (abs(c) < 1e-6 ? 1.0 : expm1(2c) / (2c)))
    r = sqrt(ej); m = (c - c0) * r; s = 1 + abs(c - c0) * r
    m - 40s, m + 40s
end

# ---------- the user-level computation with rho ----------
const COV_METHOD_MSG = "with rho != 1 the method must be :auto, :lines (the inversion on two lines with the normal component in the kernel) or :mixture (the normal average of the t-ratio's distribution function)"
"""the checks of the arguments with rho (the covariate-augmented t-ratio and the predictive regression)"""
function cov_args(fn::String, D::Determ, start::Symbol, gls, rho::Float64, method::Symbol)
    (isfinite(rho) && -1 <= rho <= 1) || throw(ArgumentError("$fn: rho must lie in [-1, 1]"))
    method in (:auto, :lines, :mixture) || throw(ArgumentError("$fn: " * COV_METHOD_MSG))
    gls === nothing || throw(ArgumentError("$fn: rho != 1 (a normal component: the covariate-augmented t-ratio, the predictive regression) is implemented for least squares detrending; GLS detrending (gls) is not supported with it"))
    D.detrend == :onestep || throw(ArgumentError("$fn: rho != 1 is implemented for the one-step regression; the statistic that detrends first (detrend_first) is not supported with it"))
    nothing
end
"""the computation of method = :auto: the lines for 0.1 <= rho^2 <= 0.99 with a zero or stationary start at c <= 0, the
mixture otherwise (near |rho| = 1 the kernel of the lines oscillates, its phase reaching x^2/(2 (1 - rho^2)), and the
mixture is the cheaper; the lines do not cover a fixed start or the vertical line above the unit root)"""
function cov_method(fn::String, method::Symbol, rho::Float64, c::Float64, fixed::Bool)
    if method === :lines
        fixed && throw(ArgumentError("$fn: method = :lines (the inversion on two lines) needs a zero or a stationary start; method = :mixture covers a fixed start"))
        c > 0 && throw(ArgumentError("$fn: method = :lines (the inversion on two lines) is implemented below and at the unit root (c <= 0); method = :mixture covers c > 0"))
        return :lines
    end
    method === :mixture && return :mixture
    (fixed || c > 0 || !(0.1 <= rho^2 <= 0.99)) ? :mixture : :lines
end
"""F and f of T = rho tau_{c0} + rb Z, 0 < rho < 1, at the finite points x by the method m"""
function cov_both(D::Determ, x::Vector{Float64}, c::Float64, rho::Float64, c0::Float64, start::Symbol, gamma0::Float64, m::Symbol; verified::Bool = false, h = nothing)
    if m === :lines
        CP = cpath_cached(startmodel(D, c, start, gamma0), c)
        F, f, _ = lines_both(CP, x, rho, c0)
        return F, f
    end
    mixture_both(D, x, c, rho, c0, start, gamma0; verified, h)
end
"""the checked computation with rho: the other route as well; where the two differ by more than vtol the mixture with half
its step on the checked distribution function of tau, with a warning"""
function cov_verified(D::Determ, x::Vector{Float64}, c::Float64, rho::Float64, c0::Float64, start::Symbol, gamma0::Float64, m::Symbol, vtol::Float64)
    F1, f1 = cov_both(D, x, c, rho, c0, start, gamma0, m)
    rb = sqrt(1 - rho^2)
    hh = (startmodel(D, c, start, gamma0).r == 1 ? 0.0625 : 0.125) / (1 + rb / rho)          # (half the default step)
    F2, f2 = m === :lines ? cov_both(D, x, c, rho, c0, start, gamma0, :mixture) :
                            cov_both(D, x, c, rho, c0, start, gamma0, :mixture; verified = true, h = hh)
    d = max(maximum(abs, F1 .- F2; init = 0.0), maximum(abs, f1 .- f2; init = 0.0))
    d <= vtol && return F1, f1
    F3, f3 = m === :lines ? cov_both(D, x, c, rho, c0, start, gamma0, :mixture; verified = true, h = hh) : (F2, f2)
    @warn "verified: the two computations of the normal component differ by $(round(d, sigdigits = 2)) (above $vtol); the normal average on the checked distribution function is returned" D.name c c0 rho
    F3, f3
end

"""
    cov_eval(fn, D, xs, c, which, start, gamma0, gls, verified, vtol, rho; method, c0) -> values

The values of `dist_cdf` (which = 1) or `dist_pdf` (2) with rho != 1: rho = 0 the standard normal law; rho in (0, 1) the
method of `cov_method`; rho in [-1, 0) the reflection P(T <= x) = 1 - P(|rho| tau + rb Z <= -x), f_T(x) = f(-x) (rho = -1:
the t-ratio's own functions at -x).
"""
function cov_eval(fn::String, D::Determ, xs::AbstractVector, c::Float64, which::Int, start::Symbol, gamma0::Real, gls, verified::Bool, vtol::Float64,
                  rho::Float64; method::Symbol = :auto, kw...)
    cov_args(fn, D, start, gls, rho, method)
    other = [string(k) for k in keys(kw) if k !== :c0]
    isempty(other) || throw(ArgumentError("$fn: with rho != 1 the options $(join(other, ", ")) do not apply"))
    c0 = Float64(get(kw, :c0, 0.0))
    isfinite(c0) || throw(ArgumentError("$fn: c0 must be a finite real scalar"))
    x = Float64.(collect(xs))
    rho == 0 && return which == 1 ? normcdf_.(x) : normpdf_.(x)
    if rho < 0
        v = rho == -1 ? (which == 1 ? dist_cdf(D, -x, c; c0, start, gamma0, verified) : dist_pdf(D, -x, c; c0, start, gamma0, verified)) :
                        cov_eval(fn, D, -x, c, which, start, gamma0, gls, verified, vtol, -rho; method, c0)
        return which == 1 ? [isnan(y) ? NaN : 1 - y for y in v] : v
    end
    fixed = start == :fixed && gamma0 != 0
    m = cov_method(fn, method, rho, c, fixed)
    nonfinite(x, which == 1) do xf
        F, f = verified ? cov_verified(D, xf, c, rho, c0, start, Float64(gamma0), m, vtol) : cov_both(D, xf, c, rho, c0, start, Float64(gamma0), m)
        which == 1 ? F : f
    end
end

"""the p-quantile with rho != 1 (Newton's method with the density, the step limited to 1, then `newton_safe`; rho = 0 the
normal quantile; rho < 0 by the reflection)"""
cov_quantile(D::Determ, p::Real, c::Float64, start::Symbol, gamma0::Real, gls, rho::Float64; kw...) =
    cov_quantile_(D, p, c, start, gamma0, gls, rho; kw...)::Float64
function cov_quantile_(D::Determ, p::Real, c::Float64, start::Symbol, gamma0::Real, gls, rho::Float64; x0::Real, tol::Real, method::Symbol = :auto, kw...)
    fn = "dist_quantile"
    cov_args(fn, D, start, gls, rho, method)
    other = [string(k) for k in keys(kw) if k !== :c0]
    isempty(other) || throw(ArgumentError("$fn: with rho != 1 the options $(join(other, ", ")) do not apply"))
    c0 = Float64(get(kw, :c0, 0.0))
    isfinite(c0) || throw(ArgumentError("$fn: c0 must be a finite real scalar"))
    rho == 0 && return sqrt(2) * erfinv(2 * Float64(p) - 1)
    if rho < 0
        rho == -1 && return -dist_quantile(D, 1 - p, c; x0 = -x0, tol, c0, start, gamma0)
        return -cov_quantile_(D, 1 - p, c, start, gamma0, gls, -rho; x0 = -x0, tol, method, c0)
    end
    m = cov_method(fn, method, rho, c, start == :fixed && gamma0 != 0)
    FF(y) = ((F, f) = cov_both(D, [Float64(y)], c, rho, c0, start, Float64(gamma0), m); (F[1], f[1]))
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

# ---------- the dispatcher of the extension keywords of dist_cdf, dist_pdf, dist_quantile ----------
const EXT_BOTH_MSG = "rho and vol together are not implemented (the limit with a normal component under a volatility path is not derived)"
const EXT_CVEC_MSG = "c is a vector (one value per regime): that needs a volatility path, vol = volpath(breaks, omega)"
"""
    ext_eval(D, xs, c, which, start, gamma0, gls, phi, hfac, verified, vtol; method, rule, vol, rho, kw...) -> values

`dist_cdf` (which = 1) and `dist_pdf` (2) with the keywords of the extensions: a volatility path (vol, and c a vector of
one value per regime; one regime with a scalar c is the engine itself) or a normal component (rho != 1).
"""
function ext_eval(D::Determ, xs::AbstractVector, c, which::Int, start::Symbol, gamma0::Real, gls, phi, hfac, verified::Bool, vtol::Float64;
                  method::Symbol = :auto, rule::Symbol = :auto, vol = nothing, rho::Real = 1.0, kw...)
    fn = which == 1 ? "dist_cdf" : "dist_pdf"
    if vol === nothing
        c isa Real || throw(ArgumentError("$fn: " * EXT_CVEC_MSG))
        (phi === nothing && hfac === nothing && rule === :auto) || throw(ArgumentError("$fn: with rho != 1 the options phi, hfac and rule of the inversion on paths do not apply"))
        return cov_eval(fn, D, xs, Float64(c), which, start, gamma0, gls, verified, vtol, Float64(rho); method, kw...)
    end
    rho == 1 || throw(ArgumentError("$fn: " * EXT_BOTH_MSG))
    VM = vol_args(fn, D, c, start, gamma0, gls, vol, method)
    VM === nothing && return dist_eval_(D, xs, c isa Real ? c : c[1], which, start, gamma0, gls, phi, hfac, verified, vtol; method, rule, kw...)
    vol_eval(D, xs, c, which, vol, phi, hfac, verified, vtol, VM; method, rule, kw...)
end
"""`dist_quantile` with the keywords of the extensions (as `ext_eval`); nothing for one regime with a scalar c (the
engine's search computes)"""
function ext_quantile(D::Determ, p::Real, c; x0::Real, tol::Real, start::Symbol, gamma0::Real, gls, phi, hfac, method::Symbol, rule::Symbol,
                      vol = nothing, rho::Real = 1.0, kw...)
    fn = "dist_quantile"
    if vol === nothing
        c isa Real || throw(ArgumentError("$fn: " * EXT_CVEC_MSG))
        (phi === nothing && hfac === nothing && rule === :auto) || throw(ArgumentError("$fn: with rho != 1 the options phi, hfac and rule of the inversion on paths do not apply"))
        return cov_quantile(D, p, Float64(c), start, gamma0, gls, Float64(rho); x0, tol, method, kw...)
    end
    rho == 1 || throw(ArgumentError("$fn: " * EXT_BOTH_MSG))
    VM = vol_args(fn, D, c, start, gamma0, gls, vol, method)
    VM === nothing && return nothing
    vol_quantile(D, p, VM; x0, tol, phi, hfac, rule, kw...)
end
