# =====================================================================================================================
# Seasonal unit roots at a harmonic frequency: the two t-ratios and the F statistic of the regression of Hylleberg,
# Engle, Granger and Yoo for a pair of complex unit roots
# =====================================================================================================================
# At a frequency strictly between 0 and pi (pi/2 for quarterly data, each pair of harmonic frequencies of monthly data) the
# limit process is complex: Z = W_a + i W_b with independent standard Brownian motions, dZ = c Z dr + dW, Z(0) = 0, and Z~
# its residual on the deterministic terms (the real trend functions of a design of the engine with complex coefficients:
# trend(0) for seasonal dummies, trend(1) for seasonal dummies and seasonal trends).  With
#
#     R = Re int Z~* dZ,   L = Im int Z~* dZ,   V = int |Z~|^2,
#
# the statistics are t3 = (R - c0 V)/sqrt(V) (the t-ratio of pi_3 in the quarterly numbering: the coefficient of y_{3,t-2},
# whose limit has the shape of a Dickey-Fuller distribution), t4 = L/sqrt(V) (pi_4, symmetric about zero at a real c) and
# F = (R^2 + L^2)/(2V) (the F statistic for pi_3 = pi_4 = 0).
#
# t3.  (R, V) is the sum of two independent copies of the lead paper's (U, V), so E exp(sR + tV) = e^{-s}/D(s,t) with the
# engine's determinant: its singularities in s are poles, and the inner integral of the inversion on paths is the sum of the
# residues at the roots s_k(t).  The density and the distribution function are single integrals over the engine's outer
# contour of closed-form functions (`t3_residues`), the kernels at the roots those of the engine (`ctr_moments`).
#
# t4 and F.  The transform of the three functionals, E exp(sR + uL + tV) = e^{-s}/D(s,u,t), by Girsanov's theorem: the tilt
# by exp(uL + t int |Z|^2) turns Z into the complex Ornstein-Uhlenbeck process with parameter -theta + i u, theta^2 = c^2 -
# 2t - u^2, and D = e^{c + theta} det{I - 2 (s A + u A_L + t B + gamma E11) S}, gamma = (c + theta)/2, with the engine's A,
# B, the area matrix A_L of the design (`seas_area`) and S the covariance of ell with the transfer functions at the two
# exponents theta -+ i u (`sigma2!`; at u = 0 the engine's Sigma).  The inversion on the lines Re u = 0, Re t = tau after
# the residues in s gives double integrals of closed-form functions of D(0, iy, tau + iw) and its roots in s, on a product
# rule in y and sqrt(w) built once per design and c (`SeasGrid`).

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

# ---------- the transform of (R, L, V) ----------
"""
    seas_area(P) -> A_L

The area matrix of the design: L - Im int Z* dZ = ell* A_L ell with A_L = (i/2)(C - C'), C the real matrix of m' G^{-1} w =
ell' C ell (m = int d Z, w = int d dZ = d(1) Z(1) - int d' Z - sum_i Delta_i Z(lambda_i)): C[z, 1] = Pm d(1), C[z, z] =
-Pm Phi_n, C[z, 1 + i] = -Pm Delta_i, with m' G^{-1} = z' Pm, Pm = Phi_d' G^{-1}, d = Phi_d g and d' = Phi_n g in the
orthonormal basis g of the prepared design (the same C whose symmetric part is in A: A = E11/2 - (C + C')/2).
"""
function seas_area(P::Prep)
    D = P.D; N = P.N; K = P.K; p = P.p; nj = P.nj; np = length(P.br) - 1
    C = zeros(N, N)
    if K > 0
        F = [hcat(D.C[j], dcoef(D.C[j]), P.g[j]) for j in 1:np]
        Gf = gram(F, P.br, D.loc)
        Phid = Gf[1:K, 2K+1:2K+p]; Phin = Gf[K+1:2K, 2K+1:2K+p]
        Pm = Phid' / P.G
        iz = (2 + nj):N
        C[iz, 1] .= Pm * dval(D, 1.0)
        C[iz, iz] .= -(Pm * Phin)
        for i in 1:nj
            lam = P.jl[i]; j = findfirst(x -> abs(x - lam) < 1e-12, P.br)
            dl = [evalpoly(lam - poff(D, j), view(D.C[j], :, k)) - evalpoly(lam - poff(D, j - 1), view(D.C[j-1], :, k)) for k in 1:K]
            C[iz, 1+i] .= -(Pm * dl)
        end
    end
    (im / 2) .* (C .- C')
end

"""the seasonal transform of a design: the prepared design P, its zero-start model, the area matrix A_L, and the static
matrices (sd: A, B, L, Dv; ALs: A_L) for N <= STATIC_MAX"""
struct SeasModel{SD,AS}
    P::Prep
    Mo::Model
    AL::Matrix{ComplexF64}
    sd::SD
    ALs::AS
end
function SeasModel(P::Prep)
    AL = seas_area(P)
    sd = sdata(P.A, P.B, P.L, P.Dv, zeros(P.N))
    ALs = sd === nothing ? nothing : SMatrix{P.N,P.N,ComplexF64}(AL)
    SeasModel(P, model(P), AL, sd, ALs)
end

"""the work space of `sigma2!`: the transfer functions at the two exponents"""
struct SeasWork
    e::Vector{ComplexF64}; a::Matrix{Float64}
    F1::Vector{ComplexF64}; F2::Vector{ComplexF64}
    Rb1::Matrix{ComplexF64}; Rb2::Matrix{ComplexF64}
    Ib1::Matrix{ComplexF64}; Ib2::Matrix{ComplexF64}
    z::Vector{Float64}; w::Vector{Float64}
    S::Matrix{ComplexF64}; M::Matrix{ComplexF64}
end
function SeasWork(P::Prep; nq::Int = 24)
    np = length(P.br) - 1; N = P.N; z, w = glrule(nq)
    SeasWork(zeros(ComplexF64, P.deg + 1), zeros(P.deg + 1, max(P.p, 1)), zeros(ComplexF64, N), zeros(ComplexF64, N),
             zeros(ComplexF64, N, np), zeros(ComplexF64, N, np), zeros(ComplexF64, max(P.p, 1), np), zeros(ComplexF64, max(P.p, 1), np),
             z, w, zeros(ComplexF64, N, N), zeros(ComplexF64, N, N))
end

"""the integrals of g over the pieces (Ib) and the transfer functions at the right ends of the pieces (Rb) at the exponent
theta, as in the engine's `sigma!`"""
function sigma2_ends!(Rb::Matrix{ComplexF64}, Ib::Matrix{ComplexF64}, e::Vector{ComplexF64}, a::Matrix{Float64}, P::Prep, θ::ComplexF64)
    p = P.p; nj = P.nj; np = length(P.br) - 1; D = P.deg; br = P.br
    @inbounds for m in 1:np
        p == 0 && break
        am = br[m]; Lm = br[m+1] - am
        ek!(e, θ * Lm, D)
        taylor!(a, P.g[m], am - P.off[m])
        for i in 1:p
            acc = zero(ComplexF64); Lk = Lm
            for k in 0:D
                acc += a[k+1, i] * Lk * e[k+1]; Lk *= Lm
            end
            Ib[i, m] = acc
        end
    end
    @inbounds for j in np:-1:1
        b = br[j+1]
        Rb[1, j] = exp(-θ * (1 - b))
        for i in 1:nj
            lam = P.jl[i]
            Rb[1+i, j] = lam >= b - 1e-14 ? exp(-θ * (lam - b)) : zero(ComplexF64)
        end
        if j == np
            for i in 1:p; Rb[1+nj+i, j] = 0; end
        else
            el = exp(-θ * (br[j+2] - br[j+1]))
            for i in 1:p; Rb[1+nj+i, j] = Ib[i, j+1] + el * Rb[1+nj+i, j+1]; end
        end
    end
    nothing
end
"""the transfer functions F at the point u = b - y of piece j (b its right end) at the exponent theta, with the Taylor
coefficients a of the basis at u given (`taylor!`, shared by the two exponents)"""
@inline function sigma2_point!(F::Vector{ComplexF64}, Rb::Matrix{ComplexF64}, e::Vector{ComplexF64}, a::Matrix{Float64}, P::Prep, j::Int,
                               y::Float64, θ::ComplexF64)
    p = P.p; nj = P.nj; D = P.deg
    ey = exp(-θ * y)
    @inbounds for i in 1:1+nj; F[i] = ey * Rb[i, j]; end
    if p > 0
        ek!(e, θ * y, D)
        @inbounds for i in 1:p
            acc = zero(ComplexF64); yk = y
            for k in 0:D
                acc += a[k+1, i] * yk * e[k+1]; yk *= y
            end
            F[1+nj+i] = acc + ey * Rb[1+nj+i, j]
        end
    end
    F
end

"""
    sigma2!(W, P, theta1, theta2) -> W.S

S = int_0^1 F(u; theta1) F(u; theta2)' du (bilinear, not symmetric), F the engine's transfer functions of ell at the
exponent theta (`sigma!`): with theta1 = theta - i u and theta2 = theta + i u, 2S is the covariance E ell ell* under the
complex Ornstein-Uhlenbeck process with parameter -theta + i u (for real theta and u), continued analytically.  The
integral over y = b - u by Gauss-Legendre panels graded from y = 0 at the scale of the larger exponent (as `sigma!`); the
entries among Z(1) and the Z(lambda_i) in closed form.
"""
function sigma2!(W::SeasWork, P::Prep, θ1::ComplexF64, θ2::ComplexF64)
    N = P.N; nj = P.nj; np = length(P.br) - 1; br = P.br
    S = W.S; fill!(S, 0)
    sigma2_ends!(W.Rb1, W.Ib1, W.e, W.a, P, θ1)
    sigma2_ends!(W.Rb2, W.Ib2, W.e, W.a, P, θ2)
    at = max(abs(θ1), abs(θ2)); rt = real(θ1 + θ2) / 2
    h0 = at > 0 ? 1 / at : Inf
    @inbounds for j in 1:np
        b = br[j+1]; Lj = b - br[j]
        y0 = 0.0
        while y0 < Lj * (1 - 1e-14)
            if at * Lj <= 2
                y1 = Lj
            else
                wdt = max(y0, h0)
                2 * rt * y0 < 80 && (wdt = min(wdt, 24h0))
                y1 = min(y0 + wdt, Lj)
                (Lj - y1) < 0.5wdt && (y1 = Lj)
            end
            for q in eachindex(W.z)
                y = y0 + (y1 - y0) * W.z[q]; wq = (y1 - y0) * W.w[q]
                P.p > 0 && taylor!(W.a, P.g[j], b - y - P.off[j])
                sigma2_point!(W.F1, W.Rb1, W.e, W.a, P, j, y, θ1)
                sigma2_point!(W.F2, W.Rb2, W.e, W.a, P, j, y, θ2)
                for jj in 1:N
                    fj = wq * W.F2[jj]
                    for ii in 1:N
                        S[ii, jj] += W.F1[ii] * fj
                    end
                end
            end
            y0 = y1
        end
    end
    # Z(1) and Z(lambda_i): int_0^m e^{-theta1 (la - u) - theta2 (lb - u)} du, m = min(la, lb)
    s12 = θ1 + θ2
    @inbounds for ia in 1:1+nj, ib in 1:1+nj
        la = ia == 1 ? 1.0 : P.jl[ia-1]; lb = ib == 1 ? 1.0 : P.jl[ib-1]; m = min(la, lb)
        S[ia, ib] = exp(-θ1 * (la - m) - θ2 * (lb - m)) * m * e0(s12 * m)
    end
    S
end

"""
    seas_parts(SM, c, u, t, W) -> (log D(0,u,t), mu)

The transform of (R, L, V) at one point (u, t): theta = sqrt(c^2 - 2t - u^2), S = `sigma2!` at theta -+ i u, and
D(s, u, t) = D(0, u, t) prod_k (1 - s mu_k) with log D(0, u, t) = c + theta + log det M0, M0 = I - 2 (u A_L + t B +
gamma E11) S, and mu_k the eigenvalues of 2 Dv L' S M0^{-1} L (`tilt_parts`).
"""
function seas_parts(SM::SeasModel{SD,AS}, c::Float64, u::ComplexF64, t::ComplexF64, W::SeasWork) where {SD,AS}
    P = SM.P
    θ = sqrt(complex(c * c - 2t - u * u))
    sigma2!(W, P, θ - im * u, θ + im * u)
    seas_tilt(SM, SM.sd, SM.ALs, W, c, u, t, θ)
end
function seas_tilt(SM::SeasModel, sd::SData{N,R}, ALs::SMatrix{N,N,ComplexF64}, W::SeasWork, c::Float64, u::ComplexF64, t::ComplexF64,
                   θ::ComplexF64) where {N,R}
    Q = MMatrix{N,N,ComplexF64}(u * ALs + t * sd.B)
    @inbounds Q[1, 1] += (c + θ) / 2
    tilt_parts(sd, SMatrix{N,N,ComplexF64}(W.S), SMatrix(Q), c + θ)
end
function seas_tilt(SM::SeasModel, ::Nothing, ::Nothing, W::SeasWork, c::Float64, u::ComplexF64, t::ComplexF64, θ::ComplexF64)
    P = SM.P
    Q = u .* SM.AL .+ t .* P.B; Q[1, 1] += (c + θ) / 2
    tilt_parts_dense(W.M, Q, W.S, P.L, P.Dv, c + θ)
end
"""
    seas_dval(SM, s, u, t; c = 0, sgn = 1) -> D(s, u, t)

The determinant of the transform of (R, L, V): E exp(sR + uL + tV) = e^{-s}/D(s, u, t) (sgn = -1 takes the other square
root theta, which does not change the value; at u = 0 the engine's `Dval`).
"""
function seas_dval(SM::SeasModel, s::Number, u::Number, t::Number; c::Real = 0.0, sgn::Int = 1)
    P = SM.P; W = SeasWork(P)
    c = Float64(c); s = ComplexF64(s); u = ComplexF64(u); t = ComplexF64(t)
    θ = sgn * sqrt(complex(c * c - 2t - u * u))
    sigma2!(W, P, θ - im * u, θ + im * u)
    Q = s .* P.A .+ u .* SM.AL .+ t .* P.B; Q[1, 1] += (c + θ) / 2
    exp(c + θ) * det(I - 2 .* (Q * W.S))
end

# ---------- the grid of the transform on the lines Re u = 0, Re t = tau ----------
"""
    SeasGrid

D(0, iy, tau + iw) and the reciprocal roots mu_k of D(., iy, tau + iw) on a product rule in y >= 0 and w >= 0
(Gauss-Legendre panels in y and in x = sqrt(w)); wy and ww the weights (ww with the Jacobian 2x); d0 the values
D(0, iy, tau + iw) (y down the rows, w across the columns), mu the r reciprocal roots at each point; edge the largest |1/d0| on
the outer edges of the grid (the part of the integrals the grid leaves out is of that order).
"""
struct SeasGrid
    c::Float64; tau::Float64; r::Int; rp::Int
    y::Vector{Float64}; wy::Vector{Float64}
    w::Vector{Float64}; ww::Vector{Float64}
    d0::Matrix{ComplexF64}
    mu::Array{ComplexF64,3}
    edge::Float64
end
function seas_panels(edges::Vector{Float64}, n::Int)
    z, w = glrule(n); x = Float64[]; wx = Float64[]
    for i in 1:length(edges)-1
        a = edges[i]; b = edges[i+1]
        append!(x, a .+ (b - a) .* z); append!(wx, (b - a) .* w)
    end
    x, wx
end
"""the rule of the grid: y up to ymax in panels of hy, x = sqrt(w) up to xmax in panels of hx1 up to x1 and hx2 beyond (the
kernels oscillate in w where w is of the order of tau), n Gauss-Legendre nodes per panel.  The prototype's panels in y
(the kernel e^{-q^2 y^2/(4t)} of t4 has the width 2 sqrt|t|/q in y: with panels of 1 the tail beyond q = 8 lost 1e-8 of
the second moment, and F at x = 60 1.3e-10, with seasonal dummies at the unit root) and twice its panels in x (within
1e-14 of the prototype's rule in t4, F and the moments of t4, at half the cost)."""
const SEAS_RULE = (ymax = 32.0, xmax = 32.0, hy = 0.5, hx1 = 0.2, x1 = 8.0, hx2 = 0.5, n = 10)
"""
    seas_grid(SM, c; tau = nothing, rule = SEAS_RULE) -> SeasGrid

The grid of the transform for the design of SM at c (tau = min(1, t_*/2), t_* the engine's first zero of D_V), on the
threads by columns of w.
"""
function seas_grid(SM::SeasModel, c::Float64; tau = nothing, rule = SEAS_RULE, closed::Int = -2)
    P = SM.P
    τ = tau === nothing ? min(1.0, tstar(SM.Mo, c)[1] / 2) : Float64(tau)
    y, wy = seas_panels(collect(range(0.0, rule.ymax; length = round(Int, rule.ymax / rule.hy) + 1)), rule.n)
    ex = [collect(range(0.0, rule.x1; length = round(Int, rule.x1 / rule.hx1) + 1));
          collect(range(rule.x1, rule.xmax; length = round(Int, (rule.xmax - rule.x1) / rule.hx2) + 1))[2:end]]
    x, wx = seas_panels(ex, rule.n)
    w = x .^ 2; ww = 2 .* x .* wx
    ny = length(y); nw = length(w); r = P.r
    d0 = Matrix{ComplexF64}(undef, ny, nw); mu = Array{ComplexF64,3}(undef, r, ny, nw)
    blas1f(dense_la(SM.Mo)) do
        Threads.@threads :dynamic for j in 1:nw
            W = SeasWork(P)
            for i in 1:ny
                u = complex(0.0, y[i]); t = complex(τ, w[j])
                if closed == -1
                    d0[i, j], mu[1, i, j] = seas_none(u, t, c)
                elseif closed == 0
                    d0[i, j], mu[1, i, j], mu[2, i, j] = seas_dummies(u, t)
                else
                    ld, μ = seas_parts(SM, c, u, t, W)
                    d0[i, j] = exp(ld)
                    for k in 1:r; mu[k, i, j] = μ[k]; end
                end
            end
        end
    end
    edge = max(maximum(i -> 1 / abs(d0[i, nw]), 1:ny), maximum(j -> 1 / abs(d0[ny, j]), 1:nw))
    SeasGrid(c, τ, r, P.rp, y, wy, w, ww, d0, mu, edge)
end
const SEASCACHE = Dict{UInt64,SeasGrid}()
const SEASORDER = UInt64[]
"""
    seas_none(u, t, c) -> (D(0, u, t), mu)

The transform without deterministic terms in closed form: D(s, u, t) = D_0(s, t + u^2/2), D_0(s, t) = e^c {cosh theta -
(s + c) sinh(theta)/theta}, theta^2 = c^2 - 2t, linear in s (one root)
"""
function seas_none(u::ComplexF64, t::ComplexF64, c::Float64)
    θ = sqrt(complex(c * c - 2t - u * u))
    sh = abs(θ) < 1e-4 ? 1 + θ * θ / 6 : sinh(θ) / θ
    d0 = exp(c) * (cosh(θ) - c * sh)
    d0, exp(c) * sh / d0
end
"""
    seas_dummies(u, t) -> (D(0, u, t), mu_+, mu_-)

Seasonal dummies at the unit root in closed form (Nabeya's formula in another form): D = sinc(a/2) sinc(b/2) {1 - (2t + s^2 +
u^2) Omega}, a, b = u -+ sqrt(u^2 + 2t), Omega = {g(a) - g(b)}/(a - b), g(x) = 1/x - cot(x/2)/2; even in s, the roots +-s
with s^2 = (1 - (2t + u^2) Omega)/Omega
"""
function seas_dummies(u::ComplexF64, t::ComplexF64)
    q = sqrt(u * u + 2t); a = u + q; b = u - q
    sc(x) = abs(x) < 1e-4 ? 1 - x * x / 24 : sin(x / 2) / (x / 2)
    g(x) = abs(x) < 1e-3 ? x / 12 + x^3 / 720 : 1 / x - cot(x / 2) / 2
    Om = abs(a - b) < 1e-6 ? complex(1 / 12) : (g(a) - g(b)) / (a - b)
    pre = sc(a) * sc(b); d0 = pre * (1 - (2t + u * u) * Om)
    s = sqrt(d0 / (pre * Om)); real(s) < 0 && (s = -s)
    d0, 1 / s, -1 / s
end
"""the closed form of the grid for the design D at c: -1 without deterministic terms (any c), 0 seasonal dummies at the
unit root, -2 none (the general transform)"""
function seas_closed(D::Determ, c::Float64)
    D.detrend == :onestep || return -2
    nreg(D) == 0 && return -1
    (c == 0 && length(D.br) == 2 && nreg(D) == 1 && size(D.C[1], 1) == 1 && D.C[1][1, 1] != 0) && return 0
    -2
end
"""the grid of the design at c, cached per (design, c, tau)"""
function seas_grid_cached(D::Determ, c::Float64; tau = nothing)
    key = hash((:seasgrid, dkey(D), c, tau === nothing ? -1.0 : Float64(tau)))
    hit = lock(CPLOCK) do; get(SEASCACHE, key, nothing); end
    hit === nothing || return hit
    G = seas_grid(SeasModel(prep_cached(D)), c; tau, closed = seas_closed(D, c))
    lock(CPLOCK) do; cache_put!(SEASCACHE, SEASORDER, key, G); end
    G
end

"""
    t4_both(G, q) -> (P(t4 <= q), density at q)

t4 = L/sqrt(V) by the inversion on the line Re u = 0, with the two signs of Im u combined (the transform is even in u at a
real c), whose kernel int_0^inf v^2 cos(q y v) e^{-t v^2} dv is elementary:

    f(q)           = (1 / (2 pi^{3/2})) Re int_0^inf int_0^inf t^{-3/2} (1 - q^2 y^2/(2t)) e^{-q^2 y^2/(4t)} / D(0, iy, t) dy dw,
    P(0 < t4 <= q) = (q / (2 pi^{3/2})) Re int_0^inf int_0^inf t^{-3/2} e^{-q^2 y^2/(4t)} / D(0, iy, t) dy dw,   t = tau + i w.
"""
function t4_both(G::SeasGrid, q::Float64)
    nw = length(G.w)
    accF = zeros(nw); accf = zeros(nw)
    Threads.@threads :static for j in 1:nw
        t = complex(G.tau, G.w[j]); t32 = 1 / (t * sqrt(t)); a = q * q / (4t)
        sF = zero(ComplexF64); sf = zero(ComplexF64)
        @inbounds for i in eachindex(G.y)
            y2 = G.y[i]^2
            e = G.wy[i] * exp(-a * y2) / G.d0[i, j]
            sF += e; sf += e * (1 - 2a * y2)
        end
        accF[j] = G.ww[j] * real(t32 * sF); accf[j] = G.ww[j] * real(t32 * sf)
    end
    k = 1 / (2 * pi^1.5)
    0.5 + q * k * sum(accF), k * sum(accf)
end

"""e^{-alpha} int_0^inf e^{-b xi - alpha xi^2} d xi with b = s - 2 alpha (the half-line integral of e^{-s xi - alpha (xi - 1)^2}),
by the scaled complementary error function; for Re z < 0 the reflection, with the two exponents combined"""
@inline function ghalf(s::ComplexF64, α::ComplexF64)
    sa = sqrt(α); z = (s - 2α) / (2sa); cc = sqrt(pi) / (2sa)
    real(z) >= 0 ? cc * exp(-α) * erfcx(z) : cc * (2 * exp(z * z - α) - exp(-α) * erfcx(-z))
end
"""int_0^inf e^{s xi - alpha (xi + 1)^2} d xi (the negative group: the half line xi < 0 of e^{-s xi - alpha (xi - 1)^2})"""
@inline function gneg(s::ComplexF64, α::ComplexF64)
    sa = sqrt(α); z = (2α - s) / (2sa)
    sqrt(pi) / (2sa) * exp(-α) * erfcx(z)
end

"""
    f_cdf1(G, x) -> P(F <= x)

F = (R^2 + L^2)/(2V).  With xi = R + 1, the density of (xi, L, V) on xi > 0 and on xi < 0 is the double inversion of the
residues of e^{-s xi}/D at the roots of the positive and of the negative group; the integral of e^{-s xi - u lambda - t v}
over the part of the region (xi - 1)^2 + lambda^2 <= 2 x v in each half-space is in closed form, which leaves

    P(F <= x) = (1/pi^2) Re int_0^inf int_0^inf (1/t) sqrt(2 pi x/t) e^{-x y^2/(2t)} { sum_+ w_k g(s_k) + sum_- w_k g(-s_k) } dy dw,

g(s) = int_0^inf e^{-s xi - (t/2x)(xi -+ 1)^2} d xi, the roots s_k = 1/mu_k of D(., iy, t), t = tau + i w, and the weights
w_k = -+1/D'(s_k) of the partial fractions 1/D = sum_+ w_k/(s_k - s) + sum_- w_k/(s - s_k).
"""
function f_cdf1(G::SeasGrid, x::Float64)
    x > 0 || return 0.0
    nw = length(G.w); r = G.r; rp = G.rp
    acc = zeros(nw)
    Threads.@threads :static for j in 1:nw
        t = complex(G.tau, G.w[j]); α = t / (2x)
        pre = sqrt(2pi * x / t) / t; bb = x / (2t)
        sj = zero(ComplexF64)
        @inbounds for i in eachindex(G.y)
            d0 = G.d0[i, j]
            g = zero(ComplexF64); np_ = 0
            for k in 1:r
                μk = G.mu[k, i, j]; sk = 1 / μk
                dD = -d0 * μk
                for l in 1:r
                    l == k || (dD *= 1 - sk * G.mu[l, i, j])
                end
                if real(sk) > 0
                    g -= ghalf(sk, α) / dD; np_ += 1
                else
                    g += gneg(sk, α) / dD
                end
            end
            np_ == rp || error("hegy_f_cdf: a root of D(., iy, tau + iw) has left its half-plane on the grid (y = $(G.y[i]), w = $(G.w[j]))")
            sj += G.wy[i] * exp(-bb * G.y[i]^2) * g
        end
        acc[j] = G.ww[j] * real(pre * sj)
    end
    sum(acc) / pi^2
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
const SEAS_T4_CMIN = -20.0
"""the degree p of the polynomial seasonal trend of the design D (-1: none, 0: seasonal dummies, 1: dummies and seasonal
trends), or an ArgumentError: t4 and F are implemented for these three"""
function seas_degree(fn::String, D::Determ)
    msg = "$fn: the second t-ratio and the F statistic are implemented for the polynomial seasonal trends trend(-1) (no seasonal terms), trend(0) (seasonal dummies) and trend(1) (seasonal dummies and seasonal trends), the one-step regression"
    D.detrend == :onestep || throw(ArgumentError(msg))
    K = nreg(D); K == 0 && return -1
    K <= 2 || throw(ArgumentError(msg))
    P = prep_cached(D)
    P.nj == 0 || throw(ArgumentError(msg))
    rs = [0.13, 0.37, 0.61, 0.89]
    X = reduce(vcat, [permutedims(dval(D, r)) for r in rs])
    Vd = [r^k for r in rs, k in 0:K-1]
    (rank(X) == K && rank(hcat(X, Vd)) == K) || throw(ArgumentError(msg))
    K - 1
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

"""
    hegy_t4_cdf(D, qs, c = 0; verified = false, vtol = 1e-9) -> F (vector)

The limit of P(t4 <= q) for the second t-ratio of the regression of Hylleberg, Engle, Granger and Yoo at a harmonic
frequency (pi/2 for quarterly data, where t4 is the t-ratio of pi_4, the coefficient of y_{3,t-1}; each pair of harmonic
frequencies of monthly data): t4 = L/sqrt(V), L = Im int Z~* dZ (see `hegy_t3_cdf`), symmetric about zero.  The design is a
polynomial seasonal trend: `trend(-1)` (no seasonal terms; then t4 is exactly standard normal, and the normal law is
returned), `trend(0)` (seasonal dummies) or `trend(1)` (seasonal dummies and seasonal trends); -20 <= c <= 0.

The inversion on the line Re u = 0 of the transform E exp(uL + tV) = 1/D(0, u, t), with the kernel int_0^inf v^2 cos(q y v)
e^{-t v^2} dv in closed form: a double integral over Im u and Im t on a grid of the transform built once per design and c
(cached; 20 to 35 ms with the closed forms without seasonal terms and with seasonal dummies at the unit root, 0.7 to 3.4 s
otherwise on four threads) and then a few ms a value.  verified = true: a second grid with another tau and a shorter range, a
warning where the two differ by more than vtol (the first is returned).

    hegy_t4_cdf(trend(0), [2.3193])        # 0.975
"""
hegy_t4_cdf(D::Determ, qs::AbstractVector, c::Real = 0.0; kw...) = t4_values("hegy_t4_cdf", D, qs, c, 1; kw...)
"""
    hegy_t4_pdf(D, qs, c = 0; verified = false) -> f (vector)

The density of the limit of t4 (`hegy_t4_cdf`).
"""
hegy_t4_pdf(D::Determ, qs::AbstractVector, c::Real = 0.0; kw...) = t4_values("hegy_t4_pdf", D, qs, c, 2; kw...)
"""the second grid of the checked computation: tau 1.5 times (at most 0.75 t_*), y and sqrt(w) up to 28"""
function seas_grid2(D::Determ, c::Float64, G::SeasGrid)
    ts = tstar(prep_cached(D) |> model, c)[1]
    seas_grid(SeasModel(prep_cached(D)), c; tau = min(1.5 * G.tau, 0.75 * ts), rule = merge(SEAS_RULE, (ymax = 28.0, xmax = 28.0)))
end
function t4_values(fn::String, D::Determ, qs::AbstractVector, c::Real, which::Int; verified::Bool = false, vtol::Real = 1e-9)
    p = seas_degree(fn, D)
    seas_cargs(fn, c, SEAS_T4_CMIN)
    x = Float64.(collect(qs))
    p == -1 && return which == 1 ? normcdf_.(x) : normpdf_.(x)
    G = seas_grid_cached(D, Float64(c))
    v = nonfinite(x, which == 1) do xf
        v1 = [t4_both(G, q)[which] for q in xf]
        if verified
            G2 = seas_grid2(D, Float64(c), G)
            v2 = [t4_both(G2, q)[which] for q in xf]
            d = maximum(abs, v1 .- v2; init = 0.0)
            d <= vtol || @warn "verified: hegy_t4: two grids differ by $(round(d, sigdigits = 2)) (above $vtol)" D.name c
        end
        v1
    end
    which == 1 && cdf_range_check(fn, D, c, :zero, 0.0, 0.0, "q", qs, v)
    v
end
"""
    hegy_t4_quantile(D, p, c = 0; x0 = nothing, tol = 1e-10) -> q

The p-quantile of the limit of t4 (`hegy_t4_cdf`; p may be a vector), by Newton's method with the density (from the normal
quantile unless x0 is given), the step limited to 1, then the safeguarded search; without seasonal terms the normal quantile.

    hegy_t4_quantile(trend(0), 0.975)      # 2.3193
"""
hegy_t4_quantile(D::Determ, p::Real, c::Real = 0.0; kw...) = t4_quantile(D, p, c; kw...)::Float64
function t4_quantile(D::Determ, p::Real, c::Real; x0 = nothing, tol::Real = 1e-10)
    fn = "hegy_t4_quantile"
    0 < p < 1 || throw(ArgumentError("$fn: p must lie in (0, 1)"))
    deg = seas_degree(fn, D)
    seas_cargs(fn, c, SEAS_T4_CMIN)
    z = sqrt(2) * erfinv(2 * Float64(p) - 1)
    deg == -1 && return z
    xs = x0 === nothing ? 1.2z : Float64(x0)
    search_args(fn, "x0", xs, tol)
    G = seas_grid_cached(D, Float64(c))
    FF(y) = t4_both(G, Float64(y))
    x = xs
    for _ in 1:30
        F, f = FF(x)
        dx = clamp((F - p) / f, -1.0, 1.0)
        isfinite(dx) || break
        x -= dx
        abs(dx) < tol && return x
    end
    newton_safe(FF, p, xs, 1.0; tol)
end
hegy_t4_quantile(D::Determ, p::AbstractVector, c::Real = 0.0; kw...) = [hegy_t4_quantile(D, pk, c; kw...) for pk in p]

"""
    hegy_f_cdf(D, xs, c = 0; verified = false, vtol = 1e-9) -> F (vector)

The limit of P(F <= x) for the F statistic of the regression of Hylleberg, Engle, Granger and Yoo for the pair of
coefficients at a harmonic frequency (pi_3 = pi_4 = 0 for quarterly data): F = (R^2 + L^2)/(2V) (see `hegy_t3_cdf`).  The
design is a polynomial seasonal trend, `trend(-1)`, `trend(0)` or `trend(1)` (as in `hegy_t4_cdf`); -20 <= c <= 0.

The inversion on the lines Re u = 0, Re t = tau after the residues in s at the roots of D(., u, t), with the integral over
the region (R + 1 - 1)^2 + L^2 <= 2 x V in closed form (the scaled complementary error function): a double integral on the
cached grid of the transform (`hegy_t4_cdf`).  There is no density.  verified = true: a second grid, a warning where they
differ by more than vtol.

    hegy_f_cdf(trend(0), [6.6455])         # 0.95
"""
function hegy_f_cdf(D::Determ, xs::AbstractVector, c::Real = 0.0; verified::Bool = false, vtol::Real = 1e-9)
    fn = "hegy_f_cdf"
    seas_degree(fn, D)
    seas_cargs(fn, c, SEAS_T4_CMIN)
    G = seas_grid_cached(D, Float64(c))
    x = Float64.(collect(xs))
    v = [isnan(xk) ? NaN : xk == Inf ? 1.0 : xk <= 0 ? 0.0 : f_cdf1(G, xk) for xk in x]
    if verified
        G2 = seas_grid2(D, Float64(c), G)
        v2 = [isnan(xk) ? NaN : xk == Inf ? 1.0 : xk <= 0 ? 0.0 : f_cdf1(G2, xk) for xk in x]
        fin = findall(isfinite, v)
        d = maximum(k -> abs(v[k] - v2[k]), fin; init = 0.0)
        d <= vtol || @warn "verified: hegy_f: two grids differ by $(round(d, sigdigits = 2)) (above $vtol)" D.name c
    end
    cdf_range_check(fn, D, c, :zero, 0.0, 0.0, "x", xs, v)
    v
end
"""
    hegy_f_quantile(D, p, c = 0; tol = 1e-9) -> x

The p-quantile of the limit of F (`hegy_f_cdf`; p may be a vector), by the rule of false position in its Illinois form on a
bracket found by doubling.

    hegy_f_quantile(trend(0), 0.95)        # 6.6455
"""
hegy_f_quantile(D::Determ, p::Real, c::Real = 0.0; kw...) = f_quantile(D, p, c; kw...)::Float64
function f_quantile(D::Determ, p::Real, c::Real; tol::Real = 1e-9)
    fn = "hegy_f_quantile"
    0 < p < 1 || throw(ArgumentError("$fn: p must lie in (0, 1)"))
    (tol > 0 && isfinite(tol)) || throw(ArgumentError("$fn: tol must be a positive finite number"))
    seas_degree(fn, D)
    seas_cargs(fn, c, SEAS_T4_CMIN)
    G = seas_grid_cached(D, Float64(c))
    Fx(x) = f_cdf1(G, x) - p
    a = 0.0; fa = -Float64(p); b = 1.0; fb = Fx(b)
    while fb < 0
        a = b; fa = fb; b *= 2; fb = Fx(b)
        b > 1e4 && error("$fn: no bracket below 1e4")
    end
    side = 0
    for _ in 1:200
        m = (a * fb - b * fa) / (fb - fa)
        fm = Fx(m)
        (abs(fm) < tol || b - a < tol) && return m
        if fm < 0
            a = m; fa = fm; side == -1 && (fb /= 2); side = -1
        else
            b = m; fb = fm; side == 1 && (fa /= 2); side = 1
        end
    end
    error("$fn: no convergence")
end
hegy_f_quantile(D::Determ, p::AbstractVector, c::Real = 0.0; kw...) = [hegy_f_quantile(D, pk, c; kw...) for pk in p]
