# =====================================================================================================================
# Volatility and persistence regimes: the Dickey-Fuller t-ratio when the standard deviation omega(r) of the innovations
# and the local-to-unity parameter c(r) are piecewise constant between known dates
# =====================================================================================================================
# The limit is the functional (U - c0 V)/sqrt(V) of the general engine, with
#
#     V = int J^2 - m' G^{-1} m,   U = (J(1)^2 - 1)/2 - m' G^{-1} w,   m = int d J,   w = int d dJ,
#
# of the process dJ = c(r) J dr + omega(r) dW, J(0) = 0, omega scaled so that int omega^2 = 1 (the variance estimate of
# the regression converges to the average of omega^2, and the statistic does not depend on the scale of omega).
#
# The transform.  On the pieces of the design refined at the regime boundaries (piece j: length L_j, c_j, omega_j) let
# theta_j = sqrt(c_j^2 - 2 t omega_j^2) and g_j = (c_j + theta_j)/(2 omega_j^2).  By Girsanov's theorem, piece by piece,
# the tilt by exp(t int J^2) turns J into the process dJ = -theta(r) J dr + omega(r) dW, further tilted by
# exp{g_last J(1)^2 + sum_j (g_j - g_{j+1}) J(b_j)^2} (b_j the right end of piece j), and
#
#     M(s,t) = E exp(sU + tV) = e^{-s/2} D(s,t)^{-1/2},
#     D(s,t) = exp{sum_j (c_j + theta_j) L_j} det{I - 2 (s A + t B + Gamma) Sigma},
#
# with ell = (J(1), J at the jumps of d, z = int g J, J at the regime boundaries that are not jumps of d), A and B the
# matrices of the design padded with zeros, Gamma the diagonal matrix of the tilts and Sigma the covariance of ell under
# the tilted process.  This is the form of the general engine with a diagonal tilt in place of gamma E11; everything
# downstream of log D_V(t) and the reciprocal roots mu_k(t) is the engine's code: the outer contour and the continuation of
# the roots (`cpath_` on a `VolModel`, a `TransformModel`), the inner integrals (`contour_eval`), the rules of one
# evaluation, the checked computation, the quantile search.

"""
    VolPath

Volatility regimes: boundaries 0 = lam[1] < lam[2] < ... < lam[end] = 1, and on regime i (lam[i] < r <= lam[i+1]) the
standard deviation om[i] of the innovations, scaled so that the integral of omega^2 is one.  Built by `volpath`; the
keyword `vol` of `dist_cdf`, `dist_pdf`, `dist_quantile`, `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf` and `hegy_t3_cdf`.
"""
struct VolPath
    lam::Vector{Float64}
    om::Vector{Float64}
end

"""
    volpath(breaks, omega = ones(length(breaks) + 1)) -> VolPath

Regimes of the innovation standard deviation (and, through a vector c, of the local-to-unity parameter) between known
dates: `breaks` the interior boundaries, strictly increasing in (0, 1) (fractions of the sample), `omega` the standard
deviations on the `length(breaks) + 1` regimes, of any positive scale (they are scaled so that the integral of omega^2 is
one; only the ratios matter).  `volpath(Float64[])` is one regime, the engine itself.

    volpath([0.3], [1.0, 3.0])              # the standard deviation triples at 30% of the sample
    volpath([0.5])                          # two regimes with a constant variance (changing persistence: c a vector)
"""
function volpath(breaks::AbstractVector{<:Real}, omega::AbstractVector{<:Real} = ones(length(breaks) + 1))
    br = Float64.(collect(breaks))
    all(b -> isfinite(b) && 0 < b < 1, br) || throw(ArgumentError("volpath: the break dates must lie in (0, 1)"))
    all(k -> br[k] < br[k+1], 1:length(br)-1) || throw(ArgumentError("volpath: the break dates must be strictly increasing"))
    length(omega) == length(br) + 1 || throw(ArgumentError("volpath: one standard deviation omega per regime: $(length(br) + 1) for $(length(br)) break dates, $(length(omega)) given"))
    om = Float64.(collect(omega))
    all(w -> isfinite(w) && w > 0, om) || throw(ArgumentError("volpath: the standard deviations omega must be finite and positive"))
    lam = [0.0; br; 1.0]
    om ./= sqrt(sum(om[i]^2 * (lam[i+1] - lam[i]) for i in eachindex(om)))
    VolPath(lam, om)
end
nregimes(V::VolPath) = length(V.om)
function Base.show(io::IO, V::VolPath)
    print(io, "volpath(", V.lam[2:end-1], ", ", round.(V.om; sigdigits = 6), ")")
end

"""the local-to-unity parameter on each regime of V: a number for all of them, or a vector with one value per regime"""
function regime_c(fn::String, V::VolPath, c)
    k = nregimes(V)
    cr = c isa Real ? fill(Float64(c), k) : Float64.(collect(c))
    length(cr) == k || throw(ArgumentError("$fn: c must be a number or a vector with one value per regime of the volatility path ($k), not $(length(cr))"))
    all(isfinite, cr) || throw(ArgumentError("$fn: c must be finite"))
    cr
end

# ---------- the model: the design on the common partition of its own pieces and the regimes ----------
"""the design D with its pieces split at the points lam (the regressors themselves unchanged)"""
function vol_refine(D::Determ, lam::Vector{Float64})
    br = copy(D.br)
    for x in lam
        any(b -> abs(b - x) < 1e-12, br) || push!(br, x)
    end
    sort!(br)
    C = Matrix{Float64}[]
    for j in 1:length(br)-1
        jo = piece(D, (br[j] + br[j+1]) / 2)
        if D.loc
            a = similar(D.C[jo]); taylor!(a, D.C[jo], br[j] - D.br[jo]); push!(C, a)
        else
            push!(C, copy(D.C[jo]))
        end
    end
    Determ(br, C, D.name, D.loc, D.detrend)
end
"""the regime of V that contains r (lam[i] < r <= lam[i+1])"""
vol_regime(V::VolPath, r::Real) = (i = 1; while i < nregimes(V) && r > V.lam[i+1]; i += 1; end; i)

"""
    VolModel

The transform model of a design with volatility regimes: the design refined at the regime boundaries (D, prepared P,
its zero-start model Mi for the inner integrals), np pieces of lengths L with omega^2 and c on each, the extended vector
ell = (J(1), J at the jumps of d, z, J at the regime boundaries that are not jumps of d) of dimension Ne, the position
bpos[j] in ell of J at the boundary between pieces j and j + 1 (0: not in ell), the padded matrices A, B and the
eigenvectors Lx of A (eigenvalues Dv), the start (:zero, g0 = 0: the fields the outer contour reads), and the static
matrices sd (`SData`, nothing for Ne > STATIC_MAX).
"""
struct VolModel{SD} <: TransformModel
    D::Determ
    P::Prep
    Mi::Model
    vol::VolPath
    cr::Vector{Float64}
    np::Int
    L::Vector{Float64}
    om2::Vector{Float64}
    c::Vector{Float64}
    Ne::Int
    bpos::Vector{Int}
    A::Matrix{Float64}; B::Matrix{Float64}
    Lx::Matrix{Float64}; Dv::Vector{Float64}
    r::Int; rp::Int; rm::Int
    start::Symbol
    g0::Float64
    sd::SD
end
inner_model(VM::VolModel) = VM.Mi
dense_la(VM::VolModel) = VM.sd === nothing

function vol_model_(D0::Determ, V::VolPath, cr::Vector{Float64})
    D = vol_refine(D0, V.lam)
    P = prep_cached(D)
    np = length(P.br) - 1
    L = diff(P.br)
    reg = [vol_regime(V, (P.br[j] + P.br[j+1]) / 2) for j in 1:np]
    om2 = V.om[reg] .^ 2; c = cr[reg]
    bpos = zeros(Int, max(np - 1, 0)); Ne = P.N
    for j in 1:np-1
        b = P.br[j+1]
        i = findfirst(x -> abs(x - b) < 1e-12, P.jl)
        if i !== nothing
            bpos[j] = 1 + i
        elseif om2[j] != om2[j+1] || c[j] != c[j+1]
            Ne += 1; bpos[j] = Ne
        end
    end
    A = zeros(Ne, Ne); B = zeros(Ne, Ne); Lx = zeros(Ne, P.r)
    A[1:P.N, 1:P.N] .= P.A; B[1:P.N, 1:P.N] .= P.B; Lx[1:P.N, :] .= P.L
    sd = sdata(A, B, Lx, copy(P.Dv), zeros(Ne))
    VolModel(D, P, model(P), V, cr, np, L, om2, c, Ne, bpos, A, B, Lx, copy(P.Dv), P.r, P.rp, P.rm, :zero, 0.0, sd)
end
"""the volatility model of the design D, the regimes V and the c of each regime, cached"""
function vol_model(D::Determ, V::VolPath, cr::Vector{Float64})
    key = hash((:volmodel, dkey(D), V.lam, V.om, cr))
    hit = lock(CPLOCK) do; get(VOLCACHE, key, nothing); end
    hit === nothing || return hit::VolModel
    VM = blas1(() -> vol_model_(D, V, cr))
    lock(CPLOCK) do; cache_put!(VOLCACHE, VOLORDER, key, VM); end
    VM
end
const VOLCACHE = Dict{UInt64,Any}()
const VOLORDER = UInt64[]

# ---------- the covariance Sigma of ell under the tilted process ----------
"""the work space of the volatility transform at one t (`MWork` of a `VolModel`)"""
struct VolWork
    e::Vector{ComplexF64}
    a::Matrix{Float64}
    F::Vector{ComplexF64}
    Rb::Matrix{ComplexF64}
    Ib::Matrix{ComplexF64}
    z::Vector{Float64}; w::Vector{Float64}
    S::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
    th::Vector{ComplexF64}
    v::Vector{ComplexF64}
    gam::Vector{ComplexF64}
end
function MWork(VM::VolModel; nq::Int = 24)
    P = VM.P; z, w = glrule(nq)
    VolWork(zeros(ComplexF64, P.deg + 1), zeros(P.deg + 1, max(P.p, 1)), zeros(ComplexF64, VM.Ne), zeros(ComplexF64, VM.Ne, VM.np),
            zeros(ComplexF64, max(P.p, 1), VM.np), z, w, zeros(ComplexF64, VM.Ne, VM.Ne), zeros(ComplexF64, VM.Ne, VM.Ne),
            zeros(ComplexF64, VM.np), zeros(ComplexF64, VM.np), zeros(ComplexF64, VM.Ne))
end

"""
    sigma_vol!(W, VM) -> W.S

The covariance (bilinear in theta, no conjugation) of ell under dJ = -theta_j J dr + omega_j dW on piece j, J(0) = 0, with
the theta_j in W.th: Sigma = int_0^1 F(u) F(u)' du with the transfer functions F(u) = omega(u) x {E(u,1) for J(1);
1{u < b} E(u,b) for J(b); int_u^1 g(r) E(u,r) dr for z}, E(u,r) = exp(-int_u^r theta).  On piece j = (a, b] and with y =
b - u, F(u)/omega_j = (the part of z inside the piece, sum_k g^{(k)}(u)/k! y^{k+1} e_k(theta_j y)) + e^{-theta_j y} R_j,
R_j the same functionals of the path started at b, by a backward recursion over the pieces.  The integral over y by
Gauss-Legendre panels graded from y = 0, as the engine's `sigma!` (which this is with one regime).  The entries among J(1)
and the J(b) are then replaced by their closed forms, from the variances at the right ends of the pieces.
"""
function sigma_vol!(W::VolWork, VM::VolModel)
    P = VM.P; N = VM.Ne; p = P.p; np = VM.np; D = P.deg; br = P.br; θ = W.th
    S = W.S; fill!(S, 0)
    e = W.e; a = W.a; F = W.F; Rb = W.Rb; Ib = W.Ib
    iz0 = 1 + P.nj
    @inbounds for m in 1:np
        p == 0 && break
        Lm = VM.L[m]
        ek!(e, θ[m] * Lm, D)
        taylor!(a, P.g[m], br[m] - P.off[m])
        for i in 1:p
            acc = zero(ComplexF64); Lk = Lm
            for k in 0:D
                acc += a[k+1, i] * Lk * e[k+1]; Lk *= Lm
            end
            Ib[i, m] = acc
        end
    end
    fill!(Rb, 0)
    Rb[1, np] = 1
    @inbounds for j in np-1:-1:1
        el = exp(-θ[j+1] * VM.L[j+1])
        for i in 1:N; Rb[i, j] = el * Rb[i, j+1]; end
        for i in 1:p; Rb[iz0+i, j] += Ib[i, j+1]; end
        pos = VM.bpos[j]
        pos > 0 && (Rb[pos, j] = 1)
    end
    @inbounds for j in 1:np
        θj = θ[j]; at = abs(θj); rt = real(θj); h0 = at > 0 ? 1 / at : Inf
        b = br[j+1]; Lj = VM.L[j]; w2 = VM.om2[j]
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
                y = y0 + (y1 - y0) * W.z[q]; wq = w2 * (y1 - y0) * W.w[q]
                ey = exp(-θj * y)
                for i in 1:N; F[i] = ey * Rb[i, j]; end
                if p > 0
                    ek!(e, θj * y, D)
                    taylor!(a, P.g[j], b - y - P.off[j])
                    for i in 1:p
                        acc = zero(ComplexF64); yk = y
                        for k in 0:D
                            acc += a[k+1, i] * yk * e[k+1]; yk *= y
                        end
                        F[iz0+i] += acc
                    end
                end
                for jj in 1:N
                    fj = wq * F[jj]
                    for ii in 1:jj
                        S[ii, jj] += fj * F[ii]
                    end
                end
            end
            y0 = y1
        end
    end
    # the variances of the path at the right ends of the pieces, and from them the entries among J(1) and the J(b)
    v = W.v; acc = zero(ComplexF64)
    @inbounds for j in 1:np
        acc = VM.om2[j] * VM.L[j] * e0(2 * θ[j] * VM.L[j]) + exp(-2 * θ[j] * VM.L[j]) * acc
        v[j] = acc
    end
    @inbounds for j in 1:np
        pj = j == np ? 1 : VM.bpos[j]; pj == 0 && continue
        S[pj, pj] = v[j]
        dec = one(ComplexF64)
        for k in j+1:np
            dec *= exp(-θ[k] * VM.L[k])
            pk = k == np ? 1 : VM.bpos[k]; pk == 0 && continue
            S[min(pj, pk), max(pj, pk)] = v[j] * dec
        end
    end
    @inbounds for jj in 1:N, ii in 1:jj-1
        S[jj, ii] = S[ii, jj]
    end
    S
end

"""the theta_j at t (principal square roots, times sgn[j] if given), Sigma into W.S and the diagonal tilt Gamma into W.gam
(g_last at J(1), g_j - g_{j+1} at the boundary between pieces j and j + 1, g_j = (c_j + theta_j)/(2 omega_j^2)); returns
the exponent sum_j (c_j + theta_j) L_j"""
function vol_tilt!(W::VolWork, VM::VolModel, t::ComplexF64, sgn = nothing)
    θ = W.th
    @inbounds for j in 1:VM.np
        θ[j] = sqrt(complex(VM.c[j]^2 - 2t * VM.om2[j]))
        sgn === nothing || (θ[j] *= sgn[j])
    end
    sigma_vol!(W, VM)
    g(j) = (VM.c[j] + θ[j]) / (2 * VM.om2[j])
    fill!(W.gam, 0)
    W.gam[1] += g(VM.np)
    ex = zero(ComplexF64)
    @inbounds for j in 1:VM.np
        ex += (VM.c[j] + θ[j]) * VM.L[j]
        if j < VM.np
            pos = VM.bpos[j]
            pos > 0 && (W.gam[pos] += g(j) - g(j + 1))
        end
    end
    ex
end

"""
    tilt_parts(sd, S, Q0, ex) -> (log D_V, mu)

The transform at one point from the process's covariance S of ell and its tilt Q0 (Girsanov's form; Q0 = t B + Gamma, or
u A_L + t B + gamma E11 with the area): log D_V = ex + log det M0, M0 = I - 2 Q0 S (the imaginary part not reduced), and
the eigenvalues mu_k of 2 Dv L' S M0^{-1} L, the reciprocal roots in s of det{I - 2 (s A + Q0) S} = det M0 prod_k (1 -
s mu_k).  Static arrays (`SData`); the shared builder of the volatility and the seasonal transforms.
"""
@inline function tilt_parts(sd::SData{N,R}, S::SMatrix{N,N,ComplexF64}, Q0::SMatrix{N,N,ComplexF64}, ex::ComplexF64) where {N,R}
    M = MMatrix{N,N,ComplexF64}(-2 * (Q0 * S))
    @inbounds for i in 1:N; M[i, i] += 1; end
    F = slu(SMatrix(M))
    X = ssolve(F, SMatrix{N,R,ComplexF64}(sd.L))
    H = 2 .* (sd.Dv .* (transpose(sd.L) * (S * X)))
    ex + slogdet(F), eigs_s(H)
end
"""`tilt_parts` with dense matrices (designs beyond STATIC_MAX): Q0 and S dense, Lx and Dv"""
function tilt_parts_dense(M0::Matrix{ComplexF64}, Q0::AbstractMatrix, S::Matrix{ComplexF64}, Lx::Matrix{Float64}, Dv::Vector{Float64}, ex::ComplexF64)
    mul!(M0, Q0, S, -2, false)
    for i in axes(M0, 1); M0[i, i] += 1; end
    F = lu(M0)
    X = F \ complex.(Lx)
    H = 2 .* (Diagonal(Dv) * (transpose(Lx) * (S * X)))
    ex + logdet(F), eigvals(H)
end

"""
    mgfparts(VM::VolModel, c, t, W) -> (theta_last, log D_V, mu, nothing)

The volatility transform at one point t (c is not used: the model carries the c of its regimes), in the form of the
engine's `mgfparts`: log D_V(t) = sum_j (c_j + theta_j) L_j + log det{I - 2 (t B + Gamma) Sigma} and the reciprocal roots.
"""
function mgfparts(VM::VolModel{SD}, c::Float64, t::ComplexF64, W::VolWork) where {SD}
    ex = vol_tilt!(W, VM, t)
    ld, μ = vol_parts(VM, VM.sd, W, t, ex)
    W.th[VM.np], ld, μ, nothing
end
function vol_parts(VM::VolModel, sd::SData{N,R}, W::VolWork, t::ComplexF64, ex::ComplexF64) where {N,R}
    S = SMatrix{N,N,ComplexF64}(W.S)
    Q = MMatrix{N,N,ComplexF64}(t * sd.B)
    @inbounds for i in 1:N; Q[i, i] += W.gam[i]; end
    tilt_parts(sd, S, SMatrix(Q), ex)
end
function vol_parts(VM::VolModel, ::Nothing, W::VolWork, t::ComplexF64, ex::ComplexF64)
    Q = t .* VM.B
    for i in 1:VM.Ne; Q[i, i] += W.gam[i]; end
    tilt_parts_dense(W.M, Q, W.S, VM.Lx, VM.Dv, ex)
end

"""
    vol_dval(VM, s, t; sgn = nothing) -> D(s,t)

The volatility transform's determinant exp{sum_j (c_j + theta_j) L_j} det{I - 2 (s A + t B + Gamma) Sigma}, so that M(s,t) =
e^{-s/2} D^{-1/2} (a check: sgn, a vector of +-1 per piece multiplying the theta_j, does not change the value).
"""
function vol_dval(VM::VolModel, s::Number, t::Number; sgn = nothing)
    W = MWork(VM)
    ex = vol_tilt!(W, VM, ComplexF64(t), sgn)
    Q = ComplexF64(s) .* VM.A .+ ComplexF64(t) .* VM.B
    for i in 1:VM.Ne; Q[i, i] += W.gam[i]; end
    M = I - 2 .* (Q * W.S)
    exp(ex) * det(M)
end

# ---------- t_*: the first positive zero of D_V ----------
"""
    vol_kernel(VM) -> (r, u) -> Cov(J(r), J(u))

The covariance of the path with the regimes, zero start: e^{C(r) + C(u)} int_0^{min(r,u)} omega^2 e^{-2 C(v)} dv with C(r) =
int_0^r c, piece by piece in closed form, each term with its exponents combined so that nothing overflows far from the unit
root (the Nystrom estimate of t_*, `eigtop` with kern).
"""
function vol_kernel(VM::VolModel)
    br = VM.P.br; np = VM.np
    Cb = zeros(np + 1)
    for j in 1:np; Cb[j+1] = Cb[j] + VM.c[j] * VM.L[j]; end
    Cat(r) = (j = 1; while j < np && r > br[j+1]; j += 1; end; Cb[j] + VM.c[j] * (r - br[j]))
    function K(r::Float64, u::Float64)
        m = min(r, u); E = Cat(r) + Cat(u); acc = 0.0
        for j in 1:np
            a = br[j]; a >= m && break
            b = min(br[j+1], m); L = b - a; cj = VM.c[j]
            Ca = Cb[j]; Cbb = Ca + cj * L
            # int_a^b e^{E - 2 C(v)} dv, C(v) = C(a) + c_j (v - a)
            acc += VM.om2[j] * (abs(2cj * L) < 1e-3 ? exp(E - 2Ca) * L * real(e0(complex(2cj * L))) :
                                (exp(E - 2Cbb) - exp(E - 2Ca)) / (-2cj))
        end
        acc
    end
    K
end

"""
    tstar(VM::VolModel, c) -> (t_*, multiplicity)

The first positive zero of D_V(t) for the volatility model: the Nystrom estimate of the largest eigenvalue of the
denominator's operator (`eigtop` with the covariance of the regimes, `vol_kernel`); D_V sampled on 81 points of +-2% around
the estimate, bisection at the first change of sign, the golden section for a double zero (lambda_1 = lambda_2, a
symmetric design at a constant variance); where the estimate is off by more than 2%, D_V scanned from zero in steps of a
thousandth of the estimate for its first change of sign (or a double zero), up to eight times the estimate.
"""
tstar(VM::VolModel, c::Real) = blas1(() -> vol_tstar(VM))
function vol_tstar(VM::VolModel)
    W = MWork(VM)
    f(t) = real(exp(mgfparts(VM, 0.0, complex(Float64(t)), W)[2]))
    l1, l2 = eigtop(VM.P, minimum(VM.c); m = NYSTROM_M[], kern = vol_kernel(VM))
    te = 1 / (2l1)
    function bisect(a, b)
        for _ in 1:100
            mid = (a + b) / 2
            f(mid) > 0 ? (a = mid) : (b = mid)
            b - a < 1e-13 * b && break
        end
        (a + b) / 2
    end
    function golden(a, b)
        g = (sqrt(5) - 1) / 2
        x1 = b - g * (b - a); x2 = a + g * (b - a); f1 = f(x1); f2 = f(x2)
        for _ in 1:80
            if f1 < f2
                b = x2; x2 = x1; f2 = f1; x1 = b - g * (b - a); f1 = f(x1)
            else
                a = x1; x1 = x2; f1 = f2; x2 = a + g * (b - a); f2 = f(x2)
            end
        end
        (a + b) / 2
    end
    grid = range(0.98te, 1.02te; length = 81)
    vals = f.(grid)
    k = findfirst(<=(0), vals)
    k !== nothing && k > 1 && return bisect(grid[k-1], grid[k]), abs(l1 - l2) < 1e-4 * l1 ? 2 : 1
    if k === nothing
        j = argmin(vals)
        (1 < j < length(grid) && vals[j] < 1e-6 * maximum(vals)) && return golden(grid[j-1], grid[j+1]), 2
    end
    # (the estimate is off: the scan from zero)
    h = te / 1000; fprev = f(h); fmax = abs(fprev); fpp = Inf; tprev = h
    for j in 2:8000
        tj = j * h; fj = f(tj)
        fj <= 0 && return bisect(tprev, tj), 1
        (fprev < fpp && fprev <= fj && fprev < 1e-6 * fmax) && return golden(tprev - h, tj), 2
        fmax = max(fmax, fj); fpp = fprev; fprev = fj; tprev = tj
    end
    error("tstar: no zero of D_V of the volatility model below $(8te) (the Nystrom estimate of t_* is $te)")
end

# ---------- the outer contour ----------
"""the outer contour of the volatility model at the angle phi and the step factor hfac (`cpath` on the model: the engine's
hyperbola, steepened while a root group leaves its side of the imaginary axis, the vertical line beyond), cached"""
function vol_path(VM::VolModel; phi = nothing, hfac::Real = 1.0)
    key = hash((:volpath, dkey(VM.D), VM.vol.lam, VM.vol.om, VM.cr, phi === nothing ? -1.0 : Float64(phi), Float64(hfac)))
    hit = lock(CPLOCK) do; get(VOLCACHE, key, nothing); end
    hit === nothing || return hit::CPath
    CP = cpath(VM, minimum(VM.cr); phi, hfac)
    lock(CPLOCK) do; cache_put!(VOLCACHE, VOLORDER, key, CP); end
    CP
end
"""the outer contour of the rule r with the step factor hf (as `rule_path`: :fine keeps the angle of the present rule's
hyperbola with half its step)"""
function vol_rule_path(VM::VolModel, phi, hf::Float64, r::Symbol)
    if phi === nothing && r in (:fine, :finestep)
        CP1 = vol_path(VM)
        isnan(CP1.h) && return CP1
        phi = CP1.phi
    end
    vol_path(VM; phi, hfac = hf)
end
"""the outer contours of the variants of the checked computation (`verify_eval`, as `variant_path`) for the volatility model"""
function vol_variant(VM::VolModel, CP::CPath, level::Int)
    isnan(CP.h) && return linecpath(VM, CP.c, CP.tstar, CP.tau; nq = level == 2 ? 48 : 32)
    level == 1 && return vol_path(VM; phi = min(CP.phi + 7.5 * pi / 180, 88 * pi / 180), hfac = 0.8)
    level == 0 && return vol_path(VM; phi = CP.phi, hfac = 0.8)
    vol_path(VM; phi = CP.phi, hfac = 0.5)
end

# ---------- the t-ratio: distribution function, density, quantiles ----------
const VOL_START_MSG = "a volatility path (vol) is implemented for the zero start only (J(0) = 0); a fixed or a stationary start is not supported with it"
const VOL_GLS_MSG = "a volatility path (vol) is implemented for least squares detrending only; GLS detrending (gls) is not supported with it"
const VOL_FIRST_MSG = "a volatility path (vol) is implemented for the one-step regression only; the statistic that detrends first (detrend_first) is not supported with it"
"""the c below which the volatility extension refuses (the general engine's range below the unit root)"""
const VOL_CMIN = -480.0
"""
    vol_args(fn, D, c, start, gamma0, gls, vol, method) -> VolModel or nothing

The checks of the arguments of the volatility extension and its model; nothing for one regime with a scalar c (the engine
itself computes).
"""
function vol_args(fn::String, D::Determ, c, start::Symbol, gamma0::Real, gls, vol, method::Symbol)
    vol isa VolPath || throw(ArgumentError("$fn: vol must be a volatility path, volpath(breaks, omega)"))
    cr = regime_c(fn, vol, c)
    (start == :zero || (start == :fixed && gamma0 == 0)) || throw(ArgumentError("$fn: " * VOL_START_MSG))
    gls === nothing || throw(ArgumentError("$fn: " * VOL_GLS_MSG))
    D.detrend == :onestep || throw(ArgumentError("$fn: " * VOL_FIRST_MSG))
    method in (:auto, :paths) || throw(ArgumentError("$fn: with a volatility path (vol) the method is the inversion on paths: method must be :auto or :paths (the line integral is not implemented for the volatility transform)"))
    all(<=(0), cr) || throw(ArgumentError("$fn: with a volatility path (vol) every c must be at most 0 (above the unit root the extension is not implemented); c = $(c)"))
    all(>=(VOL_CMIN), cr) || throw(ArgumentError("$fn: with a volatility path (vol) every c must be at least $VOL_CMIN (the range of the general engine); c = $(c)"))
    (nregimes(vol) == 1) && return nothing
    vol_model(D, vol, cr)
end

"""the values of `dist_cdf` (which = 1) or `dist_pdf` (2) with a volatility path: one evaluation of the inversion on paths on
the volatility model's outer contour by the rule of `path_region`, or the checked computation (verified = true)"""
function vol_eval(D::Determ, xs::AbstractVector, c, which::Int, vol, phi, hfac, verified::Bool, vtol::Float64, VM::VolModel;
                  method::Symbol = :auto, rule::Symbol = :auto, kw...)
    c0 = Float64(get(kw, :c0, 0.0))
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))
    check_rule(rule)
    cm = minimum(VM.cr)
    hf, n, nt, rl = path_rule(rule, verified, VM.Mi, cm, c0, hfac, kw)
    CP = vol_rule_path(VM, phi, hf, rl)
    nonfinite(xs, which == 1) do x
        v = verified ? verify_eval((CPv, o) -> contour_eval(CPv, x; o..., kw...)[which], CP, vtol; variant = (C, l) -> vol_variant(VM, C, l)) :
                       contour_eval(CP, x; kw..., n, ntail = nt)[which]
        which == 2 && tangency!(v, CP.Mo, x, c0)
        v
    end
end

"""the p-quantile with a volatility path (as `dist_quantile`: Newton's method with the density, the step limited to 1, then
`newton_safe`)"""
vol_quantile(D::Determ, p::Real, VM::VolModel; kw...) = vol_quantile_(D, p, VM; kw...)::Float64
function vol_quantile_(D::Determ, p::Real, VM::VolModel; x0::Real, tol::Real, phi, hfac, rule::Symbol, kw...)
    check_rule(rule)
    c0 = Float64(get(kw, :c0, 0.0))
    isfinite(c0) || throw(ArgumentError("c0 must be a finite real scalar"))
    hf, n, nt, rl = path_rule(rule, false, VM.Mi, minimum(VM.cr), c0, hfac, kw)
    CP = vol_rule_path(VM, phi, hf, rl)
    FF(y) = (r = contour_eval(CP, [y]; kw..., n, ntail = nt); (r[1][1], r[2][1]))
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
