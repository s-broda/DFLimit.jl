# Reference values of the regime variants (volatility and persistence regimes with GLS detrending, with the statistic
# that detrends first, and with a normal start) from the prototype that computed them for the lead paper before the
# package did: paper/VolDF.jl of the repository df-deterministics (the file as of its commit 2ad4a34 of 27 November 2026;
# it was removed from that repository once the package covered the variants, so this script runs against that commit).
# Writes test/ref/regime_ref.txt, which test/extensions/regimes.jl compares with the package.  Run from the repository
# root with the path of a checkout of df-deterministics that still holds the prototype:
#   ~/bin/jlcap --project=dev dev/regime_ref.jl ~/Documents/github/df-deterministics
# Each line starts with a tag; values are printed with 17 digits.
#   design id trend p | design id perron M lam
#   vol id nb b_1..b_nb om_1..om_nb+1           the break dates and the standard deviations as given (unscaled)
# and, with <case> = did vid nc c_1..c_nc kind par (kind = ls | gls, par cbar | first | normal, par the variance of the
# start | stat, the stationary start of the first regime, par its variance):
#   dval <case> s t D            the determinant D(s,t) at complex s, t (re im each)
#   roots <case> t n r_1..r_n    the roots in s of D(., t) at real t, increasing
#   tstar <case> tstar           the first positive zero of D(0, .)
#   dist <case> c0 n x.. F.. f.. the distribution function and the density on the prototype's contour (the engine's present rule)
#   quant <case> c0 p q          the p-quantile (tolerance 1e-13)
# The blocks headed "paper" hold the numbers behind the paper's table of sizes and critical values under one variance
# break (paper/variance.jl), its examples, and the p-values, critical values and sizes of the Treasury bill rate
# (application/tbill_vol.jl, application/tbill_vol_gls.jl), with their inputs at full precision.
length(ARGS) == 1 || error("usage: regime_ref.jl <checkout of df-deterministics with paper/VolDF.jl>")
const REPO = ARGS[1]
include(joinpath(REPO, "paper", "VolDF.jl"))
using .VolDF, DFLimit, Printf, LinearAlgebra, Statistics
const DG = DFLimit.DFDet
const OUT = joinpath(@__DIR__, "..", "test", "ref", "regime_ref.txt")
# the data and the regressions of the Treasury bill rate, as in application/tbill_vol.jl (which cannot be included beside
# the package: the prototype and the package both export volpath)
module TBill
using LinearAlgebra
function readseries(path)
    ym = Tuple{Int,Int}[]; y = Float64[]
    for (i, l) in enumerate(eachline(path))
        i == 1 && continue
        a = split(l, ','); (length(a) == 2 && !isempty(a[2]) && a[2] != ".") || continue
        push!(ym, (parse(Int, a[1][1:4]), parse(Int, a[1][6:7]))); push!(y, parse(Float64, a[2]))
    end
    ym, y
end
months(a, b) = 12 * (b[1] - a[1]) + b[2] - a[2]
function adf(y::Vector{Float64}, X::Matrix{Float64}, k::Int, t0::Int = k + 2)
    n = length(y); rows = t0:n; m = length(rows); p = size(X, 2)
    Z = Matrix{Float64}(undef, m, p + 1 + k); dy = Vector{Float64}(undef, m)
    for (i, t) in enumerate(rows)
        dy[i] = y[t] - y[t-1]
        for j in 1:p; Z[i, j] = X[t, j]; end
        Z[i, p+1] = y[t-1]
        for j in 1:k; Z[i, p+1+j] = y[t-j] - y[t-j-1]; end
    end
    F = qr(Z); b = F \ dy; e = dy - Z * b; s2 = dot(e, e) / (m - size(Z, 2))
    Ri = inv(F.R); tau = b[p+1] / sqrt(s2 * sum(abs2, @view Ri[p+1, :]))
    bic = log(dot(e, e) / m) + size(Z, 2) * log(m) / m
    aic = log(dot(e, e) / m) + 2 * size(Z, 2) / m
    (; tau, bic, aic, e, rows, b)
end
function picklag(y, X, kmax, crit::Symbol = :aic)
    best = 0; bb = Inf
    for k in 0:kmax
        a = adf(y, X, k, kmax + 2); b = crit == :bic ? a.bic : a.aic
        b < bb && ((best, bb) = (k, b))
    end
    best
end
function regs(n, TB)
    t = collect(1.0:n)
    (; one = ones(n), t, du = Float64.(t .> TB), dt = max.(t .- TB, 0.0), dtb = Float64.(t .== TB + 1))
end
function regime_sd(e, rows, T1, T2)
    s = zeros(3); m = zeros(Int, 3)
    for (i, t) in enumerate(rows)
        j = t <= T1 ? 1 : t <= T2 ? 2 : 3
        s[j] += e[i]^2; m[j] += 1
    end
    sqrt.(s ./ m), m
end
end

"""the prototype's `vmodel` for the statistic that detrends first (as `vmodel_any` of application/tbill_vol.jl)"""
function vmodel_any(D0::DG.Determ, V::VolDF.VolPath)
    D = VolDF.refine(D0, V.lam)
    P = DG.prep(D)
    np = length(P.br) - 1; L = diff(P.br)
    reg = [VolDF.regime(V, (P.br[j] + P.br[j+1]) / 2) for j in 1:np]
    om = V.om[reg]; c = V.c[reg]
    bpos = zeros(Int, max(np - 1, 0)); Ne = P.N
    for j in 1:np-1
        b = P.br[j+1]
        i = findfirst(x -> abs(x - b) < 1e-12, P.jl)
        if i !== nothing
            bpos[j] = 1 + i
        elseif reg[j] != reg[j+1]
            Ne += 1; bpos[j] = Ne
        end
    end
    A = zeros(Ne, Ne); B = zeros(Ne, Ne); Lx = zeros(Ne, P.r)
    A[1:P.N, 1:P.N] .= P.A; B[1:P.N, 1:P.N] .= P.B; Lx[1:P.N, :] .= P.L
    VolDF.VModel(D, P, DG.model(P), V, np, L, om, c, Ne, bpos, A, B, Lx, copy(P.Dv))
end

fmt(x::Real) = @sprintf("%.17g", x)
fmt(z::Complex) = @sprintf("%.17g %.17g", real(z), imag(z))
fmt(x::Integer) = string(x)
fmt(x::AbstractString) = x
fmt(x::Symbol) = String(x)

function main()
    io = open(OUT, "w")
    line(args...) = println(io, join(map(fmt, args), " "))
    println(io, "# Reference values of the regime variants from the prototype paper/VolDF.jl of df-deterministics (commit 2ad4a34);")
    println(io, "# written by dev/regime_ref.jl, see the header of that file for the tags")

    designs = Dict{Int,Tuple{DG.Determ,String}}()
    did = Ref(0)
    function design(spec::String)
        for (id, (_, s)) in designs; s == spec && return id; end
        local w = split(spec)
        local Dn = w[1] == "trend" ? trend(parse(Int, w[2])) : perron(Symbol(w[2]), parse(Float64, w[3]))
        did[] += 1; designs[did[]] = (Dn, spec); println(io, "design $(did[]) $spec"); did[]
    end
    vols = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    vid = Ref(0)
    function vol(b::Vector{Float64}, om::Vector{Float64})
        for (id, v) in vols; v == (b, om) && return id; end
        vid[] += 1; vols[vid[]] = (b, om); line("vol", vid[], length(b), b..., om...); vid[]
    end
    # the prototype's model of a case
    function vm(d::Int, v::Int, c::Vector{Float64}, kind::Symbol, par::Float64)
        local Dm = designs[d][1]; local Vm = VolDF.volpath(vols[v]...; c)
        kind == :gls ? vmodel(Dm, Vm; gls = par) : kind == :first ? vmodel_any(detrend_first(Dm), Vm) : vmodel(Dm, Vm)
    end
    case(d, v, c, kind, par) = (d, v, length(c), c..., kind, par)
    v0of(M, kind, par) = kind == :stat ? M.V.om[1]^2 / (-2 * M.V.c[1]) : par
    PTS = ((0.3, -2.0), (-0.4, 0.5), (0.2 + 0.7im, 0.3 - 4.0im), (0.1, -50.0 + 20im))
    function dvals(d, v, c, kind, par)
        local M = vm(d, v, c, kind, par)
        for (s, t) in PTS
            local Dv = kind in (:normal, :stat) ? vDstart(M, s, t, v0of(M, kind, par)) : vDval(M, s, t)
            line("dval", case(d, v, c, kind, kind == :stat ? v0of(M, kind, par) : par)..., complex(s), complex(t), complex(Dv))
        end
    end
    function roots(d, v, c, kind, par, t::Float64)
        local M = vm(d, v, c, kind, par)
        if kind in (:normal, :stat)
            # (as paper/variance.jl: the determinant of the start is a polynomial in s of the degree of the rank of the
            # extended numerator, two with an intercept; its roots from its values at -1, 0 and 1)
            local v0 = v0of(M, kind, par)
            local dd = [real(vDstart(M, x, t, v0)) for x in (-1.0, 0.0, 1.0)]
            local a2 = (dd[1] + dd[3]) / 2 - dd[2]; local a1 = (dd[3] - dd[1]) / 2
            local rts = sort(real.([(-a1 + sg * sqrt(complex(a1^2 - 4 * a2 * dd[2]))) / (2 * a2) for sg in (-1, 1)]))
            line("roots", case(d, v, c, kind, kind == :stat ? v0 : par)..., t, length(rts), rts...)
        else
            local rts = sort(real.(1 ./ vparts(M, t)[2]))
            line("roots", case(d, v, c, kind, par)..., t, length(rts), rts...)
        end
    end
    function dist(d, v, c, kind, par, c0::Float64, xs::Vector{Float64}; CP = vpath(vm(d, v, c, kind, par)))
        local F, f = vol_both(CP, xs; c0)
        line("dist", case(d, v, c, kind, par)..., c0, length(xs), xs..., F..., f...)
        CP
    end
    function quant(d, v, c, kind, par, c0::Float64, p::Float64, x0::Float64; CP = vpath(vm(d, v, c, kind, par)))
        local qq = vol_quantile(CP, p; x0, tol = 1e-13, c0)
        line("quant", case(d, v, c, kind, par)..., c0, p, qq)
        qq
    end

    # ---------------------------------------------------------------------------------------------------------------
    println(io, "# the transform, the roots, t_*, the distribution and the quantiles of a set of cases")
    tr1 = design("trend -1"); tr0 = design("trend 0"); tr = design("trend 1"); tr2 = design("trend 2")
    pB = design("perron B 0.4"); pC = design("perron C 0.3"); pA = design("perron A 0.5")
    va = vol([0.3], [1.0, 3.0]); vb = vol([0.2, 0.7], [2.0, 1.0, 0.5]); vc = vol([0.5], [1.0, 1.0]); vd = vol([0.4], [3.0, 1.0]); ve = vol([0.1], [1.0, 3.0])
    ca = [0.0, 0.0]; cb = [-3.0, 0.0, -10.0]; cc = [0.0, -20.0]; cd = [-5.0, -5.0]; ce = [-4.0, -4.0]
    xs = [-4.0, -2.86, -1.5, 0.3]
    for (d, cbar) in ((tr, -13.5), (pB, -18.2), (tr0, -7.0), (pC, -17.0), (tr2, -10.0)), (v, c) in ((va, ca), (vb, cb), (vc, cc), (vd, cd))
        dvals(d, v, c, :gls, cbar)
        roots(d, v, c, :gls, cbar, 0.5); roots(d, v, c, :gls, cbar, -2.0)
        M = vm(d, v, c, :gls, cbar); line("tstar", case(d, v, c, :gls, cbar)..., vtstar(M))
        CP = vpath(M)
        dist(d, v, c, :gls, cbar, 0.0, xs; CP)
        (d, v) in ((tr, va), (pB, vb)) && dist(d, v, c, :gls, cbar, -2.0, xs; CP)
        for p in (0.01, 0.05, 0.5); quant(d, v, c, :gls, cbar, 0.0, p, -2.5; CP); end
        flush(io)
    end
    for d in (tr, pB, tr2, pA), (v, c) in ((va, ca), (vb, cb), (vc, cc), (vd, cd))
        dvals(d, v, c, :first, 0.0)
        roots(d, v, c, :first, 0.0, 0.5); roots(d, v, c, :first, 0.0, -2.0)
        M = vm(d, v, c, :first, 0.0); line("tstar", case(d, v, c, :first, 0.0)..., vtstar(M))
        CP = vpath(M)
        dist(d, v, c, :first, 0.0, 0.0, xs; CP)
        (d, v) in ((tr, va), (pB, vb)) && dist(d, v, c, :first, 0.0, -2.0, xs; CP)
        for p in (0.01, 0.05, 0.5); quant(d, v, c, :first, 0.0, 0.0, p, -3.0; CP); end
        flush(io)
    end
    # a normal start: the determinant of the start, and beside it that of the zero start at the same points (the two
    # yield the exponent of the fixed start: D_N = D_0 (1 - 2 v0 Q), the fixed start averaged over a normal one)
    for d in (tr1, tr0, tr, pC, pA), (v, c) in ((vb, cb), (ve, ce), (vd, cd), (va, [-2.0, 0.0]))
        dvals(d, v, c, :ls, 0.0)
        dvals(d, v, c, :stat, 0.0)
        dvals(d, v, c, :normal, 0.7)
        roots(d, v, c, :ls, 0.0, 0.5)
        d == tr0 && (roots(d, v, c, :stat, 0.0, 0.5); roots(d, v, c, :normal, 0.7, 0.5))
    end
    flush(io)

    # ---------------------------------------------------------------------------------------------------------------
    println(io, "# paper: the table of paper/variance.jl (one break in the variance; sizes at the 5% value of the constant variance,")
    println(io, "# which is the quantile of the first line of each block, and the 5% values under the break)")
    dn = design("trend -1")
    for (d, kind, par) in ((tr1, :ls, 0.0), (tr0, :ls, 0.0), (tr, :ls, 0.0), (tr, :gls, -13.5))
        D = designs[d][1]
        cv = vol_quantile(D, VolDF.volpath(0.0), 0.05; x0 = -2.5, gls = kind == :gls ? par : nothing)
        line("papercv", d, kind, par, cv)
        for δ in (1 / 5, 1 / 3, 1 / 2, 2.0, 3.0, 5.0), λ in 0.1:0.1:0.9
            v = vol([λ], [1.0, δ]); c = [0.0, 0.0]
            CP = dist(d, v, c, kind, par, 0.0, [cv])
            q = vol_quantile(CP, 0.05; x0 = cv, tol = 1e-13)
            line("quant", case(d, v, c, kind, par)..., 0.0, 0.05, q)
        end
        flush(io)
    end
    println(io, "# paper: the examples of paper/variance.jl (the roots at t = 1/2 for a problem and its reversal; the stationary start")
    println(io, "# of the first regime; the power under a persistence that changes, at the 5% value of the intercept)")
    roots(tr0, vol([0.3], [1.0, 3.0]), [0.0, 0.0], :ls, 0.0, 0.5)
    roots(tr0, vol([0.7], [3.0, 1.0]), [0.0, 0.0], :ls, 0.0, 0.5)
    roots(tr0, vol([0.3], [1.0, 3.0]), [-4.0, -4.0], :stat, 0.0, 0.5)
    roots(tr0, vol([0.3], [1.0, 1.0]), [-4.0, -4.0], :stat, 0.0, 0.5)
    cv0 = dist_quantile(trend(0), 0.05, 0.0)
    line("papercv", tr0, :pkg, 0.0, cv0)
    dist(tr0, vol([0.5], [1.0, 1.0]), [0.0, -20.0], :ls, 0.0, 0.0, [cv0])
    dist(tr0, vol([0.5], [1.0, 1.0]), [-20.0, 0.0], :ls, 0.0, 0.0, [cv0])
    flush(io)

    # ---------------------------------------------------------------------------------------------------------------
    # the Treasury bill rate: the inputs (regime dates, residual standard deviations, the statistic) from the data and the
    # regressions of application/tbill_vol.jl
    println(io, "# paper: the Treasury bill rate (application/tbill_vol.jl): the statistic tau, the three regimes and one break;")
    println(io, "# tbill <did> <kind> <par> <lag rule> k tau q0 | then dist at tau and at q0, and the quantiles")
    readseries = TBill.readseries; months = TBill.months; adf = TBill.adf; picklag = TBill.picklag; regs = TBill.regs; regime_sd = TBill.regime_sd
    ym, y = readseries(joinpath(REPO, "application", "data", "TB3MS.csv"))
    first = (1951, 4); last = (2007, 12); b1 = (1979, 10); b2 = (1982, 10); KMAX = 24
    i1 = findfirst(==(first), ym); i2 = findfirst(==(last), ym); yy = y[i1:i2]; n = length(yy)
    T1 = months(first, b1); T2 = months(first, b2); l1 = T1 / n; l2 = T2 / n
    R = regs(n, T1)
    Zb = hcat(R.one, R.t, R.dt); Xn = zeros(n, 0)
    stats = [("trend 1", :ls, yy, hcat(R.one, R.t), -3.4), (@sprintf("perron C %.17g", l1), :ls, yy, hcat(R.one, R.t, R.du, R.dt, R.dtb), -4.2),
             (@sprintf("perron B %.17g", l1), :first, yy - Zb * (Zb \ yy), Xn, -3.9)]
    for (spec, kind, ys, X, x0) in stats
        d = design(spec); D = kind == :first ? detrend_first(designs[d][1]) : designs[d][1]
        for (lab, k) in (("AIC", picklag(ys, X, KMAX, :aic)), ("BIC", picklag(ys, X, KMAX, :bic)), ("12lags", 12))
            a = adf(ys, X, k); sd, m = regime_sd(a.e, a.rows, T1, T2)
            q0 = dist_quantile(D, 0.05, 0.0)
            line("tbill", d, kind, 0.0, lab, k, a.tau, q0)
            v3 = vol([l1, l2], sd); v2 = vol([l1], [sd[1], sqrt((sd[2]^2 * m[2] + sd[3]^2 * m[3]) / (m[2] + m[3]))]); v1 = vol([l2], [1.0, 1.0])
            CP = dist(d, v3, zeros(3), kind, 0.0, 0.0, [a.tau, q0])
            quant(d, v3, zeros(3), kind, 0.0, 0.0, 0.05, x0; CP); quant(d, v3, zeros(3), kind, 0.0, 0.0, 0.01, x0 - 0.6; CP); quant(d, v3, zeros(3), kind, 0.0, 0.0, 0.10, x0 + 0.3; CP)
            CP2 = dist(d, v2, zeros(2), kind, 0.0, 0.0, [a.tau])
            quant(d, v2, zeros(2), kind, 0.0, 0.0, 0.05, x0; CP = CP2)
            dist(d, v1, zeros(2), kind, 0.0, 0.0, [a.tau])
            flush(io)
        end
    end
    println(io, "# paper: the GLS-detrended statistic of the Treasury bill rate (application/tbill_vol_gls.jl)")
    # (the detrending and the choice of cbar of that file, which runs its simulations when included)
    gls_resid(y, Z, cbar) = (nn = length(y); a = 1 + cbar / nn;
        ya = vcat(y[1], y[2:end] .- a .* y[1:end-1]); Za = vcat(Z[1:1, :], Z[2:end, :] .- a .* Z[1:end-1, :]); y - Z * (Za \ ya))
    CBAR = [-16.0, -17.6, -18.2, -18.2, -17.8, -17.2, -16.4, -15.6, -14.8]
    cbar_ckp(lam) = (x = clamp(lam, 0.1, 0.9) * 10; i = clamp(floor(Int, x), 1, 8); w = x - i; (1 - w) * CBAR[i] + w * CBAR[i+1])
    cbr = cbar_ckp(l1); d = design(@sprintf("perron B %.17g", l1)); D = designs[d][1]; ys = gls_resid(yy, Zb, cbr)
    for (lab, k) in (("AIC", picklag(ys, Xn, KMAX, :aic)), ("BIC", picklag(ys, Xn, KMAX, :bic)), ("12lags", 12))
        a = adf(ys, Xn, k); sd, m = regime_sd(a.e, a.rows, T1, T2)
        q0 = dist_quantile(D, 0.05, 0.0; gls = cbr)
        line("tbill", d, :gls, cbr, lab, k, a.tau, q0)
        v3 = vol([l1, l2], sd); v1 = vol([l2], [1.0, 1.0])
        CP = dist(d, v3, zeros(3), :gls, cbr, 0.0, [a.tau, q0])
        quant(d, v3, zeros(3), :gls, cbr, 0.0, 0.05, q0; CP); quant(d, v3, zeros(3), :gls, cbr, 0.0, 0.01, q0 - 0.6; CP); quant(d, v3, zeros(3), :gls, cbr, 0.0, 0.10, q0 + 0.3; CP)
        dist(d, v1, zeros(2), :gls, cbr, 0.0, [a.tau])
        flush(io)
    end
    close(io)
    println("wrote ", OUT, ": ", countlines(OUT), " lines")
end
main()
