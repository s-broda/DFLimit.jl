# The regime variants (src/DFDet_vol.jl): volatility and persistence regimes with GLS detrending, with the statistic that
# detrends first, and with a normal or a fixed start.
#
# 1. Against the prototype that computed these variants for the lead paper before the package did (test/ref/regime_ref.txt,
#    written by dev/regime_ref.jl from paper/VolDF.jl of df-deterministics): the determinant, the roots, t_*, the
#    distribution function, the density and the quantiles of a set of cases, and the numbers behind the paper's table of
#    sizes and critical values under one variance break, its examples and the Treasury bill rate.  The prototype has the
#    determinant of a normal start; its ratio to that of the zero start is the exponent of the fixed start averaged over
#    the start, which checks the fixed start's exponent.
# 2. The stationary start, which the prototype did not invert: a dummy boundary at a constant variance against the
#    engine's stationary start; the finite-sample moment generating function of the regression with a normal start; the
#    normal start as the average of the fixed one, at the level of the distribution function (the fixed start by the
#    engine's paths on a steep outer contour, an internal computation: the user-level functions refuse a fixed start
#    under regimes); the density against the distribution function; the checked computation; the classical functions;
#    argument errors.
using DFLimit, Test, LinearAlgebra
const DD = DFLimit.DFDet
const REFR = joinpath(@__DIR__, "..", "ref", "regime_ref.txt")
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end
rel(a, b) = abs(a - b) / abs(b)

"""the package's model of a case of the reference file"""
function ref_model(D::Determ, V::VolPath, c::Vector{Float64}, kind::String, par::Float64)
    kind == "gls" ? DD.vol_model(D, V, c; gls = par) : kind == "first" ? DD.vol_model(detrend_first(D), V, c) :
    kind == "stat" ? DD.vol_model(D, V, c; start = :stationary) : kind == "normal" ? DD.vol_model(D, V, c; start = :stationary, v0 = par) :
    DD.vol_model(D, V, c)
end
"""the distribution function with the fixed start J(0) = g under regimes, by the engine's paths of the fixed start on the
outer contour of the model (an internal computation: dependable for starts of moderate size on a steep contour only)"""
fixed_cdf(D::Determ, xs, c::Vector{Float64}, V::VolPath, g::Real; phi = nothing) =
    DD.vol_eval(D, xs, c, 1, V, phi, nothing, false, 1e-9, DD.vol_model(D, V, c; start = :fixed, g0 = g))
"""the design and the keywords of the user-level call of a case (least squares, GLS detrending, detrending first)"""
ref_call(D::Determ, kind::String, par::Float64) = kind == "gls" ? (D, (; gls = par)) : kind == "first" ? (detrend_first(D), (;)) : (D, (;))

"""
the exact joint moment generating function of the numerator and the denominator of the regression in a sample of n with
the start u_0 = sqrt(n v0) e_0, e_0 standard normal and independent of the innovations (Gaussian innovations; the
quadratic forms of the regression in (e_0, e_1, ..., e_n), real s and t)
"""
function discrete_mgf_start(D::Determ, V::VolPath, cr::Vector{Float64}, v0::Float64, n::Int, s::Real, t::Real)
    reg(r) = (i = 1; while i < length(V.om) && r > V.lam[i+1]; i += 1; end; i)
    om = [V.om[reg(k / n)] for k in 1:n]; ph = [1 + cr[reg(k / n)] / n for k in 1:n]
    S = zeros(n, n + 1)                               # u_k = S[k, :] (e_0, e_1, ..., e_n)
    S[1, 1] = ph[1] * sqrt(n * v0); S[1, 2] = om[1]
    for k in 2:n
        S[k, :] .= ph[k] .* S[k-1, :]; S[k, k+1] = om[k]
    end
    Sl = zeros(n, n + 1); Sl[1, 1] = sqrt(n * v0); Sl[2:n, :] .= S[1:n-1, :]
    X = DD.regmat(D, n)
    PSl = size(X, 2) == 0 ? Sl : (Q = Matrix(qr(X).Q); Sl .- Q * (transpose(Q) * Sl))
    N = transpose(PSl) * (S .- Sl) ./ n
    A = (N .+ transpose(N)) ./ 2; T = transpose(PSl) * PSl ./ n^2
    exp(-logdet(cholesky(Symmetric(I - 2s .* A .- 2t .* T))) / 2)
end

@testset "regime variants" begin
    lines = [split(l) for l in eachline(REFR) if !startswith(l, "#") && !isempty(strip(l))]
    num(s) = parse(Float64, s)
    designs = Dict{Int,Determ}(); vols = Dict{Int,VolPath}()
    for v in lines
        if v[1] == "design"
            designs[parse(Int, v[2])] = v[3] == "trend" ? trend(parse(Int, v[4])) : perron(Symbol(v[4]), num(v[5]))
        elseif v[1] == "vol"
            nb = parse(Int, v[3])
            vols[parse(Int, v[2])] = volpath(num.(v[4:3+nb]), num.(v[4+nb:4+2nb]))
        end
    end
    # the case of a line from position 2: (D, V, c, kind, par, next position)
    function case(v)
        D = designs[parse(Int, v[2])]; V = vols[parse(Int, v[3])]; nc = parse(Int, v[4])
        D, V, num.(v[5:4+nc]), String(v[5+nc]), num(v[6+nc]), 7 + nc
    end
    counts = Dict{String,Int}(); worst = Dict{String,Float64}()
    note(tag, e) = (counts[tag] = get(counts, tag, 0) + 1; worst[tag] = max(get(worst, tag, 0.0), e); e)
    d0 = Dict{Any,ComplexF64}()                        # the zero-start determinants, for the fixed start's exponent
    for v in lines
        v[1] == "dval" || continue
        D, V, c, kind, par, k = case(v)
        kind == "ls" && (d0[(v[2], v[3], c, v[k:k+3])] = complex(num(v[k+4]), num(v[k+5])))
    end
    @testset "against the prototype" begin
        paper = false                               # the blocks of the paper's numbers begin with the first line "papercv"
        for v in lines
            tag = String(v[1])
            tag == "papercv" && (paper = true)
            tag in ("dval", "roots", "tstar", "dist", "quant") || continue
            D, V, c, kind, par, k = case(v)
            if tag == "dval"
                s = complex(num(v[k]), num(v[k+1])); t = complex(num(v[k+2]), num(v[k+3])); ref = complex(num(v[k+4]), num(v[k+5]))
                VM = ref_model(D, V, c, kind, par)
                kind == "stat" && @test VM.v0 ≈ par rtol = 1e-14
                @test note("the determinant (" * kind * ")", rel(DD.vol_dval(VM, s, t), ref)) < 1e-10
                if kind in ("stat", "normal")
                    # the fixed start's exponent from the two determinants of the prototype: D_N = D_0 (1 - 2 v0 Q)
                    q = (1 - ref / d0[(v[2], v[3], c, v[k:k+3])]) / (2 * par)
                    @test note("the exponent of the fixed start", rel(DD.vol_qf(DD.vol_model(D, V, c; start = :fixed, g0 = 1.5), s, t), q)) < 1e-9
                end
            elseif tag == "roots"
                t = num(v[k]); n = parse(Int, v[k+1]); ref = num.(v[k+2:k+1+n])
                r = DD.vol_roots(ref_model(D, V, c, kind, par), t)
                @test length(r) == n
                @test note("the roots (" * kind * ")", maximum(abs.(r .- ref) ./ abs.(ref))) < 1e-9
            elseif tag == "tstar"
                @test note("t_*", rel(DD.tstar(ref_model(D, V, c, kind, par), 0.0)[1], num(v[k]))) < 1e-10
            elseif tag == "dist"
                c0 = num(v[k]); n = parse(Int, v[k+1]); xs = num.(v[k+2:k+1+n]); F = num.(v[k+2+n:k+1+2n]); f = num.(v[k+2+2n:k+1+3n])
                Dc, kw = ref_call(D, kind, par)
                # (the prototype's nodes are those of the rule :present, its outer contour the engine's 45-degree hyperbola;
                # the package's hyperbola under regimes starts at 70 degrees since the end of November 2026 (`VOL_PHI`), and
                # on it the rule :present is within 4.6e-12 of the prototype in F and 1.3e-12 in f, where it was 1.7e-15 on
                # the prototype's own contour; the default rule of one evaluation beside it)
                @test note("F, the present rule", maximum(abs, dist_cdf(Dc, xs, c; vol = V, c0, rule = :present, kw...) .- F)) < 2e-11
                @test note("f, the present rule", maximum(abs, dist_pdf(Dc, xs, c; vol = V, c0, rule = :present, kw...) .- f)) < 2e-11
                # (the default rule, the thin outer step on the 70-degree hyperbola: within 9.2e-12 of the prototype in F and
                # 4.7e-12 in f; on the 45-degree hyperbola it was off by 6.5e-9 in F and 3.0e-8 in f for the quadratic trend
                # detrended first under three regimes with their own c.  On the cells of the paper's table and of its
                # illustration the test leaves it out for the time it takes)
                paper && continue
                @test note("F, the default rule", maximum(abs, dist_cdf(Dc, xs, c; vol = V, c0, kw...) .- F)) < 1e-10
                @test note("f, the default rule", maximum(abs, dist_pdf(Dc, xs, c; vol = V, c0, kw...) .- f)) < 1e-10
            elseif tag == "quant"
                c0 = num(v[k]); p = num(v[k+1]); q = num(v[k+2])
                Dc, kw = ref_call(D, kind, par)
                # (the search to 1e-10: below 1e-12 the computed F has steps of the order 1e-13, where the paths are placed anew)
                @test note("the quantiles", abs(dist_quantile(Dc, p, c; vol = V, c0, x0 = -2.5, tol = 1e-10, rule = :present, kw...) - q)) < 1e-9
            end
        end
        @test counts["F, the present rule"] >= 280 && counts["the quantiles"] >= 350 && counts["the determinant (gls)"] == 80 &&
              counts["the determinant (first)"] == 64 && counts["the exponent of the fixed start"] == 160
        if get(ENV, "DFLIMIT_REGIME_REPORT", "") != ""
            for tag in sort(collect(keys(worst))); println(rpad(tag, 36), lpad(counts[tag], 5), "  largest difference ", worst[tag]); end
        end
    end

    xs = [-4.0, -2.86, -1.0, 0.5]
    V2 = volpath([0.3], [1.0, 3.0]); V3 = volpath([0.2, 0.7], [2.0, 1.0, 0.5])
    @testset "a dummy boundary at a constant variance: the engine's variants" begin
        V = volpath([0.4], [1.0, 1.0])
        for D in (trend(-1), trend(0), trend(1), perron(:C, 0.6)), c in (-2.0, -10.0)
            @test maximum(abs, dist_cdf(D, xs, c; vol = V, start = :stationary) .- dist_cdf(D, xs, c; start = :stationary, method = :paths)) < 1e-12
            @test maximum(abs, dist_pdf(D, xs, c; vol = V, start = :stationary) .- dist_pdf(D, xs, c; start = :stationary, method = :paths)) < 1e-12
            # (the fixed start's data along the contour, where the paths are those of the engine: a constant variance)
            @test maximum(abs, fixed_cdf(D, xs, [c, c], V, 1.5) .- dist_cdf(D, xs, c; start = :fixed, gamma0 = 1.5, method = :paths)) < 1e-9
        end
        for (D, cbar) in ((trend(1), -13.5), (perron(:B, 0.4), -18.0)), c in (0.0, -10.0)
            @test maximum(abs, dist_cdf(D, xs, c; vol = V, gls = cbar) .- dist_cdf(D, xs, c; gls = cbar, method = :paths)) < 1e-12
            @test maximum(abs, dist_cdf(detrend_first(D), xs, c; vol = V) .- dist_cdf(detrend_first(D), xs, c; method = :paths)) < 1e-12
        end
        # one regime is the engine itself, with its start and its detrending
        @test dist_cdf(trend(0), xs, -5.0; vol = volpath(Float64[]), start = :stationary) == dist_cdf(trend(0), xs, -5.0; start = :stationary)
        @test dist_cdf(trend(1), xs, -5.0; vol = volpath(Float64[]), gls = -13.5) == dist_cdf(trend(1), xs, -5.0; gls = -13.5)
    end
    @testset "a normal start: the transform against the finite-sample moment generating function" begin
        # (Richardson's extrapolation in 1/n of the exact transform at n = 200, 400, 800; the stationary start of the first regime)
        for (D, V, cr) in ((trend(0), V2, [-4.0, -4.0]), (trend(1), V3, [-3.0, 0.0, -10.0])), (s, t) in ((0.3, -2.0), (-0.5, 0.4))
            VM = DD.vol_model(D, V, cr; start = :stationary)
            ex = real(exp(-s / 2) / sqrt(DD.vol_dval(VM, s, t)))
            m1, m2, m4 = (discrete_mgf_start(D, V, cr, VM.v0, n, s, t) for n in (200, 400, 800))
            @test abs((4 * (2m4 - m2) - (2m2 - m1)) / 3 - ex) < 2e-5
        end
    end
    @testset "the normal start is the average of the fixed one" begin
        # F with the stationary start of the first regime (J(0) one more functional of a determinant of higher rank) against
        # the Gauss-Hermite average over gamma0 ~ N(0, v0) of F with the fixed start (the zero start's roots and the
        # exponent of the start): two computations that share the law of the path and nothing else
        J = SymTridiagonal(zeros(40), [sqrt(k / 2) for k in 1:39]); E = eigen(J); z = E.values; w = E.vectors[1, :] .^ 2
        for (D, V, c) in ((trend(0), V2, [-4.0, -4.0]), (trend(1), V3, [-8.0, 0.0, -10.0]), (perron(:C, 0.4), volpath([0.6], [3.0, 1.0]), [-5.0, -5.0]))
            v0 = DD.vol_model(D, V, c; start = :stationary).v0
            acc = zeros(length(xs))
            for k in eachindex(z)
                w[k] < 1e-12 && continue
                acc .+= w[k] .* fixed_cdf(D, xs, c, V, sqrt(2v0) * z[k]; phi = 85 * pi / 180)
            end
            @test maximum(abs, dist_cdf(D, xs, c; vol = V, start = :stationary) .- acc) < 1e-9
        end
    end
    @testset "the density, the checked computation, the quantiles" begin
        h = 1e-3; x = [-3.0, -2.0, -0.5]
        for (D, V, c, kw) in ((trend(0), V2, [-4.0, -4.0], (; start = :stationary)), (trend(1), V3, [-3.0, 0.0, -10.0], (; start = :stationary)),
                              (trend(1), V2, [0.0, 0.0], (; gls = -13.5)), (detrend_first(perron(:B, 0.4)), volpath([0.6], [3.0, 1.0]), [-5.0, -5.0], (;)))
            F = dist_cdf(D, [-2.0 - h, -2.0 + h], c; vol = V, kw...)
            @test abs((F[2] - F[1]) / (2h) - dist_pdf(D, [-2.0], c; vol = V, kw...)[1]) < 1e-6
            @test maximum(abs, dist_cdf(D, x, c; vol = V, verified = true, kw...) .- dist_cdf(D, x, c; vol = V, kw...)) < 1e-9
            q = dist_quantile(D, 0.05, c; vol = V, kw...)
            @test abs(dist_cdf(D, [q], c; vol = V, kw...)[1] - 0.05) < 1e-9
        end
    end
    @testset "the classical functions take the variants" begin
        x = [-3.0, -2.0]
        @test ltratio_cdf(x, "SU", -5; vol = V2) == dist_cdf(trend(0), x, -5.0; vol = V2, start = :stationary)
        @test ltratio_cdf(x, "DG", -5; vol = V2) == dist_cdf(trend(1), x, -5.0; vol = V2, gls = -13.5)
        @test occursin("fixed start (gamma0 != 0) is not computed", errmsg(() -> ltratio_cdf(x, "DU", -5; vol = V2, g0 = 0.5)))
        @test ltratio_pdf(x, trend(1), 0; vol = V2, gls = -10.0) == dist_pdf(trend(1), x, 0.0; vol = V2, gls = -10.0)
        @test ltratio_cdf(x, detrend_first(perron(:B, 0.4)), [0.0, -5.0]; vol = V2) == dist_cdf(detrend_first(perron(:B, 0.4)), x, [0.0, -5.0]; vol = V2)
        @test abs(ltratio_cdf(ltratio_icdf(0.05, "DG", 0; vol = V2), "DG", 0; vol = V2) - 0.05) < 1e-9
        @test abs(ltratio_cdf(ltratio_icdf(0.05, "SU", -5; vol = V2), "SU", -5; vol = V2) - 0.05) < 1e-9
    end
    @testset "the outer contour under regimes: a root near the imaginary axis far out" begin
        # (cases in which the thin step on the 45-degree hyperbola lost 1.4e-6 to 3.8e-4 at every x, and in the far upper
        # tail `cdf_range_check` refused the value: an early regime of high variance with a break in the deterministic terms
        # or a persistence that changes; see `VOL_PHI`)
        xs = [-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        for (D, V, c, kw) in ((perron(:B, 0.5), volpath([0.2, 0.7], [2.0, 1.0, 0.5]), [-5.0, 0.0, -10.0], (;)),
                              (perron(:C, 0.4), volpath([0.3], [3.0, 1.0]), [-5.0, -5.0], (; start = :stationary)),
                              (perron(:B, 0.3), volpath([0.2], [5.0, 1.0]), [-5.0, -20.0], (;)),
                              (perron(:B, 0.5), volpath([0.2, 0.7], [2.0, 1.0, 0.5]), [-20.0, 0.0, -10.0], (; c0 = 1.0)))
            F = dist_cdf(D, xs, c; vol = V, kw...)
            @test maximum(abs, F .- dist_cdf(D, xs, c; vol = V, verified = true, kw...)) < 1e-10
            @test all(diff(F) .> -1e-11) && 1 - F[end] < 1e-11
            @test minimum(dist_pdf(D, xs, c; vol = V, kw...)) > -1e-11
        end
        VM = DD.vol_model(perron(:B, 0.5), volpath([0.2, 0.7], [2.0, 1.0, 0.5]), [-5.0, 0.0, -10.0])
        @test DD.vol_path(VM).phi == DD.VOL_PHI == 70 * pi / 180 && DD.vol_hfac(1.5, :thin) == DD.VOL_HFAC == 2.0 && DD.vol_hfac(1.0, :present) == 1.0
        # (the engine's designs keep the 45-degree hyperbola)
        @test DD.cpath_cached(DD.model(DD.prep_cached(trend(1), nothing), :zero, 0.0), -5.0).phi == pi / 4
    end
    @testset "argument errors" begin
        x = [-2.0]
        @test occursin("fixed start (gamma0 != 0) is not computed", errmsg(() -> dist_cdf(trend(0), x, -5.0; vol = V2, start = :fixed, gamma0 = 1.0)))
        @test occursin("fixed start (gamma0 != 0) is not computed", errmsg(() -> dist_quantile(trend(0), 0.05, -5.0; vol = V2, start = :fixed, gamma0 = 1.0)))
        @test occursin("zero start only", errmsg(() -> dist_cdf(trend(1), x, -5.0; vol = V2, gls = -13.5, start = :stationary)))
        @test occursin("zero start only", errmsg(() -> dist_cdf(trend(1), x, -5.0; vol = V2, gls = -13.5, start = :fixed, gamma0 = 1.0)))
        @test occursin("zero start only", errmsg(() -> dist_cdf(detrend_first(trend(1)), x, -5.0; vol = V2, start = :stationary)))
        @test occursin("detrends first already", errmsg(() -> dist_cdf(detrend_first(trend(1)), x, -5.0; vol = V2, gls = -13.5)))
        @test occursin("cbar < 0", errmsg(() -> dist_cdf(trend(1), x, -5.0; vol = V2, gls = 1.0)))
        @test occursin("c < 0 on the first regime", errmsg(() -> dist_cdf(trend(0), x, [0.0, -5.0]; vol = V2, start = :stationary)))
        @test occursin("start must be", errmsg(() -> dist_cdf(trend(0), x, -5.0; vol = V2, start = :random)))
        @test occursin("gamma0 must be", errmsg(() -> dist_cdf(trend(0), x, -5.0; vol = V2, start = :fixed, gamma0 = Inf)))
        @test occursin("must have a fixed start", errmsg(() -> DD.vol_qf(DD.vol_model(trend(0), V2, -5.0), 0.3, -2.0)))
        @test occursin("variance of a normal start", errmsg(() -> DD.vol_model(trend(0), V2, [0.0, -5.0]; start = :stationary)))
        # (gamma0 = 0 is the zero start, with GLS detrending too)
        @test dist_cdf(trend(1), x, -5.0; vol = V2, gls = -13.5, start = :fixed, gamma0 = 0.0) == dist_cdf(trend(1), x, -5.0; vol = V2, gls = -13.5)
    end
    @testset "type stability" begin
        PT = Tuple{ComplexF64,ComplexF64,Vector{ComplexF64},Union{Nothing,DD.FixT}}
        for VM in (DD.vol_model(trend(1), V2, [0.0, -5.0]; gls = -13.5), DD.vol_model(trend(1), V2, [-1.0, -5.0]; start = :stationary),
                   DD.vol_model(trend(1), V2, [-1.0, -5.0]; start = :fixed, g0 = 1.0), DD.vol_model(detrend_first(trend(1)), V2, [0.0, -5.0]))
            W = DD.MWork(VM)
            @inferred PT DD.mgfparts(VM, 0.0, 1.0 + 2.0im, W)
            @inferred DD.sigma_vol!(W, VM)
            @inferred DD.vol_start!(W, VM)
        end
        @inferred dist_cdf(trend(1), [-2.0], 0.0; vol = V2, gls = -13.5)
        @inferred dist_cdf(trend(0), [-2.0], -1.0; vol = V2, start = :stationary)
        @inferred dist_quantile(detrend_first(trend(1)), 0.05, 0.0; vol = V2)
    end
end
