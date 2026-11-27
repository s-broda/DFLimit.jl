# The first t-ratio of a pair of complex unit roots at a harmonic frequency (src/DFDet_seas.jl): the sum of the residues
# against the engine's inversion on paths of the doubled transform, the exponential law without deterministic terms, the
# elementary form at the unit root (spherical Bessel functions), the moments in the table of Nabeya (2001); the reference
# values of the prototype (df-deterministics, extensions/seasonal, table.txt: critical values and powers at four decimals);
# compositions with a volatility path; argument errors and type stability.
using DFLimit, Test
const DD = DFLimit.DFDet
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end

"""the spherical Bessel function of order nu = -1..3 divided by its leading term at the origin: cos x, sin(x)/x, ..."""
function jhat(nu::Int, x::ComplexF64)
    sx = sin(x); cx = cos(x)
    nu == -1 && return cx
    nu == 0 && return sx / x
    nu == 1 && return 3 * (sx - x * cx) / x^3
    nu == 2 && return 15 * ((3 - x^2) * sx - 3x * cx) / x^5
    105 * ((15 - 6x^2) * sx - (15x - x^3) * cx) / x^7
end
"""t3 at the unit root for the polynomial seasonal trend of degree p = 0, 1, 2 from elementary functions: D(s,t) =
jhat_p {jhat_{p-1} - c_p s^2 jhat_{p+1}} at x = sqrt(t/2), c_p = 1/(4 (2p+1)(2p+3)); the roots +-s(t), the weight
1/(2 c_p jhat_p jhat_{p+1} s), M_V = 1/(jhat_{p-1} jhat_p); only the nodes of the outer contour CP are taken"""
function t3_unitroot(p::Int, xs::Vector{Float64}, CP)
    cp = 1 / (4 * (2p + 1) * (2p + 3))
    F = zeros(length(xs)); f = zeros(length(xs))
    pcs = [DD.xpieces(x, 0.0, 1.0) for x in xs]
    for k in eachindex(CP.t)
        t = CP.t[k]; x = sqrt(t / 2)
        jm = jhat(p - 1, x); j0 = jhat(p, x); jp = jhat(p + 1, x)
        s = sqrt(jm / (cp * jp)); real(s) < 0 && (s = -s)
        w = 1 / (2 * cp * j0 * jp * s); MV = 1 / (jm * j0)
        for ix in eachindex(xs)
            plus, minus = pcs[ix]
            _, M1, M2 = DD.ctr_moments(t, s, 0.0, xs[ix], 1.0, plus)
            gf = w * 2 * M2; gF = w * 2 * M1 / s
            if !isempty(minus)
                _, N1, N2 = DD.ctr_moments(t, -s, 0.0, xs[ix], 1.0, minus)
                gf += w * 2 * N2; gF += w * 2 * N1 / (-s)
                gF += MV * exp(-t * minus[1][1]^2) / t
            end
            F[ix] += CP.wt[k] * imag(gF * CP.dt[k]); f[ix] += CP.wt[k] * imag(gf * CP.dt[k])
        end
    end
    1 .- F, f
end

@testset "seasonal unit roots" begin
    xs = [-4.0, -2.5, -1.0, 0.0, 1.2]
    @testset "t3: the residues against the doubled transform on the paths" begin
        for D in (trend(-1), trend(0), trend(1), perron(:A, 0.4), perron(:C, 0.6)), (c, c0) in ((0.0, 0.0), (-5.0, 0.0), (-5.0, -2.0), (0.0, 1.5), (-20.0, 0.0))
            CP = DD.cpath_cached(DD.startmodel(D, c, :zero, 0.0), c)
            F2, f2 = DD.contour_eval(DD.t3_doubled(CP), xs; c0)[1:2]
            @test maximum(abs, hegy_t3_cdf(D, xs, c; c0) .- F2) < 1e-11
            @test maximum(abs, hegy_t3_pdf(D, xs, c; c0) .- f2) < 1e-11
        end
        @test abs(hegy_t3_cdf(trend(-1), [0.0])[1] - (1 - exp(-1))) < 1e-13          # |Z(1)|^2/2 is exponential
        for p in 0:2
            CP = DD.cpath_cached(DD.startmodel(trend(p), 0.0, :zero, 0.0), 0.0)
            Fe, fe = t3_unitroot(p, xs, CP)
            @test maximum(abs, hegy_t3_cdf(trend(p), xs) .- Fe) < 1e-12
            @test maximum(abs, hegy_t3_pdf(trend(p), xs) .- fe) < 1e-12
        end
        # far from the unit root, and the checked computation
        @test maximum(abs, hegy_t3_cdf(trend(0), xs, -50.0; verified = true) .- hegy_t3_cdf(trend(0), xs, -50.0)) < 1e-11
    end
    @testset "the moments of Nabeya (2001) from the densities" begin
        # (seasonal dummies: the mean and the standard deviation of t3, at eight decimals)
        z, w = DD.glrule(400)
        q = -12 .+ 24 .* z; f = hegy_t3_pdf(trend(0), q)
        m1 = 24 * sum(w .* q .* f); m2 = 24 * sum(w .* q .^ 2 .* f)
        @test abs(m1 + 1.96870124) < 1e-8
        @test abs(sqrt(m2 - m1^2) - 0.86627092) < 1e-8
    end
    @testset "the prototype's table: critical values of t3 and the power of its 5% test (four decimals)" begin
        for (p, r3, pw) in ((-1, [-2.5558, -2.2135, -1.9217, -1.5879], 0.7227), (0, [-3.9089, -3.6038, -3.3442, -3.0477], 0.2017),
                            (1, [-4.7096, -4.4181, -4.1710, -3.8901], 0.1144))
            D = trend(p)
            q3 = hegy_t3_quantile(D, [0.01, 0.025, 0.05, 0.10]; x0 = -2.0 - p)
            @test maximum(abs, q3 .- r3) < 1e-4
            @test abs(hegy_t3_cdf(D, [q3[3]], -5.0)[1] - pw) < 1e-4
        end
    end
    @testset "with a volatility path" begin
        V = volpath([0.5], [1.0, 3.0])
        for D in (trend(-1), trend(0), trend(1))
            VM = DD.vol_model(D, V, [0.0, 0.0])
            F2 = DD.contour_eval(DD.t3_doubled(DD.vol_path(VM)), xs)[1]
            @test maximum(abs, hegy_t3_cdf(D, xs; vol = V) .- F2) < 1e-11
        end
        @test hegy_t3_cdf(trend(0), xs; vol = volpath(Float64[])) == hegy_t3_cdf(trend(0), xs)
        @test abs(hegy_t3_cdf(trend(0), [-2.0], [-5.0, -5.0]; vol = volpath([0.3], [1.0, 1.0]))[1] - hegy_t3_cdf(trend(0), [-2.0], -5.0)[1]) < 1e-12
    end
    @testset "argument errors" begin
        @test occursin("at most 0", errmsg(() -> hegy_t3_cdf(trend(0), [1.0], 1.0)))
        @test occursin("p must lie in (0, 1)", errmsg(() -> hegy_t3_quantile(trend(0), 1.0)))
        @test occursin("c0 must be", errmsg(() -> hegy_t3_cdf(trend(0), [1.0]; c0 = Inf)))
        # (the second t-ratio and the F statistic are not part of this release)
        @test !isdefined(DFLimit, :hegy_t4_cdf) && !isdefined(DFLimit, :hegy_f_cdf) && !isdefined(DD, :SeasGrid)
    end
    @testset "type stability" begin
        @inferred hegy_t3_cdf(trend(0), [-2.0])
        @inferred hegy_t3_quantile(trend(0), 0.05)
    end
end
