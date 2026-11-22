# The prototypes of the devices of Bao and Kan (src/DFKan*.jl)
using Test, DFLimit, Logging
const KAN = DFLimit.DFKan

@testset "DFKan: devices of Bao and Kan" begin
    X = DFLimit.DFExact; G = DFLimit.DFDet
    @testset "jets" begin
        x = KAN.jetvar(Val(6), 0.3)
        @test maximum(abs, (KAN.logp(exp(x)) - x).c) < 1e-14
        @test maximum(abs, (sqrt(x * x) - x).c) < 1e-14
        y = x / (1 + x)
        @test maximum(abs, (y * (1 + x) - x).c) < 1e-14
        g(u) = exp(sqrt(1 + u^2)) / (2 + u)
        f = g(x); h = 1e-4
        @test abs(KAN.coef(f, 1) - (g(0.3 + h) - g(0.3 - h)) / 2h) < 1e-7
        @test abs(2 * KAN.coef(f, 2) - (g(0.3 + h) - 2g(0.3) + g(0.3 - h)) / h^2) < 1e-6
    end
    @testset "the transform against D(s, t), branch at the point" begin
        for (cas, model, stat) in (("DK", 1, false), ("DU", 2, false), ("DT", 3, false), ("SK", 1, true), ("SU", 2, true), ("ST", 3, true)),
            c in (1.5, 0.0, -0.7, -6.0), (s, t) in ((0.7im, -1.3im), (-2.0im, 9.0im), (0.1im, 0.02im), (5.0im, 400.0im))
            (stat && c >= 0) && continue
            F = KAN.limcf(cas, c)
            D = X.Dfun(model, c, s, t; stat)
            want = exp(-s - ((model == 1 && !stat) ? c : 0.0)) / D
            @test isapprox(exp(2 * KAN.logmgf(F, s, t)), want; rtol = 1e-8)
        end
        # the modulus never exceeds one and the transform is continuous along a ray through many turns of the phase
        F = KAN.limcf("DT", -5.0); prev = 0.0im
        for u in 0.0:0.05:400.0
            L = KAN.logcf(F, u, 3.0u)
            @test real(L) <= 1e-12
            @test abs(L - prev) < 2.0
            prev = L
        end
    end
    @testset "P(t), M0 and the roots at a point against the tracked branches" begin
        for model in 1:4, stat in (false, true), c in (5.0, 0.0, -1.0, -20.0, -100.0)
            (stat && (c >= 0 || model == 4)) && continue
            tau = X.default_tau(model, c; stat)
            ts = [tau + im * tau * sinh(u) for u in 0.0:0.075*X.hufac(c):asinh(X.wmaxc(model, c) / tau)]
            A = X.tparts(model, c, ts; stat); B = KAN.tparts_pointwise(model, c, ts; stat)
            n = X.kernel(model, stat) == 1 ? 2 : 4
            for i in 1:n
                @test maximum(abs.(B[i] .- A[i]) ./ abs.(A[i])) < 1e-9
            end
            # a single far point, with nothing to track from
            t1 = ts[end]; tp = KAN.TPoint(model, c, stat)
            @test isapprox(KAN.tparts_point(tp, t1)[1], A[1][end]; rtol = 1e-9)
            @test isapprox(KAN.tparts_point(tp, conj(t1))[1], conj(A[1][end]); rtol = 1e-9)
        end
        v0 = ltratio_cdf(-2.0, "DU", -5); w0 = ltratio_pdf(-1.0, "ST", -3)
        KAN.pointwise!(true)
        try
            @test isapprox(ltratio_cdf(-2.0, "DU", -5), v0; atol = 1e-13)
            @test isapprox(ltratio_pdf(-1.0, "ST", -3), w0; atol = 1e-13)
        finally
            KAN.pointwise!(false)
        end
    end
    @testset "the coefficient statistic by the single inversion" begin
        # quantiles of n(phi^ - phi) from AR1LIM_ICDF of Bao and Kan (MATLAB): (case, c, par-free start g0, p, x)
        his = (("DK", 0.0, 0.0, 0.05, -8.039128902741016), ("DU", 0.0, 0.0, 0.05, -14.093590831289385), ("DT", 0.0, 0.0, 0.05, -21.711224283138424),
               ("SU", -5.0, 0.0, 0.5, -3.5578742468854818))
        for (cas, c, g0, p, x) in his
            @test abs(KAN.coef_cdf(x + c, cas, c; g0) - p) < 1e-9
            @test abs(KAN.coef_icdf(p, cas, c; g0) - (x + c)) < 1e-7
        end
        for (cas, c, g0, y) in (("DK", -5.0, 0.0, -9.0), ("DU", -5.0, 1.0, -12.0), ("DT", 1.0, 0.0, -10.0), ("SK", -1.0, 0.0, -3.0), ("SU", -10.0, 0.0, -20.0), ("ST", -5.0, 0.0, -25.0))
            v = with_logger(() -> coef_cdf(y, cas, c; g0), NullLogger())
            @test abs(KAN.coef_cdf(y, cas, c; g0) - v) < 1e-9
            h = 1e-4
            @test abs(KAN.coef_pdf(y, cas, c; g0) - (KAN.coef_cdf(y + h, cas, c; g0) - KAN.coef_cdf(y - h, cas, c; g0)) / 2h) < 1e-6
        end
        # at the chi-square point (y = -c0) the tail carries the slow decay: the closed form
        for (cas, c) in (("DK", 0.0), ("DK", -5.0), ("DU", 0.0), ("DU", -5.0), ("SK", -2.0), ("SU", -2.0))
            @test abs(KAN.coef_cdf(0.0, cas, c) - KAN.ray_cdf(cas, c)) < 1e-9
        end
        @test abs(KAN.ray_cdf("DK", 0.0) - 0.6826894921370859) < 1e-14
        @test abs(KAN.ray_cdf("DU", 0.0) - 0.95756559) < 1e-8
        m = KAN.coef_mom(1, "DK", 0.0); @test abs(m[1] + 1.78143) < 1e-5
        m = KAN.coef_mom(1, "DU", 0.0); @test abs(m[1] + 5.37913) < 1e-5
        @test abs(KAN.lin_cdf(0.3, -2.5, "DK", -5.0) - ltratio_lincdf(0.3, -2.5, "DK", -5.0; engine = :classical)) < 1e-9
        @test abs(KAN.envelope(-7.0) - power_envelope(-7.0)) < 1e-8
    end
    @testset "moments of the t-ratio from jets" begin
        for (cas, c, c0) in (("DK", 0.0, 0.0), ("DU", -5.0, 0.0), ("DT", -5.0, -5.0), ("SU", -3.0, 0.0))
            a = KAN.tratio_mom(4, cas, c; c0); b = with_logger(() -> ltratio_mom(4, cas, c; c0), NullLogger())
            @test maximum(abs.(a .- b) ./ abs.(b)) < 1e-8
        end
    end
    @testset "the general engine's determinant from the pivots" begin
        for (D, cas, start) in ((trend(-1), "DK", :zero), (trend(0), "DU", :zero), (trend(1), "DT", :zero), (trend(0), "SU", :stationary), (trend(1), "ST", :stationary)), c in (-4.0, 0.0)
            (start == :stationary && c >= 0) && continue
            F1 = KAN.limcf(cas, c); F2 = KAN.DetCF(G.startmodel(D, c, start, 0.0, nothing), c)
            for (t1, t2) in ((0.3, -0.2), (4.0, 30.0), (-25.0, 700.0), (60.0, -2000.0))
                @test isapprox(KAN.logcf(F2, t1, t2), KAN.logcf(F1, t1, t2); atol = 1e-8)
            end
        end
        D = perron(:C, 0.5)
        @test abs(KAN.gcoef_cdf(D, -15.0, -5.0)[1] - G.coef_cdf(D, [-15.0], -5.0)[1]) < 1e-9
        @test abs(KAN.gcoef_cdf(trend(1), -20.0, 0.0; gls = -13.5)[1] - G.coef_cdf(trend(1), [-20.0], 0.0; gls = -13.5)[1]) < 1e-9
        @test abs(KAN.pt_cdf(trend(1), 5.0, -13.5, 0.0)[1] - G.pt_cdf(trend(1), [5.0], -13.5, 0.0)[1]) < 1e-9
        @test abs(KAN.zero_cdf(D, -5.0)[1] - dist_cdf(D, [0.0], -5.0)[1]) < 1e-9
        # the line integral's provider: the branch at each point against the continuation
        v0 = dist_cdf(trend(2), [-2.5], -5.0; method = :line)[1]
        G.empty_caches!(); KAN.line_pointwise!(true)
        try
            @test isapprox(dist_cdf(trend(2), [-2.5], -5.0; method = :line)[1], v0; atol = 1e-13)
        finally
            KAN.line_pointwise!(false); G.empty_caches!()
        end
        # the outer contour: roots and log P at single nodes against the continuation along the path
        Mo = G.startmodel(D, -5.0, :zero, 0.0, nothing); CP = G.cpath(Mo, -5.0)
        for k in (2, 17, length(CP.t) ÷ 2, length(CP.t))
            lP, pos, neg = KAN.contour_parts(Mo.P, -5.0, CP.t[k])
            @test abs(lP - CP.lP[k]) < 1e-8
            @test length(pos) == Mo.rp && length(neg) == Mo.rm
        end
    end
end
