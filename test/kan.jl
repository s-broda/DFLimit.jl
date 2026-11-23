# The devices of Bao and Kan ("Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order
# Autoregressions: O(1) Algorithms"; src/DFKan*.jl): the jets, the closed-form transform with the branch at each point,
# and the package's functions that take them (the coefficient statistic, the linear form, the power envelope and the
# moments of the t-ratio by the single inversion or from jets).  The tolerances are at least ten times the differences
# measured in November 2026 (scratch/int of the branch kan-integration).
using Test, DFLimit, Logging
const KAN = DFLimit.DFKan
quietly(f) = with_logger(f, NullLogger())
errm(f) = try; f(); ""; catch e; sprint(showerror, e); end

@testset "the devices of Bao and Kan" begin
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
        # the default against the engines' computation (method = :engine)
        for (cas, c, g0, y) in (("DK", -5.0, 0.0, -9.0), ("DU", -5.0, 1.0, -12.0), ("DT", 1.0, 0.0, -10.0), ("SK", -1.0, 0.0, -3.0), ("SU", -10.0, 0.0, -20.0),
                                ("ST", -5.0, 0.0, -25.0), ("DU", -5.0, 0.0, -14.0), ("DK", 0.0, 0.0, -8.039), ("DT", -2.0, 0.0, -4.0))
            @test abs(coef_cdf(y, cas, c; g0) - quietly(() -> coef_cdf(y, cas, c; g0, method = :engine))) < 1e-11       # (measured 8.2e-13)
        end
        # quantiles of n(phi^ - phi) from AR1LIM_ICDF of Bao and Kan (their MATLAB programs): (case, c, g0, p, x)
        for (cas, c, g0, p, x) in (("DK", 0.0, 0.0, 0.05, -8.039128902741016), ("DU", 0.0, 0.0, 0.05, -14.093590831289385), ("DT", 0.0, 0.0, 0.05, -21.711224283138424),
                                   ("SU", -5.0, 0.0, 0.5, -3.5578742468854818))
            @test abs(coef_cdf(x + c, cas, c; g0) - p) < 1e-11
            @test abs(coef_icdf(p, cas, c; g0) - (x + c)) < 1e-9
        end
        @test abs(coef_icdf(0.05, "DU") - quietly(() -> coef_icdf(0.05, "DU"; method = :engine))) < 1e-9         # (7.6e-11: the search of before stops at a width of 1e-9)
        # the error estimates, the options and the computation of before
        P, E = coef_cdf([-14.0, -5.0], "DU", -5; err = true)
        @test P == coef_cdf([-14.0, -5.0], "DU", -5) && all(0 .< E .< 1e-12)
        @test coef_cdf(-14.0, "DU", -5; method = :engine) == coef_cdf(-14.0, "DU", -5; engine = :classical) == ltratio_cdf(0.0, "DU", -5; c0 = -14.0)
        @test coef_cdf(-14.0, "DU", -5; method = :inversion) == coef_cdf(-14.0, "DU", -5)
        @test coef_cdf(-14.0, trend(0), -5) == coef_cdf(-14.0, "DU", -5)
        @test isnan(coef_cdf(-14.0, "DU", -5; method = :engine, err = true)[2])
        @test coef_cdf(-16.59, "DG", -13.5) == coef_cdf(-16.59, "DG", -13.5; method = :engine)          # (case DG: the engines)
        @test occursin("method = :inversion applies", errm(() -> coef_cdf(-16.59, "DG", -13.5; method = :inversion)))
        @test occursin("does not apply with engine = :general", errm(() -> coef_cdf(-14.0, "DU", -5; method = :inversion, engine = :general)))
        @test occursin("method must be", errm(() -> coef_cdf(-14.0, "DU", -5; method = :fast)))
        @test isequal(coef_cdf([-Inf, NaN, Inf], "DK"), [0.0, NaN, 1.0])
        # case DU with a fixed start far from the unit root (c = -60, g0 = 1.5): the engines' one evaluation (the contour
        # integral) is off by up to 1.5e-6 there; the single inversion is within 4.3e-12 of their checked computation (`verified = true`)
        F = KAN.limcf("DU", -60.0, 1.5)
        for p in (0.001, 0.05, 0.5, 0.95)
            y = KAN.ratio_icdf(F, p) - 60.0
            @test abs(coef_cdf(y, "DU", -60.0; g0 = 1.5) - quietly(() -> coef_cdf(y, "DU", -60.0; g0 = 1.5, method = :engine, verified = true))) < 5e-11
        end
        # at the chi-square point (y = -c0) the tail carries the slow decay: the closed form of the distribution function at
        # zero (AR1LIM_RAY of Bao and Kan)
        for (cas, c) in (("DK", 0.0), ("DK", -5.0), ("DU", 0.0), ("DU", -5.0), ("SK", -2.0), ("SU", -2.0))
            @test abs(coef_cdf(0.0, cas, c) - KAN.ray_cdf(cas, c)) < 1e-11
        end
        # the density against the general engine (the kernel 2 v^3 of its contour integral) and against differences
        for (cas, c, g0, y) in (("DK", -5.0, 0.0, -9.0), ("DU", -5.0, 1.0, -12.0), ("DT", 1.0, 0.0, -10.0), ("SK", -1.0, 0.0, -3.0), ("SU", -10.0, 0.0, -20.0), ("ST", -5.0, 0.0, -25.0))
            f = coef_pdf(y, cas, c; g0)
            @test abs(f - quietly(() -> coef_pdf(y, cas, c; g0, method = :engine))) < 5e-12          # (2.1e-13)
            h = 1e-4
            @test abs(f - (coef_cdf(y + h, cas, c; g0) - coef_cdf(y - h, cas, c; g0)) / 2h) < 1e-7
        end
        @test coef_pdf(-8.039, trend(-1)) == coef_pdf(-8.039, "DK")
        # the moments: the mean of n(phi^ - 1) at the unit root (-1.7814 without deterministic terms, -5.3791 with an
        # intercept), and the first two moments against the integrals of the density
        @test abs(coef_mom(1, "DK")[1] + 1.78143) < 1e-5 && abs(coef_mom(1, "DU")[1] + 5.37913) < 1e-5
        for (cas, c) in (("DK", -5.0), ("SU", -10.0))
            m = coef_mom(2, cas, c)
            # (on [-300, 100]: far out the density is rounding noise of 1e-15 and below, which y^2 weighs up)
            m1 = KAN.quadgk(y -> y * coef_pdf(y, cas, c), -300.0, 100.0; rtol = 1e-11)[1]
            m2 = KAN.quadgk(y -> y^2 * coef_pdf(y, cas, c), -300.0, 100.0; rtol = 1e-11)[1]
            @test abs(m1 / m[1] - 1) < 2e-11 && abs(m2 / m[2] - 1) < 2e-11        # (1.3e-12)
        end
        @test coef_mom(2, trend(0), -5) == coef_mom(2, "DU", -5)
        @test occursin("cases DK, DU, DT, SK, SU, ST", errm(() -> coef_mom(1, "DG")))
    end
    @testset "the linear form and the power envelope by the single inversion" begin
        for (b, a, c) in ((-0.2639, -3.5, 0.0), (0.3, -2.5, -5.0), (-0.1, 1.0, -13.5), (5.0, -3.5, 2.0))
            @test abs(ltratio_lincdf(b, a, "DK", c) - ltratio_lincdf(b, a, "DK", c; method = :engine)) < 1e-10          # (5.4e-12)
        end
        # the other classical cases through their designs (the general engine before)
        for (D, g0, st) in ((trend(0), 0.0, nothing), (trend(1), 0.0, nothing), (trend(0), 0.0, :stationary), (trend(-1), 1.5, nothing))
            @test abs(ltratio_lincdf(-0.2639, -3.5, D, -5; g0, start = st) - ltratio_lincdf(-0.2639, -3.5, D, -5; g0, start = st, method = :engine)) < 1e-10          # (5.4e-12)
        end
        # a = 0 (the law of U alone, whose transform decays like a power): the engines' closed form
        @test ltratio_lincdf(-0.3, 0.0, "DK", -5.0) == ltratio_lincdf(-0.3, 0.0, "DK", -5.0; method = :engine)
        P, E = ltratio_lincdf([-0.3, 0.1], -3.5, "DK", 0; err = true)
        @test all(E .< 1e-12)
        # the envelope against the general engine's (the critical value to 1e-10)
        for c in (-7.0, -13.5, -1.0, -0.01, -1e-4)
            @test abs(power_envelope(c) - power_envelope(c; engine = :general)) < 1e-11       # (7.4e-13)
        end
        @test power_envelope([-7.0, 0.0])[2] == 0.05
        @test power_envelope(-13.5, "DG") == power_envelope(-13.5, "DG"; method = :engine)
    end
    @testset "moments of the t-ratio from jets" begin
        # near the unit root against the engines' circles (within 1.4e-11 relative)
        for (cas, c, c0) in (("DK", 0.0, 0.0), ("DU", -5.0, 0.0), ("DT", -5.0, -5.0), ("SU", -3.0, 0.0))
            a = ltratio_mom(4, cas, c; c0); b = quietly(() -> ltratio_mom(4, cas, c; c0, method = :engine))
            @test maximum(abs.(a .- b) ./ abs.(b)) < 2e-10
        end
        @test ltratio_mom(2, "DU")[2] ≈ 3.05599535 atol = 1e-8                  # Nabeya (1999)
        # the stored values of the classical engine's circles: within 3.7e-9 relative in test/ref/mom_ref.txt (c = c0 = -60,
        # where the integral of x^j times the density is within 1.5e-9 of the jets and 5.2e-9 of the circles), and in
        # test/engine/largec_ref.txt within 3.2e-10 at c0 = 0, 6e-9 with c0 != 0 and a zero start, 2e-7 with c0 = c and a
        # stationary start (c from -100 to -480, where the circles move towards the jets under refinement)
        casecode(m, st) = (st == 1 ? "S" : "D") * "KUT"[Int(m)]
        for l in eachline(joinpath(@__DIR__, "ref", "mom_ref.txt"))
            (startswith(l, "#") || isempty(strip(l))) && continue
            p = parse.(Float64, split(l)); m, c, c0, g0, st = p[1:5]; mr = p[6:9]
            @test maximum(abs.(ltratio_mom(4, casecode(m, st), c; c0, g0) ./ mr .- 1)) < 4e-8
        end
        for l in eachline(joinpath(@__DIR__, "engine", "largec_ref.txt"))
            p = split(l); p[1] == "mom" || continue
            cas = String(p[2]); c, c0 = parse.(Float64, p[3:4]); mr = parse.(Float64, p[5:8])
            tol = c0 == 0 ? 4e-9 : cas[1] == 'S' ? 2e-6 : 6e-8
            @test maximum(abs.(ltratio_mom(4, cas, c; c0) ./ mr .- 1)) < tol
        end
        # with a fixed start above the unit root the engines' one evaluation is off (by 2e-4 at c = 1 and 0.65 at c = 2 with
        # c0 = -2, g0 = 1, case DK); their circles with Nu = 256, h = 0.0125 are within 3.4e-13 of the jets at c = 2
        m = ltratio_mom(4, "DK", 2.0; c0 = -2.0, g0 = 1.0)
        mr = DFLimit.DFExact.moments(1; c = 2.0, c0 = -2.0, g0 = 1.0, pmax = 4, Nu = 256, h = 0.0125)
        @test maximum(abs.(m ./ mr .- 1)) < 5e-12
        mm, e = ltratio_mom(4, "DK", 20.0; c0 = 20.0, err = true)
        @test all(e .< 1e-9) && abs(mm[2] - 1) < 1e-5
        @test ltratio_mom(4, "DU", -5; engine = :classical) == DFLimit.DFExact.moments(2; c = -5.0, pmax = 4)   # (engine = :classical: the circles)
        @test occursin("does not apply with engine = :general", errm(() -> ltratio_mom(4, "DU", -5; method = :jets, engine = :general)))
    end
    @testset "the general engine's determinant from the pivots" begin
        for (D, cas, start) in ((trend(-1), "DK", :zero), (trend(0), "DU", :zero), (trend(1), "DT", :zero), (trend(0), "SU", :stationary), (trend(1), "ST", :stationary)), c in (-4.0, 0.0)
            (start == :stationary && c >= 0) && continue
            F1 = KAN.limcf(cas, c); F2 = G.DetCF(G.startmodel(D, c, start, 0.0, nothing), c)
            for (t1, t2) in ((0.3, -0.2), (4.0, 30.0), (-25.0, 700.0), (60.0, -2000.0))
                @test isapprox(KAN.logcf(F2, t1, t2), KAN.logcf(F1, t1, t2); atol = 1e-8)
            end
        end
        D = perron(:C, 0.5)
        @test abs(coef_cdf_gp(D, [-15.0], -5.0)[1] - G.coef_cdf(D, [-15.0], -5.0)[1]) < 1e-9
        @test abs(coef_cdf_gp(trend(1), [-20.0], 0.0; gls = -13.5)[1] - G.coef_cdf(trend(1), [-20.0], 0.0; gls = -13.5)[1]) < 1e-9
        @test abs(pt_cdf(trend(1), [5.0], -13.5, 0.0)[1] - pt_cdf(trend(1), [5.0], -13.5, 0.0; method = :grid)[1]) < 1e-12
        # the point-optimal statistic by the single inversion against the contour integral of the general engine on its
        # quadratic form (the paths around the roots: an independent evaluation), within 3.3e-12 from c = -30 to 4, also where
        # the grid refuses (c = 4 with an intercept and a linear trend) and beyond its range of q
        ptc(D, q, cbar, c, which) = (P = G.prep_pt(D, cbar); CP = G.cpath(G.model(P), c);
                                     G.contour_eval_specs(G.line_refined(CP, Float64[]; amax = [abs(CP.Mo.a + q - 0.5)]), [[G.XSpec(-cbar^2, 0.0, CP.Mo.a + q - 0.5; dA = 1.0)]])[which][1])
        for (D, cbar, c, q) in ((trend(0), -7.0, 0.0, 18.1014815), (trend(1), -13.5, 4.0, 969.3008382), (perron(:B, 0.5), -17.8, 2.0, 16.5697), (trend(0), -7.0, 4.0, 2223.714162),
                                (trend(1), -13.5, -30.0, 1.5))
            @test abs(pt_cdf(D, [q], cbar, c)[1] - ptc(D, q, cbar, c, 1)) < 5e-11
            @test abs(pt_pdf(D, [q], cbar, c)[1] - ptc(D, q, cbar, c, 2)) < 5e-10
        end
        @test occursin("does not resolve", errm(() -> pt_cdf(trend(1), [969.3008382], -13.5, 4.0; method = :grid)))
        @test abs(pt_quantile(trend(1), 0.05, -13.5) - pt_quantile(trend(1), 0.05, -13.5; method = :grid)) < 1e-10
        @test abs(pt_envelope(trend(0), [-7.0])[1] - pt_envelope(trend(0), [-7.0]; method = :grid)[1]) < 1e-10
        P, E = pt_cdf(trend(1), [5.0, 10.0], -13.5, 0.0; err = true); @test all(E .< 1e-12) && P == pt_cdf(trend(1), [5.0, 10.0], -13.5, 0.0)
        # coef_cdf_gp by the single inversion against the contour integral (within 6e-14 from c = -30 to 5)
        for (D, c, y, kw) in ((trend(0), -5.0, -13.83853904, (;)), (trend(2), 0.0, -20.0, (;)), (perron(:C, [0.3, 0.7]), -13.5, -15.0, (;)), (trend(1), -13.5, -20.0, (start = :stationary,)), (trend(0), 5.0, 9.942147816, (;)))
            @test abs(coef_cdf_gp(D, [y], c; kw...)[1] - G.coef_cdf(D, [y], c; kw...)[1]) < 1e-11
        end
        @test coef_cdf_gp(trend(0), [-14.0], -5.0; method = :grid) == G.coef_cdf_grid(trend(0), [-14.0], -5.0)
        @test abs(G.zero_cdf(D, -5.0)[1] - dist_cdf(D, [0.0], -5.0)[1]) < 1e-9
        # the line integral's provider: the branch at each point against the continuation
        v0 = dist_cdf(trend(2), [-2.5], -5.0; method = :line)[1]
        G.empty_caches!(); G.line_pointwise!(true)
        try
            @test isapprox(dist_cdf(trend(2), [-2.5], -5.0; method = :line)[1], v0; atol = 1e-13)
        finally
            G.line_pointwise!(false); G.empty_caches!()
        end
        # the outer contour: roots and log P at single nodes against the continuation along the path
        Mo = G.startmodel(D, -5.0, :zero, 0.0, nothing); CP = G.cpath(Mo, -5.0)
        for k in (2, 17, length(CP.t) ÷ 2, length(CP.t))
            lP, pos, neg = G.contour_parts(Mo.P, -5.0, CP.t[k])
            @test abs(lP - CP.lP[k]) < 1e-8
            @test length(pos) == Mo.rp && length(neg) == Mo.rm
        end
    end
end
