# The merged package: the choice of engine (DFLimit.autoengine, the table "Which engine" of the README) cell by cell, the
# calls taking the engine of their cell, the two engines forced against each other where both are valid, and the classical
# functions with a design of the general engine in place of the case.
using DFLimit, Test, Logging
const MD = DFLimit; const DXM = DFLimit.DFExact; const GEM = DFLimit.DFDet
errm(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end
quietly(f) = with_logger(NullLogger()) do; f(); end
spec(cas; g0 = 0.0) = MD.case_spec(cas, g0)

@testset "merged package: interface and choice of engine" begin
    @testset "the rule, cell by cell" begin
        ae(f, S, c, c0, x) = MD.autoengine(f, S, Float64(c), Float64(c0), Float64(x))
        # the t-ratio
        @test ae(:dist, spec("DU"), 0, 0, -2) == :classical                 # zero start, c0 = 0
        @test ae(:dist, spec("DK"), -300, 0, -2) == :classical
        @test ae(:dist, spec("SU"), -10, 0, -2) == :classical               # stationary start, c0 = 0
        @test ae(:dist, spec("DU"), -10, -10, -2) == :classical             # zero start, c0 < 0, -100 <= c <= 0
        @test ae(:dist, spec("DU"), -5, 2, -3) == :general                  # zero start, c0 > 0, -100 < c < 0
        @test ae(:dist, spec("DU"), -100, 3, -3) == :general && ae(:dist, spec("DU"), 2, 1, -3) == :classical
        @test ae(:dist, spec("DK"), -5, 2, -3) == :general && ae(:dist, spec("DK"), 0, 2, -2 - 1e-4) == :general    # case DK below the tangency x* = -2
        @test ae(:dist, spec("DK"), -5, 2, -2 + 1e-4) == :classical                                              # just above it
        @test ae(:dist, spec("DT"), -300, -300, -1) == :general             # zero start, c0 != 0, c < -100
        @test ae(:dist, spec("SU"), -10, -10, -1) == :general               # stationary start, c0 != 0, c <= -0.1
        @test ae(:dist, spec("SK"), -1e-4, -1e-4, 0.5) == :classical        # next to the unit root
        @test ae(:dist, spec("DK"), 5, -3, 20) == :classical                # c0 < 0 above the unit root (the general engine off by 7e-3)
        @test ae(:dist, spec("DT"), 7, 0, 200) == :classical                # the bulk above the unit root
        @test ae(:dist, spec("DK"; g0 = 1.0), 0, 0, -1) == :general         # fixed start, case DK, (c g0)^2/2 <= 400
        @test ae(:dist, spec("DK"; g0 = 2.0), -20, 0, -1) == :classical
        @test ae(:dist, spec("DK"; g0 = 1.0), 2, 0, -1) == :classical
        # (c0 < 0 near the unit root: the classical engine; the general engine's one evaluation of before was off in the upper
        # tail, by 2.7e-6 at the 99.99% point at c = 0.5, c0 = -5, g0 = 1)
        @test ae(:dist, spec("DK"; g0 = 1.0), 0.5, -5, 25) == :classical && ae(:dist, spec("DK"; g0 = 1.0), -2, -5, 10) == :general
        @test ltratio_cdf(25.173, "DK", 0.5; c0 = -5.0, g0 = 1.0) ≈ ltratio_cdf(25.173, "DK", 0.5; c0 = -5.0, g0 = 1.0, engine = :classical, verified = true) atol = 1e-10
        @test ae(:dist, spec("DU"; g0 = 1.0), -10, 0, -1) == :classical     # fixed start, cases DU, DT
        # (far from the unit root the general engine: c0 > 0 below c = -20, or below c = -10 where |c g0| > 20, and any c0
        # below c = -20 where |c g0| > 60; the classical contour integral's one evaluation was off there by up to 2.2e-2 of
        # the peak of the density and 3.9e-5 in F, its checked computation by 8.4e-6 in the density)
        @test ae(:dist, spec("DU"; g0 = 0.5), -30, 2, -3) == :general && ae(:dist, spec("DU"; g0 = 0.5), -15, 2, -3) == :classical
        @test ae(:dist, spec("DU"; g0 = 1.5), -13.5, 2, -3) == :general && ae(:dist, spec("DT"; g0 = 1.0), -13.5, 2, -3) == :classical
        @test ae(:dist, spec("DU"; g0 = 1.5), -60, -2, -3) == :general && ae(:dist, spec("DT"; g0 = 1.0), -60, -2, -3) == :classical
        let x = -9.1684, h = 2e-3      # the density is the derivative of F (Richardson's central difference; 8e-12 measured)
            Fh = ltratio_cdf(x .+ h .* [-2, -1, 1, 2], "DU", -60; c0 = 2.0, g0 = 1.5)
            @test abs(ltratio_pdf(x, "DU", -60; c0 = 2.0, g0 = 1.5) - (8 * (Fh[3] - Fh[2]) - (Fh[4] - Fh[1])) / (12h)) < 1e-8
        end
        @test ae(:dist, spec("DG"), -10, 0, -2) == :classical
        # x = 0, the points of the coefficient statistic: c0 > 0 the classical engine, c0 < 0 in case DK for c > -5 the general
        @test ae(:dist, spec("DU"), -5, 2, 0) == :classical && ae(:dist, spec("DK"), 0, 2, 0) == :classical && ae(:dist, spec("DT"), -80, 2, 0) == :classical
        @test ae(:dist, spec("DK"), 0, -3, 0) == :general && ae(:dist, spec("DK"), -5, -3, 0) == :classical && ae(:dist, spec("DU"), -2, -3, 0) == :classical
        @test ae(:dist, spec("DU"), -300, 2, 0) == :general && ae(:dist, spec("DU"), 2, 1, 0) == :classical      # (beyond: the rule of the t-ratio)
        @test coef_cdf([-5.0, 1.0], "DK", 0; method = :engine) == [ltratio_cdf(0.0, "DK", 0; c0 = -5.0, engine = :general), ltratio_cdf(0.0, "DK", 0; c0 = 1.0, engine = :classical)]
        # the other functions
        @test ae(:icdf, spec("DT"), -5, 0, 0) == :general && ae(:icdf, spec("ST"), -10, 0, 0) == :general && ae(:icdf, spec("ST"), -0.1, 0, 0) == :classical
        @test ae(:mom, spec("SU"), -0.1, 0, 0) == :classical && ae(:mom, spec("SK"), -0.01, -0.01, 0) == :general && ae(:mom, spec("ST"), -2, 0, 0) == :general
        @test ae(:icdf, spec("DU"), 0, 0, 0) == :general && ae(:icdf, spec("DU"), -10, 0, 0) == :classical
        @test ae(:icdf, spec("DK"), 0, 0, 0) == :classical && ae(:icdf, spec("DG"), 0, 0, 0) == :classical
        @test ae(:icdf, spec("DU"; g0 = 1.0), -5, 0, 0) == :general && ae(:icdf, spec("DU"; g0 = 0.5), -20, 2, 0) == :classical
        @test ae(:mom, spec("DT"), -5, 0, 0) == :general && ae(:mom, spec("SK"), -300, -300, 0) == :classical && ae(:mom, spec("DU"), -10, -10, 0) == :classical
        lk = MD.lin_spec("DK", -13.5); lg = MD.lin_spec("DG", -13.5)
        @test ae(:lincdf, lk, 0, 0, NaN) == :general && ae(:lincdf, lk, 0, 0, 0.0) == :classical && ae(:lincdf, lk, 0, 0, -3.5) == :classical
        @test ae(:lincdf, lk, 5, 0, 0.0) == :classical && ae(:lincdf, lk, 5, 0, NaN) == :general
        @test ae(:lincdf, lg, 0, 0, -7.3) == :classical && ae(:lincdf, lg, 4, 0, 0.1) == :general && ae(:lincdf, lg, 4.5, 0, 0.1) == :general
        @test ae(:lincdf, lg, 0, 0, 0.0) == :classical && ae(:lincdf, MD.lin_spec("DG", -5.0), 0, 0, -2.0) == :classical
        @test ae(:lincdf, MD.lin_spec("DG", -1.0), 0, 0, -2.0) == :classical && ae(:lincdf, MD.lin_spec("DG", -1.0), 2, 0, -2.0) == :general
        @test ae(:adfmax, spec("SU"), -10, 0, 0) == :general && ae(:adfmax, spec("SU"), -1e-4, 0, 0) == :classical && ae(:adfmax, spec("DU"), 0, 0, 0) == :classical
        @test ae(:envelope, spec("DK"), -7, 0, 0) == :general && ae(:envelope, spec("DK"), -1e-4, 0, 0) == :classical
        @test ae(:envelope, spec("DG"), -13.5, 0, 0) == :classical && ae(:envelope, spec("DG"), -0.5, 0, 0) == :classical
        @test ae(:region, spec("DK"), -5, 0, 0) == :classical
    end

    @testset "the rule of November 26, 2026: the moved cells, on both sides of each new border" begin
        ae(f, S, c, c0, x) = MD.autoengine(f, S, Float64(c), Float64(c0), Float64(x))
        # zero start, c0 > 0: the general engine for c <= 1 at every c (c = -100 included), and in case DT up to c = 2
        @test ae(:dist, spec("DU"), -100, 3, -3) == :general && ae(:dist, spec("DK"), -100, 1, -3) == :general && ae(:dist, spec("DT"), -100, 10, -3) == :general
        @test ae(:dist, spec("DT"), 2, 1, -2) == :general && ae(:dist, spec("DT"), 2.5, 1, -2) == :classical
        @test ae(:dist, spec("DU"), 1.5, 1, -2) == :classical && ae(:dist, spec("DK"), 1.5, 1, -1) == :classical
        @test ae(:dist, spec("DT"), 1.5, 2, 0) == :classical                # (x = 0, the coefficient statistic: the classical line integral)
        # zero start, c0 < 0: case DK for -5 < c <= 2, case DT for -50 <= c <= -10
        @test ae(:dist, spec("DK"), 0, -2, 1) == :general && ae(:dist, spec("DK"), -4.9, -5, 1) == :general && ae(:dist, spec("DK"), 2, -2, 1) == :general
        @test ae(:dist, spec("DK"), -5, -5, 1) == :classical && ae(:dist, spec("DK"), 2.5, -2, 1) == :classical
        @test ae(:dist, spec("DT"), -10, -10, 1) == :general && ae(:dist, spec("DT"), -50, -5, 1) == :general && ae(:dist, spec("DT"), -30, -30, 0) == :general
        @test ae(:dist, spec("DT"), -9, -9, 1) == :classical && ae(:dist, spec("DT"), -51, -5, 1) == :classical && ae(:dist, spec("DU"), -10, -10, 1) == :classical
        # zero start, c0 = 0 above the unit root: the general engine's paths in case DU for 0 < c <= 1.5, in case DT for 0 < c <= 2
        @test ae(:dist, spec("DU"), 0.1, 0, -1) == :general && ae(:dist, spec("DU"), 1.5, 0, -1) == :general && ae(:dist, spec("DU"), 1.6, 0, -1) == :classical
        @test ae(:dist, spec("DT"), 2, 0, -1) == :general && ae(:dist, spec("DT"), 2.1, 0, -1) == :classical && ae(:dist, spec("DT"), 0, 0, -1) == :classical
        @test ae(:dist, spec("DK"), 1, 0, -1) == :classical && ae(:dist, spec("SU"), -1, 0, -1) == :classical
        # the fixed start of case DK: also |g0| <= 1.5 with |c g0| <= 50 and c >= -100; |g0| (a negative g0 as its absolute value)
        @test ae(:dist, spec("DK"; g0 = 1.5), -20, 0, -1) == :general && ae(:dist, spec("DK"; g0 = 1.0), -50, 2, -1) == :general
        @test ae(:dist, spec("DK"; g0 = 1.0), -51, 0, -1) == :classical && ae(:dist, spec("DK"; g0 = 2.0), -15, 0, -1) == :classical
        @test ae(:dist, spec("DK"; g0 = -1.0), -100, 0, -1) == :classical && ae(:dist, spec("DK"; g0 = -0.5), -100, 0, -1) == :general
        @test ae(:dist, spec("DK"; g0 = 0.4), -101, 0, -1) == :classical
        # the quantiles: cases DU (0 <= c <= 1.5) and DT (c <= 2) by the general engine
        @test ae(:icdf, spec("DU"), 1.5, 0, 0) == :general && ae(:icdf, spec("DU"), 2, 0, 0) == :classical && ae(:icdf, spec("DU"), -1, 0, 0) == :classical
        @test ae(:icdf, spec("DT"), 2, 0, 0) == :general && ae(:icdf, spec("DT"), 2.5, 0, 0) == :classical
        # the moments with c0 != 0: the classical engine below c = -5 (case DT: from c < -5), the general nearer the unit root
        @test ae(:mom, spec("SK"), -5, -5, 0) == :classical && ae(:mom, spec("SK"), -4, -4, 0) == :general && ae(:mom, spec("ST"), -5, 2, 0) == :classical
        @test ae(:mom, spec("ST"), -1, -1, 0) == :general && ae(:mom, spec("SK"), -50, 0, 0) == :general
        @test ae(:mom, spec("DT"), -5.5, -5.5, 0) == :classical && ae(:mom, spec("DT"), -5, 2, 0) == :general && ae(:mom, spec("DT"), 0, -2, 0) == :general
        # the linear form of case DG: the classical code at and below the unit root, the general engine above it (cbar <= -1)
        lg = MD.lin_spec("DG", -13.5)
        @test ae(:lincdf, lg, 0, 0, -7.3) == :classical && ae(:lincdf, lg, -50, 0, -2.0) == :classical && ae(:lincdf, lg, 0.5, 0, -7.3) == :general
        @test ae(:lincdf, MD.lin_spec("DG", -0.5), 1, 0, -2.0) == :classical && ae(:lincdf, MD.lin_spec("DG", -0.5), 4.5, 0, -2.0) == :general
        # the parabola of the t-ratio at c0 = 0: the general engine (x = 1 marks the parabola)
        @test ae(:region, spec("DK"), -7, 0, 1) == :general && ae(:region, spec("DK"), -7, -2, 1) == :classical && ae(:region, spec("DK"), -7, 0, 0) == :classical
        @test ae(:region, spec("DK"), 0, 0, 1) == :general && ae(:region, spec("DK"), 2, 0, 1) == :classical
        @test MD.region_rule_args((type = "parabola", x = -1.9)) == (1.0, 0.0) && MD.region_rule_args((type = "parabola", x = -1.9, v0 = 0.1)) == (0.0, 0.0)
        @test MD.region_rule_args((type = "line", a = -3.5, b = 0.1)) == (0.0, 0.0) && MD.region_rule_args(Dict("type" => "parabola", "x" => 1.0, "c0" => 2)) == (1.0, 2.0)
        # the general engine's method :auto above the unit root (DFDet.line_auto): the paths for 0 < c <= 1.5 with one
        # regressor and 0 < c <= 2 with two or more, the line integral otherwise
        la(D, c) = GEM.line_auto(D, Float64(c), 0.0)
        @test !la(trend(0), 1.5) && la(trend(0), 1.6) && la(trend(0), 0.0) && !la(trend(1), 2.0) && la(trend(1), 2.1) && !la(trend(3), 0.5)
        @test la(trend(-1), 0.5) && la(trend(-1), 2.0) && !GEM.line_auto(trend(1), -5.0, -2.0)
        @test !la(detrend_first(trend(1)), 2.0)
    end

    x = [-3.0, -2.0, -1.0]
    @testset "a call takes the engine of its cell" begin
        @test ltratio_cdf(x, "DU", -5) == ltratio_cdf(x, "DU", -5; engine = :classical)
        @test ltratio_pdf(x, "DT", -300; c0 = -300) == ltratio_pdf(x, "DT", -300; c0 = -300, engine = :general)
        @test ltratio_cdf(x, "SU", -10; c0 = -10) == ltratio_cdf(x, "SU", -10; c0 = -10, engine = :general)
        # (a tail: the classical conditional integral, also where the classical rule takes the line integral, DFExact.stat_line)
        @test ltratio_cdf(-9.0, "SU", -1.2; c0 = 1.2) == ltratio_cdf(-9.0, "SU", -1.2; c0 = 1.2, engine = :classical, method = :double)
        @test ltratio_cdf(x, "DU", -5; c0 = 2.0, method = :single) == ltratio_cdf(x, "DU", -5; c0 = 2.0, engine = :classical, method = :single)
        @test ltratio_cdf(x, "SU", -10; c0 = -10, verified = true) == ltratio_cdf(x, "SU", -10; c0 = -10, verified = true, engine = :classical)
        @test ltratio_mom(4, "DT", -5; method = :engine) == ltratio_mom(4, "DT", -5; engine = :general)
        @test ltratio_mom(4, "DT", -100; c0 = -100, method = :engine) == ltratio_mom(4, "DT", -100; c0 = -100, engine = :classical)
        @test ltratio_mom(4, "DU", -100; c0 = -100, method = :engine) == ltratio_mom(4, "DU", -100; c0 = -100, engine = :classical)
        @test ltratio_icdf(0.05, "DT", -5) == ltratio_icdf(0.05, "DT", -5; engine = :general)
        @test ltratio_icdf(0.05, "DK") == ltratio_icdf(0.05, "DK"; engine = :classical)
        @test ltratio_lincdf([0.1, 0.3], NaN, "DK", -5) == ltratio_lincdf([0.1, 0.3], NaN, "DK", -5; engine = :general)
        @test ltratio_lincdf(-0.2639, -3.5, "DK", 0; method = :engine) == ltratio_lincdf(-0.2639, -3.5, "DK", 0; engine = :classical)
        @test ltratio_lincdf([-0.5, 0.0], -7.3, "DG", -1) == ltratio_lincdf([-0.5, 0.0], -7.3, "DG", -1; engine = :classical)
        @test ltratio_lincdf([-0.5, 0.0], -7.3, "DG", 1) == ltratio_lincdf([-0.5, 0.0], -7.3, "DG", 1; engine = :general)
        @test maximum(abs, ltratio_lincdf([-0.5, 0.0], -7.3, "DG", -1; engine = :general) - ltratio_lincdf([-0.5, 0.0], -7.3, "DG", -1; engine = :classical)) < 1e-10
        @test adfmax_cdf([-3.0, -2.4], "SU", -10) == adfmax_cdf([-3.0, -2.4], "SU", -10; engine = :general)
    end

    @testset "the moved cells of November 26, 2026: the default against the reference" begin
        # (the one evaluation of the engine the rule now takes, against the other engine's checked computation, verified = true;
        # the measurements behind the rule on grids of x from the 0.01% to the 99.99% point: scratch/rules/acc.jl)
        Fc(x, cas, c; kw...) = ltratio_cdf(x, cas, c; engine = :classical, verified = true, kw...)
        fc(x, cas, c; kw...) = ltratio_pdf(x, cas, c; engine = :classical, verified = true, kw...)
        for (cas, c, c0, g0) in (("DU", -100.0, 3.0, 0.0), ("DT", 2.0, 1.0, 0.0), ("DK", 0.0, -2.0, 0.0), ("DK", 1.0, -2.0, 0.0),
                                 ("DT", -20.0, -20.0, 0.0), ("DU", 1.0, 0.0, 0.0), ("DT", 2.0, 0.0, 0.0), ("DK", -30.0, 0.0, 1.0))
            # (the 1%, 50% and 99% points by the classical engine)
            xs = ltratio_icdf([0.01, 0.5, 0.99], cas, c; c0, g0, engine = :classical)
            @test ltratio_cdf(xs, cas, c; c0, g0) == ltratio_cdf(xs, cas, c; c0, g0, engine = :general)
            # (the bar of a move: within 1e-7; the classical checked computation at c = -100 with c0 > 0 is itself off by up
            # to 4e-8, scratch/rules/FINDINGS.md)
            @test maximum(abs, ltratio_cdf(xs, cas, c; c0, g0) .- Fc(xs, cas, c; c0, g0)) < 1e-7
            f0 = fc(xs, cas, c; c0, g0)
            @test maximum(abs, ltratio_pdf(xs, cas, c; c0, g0) .- f0) < 1e-7 * max(1.0, maximum(f0))
        end
        for (cas, c) in (("DU", 1.0), ("DT", 2.0))
            @test maximum(abs, ltratio_icdf([0.05, 0.5], cas, c) .- ltratio_icdf([0.05, 0.5], cas, c; engine = :classical)) < 1e-8
        end
        # the moments with c0 != 0 far from the unit root: the classical engine, against the moments from jets
        @test ltratio_mom(4, "SK", -20; c0 = -20, method = :engine) == ltratio_mom(4, "SK", -20; c0 = -20, engine = :classical)
        let mj = ltratio_mom(4, "ST", -20; c0 = -20, method = :jets, tol = 1e-13), me = ltratio_mom(4, "ST", -20; c0 = -20, method = :engine)
            @test maximum(abs.(me .- mj) ./ [max(abs(mj[j]), abs(mj[2])^(j / 2)) for j in 1:4]) < 1e-8
        end
        # the linear form of case DG below the unit root: the classical code, against the general engine's checked computation
        let bs = [-0.9, -0.5, 0.0, 0.5]
            @test ltratio_lincdf(bs, -7.3, "DG", -5) == ltratio_lincdf(bs, -7.3, "DG", -5; engine = :classical)
            @test maximum(abs, ltratio_lincdf(bs, -7.3, "DG", -5) .- GEM.lin_cdf(trend(1), bs, -7.3, -5.0; gls = -13.5, verified = true)) < 1e-10
        end
        # the parabola of the t-ratio at c0 = 0: the general engine, against the classical checked t-ratio (at the 5% point)
        for c in (0.0, -7.0)
            xq = ltratio_icdf(0.05, "DK", c; engine = :classical); bp = (type = "parabola", x = xq)
            @test ltratio_region(bp, "DK", c) == ltratio_region(bp, "DK", c; engine = :general)
            @test abs(ltratio_region(bp, "DK", c) - Fc(xq, "DK", c)) < 1e-10
        end
        # the general engine's method :auto above the unit root: the paths, within 1e-9 of the line integral (at the 1%, 50%
        # and 99% points by the line integral)
        for (D, c) in ((trend(0), 1.0), (trend(2), 2.0), (detrend_first(trend(1)), 1.5))
            xs = [dist_quantile(D, p, c; x0 = -2.0, method = :line) for p in (0.01, 0.5, 0.99)]
            @test dist_cdf(D, xs, c) == dist_cdf(D, xs, c; method = :paths)
            @test maximum(abs, dist_cdf(D, xs, c) .- dist_cdf(D, xs, c; method = :line)) < 1e-9
        end
        @test dist_cdf(trend(0), [-2.0], 2.0) == dist_cdf(trend(0), [-2.0], 2.0; method = :line)
        # the stationary start's quantiles at c0 = 0: cases SU, ST from c = -0.5 on by the general engine
        @test MD.autoengine(:icdf, spec("SU"), -0.5, 0.0, 0.0) == :general && MD.autoengine(:icdf, spec("SU"), -0.4, 0.0, 0.0) == :classical
        @test abs(ltratio_icdf(0.05, "SU", -1.0) - ltratio_icdf(0.05, "SU", -1.0; engine = :classical)) < 1e-8
    end

    @testset "the brittle region far above the unit root (c > 5.5) warns" begin
        # (designs beyond the classical cases and a fixed start with deterministic terms, where the paths compute: one
        # evaluation was off by more than 1e-5 in the upper tail from c = 6 on; November 26, 2026)
        @test_logs (:warn, r"far above the unit root") match_mode = :any dist_cdf(perron(:C, 0.3), [0.0], 6.0)
        @test_logs min_level = Logging.Warn dist_cdf(perron(:C, 0.3), [0.0], 5.0)
        @test_logs (:warn, r"far above the unit root") match_mode = :any ltratio_cdf(5.0, trend(1), 6.0; g0 = 1.0)
        @test_logs min_level = Logging.Warn ltratio_cdf(5.0, trend(1), 5.0; g0 = 1.0)
        @test_logs min_level = Logging.Warn dist_cdf(trend(2), [0.0], 6.0)      # (the line integral computes)
        @test_logs min_level = Logging.Warn ltratio_cdf(0.0, "DT", 6.0)          # (the classical engine computes)
    end

    @testset "the rules and the extensions together: one warning per call, the reflection by the rule" begin
        # (the counts of each kind of warning in a call)
        kinds(f) = (r = Test.collect_test_logs(f; min_level = Logging.Warn)[1];
                    (far = count(l -> occursin("far above", string(l.message)), r), line = count(l -> occursin("vertical line", string(l.message)), r), all = length(r)))
        PC = perron(:C, 0.3); TR = fdesign(r -> [1.0, sin(2pi * r), cos(2pi * r)]; name = "trig")
        # far above the unit root: once in a call that evaluates many times (the quantiles of a vector, their Newton steps;
        # the normal average of rho, which calls the t-ratio's distribution function at its points), and once per call
        @test kinds(() -> dist_quantile(PC, [0.5, 0.6], 6.0; x0 = 1.0)) == (far = 1, line = 0, all = 1)
        @test kinds(() -> ltratio_icdf([0.5, 0.6], PC, 6.0)) == (far = 1, line = 0, all = 1)
        @test kinds(() -> dist_cdf(PC, [0.0, 1.0], 6.0; rho = 0.995)) == (far = 1, line = 1, all = 2)
        @test GEM.once_per_call(() -> (GEM.warn_first(:k), GEM.warn_first(:k), GEM.warn_first(:j))) == (true, false, true)
        @test (GEM.warn_first(:k), GEM.warn_first(:k)) == (true, true)                       # (outside a call: every time)
        # the line integral of c0 > 0 on the vertical line computes the points x < 0, the paths the others: the warning
        # comes where the paths compute
        @test kinds(() -> dist_cdf(TR, [-3.0, 1.0], 6.0; c0 = 2.0)) == (far = 1, line = 0, all = 1)
        @test kinds(() -> dist_cdf(TR, [-3.0], 6.0; c0 = 2.0)) == (far = 0, line = 0, all = 0)
        # rho = -1 for a classical case: the t-ratio's own functions at -x by the engine of the rule (here the classical
        # engine), within 1e-10 of the general engine's reflection
        x = [-1.0, 0.5, 2.0]
        for (cas, c, c0) in (("DU", -10.0, -10.0), ("DK", -7.0, -3.0), ("DT", -60.0, -5.0))
            S = spec(cas)
            @test MD.autoengine(:dist, S, c, c0, -0.5) == :classical
            @test ltratio_cdf(x, cas, c; rho = -1, c0) == 1 .- ltratio_cdf(-x, cas, c; c0)
            @test ltratio_pdf(x, cas, c; rho = -1, c0) == ltratio_pdf(-x, cas, c; c0)
            @test maximum(abs, ltratio_cdf(x, cas, c; rho = -1, c0) .- dist_cdf(S.D, x, c; rho = -1.0, c0, start = S.start)) < 1e-10
            @test ltratio_icdf([0.05, 0.9], cas, c; rho = -1, c0) == -ltratio_icdf(1 .- [0.05, 0.9], cas, c; c0)
        end
        @test ltratio_cdf(x, "DU", -10; rho = -1, c0 = -10, engine = :general) == dist_cdf(trend(0), x, -10.0; rho = -1.0, c0 = -10.0)
        @test occursin("GLS detrending", errm(() -> ltratio_cdf(x, "DG", -5; rho = -1)))     # (refused, as by the general engine)
    end

    @testset "far from the unit root: the merged default against the stored values" begin
        # (test/engine/largec_ref.txt, the classical engine's values, which test_dfexact.jl holds it to 1e-10).  The moments of
        # the merged default come from the general engine at c0 = 0 and with a stationary start: within 1e-10 except with
        # c0 = c and a stationary start, where both engines move by 1e-8 to 5e-7 relative under refinement of their rules
        # and differ by up to 4.1e-9 with Accelerate or IntelVectorMath and by 5.4e-9 with the plain loops (Linux and
        # Windows without a vector library, where the sums round differently): 2e-8 there; with c0 = c and a zero start
        # below c = -100 both move by up to 1e-10 and differ by up to 5.2e-10 (the general engine takes case DT): 5e-10
        for l in eachline(joinpath(@__DIR__, "engine", "largec_ref.txt"))
            p = split(l); p[1] == "mom" || continue
            cas = String(p[2]); c, c0 = parse.(Float64, p[3:4]); mr = parse.(Float64, p[5:8])
            tol = (cas[1] == 'S' && c0 != 0) ? 2e-8 : (c0 != 0 && c <= -200) ? 5e-10 : 1e-10
            @test maximum(abs.(quietly(() -> ltratio_mom(4, cas, c; c0, method = :engine)) ./ mr .- 1)) < tol
        end
    end

    @testset "the engines forced against each other" begin
        for (cas, c, c0, g0, xs) in (("DK", 0.0, 0.0, 0.0, [-2.5, -1.9, -1.0, 0.0, 1.0]), ("DU", -10.0, 0.0, 0.0, [-4.5, -3.5, -2.5, -1.5]),
                                     ("DT", -5.0, -5.0, 0.0, [-1.5, -0.5, 0.5, 1.5]), ("SU", -10.0, -10.0, 0.0, [-1.5, -0.5, 0.5, 1.5]),
                                     ("DK", 0.0, 0.0, 1.0, [-2.0, -1.0, 0.0, 1.0]), ("DT", -300.0, -300.0, 0.0, [-1.5, 0.0, 1.5]),
                                     ("DU", -5.0, 2.0, 0.0, [-6.0, -4.0, -3.0]), ("DU", -10.0, 0.0, 1.0, [-4.0, -3.0, -2.0]), ("DG", -10.0, 0.0, 0.0, [-4.5, -3.0, -1.5]))
            for f in (ltratio_cdf, ltratio_pdf)
                @test maximum(abs, f(xs, cas, c; c0, g0, engine = :classical) - f(xs, cas, c; c0, g0, engine = :general)) < 1e-8
            end
        end
        for (cas, c) in (("DK", -5.0), ("DT", -5.0), ("SU", -5.0))
            @test maximum(abs.(ltratio_mom(4, cas, c; engine = :general) ./ ltratio_mom(4, cas, c; engine = :classical) .- 1)) < 1e-10
        end
        @test abs(ltratio_icdf(0.05, "DU"; engine = :general) - ltratio_icdf(0.05, "DU"; engine = :classical)) < 1e-9
        @test abs(coef_cdf(-14.094, "DU"; engine = :general) - coef_cdf(-14.094, "DU"; engine = :classical)) < 1e-9
        @test maximum(abs, ltratio_lincdf([-0.6, -0.2639, 0.3], -3.5, "DK", 0; engine = :general) - ltratio_lincdf([-0.6, -0.2639, 0.3], -3.5, "DK", 0; engine = :classical)) < 1e-10
        @test maximum(abs, adfmax_cdf([-3.0, -2.4], "DU", 0; engine = :general) - adfmax_cdf([-3.0, -2.4], "DU", 0; engine = :classical)) < 1e-10
        @test abs(ltratio_region((type = "line", a = -3.5, b = -0.2639), "DK", -5; engine = :general) -
                  ltratio_region((type = "line", a = -3.5, b = -0.2639), "DK", -5; engine = :classical)) < 1e-10
        @test abs(power_envelope(-10.0, "DK"; engine = :general) - power_envelope(-10.0, "DK"; engine = :classical)) < 1e-8
    end

    @testset "the classical functions with a design" begin
        @test ltratio_cdf(x, trend(0), -5) == ltratio_cdf(x, "DU", -5)                       # a classical design is its case
        @test ltratio_pdf(x, trend(1), -10; gls = -13.5) == ltratio_pdf(x, "DG", -10)
        @test ltratio_cdf(x, trend(1), -10; start = :stationary) == ltratio_cdf(x, "ST", -10)
        @test ltratio_cdf(x, trend(-1), -5; g0 = 1.5) == ltratio_cdf(x, "DK", -5; g0 = 1.5)
        # beyond the classical engine's checks a design goes to the general engine; the case string keeps the check
        @test quietly(() -> ltratio_cdf(-2.0, trend(-1), 0; g0 = 2.5)) == quietly(() -> dist_cdf(trend(-1), [-2.0], 0.0; start = :fixed, gamma0 = 2.5)[1])
        @test errm(() -> ltratio_cdf(-2.0, "DK", 0; g0 = 2.5)) == DXM.FIXED_MSG
        D = perron(:C, 0.3)
        @test ltratio_cdf(x, D, -5) == dist_cdf(D, x, -5.0) && ltratio_pdf(-2.0, D, -5) == dist_pdf(D, [-2.0], -5.0)[1]
        @test ltratio_cdf(x, D, -5; c0 = -5) == dist_cdf(D, x, -5.0; c0 = -5.0)
        @test abs(ltratio_icdf(0.05, D) - dist_quantile(D, 0.05; x0 = -4.0)) < 1e-8
        @test ltratio_mom(2, D, -5) == dist_moments(D, -5.0; p = 2)[1]
        @test coef_cdf([-20.0], D) == GEM.coef_cdf(D, [-20.0], 0.0)
        @test abs(coef_icdf(0.05, D) - coef_quantile(D, 0.05)) < 1e-8
        @test adfmax_cdf(-3.0, D) == GEM.adfmax_cdf(D, [-3.0], 0.0)[1]
        @test ltratio_lincdf(-0.3, -3.5, D, -5) == lin_cdf(D, [-0.3], -3.5, -5.0)[1]
        @test ltratio_region((type = "parabola", x = -3.0), D, -5) == dist_cdf(D, [-3.0], -5.0)[1]
        @test ltratio_cdf(x, D, -5; start = :stationary) == dist_cdf(D, x, -5.0; start = :stationary)
        @test ltratio_cdf(-2.0, detrend_first(perron(:B, 0.5)), -5) == dist_cdf(detrend_first(perron(:B, 0.5)), [-2.0], -5.0)[1]
        @test power_envelope(-10.0, trend(0)) == power_envelope(-10.0, "DK") && power_envelope(-10.0, trend(1)) == power_envelope(-10.0, "DG")
        @test round(power_envelope(-10.0, perron(:C, 0.5)), digits = 4) == 0.1789                # paper/envelope.txt of df-deterministics
        # the argument errors of the interface
        @test occursin("the classical engine has the cases", errm(() -> ltratio_cdf(-2.0, D, 0; engine = :classical)))
        @test occursin("does not apply with engine = :general", errm(() -> ltratio_cdf(-2.0, "DU", 0; method = :double, engine = :general)))
        @test occursin("engine must be", errm(() -> ltratio_cdf(-2.0, "DU", 0; engine = :fast)))
        @test occursin("exclude each other", errm(() -> ltratio_cdf(-2.0, D, -5; g0 = 1.0, start = :stationary)))
    end

    @testset "regions and the linear form without deterministic terms, with a fixed start" begin
        # (the classical regions have the zero start only, and the fixed start was dropped: the zero start's 0.285 for
        # 0.975 below).  The parabola is the t-ratio, which the classical engine has with the fixed start
        bp = (type = "parabola", x = -2.0); bl = (type = "line", a = -3.5, b = -0.2639)
        @test abs(ltratio_region(bp, trend(-1), -5; g0 = 1.5) - ltratio_cdf(-2.0, "DK", -5; g0 = 1.5, engine = :classical)) < 1e-10
        Fl = lin_cdf(trend(-1), [-0.2639], -3.5, -5.0; start = :fixed, gamma0 = 1.5)[1]
        @test ltratio_region(bl, trend(-1), -5; g0 = 1.5) == Fl && abs(Fl - ltratio_region(bl, "DK", -5)) > 0.1
        @test occursin("case DK only", errm(() -> ltratio_region(bl, trend(-1), -5; g0 = 1.5, engine = :classical)))
        # the linear form likewise
        @test ltratio_lincdf(-0.2639, -3.5, trend(-1), -5; g0 = 1.5, method = :engine) == Fl
        @test abs(ltratio_lincdf(-0.2639, -3.5, trend(-1), -5; g0 = 1.5) - Fl) < 1e-10      # (the single inversion of Bao and Kan: 5.4e-12)
        @test occursin("cases DK and DG only", errm(() -> ltratio_lincdf(-0.2639, -3.5, trend(-1), -5; g0 = 1.5, engine = :classical)))
    end

    @testset "the quantile searches: the start from the moments from jets (mom_start)" begin
        # the quantiles of before to the search tolerance (2.4e-11 measured)
        m = MD.mom_start(2, "DU", -5.0; c0 = -2.0)
        @test m !== nothing && maximum(abs.(m ./ ltratio_mom(2, "DU", -5; c0 = -2.0, method = :engine) .- 1)) < 1e-9
        @test MD.mom_start(2, "DG", -5.0) === nothing && MD.mom_start(2, "DU", 25.0) === nothing
        for (cas, c, c0) in (("DU", -5.0, 0.0), ("DT", -5.0, -2.0), ("DK", -5.0, 2.0))
            q1 = ltratio_icdf([0.05, 0.5], cas, c; c0)
            MD.MOM_START[] = false
            q0 = try; ltratio_icdf([0.05, 0.5], cas, c; c0); finally; MD.MOM_START[] = true; end
            @test maximum(abs, q1 .- q0) < 1e-9
        end
    end

    @testset "the quantile searches: F and f in one evaluation where the general engine computes (route_dist2)" begin
        # the values of ltratio_cdf and ltratio_pdf bit for bit, and so the quantiles of the two evaluations
        for (cas, c, c0, g0) in (("DU", -5.0, 2.0, 0.0), ("SU", -5.0, -5.0, 0.0), ("DK", -5.0, 0.0, 1.0))
            Ff = MD.route_dist2(-2.5, cas, c; c0, g0)
            @test Ff !== nothing && Ff[1] == ltratio_cdf(-2.5, cas, c; c0, g0) && Ff[2] == ltratio_pdf(-2.5, cas, c; c0, g0)
            q1 = ltratio_icdf([0.05, 0.5], cas, c; c0, g0)
            MD.DIST2[] = false
            q0 = try; ltratio_icdf([0.05, 0.5], cas, c; c0, g0); finally; MD.DIST2[] = true; end
            @test q1 == q0
        end
        @test MD.route_dist2(-2.5, "DU", -5.0; c0 = 0.0) === nothing        # (the classical engine's point)
    end
end
