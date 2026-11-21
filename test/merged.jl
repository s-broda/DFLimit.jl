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
        @test ae(:dist, spec("DU"), -100, 3, -3) == :classical && ae(:dist, spec("DK"), -5, 2, -3) == :classical
        @test ae(:dist, spec("DT"), -300, -300, -1) == :general             # zero start, c0 != 0, c < -100
        @test ae(:dist, spec("SU"), -10, -10, -1) == :general               # stationary start, c0 != 0, c <= -0.1
        @test ae(:dist, spec("SK"), -1e-4, -1e-4, 0.5) == :classical        # next to the unit root
        @test ae(:dist, spec("DK"), 5, -3, 20) == :classical                # c0 < 0 above the unit root (the general engine off by 7e-3)
        @test ae(:dist, spec("DT"), 7, 0, 200) == :classical                # the bulk above the unit root
        @test ae(:dist, spec("DK"; g0 = 1.0), 0, 0, -1) == :general         # fixed start, case DK, (c g0)^2/2 <= 400
        @test ae(:dist, spec("DK"; g0 = 2.0), -20, 0, -1) == :classical
        @test ae(:dist, spec("DK"; g0 = 1.0), 2, 0, -1) == :classical
        @test ae(:dist, spec("DU"; g0 = 1.0), -10, 0, -1) == :classical     # fixed start, cases DU, DT
        @test ae(:dist, spec("DG"), -10, 0, -2) == :classical
        # the other functions
        @test ae(:icdf, spec("DT"), -5, 0, 0) == :general && ae(:icdf, spec("ST"), -10, 0, 0) == :general
        @test ae(:icdf, spec("DU"), 0, 0, 0) == :general && ae(:icdf, spec("DU"), -10, 0, 0) == :classical
        @test ae(:icdf, spec("DK"), 0, 0, 0) == :classical && ae(:icdf, spec("DG"), 0, 0, 0) == :classical
        @test ae(:icdf, spec("DU"; g0 = 1.0), -5, 0, 0) == :general && ae(:icdf, spec("DU"; g0 = 0.5), -20, 2, 0) == :classical
        @test ae(:mom, spec("DT"), -5, 0, 0) == :general && ae(:mom, spec("SK"), -300, -300, 0) == :classical
        lk = MD.lin_spec("DK", -13.5); lg = MD.lin_spec("DG", -13.5)
        @test ae(:lincdf, lk, 0, 0, NaN) == :general && ae(:lincdf, lk, 0, 0, 0.0) == :general && ae(:lincdf, lk, 0, 0, -3.5) == :classical
        @test ae(:lincdf, lk, 5, 0, 0.0) == :classical && ae(:lincdf, lk, 5, 0, NaN) == :classical    # (the general lin_cdf is wrong at c = 5)
        @test ae(:lincdf, lg, 0, 0, -7.3) == :classical && ae(:lincdf, lg, 0, 0, 0.1) == :classical
        @test ae(:adfmax, spec("SU"), -10, 0, 0) == :classical && ae(:adfmax, spec("DU"), 0, 0, 0) == :classical
        @test ae(:envelope, spec("DK"), 0, 0, 0) == :classical && ae(:region, spec("DK"), -5, 0, 0) == :classical
    end

    x = [-3.0, -2.0, -1.0]
    @testset "a call takes the engine of its cell" begin
        @test ltratio_cdf(x, "DU", -5) == ltratio_cdf(x, "DU", -5; engine = :classical)
        @test ltratio_pdf(x, "DT", -300; c0 = -300) == ltratio_pdf(x, "DT", -300; c0 = -300, engine = :general)
        @test ltratio_cdf(x, "SU", -10; c0 = -10) == ltratio_cdf(x, "SU", -10; c0 = -10, engine = :general)
        @test ltratio_cdf(-9.0, "SU", -1.2; c0 = 1.2) == ltratio_cdf(-9.0, "SU", -1.2; c0 = 1.2, engine = :classical)   # a tail: classical
        @test ltratio_cdf(x, "DU", -5; c0 = 2.0, method = :single) == ltratio_cdf(x, "DU", -5; c0 = 2.0, engine = :classical, method = :single)
        @test ltratio_cdf(x, "SU", -10; c0 = -10, verified = true) == ltratio_cdf(x, "SU", -10; c0 = -10, verified = true, engine = :classical)
        @test ltratio_mom(4, "DT", -5) == ltratio_mom(4, "DT", -5; engine = :general)
        @test ltratio_mom(4, "DT", -100) == ltratio_mom(4, "DT", -100; engine = :classical)
        @test ltratio_icdf(0.05, "DT", -5) == ltratio_icdf(0.05, "DT", -5; engine = :general)
        @test ltratio_icdf(0.05, "DK") == ltratio_icdf(0.05, "DK"; engine = :classical)
        @test ltratio_lincdf([0.1, 0.3], NaN, "DK", -5) == ltratio_lincdf([0.1, 0.3], NaN, "DK", -5; engine = :general)
        @test ltratio_lincdf(-0.2639, -3.5, "DK", 0) == ltratio_lincdf(-0.2639, -3.5, "DK", 0; engine = :classical)
        @test adfmax_cdf([-3.0, -2.4], "SU", -10) == adfmax_cdf([-3.0, -2.4], "SU", -10; engine = :classical)
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
        @test occursin("does not apply with engine = :general", errm(() -> ltratio_cdf(-2.0, "DU", 0; method = :single, engine = :general)))
        @test occursin("engine must be", errm(() -> ltratio_cdf(-2.0, "DU", 0; engine = :fast)))
        @test occursin("exclude each other", errm(() -> ltratio_cdf(-2.0, D, -5; g0 = 1.0, start = :stationary)))
    end
end
