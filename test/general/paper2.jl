# The statistics of the paper on general deterministic terms as functions (src/DFDet_paper2.jl) against the outputs of the
# scripts they come from, paper/envelope.txt and paper/lmtest.txt of the repository df-deterministics (the digits printed
# there), and the envelope of the general construction against the engine's for the intercept and the linear trend.
using DFLimit, Test
const GE = DFLimit.DFDet

@testset "paper 2: point-optimal statistic, envelope, LM, MZ_alpha, MSB" begin
    # the envelope of the general construction and the engine's (no deterministic terms needs none: Q of trend(-1) is the
    # line U - aV), envelope.txt: c = -5, -10, -20
    cs = [-5.0, -10.0, -20.0]
    for (D, ref) in ((trend(0), [0.31947655, 0.75818325, 0.99884511]), (trend(1), [0.11484596, 0.31005163, 0.85525722]))
        e = pt_envelope(D, cs)
        @test maximum(abs, e - ref) < 6e-9
        @test maximum(abs, e - GE.power_envelope(D, cs)) < 1e-8
    end
    # the break in the slope (Carrion-i-Silvestre, Kim and Perron), envelope.txt: lambda, cbar, P_T, MSB, MZ_alpha, the envelope at cbar
    for (lam, cb, pt5, msb5, mza5, env) in ((0.1, -16.0, 6.0936, 0.14783, -22.387, 0.4741), (0.5, -17.8, 6.9246, 0.14427, -23.659, 0.4977))
        D = perron(:B, lam)
        @test round(pt_quantile(D, 0.05, cb), digits = 4) == pt5
        @test round(msb_quantile(D, 0.05; gls = cb), digits = 5) == msb5
        @test round(mza_quantile(D, 0.05; gls = cb, y0 = -25.0), digits = 3) == mza5
        @test round(pt_envelope(D, [cb])[1], digits = 4) == env
    end
    # the size of the simulated values at their critical value (envelope.txt: P_T 6.789 at 0.5 has size 0.0455, MSB 0.144 0.0491)
    @test round(pt_cdf(perron(:B, 0.5), [6.789], -17.8)[1], digits = 4) == 0.0455
    @test round(msb_cdf(perron(:B, 0.5), [0.144]; gls = -17.8)[1], digits = 4) == 0.0491
    # the LM statistic, lmtest.txt: the 1%, 5%, 10% values and the power of the 5% test of the linear trend
    for (D, q) in ((trend(1), [-3.5617, -3.0230, -2.7501]), (trend(2), [-4.0594, -3.5304, -3.2603]), (perron(:C, 0.5), [-4.1609, -3.6475, -3.3873]))
        @test round.([lm_quantile(D, p) for p in (0.01, 0.05, 0.10)], digits = 4) == q
    end
    q5, pw = lm_power(trend(1), [-5.0, -10.0, -15.0, -20.0])
    @test round.(pw, digits = 4) == [0.1059, 0.2656, 0.5149, 0.7599]
    @test lm_cdf(trend(1), [q5])[1] ≈ 0.05 atol = 1e-9
    @test lm_cdf(trend(1), [0.5, -Inf]) == [1.0, 0.0]
    @test_throws ArgumentError lm_quantile(trend(0), 0.05)              # no linear trend: J_0(1) does not vanish
    # the edge t_* of the LM statistic's transform: the first zero of D_V (the estimate took the orthogonal projection on the
    # design, whose largest eigenvalue is not that of the LM statistic's V: 231 for 139 with the linear trend at c = -20, and
    # above the unit root up to 2.3 times the edge, where the path at t_*/2 lay beyond it and failed)
    for D in (trend(1), trend(2), perron(:A, 0.3), perron(:B, 0.3), perron(:C, 0.5)), c in (-20.0, 2.0, 5.0)
        Mo = GE.model(GE.lmprep(D)); ts = GE.tstar(Mo, c)[1]
        W = GE.MWork(Mo); dv(t) = real(exp(GE.mgfparts(Mo, c, complex(t), W)[2]))
        @test all(t -> dv(t) > 0, range(0.01ts, 0.99ts; length = 200)) && dv(1.01ts) < 0
    end
    # with a level break (Perron's model A) the LM statistic has the law of the linear trend's (J_0 and Sbar are the same
    # functionals of J); at c = 3 and 4 the break at 0.3 failed (the path beyond t_*)
    xs = [-0.5, -0.2, -0.1, -0.05]
    for c in (3.0, 4.0)
        Ft = lm_cdf(trend(1), xs, c)
        @test maximum(abs, lm_cdf(perron(:A, 0.3), xs, c) - Ft) < 1e-10 && maximum(abs, lm_cdf(perron(:A, 0.5), xs, c) - Ft) < 1e-10
    end
    # above the unit root the LM statistic takes the path of V alone (lmpath): the vertical line of the t-ratio's path was
    # off by 6e-3 at c = 5 with a linear trend (F above 1); now a hyperbola, within 1e-13 of hyperbolas at other angles and
    # positions at c = 5 (1e-11 at c = 7) and within Monte Carlo error of the simulated limit (1.7 standard errors, 1e6
    # replications); beyond c = 10 an error
    P = GE.lmprep(trend(1)); Mo = GE.model(P)
    ys = 10 .^ range(-2, 4; length = 25); xs = -1 ./ (2 .* sqrt.(ys))
    for c in (5.0, 7.0)
        F = lm_cdf(trend(1), xs, c); CP = GE.lmpath(P, c)
        Fv = [GE.lmV(GE.cpath(Mo, c; phi = pi / 3, tau = 0.3 * CP.tstar), y) for y in ys]
        @test !isnan(CP.h) && maximum(abs, F - Fv) < 1e-10 && all(-1e-10 .<= F .<= 1) && all(diff(F) .>= -1e-14)
    end
    @test_throws ArgumentError lm_cdf(trend(1), [-0.1], 12.0)
    @test_throws ArgumentError lm_power(trend(1), [-5.0, 12.0])
end
