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
end
