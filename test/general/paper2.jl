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
    # the limit computed holds only for trend functions whose span contains the constant (a referee: false otherwise); an
    # ArgumentError from each function of the statistic: e^r alone, r with e^r, r alone, a level dummy with r, none
    lmerr(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end
    for D in (fdesign(r -> [exp(r)]; name = "e^r"), fdesign(r -> [r, exp(r)]; name = "r, e^r"), fdesign(r -> [r]; name = "r"),
              fdesign(r -> [r > 0.4 ? 1.0 : 0.0, r], [0.4]; name = "a level dummy and r"), trend(-1),
              fdesign(r -> [exp(r), exp(-r)]; name = "e^r, e^-r"))
        @test !GE.span_has_constant(D)
        for f in (() -> lm_cdf(D, [-2.0]), () -> lm_quantile(D, 0.05), () -> lm_power(D, [-5.0]))
            @test occursin("span contains the constant", lmerr(f))
        end
    end
    # the span contains it: polynomial trends, breaks, the Legendre basis, functions of which only a combination is
    # constant (to the interpolants' tolerance), a design from a matrix (decided on values: the high-order local Taylor
    # coefficients of a function design carry rounding up to 1e-2, the trigonometric design's 1.3e-2, which cancels in the
    # values); and then the statistic is that of the span
    for D in (trend(0), trend(1), trend(3), perron(:A, 0.5), perron(:C, [0.3, 0.7]), ltrend(4),
              fdesign(r -> [sin(pi * r)^2, cos(pi * r)^2, r]; name = "sin^2, cos^2, r"), fdesign(r -> [1.0 + r, r]; name = "1 + r, r"),
              fdesign(r -> [1.0, sin(2pi * r), cos(2pi * r)]; name = "an intercept with a sine and a cosine"),
              fdesign(r -> [sin(4pi * r)^2, cos(4pi * r)^2]; name = "sin^2, cos^2 of 4 pi r"),
              design_from_matrix(hcat(ones(200), (1:200) ./ 200, ((1:200) ./ 200) .^ 2)))
        @test GE.span_has_constant(D)
    end
    @test abs(lm_cdf(fdesign(r -> [1.0 + r, r]; name = "1 + r, r"), [-3.0230])[1] - lm_cdf(trend(1), [-3.0230])[1]) < 1e-10
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
    # the point-optimal statistic where the grid of the inversion of Gil-Pelaez does not resolve the transform (F from -0.1
    # to 3.4 at c = 5, above c = 4, where the grid stays the default), and at points beyond the range of the grid (F = 3 at
    # q = 1e5 at c = 0; the single inversion, the default at c = 0, has no such range): an error; with the N the error asks
    # for, c = 5 within Monte Carlo error of the simulated limit (P(Q <= 73.28) = 0.05, 1e6 replications, n = 4000)
    @test_throws ArgumentError pt_cdf(trend(1), [100.0], -13.5, 5.0)
    @test abs(pt_cdf(trend(1), [73.28], -13.5, 5.0; N = 1_000_000)[1] - 0.05) < 1e-3
    @test_throws ArgumentError pt_cdf(trend(1), [1e5], -13.5, 0.0; method = :grid)
    @test 0 <= 1 - pt_cdf(trend(1), [1e5], -13.5, 0.0)[1] < 1e-12
    @test pt_cdf(trend(1), [1e3, Inf, -Inf], -13.5, 0.0) ≈ [1.0, 1.0, 0.0] atol = 1e-9
end
