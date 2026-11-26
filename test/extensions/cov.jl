# A normal component (src/DFDet_cov.jl): T = rho tau_{c0} + sqrt(1 - rho^2) Z, the limit of the covariate-augmented
# t-ratio (Hansen 1995) and of the t-ratio of the slope in a predictive regression.  The limiting cases rho = 1 (the
# t-ratio itself), rho = -1, rho = 0 and the reflection for a negative rho; the inversion on two lines against the
# normal average; the reference values of the prototype (df-deterministics, extensions/covariates, CHECKS.md and
# table.txt: critical values at five decimals, powers at four) and of the companion paper's section on the predictive
# regression (its table of sizes at four decimals and its worked example, from the normal average of the classical
# engine's distribution function); the incomplete Gaussian integrals; argument errors and type stability.
using DFLimit, Test, QuadGK, Logging
const DD = DFLimit.DFDet
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end
Phi(x) = DD.normcdf_(x)

@testset "a normal component" begin
    @testset "the local null values" begin
        @test cadf_c0(0.0, 0.5, 0.3) == 0
        @test cadf_c0(-5.0, 1.0) == 0
        @test cadf_c0(-5.0, sqrt(0.5)) ≈ 5.0                       # -c (1 - rho^2)/rho^2 when R = rho
        @test cadf_c0(-5.0, sqrt(0.7), sqrt(0.3)) ≈ -5 * (1 - 1 / sqrt(0.21))
        @test cadf_c0(-5.0, -0.7) ≈ -5 * (1 + 1 / 0.49)             # (a negative rho, R = |rho|: c0 = -15.2)
        @test predictive_c0(-5.0, -0.95) == -5.0
        @test predictive_c0(-5.0, -0.95, 5.0) ≈ -5 + 5 / 0.95
        @test occursin("not be 0", errmsg(() -> cadf_c0(-5, 0.0)))
        @test occursin("[-1, 1]", errmsg(() -> cadf_c0(-5, 1.2)))
        @test occursin("R must be finite and positive", errmsg(() -> cadf_c0(-5, 0.5, -1.0)))
        @test occursin("not be 0", errmsg(() -> predictive_c0(-5, 0.0)))
    end
    @testset "the incomplete Gaussian integrals for any complex argument" begin
        for z in (0.3 + 0.2im, 2.0 - 5.0im, -0.4 + 0.1im, -1.5 - 2.0im, 0.0 + 3.0im, -0.01 - 7.0im)
            h = DD.hfull(ComplexF64(z))
            for m in 0:2
                v, _ = quadgk(u -> u^m * exp(-u^2 - 2z * u), 0, Inf; rtol = 1e-12)
                @test abs(h[m+1] - v) < 1e-9 * max(1, abs(v))
            end
        end
    end
    xs = [-4.0, -2.5, -1.0, 0.0, 1.5]
    @testset "the limiting cases" begin
        for D in (trend(-1), trend(0), perron(:A, 0.3)), c in (0.0, -10.0)
            @test dist_cdf(D, xs, c; rho = 1.0) == dist_cdf(D, xs, c)
            @test dist_pdf(D, xs, c; rho = 1) == dist_pdf(D, xs, c)
            @test dist_cdf(D, xs, c; rho = -1.0) == 1 .- dist_cdf(D, -xs, c)
            @test dist_pdf(D, xs, c; rho = -1.0) == dist_pdf(D, -xs, c)
            @test dist_cdf(D, xs, c; rho = 0.0) == Phi.(xs)
        end
        @test dist_quantile(trend(0), 0.05, 0.0; rho = 1.0) == dist_quantile(trend(0), 0.05, 0.0)
        @test dist_quantile(trend(0), 0.3, 0.0; rho = 0.0) ≈ -0.5244005127080407
        # a negative rho is a reflection
        F = dist_cdf(trend(0), xs; rho = 0.7); Fn = dist_cdf(trend(0), -xs; rho = -0.7)
        @test maximum(abs, Fn .- (1 .- F)) < 1e-12
        @test maximum(abs, dist_pdf(trend(0), -xs; rho = -0.7) .- dist_pdf(trend(0), xs; rho = 0.7)) < 1e-12
        @test abs(dist_quantile(trend(0), 0.95, -5.0; rho = -0.7, c0 = 3.0) + dist_quantile(trend(0), 0.05, -5.0; rho = 0.7, c0 = 3.0, x0 = -2.0)) < 1e-8
        # close to rho = 1 the lines approach the t-ratio's distribution function at the rate 1 - rho^2
        F1 = dist_cdf(trend(1), xs, 0.0)
        d1 = maximum(abs, dist_cdf(trend(1), xs; rho = sqrt(0.99), method = :lines) .- F1)
        d2 = maximum(abs, dist_cdf(trend(1), xs; rho = sqrt(0.999), method = :lines) .- F1)
        @test d2 < 2e-3 && 5 < d1 / d2 < 20
    end
    @testset "the two computations agree" begin
        # (F and f of the inversion on two lines and of the normal average, DD.cov_both: both from one evaluation)
        for (D, rho2) in ((trend(-1), 0.2), (trend(-1), 0.95), (trend(0), 0.2), (trend(1), 0.6), (perron(:A, 0.5), 0.95), (perron(:C, 0.3), 0.6))
            rho = sqrt(rho2)
            Fl, fl = DD.cov_both(D, xs, 0.0, rho, 0.0, :zero, 0.0, :lines)
            Fm, fm = DD.cov_both(D, xs, 0.0, rho, 0.0, :zero, 0.0, :mixture)
            @test maximum(abs, Fl .- Fm) < 1e-9
            @test maximum(abs, fl .- fm) < 1e-9
        end
        # local alternatives, R different from rho (c0 = 5.9 > 0: the normal average graded toward the kink), a negative
        # c0, the stationary start, far from the unit root
        for (D, c, rho, c0, st, x) in ((trend(0), -5.0, sqrt(0.7), cadf_c0(-5.0, sqrt(0.7), sqrt(0.3)), :zero, [-2.0]),
                                       (trend(1), -10.0, sqrt(0.8), cadf_c0(-10.0, sqrt(0.8), sqrt(2.0)), :zero, [-3.0, -2.0, 0.5]),
                                       (trend(-1), -5.0, sqrt(0.5), 5.0, :zero, [-2.0]), (trend(0), -5.0, sqrt(0.5), -2.0, :stationary, [-3.0, -2.0, 0.5]),
                                       (trend(1), -50.0, 0.8, -10.0, :zero, [-3.0, 0.5]))
            Fl, fl = DD.cov_both(D, x, c, rho, c0, st, 0.0, :lines)
            Fm, fm = DD.cov_both(D, x, c, rho, c0, st, 0.0, :mixture)
            @test maximum(abs, Fl .- Fm) < 1e-9
            @test maximum(abs, fl .- fm) < 1e-9
        end
        # the public functions take the two routes
        @test dist_cdf(trend(0), xs; rho = sqrt(0.6), method = :lines) == DD.cov_both(trend(0), xs, 0.0, sqrt(0.6), 0.0, :zero, 0.0, :lines)[1]
        @test dist_pdf(trend(0), xs; rho = sqrt(0.6)) == DD.cov_both(trend(0), xs, 0.0, sqrt(0.6), 0.0, :zero, 0.0, :lines)[2]
        @test dist_cdf(trend(0), xs; rho = sqrt(0.995)) == DD.cov_both(trend(0), xs, 0.0, sqrt(0.995), 0.0, :zero, 0.0, :mixture)[1]
    end
    @testset "the prototype's table: critical values (five decimals) and the power of the 5% tests (four)" begin
        for (D, rho2, ref, pw) in ((trend(-1), 0.3, [-2.51794, -1.85487, -1.50044], (0.8514, 0.9925)), (trend(0), 0.5, [-3.20628, -2.58675, -2.25758], (0.3683, 0.7996)),
                                   (trend(1), 0.9, [-3.90301, -3.34478, -3.05456], (0.0992, 0.2350)), (perron(:A, 0.5), 0.2, [-3.30813, -2.64809, -2.29583], (0.6227, 0.9639)),
                                   (perron(:C, 0.5), 0.7, [-4.40407, -3.82543, -3.51962], (0.1134, 0.2586)), (trend(0), 0.1, [-2.77397, -2.10402, -1.74676], (0.9602, 0.9998)))
            rho = sqrt(rho2)
            q = dist_quantile(D, [0.01, 0.05, 0.10], 0.0; rho)
            @test maximum(abs, q .- ref) < 1e-5
            for (c, p) in zip((-5.0, -10.0), pw)
                @test abs(dist_cdf(D, [q[2]], c; rho, c0 = cadf_c0(c, rho))[1] - p) < 1e-4
            end
        end
        # the break date and its mirror image at the unit root (ten digits)
        for rho2 in (0.5,)
            @test abs(dist_quantile(perron(:A, 0.3), 0.05; rho = sqrt(rho2), tol = 1e-11) + 3.2161800765) < 1e-9
            @test abs(dist_quantile(perron(:A, 0.7), 0.05; rho = sqrt(rho2), tol = 1e-11) + 3.2161800765) < 1e-9
        end
    end
    @testset "the predictive regression: the companion paper's table and example" begin
        z95 = 1.6448536269514722; z975 = 1.959963984540054
        for (c, rho, pub) in ((0, -0.95, (0.0005, 0.4213, 0.2776)), (-5, -0.95, (0.0042, 0.1954, 0.1172)), (-20, -0.99, (0.0161, 0.1155, 0.0713)),
                              (-100, -0.95, (0.0321, 0.0739, 0.0541)), (0, -0.5, (0.0065, 0.1806, 0.1095)))
            F = ltratio_cdf([-z95, z95, -z975, z975], "DU", c; rho, c0 = predictive_c0(c, rho))
            @test maximum(abs, (F[1], 1 - F[2], F[3] + 1 - F[4]) .- pub) < 1e-4
        end
        rho = -0.95
        for (c, pub) in ((0, (0.261, 2.821, -0.324)), (-5, (0.108, 2.378, -0.988)))
            c0 = predictive_c0(c, rho)
            @test abs(1 - ltratio_cdf(2.0, "DU", c; rho, c0) - pub[1]) < 1e-3
            @test maximum(abs, ltratio_icdf([0.95, 0.025], "DU", c; rho, c0) .- pub[2:3]) < 1e-3
        end
        @test abs(1 - ltratio_cdf(1.645, "DU", -5; rho, c0 = predictive_c0(-5, rho, 10.0)) - 0.999) < 1e-3
    end
    @testset "a distribution function with its density; the fixed start; verified" begin
        D = perron(:A, 0.4); rho = sqrt(0.6)
        g = collect(-7.0:0.5:5.0)
        F = dist_cdf(D, g; rho); f = dist_pdf(D, g; rho)
        @test all(diff(F) .> 0) && F[1] < 1e-8 && F[end] > 1 - 1e-6 && all(f .> 0)
        h = 1e-3
        fd = (dist_cdf(D, [-2.0 + h]; rho)[1] - dist_cdf(D, [-2.0 - h]; rho)[1]) / (2h)
        @test abs(fd - dist_pdf(D, [-2.0]; rho)[1]) < 1e-6
        q = dist_quantile(D, 0.05; rho)
        @test abs(dist_cdf(D, [q]; rho)[1] - 0.05) < 1e-9
        # the fixed start by the normal average; agrees with the lines at a tiny fixed start
        @test ltratio_cdf([-2.0], "DU", -5; rho, g0 = 1.0) == dist_cdf(trend(0), [-2.0], -5.0; rho, start = :fixed, gamma0 = 1.0)
        @test abs(dist_cdf(trend(0), [-2.0], -5.0; rho, start = :fixed, gamma0 = 1e-6)[1] - dist_cdf(trend(0), [-2.0], -5.0; rho)[1]) < 1e-9
        @test maximum(abs, dist_cdf(trend(1), [-3.0, -1.0], -5.0; rho, verified = true) .- dist_cdf(trend(1), [-3.0, -1.0], -5.0; rho)) < 1e-9
        @test dist_cdf(trend(0), [-Inf, Inf], -5.0; rho) == [0.0, 1.0] && isnan(dist_cdf(trend(0), [NaN], -5.0; rho)[1])
        # small rho: close to the standard normal law, by the normal average (rho^2 < 0.1)
        @test maximum(abs, dist_cdf(trend(0), [-1.0, 0.0, 1.0]; rho = 0.01) .- Phi.([-1.0, 0.0, 1.0])) < 0.01
    end
    @testset "argument errors" begin
        x = [-2.0]
        @test occursin("rho must lie in [-1, 1]", errmsg(() -> dist_cdf(trend(0), x, -5.0; rho = 1.5)))
        @test occursin("rho must lie in [-1, 1]", errmsg(() -> dist_cdf(trend(0), x, -5.0; rho = NaN)))
        @test occursin("GLS detrending", errmsg(() -> dist_cdf(trend(1), x, -5.0; rho = 0.5, gls = -13.5)))
        @test occursin("detrend_first", errmsg(() -> dist_cdf(detrend_first(trend(1)), x, -5.0; rho = 0.5)))
        @test occursin("method must be :auto, :lines", errmsg(() -> dist_cdf(trend(0), x, -5.0; rho = 0.5, method = :paths)))
        @test occursin("needs a zero or a stationary start", errmsg(() -> dist_cdf(trend(0), x, -5.0; rho = 0.5, start = :fixed, gamma0 = 1.0, method = :lines)))
        @test occursin("below and at the unit root", errmsg(() -> dist_cdf(trend(0), x, 2.0; rho = 0.5, method = :lines)))
        @test occursin("do not apply", errmsg(() -> dist_cdf(trend(0), x, -5.0; rho = 0.5, n = 32)))
        @test occursin("do not apply", errmsg(() -> dist_cdf(trend(0), x, -5.0; rho = 0.5, hfac = 0.5)))
        @test occursin("engine = :classical", errmsg(() -> ltratio_cdf(x, "DU", 0; rho = 0.5, engine = :classical)))
        @test occursin("p must lie in (0, 1)", errmsg(() -> dist_quantile(trend(0), 1.5, 0.0; rho = 0.5)))
    end
    @testset "type stability" begin
        CP = DD.cpath_cached(DD.startmodel(trend(0), 0.0, :zero, 0.0), 0.0)
        @inferred DD.lines_both(CP, [-2.0, 1.0], 0.7, 0.0)
        @inferred dist_cdf(trend(0), [-2.0], 0.0; rho = 0.7)
        @inferred dist_pdf(trend(0), [-2.0], 0.0; rho = 0.7, method = :mixture)
        @inferred dist_quantile(trend(0), 0.05, 0.0; rho = 0.7)
    end
end
