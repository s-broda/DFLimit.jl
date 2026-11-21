# The tests of the general engine (DFLimit.DFDet; the file test/runtests.jl of the repository df-deterministics, with this
# header), about a minute: the covariance Sigma against nested adaptive quadrature, the determinant against DFExact.Dfun
# in cases DK, DU, DT, the distribution function against DFLimit's ltratio_cdf, t_* at a double zero, the inertia of A,
# the parity with the classical engine, the speed stage, the path tables, and regression tests of critical values.  The
# comparisons with the classical functions force their engine (engine = :classical) where the rule of the merged package
# would take the general engine itself.
using DFLimit
using DFLimit.DFDet
using Test, QuadGK, LinearAlgebra, Logging
const DX = DFLimit.DFExact

function sigma_bf(P, θ)
    D = P.D; br = P.br; N = P.N
    gval(i, r) = (j = DFDet.piece(D, r); evalpoly(r, view(P.g[j], :, i)))
    function Fv(s)
        F = zeros(ComplexF64, N)
        F[1] = exp(-θ * (1 - s))
        for i in 1:P.nj; F[1+i] = s < P.jl[i] ? exp(-θ * (P.jl[i] - s)) : 0; end
        for i in 1:P.p
            F[1+P.nj+i] = quadgk(r -> gval(i, r) * exp(-θ * (r - s)), s, [b for b in br if b > s]...; rtol = 1e-13, atol = 1e-15)[1]
        end
        F
    end
    S = zeros(ComplexF64, N, N)
    for a in 1:N, b in a:N
        S[a, b] = S[b, a] = quadgk(s -> (F = Fv(s); F[a] * F[b]), br...; rtol = 1e-12, atol = 1e-15)[1]
    end
    S
end

@testset "DFDet" begin
    @testset "Sigma" begin
        P = prep(perron(:C, 0.3))
        for θ in (0.01 + 0im, 2.0 - 3.0im, 0.0 + 3.0im, 30.0 - 60.0im)
            Sb = sigma_bf(P, θ)
            @test maximum(abs, sigma(P, θ) - Sb) < 1e-13 * maximum(abs, Sb)
        end
    end
    @testset "determinant, cases DK DU DT" begin
        for (model, p) in ((1, -1), (2, 0), (3, 1)), c in (0.0, -5.0)
            P = prep(trend(p))
            for (s, t) in ((0.3 + 2.0im, 0.5 - 7.0im), (-0.8 - 5.0im, -3.0 + 40.0im), (1.1 + 0.0im, 0.9 + 0.0im))
                Dc = DX.Dfun(model, c, s, t) * (model == 1 ? exp(c) : 1.0)
                @test abs(Dval(P, c, s, t) - Dc) < 1e-11 * abs(Dc)
            end
        end
    end
    @testset "inertia" begin
        @test (prep(trend(3)).rp, prep(trend(3)).rm) == (1, 1)
        P = prep(perron(:C, 0.3)); @test (P.r, P.rp, P.rm) == (4, 2, 2)
    end
    @testset "t_* at a double zero" begin
        ts, mult = tstar(prep(perron(:C, 0.5)), 0.0)
        @test abs(ts - 8pi^2) < 1e-8 && mult == 2
    end
    @testset "distribution function, cases DK DU DT" begin
        xs = [-3.5, -2.0, -0.7, 0.4]
        for (cas, p) in (("DK", -1), ("DU", 0), ("DT", 1))
            @test maximum(abs, dist_cdf(trend(p), xs, -5.0) - ltratio_cdf(xs, cas, -5.0; engine = :classical)) < 1e-10
            @test maximum(abs, dist_pdf(trend(p), xs, -5.0) - ltratio_pdf(xs, cas, -5.0; engine = :classical)) < 1e-10
        end
    end
    @testset "fixed start: Qf against DFLimit's qstart" begin
        for (p, m) in ((-1, 1), (0, 2), (1, 3)), c in (-5.0, 0.5)
            Mo = model(prep(trend(p)), :fixed, 1.0)
            for (s, t) in ((0.3 + 2.0im, 0.5 - 7.0im), (-0.8 - 5.0im, -3.0 + 40.0im))
                q = DX.qstart(m, c, s, t)
                @test abs(Qf(Mo, c, s, t) - q) < 1e-11 * max(1, abs(q))
                @test abs(DFDet.Qf_direct(prep(trend(p)), c, s, t) - q) < 1e-11 * max(1, abs(q))
            end
        end
        Mo = model(prep(perron(:C, 0.3)), :fixed, 2.0)
        @test abs(Qf(Mo, 0.0, 0.7 - 3.0im, 2.0 + 9.0im)) < 1e-12 && Mo.kappa == 0 && Mo.a == 0.5
        @test model(prep(trend(-1)), :fixed, 2.0).a == 2.5
    end
    @testset "fixed start: the shift, kappa in {0, 1/2} and the span criterion" begin
        pw(cols...) = Determ([0.0, 1.0], [reduce(hcat, [Float64.(c) for c in cols])], "")
        designs = [(Determ([0.0, 0.4, 1.0], [[1.0 1.0; 1.0 0.0], [1.0 0.0; 1.0 0.0]], "1 + r, 1{r < 0.4}"), 0.5),
                   (fdesign(r -> [exp(r)]), 0.0), (pw([0, 1]), 0.5), (pw([0, 1, 0], [0, 0, 1]), 0.5), (pw([1, 1], [0, 1]), 0.0),
                   (trend(-1), 0.5), (trend(1), 0.0), (perron(:C, 0.3), 0.0), (pw([1, 1, 0], [0, 0, 1]), 0.0),
                   (Determ([0.0, 0.5, 1.0], [[1.0 0.0; 0.0 0.0], [0.5 1.0; 1.0 0.0]], "1 + (r - 0.5)+, 1{r > 0.5}"), 0.0),
                   (fdesign(r -> [sin(2pi * r), cos(2pi * r)]), 0.0), (fdesign(r -> [cos(r)]), 0.5)]
        for (D, k) in designs
            P = prep(D)
            y, kr, res = DFDet.fixed_shift(P)
            @test res < 1e-12
            @test abs(kr - k) < 1e-10
            @test DFDet.kappa_span(D)[1] == k
            Mo = model(P, :fixed, 1.0)
            @test Mo.kappa == k && Mo.a == 0.5 + k
            if k == 0
                al = DFDet.kappa_span(D)[2]
                e = DFDet.shift_functionals(P, al); ph0 = dot(al, DFDet.dval(D, 0.0))
                @test norm(2 .* (P.A * e) .+ ph0 .* P.b) < 1e-12 * (norm(P.b) * abs(ph0) + norm(P.A * e))
            end
        end
    end
    @testset "stationary start: determinant against DFLimit" begin
        for (p, m) in ((-1, 1), (0, 2), (1, 3)), c in (-0.5, -8.0)
            Ms = model(prep(trend(p)), :stationary)
            for (s, t) in ((0.3 + 2.0im, 0.5 - 7.0im), (-0.8 - 5.0im, -3.0 + 40.0im))
                d = DX.Dfun(m, c, s, t; stat = true)
                @test abs(Dval(Ms, c, s, t) - d) < 1e-11 * abs(d)
            end
        end
        Ms = model(prep(perron(:A, 0.5)), :stationary); @test (Ms.r, Ms.rp, Ms.rm) == (4, 2, 2)
    end
    @testset "distribution function: fixed and stationary start" begin
        xs = [-3.5, -2.0, -0.7, 0.4]
        @test maximum(abs, dist_cdf(trend(1), xs, -5.0; start = :fixed, gamma0 = 2.0) - ltratio_cdf(xs, "DT", -5.0; g0 = 2.0)) < 1e-10
        @test maximum(abs, dist_cdf(trend(-1), xs, -5.0; start = :fixed, gamma0 = 1.5) - ltratio_cdf(xs, "DK", -5.0; g0 = 1.5, engine = :classical)) < 1e-10
        @test maximum(abs, dist_cdf(trend(0), xs, -5.0; start = :stationary) - ltratio_cdf(xs, "SU", -5.0)) < 1e-10
        @test maximum(abs, dist_pdf(trend(1), xs, -10.0; start = :stationary) - ltratio_pdf(xs, "ST", -10.0)) < 1e-10
    end
    @testset "fixed start far from the unit root" begin
        # (the cases that failed with the earlier placement grid, by 11 and 2.4; now without a warning)
        xs = [-8.0, -5.0, -3.0, -1.0, 1.0]
        for (D, cas, g) in ((trend(0), "DU", 1.5), (trend(1), "DT", 3.0))
            F = @test_logs min_level = Logging.Warn dist_cdf(D, xs, -40.0; start = :fixed, gamma0 = g)
            @test maximum(abs, F - ltratio_cdf(xs, cas, -40.0; g0 = g, verified = true)) < 1e-9
        end
        # the panels cut to the rate of log Mt (case DT at c = -50, g0 = 5: off by 9e-4 without)
        @test maximum(abs, dist_cdf(trend(1), [-3.0, 0.0, 3.0], -50.0; start = :fixed, gamma0 = 5.0) .- 1) < 1e-9
        P = prep(trend(0)); Mo = model(P, :fixed, 1.5)
        CP = DFDet.cpath_cached(Mo, -40.0)
        F, f, _, er = contour_eval(CP, [-3.0, 0.0])
        @test maximum(er.F) < 1e-10 && maximum(er.f) < 1e-10
    end
    @testset "local null value c0" begin
        xs = [-3.5, -2.0, -0.7, 0.4, 2.5]
        for c0 in (3.0, -3.0)
            @test maximum(abs, dist_cdf(trend(1), xs, -5.0; c0) - ltratio_cdf(xs, "DT", -5.0; c0, verified = true)) < 1e-10
            @test maximum(abs, dist_pdf(trend(0), xs, -5.0; c0) - ltratio_pdf(xs, "DU", -5.0; c0, verified = true)) < 1e-10
        end
    end
    @testset "coefficient statistic and moments" begin
        ys = [-20.0, -8.0, -1.0]
        @test maximum(abs, DFDet.coef_cdf(trend(1), ys, -5.0) - DFLimit.coef_cdf(ys, "DT", -5.0; engine = :classical)) < 1e-10
        @test maximum(abs, coef_cdf_gp(trend(0), ys, 0.0) - DFLimit.coef_cdf(ys, "DU", 0.0; engine = :classical)) < 1e-10
        @test abs(DFDet.coef_quantile(trend(-1), 0.05) - (-8.039128903)) < 1e-8
        m, _ = dist_moments(trend(0), -5.0; c0 = -5.0)
        @test maximum(abs, m - DX.moments(2; c = -5.0, c0 = -5.0)) < 1e-9
        m, _ = dist_moments(trend(1), -5.0; start = :stationary)
        @test maximum(abs, m - DX.moments(3; c = -5.0, start = :stationary)) < 1e-9
    end
    @testset "designs: functions, Legendre trends, several breaks, from a matrix" begin
        xs = [-4.0, -3.0, -2.0]
        @test maximum(abs, dist_cdf(ltrend(3), xs) - dist_cdf(trend(3), xs)) < 1e-11
        @test maximum(abs, dist_cdf(fdesign(r -> [1.0, r, r > 0.4 ? 1.0 : 0.0, max(r - 0.4, 0.0)], [0.4]), xs, -5.0) - dist_cdf(perron(:C, 0.4), xs, -5.0)) < 1e-11
        D = design_from_matrix(regmat(perron(:C, 0.3), 500))
        @test D.br == [0.0, 0.3, 1.0]
        @test maximum(abs, dist_cdf(D, xs) - dist_cdf(perron(:C, 0.3), xs)) < 1e-11
        @test_throws ArgumentError design_from_matrix(hcat(ones(500), (1:500) ./ 500, [t == 250 ? 1.0 : 0.0 for t in 1:500]))
        P = prep(perron(:C, [0.25, 0.5, 0.75])); @test (P.r, P.rp, P.rm) == (8, 4, 4)
    end
    @testset "GLS detrending and c > 0" begin
        xs = [-3.0, -2.0, -1.0, 0.5]
        @test maximum(abs, dist_cdf(trend(1), xs, -5.0; gls = -13.5) - ltratio_cdf(xs, "DG", -5.0)) < 1e-10
        @test maximum(abs, dist_cdf(perron(:A, 0.4), xs, -5.0; gls = -13.5) - dist_cdf(trend(1), xs, -5.0; gls = -13.5)) < 1e-12
        @test maximum(abs, dist_cdf(trend(0), xs, -5.0; gls = -13.5) - dist_cdf(trend(-1), xs, -5.0)) < 1e-12
        xs = [-1.0, 1.0, 3.0]
        @test maximum(abs, dist_cdf(trend(0), xs, 4.0) - ltratio_cdf(xs, "DU", 4.0)) < 1e-10
    end
    @testset "GLS detrending with c0, the coefficient statistic, moments; no start" begin
        xs = [-3.0, -2.0, -1.0, 0.5]
        @test maximum(abs, dist_cdf(trend(1), xs, -5.0; c0 = 3.0, gls = -13.5) - ltratio_cdf(xs, "DG", -5.0; c0 = 3.0, verified = true)) < 1e-10
        @test maximum(abs, dist_pdf(trend(1), xs, -10.0; c0 = -10.0, gls = -13.5) - ltratio_pdf(xs, "DG", -10.0; c0 = -10.0, verified = true)) < 1e-10
        ys = [-20.0, -8.0, -1.0]
        @test maximum(abs, DFDet.coef_cdf(trend(1), ys, -5.0; gls = -13.5) - DFLimit.coef_cdf(ys, "DG", -5.0; verified = true)) < 1e-10
        # the moments (the radius of the Cauchy circles capped at 2 with GLS detrending): d = (1) is case DK, and the mean of
        # d = (1, r) against the integral of its distribution function (check 3)
        m, _ = dist_moments(trend(0), 0.0; gls = -13.5)
        @test maximum(abs, m - DX.moments(1; c = 0.0)) < 1e-9
        m, _ = dist_moments(trend(1), 0.0; gls = -13.5)
        @test abs(m[1] - (-1.650715212)) < 1e-8 && abs(m[2] - 3.207920935) < 1e-8
        @test_throws ArgumentError dist_cdf(trend(1), xs, -5.0; gls = -13.5, start = :fixed, gamma0 = 1.0)
        @test_throws ArgumentError dist_cdf(trend(1), xs, -5.0; gls = -13.5, start = :stationary)
    end
    @testset "the statistic that detrends first" begin
        # closed under differentiation: the same matrix and the same law as the one-step regression
        xs = [-4.0, -3.0, -1.5, 0.5]
        for D in (trend(1), trend(2), perron(:A, 0.3), perron(:C, 0.3))
            @test maximum(abs, prep(detrend_first(D)).A - prep(D).A) < 1e-12
            @test maximum(abs, dist_cdf(detrend_first(D), xs, -5.0) - dist_cdf(D, xs, -5.0)) < 1e-12
        end
        # the slope break: rank two, and the 5% critical values of the prototype (four decimals)
        P = prep(detrend_first(perron(:B, 0.3))); @test (P.r, P.rp, P.rm) == (2, 1, 1)
        @test DFDet.closed_under_derivative(perron(:C, 0.3)) && !DFDet.closed_under_derivative(perron(:B, 0.3))
        cv = [-3.4604, -3.6468, -3.7625, -3.8284, -3.8592, -3.8609, -3.8334, -3.7701, -3.6582]
        for (k, lam) in enumerate(0.1:0.1:0.9)
            q = dist_quantile(detrend_first(perron(:B, round(lam, digits = 1))), 0.05; x0 = -3.6)
            @test abs(q - cv[k]) < 1e-4
        end
        D = detrend_first(perron(:B, 0.5))
        @test_throws ArgumentError dist_cdf(D, xs, -5.0; start = :fixed, gamma0 = 1.0)
        @test_throws ArgumentError dist_cdf(D, xs, -5.0; start = :stationary)
        @test_throws ArgumentError dist_cdf(D, xs, -5.0; gls = -13.5)
        @test_throws ArgumentError dist_moments(D, -5.0)
    end
    @testset "parity: ADFmax, the line U - aV <= b, the law of V, the power envelope, regions" begin
        xs = [-3.0, -2.0, -1.0, 0.0, 1.0]
        @test maximum(abs, DFDet.adfmax_cdf(trend(0), xs, 0.0) - DFLimit.adfmax_cdf(xs, "DU", 0.0)) < 1e-10
        @test maximum(abs, DFDet.adfmax_pdf(trend(1), xs, -10.0; start = :stationary) - DFLimit.adfmax_pdf(xs, "ST", -10.0)) < 1e-10
        # time reversal at c = 0 with an intercept: P(ADFmax <= x) = 2 P(tau <= x) - 1 for x >= 0
        D = perron(:C, 0.3)
        @test maximum(abs, DFDet.adfmax_cdf(D, [0.0, 0.7, 2.0], 0.0) - (2 .* dist_cdf(D, [0.0, 0.7, 2.0], 0.0) .- 1)) < 1e-12
        @test abs(DFDet.adfmax_quantile(trend(0), 0.05) - DFLimit.adfmax_icdf(0.05, "DU")) < 1e-8
        bs = [-2.0, -0.3, 0.0, 0.5, 2.0]
        @test maximum(abs, lin_cdf(trend(-1), bs, -3.5, -7.0) - [ltratio_lincdf(b, -3.5, "DK", -7.0) for b in bs]) < 1e-11
        @test maximum(abs, lin_cdf(trend(1), bs, 1.0, 0.0; gls = -13.5) - [ltratio_lincdf(b, 1.0, "DG", 0.0) for b in bs]) < 1e-11
        @test maximum(abs, v_cdf(trend(-1), [0.05, 0.3, 2.0], 0.0) - [ltratio_lincdf(y, NaN, "DK", 0.0) for y in (0.05, 0.3, 2.0)]) < 1e-11
        # the density of U - aV against a difference quotient of its distribution function
        h = 1e-4
        @test abs(lin_pdf(trend(1), [0.3], -2.0, -5.0)[1] - (lin_cdf(trend(1), [0.3 + h], -2.0, -5.0)[1] - lin_cdf(trend(1), [0.3 - h], -2.0, -5.0)[1]) / 2h) < 1e-7
        @test abs(DFDet.power_envelope(trend(-1), [-7.0])[1] - DFLimit.power_envelope(-7.0, "DK")) < 1e-8
        @test abs(region_prob(trend(-1), (type = "line", a = -3.5, b = -0.26), -7.0) - ltratio_region((type = "line", a = -3.5, b = -0.26), "DK", -7.0)) < 1e-10
        @test_throws ArgumentError region_prob(trend(-1), (type = "wap", k = 3.6658, omega = 10), 0.0)
        @test_throws ArgumentError DFDet.power_envelope(perron(:A, 0.5), [-7.0])
    end
    @testset "parity: the range of c, verified, non-finite x, the infinite density" begin
        # far from the unit root (the hyperbola chosen by the largest term, ending at 20 c^2)
        for (D, cas, st) in ((trend(0), "DU", :zero), (trend(1), "ST", :stationary))
            m = ltratio_mom(2, cas, -300.0); xs = m[1] .+ sqrt(m[2] - m[1]^2) .* [-2.0, 0.0, 2.0]
            @test maximum(abs, dist_cdf(D, xs, -300.0; start = st) - ltratio_cdf(xs, cas, -300.0; verified = true)) < 1e-10
            @test maximum(abs, dist_pdf(D, xs, -300.0; start = st) - ltratio_pdf(xs, cas, -300.0; verified = true)) < 1e-10
        end
        # the explosive side: the upper tail on the refined vertical line (c = 5, mean + 3 sd)
        @test abs(dist_cdf(trend(1), [78.23], 5.0)[1] - ltratio_cdf(78.23, "DT", 5.0; verified = true)) < 1e-10
        @test abs(dist_cdf(trend(-1), [193.8], 5.0)[1] - ltratio_cdf(193.8, "DK", 5.0; verified = true)) < 1e-10
        xs = [-3.5, -2.0, -0.7]
        F = @test_logs min_level = Logging.Warn dist_cdf(perron(:B, 0.5), xs, -5.0; verified = true)
        @test maximum(abs, F - dist_cdf(perron(:B, 0.5), xs, -5.0)) < 1e-9
        F = dist_cdf(trend(1), [-Inf, NaN, -2.0, Inf], -5.0)
        @test F[1] == 0 && isnan(F[2]) && F[4] == 1 && abs(F[3] - dist_cdf(trend(1), [-2.0], -5.0)[1]) == 0
        @test dist_pdf(trend(-1), [-2.0, -1.5], 0.0; c0 = 2.0)[1] == Inf
        @test isfinite(dist_pdf(trend(0), [-2.0], 0.0; c0 = 2.0)[1])
        q = dist_quantile(trend(0), [0.05, 0.10]; x0 = -2.8)
        @test maximum(abs, q - ltratio_icdf([0.05, 0.10], "DU"; engine = :classical)) < 1e-8
    end
    @testset "fixed start with c0 > 0 above the unit root; the line with c0 < 0 warns; GLS at a small cbar" begin
        m = ltratio_mom(2, "DU", 2.0; c0 = 3.0, g0 = 1.5); xs = m[1] .+ sqrt(m[2] - m[1]^2) .* [0.0, 2.0, 4.0]
        @test maximum(abs, dist_cdf(trend(0), xs, 2.0; c0 = 3.0, start = :fixed, gamma0 = 1.5) - ltratio_cdf(xs, "DU", 2.0; c0 = 3.0, g0 = 1.5, verified = true)) < 1e-10
        @test_logs (:warn, r"vertical line") match_mode = :any dist_cdf(trend(-1), [94.6], 5.0; c0 = -3.0)
        @test abs(DFDet.power_envelope(trend(1), [-2.0])[1] - DFLimit.power_envelope(-2.0, "DG")) < 1e-8
    end
    @testset "bounded caches, no second refinement of a line" begin
        old = DFDet.CACHE_MAX[]; DFDet.CACHE_MAX[] = 3
        try
            for c in (-1.0, -2.0, -3.0, -4.0, -5.0); dist_cdf(trend(0), [-2.0], c); end
            @test length(DFDet.CPCACHE) <= 3 && length(DFDet.CPORDER) == length(DFDet.CPCACHE)
        finally
            DFDet.CACHE_MAX[] = old
        end
        empty_caches!(); @test isempty(DFDet.CPCACHE) && isempty(DFDet.LINECACHE) && isempty(DFDet.PREPCACHE)
        CP = DFDet.cpath_cached(model(prep(trend(-1))), 5.0)
        CPr = DFDet.line_refined(CP, [200.0]); @test length(CPr.t) > length(CP.t)
        @test DFDet.line_refine(CPr, 4096.0) === CPr                  # a refined line is not refined again
    end
    @testset "speed stage: e_k, t_* far from the unit root, the BLAS threads" begin
        # e_k against the series in BigFloat, where the power series in double precision lost eight digits (D = 26, |z| = 20)
        function ekbig(z, D)
            setprecision(320) do
                zb = Complex{BigFloat}(z); e = [Complex{BigFloat}(1) / (k + 1) for k in 0:D]; term = Complex{BigFloat}(1)
                for n in 1:2000
                    term *= -zb / n
                    for k in 0:D; e[k+1] += term / (n + k + 1); end
                    abs(term) < big(1e-70) && break
                end
                ComplexF64.(e)
            end
        end
        for (D, z) in ((26, 20.0 + 0im), (26, 7.654 + 18.48im), (8, 5.0 - 2.0im), (3, 2.5 + 0.4im), (1, 0.3 - 1.1im))
            ex = ekbig(z, D); sc = [max(1.0, exp(-real(z))) / (k + 1) for k in 0:D]
            @test maximum(abs.(DFDet.ek!(zeros(ComplexF64, D + 1), z, D) - ex) ./ sc) < 5e-15
        end
        # case DK: D_V is proportional to cos(omega) - c sin(omega)/omega at theta = i omega, t_* = (c^2 + omega^2)/2
        for c in (-20.0, -300.0, -480.0)
            a, b = pi / 2, pi
            for _ in 1:200; m = (a + b) / 2; (cos(m) - c * sin(m) / m > 0) ? (a = m) : (b = m); end
            @test abs(tstar(model(prep(trend(-1))), c)[1] - (c^2 + a^2) / 2) < 1e-12 * c^2
        end
        nb = BLAS.get_num_threads()
        DFDet.cpath(model(prep(trend(1))), 0.0); @test BLAS.get_num_threads() == nb
    end
    @testset "static arrays: the transform against the dense code, the eigenvalues against LAPACK" begin
        for (D, st, g) in ((trend(1), :zero, 0.0), (perron(:C, [0.3, 0.7]), :zero, 0.0), (perron(:C, 0.3), :fixed, 1.0), (trend(1), :stationary, 0.0))
            Mo = model(prep(D), st, g); Ws = DFDet.MWork(Mo)
            DFDet.USE_STATIC[] = false; Wd = DFDet.MWork(Mo); DFDet.USE_STATIC[] = true
            @test Ws.sd !== nothing && Wd.sd === nothing
            for t in (complex(1.0, 0.0), complex(3.0, 40.0), complex(-50.0, 300.0))
                a = DFDet.mgfparts(Mo, -5.0, t, Ws); b = DFDet.mgfparts(Mo, -5.0, t, Wd)
                @test abs(a[2] - b[2]) < 1e-12
                ra = sort(1 ./ a[3]; by = z -> (real(z), imag(z))); rb = sort(1 ./ b[3]; by = z -> (real(z), imag(z)))
                @test maximum(abs.(ra - rb) ./ abs.(rb)) < 1e-12
                st == :fixed && @test abs(a[4].cf - b[4].cf) < 1e-12 && maximum(abs, a[4].w1 - b[4].w1) < 1e-12 * maximum(abs, b[4].w1)
            end
        end
        for R in (3, 4, 6, 8), k in 1:20
            A = [cis(0.7 * i * j + k) * exp(2 * sin(3i + 5j + 11k)) for i in 1:R, j in 1:R]     # deterministic, graded
            e = DFDet.seigvals(DFDet.SMatrix{R,R,ComplexF64}(A)); el = eigvals(A)
            @test e !== nothing && maximum(minimum(abs.(collect(e) .- z)) for z in el) < 1e-13 * maximum(abs, el)
        end
    end
    @testset "fixed start: the exponent in the terms of the proof, its bound; designs without a proper intercept" begin
        # E1(s) = -1/2 m'(Theta - Lambda^{-1}/(2s))^{-1} m with Theta = Lambda^{-1} H/2, and |Q - Q(inf)| <= 1/2 on |s| = B
        Mo = model(prep(perron(:C, 0.3)), :fixed, 1.5); CP = DFDet.cpath(Mo, -10.0)
        for k in (1, 20, 60)
            fx = CP.fx[k]; Bf = DFDet.fixbuf(Mo.r); Θ = (fx.H ./ Mo.Dv) ./ 2
            for s in (complex(0.3, 0.1), complex(-2.0, 5.0))
                q = -0.5 * transpose(fx.mt) * ((Θ - Diagonal(1 ./ Mo.Dv) ./ (2s)) \ fx.mt)
                @test abs(DFDet.fixE1(fx, s, Bf) - q) < 1e-10 * max(1, abs(q))
            end
            B, φ = DFDet.fixbound(fx, Mo.Dv, CP.t[k]); qinf = -0.5 * transpose(fx.mt) * (Θ \ fx.mt)
            @test isfinite(B) && maximum(abs(DFDet.fixE1(fx, B * cis(a), Bf) - qinf) for a in 0:0.5:6) <= 0.5
        end
        # d(r) = 1 + r: the positive roots leave Re s > 0 on the 45-degree hyperbola, the path is the vertical line
        D = Determ([0.0, 1.0], [reshape([1.0, 1.0], 2, 1)], "1+r")
        @test isnan(DFDet.cpath(model(prep(D)), 0.0).h)
        F = with_logger(NullLogger()) do; dist_cdf(D, [-2.0, -1.0], 0.0); end
        @test abs(F[1] - 0.32301) < 1e-4 && abs(F[2] - 0.81626) < 1e-4
    end
    @testset "the path tables of cases DK, DU, DT at c = 0" begin
        xs = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        for p in -1:1
            Mo = model(prep(trend(p)))
            @test DFDet.table_case(Mo, 0.0) == p && isfile(DFDet.table_file(p))
            CPt = DFDet.read_path_table(DFDet.table_file(p), Mo); CPc = DFDet.cpath(Mo, 0.0)
            @test length(CPt.t) == length(CPc.t) && maximum(abs, CPt.t - CPc.t) <= 1e-12 * maximum(abs, CPc.t)
            Ft, ft = contour_eval(CPt, xs)[1:2]; Fc, fc = contour_eval(CPc, xs)[1:2]
            @test maximum(abs, Ft - Fc) < 1e-13 && maximum(abs, ft - fc) < 1e-13
        end
        @test DFDet.table_case(model(prep(trend(2))), 0.0) == -2 && DFDet.table_case(model(prep(trend(1))), -1.0) == -2
        @test DFDet.table_case(model(prep(trend(1)), :fixed, 1.0), 0.0) == -2 && DFDet.table_case(model(prep_gls(trend(1), -13.5)), 0.0) == -2
        @test DFDet.table_case(model(prep(detrend_first(trend(1)))), 0.0) == -2
        empty_caches!(); F1 = dist_cdf(trend(0), xs)
        DFDet.USE_TABLES[] = false
        try
            empty_caches!(); F2 = dist_cdf(trend(0), xs); @test maximum(abs, F1 - F2) < 1e-13
        finally
            DFDet.USE_TABLES[] = true; empty_caches!()
        end
    end
    @testset "critical values" begin
        q = dist_quantile(perron(:A, 0.5), 0.05; x0 = -4.0)
        @test abs(q - (-3.7147)) < 1e-4
        # the time reversal at c = 0: lambda and 1 - lambda give the same law
        @test abs(dist_cdf(perron(:C, 0.2), [-4.0])[1] - dist_cdf(perron(:C, 0.8), [-4.0])[1]) < 1e-12
    end
end
