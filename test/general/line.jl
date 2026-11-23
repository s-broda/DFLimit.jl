# The line integral with the explicit kernel for the designs whose numerator has rank one or two (src/DFDet_line.jl): the
# determinant, the roots and P(t) of polynomial trends at the unit root against the spherical Bessel formula, the provider
# against the classical engine's closed forms in the classical designs, the line integral against the inversion on paths
# on a reduced grid, the refusals of method = :line, the automatic choice, the quantiles and the checked computation.
using DFLimit, Test, Logging
const GL = DFLimit.DFDet; const DXL = DFLimit.DFExact
errl(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end
quietl(f) = with_logger(NullLogger()) do; f(); end

# the spherical Bessel function of order nu divided by its leading term at zero, by its power series (|z| <= 3 here):
# sum_k (-z^2/2)^k / (k! prod_{i=1..k} (2 nu + 2 i + 1)); nu = -1 gives cos
function jhat(nu::Int, z)
    acc = zero(z); term = one(z)
    for k in 0:80
        acc += term
        term *= -z * z / 2 / ((k + 1) * (2nu + 2k + 3))
    end
    acc
end
const TRIG = GL.fdesign(r -> [1.0, sin(2pi * r), cos(2pi * r)]; name = "intercept, sine and cosine")

@testset "the line integral of rank one or two" begin

@testset "the unit root: the spherical Bessel formula, trend of order $p" for p in 2:4
    # D(s,t) = jh_p(x) {jh_{p-1}(x) - c_p s^2 jh_{p+1}(x)}, x = (t/2)^{1/2}, c_p = 1/(4 (2p+1)(2p+3))
    P = GL.prep(GL.trend(p)); cp = 1 / (4 * (2p + 1) * (2p + 3))
    for t in (complex(0.7), complex(1.0, 2.0), complex(1.0, 8.0), complex(-3.0, 1.0)), s in (complex(0.0), complex(1.5), complex(0.3, -2.0))
        x = sqrt(t / 2)
        @test GL.Dval(P, 0.0, s, t) ≈ jhat(p, x) * (jhat(p - 1, x) - cp * s^2 * jhat(p + 1, x)) rtol = 1e-10
    end
    # the provider on the line t = 1 + i w: the roots -+ (jh_{p-1}/(c_p jh_{p+1}))^{1/2} and P(t)^{-2} = c_p jh_p jh_{p+1}
    lm, why = GL.line_model(GL.trend(p), 0.0, :zero, 0.0, nothing)
    @test why === nothing && lm.r == 2
    ts = [complex(1.0, w) for w in range(0, 16; length = 120)]
    Pt, s1, s2, M0 = DXL.tparts(lm, 0.0, ts)
    for j in (1, 30, 80, 120)
        x = sqrt(ts[j] / 2); r = sqrt(jhat(p - 1, x) / (cp * jhat(p + 1, x)))
        @test s2[j] ≈ r rtol = 1e-10
        @test s1[j] ≈ -r rtol = 1e-10
        @test Pt[j]^(-2) ≈ cp * jhat(p, x) * jhat(p + 1, x) rtol = 1e-10
        @test M0[j]^(-2) ≈ -s1[j] * s2[j] rtol = 1e-12
    end
    @test imag(Pt[1]) == 0 && real(Pt[1]) > 0                        # (P is positive at the real point)
end

@testset "the classical designs: the provider against the closed forms" begin
    # the classical integrator with P and the roots of the general engine against itself with the closed forms (the classical
    # engine's line integral, method = :single): the provider is all that differs
    for (cas, D, st, cs) in (("DK", GL.trend(-1), :zero, (0.0, -13.5, 2.0)), ("DU", GL.trend(0), :zero, (0.0, -50.0, 5.0)),
                             ("DT", GL.trend(1), :zero, (0.0, -13.5, 2.0)), ("SK", GL.trend(-1), :stationary, (-0.5, -13.5)),
                             ("SU", GL.trend(0), :stationary, (-13.5,)), ("ST", GL.trend(1), :stationary, (-5.0,)))
        for c in cs
            xs = [-4.0, -2.2, -1.0, 0.0, 1.3]
            F = GL.dist_cdf(D, xs, c; start = st, method = :line); f = GL.dist_pdf(D, xs, c; start = st, method = :line)
            Fc = DXL.ltratio_cdf(xs, cas, c; method = :single, engine = :classical); fc = DXL.ltratio_pdf(xs, cas, c; method = :single, engine = :classical)
            # (within 1e-14 at c = 5, where the classical engine takes the Girsanov forms; 1.4e-12 in the density at c = -50,
            # where |s_1| ~ c^2 = 2500 and the roots of the general engine and the closed forms differ in the last digits)
            @test maximum(abs, F .- Fc) < 1e-12
            @test maximum(abs, f .- fc) < (c < -20 ? 1e-11 : 1e-12)
        end
    end
    # a local null value (the classical integrator's general kernel)
    for (cas, D) in (("DU", GL.trend(0)), ("DK", GL.trend(-1))), c0 in (-3.0, 3.0)
        xs = [-3.0, -1.0, 1.0]
        F = GL.dist_cdf(D, xs, -5.0; c0, method = :line); Fc = DXL.ltratio_cdf(xs, cas, -5.0; c0, method = :single, engine = :classical)
        @test maximum(abs, F .- Fc) < 1e-12
    end
end

@testset "general designs against the inversion on paths" begin
    # a reduced grid: polynomial trends of order 2 and 4, an intercept with the first sine and cosine, the statistic that
    # detrends first with a break in the slope; c in {0, -13.5, 2}; zero and stationary starts; c0 in {0, -3, 3} at c = -5;
    # x across the bulk and both tails (the 1e-4 and 1 - 1e-4 points of the paths' distribution function)
    cells = [(GL.trend(2), :zero, 0.0, 0.0), (GL.trend(2), :zero, -13.5, 0.0), (GL.trend(2), :zero, 2.0, 0.0), (GL.trend(2), :stationary, -13.5, 0.0),
             (GL.trend(4), :zero, 0.0, 0.0), (GL.trend(4), :zero, -13.5, 0.0), (GL.trend(4), :stationary, -13.5, 0.0),
             (TRIG, :zero, 0.0, 0.0), (TRIG, :zero, -13.5, 0.0), (TRIG, :stationary, -13.5, 0.0), (TRIG, :zero, 2.0, 0.0),
             (GL.detrend_first(GL.perron(:B, 0.5)), :zero, -5.0, 0.0),
             (GL.trend(2), :zero, -5.0, -3.0), (GL.trend(2), :zero, -5.0, 3.0), (TRIG, :zero, -5.0, -3.0)]
    for (D, st, c, c0) in cells
        lo = GL.dist_quantile(D, 1e-4, c; c0, start = st, method = :paths, x0 = -4.0)
        hi = GL.dist_quantile(D, 1 - 1e-4, c; c0, start = st, method = :paths, x0 = -1.0)
        xs = collect(range(lo, hi; length = 5))
        Fl = GL.dist_cdf(D, xs, c; c0, start = st, method = :line); Fp = GL.dist_cdf(D, xs, c; c0, start = st, method = :paths)
        fl = GL.dist_pdf(D, xs, c; c0, start = st, method = :line); fp = GL.dist_pdf(D, xs, c; c0, start = st, method = :paths)
        @test maximum(abs, Fl .- Fp) < 1e-10
        @test maximum(abs, fl .- fp) < 1e-10
    end
end

@testset "refusals of method = :line, and the automatic choice falls back" begin
    xs = [-2.0, -1.0]
    origin = GL.fdesign(r -> [r]; name = "trend through the origin")
    @test occursin("rank 3", errl(() -> GL.dist_cdf(origin, xs, 0.0; method = :line)))                   # d(0) = 0, not closed: rank 3
    @test occursin("d(0) = 0", errl(() -> GL.dist_cdf(GL.detrend_first(origin), xs, 0.0; method = :line)))           # rank 1
    @test occursin("a jump", errl(() -> GL.dist_cdf(GL.perron(:A, 0.5), xs, 0.0; method = :line)))                  # rank 4
    @test occursin("GLS", errl(() -> GL.dist_cdf(GL.trend(2), xs, -5.0; gls = -13.5, method = :line)))
    @test occursin("fixed start", errl(() -> GL.dist_cdf(GL.trend(2), xs, -5.0; start = :fixed, gamma0 = 1.0, method = :line)))
    @test occursin("belong to the inversion on paths", errl(() -> GL.dist_cdf(GL.trend(2), xs, 0.0; n = 32, method = :line)))
    @test occursin("hfac", errl(() -> GL.dist_cdf(GL.trend(2), xs, 0.0; hfac = 0.5, method = :line)))
    @test occursin("phi", errl(() -> GL.dist_quantile(GL.trend(2), 0.05, 0.0; phi = 1.0, method = :line)))
    @test occursin("c0 must be", errl(() -> GL.dist_cdf(GL.trend(2), xs, 0.0; c0 = NaN, method = :line)))
    @test occursin("c <= -0.0001", errl(() -> GL.dist_cdf(GL.trend(2), xs, -1e-5; start = :stationary, method = :line)))
    @test occursin("method must be", errl(() -> GL.dist_cdf(GL.trend(2), xs, 0.0; method = :single)))
    @test occursin("method must be", errl(() -> GL.dist_quantile(GL.trend(2), 0.05, 0.0; method = :contour)))
    # the automatic choice takes the paths there, silently
    for (D, kw) in ((origin, (;)), (GL.perron(:A, 0.5), (;)), (GL.trend(2), (gls = -13.5,)), (GL.trend(2), (start = :fixed, gamma0 = 1.0)))
        @test (@test_logs min_level = Logging.Warn GL.dist_cdf(D, xs, -5.0; kw...)) == GL.dist_cdf(D, xs, -5.0; kw..., method = :paths)
    end
    # beyond the bounds of the classical integrator's paths (c0 < 0 near zero, x > 0: a root of X(v) far out): an
    # ArgumentError with method = :line
    @test occursin("more nodes than the engine allows", errl(() -> GL.dist_cdf(GL.trend(2), [1.5], 0.0; c0 = -1e-4, method = :line)))
end

@testset "the automatic choice takes the faster method" begin
    xs = [-3.5, -2.5, -1.5]
    for (D, c, c0, st) in ((GL.trend(2), 0.0, 0.0, :zero), (TRIG, -13.5, 0.0, :stationary), (GL.trend(3), 2.0, 0.0, :zero))
        @test GL.dist_cdf(D, xs, c; c0, start = st) == GL.dist_cdf(D, xs, c; c0, start = st, method = :line)
        @test GL.dist_pdf(D, xs, c; c0, start = st) == GL.dist_pdf(D, xs, c; c0, start = st, method = :line)
    end
    for (D, c, c0) in ((GL.trend(2), -5.0, -3.0), (GL.trend(2), -5.0, 3.0))
        @test GL.dist_cdf(D, xs, c; c0) == GL.dist_cdf(D, xs, c; c0, method = :paths)
    end
    # the providers are cached per design, start and c, at most CACHE_MAX[] of them
    old = GL.CACHE_MAX[]; GL.CACHE_MAX[] = 3
    try
        for c in (-1.0, -2.0, -3.0, -4.0, -5.0); GL.dist_cdf(GL.trend(2), [-2.0], c); end
        @test length(GL.LMCACHE) <= 3 && length(GL.LMORDER) == length(GL.LMCACHE)
    finally
        GL.CACHE_MAX[] = old
    end
    empty_caches!(); @test isempty(GL.LMCACHE)
    # the merged interface takes it through the general functions
    @test ltratio_cdf(xs, GL.trend(2), -5.0) == GL.dist_cdf(GL.trend(2), xs, -5.0; method = :line)
end

@testset "the merged interface: method for a design" begin
    xs = [-3.5, -2.5, -1.5]
    # a design the classical engine does not have: :line and :paths are the general engine's
    @test ltratio_cdf(xs, GL.trend(2), -5.0; method = :line) == GL.dist_cdf(GL.trend(2), xs, -5.0; method = :line)
    @test ltratio_pdf(xs, GL.trend(2), -5.0; method = :paths) == GL.dist_pdf(GL.trend(2), xs, -5.0; method = :paths)
    @test ltratio_icdf(0.05, TRIG, 0.0; method = :paths) ≈ ltratio_icdf(0.05, TRIG, 0.0; method = :line) atol = 1e-9
    @test occursin("a jump", errl(() -> ltratio_cdf(xs, GL.perron(:A, 0.5), 0.0; method = :line)))
    @test occursin("does not apply with engine = :general", errl(() -> ltratio_cdf(xs, GL.trend(2), -5.0; method = :double)))
    @test ltratio_cdf(xs, GL.trend(2), -5.0; method = :single) == GL.dist_cdf(GL.trend(2), xs, -5.0; method = :line)   # (:single: the line integral)
    # a design that is a classical case: :line is the classical line integral, :paths the general engine's paths
    @test ltratio_cdf(xs, GL.trend(0), -5.0; method = :line) == ltratio_cdf(xs, "DU", -5.0; method = :single)
    @test ltratio_cdf(xs, GL.trend(0), -5.0; method = :paths) == GL.dist_cdf(GL.trend(0), xs, -5.0; method = :paths)
    # engine = :general: the general engine's methods for a case string and for a design
    @test ltratio_cdf(xs, "DU", -5.0; engine = :general, method = :line) == GL.dist_cdf(GL.trend(0), xs, -5.0; method = :line)
    @test ltratio_cdf(xs, GL.trend(0), -5.0; engine = :general, method = :paths) == GL.dist_cdf(GL.trend(0), xs, -5.0; method = :paths)
    @test ltratio_icdf(0.05, "DT", -5.0; engine = :general, method = :line) ≈ ltratio_icdf(0.05, "DT", -5.0; engine = :classical) atol = 1e-9
end

@testset "quantiles and the checked computation" begin
    for (D, c, st) in ((GL.trend(2), 0.0, :zero), (TRIG, -13.5, :stationary))
        ql = GL.dist_quantile(D, 0.05, c; start = st, method = :line, x0 = -3.0)
        @test ql ≈ GL.dist_quantile(D, 0.05, c; start = st, method = :paths, x0 = -3.0) atol = 1e-9
        @test GL.dist_quantile(D, 0.05, c; start = st, x0 = -3.0) == ql
        @test GL.dist_cdf(D, [ql], c; start = st, method = :line)[1] ≈ 0.05 atol = 1e-12
    end
    xs = [-3.0, -2.0, 0.5]
    @test GL.dist_cdf(GL.trend(2), xs, -5.0; verified = true) == GL.dist_cdf(GL.trend(2), xs, -5.0; method = :line)
    @test GL.dist_pdf(TRIG, xs, 0.0; verified = true, method = :line) == GL.dist_pdf(TRIG, xs, 0.0; method = :line)
    # non-finite points
    @test GL.dist_cdf(GL.trend(2), [-Inf, NaN, Inf], 0.0; method = :line) |> v -> v[1] == 0 && isnan(v[2]) && v[3] == 1
end

end # @testset "the line integral of rank one or two"
