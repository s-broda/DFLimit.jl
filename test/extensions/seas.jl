# The statistics of a pair of complex unit roots at a harmonic frequency (src/DFDet_seas.jl).  t3: the sum of the residues
# against the engine's inversion on paths of the doubled transform, the exponential law without deterministic terms, the
# elementary form at the unit root (spherical Bessel functions), the moments in the table of Nabeya (2001); the transform of
# (R, L, V) against its closed forms (no terms; seasonal dummies at the unit root; Nabeya's transforms with seasonal dummies
# and with seasonal trends) and against
# the engine's determinant at u = 0; t4 and F against the normal law and the convolution (t3^2 + chi^2_1)/2 without
# deterministic terms; the reference values of the prototype (df-deterministics, extensions/seasonal, table.txt: critical
# values and powers at four decimals); compositions with a volatility path; argument errors and type stability.
using DFLimit, Test, Logging
const DD = DFLimit.DFDet
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end
Phi(x) = DD.normcdf_(x)

"""the spherical Bessel function of order nu = -1..3 divided by its leading term at the origin: cos x, sin(x)/x, ..."""
function jhat(nu::Int, x::ComplexF64)
    sx = sin(x); cx = cos(x)
    nu == -1 && return cx
    nu == 0 && return sx / x
    nu == 1 && return 3 * (sx - x * cx) / x^3
    nu == 2 && return 15 * ((3 - x^2) * sx - 3x * cx) / x^5
    105 * ((15 - 6x^2) * sx - (15x - x^3) * cx) / x^7
end
"""t3 at the unit root for the polynomial seasonal trend of degree p = 0, 1, 2 from elementary functions: D(s,t) =
jhat_p {jhat_{p-1} - c_p s^2 jhat_{p+1}} at x = sqrt(t/2), c_p = 1/(4 (2p+1)(2p+3)); the roots +-s(t), the weight
1/(2 c_p jhat_p jhat_{p+1} s), M_V = 1/(jhat_{p-1} jhat_p); only the nodes of the outer contour CP are taken"""
function t3_unitroot(p::Int, xs::Vector{Float64}, CP)
    cp = 1 / (4 * (2p + 1) * (2p + 3))
    F = zeros(length(xs)); f = zeros(length(xs))
    pcs = [DD.xpieces(x, 0.0, 1.0) for x in xs]
    for k in eachindex(CP.t)
        t = CP.t[k]; x = sqrt(t / 2)
        jm = jhat(p - 1, x); j0 = jhat(p, x); jp = jhat(p + 1, x)
        s = sqrt(jm / (cp * jp)); real(s) < 0 && (s = -s)
        w = 1 / (2 * cp * j0 * jp * s); MV = 1 / (jm * j0)
        for ix in eachindex(xs)
            plus, minus = pcs[ix]
            _, M1, M2 = DD.ctr_moments(t, s, 0.0, xs[ix], 1.0, plus)
            gf = w * 2 * M2; gF = w * 2 * M1 / s
            if !isempty(minus)
                _, N1, N2 = DD.ctr_moments(t, -s, 0.0, xs[ix], 1.0, minus)
                gf += w * 2 * N2; gF += w * 2 * N1 / (-s)
                gF += MV * exp(-t * minus[1][1]^2) / t
            end
            F[ix] += CP.wt[k] * imag(gF * CP.dt[k]); f[ix] += CP.wt[k] * imag(gf * CP.dt[k])
        end
    end
    1 .- F, f
end
"""the transform without deterministic terms: D(s,u,t) = D_0(s, t + u^2/2), D_0(s,t) = e^c {cosh theta - (s + c) sinh(theta)/theta}"""
function d3none(s, u, t; c = 0.0)
    tt = t + u^2 / 2; θ = sqrt(complex(c^2 - 2tt))
    exp(c) * (cosh(θ) - (s + c) * (abs(θ) < 1e-4 ? 1 + θ^2 / 6 : sinh(θ) / θ))
end
"""seasonal dummies at the unit root: D = sinc(a/2) sinc(b/2) {1 - (2t + s^2 + u^2) Omega}, a, b = u +- sqrt(u^2 + 2t),
Omega = {-cot(a/2)/2 + 1/a + cot(b/2)/2 - 1/b}/(a - b)"""
function d3dummies(s, u, t)
    q = sqrt(complex(u^2 + 2t)); a = u + q; b = u - q
    sc(x) = abs(x) < 1e-4 ? 1 - x^2 / 24 : sin(x / 2) / (x / 2)
    g(x) = abs(x) < 1e-3 ? x / 12 + x^3 / 720 : 1 / x - cot(x / 2) / 2
    sc(a) * sc(b) * (1 - (2t + s^2 + u^2) * (g(a) - g(b)) / (a - b))
end
"""P(F <= x) without deterministic terms: t4 is standard normal and independent of t3, 2F = t3^2 + chi^2_1"""
function f_none(x::Float64, c::Float64; n::Int = 200)
    z, w = DD.glrule(n); r = sqrt(2x); a = (pi / 2) .* z; q = r .* cos.(a)
    Fp = hegy_t3_cdf(trend(-1), q, c); Fm = hegy_t3_cdf(trend(-1), -q, c)
    2 * sum((pi / 2) * w[i] * r * cos(a[i]) * exp(-(r * sin(a[i]))^2 / 2) / sqrt(2pi) * (Fp[i] - Fm[i]) for i in 1:n)
end

@testset "seasonal unit roots" begin
    xs = [-4.0, -2.5, -1.0, 0.0, 1.2]
    @testset "t3: the residues against the doubled transform on the paths" begin
        for D in (trend(-1), trend(0), trend(1), perron(:A, 0.4), perron(:C, 0.6)), (c, c0) in ((0.0, 0.0), (-5.0, 0.0), (-5.0, -2.0), (0.0, 1.5), (-20.0, 0.0))
            CP = DD.cpath_cached(DD.startmodel(D, c, :zero, 0.0), c)
            F2, f2 = DD.contour_eval(DD.t3_doubled(CP), xs; c0)[1:2]
            @test maximum(abs, hegy_t3_cdf(D, xs, c; c0) .- F2) < 1e-11
            @test maximum(abs, hegy_t3_pdf(D, xs, c; c0) .- f2) < 1e-11
        end
        @test abs(hegy_t3_cdf(trend(-1), [0.0])[1] - (1 - exp(-1))) < 1e-13          # |Z(1)|^2/2 is exponential
        for p in 0:2
            CP = DD.cpath_cached(DD.startmodel(trend(p), 0.0, :zero, 0.0), 0.0)
            Fe, fe = t3_unitroot(p, xs, CP)
            @test maximum(abs, hegy_t3_cdf(trend(p), xs) .- Fe) < 1e-12
            @test maximum(abs, hegy_t3_pdf(trend(p), xs) .- fe) < 1e-12
        end
        # far from the unit root, and the checked computation
        @test maximum(abs, hegy_t3_cdf(trend(0), xs, -50.0; verified = true) .- hegy_t3_cdf(trend(0), xs, -50.0)) < 1e-11
    end
    @testset "the moments of Nabeya (2001) from the densities" begin
        # (seasonal dummies: mean and standard deviation of t3, the standard deviation of t4, at eight decimals)
        z, w = DD.glrule(400)
        q = -12 .+ 24 .* z; f = hegy_t3_pdf(trend(0), q)
        m1 = 24 * sum(w .* q .* f); m2 = 24 * sum(w .* q .^ 2 .* f)
        @test abs(m1 + 1.96870124) < 1e-8
        @test abs(sqrt(m2 - m1^2) - 0.86627092) < 1e-8
        z, w = DD.glrule(160)
        q = 12 .* z; f4 = hegy_t4_pdf(trend(0), q)                       # (t4 is symmetric about zero)
        @test abs(sqrt(2 * 12 * sum(w .* q .^ 2 .* f4)) - 1.19122237) < 1e-8
        f4 = hegy_t4_pdf(trend(1), q)                                     # (dummies and seasonal trends)
        @test abs(sqrt(2 * 12 * sum(w .* q .^ 2 .* f4)) - 1.27390322) < 1e-8
        # the closed forms of the grid against the general transform (no seasonal terms at c = -5, dummies at the unit root)
        for (D, c) in ((trend(-1), -5.0), (trend(0), 0.0))
            G1 = DD.seas_grid_cached(D, c); G2 = DD.seas_grid(DD.SeasModel(DD.prep(D)), c)
            @test maximum(abs, G1.d0 ./ G2.d0 .- 1) < 1e-12
            @test maximum(abs, [DD.t4_both(G1, x)[1] - DD.t4_both(G2, x)[1] for x in (0.5, 2.0)]) < 1e-13
            @test maximum(abs, [DD.f_cdf1(G1, x) - DD.f_cdf1(G2, x) for x in (2.0, 9.0)]) < 1e-12
        end
    end
    @testset "the transform of (R, L, V)" begin
        pts = ((0.3, 0.0, 0.5), (0.2, 0.4, 0.3), (0.2 + 1.0im, 0.3im, 0.4 + 2.0im), (0.0, 2.0im, 1.0 + 30.0im))
        for p in (-1, 0, 1), c in (0.0, -3.0), (s, u, t) in pts
            SM = DD.SeasModel(DD.prep(trend(p)))
            f = DD.seas_dval(SM, s, u, t; c)
            # (the sign of theta: with -theta the exponentials of the covariance grow like e^{2 |Re theta|}, e^{12} at t = 1 + 30i)
            @test abs(f / DD.seas_dval(SM, s, u, t; c, sgn = -1) - 1) < (abs(t) > 10 ? 1e-9 : 1e-10)
            p == -1 && @test abs(f / d3none(s, u, t; c) - 1) < 1e-12
            (p == 0 && c == 0) && @test abs(f / d3dummies(s, u, t) - 1) < 1e-12
            (p >= 0 && c == 0) && @test abs(f - DD.seas_dval(SM, -s, u, t)) < 1e-11 * abs(f)     # even in s at the unit root
            u == 0 && @test abs(f / DD.Dval(DD.prep(trend(p)), c, s, t) - 1) < 1e-11
        end
        # at u = 0 the engine's determinant for designs with breaks; Nabeya's (2001) closed forms with seasonal dummies and
        # with seasonal trends
        for D in (perron(:A, 0.4), perron(:C, 0.6)), (s, t) in ((0.3, 0.5), (0.2 + 1.0im, 0.4 + 2.0im))
            @test abs(DD.seas_dval(DD.SeasModel(DD.prep(D)), s, 0.0, t; c = -2.0) / DD.Dval(DD.prep(D), -2.0, s, t) - 1) < 1e-11
        end
        let u2 = 0.2, u3 = 0.4, v = 0.3, q = sqrt(u3^2 + 2v), w = u2^2 + u3^2
            n8 = (1 + w / (2v)) * sin(q) / q - w / (2v^2) * (cos(u3) - cos(q))
            n9 = (6 / v^2 - (6 / v^3 + 18 / v^4) * w) * cos(u3) - 18u3 / v^4 * w * sin(u3) + (-6 / v^2 + (-12 / v^3 + 18 / v^4) * w) * cos(q) +
                 (-6 / v + (-3 / v^2 + 36 / v^3 + 18u3^2 / v^4) * w) * sin(q) / q
            @test abs(real(DD.seas_dval(DD.SeasModel(DD.prep(trend(0))), u2, u3, v)) / n8 - 1) < 1e-12
            @test abs(real(DD.seas_dval(DD.SeasModel(DD.prep(trend(1))), u2, u3, v)) / n9 - 1) < 1e-10
        end
    end
    @testset "t4 and F without deterministic terms" begin
        qs = [0.5, 1.645, 2.5]; xF = [1.0, 3.0, 6.0]
        @test hegy_t4_cdf(trend(-1), qs) == Phi.(qs)
        @test hegy_t4_quantile(trend(-1), 0.975) ≈ 1.959963984540054
        for c in (0.0, -5.0)
            G = DD.seas_grid_cached(trend(-1), c)
            @test maximum(abs, [DD.t4_both(G, q)[1] for q in qs] .- Phi.(qs)) < 1e-11              # the lines: normal
            @test maximum(abs, hegy_f_cdf(trend(-1), xF, c) .- [f_none(x, c) for x in xF]) < 1e-11  # (t3^2 + chi2_1)/2
        end
    end
    @testset "the prototype's table: critical values and the power of the 5% tests (four decimals)" begin
        for (p, r3, r4, rF, pw) in ((-1, [-2.5558, -2.2135, -1.9217, -1.5879], [1.2816, 1.6449, 1.9600, 2.3263], [2.4096, 3.1119, 3.8116, 4.7342], (0.7227, 0.0500, 0.4674)),
                                    (0, [-3.9089, -3.6038, -3.3442, -3.0477], [1.5384, 1.9603, 2.3193, 2.7284], [5.6303, 6.6455, 7.6020, 8.8074], (0.2017, 0.0409, 0.1725)),
                                    (1, [-4.7096, -4.4181, -4.1710, -3.8901], [1.6487, 2.0965, 2.4754, 2.9044], [8.5763, 9.7513, 10.8479, 12.2172], (0.1144, 0.0408, 0.1056)))
            D = trend(p)
            q3 = hegy_t3_quantile(D, [0.01, 0.025, 0.05, 0.10]; x0 = -2.0 - p)
            q4 = hegy_t4_quantile(D, [0.90, 0.95, 0.975, 0.99]); qF = hegy_f_quantile(D, [0.90, 0.95, 0.975, 0.99])
            @test maximum(abs, q3 .- r3) < 1e-4
            @test maximum(abs, q4 .- r4) < 1e-4
            @test maximum(abs, qF .- rF) < 1e-4
            @test abs(hegy_t3_cdf(D, [q3[3]], -5.0)[1] - pw[1]) < 1e-4
            @test abs(2 * (1 - hegy_t4_cdf(D, [q4[3]], -5.0)[1]) - pw[2]) < 1e-4
            @test abs(1 - hegy_f_cdf(D, [qF[2]], -5.0)[1] - pw[3]) < 1e-4
        end
        # the density of t4 against its distribution function; another grid (verified = true)
        h = 1e-3
        @test abs((hegy_t4_cdf(trend(1), [1.0 + h])[1] - hegy_t4_cdf(trend(1), [1.0 - h])[1]) / (2h) - hegy_t4_pdf(trend(1), [1.0])[1]) < 1e-6
        @test_logs min_level = Logging.Warn hegy_t4_cdf(trend(0), [1.0, 2.5], -5.0; verified = true)
        @test_logs min_level = Logging.Warn hegy_f_cdf(trend(0), [3.0, 9.0], -5.0; verified = true)
    end
    @testset "with a volatility path" begin
        V = volpath([0.5], [1.0, 3.0])
        for D in (trend(-1), trend(0), trend(1))
            VM = DD.vol_model(D, V, [0.0, 0.0])
            F2 = DD.contour_eval(DD.t3_doubled(DD.vol_path(VM)), xs)[1]
            @test maximum(abs, hegy_t3_cdf(D, xs; vol = V) .- F2) < 1e-11
        end
        @test hegy_t3_cdf(trend(0), xs; vol = volpath(Float64[])) == hegy_t3_cdf(trend(0), xs)
        @test abs(hegy_t3_cdf(trend(0), [-2.0], [-5.0, -5.0]; vol = volpath([0.3], [1.0, 1.0]))[1] - hegy_t3_cdf(trend(0), [-2.0], -5.0)[1]) < 1e-12
    end
    @testset "argument errors" begin
        @test occursin("polynomial seasonal trends", errmsg(() -> hegy_t4_cdf(perron(:A, 0.5), [1.0])))
        @test occursin("polynomial seasonal trends", errmsg(() -> hegy_f_cdf(trend(2), [1.0])))
        @test occursin("polynomial seasonal trends", errmsg(() -> hegy_t4_quantile(detrend_first(trend(1)), 0.5)))
        @test occursin("at most 0", errmsg(() -> hegy_t3_cdf(trend(0), [1.0], 1.0)))
        @test occursin("at most 0", errmsg(() -> hegy_t4_cdf(trend(0), [1.0], 0.5)))
        @test occursin("at least -20", errmsg(() -> hegy_f_cdf(trend(0), [1.0], -30.0)))
        @test occursin("p must lie in (0, 1)", errmsg(() -> hegy_f_quantile(trend(0), 1.0)))
        @test occursin("c0 must be", errmsg(() -> hegy_t3_cdf(trend(0), [1.0]; c0 = Inf)))
    end
    @testset "type stability" begin
        @inferred hegy_t3_cdf(trend(0), [-2.0])
        @inferred hegy_t3_quantile(trend(0), 0.05)
        @inferred hegy_t4_cdf(trend(0), [1.0])
        @inferred hegy_t4_quantile(trend(0), 0.9)
        @inferred hegy_f_cdf(trend(0), [5.0])
        @inferred hegy_f_quantile(trend(0), 0.9)
        SM = DD.SeasModel(DD.prep(trend(1))); W = DD.SeasWork(SM.P)
        @inferred DD.seas_parts(SM, 0.0, 1.0im, 1.0 + 2.0im, W)
    end
end
