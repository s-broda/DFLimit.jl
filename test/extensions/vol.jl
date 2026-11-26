# The volatility and persistence regimes (src/DFDet_vol.jl): the transform against the engine's at a constant variance,
# against the product of the transfer matrices of the regimes without deterministic terms and against the finite-sample
# moment generating function of the regression; the distribution function against the engine's with one regime and with a
# dummy boundary; the reference values of the prototype (df-deterministics, extensions/volatility, CHECKS.md: the table
# of the note at eight decimals, the quantiles of its simulations at four, the power under changing persistence);
# the checked computation, the density against the distribution function, argument errors and type stability.
using DFLimit, Test, LinearAlgebra, Logging
const DD = DFLimit.DFDet
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end
rel(a, b) = abs(a - b) / abs(b)

"""the determinant without deterministic terms by the transfer matrices of the regimes: D = e_1' T_1 ... T_m (1, s)',
T_i = e^{c_i l_i} [cosh(theta_i l_i) - (c_i/theta_i) sinh(theta_i l_i), -(omega_i^2/theta_i) sinh(theta_i l_i);
(2t/theta_i) sinh(theta_i l_i), cosh(theta_i l_i) + (c_i/theta_i) sinh(theta_i l_i)] (the backward equation)"""
function transfer_d0(V::VolPath, cr::Vector{Float64}, s::Number, t::Number)
    x = ComplexF64[1.0, s]
    for i in length(V.om):-1:1
        l = V.lam[i+1] - V.lam[i]; c = cr[i]; w2 = V.om[i]^2
        θ = sqrt(complex(c^2 - 2t * w2))
        ch = cosh(θ * l); sh = abs(θ * l) < 1e-6 ? l * (1 + (θ * l)^2 / 6) : sinh(θ * l) / θ
        x = exp(c * l) .* [(ch - c * sh) * x[1] - w2 * sh * x[2], 2t * sh * x[1] + (ch + c * sh) * x[2]]
    end
    x[1]
end
"""the exact joint moment generating function of the numerator and the denominator of the regression in a sample of n
(Gaussian innovations; the quadratic forms of the regression, real s and t)"""
function discrete_mgf(D::Determ, V::VolPath, cr::Vector{Float64}, n::Int, s::Real, t::Real)
    reg(r) = (i = 1; while i < length(V.om) && r > V.lam[i+1]; i += 1; end; i)
    om = [V.om[reg(k / n)] for k in 1:n]; ph = [1 + cr[reg(k / n)] / n for k in 1:n]
    S = zeros(n, n)
    for k in 1:n
        S[k, k] = om[k]
        for j in k+1:n; S[j, k] = ph[j] * S[j-1, k]; end
    end
    Sl = zeros(n, n); Sl[2:n, :] .= S[1:n-1, :]
    X = DD.regmat(D, n)
    PSl = size(X, 2) == 0 ? Sl : (Q = Matrix(qr(X).Q); Sl .- Q * (transpose(Q) * Sl))
    N = transpose(PSl) * (S .- Sl) ./ n
    A = (N .+ transpose(N)) ./ 2; T = transpose(PSl) * PSl ./ n^2
    exp(-logdet(cholesky(Symmetric(I - 2s .* A .- 2t .* T))) / 2)
end

const PTS = ((0.3, -2.0), (-0.4, 0.5), (0.2 + 0.7im, 0.3 - 4.0im), (0.1, -50.0 + 20im))
const V2 = volpath([0.3], [1.0, 3.0])
const V3 = volpath([0.2, 0.7], [2.0, 1.0, 0.5]); const C3 = [-3.0, 0.0, -10.0]

@testset "volatility regimes" begin
    @testset "the path" begin
        V = volpath([0.25, 0.5], [1.0, 2.0, 4.0])
        @test sum(V.om .^ 2 .* diff(V.lam)) ≈ 1 atol = 1e-15
        @test V.om[2] / V.om[1] ≈ 2 && V.lam == [0.0, 0.25, 0.5, 1.0]
        @test DD.nregimes(volpath(Float64[])) == 1 && volpath(Float64[]).om == [1.0]
        @test occursin("lie in (0, 1)", errmsg(() -> volpath([1.0])))
        @test occursin("lie in (0, 1)", errmsg(() -> volpath([0.0])))
        @test occursin("strictly increasing", errmsg(() -> volpath([0.5, 0.5])))
        @test occursin("one standard deviation omega per regime", errmsg(() -> volpath([0.5], [1.0])))
        @test occursin("finite and positive", errmsg(() -> volpath([0.5], [1.0, 0.0])))
        @test occursin("finite and positive", errmsg(() -> volpath([0.5], [1.0, Inf])))
    end
    @testset "the transform at a constant variance is the engine's" begin
        for D in (trend(-1), trend(0), trend(1), perron(:A, 0.3), perron(:C, 0.6)), c in (0.0, -5.0)
            VM = DD.vol_model(D, volpath([0.4], [1.0, 1.0]), [c, c]); P = DD.prep(D)
            for (s, t) in PTS
                @test rel(DD.vol_dval(VM, s, t), DD.Dval(P, c, s, t)) < 1e-10
            end
        end
    end
    @testset "no deterministic terms: the product of the transfer matrices" begin
        for (V, cr) in ((V2, [0.0, 0.0]), (V3, C3), (volpath([0.1, 0.3, 0.6, 0.8], [1.0, 2.0, 0.5, 3.0, 1.5]), [-1.0, 0.0, -4.0, -2.0, 0.0])), (s, t) in (PTS..., (0.0, 3.0))
            @test rel(DD.vol_dval(DD.vol_model(trend(-1), V, cr), s, t), transfer_d0(V, cr, s, t)) < 1e-11
        end
    end
    @testset "the signs of theta do not matter" begin
        VM = DD.vol_model(trend(1), V3, C3)
        @test VM.np == 3
        for sg in ([-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, -1]), (s, t) in ((0.3, -1.0), (0.2 + 0.5im, 0.4 + 3im))
            @test rel(DD.vol_dval(VM, s, t; sgn = sg), DD.vol_dval(VM, s, t)) < 1e-8
        end
    end
    @testset "the transform against the finite-sample moment generating function" begin
        # (Richardson's extrapolation in 1/n of the exact transform at n = 200, 400, 800)
        for (D, V, cr) in ((trend(0), V2, [0.0, 0.0]), (trend(1), V3, C3)), (s, t) in ((0.3, -2.0), (-0.5, 0.4))
            ex = real(exp(-s / 2) / sqrt(DD.vol_dval(DD.vol_model(D, V, cr), s, t)))
            m1, m2, m4 = (discrete_mgf(D, V, cr, n, s, t) for n in (200, 400, 800))
            @test abs((4 * (2m4 - m2) - (2m2 - m1)) / 3 - ex) < 2e-5
        end
    end
    @testset "the reversal at the unit root" begin
        # D is not even in s under a variance break; the reversed design with the reversed path has D(-s, t) = D(s, t)
        VM = DD.vol_model(trend(0), V2, [0.0, 0.0]); VR = DD.vol_model(trend(0), volpath([0.7], [3.0, 1.0]), [0.0, 0.0])
        @test abs(DD.vol_dval(VM, 0.5, -2.0) - 1.6815863197) < 1e-9
        @test abs(DD.vol_dval(VM, -0.5, -2.0) - 2.0691752124) < 1e-9
        @test abs(DD.vol_dval(VR, -0.5, -2.0) - 1.6815863197) < 1e-9
        @test abs(DD.vol_dval(VM, 1.0, 1.0) - 0.4215903315) < 1e-9
        rts = sort(real.(1 ./ DD.mgfparts(VM, 0.0, 0.5 + 0im, DD.MWork(VM))[3]))
        @test abs(rts[1] + 7.1969) < 1e-4 && abs(rts[2] - 2.4178) < 1e-4
        VT = DD.vol_model(trend(1), V2, [0.0, 0.0])
        @test abs(DD.vol_dval(VT, 0.5, -2.0) - 1.2374269682) < 1e-9 && abs(DD.vol_dval(VT, -0.5, -2.0) - 1.3528083151) < 1e-9
    end
    @testset "t_*" begin
        # (at a constant variance the engine's; with model C at 1/2 a double zero, 8 pi^2)
        for (D, c) in ((trend(0), 0.0), (trend(1), -5.0), (perron(:C, 0.5), 0.0), (trend(-1), -100.0))
            VM = DD.vol_model(D, volpath([0.4], [1.0, 1.0]), [c, c])
            ts, m = DD.tstar(VM, c); te, me = DD.tstar(DD.model(DD.prep(D)), c)
            @test abs(ts - te) < (me == 2 ? 1e-6 : 1e-10) * te
            @test m == me
        end
    end
    xs = [-4.0, -2.86, -1.0, 0.5]
    @testset "one regime is the engine itself" begin
        for D in (trend(0), perron(:B, 0.5)), c in (0.0, -10.0)
            @test dist_cdf(D, xs, c; vol = volpath(Float64[])) == dist_cdf(D, xs, c)
            @test dist_pdf(D, xs, c; vol = volpath(Float64[])) == dist_pdf(D, xs, c)
            @test dist_cdf(D, xs, [c]; vol = volpath(Float64[])) == dist_cdf(D, xs, c)
        end
        @test dist_quantile(trend(1), 0.05, 0.0; vol = volpath(Float64[])) == dist_quantile(trend(1), 0.05, 0.0)
    end
    @testset "a dummy boundary at a constant variance: the engine's paths to rounding" begin
        for D in (trend(-1), trend(0), trend(1), perron(:C, 0.6)), c in (0.0, -10.0, -100.0)
            V = volpath([0.4], [1.0, 1.0])
            @test maximum(abs, dist_cdf(D, xs, c; vol = V) .- dist_cdf(D, xs, c; method = :paths)) < 1e-12
            @test maximum(abs, dist_pdf(D, xs, c; vol = V) .- dist_pdf(D, xs, c; method = :paths)) < 1e-12
        end
        # (and as a vector of equal c, with c0)
        V = volpath([0.25, 0.75], [1.0, 1.0, 1.0])
        @test maximum(abs, dist_cdf(trend(0), xs, [-5.0, -5.0, -5.0]; vol = V, c0 = -2.0) .- dist_cdf(trend(0), xs, -5.0; c0 = -2.0, method = :paths)) < 1e-12
    end
    @testset "the prototype's table: F at the constant-variance 5% values (eight decimals)" begin
        # (the points are the constant-variance critical values rounded to six decimals, as the prototype took them)
        cvs = Dict(-1 => -1.940873, 0 => -2.861569, 1 => -3.410454)
        for (p, del, lam, ref) in ((-1, 0.2, 0.1, 0.09286228), (-1, 5.0, 0.9, 0.13487886), (-1, 1 / 3, 0.5, 0.06764708),
                                   (0, 0.2, 0.1, 0.51947131), (0, 5.0, 0.1, 0.02151061), (0, 3.0, 0.1, 0.02278491), (0, 1 / 3, 0.9, 0.05823416),
                                   (1, 0.2, 0.5, 0.13496040), (1, 5.0, 0.9, 0.13979272), (1, 3.0, 0.1, 0.03014518))
            @test abs(dist_cdf(trend(p), [cvs[p]], 0.0; vol = volpath([lam], [1.0, del]))[1] - ref) < 1e-8
        end
    end
    @testset "the prototype's quantiles (four decimals) and the power under changing persistence" begin
        ps = [0.01, 0.05, 0.10, 0.50, 0.90]
        for (D, V, c, ref) in ((trend(0), volpath([0.1], [3.0, 1.0]), 0.0, [-5.5655, -4.3802, -3.8224, -2.2635, -1.0436]),
                               (trend(0), volpath([0.1], [1.0, 3.0]), 0.0, [-3.1355, -2.5800, -2.3121, -1.4497, -0.3326]),
                               (trend(1), volpath([0.2, 0.7], [2.0, 1.0, 0.5]), C3, [-4.4726, -3.7390, -3.3627, -2.1126, -1.0092]),
                               (perron(:C, 0.4), volpath([0.6], [3.0, 1.0]), -5.0, [-5.2725, -4.6747, -4.3647, -3.3229, -2.3449]))
            @test maximum(abs, dist_quantile(D, ps, c; vol = V) .- ref) < 1e-4
        end
        cv = dist_quantile(trend(0), 0.05)
        @test abs(dist_cdf(trend(0), [cv], [0.0, -20.0]; vol = volpath([0.5]))[1] - 0.1319) < 1e-4
        @test abs(dist_cdf(trend(0), [cv], [-20.0, 0.0]; vol = volpath([0.5]))[1] - 0.1077) < 1e-4
    end
    @testset "the classical functions take vol" begin
        x = [-3.0, -2.0]
        a = ltratio_cdf(x, "DU", -5; vol = V2)
        @test a == ltratio_cdf(x, trend(0), -5; vol = V2) == dist_cdf(trend(0), x, -5.0; vol = V2)
        @test ltratio_pdf(-2.0, "DT", 0; vol = V2) == dist_pdf(trend(1), [-2.0], 0.0; vol = V2)[1]
        @test ltratio_cdf(x, "DU", [0.0, -20.0]; vol = volpath([0.5])) == dist_cdf(trend(0), x, [0.0, -20.0]; vol = volpath([0.5]))
        q = ltratio_icdf(0.05, "DU", 0; vol = V2)
        @test abs(ltratio_cdf(q, "DU", 0; vol = V2) - 0.05) < 1e-9
        @test abs(ltratio_icdf(0.05, trend(1), [-3.0, 0.0, -10.0]; vol = V3) + 3.7390) < 1e-4
        @test ltratio_cdf(x, "DU", -5; vol = nothing) == ltratio_cdf(x, "DU", -5)
    end
    @testset "the checked computation, the density, non-finite points" begin
        V = volpath([0.1], [1.0, 3.0]); x = [-3.0, -2.0, -0.5]
        @test maximum(abs, dist_cdf(trend(0), x, -5.0; vol = V, verified = true) .- dist_cdf(trend(0), x, -5.0; vol = V)) < 1e-9
        @test maximum(abs, dist_pdf(trend(0), x, -5.0; vol = V, verified = true) .- dist_pdf(trend(0), x, -5.0; vol = V)) < 1e-9
        h = 1e-3
        for (D, V, c) in ((trend(-1), volpath([0.7], [1.0, 3.0]), 0.0), (trend(1), V3, C3))
            F = dist_cdf(D, [-2.0 - h, -2.0 + h], c; vol = V)
            @test abs((F[2] - F[1]) / (2h) - dist_pdf(D, [-2.0], c; vol = V)[1]) < 1e-6
        end
        @test dist_cdf(trend(0), [-Inf, Inf, NaN], -5.0; vol = V)[1:2] == [0.0, 1.0] && isnan(dist_cdf(trend(0), [NaN], -5.0; vol = V)[1])
    end
    @testset "argument errors" begin
        x = [-2.0]
        @test occursin("needs a volatility path", errmsg(() -> dist_cdf(trend(0), x, [0.0, -1.0])))
        @test occursin("one value per regime of the volatility path (2), not 3", errmsg(() -> dist_cdf(trend(0), x, [0.0, -1.0, 0.0]; vol = V2)))
        @test occursin("at most 0", errmsg(() -> dist_cdf(trend(0), x, 1.0; vol = V2)))
        @test occursin("at most 0", errmsg(() -> dist_cdf(trend(0), x, [0.0, 1.0]; vol = V2)))
        @test occursin("at least -480", errmsg(() -> dist_cdf(trend(0), x, -500.0; vol = V2)))
        @test occursin("zero start only", errmsg(() -> dist_cdf(trend(0), x, -5.0; vol = V2, start = :stationary)))
        @test occursin("zero start only", errmsg(() -> dist_cdf(trend(0), x, -5.0; vol = V2, start = :fixed, gamma0 = 1.0)))
        @test occursin("GLS detrending", errmsg(() -> dist_cdf(trend(1), x, -5.0; vol = V2, gls = -13.5)))
        @test occursin("detrend_first", errmsg(() -> dist_cdf(detrend_first(trend(1)), x, -5.0; vol = V2)))
        @test occursin("method must be :auto or :paths", errmsg(() -> dist_cdf(trend(0), x, -5.0; vol = V2, method = :line)))
        @test occursin("rho and vol together", errmsg(() -> dist_cdf(trend(0), x, -5.0; vol = V2, rho = 0.5)))
        @test occursin("must be a volatility path", errmsg(() -> dist_cdf(trend(0), x, -5.0; vol = [0.3])))
        @test occursin("engine = :classical", errmsg(() -> ltratio_cdf(x, "DU", 0; vol = V2, engine = :classical)))
        @test occursin("needs a volatility path", errmsg(() -> ltratio_cdf(x, "DU", [0.0, -1.0])))
    end
    @testset "type stability" begin
        VM = DD.vol_model(trend(1), V2, [0.0, -5.0]); W = DD.MWork(VM)
        @inferred DD.mgfparts(VM, 0.0, 1.0 + 2.0im, W)
        @inferred DD.sigma_vol!(W, VM)
        @inferred dist_cdf(trend(0), [-2.0], 0.0; vol = V2)
        @inferred dist_pdf(trend(0), [-2.0], [0.0, -1.0]; vol = V2)
        @inferred dist_quantile(trend(0), 0.05, 0.0; vol = V2)
    end
end
