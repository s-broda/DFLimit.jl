# The tests of DFLimit: the engine's values against values stored by the paper's code, at the tolerances of the existing
# checks (test/engine/quickcheck.jl and test/engine/test_dfexact.jl, the C engine's tests c/test of dflimit-matlab, the
# MATLAB suite matlab/ltratio_test.m there and the tests of the R package dflimit), closed forms, and the argument errors;
# then the engine's own checks, test/engine/quickcheck.jl and test/engine/test_dfexact.jl.  The stored values are in
# test/ref/ (the first line of each file says where it comes from) and test/engine/.  About six minutes on a laptop; run with
#   julia --project -e 'using Pkg; Pkg.test()'                         (one thread)
#   JULIA_NUM_THREADS=auto julia --project -e 'using Pkg; Pkg.test()'  (all performance cores)
using DFLimit, Test, Printf, Logging
using SpecialFunctions: erf

const D = DFLimit.DFExact
const REF = joinpath(@__DIR__, "ref")

readref(name) = [split(l) for l in eachline(joinpath(REF, name)) if !startswith(l, "#") && !isempty(strip(l))]
num(p) = parse.(Float64, p)
casecode(m, st) = m == 4 ? "DG" : (st == 1 ? "S" : "D") * "KUT"[Int(m)]
# the message of the ArgumentError that f() throws ("" if it does not throw, "other: ..." for another exception)
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end
# f() without a warning: the conditional integral warns where its error estimate is not met
nowarn(f) = @test_logs min_level = Logging.Warn f()

const T0 = time()
@printf("DFLimit tests: Julia %s, %d threads, %d workers, vector math: %s\n", VERSION, Threads.nthreads(), D.NWORKERS[], D.vecmath())

@testset "DFLimit" begin

@testset "exports" begin
    ex = [:ltratio_cdf, :ltratio_pdf, :ltratio_icdf, :ltratio_mom, :adfmax_cdf, :adfmax_pdf, :adfmax_icdf, :coef_cdf,
          :coef_icdf, :ltratio_region, :ltratio_lincdf, :power_envelope, :ltratio_bisect]
    @test all(n -> Base.isexported(DFLimit, n), ex)
    @test all(n -> getfield(DFLimit, n) === getfield(D, n), ex)
    @test Set(filter(!=(:DFExact), names(D))) ⊆ Set(names(DFLimit))
    @test D.cdf isa Function && D.tr_limdist isa Function && D.tr_adfmax isa Function
end

# the twelve points of test/engine/python_exact_reference.json: the line integral at c0 = 0 (quickcheck.jl, section 2)
@testset "stored reference values (line integral, models 1-3)" begin
    for p in readref("exact_ref.txt")
        m = parse(Int, p[1]); c, q, Fr, fr = num(p[2:5])
        @test D.cdf(q, m; c) ≈ Fr atol = 1e-12
        @test D.density(q, m; c) ≈ fr rtol = 1e-10
    end
end

# the C engine's criterion (c/test, tests/testthat of dflimit): F to 1e-11, f to 1e-9 relative or 1e-11 absolute
@testset "line integral at c0 = 0: the six cases and DG" begin
    for p in readref("single_ref.txt")
        m, c, st, x, Fr, fr = num(p); cas = casecode(m, st)
        @testset "$cas c = $c x = $x" begin
            @test ltratio_cdf(x, cas, c; method = :single) ≈ Fr atol = 1e-11
            @test ltratio_pdf(x, cas, c; method = :single) ≈ fr rtol = 1e-9 atol = 1e-11
        end
    end
end

@testset "line integral with a local null value c0" begin
    for p in readref("gen_ref.txt")
        m, c, c0, st, x, Fr, fr = num(p); cas = casecode(m, st)
        @testset "$cas c = $c c0 = $c0 x = $x" begin
            @test ltratio_cdf(x, cas, c; c0, method = :single) ≈ Fr atol = 1e-11
            @test ltratio_pdf(x, cas, c; c0, method = :single) ≈ fr rtol = 1e-9 atol = 1e-11
        end
    end
end

@testset "line integral with a fixed start g0" begin
    for p in readref("fixed_ref.txt")
        m, c, c0, g0, st, x, Fr, fr = num(p); cas = casecode(m, st)
        @testset "$cas c = $c c0 = $c0 g0 = $g0 x = $x" begin
            @test ltratio_cdf(x, cas, c; c0, g0, method = :single) ≈ Fr atol = 1e-11
            @test ltratio_pdf(x, cas, c; c0, g0, method = :single) ≈ fr rtol = 1e-9 atol = 1e-11
        end
    end
end

# the contour integral (a fixed start in cases DU, DT; the choice of :auto there): F, f to max(1e-12, the stored error
# estimate), no warning; :auto gives the same value (above 1e-3, where it does not recompute by the conditional integral)
@testset "contour integral with a fixed start g0" begin
    for p in readref("contour_ref.txt")
        m, c, c0, g0, x, Fr, fr, eF, ef = num(p); cas = casecode(m, 0)
        @testset "$cas c = $c c0 = $c0 g0 = $g0 x = $x" begin
            # verified: the stored values (the checked computation); the default (one evaluation) within 5e-9
            F = nowarn(() -> ltratio_cdf(x, cas, c; c0, g0, method = :contour, verified = true)); f = nowarn(() -> ltratio_pdf(x, cas, c; c0, g0, method = :contour, verified = true))
            @test F ≈ Fr atol = max(1e-12, eF)
            @test f ≈ fr atol = max(1e-12, ef)
            (F >= 1e-3 && f >= 1e-3) && @test ltratio_cdf(x, cas, c; c0, g0, verified = true) == F
            @test ltratio_cdf(x, cas, c; c0, g0, method = :contour) ≈ Fr atol = 5e-9
            @test ltratio_pdf(x, cas, c; c0, g0, method = :contour) ≈ fr atol = 5e-9
        end
    end
    @test errmsg(() -> ltratio_cdf(-2.0, "DK", -5; g0 = 1, method = :contour)) == D.CONTOUR_MSG
    @test errmsg(() -> ltratio_cdf(-2.0, "DU", -5; method = :contour)) == D.CONTOUR_MSG
end

# the C engine's criterion for the conditional integral with verified = true: F to 1e-10, f to 1e-8 relative or 1e-10
# absolute, no warning; the default (one evaluation) to 5e-7 in F and 1e-5 absolute or 1e-6 relative in f, outside the
# regimes in which it warns (DFExact.brittle_msg)
@testset "conditional integral" begin
    for p in readref("dbl_ref.txt")
        m, c, c0, g0, st, x, Fr, fr = num(p); cas = casecode(m, st)
        @testset "$cas c = $c c0 = $c0 g0 = $g0 x = $x" begin
            @test nowarn(() -> ltratio_cdf(x, cas, c; c0, g0, method = :double, verified = true)) ≈ Fr atol = 1e-10
            @test nowarn(() -> ltratio_pdf(x, cas, c; c0, g0, method = :double, verified = true)) ≈ fr rtol = 1e-8 atol = 1e-10
            if D.brittle_msg(Int(m), st == 1, c, c0, g0, Set([:double])) === nothing
                @test ltratio_cdf(x, cas, c; c0, g0, method = :double) ≈ Fr atol = 5e-7
                @test ltratio_pdf(x, cas, c; c0, g0, method = :double) ≈ fr rtol = 1e-6 atol = 1e-5
            end
        end
    end
    x = [-3.0, -1.5, 0.5]                                   # the two representations, and the names :line, :conditional
    @test ltratio_cdf(x, "DU", -2; method = :line) == ltratio_cdf(x, "DU", -2; method = :single)
    @test ltratio_cdf(x, "DU", -2; method = :conditional) == ltratio_cdf(x, "DU", -2; method = :double)
    @test ltratio_cdf(x, "DU", -2; method = :line) ≈ ltratio_cdf(x, "DU", -2; method = :conditional) atol = 1e-10
end

# the default (one evaluation) and verified = true (the checked computation before November 2026: stored values of that
# engine); the warnings of the brittle regimes where they should fire, once per call, and not elsewhere
@testset "one evaluation and verified" begin
    for (cas, c, c0, g0, x, F, f) in (("DU", -5.0, 2.0, 0.0, -3.0, 0.1976948237625359, 0.4648778393052269),
                                      ("DK", -10.0, -5.0, 0.0, -2.0, 0.16726497409337868, 0.3125724838563969),
                                      ("DT", -5.0, 0.0, 2.0, -3.0, 0.28899703887072636, 0.4636733678782745),
                                      ("SU", -5.0, -5.0, 0.0, -1.0, 0.44272919976045844, 0.4264723317698583))
        @test ltratio_cdf(x, cas, c; c0, g0, verified = true) ≈ F atol = 1e-14
        @test ltratio_pdf(x, cas, c; c0, g0, verified = true) ≈ f atol = 1e-14
        @test ltratio_cdf(x, cas, c; c0, g0) ≈ F atol = 1e-9
        @test ltratio_pdf(x, cas, c; c0, g0) ≈ f atol = 1e-9
    end
    @test_logs (:warn, r"one evaluation") ltratio_cdf(-3.0, "DU", -5; c0 = 2.0, method = :double)
    @test_logs (:warn, r"one evaluation") ltratio_cdf(-5.0, "DU", -20; c0 = 2.0, g0 = 0.5)
    @test_logs (:warn, r"one evaluation") ltratio_icdf([0.05, 0.5], "DU", -20; c0 = 2.0, g0 = 0.5)
    @test_logs min_level = Logging.Warn ltratio_cdf(-3.0, "DU", -5; c0 = 2.0)
    @test_logs min_level = Logging.Warn ltratio_cdf(-3.0, "DU", -5; c0 = 2.0, method = :double, verified = true)
    @test_logs min_level = Logging.Warn ltratio_cdf(-2.0, "DU", -5; g0 = 1.5)
    @test_logs (:warn, r"contour integral by one evaluation fails") ltratio_cdf(-8.4, "DU", 2; c0 = 5.0, g0 = 1.5, method = :contour)
    # case DU with a fixed start, c0 > 3 at c > 1.5: the contour integral fails there, :auto takes the conditional integral
    Fa = @test_logs min_level = Logging.Warn ltratio_cdf(-8.4075, "DU", 2; c0 = 5.0, g0 = 1.5)
    @test Fa ≈ ltratio_cdf(-8.4075, "DU", 2; c0 = 5.0, g0 = 1.5, method = :double, verified = true) atol = 1e-9
end

@testset "moments" begin
    for p in readref("mom_ref.txt")
        m, c, c0, g0, st = num(p[1:5]); mr = num(p[6:9]); cas = casecode(m, st)
        @test maximum(abs.(ltratio_mom(4, cas, c; c0, g0) ./ mr .- 1)) < 1e-10
    end
    @test ltratio_mom(2, "DU")[2] ≈ 3.05599535 atol = 1e-8                  # Nabeya (1999); quickcheck.jl, section 8
end

# far from the unit root: the lines of test/engine/largec_ref.txt that quickcheck.jl checks (section 15), the line integral to
# 1e-11 and the automatic method to 1e-10; the moments to 1e-10 relative; the 5% quantile of DU to 1e-9
@testset "far from the unit root (c = -100, -480)" begin
    for p in readref("largec_ref.txt")
        cas = String(p[2])
        if p[1] == "val"
            c, c0, z, x, Fr, fr = num(p[3:8])
            @test ltratio_cdf(x, cas, c; c0, method = :single) ≈ Fr atol = 1e-11
            @test ltratio_pdf(x, cas, c; c0, method = :single) ≈ fr atol = 1e-11
            @test ltratio_cdf(x, cas, c; c0) ≈ Fr atol = 1e-10
            @test ltratio_pdf(x, cas, c; c0) ≈ fr atol = 1e-10
        elseif p[1] == "mom"
            c, c0 = num(p[3:4]); mr = num(p[5:8])
            @test maximum(abs.(ltratio_mom(4, cas, c; c0) ./ mr .- 1)) < 1e-10
        else
            c, c0, pr, qr = num(p[3:6])
            q = ltratio_icdf(pr, cas, c; c0)
            @test q ≈ qr atol = 1e-9
            @test ltratio_cdf(q, cas, c; c0) ≈ pr atol = 1e-9
        end
    end
    @test ltratio_cdf(-15.446799628855823, "DG", -480) ≈ 0.49999999999999922 atol = 1e-11    # quickcheck.jl, section 18
    @test ltratio_pdf(-15.446799628855823, "DG", -480) ≈ 0.77592587896902709 atol = 1e-11
end

@testset "automatic method, closed forms and published values" begin
    @test ltratio_cdf(-2.86157, "DU", -5) ≈ 0.116513562093 rtol = 1e-9        # the R package's stored value
    @test ltratio_cdf(-2.86157, "SU", -5) ≈ 0.125 atol = 5e-4                  # the paper's table (quickcheck.jl, section 5)
    @test ltratio_cdf(0, "DK", 2; c0 = 2) ≈ 0.650436 atol = 1e-6               # Larsson's table (section 9)
    @test D.cdf(0.0, 2; c = 10.0, c0 = 5.0) ≈ 0.000302655553129 atol = 1e-11   # a one-dimensional inversion (section 9)
    @test ltratio_cdf(-2, "DU", -5; g0 = 1.5) ≈ 0.864977240077167 atol = 1e-11  # the MATLAB programs (section 11)
    @test ltratio_cdf(-3.0, "DT", -5; g0 = 2) ≈ 0.2889970388707237 atol = 1e-11 # c/test/fixed_ref.txt (section 16)
    let c = -5.0, g0 = 1.5, v = (exp(-10) - 1) / (-10), m = 1.5 * exp(-5), r = sqrt(1 + 1.5^2)   # P[J(1)^2 <= 1 + g0^2]
        @test ltratio_cdf(0.0, "DK", c; g0) ≈ (erf((r - m) / sqrt(2v)) + erf((r + m) / sqrt(2v))) / 2 atol = 1e-11
    end
    lamg = D.lam_gls(D.CBAR_GLS)                                               # case DG at c = 0: P{V(1)^2 <= 1}
    @test ltratio_cdf(0.0, "DG", 0.0) ≈ erf(1 / sqrt(0.4 * (1 - lamg)^2)) atol = 1e-12
    F = ltratio_cdf([-Inf, 0.0, Inf, NaN], "DK", -5)                           # non-finite x; F(0) of DK in closed form
    @test F[1] == 0 && F[3] == 1 && isnan(F[4])
    @test F[2] ≈ erf(1 / sqrt(2 * (exp(-10) - 1) / (-10))) atol = 1e-12
    @test ltratio_pdf([-Inf, Inf], "DK") == [0.0, 0.0]
    @test isinf(ltratio_pdf(-2.0, "DK"; c0 = 2.0))                             # f(-sqrt(2 c0)) = Inf in case DK
    @test size(ltratio_cdf([-1.0 0.0; 0.5 1.0], "DK")) == (2, 2)
    @test ltratio_cdf(Float64[], "DU") == Float64[]
    @test ltratio_cdf(-1, "dk") == ltratio_cdf(-1.0, "DK")
end

@testset "quantiles" begin
    @test ltratio_icdf(0.05, "DU") ≈ -2.86156886 atol = 1e-7                  # quickcheck.jl, section 12
    @test D.quantile(0.05, 2) ≈ -2.861568864 atol = 1e-8                      # section 4
    @test ltratio_icdf(0.05, "DK") ≈ -1.94087 atol = 1e-5                     # the published 5% critical value
    for (q5, cas) in ((-1.94087325, "DK"), (-2.86156886, "DU"), (-3.41045363, "DT"))   # nine-digit 5% points (ltratio_test.m)
        @test ltratio_cdf(q5, cas) ≈ 0.05 atol = 2e-9
    end
    @test ltratio_icdf(0.05, "DG") ≈ -2.8456308262 atol = 1e-8               # ltratio_test.m
    q = ltratio_icdf([0.01, 0.05, 0.10], "DU")
    @test ltratio_cdf(q, "DU") ≈ [0.01, 0.05, 0.10] atol = 1e-9
    q = ltratio_icdf([0.1, 0.9], "ST", -10)
    @test ltratio_cdf(q, "ST", -10) ≈ [0.1, 0.9] atol = 1e-9
end

# the application functions against the stored values of c/test/app_ref.txt (as in the R package: F, f, P to 1e-10),
# quickcheck.jl (section 19) and the MATLAB suite (matlab/ltratio_test.m of dflimit-matlab, section 10)
@testset "ADFmax" begin
    for p in readref("app_ref.txt")
        p[1] == "adfmax" || continue
        m, st, c, x, Fr, fr = num(p[2:7]); cas = casecode(m, st)
        @test adfmax_cdf(x, cas, c) ≈ Fr atol = 1e-10
        @test adfmax_pdf(x, cas, c) ≈ fr atol = 1e-10
    end
    F = adfmax_cdf([-2.5, -Inf, Inf, NaN], "DU", 0)
    @test F[1] ≈ 0.042709500744894657 atol = 1e-12
    @test F[2] == 0 && F[3] == 1 && isnan(F[4])
    q = adfmax_icdf(0.05, "DU")
    @test q ≈ -2.4354164994325092 atol = 1e-9
    @test adfmax_cdf(q, "DU") ≈ 0.05 atol = 1e-9
end

@testset "coefficient statistic" begin
    @test coef_cdf(-14.0, "DU", -5) ≈ 0.1938495145682877 atol = 1e-10
    @test coef_cdf(-14.0, "SU", -10) ≈ 0.45401150754561903 atol = 1e-10
    @test coef_cdf(-16.59, "DG", -13.5) ≈ 0.51132404681639121 atol = 1e-10
    @test coef_icdf(0.05, "DU") ≈ -14.093590831267115 atol = 1e-7
    @test round(coef_icdf(0.05, "DK"), digits = 3) == -8.039                   # the paper: -8.039, -14.094, -21.711
    @test round(coef_icdf(0.05, "DT"), digits = 3) == -21.711
    @test coef_cdf(-5, "DT", -2; c0 = 1) ≈ ltratio_cdf(0, "DT", -2; c0 = -4) rtol = 1e-13   # the definition
    @test coef_cdf([-Inf, NaN, Inf], "DK")[[1, 3]] == [0.0, 1.0] && isnan(coef_cdf(NaN, "DK"))
end

@testset "regions, lincdf, power envelope, bisection" begin
    for p in readref("app_ref.txt")
        p[1] == "region" || continue
        ty, p1, p2, v0, c, Pr = num(p[2:7])
        bnd = ty == 1 ? (type = "line", a = p1, b = p2, v0 = v0) : ty == 2 ? (type = "parabola", x = p1, c0 = p2, v0 = v0) :
                        (type = "wap", k = p1, omega = p2, v0 = v0)
        @test ltratio_region(bnd, "DK", c) ≈ Pr atol = 1e-10
    end
    @test ltratio_region((type = "line", a = -3.5, b = -0.2639), "DK", 0) ≈ 0.049996781931320872 atol = 1e-12
    @test ltratio_region((type = "wap", k = 3.6658, omega = 10), "DK", -10) ≈ 0.75806440265373276 atol = 1e-12
    @test ltratio_region((type = "parabola", x = -1.94087), "DK", -7) ≈ ltratio_cdf(-1.94087, "DK", -7; method = :single) atol = 1e-11
    # a function g (check_applications.jl holds the general boundaries to 1e-10)
    @test ltratio_region((type = "fun", g = v -> -1.94087 * sqrt(v)), "DK", -7) ≈ ltratio_cdf(-1.94087, "DK", -7; method = :single) atol = 1e-10
    @test ltratio_lincdf(-0.2639, -3.5, "DK", 0) ≈ 0.049996781931325174 atol = 1e-12
    E = power_envelope([-7.0, 0.0])
    @test E[1] ≈ 0.49940198260314117 atol = 1e-10
    @test E[2] == 0.05
    @test power_envelope(-13.5, "DG") ≈ 0.51441347319065545 atol = 1e-10
    @test ltratio_bisect(x -> x^3 - 2, 0, 2, 1e-12) ≈ cbrt(2) atol = 1e-12
    @test_throws ErrorException ltratio_bisect(x -> x^2 + 1, -1, 1)
end

# The linear form near a = 0 (November 2026): the inversion of Gil-Pelaez truncated at u = 4000 was off by up to 7e-2 near
# the edge b = -1/2 of the law of U alone (a = 0, a gamma(1/2) variable less 1/2) and 1e-3 in the bulk for |a| up to about
# 0.3, and for c > 0 where U - aV is wide; now a conditional integral serves there (the region in case DK, the conditional
# integral given U in case DG).  References: the closed form at a = 0, P(U <= b) = erf(sqrt(kappa (b + 1/2))) with kappa
# = 2c/(e^{2c} - 1) (case DK); the conditional integral given U (lincond, case DK) against the region; in case DG the
# inversion of Gil-Pelaez on a grid 40 times finer up to u = 4e5 and the contour integral of the newer engine (DFDet).
@testset "the linear form near a = 0" begin
    kap(c) = c == 0 ? 1.0 : 2c / expm1(2c)
    for c in (-50.0, -5.0, 0.0, 2.0), b in (-0.6, -0.5 - 1e-6, -0.5 + 1e-6, -0.49, -0.3, 0.5, 2.0)
        F = ltratio_lincdf(b, 0.0, "DK", c)
        @test F ≈ (b > -0.5 ? erf(sqrt(kap(c) * (b + 0.5))) : 0.0) atol = 1e-13
    end
    for c in (-5.0, 0.0), a in (1e-6, -1e-6, 1e-3, -1e-3, 0.1, -0.1, 0.3), b in (-0.6, -0.51, -0.5 + 1e-5, -0.49, 0.5)
        @test ltratio_lincdf(b, a, "DK", c) ≈ D.lincond(b, a, "DK", c) atol = 1e-12
    end
    @test ltratio_lincdf(-0.51, 0.3, "DG", -5.0) ≈ 0.291315235195 atol = 1e-11
    @test ltratio_lincdf(-0.51, 0.3, "DG", 0.0) ≈ 0.287795093100 atol = 1e-11
    @test ltratio_lincdf(-0.51, 0.3, "DG", -13.5) ≈ 0.102096301375 atol = 1e-11
    kg = D.lincond_setup("DG", 0.0, -13.5).kap
    @test ltratio_lincdf(-0.49, 0.0, "DG", 0.0) ≈ erf(sqrt(kg * 0.01)) atol = 1e-14
    F = ltratio_lincdf(collect(range(-1.0, 1.0; length = 41)), 1e-4, "DG", -1.0)
    @test all(0 .<= F .<= 1) && issorted(F)
    @test ltratio_lincdf(-0.6, -1e-3, "DK", 0.0) == 0.0                   # (below the support: a <= 0, b < -1/2)
    # c > 0 against the inversion of Gil-Pelaez on a grid of 1.2e6 nodes up to 1e5; detrended at cbar = -0.1, where the
    # conditional law of V given R turns within a fraction of the scale of R (the adaptive panels of the outer integral)
    @test ltratio_lincdf(-0.45, -0.3, "DG", 2.0) ≈ 0.270133915145322 atol = 1e-12
    @test ltratio_lincdf(-0.32, -1.0, "DG", 2.0; cbar = -0.1) ≈ 0.455160024556188 atol = 1e-12
    @test isequal(ltratio_lincdf([-Inf, NaN, Inf], 0.0, "DK", 0.0), [0.0, NaN, 1.0])
    @test isequal(ltratio_lincdf([-Inf, NaN, Inf], -3.5, "DK", 0.0), [0.0, NaN, 1.0])
    # the power envelope near c = 0: against the critical value and the power by the region (case DK), and alpha + O(c^2) in
    # case DG
    @test power_envelope(-0.01, "DK") ≈ 0.050249919491 atol = 1e-10
    @test power_envelope(-1e-4, "DK") ≈ 0.050002496746 atol = 1e-10
    @test 0.05 < power_envelope(-1e-3, "DG") < 0.05 + 1e-8
    # the law of V far from the unit root (the transform decays only from u ~ 15 |c|^{3/2}): against a grid up to 4e6
    @test ltratio_lincdf(D.EVdk(-200.0), NaN, "DK", -200.0) ≈ 0.519901089453532 atol = 1e-11
    @test occursin("at least -550", errmsg(() -> ltratio_lincdf(0.0, 1.0, "DK", -600)))
    @test occursin("case DG is available for c up to about 3.5", errmsg(() -> ltratio_lincdf(0.0, 0.01, "DG", 6.0)))
end

# the values do not depend on the number of tasks (the order of summation can change the last digit); with one thread the
# tasks run in turn, which still exercises the threaded loops
@testset "number of tasks" begin
    nw = D.NWORKERS[]
    x = [-2.0, 0.0]
    try
        D.NWORKERS[] = 1
        F1 = ltratio_cdf(x, "DT", -3); f1 = ltratio_pdf(x, "DK", -3); Fd1 = ltratio_cdf(x, "SU", -3; method = :double)
        D.NWORKERS[] = max(nw, 3)
        @test ltratio_cdf(x, "DT", -3) ≈ F1 rtol = 1e-14
        @test ltratio_pdf(x, "DK", -3) ≈ f1 rtol = 1e-14
        @test ltratio_cdf(x, "SU", -3; method = :double) ≈ Fd1 rtol = 1e-14
    finally
        D.NWORKERS[] = nw
    end
end

# the argument errors (quickcheck.jl, sections 12, 16 and 18, and the R package's interface tests)
@testset "argument errors" begin
    @test errmsg(() -> ltratio_cdf(0, "DU", -600)) == D.CMIN_MSG
    @test errmsg(() -> ltratio_pdf(-20.0, "SK", -700)) == D.CMIN_MSG
    @test errmsg(() -> ltratio_mom(2, "DT", -551)) == D.CMIN_MSG
    @test errmsg(() -> ltratio_icdf(0.05, "DG", -551)) == D.CMIN_MSG
    @test errmsg(() -> ltratio_cdf(-17.0, "DU", -550)) == ""
    @test errmsg(() -> ltratio_cdf(-2.0, "DK", 0; g0 = 2.5)) == D.FIXED_MSG
    @test errmsg(() -> ltratio_pdf(-2.0, "DU", -10; g0 = 2)) == D.FIXED_MSG
    @test errmsg(() -> ltratio_cdf(-9.0, "DT", -110; g0 = 0.5)) == D.FIXED_MSG
    @test errmsg(() -> ltratio_mom(2, "DU", 3; g0 = 1)) == D.FIXED_MSG
    @test errmsg(() -> ltratio_cdf(0, "SU", 1)) == D.STAT_MSG
    @test occursin("applies to cases DK, DU, DT", errmsg(() -> ltratio_cdf(0, "SU", -5; g0 = 1)))
    @test occursin("not available for case DG", errmsg(() -> ltratio_mom(2, "DG")))
    @test occursin("line integral representation only", errmsg(() -> ltratio_cdf(0, "DG"; method = :double)))
    @test occursin("(0, 1)", errmsg(() -> ltratio_icdf(1, "DU")))
    @test occursin("finite real scalar", errmsg(() -> ltratio_cdf(0, "DU", NaN)))
    @test occursin("finite real scalar", errmsg(() -> ltratio_cdf(0, "DU", 0; c0 = Inf)))
    @test occursin("the case must be", errmsg(() -> ltratio_cdf(0, "XX")))
    @test occursin("method must be", errmsg(() -> ltratio_cdf(0, "DU"; method = :fast)))
    @test occursin("positive integer", errmsg(() -> ltratio_mom(0, "DU")))
    @test occursin("the case must be one of DU, DT, SU, ST", errmsg(() -> adfmax_cdf(0, "DK")))
    @test errmsg(() -> adfmax_cdf(0, "SU", 1)) == D.STAT_MSG
    @test occursin("(0, 1)", errmsg(() -> adfmax_icdf(1.5, "DU")))
    @test occursin("(0, 1)", errmsg(() -> coef_icdf(0, "DU")))
    @test occursin("needs the field b", errmsg(() -> ltratio_region((type = "line", a = 1))))
    @test occursin("unknown boundary type", errmsg(() -> ltratio_region((type = "circle",))))
    @test occursin("k > 0 and omega > 0", errmsg(() -> ltratio_region((type = "wap", k = -1, omega = 1))))
    @test occursin("the case must be DK", errmsg(() -> ltratio_region((type = "line", a = -3.5, b = -0.2639), "DU")))
    @test occursin("the case must be DK or DG", errmsg(() -> ltratio_lincdf(0.0, 1.0, "DU")))
    @test occursin("the case must be DK (or DU) or DG", errmsg(() -> power_envelope(-5, "DT")))
end

@testset "stationary start near the unit root" begin
    # for c > -1e-4 the engine refuses (CSTAT, STAT_MSG): the variance -1/(2c) of the start is too large for the quadrature
    for f in (() -> ltratio_cdf(-1.94087, "SK", -9e-5), () -> ltratio_pdf(-2.0, "SU", -9e-5), () -> ltratio_mom(2, "ST", -9e-5),
              () -> ltratio_icdf(0.05, "SK", -9e-5), () -> coef_cdf(-8.039, "SK", -9e-5), () -> adfmax_cdf(-2.4, "SU", -9e-5))
        @test errmsg(f) == D.STAT_MSG
    end
    # at c = -1e-4 (test/engine/quickcheck.jl, section 18b), and the two errors fixed in November 2026: the batched
    # distribution function of case SK at tau < 1 and the lower end of the conditional integral with a stationary start
    @test ltratio_cdf(-1.94087, "SK", -1e-4) ≈ 0.028490044498374834 atol = 1e-11
    @test coef_cdf(-8.039, "SK", -1e-4) ≈ 0.00032997244558452543 atol = 1e-12
    @test adfmax_cdf(-2.43542, "SU", -1e-4) ≈ 0.050000718603976146 atol = 1e-11
    @test ltratio_cdf(-1.94087, "SK", -0.1) ≈ 0.059510986573375835 atol = 1e-11
    @test coef_cdf(-8.039, "SK", -0.001) ≈ 0.0010438037922007037 atol = 1e-12
end

# Memory (November 2026): two calls that took more than 5 GB.  (1) Case DT at c = 7 in the bulk (x in the hundreds): every
# new Gauss-Legendre rule of the inner integrals' saddle segment, whose node count grows like x^2, was computed with its
# n x n matrix of eigenvectors; now by Newton's method above GLEIG nodes, at most RULEMAX nodes, and beyond that the
# conditional integral.  (2) Case SK at c = c0 = -1e-4 with x > 0: the root of X(v) at x/|c0| made the general kernel's
# paths billions of nodes long, each stored, with two boxed variables allocating about 1 kB per node; now the paths are
# lists of pieces summed as they go, at most NODEMAX nodes, and beyond that the conditional integral.  The values: the C
# engine (dflimit 0.4.0) and the contour integral of the newer engine (DFDet) where the C engine fails.
footprint() = Sys.isapple() ? (buf = zeros(UInt8, 96); ccall(:proc_pid_rusage, Cint, (Cint, Cint, Ptr{UInt8}), getpid(), 0, buf) == 0 ?
                               Float64(reinterpret(UInt64, buf[73:80])[1]) : Float64(Sys.maxrss())) : Float64(Sys.maxrss())
@testset "memory: rules, paths and caches" begin
    @test (@allocated D.gl(4096)) < 2^20                           # (Golub-Welsch: 129 MB)
    @test maximum(abs.(D.gl(2048)[1] .- D.gl_newton(2048)[1])) < 1e-15
    @test D.pathtoolarge(try; D.rule(D.RULEMAX + 1); catch e; e; end) isa D.PathTooLarge
    f0 = footprint()
    for (x, Fr) in ((200.0, 0.5122687264383423), (300.0, 0.7020412425838507), (403.7, 0.838656678280211))
        r = Ref(NaN); a = @allocated (r[] = ltratio_cdf(x, "DT", 7.0))
        @test r[] ≈ Fr atol = 1e-12
        @test a < 0.3 * 2^30                                        # (0.42 GB at x = 200 before; 3 GB at 403.7)
    end
    @test ltratio_cdf(751.0, "DT", 7.0) ≈ 0.990823749760809 atol = 1e-12   # (the C engine segfaults here; more than 5 GB before)
    @test ltratio_pdf(751.0, "DT", 7.0) ≈ 9.2924954e-5 rtol = 1e-6
    @test footprint() - f0 < 0.3 * 2^30
    # the neighbour of call 2 by the line integral itself (12 GB and 47 GB allocated before)
    for (x, Fr) in ((0.5, 0.780675601976397), (1.0, 0.893885483036904))
        r = Ref(NaN); a = @allocated (r[] = ltratio_cdf(x, "SK", -0.1; c0 = -0.1))
        @test r[] ≈ Fr atol = 1e-12
        @test a < 0.2 * 2^30
    end
    # call 2: x <= 0 by the line integral, x > 0 by the conditional integral, which flags some of them (a warning)
    f0 = footprint()
    F = @test_logs (:warn, r"the conditional integral that takes these points flags") match_mode = :any ltratio_cdf([-1.0, 0.05, 0.5, 1.0, 1.5], "SK", -1e-4; c0 = -1e-4)
    @test F ≈ [0.1666135587511895, 0.531233538152, 0.700742367308, 0.847290600876, 0.936188673511] atol = 1e-6
    @test footprint() - f0 < 0.5 * 2^30
    # a path beyond the bounds: the engine refuses it before evaluating it, the user-level functions name the arguments
    e = errmsg(() -> ltratio_cdf(1.0, "SK", -1e-4; c0 = -1e-4, method = :single))
    @test occursin("the line integral needs more nodes than the engine allows", e) && occursin("c0 = -0.0001, x = 1.0", e)
    @test occursin("at large x above the unit root", errmsg(() -> ltratio_cdf(751.0, "DT", 7.0; method = :single)))
    @test D.pathtoolarge(try; D.cdf(1.0, 1; c = -1e-4, c0 = -1e-4, start = :stationary); catch e; e; end) isa D.PathTooLarge
    nm = D.NODEMAX[]; D.NODEMAX[] = 1000
    try
        @test occursin("case DG has the line integral representation only", errmsg(() -> ltratio_cdf(0.5, "DG", -1.0; c0 = -0.01)))
        @test ltratio_cdf(0.5, "DK", 0.0; c0 = -0.01) ≈ 0.821459268187 atol = 1e-11    # (routed: the conditional integral)
    finally
        D.NODEMAX[] = nm
    end
    # the caches kept per point or per c are bounded
    ltratio_cdf(collect(range(-4, 1; length = 100)), "DU", -5)
    @test length(D.LAYOUTSK) <= D.CACHEMAX && length(D.LAYOUTS) <= D.CACHEMAX
end

# A fixed start with c0 < 0 near zero at x > 0 (November 2026): the noncentral exponent rises into the segment at v = 0,
# and the descent piece of segint, clamped to 80 degrees from it, climbed at the real point of the line to e^{213} and
# cancelled against the connection (F = 6e74 at c = 0, c0 = -0.01, g0 = 1, x = 1.5; the C engine the same); now cut back
# where it climbs above the larger end of the segment (noclimb).  The values: the contour integral of the newer engine.
@testset "fixed start: a descent piece that climbs" begin
    @test ltratio_cdf(1.5, "DK", 0; c0 = -0.01, g0 = 1) ≈ 0.960991862318 atol = 1e-10
    @test ltratio_pdf(1.5, "DK", 0; c0 = -0.01, g0 = 1) ≈ 0.079242719297 atol = 1e-10
    @test ltratio_cdf(0.5, "DK", 0; c0 = -0.01, g0 = 1) ≈ 0.798334456036 atol = 1e-10     # (was off by 7e-8)
    @test ltratio_cdf(1.5, "DK", 2; c0 = -0.01, g0 = 1) ≈ 0.086764388544 atol = 1e-10     # (was off by 3e-3)
    @test ltratio_cdf(0.5, "DK", 0; c0 = -0.01, g0 = 2) ≈ 0.755768817096 atol = 1e-10
    @test ltratio_cdf(1.5, "DU", 2; c0 = -0.01, g0 = 1, method = :single) ≈ 0.207535597435 atol = 1e-10   # (was -5e29)
end

# the engine's own checks (test/engine/, from the directory julia/ of the paper's repository, where the engine was
# developed): quickcheck.jl (K_0 against AMOS, node doubling, contour independence, closed forms, the applications, the
# contour integral, the default and verified) and test_dfexact.jl (all the stored values far from the unit root and of
# the contour integral).  Each is a script that prints its comparisons and raises an error when a check fails; it runs
# here in a module of its own.
@testset "engine check $f" for f in ("quickcheck.jl", "test_dfexact.jl")
    @test (Base.include(Module(), joinpath(@__DIR__, "engine", f)); true)
end

end # @testset "DFLimit"

@printf("DFLimit tests: %.0f s\n", time() - T0)
