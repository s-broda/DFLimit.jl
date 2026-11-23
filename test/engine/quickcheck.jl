# Quick check of DFExact.jl (about a minute after compilation): the stored reference values, node doubling and
# contour independence at a few points of every model for c = 0, -5 and 3, the K_0 evaluation against AMOS,
# the normalization of the density, a few points far from the unit root (c = -100, -480), and the application functions
# (adfmax_cdf, coef_cdf, ltratio_region, ltratio_lincdf, power_envelope, ...) against stored values.  Exits with an error if anything is off; the full certification is
# dev/certify_dfexact.jl.  Part of Pkg.test() (test/runtests.jl); alone: julia -t auto --project=dev test/engine/quickcheck.jl
using Printf, SpecialFunctions, Random
using DFLimit; const DFExact = DFLimit.DFExact; const D = DFExact
@printf("Julia %s, %s, %d threads, %d workers, vector math: %s\n", VERSION, Sys.MACHINE, Threads.nthreads(), D.NWORKERS[], D.vecmath())
fails = String[]
check(ok, msg) = (ok || push!(fails, msg); @printf("  %s %s\n", ok ? "ok  " : "FAIL", msg))

# 1. K_0 against AMOS
Random.seed!(1); worst = 0.0
for _ in 1:20000
    z = complex(rand() * 40, (rand() - 0.5) * 80)
    ref = besselkx(0, z); global worst = max(worst, abs(D.k0x(z) - ref) / abs(ref))
end
check(worst < 5e-14, @sprintf("K_0 vs AMOS: worst relative error %.1e", worst))

# 2. stored reference values (the twelve points of test_dfexact.jl)
import JSON
ref = JSON.parsefile(joinpath(@__DIR__, "python_exact_reference.json"))
wF = wf = 0.0
for r in ref
    F = D.cdf(Float64(r["q"]), r["model"]; c = Float64(r["c"])); f = D.density(Float64(r["q"]), r["model"]; c = Float64(r["c"]))
    global wF = max(wF, abs(F - r["cdf"])); global wf = max(wf, abs(f - r["density"]) / abs(r["density"]))
end
check(wF < 1e-12 && wf < 1e-10, @sprintf("reference values: |dF| %.1e, |df/f| %.1e", wF, wf))

# 3. node doubling and contour independence at a few points
wd = 0.0; wt = 0.0
for model in 1:3, (c, q) in ((0.0, -2.0), (0.0, 0.5), (-5.0, -3.0), (3.0, 2.0), (3.0, 8.0))
    F = D.cdf(q, model; c = c); f = D.density(q, model; c = c)
    D.NODEFAC[] = 2; F2 = D.cdf(q, model; c = c); f2 = D.density(q, model; c = c); D.NODEFAC[] = 1
    global wd = max(wd, abs(F - F2), abs(f - f2) / abs(f))
    if c > 0
        F3 = D.cdf(q, model; c = c, tau = D.tstar(model, c) / 4); global wt = max(wt, abs(F - F3))
    end
end
check(wd < 1e-10, @sprintf("node doubling: worst change %.1e", wd))
check(wt < 1e-10, @sprintf("contour independence (c = 3): worst change %.1e", wt))

# 4. normalization of the density (M2, c = 0) and the 5% quantile of M2
qs = range(-8, 6, length = 561); h = step(qs)
mass = sum(D.density(q, 2) for q in qs) * h
check(abs(mass - 1) < 1e-9, @sprintf("density of M2 integrates to %.12f", mass))
q05 = D.quantile(0.05, 2)
check(abs(q05 + 2.861568864) < 1e-8, @sprintf("M2 5%% quantile %.9f", q05))

# 5. the stationary start: the limiting power of the 5% test of M2 at c = -5 (0.125 in Bao, Broda and Kan's table)
Fs = D.cdf(-2.86157, 2; c = -5.0, start = :stationary)
D.NODEFAC[] = 2; Fs2 = D.cdf(-2.86157, 2; c = -5.0, start = :stationary); D.NODEFAC[] = 1
check(abs(Fs - 0.125) < 5e-4 && abs(Fs - Fs2) < 1e-10, @sprintf("stationary start, M2 c=-5 power %.6f (doubling %.1e)", Fs, abs(Fs - Fs2)))

# 6. small |q|: the fixed layouts hand over to the adaptive code below |q| = 0.034; both paths and the uniform rule agree
wq = 0.0
for (m, q) in ((1, -0.0061), (2, -0.0061), (1, -0.02), (3, -0.02))
    fb = D.density(q, m); D.BATCHED[] = false; fs = D.density(q, m); D.BATCHED[] = true
    fu = D.density_uniform(q, m; hw = 0.1)
    global wq = max(wq, abs(fb - fs) / fs, abs(fb - fu) / fs)
end
check(wq < 1e-10, @sprintf("small |q| (0.006, 0.02): batched vs adaptive vs uniform %.1e", wq))

# 7. large c: the roots and P(t) from the Girsanov forms (M2, c = 10, x = 20; the determinant itself loses 8 digits there)
F10 = D.cdf(20.0, 2; c = 10.0); D.NODEFAC[] = 2; F10b = D.cdf(20.0, 2; c = 10.0); D.NODEFAC[] = 1
check(abs(F10 - 0.001645968250) < 1e-11 && abs(F10 - F10b) < 1e-11, @sprintf("M2 c=10: F(20) = %.12f (doubling %.1e)", F10, abs(F10 - F10b)))
# the Girsanov determinant against D(0, t) where both are exact: e^c cosh(th) det(I - Q Sig) = D(0, t)
wg = 0.0
for m in 2:3, c in (2.5, -1.0), t in (complex(0.05), complex(0.3, 0.7), complex(2.0, -4.0), complex(c^2 / 2 + 0.2, 0.3))
    G = D.girsanov(m, c, t)
    global wg = max(wg, abs(exp(c) * cosh(G.th) * G.D / D.Dfun(m, c, complex(0.0), t) - 1))
end
check(wg < 1e-12, @sprintf("Girsanov determinant vs D(0, t): worst relative difference %.1e", wg))
# the domain end t_* of case DK at c < 0: the first zero of D(0, t) lies just below the pole of eta at (c^2 + pi^2)/2, which
# a bracketing on the sign of eta alone could skip (61.6 for 49.2 at c = -9.5, 313 for 204.5 at c = -20)
wts = maximum(abs(D.tstar(1, c) - D.dksmax(c)) / D.dksmax(c) for c in (-9.5, -20.0, -30.0))
check(wts < 1e-12 && abs(D.tstar(1, -9.5) - 49.18660310687472) < 1e-9, @sprintf("t_* of case DK at c = -9.5, -20, -30 vs dksmax: %.1e", wts))
# the GLS-detrended trend case (model 4, case DG): at c = 0, P(tau <= 0) = P{V(1)^2 <= 1} with Var V(1) = 0.2 (1 - lam)^2 in
# closed form; node doubling and the contour at the 5% point
lamg = D.lam_gls(D.CBAR_GLS); F0g = D.ltratio_cdf(0.0, "DG", 0.0); F0c = erf(1 / sqrt(0.4 * (1 - lamg)^2))
Fg = D.cdf(-2.8456, 4; c = -5.0); D.NODEFAC[] = 2; Fg2 = D.cdf(-2.8456, 4; c = -5.0); D.NODEFAC[] = 1; Fg3 = D.cdf(-2.8456, 4; c = -5.0, tau = 0.5)
check(abs(F0g - F0c) < 1e-12 && abs(Fg - Fg2) < 1e-11 && abs(Fg - Fg3) < 1e-11, @sprintf("case DG (GLS, trend): P(tau <= 0) at c = 0 vs the closed form %.1e; doubling %.1e, contour %.1e", abs(F0g - F0c), abs(Fg - Fg2), abs(Fg - Fg3)))

# 8. the moments: E tau_mu^2 at the unit root is 3.05599535 (Nabeya, 1999); the first implementation agrees
m2 = D.moments(2; pmax = 2)[2]
check(abs(m2 - 3.05599535) < 1e-8 && abs(m2 - D.moments_dfx(2; pmax = 2)[2]) < 1e-12, @sprintf("E tau_mu^2 at c = 0: %.8f", m2))

# 9. the local null value c0: the t-ratio centered at the true value (DK, c = c0 = 2, x = 0: 0.650436 in Larsson's table),
#    P(U - c0 V <= 0) at c = 10, c0 = 5 (DU: 0.000302655553129 by a one-dimensional inversion), node doubling with two
#    breakpoints (DT, c = -2.5, c0 = -5), and the general kernel at c0 = 0 against the specialized code
F22 = D.cdf(0.0, 1; c = 2.0, c0 = 2.0)
check(abs(F22 - 0.650436) < 1e-6, @sprintf("DK c = c0 = 2: F(0) = %.6f (0.650436)", F22))
F105 = D.cdf(0.0, 2; c = 10.0, c0 = 5.0)
check(abs(F105 - 0.000302655553129) < 1e-11, @sprintf("DU c = 10, c0 = 5: F(0) = %.15f", F105))
Fg = D.cdf(-1.1, 3; c = -2.5, c0 = -5.0); fg = D.density(-1.1, 3; c = -2.5, c0 = -5.0)
D.NODEFAC[] = 2; Fg2 = D.cdf(-1.1, 3; c = -2.5, c0 = -5.0); fg2 = D.density(-1.1, 3; c = -2.5, c0 = -5.0); D.NODEFAC[] = 1
check(abs(Fg - Fg2) < 1e-10 && abs(fg - fg2) / fg < 1e-10, @sprintf("DT c = -2.5, c0 = -5: F(-1.1) = %.12f, doubling %.1e / %.1e", Fg, abs(Fg - Fg2), abs(fg - fg2) / fg))
Fs = D.cdf(-2.86, 2); Fsg = D.cdf(-2.86, 2; gen = true); fs = D.density(-2.86, 2); fsg = D.density(-2.86, 2; gen = true)
check(abs(Fs - Fsg) < 1e-12 && abs(fs - fsg) / fs < 1e-11, @sprintf("general kernel at c0 = 0 vs the specialized code: %.1e / %.1e", abs(Fs - Fsg), abs(fs - fsg) / fs))
check(isinf(D.density(-2.0, 1; c0 = 2.0)), "DK c0 = 2: f(-sqrt(2 c0)) = Inf")

# 10. moments with a local null value and a start: the mean of DU at c = -5 against the Gauss-Legendre integral of the density
#     (192 nodes on [-12, 6]), and the stationary start as the Gauss-Hermite average over the start of the fixed-start moments
using LinearAlgebra
let n = 192, a = -12.0, b = 6.0
    E = eigen(SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:n-1]))
    xg = a .+ (b - a) .* (E.values .+ 1) ./ 2; wg = (b - a) .* E.vectors[1, :] .^ 2
    f = [D.density(x, 2; c = -5.0) for x in xg]
    m1 = D.moments(2; c = -5.0, pmax = 1)[1]
    check(abs(sum(wg .* xg .* f) - m1) < 1e-9 && abs(sum(wg .* f) - 1) < 1e-9, @sprintf("DU c=-5: mean %.10f vs the density integral (%.1e), mass 1 %+.1e", m1, abs(sum(wg .* xg .* f) - m1), sum(wg .* f) - 1))
    E = eigen(SymTridiagonal(zeros(16), sqrt.(1:15))); gz = E.values; gw = E.vectors[1, :] .^ 2
    mst = D.moments(2; c = -5.0, start = :stationary, pmax = 2)
    mav = sum(gw[k] .* D.moments(2; c = -5.0, g0 = sqrt(0.1) * gz[k], pmax = 2) for k in 1:16)
    check(maximum(abs.(mst .- mav)) < 1e-7, @sprintf("SU c=-5: moments %s vs the Gauss-Hermite average of the fixed start %.1e", mst, maximum(abs.(mst .- mav))))
end

# 11. the double integral (the conditional integral representation; the only route for a fixed start in cases DU, DT): against the
#     single integral with a zero start (DU, c = 2, c0 = 1) and a stationary start (SU, c = -5), the closed form of
#     P[J(1)^2 <= 1 + g0^2] in case DK with a fixed start, and the MATLAB programs' value for DU, c = -5, g0 = 1.5, x = -2
Fd = ltratio_cdf([-0.5, 1.3], "DU", 2; c0 = 1, method = :double); Fs = ltratio_cdf([-0.5, 1.3], "DU", 2; c0 = 1)
check(maximum(abs.(Fd .- Fs)) < 1e-11, @sprintf("double vs single integral, DU c = 2 c0 = 1: %.1e", maximum(abs.(Fd .- Fs))))
Fd = ltratio_cdf(-2.0, "SU", -5; method = :double); Fs = ltratio_cdf(-2.0, "SU", -5)
check(abs(Fd - Fs) < 1e-11, @sprintf("double vs single integral, SU c = -5: %.1e", abs(Fd - Fs)))
let c = -5.0, g0 = 1.5, v = (exp(-10) - 1) / (-10), m = 1.5 * exp(-5), r = sqrt(1 + 1.5^2)
    Fex = (erf((r - m) / sqrt(2v)) + erf((r + m) / sqrt(2v))) / 2
    Fg = ltratio_cdf(0.0, "DK", c; g0)
    check(abs(Fg - Fex) < 1e-11, @sprintf("DK c = -5 g0 = 1.5: F(0) = %.15f vs the closed form (%.1e)", Fg, abs(Fg - Fex)))
end
Fg = ltratio_cdf(-2.0, "DU", -5; g0 = 1.5)
check(abs(Fg - 0.864977240077167) < 1e-11, @sprintf("DU c = -5 g0 = 1.5: F(-2) = %.15f vs the MATLAB programs (%.1e)", Fg, abs(Fg - 0.864977240077167)))

# 12. the user-level interface: the 5% critical value of tau_mu, non-finite x, invalid arguments
q05 = ltratio_icdf(0.05, "DU")
check(abs(q05 + 2.86156886) < 1e-7, @sprintf("ltratio_icdf(0.05, \"DU\") = %.8f", q05))
Fx = ltratio_cdf([-Inf, 0.0, Inf, NaN], "DK", -5)
check(Fx[1] == 0 && Fx[3] == 1 && isnan(Fx[4]) && abs(Fx[2] - erf(1 / sqrt(2 * (exp(-10) - 1) / (-10)))) < 1e-12, "ltratio_cdf: non-finite x, and F(0) of DK at c = -5 in closed form")
check(try; ltratio_cdf(-2.0, "SU", 1); false; catch e; e isa ArgumentError; end, "ltratio_cdf rejects a stationary start with c > 0")

# 13. the tails with c0 != 0 (Raymond Kan, November 2026): the far roots of X(v) are kept, so the single integral does not
# fail there, and with method = :auto the values below 1e-3 come from the double integral (the single integral's error is
# absolute); a negative computed density is returned as zero
fe = try; ltratio_pdf([9.0, 3.0], "SU", -1.2; c0 = -1.2, method = :single); ltratio_pdf(-9.0, "SU", -1.2; c0 = 1.2, method = :single); ltratio_pdf(3.0, "SU", -1.2; c0 = -0.05, method = :single); true; catch; false; end
check(fe, "no error at the far roots of X(v): SU c = -1.2, c0 = -1.2, 1.2, -0.05")
fs = ltratio_pdf(3.0, "SU", -1.2; c0 = -0.05, method = :single); fd = ltratio_pdf(3.0, "SU", -1.2; c0 = -0.05, method = :double)
check(abs(fs / fd - 1) < 1e-10, @sprintf("SU c = -1.2 c0 = -0.05, x = 3: single (far root kept) vs double, f = %.6e (%.1e relative)", fd, abs(fs / fd - 1)))
fa = ltratio_pdf(-9.0, "SU", -1.2; c0 = 1.2); fd = ltratio_pdf(-9.0, "SU", -1.2; c0 = 1.2, method = :double)
check(fa == fd && fa > 0, @sprintf("SU c = -1.2 c0 = 1.2, x = -9: the automatic method takes the double integral in the tail, f = %.3e", fa))
check(all(ltratio_pdf([8.5, 9.0, 12.0], "SU", -1.2; c0 = -1.2, method = :single) .>= 0), "a negative computed density is returned as zero")

# 14. the conditional integral in case T on the explosive side (November 2026): for c between 2.04 and 2.38 the step grid
# of the conditional moments has points at which the conditional variance is not positive; they are skipped (as in the
# C engine), where the square root used to throw
fs = ltratio_cdf([-3.0, -1.0], "DT", 2.3; c0 = 2.3, method = :single)
fd = try; ltratio_cdf([-3.0, -1.0], "DT", 2.3; c0 = 2.3, method = :double); catch; [NaN, NaN]; end
check(maximum(abs.(fd .- fs)) < 1e-10, @sprintf("DT c = c0 = 2.3: double vs single, F = %.10f, %.10f (%.1e)", fd[1], fd[2], maximum(abs.(fd .- fs))))

# 15. far from the unit root (November 2026): a few of the regression points of largec_ref.txt (all of them in
# test_dfexact.jl), the moments and quantiles at c = -100 and -480, and t_* there (the first zero of D(0, t) lies at
# phi = sqrt(2t - c^2) of about 6.2 in case DU; the 5% steps in t used to skip pairs of zeros, giving 18.7 at c = -100)
let LR = [split(l) for l in readlines(joinpath(@__DIR__, "largec_ref.txt"))], w = 0.0, wa = 0.0, wm = 0.0, wq = 0.0
    sub = (("DU", -100.0, 0.0, -2.0), ("DU", -480.0, 0.0, 2.0), ("DT", -480.0, 0.0, 0.0), ("SU", -480.0, 0.0, -2.0),
           ("DK", -480.0, 0.0, 5.0), ("ST", -100.0, 0.0, 0.0), ("DK", -100.0, -100.0, -2.0))
    for p in LR
        p[1] == "val" || continue
        cas = String(p[2]); c, c0, z, x, F, f = parse.(Float64, p[3:8])
        (cas, c, c0, z) in sub || continue
        w = max(w, abs(ltratio_cdf(x, cas, c; c0, method = :single) - F), abs(ltratio_pdf(x, cas, c; c0, method = :single) - f))
        wa = max(wa, abs(ltratio_cdf(x, cas, c; c0) - F), abs(ltratio_pdf(x, cas, c; c0) - f))
    end
    for p in LR
        if p[1] == "mom" && p[2] in ("DU", "SU") && p[4] == "0"
            c = parse(Float64, p[3]); mr = parse.(Float64, p[5:8])
            wm = max(wm, maximum(abs.(ltratio_mom(4, String(p[2]), c; method = :engine) .- mr) ./ abs.(mr)))
        elseif p[1] == "icdf" && p[2] == "DU" && p[5] == "0.05"
            c = parse(Float64, p[3]); q = ltratio_icdf(0.05, "DU", c)
            wq = max(wq, abs(q - parse(Float64, p[6])), abs(ltratio_cdf(q, "DU", c) - 0.05))
        end
    end
    check(w < 1e-11 && wa < 1e-10, @sprintf("c = -100, -480: seven points of largec_ref.txt, single %.1e, auto %.1e", w, wa))
    check(wm < 1e-10 && wq < 1e-9, @sprintf("c = -100, -480: moments of DU, SU (%.1e relative), 5%% quantile of DU (%.1e)", wm, wq))
    ph = [sqrt(2D.tstar(2, c) - c^2) for c in (-100.0, -480.0)]
    check(all(6.1 .< ph .< 6.3), @sprintf("t_* of case DU at c = -100, -480: phi = %.4f, %.4f (the first zero)", ph...))
end

# 16. a fixed start far from the unit root (November 2026): in cases DU, DT the constant e^{-lam} of the noncentral kernel
# (lam ~ (c g0)^2/2 in case DU) goes into the exponent of the scaled factor, where the line integral returned NaN for
# lam > 709 (DU c = -18, g0 = 3: F at the mean, stable under node doubling to 1e-14); in case DK P[J(1)^2 <= 1 + g0^2],
# of the size e^{-lam} against the factor pref of the size e^{lam}, is formed from erfc (the sum of the two erf cancelled:
# F off by 0.33 at c = 0, g0 = 3, against the conditional integral)
# (task 8) the l-rule of P[R <= 1/2] in case DU: a narrow peak of the Gaussian of X2 near the edge scale got too few nodes,
# an error of F that did not depend on x (3e-7 at c = -60, g0 = 1.5): F far in the lower tail is 0; the region of a fixed
# start: an error outside it in each case, a value inside (DT against c/test/fixed_ref.txt)
let Fu = ltratio_cdf(-9.0806812422194216, "DU", -18; g0 = 1.5, method = :single), Fk = ltratio_cdf(-0.24428, "DK", 0; g0 = 2), Fkd = ltratio_cdf(-0.24428, "DK", 0; g0 = 2, method = :double),
    Fl = ltratio_cdf(-9.836211826068519 - 12 * 0.51109, "DU", -60; g0 = 1.5, method = :single), Ft = ltratio_cdf(-3.0, "DT", -5; g0 = 2)
    check(isfinite(Fu) && abs(Fl) < 1e-11, @sprintf("DU c = -60 g0 = 1.5: F(mean - 12 sd) = %.1e (the l-rule of ncP)", Fl))
    check(abs(Fk - Fkd) < 1e-13, @sprintf("DK c = 0 g0 = 2: F(-0.24428) = %.15f, line vs conditional integral %.1e", Fk, abs(Fk - Fkd)))
    check(abs(Ft - 0.2889970388707237) < 1e-11, @sprintf("DT c = -5 g0 = 2: F(-3) = %.15f", Ft))
    msg(f) = (try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other"; end)
    ms = (msg(() -> ltratio_cdf(-2.0, "DK", 0; g0 = 2.5)), msg(() -> ltratio_pdf(-2.0, "DU", -10; g0 = 2)), msg(() -> ltratio_cdf(-9.0, "DT", -110; g0 = 0.5)), msg(() -> ltratio_mom(2, "DU", 3; g0 = 1)))
    check(all(==(D.FIXED_MSG), ms), "outside the region of a fixed start: the error \"" * D.FIXED_MSG * "\"")
end

# 17. the double zero x = -sqrt(2 c0) (c = -2.5, c0 = 2, x = -2): the conditional integral returned NaN with status 0 with a
# stationary start (a node with a(b) = 0 exactly, where K_0 is infinite) and an infinite density in cases DU, DT (right in
# case DK only); now the default is finite in the six cases (the density of case DK is infinite at x = -2), its distribution
# function agrees with the line integral's to 1e-9, its density to 1e-4 (near the double zero the density has a cusp: just
# below x = -2 the line integral's density is off by up to 5e-5, changing by as much under refinement, while the
# conditional integral's is stable to 1e-13; at x = -2 itself both are good to about 1e-6 only)
let wF = 0.0, wf = 0.0, fin = true
    for cas in ("DK", "DU", "DT", "SK", "SU", "ST"), x in (-2.0, -2.0 + 1e-9, -2.000001, -1.999)
        Fa = ltratio_cdf(x, cas, -2.5; c0 = 2); fa = ltratio_pdf(x, cas, -2.5; c0 = 2)
        Fs = ltratio_cdf(x, cas, -2.5; c0 = 2, method = :single); fs = ltratio_pdf(x, cas, -2.5; c0 = 2, method = :single)
        dkinf = cas == "DK" && x == -2.0
        fin &= isfinite(Fa) && (dkinf ? fa == Inf : isfinite(fa))
        wF = max(wF, abs(Fa - Fs)); dkinf || (wf = max(wf, abs(fa - fs) / fs))
    end
    check(fin && wF < 1e-9 && wf < 1e-4, @sprintf("the double zero x = -2 (c = -2.5, c0 = 2), six cases: finite (%s), auto vs line integral |dF| %.1e, |df/f| %.1e", fin, wF, wf))
end

# 18. the range of c (November 2026): below c = -550 the user-level functions raise an error (the moments of cases DT, ST
# fail at -600, the line integral from -650), with the same message as the C engine and the MATLAB programs; case DG far
# from the unit root (c = -480, at the median: stable to 2e-12 under node doubling, the adaptive rules and a second line)
let msg(f) = (try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end)
    ms = (msg(() -> ltratio_cdf(-20.0, "DU", -551)), msg(() -> ltratio_pdf(-20.0, "SK", -700)), msg(() -> ltratio_mom(2, "DT", -551)), msg(() -> ltratio_icdf(0.05, "DG", -551)))
    check(all(==(D.CMIN_MSG), ms) && msg(() -> ltratio_cdf(-17.0, "DU", -550)) == "", "c below -550: the error \"" * D.CMIN_MSG * "\"")
    Fg = ltratio_cdf(-15.446799628855823, "DG", -480); fg = ltratio_pdf(-15.446799628855823, "DG", -480)
    check(abs(Fg - 0.49999999999999922) < 1e-11 && abs(fg - 0.77592587896902709) < 1e-11, @sprintf("case DG c = -480: F(median) = %.15f, f = %.15f", Fg, fg))
end

# 18b. the stationary start near the unit root (November 2026): for c > -1e-4 an error (CSTAT, STAT_MSG; the same in the C
# engine and the MATLAB programs), at c = -1e-4 values; case SK at c = -1e-4 against the exact P(t <= 0) = P(Z1 Z2 <= k),
# k = -c/sqrt(1 - e^{2c}), = 1/2 + (1/pi) int_0^k K_0; and the two errors fixed then: the batched distribution function of
# case SK at tau < 1 (c = -0.1: it was 0.059632, off by 1.2e-4 at every x) and the lower end of the double integral with a
# stationary start (the coefficient statistic of case SK at c = -0.001: it was 6e-6 for 0.0010438)
let msg(f) = (try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end)
    ms = (msg(() -> ltratio_cdf(-1.94087, "SK", -9e-5)), msg(() -> ltratio_pdf(-2.0, "SU", -9e-5)), msg(() -> ltratio_mom(2, "ST", -9e-5)),
          msg(() -> ltratio_icdf(0.05, "SK", -9e-5)), msg(() -> coef_cdf(-8.039, "SK", -9e-5)), msg(() -> adfmax_cdf(-2.4, "SU", -9e-5)),
          msg(() -> ltratio_cdf(-2.0, "SU", 1)), msg(() -> D.tr_limdist([-1.0], 1, -9e-5; start = :stationary)))
    check(all(==(D.STAT_MSG), ms), "stationary start, c > -1e-4 (and c > 0): the error \"" * D.STAT_MSG * "\"")
    k = 1e-4 / sqrt(-expm1(-2e-4)); F0 = 0.5 + D.quadgk(w -> besselk(0, w), 0.0, k / 1e6, k; rtol = 1e-14)[1] / pi
    Fa = ltratio_cdf([0.0, -1.94087], "SK", -1e-4); Fc = coef_cdf(-8.039, "SK", -1e-4); Fm = adfmax_cdf(-2.43542, "SU", -1e-4); Ft = ltratio_cdf(-3.41045, "ST", -1e-4)
    check(abs(Fa[1] - F0) < 1e-12 && abs(Fa[2] - 0.028490044498374834) < 1e-11 && abs(Fc - 0.00032997244558452543) < 1e-12 && abs(Fm - 0.050000718603976146) < 1e-11 && abs(Ft - 0.050000474221289615) < 1e-11,
          @sprintf("c = -1e-4: SK F(0) = %.14f (exact %.14f), F(-1.94087) = %.12f, coef F(-8.039) = %.12e; SU ADFmax %.12f; ST F(-3.41045) %.12f", Fa[1], F0, Fa[2], Fc, Fm, Ft))
    F1 = ltratio_cdf(-1.94087, "SK", -0.1); F2 = ltratio_cdf(-1.94087, "SK", -0.1; method = :double); Fc3 = coef_cdf(-8.039, "SK", -0.001)
    check(abs(F1 - 0.059510986573375835) < 1e-11 && abs(F1 - F2) < 1e-11 && abs(Fc3 - 0.0010438037922007037) < 1e-12,
          @sprintf("case SK: F(-1.94087) at c = -0.1 %.12f (double %.1e), coef F(-8.039) at c = -0.001 %.12e", F1, abs(F1 - F2), Fc3))
end

# 19. the application functions (the applications of the paper; the API of the MATLAB programs) against stored values of
# this code (c/test/app_ref.txt and the values of the MATLAB suite ltratio_test, section 10); their cross-checks are
# dev/check_applications.jl
let t = time()
    F = adfmax_cdf([-2.5, -Inf, Inf, NaN], "DU", 0); f = adfmax_pdf(-2.5, "DU")
    check(abs(F[1] - 0.042709500744894657) < 1e-12 && abs(f - 0.10563355056449016) < 1e-12 && F[2] == 0 && F[3] == 1 && isnan(F[4]),
          @sprintf("adfmax_cdf, adfmax_pdf DU c = 0 x = -2.5: %.15f, %.15f; non-finite x", F[1], f))
    Fs = adfmax_cdf(-2.5, "SU", -10; engine = :classical); fs = adfmax_pdf(-2.5, "SU", -10; engine = :classical)   # (this engine's: the merged package takes the general engine here)
    check(abs(Fs - 0.45543929180199272) < 1e-12 && abs(fs - 0.69145494355731718) < 1e-12, @sprintf("adfmax_cdf, adfmax_pdf SU c = -10 x = -2.5: %.15f, %.15f", Fs, fs))
    q = adfmax_icdf(0.05, "DU")
    check(abs(q + 2.4354164994325092) < 1e-9, @sprintf("adfmax_icdf(0.05, \"DU\") = %.10f", q))
    Fc = coef_cdf(-14.0, "DU", -5); qc = coef_icdf(0.05, "DU")
    check(abs(Fc - 0.1938495145682877) < 1e-10 && abs(qc + 14.093590831267115) < 1e-7, @sprintf("coef_cdf(-14, \"DU\", -5) = %.12f, coef_icdf(0.05, \"DU\") = %.9f", Fc, qc))
    Pl = ltratio_region((type = "line", a = -3.5, b = -0.2639), "DK", 0); Pg = ltratio_lincdf(-0.2639, -3.5, "DK", 0)
    Pw = ltratio_region((type = "wap", k = 3.6658, omega = 10), "DK", -10)
    Pp = ltratio_region((type = "parabola", x = -1.94087), "DK", -7); Pt = ltratio_cdf(-1.94087, "DK", -7; method = :single)
    check(abs(Pl - 0.049996781931320872) < 1e-12 && abs(Pg - 0.049996781931325174) < 1e-12 && abs(Pw - 0.75806440265373276) < 1e-12 && abs(Pp - Pt) < 1e-11,
          @sprintf("ltratio_region (line, wap, parabola vs ltratio_cdf %.1e) and ltratio_lincdf: %.15f %.15f %.15f", abs(Pp - Pt), Pl, Pg, Pw))
    Ek = power_envelope([-7.0, 0.0]; engine = :classical); Eg = power_envelope(-13.5, "DG"; engine = :classical)   # (this engine's)
    check(abs(Ek[1] - 0.49940198260314117) < 1e-10 && Ek[2] == 0.05 && abs(Eg - 0.51441347319065545) < 1e-10, @sprintf("power_envelope DK c = -7 %.12f, DG c = -13.5 %.12f", Ek[1], Eg))
    xb = ltratio_bisect(x -> x^3 - 2, 0, 2, 1e-12)
    check(abs(xb - cbrt(2)) < 1e-12 && (try; ltratio_bisect(x -> x^2 + 1, -1, 1); false; catch; true; end), "ltratio_bisect: the root of x^3 - 2, and no bracket")
    @printf("  (the application functions: %.1f s)\n", time() - t)
end

# 20. the contour integral representation (method :contour, tr_contour; November 2026): a fixed start in cases DU, DT as a
# double integral with an explicit integrand, the choice of :auto there.  Stored values (c/test/contour_ref.txt, checked
# against the line and the conditional integral; the points include two roots of X(v) with c0 > 0, c0 < 0, the
# neighbourhood of the double root, c = 1.9 in the upper tail and c = -94.4), the automatic method, the three
# representations against each other and the scope of the method
let t = time(), msg(f) = (try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other"; end)
    CR = [parse.(Float64, split(l)) for l in readlines(joinpath(@__DIR__, "contour_ref.txt"))]
    wv = 0.0; ws = 0; w1 = 0.0
    for r in CR
        model = Int(r[1]); c, c0, g0, x, F, f = r[2:7]
        Fc, iF = D.tr_contour([x], model, c; c0, g0, warn = false, verified = true); fc, i_f = D.tr_contour([x], model, c; c0, g0, pdf = true, warn = false, verified = true)
        wv = max(wv, abs(Fc[1] - F), abs(fc[1] - f)); ws += iF.status[1] + i_f.status[1]
        F1, _ = D.tr_contour([x], model, c; c0, g0); f1, _ = D.tr_contour([x], model, c; c0, g0, pdf = true)    # (the default: one evaluation)
        w1 = max(w1, abs(F1[1] - F), abs(f1[1] - f))
    end
    # (6e-11: with the branch at each point, November 2026, the corner (c g0)^2/2 = 7200 of case DT, c = -40, g0 = 3, x = -13,
    # moved by 5.9e-12, to within 2e-12 of the general engine's checked value, the stored one within 3.9e-12 of it; the other
    # 29 points within 5e-13)
    check(wv < 6e-11 && ws == 0, @sprintf("contour integral, verified: %d stored points (c/test/contour_ref.txt), worst |dF|, |df| %.1e, flagged %d", length(CR), wv, ws))
    check(w1 < 5e-9, @sprintf("contour integral, one evaluation (the default): worst |dF|, |df| against the stored values %.1e (5e-9 at the corner (c g0)^2/2 = 7200)", w1))
    Fa = ltratio_cdf(-2.0, "DU", -5; g0 = 1.5); Fk = ltratio_cdf(-2.0, "DU", -5; g0 = 1.5, method = :contour)
    Fs = ltratio_cdf(-2.0, "DU", -5; g0 = 1.5, method = :single); Fd = ltratio_cdf(-2.0, "DU", -5; g0 = 1.5, method = :double)
    fa = ltratio_pdf(-3.0, "DT", -5; g0 = 2); fs = ltratio_pdf(-3.0, "DT", -5; g0 = 2, method = :single); fd = ltratio_pdf(-3.0, "DT", -5; g0 = 2, method = :double)
    check(Fa == Fk && abs(Fa - 0.864977240077167) < 1e-12 && abs(Fa - Fs) < 1e-12 && abs(Fa - Fd) < 1e-10 && abs(fa - fs) < 1e-10 && abs(fa - fd) < 1e-8,
          @sprintf("DU c = -5 g0 = 1.5: F(-2) auto = contour %.15f (MATLAB 0.864977240077167), single %.1e, double %.1e; DT c = -5 g0 = 2 f(-3): single %.1e, double %.1e",
                   Fa, abs(Fa - Fs), abs(Fa - Fd), abs(fa - fs), abs(fa - fd)))
    q = ltratio_icdf(0.05, "DT", -10; g0 = 0.5); Fq = ltratio_cdf(q, "DT", -10; g0 = 0.5, method = :double, verified = true)
    check(abs(Fq - 0.05) < 1e-10, @sprintf("ltratio_icdf(0.05, \"DT\", -10; g0 = 0.5) = %.10f, F there by the conditional integral %.1e off", q, abs(Fq - 0.05)))
    ms = (msg(() -> ltratio_cdf(-2.0, "DK", -5; g0 = 1, method = :contour)), msg(() -> ltratio_cdf(-2.0, "DU", -5; method = :contour)),
          msg(() -> ltratio_cdf(-2.0, "SU", -5; method = :contour)))
    check(all(==(D.CONTOUR_MSG), ms), "method :contour outside a fixed start in cases DU, DT: the error \"" * D.CONTOUR_MSG * "\"")
    @printf("  (the contour integral: %.1f s)\n", time() - t)
end

# 21. the default and the option verified (November 2026): every value is one evaluation by one method; verified = true
# reproduces the checked computation before the change (stored values of that engine, the rule before and its tail
# recomputation included); the warnings of the brittle regimes fire where they should, once per call, and not elsewhere
let t = time()
    VR = (("DU", -5.0, 2.0, 0.0, -3.0, 0.1976948237625359, 0.4648778393052269),
          ("DK", -10.0, -5.0, 0.0, -2.0, 0.16726497409337868, 0.3125724838563969),
          ("DT", -5.0, 0.0, 2.0, -3.0, 0.28899703887072636, 0.4636733678782745),
          ("SU", -5.0, -5.0, 0.0, -1.0, 0.44272919976045844, 0.4264723317698583),
          ("DK", -5.0, 0.0, 1.0, -2.0, 0.7176319327534152, 0.6968409849591408),
          ("DU", -5.0, 0.0, 0.0, -4.5, 0.0005234905959262681, 0.0022074999304135296))
    wv = 0.0; w1 = 0.0
    for (cas, c, c0, g0, x, F, f) in VR
        wv = max(wv, abs(ltratio_cdf(x, cas, c; c0, g0, verified = true) - F), abs(ltratio_pdf(x, cas, c; c0, g0, verified = true) - f))
        w1 = max(w1, abs(ltratio_cdf(x, cas, c; c0, g0) - F), abs(ltratio_pdf(x, cas, c; c0, g0) - f))
    end
    check(wv < 1e-14 && w1 < 1e-9, @sprintf("verified = true reproduces the checked computation at %d points to %.1e; the default (one evaluation) is within %.1e", length(VR), wv, w1))
    nwarn(f) = (io = IOBuffer(); Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(io, Base.CoreLogging.Warn)) do; f(); end; count("one evaluation", String(take!(io))))
    fire = (nwarn(() -> ltratio_cdf(-3.0, "DU", -5; c0 = 2.0, method = :double)), nwarn(() -> ltratio_cdf([-3.0, 1.0], "DK", 2; c0 = -5.0, method = :double)),
            nwarn(() -> ltratio_cdf(-5.0, "DU", -20; c0 = 2.0, g0 = 0.5)), nwarn(() -> ltratio_icdf([0.05, 0.5], "DU", -20; c0 = 2.0, g0 = 0.5)),
            nwarn(() -> ltratio_cdf(-2.0, "SK", -0.1; c0 = 1.0, method = :double)), nwarn(() -> ltratio_cdf(-2.0, "DU", 2; c0 = -2.0, g0 = 1.0, method = :contour)),
            nwarn(() -> ltratio_cdf(-8.4, "DU", 2; c0 = 5.0, g0 = 1.5, method = :contour)), nwarn(() -> ltratio_cdf(-6.0, "DK", -25; g0 = 1.0, method = :double)))
    quiet = (nwarn(() -> ltratio_cdf(-3.0, "DU", -5; c0 = 2.0)), nwarn(() -> ltratio_cdf(-2.0, "DU", -5)), nwarn(() -> ltratio_cdf(-2.0, "DU", -5; g0 = 1.5)),
             nwarn(() -> ltratio_cdf(-3.0, "DU", -5; c0 = 2.0, method = :double, verified = true)), nwarn(() -> ltratio_icdf(0.05, "DU", -5; c0 = -2.0)),
             nwarn(() -> coef_cdf([-10.0, -5.0], "DU", -5)))
    check(all(==(1), fire) && all(==(0), quiet), "the warnings of the brittle regimes: once each where they should (" * join(fire, " ") * "), none elsewhere (" * join(quiet, " ") * ")")
    # case DU with a fixed start, c0 > 3 at c > 1.5: the contour integral fails (1e44 at c = 2, c0 = 5, g0 = 1.5), :auto takes
    # the conditional integral, without a warning
    Fa = 0.0; na = nwarn(() -> (Fa = ltratio_cdf(-8.4075, "DU", 2; c0 = 5.0, g0 = 1.5)))
    Fl = ltratio_cdf(-8.4075, "DU", 2; c0 = 5.0, g0 = 1.5, method = :single)
    check(abs(Fa - Fl) < 1e-9 && na == 0, @sprintf("DU c = 2 c0 = 5 g0 = 1.5: F(-8.4075) by :auto %.12f, line integral %.1e off, %d warnings", Fa, abs(Fa - Fl), na))
    @printf("  (the default and verified: %.1f s)\n", time() - t)
end

isempty(fails) || error("quick check failed: " * join(fails, "; "))
println("quick check passed")
