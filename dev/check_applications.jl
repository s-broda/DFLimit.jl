# Cross-checks of the application functions of DFExact.jl (adfmax_cdf, adfmax_pdf, adfmax_icdf, coef_cdf, ltratio_region,
# ltratio_lincdf, power_envelope), formerly in the scripts julia/application/power.jl, wap.jl and adfmax.jl of the paper's
# repository; the reproduction of the paper is julia/replication/ltratio_paper.jl there.  Exits with an error if a check fails.
#   (1) the coefficient statistic, line against conditional integral; Gil-Pelaez against the engine (cases DK, DG); the
#       envelopes; case DG at c = 0 in closed form
#   (2) the region {U <= g(V)}: a line against Gil-Pelaez, the t-ratio's parabola against the line integral, a
#       horizontal boundary against the closed form, P(V < v0) against a boundary far above the support
#   (3) ADFmax: the outer integral without the reflected term against the engine's t-ratio, the symmetry where it holds,
#       the distribution function against the integral of the density, monotonicity and the tails, node doubling
#   (4) ADFmax: Monte Carlo of the finite-sample statistic (n = 250 and 1000, 4 million replications each), and the
#       published asymptotic critical values of Otero and Smith (2012)
# Run from the repository root: julia --project=dev -t auto dev/check_applications.jl [nomc]   (about 5 minutes on a
# ten-thread laptop; the argument nomc leaves out the Monte Carlo)
using DFLimit; const DFExact = DFLimit.DFExact; const D = DFExact
using Printf, Random, SpecialFunctions

const LEVELS = (0.01, 0.025, 0.05, 0.10)
fails = String[]
note(ok, msg) = (ok || push!(fails, msg); ok ? "PASS" : "FAIL")
T0 = time()
@printf("threads %d, engine workers %d\n", Threads.nthreads(), D.NWORKERS[])

# =====================================================================================================================
# (1) the coefficient statistic and the linear forms (power.jl)
println("\n(1) the coefficient statistic, Gil-Pelaez and the envelopes:")
let w = 0.0
    for (cas, c, x) in (("DK", -5.0, -8.0391), ("DU", -10.0, -14.0936), ("DT", -15.0, -21.7112), ("DK", -20.0, -8.0391))
        Fs = coef_cdf(x, cas, c; method = :single); Fd = coef_cdf(x, cas, c; method = :double)
        w = max(w, abs(Fs - Fd))
        @printf("  line vs conditional: %s c=%g c0=%g  %.10f  %.10f  diff %.1e\n", cas, c, x, Fs, Fd, Fs - Fd)
    end
    @printf("  coefficient statistic, line vs conditional integral: %.1e  %s\n", w, note(w < 1e-11, "coef line vs conditional"))
end
let w = 0.0
    for (cas, c, a) in (("DK", 0.0, -8.0), ("DK", -5.0, -8.0), ("DK", 0.0, -2.0), ("DG", 0.0, -20.0), ("DG", -13.5, -20.0), ("DG", -5.0, -8.0))
        p1 = ltratio_lincdf(0.0, a, cas, c); p2 = coef_cdf(a, cas, c; method = :single)
        w = max(w, abs(p1 - p2))
        @printf("  Gil-Pelaez vs engine: %s c=%g a=%g  %.10f  %.10f  diff %.1e\n", cas, c, a, p1, p2, p1 - p2)
    end
    @printf("  P(U - aV < 0) by Gil-Pelaez against the coefficient statistic: %.1e  %s\n", w, note(w < 1e-10, "Gil-Pelaez vs engine"))
end
let e7 = power_envelope(-7.0), e13 = power_envelope(-13.5, "DG")
    @printf("  envelope at -7 (ERS: 0.50) %.4f, trend envelope at -13.5 (ERS: 0.50) %.4f  %s\n", e7, e13, note(abs(e7 - 0.5) < 0.01 && abs(e13 - 0.5) < 0.02, "envelopes"))
end
for cb in (-5.0, -20.0)
    a = D.po_weight(cb, "DG")                                       # cb^2/(2 (cb - 1)), the point-optimal weight of case DG
    p1 = power_envelope(cb, "DG")
    b2 = ltratio_bisect(b -> ltratio_lincdf(b, a, "DG", 0.0) - 0.05, -50.0, 50.0); p2 = ltratio_lincdf(b2, a, "DG", cb)
    @printf("  trend envelope at cb = %g: detrended at cb %.6f, detrended at -13.5 %.6f\n", cb, p1, p2)
end
let lam = D.lam_gls(D.CBAR_GLS), F = ltratio_cdf(0.0, "DG", 0.0), Fc = erf(1 / sqrt(0.4 * (1 - lam)^2))
    @printf("  DG, c = 0: P(tau <= 0) = %.12f, closed form erf(1/sqrt(0.4 (1 - lam)^2)) = %.12f  %s\n", F, Fc, note(abs(F - Fc) < 1e-12, "DG closed form"))
end

# =====================================================================================================================
# (2) the region {U <= g(V)} (wap.jl): a line against Gil-Pelaez, the t-ratio's parabola against the line integral, and a
# horizontal boundary against the closed form P(J(1)^2 <= 1 + 2k), J(1) ~ N(0, (e^{2c} - 1)/(2c))
sJ2(c) = abs(c) < 1e-12 ? 1.0 : (exp(2c) - 1) / (2c)
P_score(c, k) = 1 + 2k > 0 ? erf(sqrt((1 + 2k) / sJ2(c) / 2)) : 0.0   # P(chi^2_1 s^2 <= 1 + 2k)
println("\n(2) the conditional integral with a general boundary:")
let w = 0.0
    for (c, a, p) in ((0.0, -3.5, 0.05), (-5.0, -3.5, 0.3), (-10.0, -1.0, 0.7), (-20.0, -3.5, 0.95))
        b = D.quant_lin(c, a, p)
        t = @elapsed (p1, inf) = D.tr_region((type = "line", a = a, b = b), c)
        p2 = ltratio_lincdf(b, a, "DK", c); w = max(w, abs(p1 - p2))
        @printf("  line b + a v, c=%5.1f a=%4.1f b=%4.1f: cond %.12f  Gil-Pelaez %.12f  diff %.1e  err est %.1e  (%.2fs, %d pairs)\n", c, a, b, p1, p2, p1 - p2, inf.err, t, inf.npairs)
    end
    for (c, x, c0) in ((0.0, -1.94087, 0.0), (-7.0, -1.94087, 0.0), (-5.0, -2.0, -5.0), (-15.0, 1.0, -8.0))
        t = @elapsed (p1, inf) = D.tr_region((type = "parabola", x = x, c0 = c0), c)
        p2 = ltratio_cdf(x, "DK", c; c0 = c0, method = :single); w = max(w, abs(p1 - p2))
        @printf("  t-ratio x=%6.3f c0=%4.1f c=%5.1f: cond %.12f  line integral %.12f  diff %.1e  err est %.1e  (%.2fs)\n", x, c0, c, p1, p2, p1 - p2, inf.err, t)
    end
    for (c, k) in ((0.0, -0.498), (-5.0, -0.498), (-10.0, -0.3))
        t = @elapsed (p1, inf) = D.tr_region((type = "fun", g = v -> k), c)
        p2 = P_score(c, k); w = max(w, abs(p1 - p2))
        @printf("  horizontal k=%6.3f c=%5.1f: cond %.12f  closed form %.12f  diff %.1e  err est %.1e  (%.2fs)\n", k, c, p1, p2, p1 - p2, inf.err, t)
    end
    # P(V < v0) (the part of a region where g = +Inf) against the conditional integral with a boundary far above the support
    for (c, v0) in ((0.0, 0.05), (-5.0, 0.03))
        p1 = ltratio_lincdf(v0, NaN, "DK", c); p2 = ltratio_region((type = "fun", g = v -> v < v0 ? 1e3 : -1e3), "DK", c)
        w = max(w, abs(p1 - p2))
        @printf("  P(V < %g) at c=%g: Gil-Pelaez %.12f  conditional %.12f  diff %.1e\n", v0, c, p1, p2, p1 - p2)
    end
    @printf("  worst difference %.1e  %s\n", w, note(w < 1e-10, "general boundary"))
end

# =====================================================================================================================
# (3) ADFmax, internal checks (adfmax.jl)
println("\n(3b0) the outer integral of tr_adfmax without the reflected term (the t-ratio) against the engine:")
let d0 = 0.0, d1 = 0.0
    for (cas, c, x) in (("DU", 0.0, -2.5), ("DT", 0.0, -3.4), ("DT", 0.0, 0.5), ("DU", -10.0, -2.0), ("DT", -10.0, -3.0),
                        ("SU", -10.0, -2.5), ("ST", -10.0, -3.0), ("ST", -1.0, -1.0), ("SU", -25.0, -4.0))
        F = D.tr_adfmax(x, cas, c; refl = false, trunc = false)[1]; f = D.tr_adfmax(x, cas, c; pdf = true, refl = false, trunc = false)[1]
        Fd = ltratio_cdf(x, cas, c; method = :double); Fs = ltratio_cdf(x, cas, c; method = :single); fs = ltratio_pdf(x, cas, c; method = :single)
        d0 = max(d0, abs(F - Fd)); d1 = max(d1, abs(F - Fs), abs(f - fs))
        @printf("  %s c=%6.1f x=%5.2f: F %.12f  engine double %.12f  single %.12f | f %.12f  single %.12f\n", cas, c, x, F, Fd, Fs, f, fs)
    end
    @printf("  max |diff| against the engine's conditional integral %.1e, against its line integral (F and f) %.1e: %s\n", d0, d1, note(d0 < 1e-12 && d1 < 1e-10, "ADFmax b0"))
end
println("\n(3b1) symmetry where it holds (c = 0; stationary start): F_max = 2 F_tau - 1 (x >= 0), = 2 F_fwd,trunc - 1 (x < 0):")
for (cas, c) in (("DU", 0.0), ("DT", 0.0), ("SU", -10.0), ("ST", -10.0), ("ST", -2.0), ("SU", -25.0))
    dm = 0.0
    for x in (-4.0, -3.0, -2.0, -1.0, -0.3, 0.0, 0.5, 1.5)
        F = adfmax_cdf(x, cas, c)
        S = x >= 0 ? 2ltratio_cdf(x, cas, c; method = :single) - 1 : 2D.tr_adfmax(x, cas, c; refl = false, trunc = true)[1] - 1
        dm = max(dm, abs(F - S))
    end
    @printf("  %s c=%6.1f: max |F_max - symmetric form| over x in {-4,-3,-2,-1,-0.3,0,0.5,1.5}: %.1e  %s\n", cas, c, dm, note(dm < 1e-10, "ADFmax symmetry $cas $c"))
end
let x = -2.5, c = -10.0
    for cas in ("DU", "DT")
        F = adfmax_cdf(x, cas, c); S = 2D.tr_adfmax(x, cas, c; refl = false, trunc = true)[1] - 1
        @printf("  (no symmetry with a zero start at c < 0: %s c = %g, x = %g: F_max %.6f, the symmetric form %.6f)\n", cas, c, x, F, S)
    end
end
cvmax = Dict(cas => adfmax_icdf(collect(LEVELS), cas, 0.0) for cas in ("DU", "DT"))
println("\n(3b2) the distribution function against the integral of the density (16-point Gauss-Legendre on each interval):")
let (zl, wl) = D.rule(16)
    for (cas, c) in (("DU", 0.0), ("DT", 0.0), ("DU", -10.0), ("DT", -10.0), ("SU", -10.0), ("ST", -10.0))
        dm = 0.0
        for (x1, x2) in ((-6.0, -4.5), (-4.5, -3.5), (-3.5, -2.5), (-2.5, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 2.0))
            I = sum(wl[k] * (x2 - x1) * adfmax_pdf(x1 + (x2 - x1) * zl[k], cas, c) for k in eachindex(zl))
            dm = max(dm, abs(I - (adfmax_cdf(x2, cas, c) - adfmax_cdf(x1, cas, c))))
        end
        @printf("  %s c=%6.1f: max |int f - (F(x2) - F(x1))| over 7 intervals covering [-6, 2]: %.1e  %s\n", cas, c, dm, note(dm < 1e-10, "ADFmax b2 $cas $c"))
    end
end
println("(3b3) P(ADFmax <= x) <= P(tau <= x) and F_max increasing on x = -6:0.25:4, and the tails (1 - F_max(4) < 1e-6):")
for (cas, c) in (("DU", 0.0), ("DT", 0.0), ("DU", -10.0), ("DT", -10.0), ("SU", -10.0), ("ST", -10.0))
    xs = collect(-6.0:0.25:4.0)
    Fm = adfmax_cdf(xs, cas, c); Ft = ltratio_cdf(xs, cas, c)
    viol = maximum(Fm .- Ft); mono = minimum(diff(Fm))
    @printf("  %s c=%6.1f: max (F_max - F_tau) = %.1e, min increment %.1e; F_max(-6) = %.2e, 1 - F_max(x) at x = 1, 2, 3, 4: %.2e %.2e %.2e %.2e  %s\n",
            cas, c, viol, mono, Fm[1], 1 - Fm[29], 1 - Fm[33], 1 - Fm[37], 1 - Fm[41], note(viol <= 1e-12 && mono >= -1e-13 && 1 - Fm[41] < 1e-6, "ADFmax b3 $cas $c"))
end
println("(3b4) node doubling (M 64 -> 128, Nv 96 -> 192, pieces of length 2 -> 1, outer tolerance 1e-12 -> 1e-14; stationary start Nl 96 -> 192):")
let dm = 0.0, dmf = 0.0
    for (cas, c, x) in (("DU", 0.0, cvmax["DU"][1]), ("DU", 0.0, cvmax["DU"][3]), ("DT", 0.0, cvmax["DT"][1]), ("DT", 0.0, cvmax["DT"][3]),
                        ("DT", 0.0, 0.3), ("DU", -10.0, cvmax["DU"][3]), ("DT", -20.0, cvmax["DT"][3]), ("DT", -5.0, -0.2),
                        ("SU", -10.0, cvmax["DU"][3]), ("ST", -20.0, cvmax["DT"][3]), ("ST", -5.0, -0.2), ("ST", -0.5, cvmax["DT"][1]))
        kw = cas[1] == 'S' ? (M = 128, Nv = 192, plen = 1.0, otol = 1e-14, Nl = 192) : (M = 128, Nv = 192, plen = 1.0, otol = 1e-14)
        F1 = adfmax_cdf(x, cas, c); F2 = adfmax_cdf(x, cas, c; kw...)
        f1 = adfmax_pdf(x, cas, c); f2 = adfmax_pdf(x, cas, c; kw...)
        dm = max(dm, abs(F1 - F2)); dmf = max(dmf, abs(f1 - f2))
        @printf("  %s c=%6.1f x=%8.5f: F %.14f  doubled %.14f  diff %.1e | f diff %.1e\n", cas, c, x, F1, F2, F1 - F2, f1 - f2)
    end
    @printf("  max change: F %.1e, f %.1e  %s\n", dm, dmf, note(dm < 1e-9 && dmf < 1e-9, "ADFmax node doubling"))
end

# =====================================================================================================================
# (4) Monte Carlo of the finite-sample statistic: Gaussian AR(1) y_t = rho y_{t-1} + e_t, rho = 1 + c/n, t = 1..n, y_0 = 0
# (zero start) or y_0 ~ N(0, 1/(1 - rho^2)) (stationary start); the forward regression of y_t on 1, (t,) y_{t-1} and the
# reverse regression of y_{t-1} on 1, (t,) y_t over the same n pairs (the reverse regression of y_t on y_{t+1},
# t = 0..n-1); OLS t-statistics of the coefficient one, s^2 with n - k degrees of freedom.  Both from the same moments of
# the demeaned (detrended) pairs (a_i, b_i) = (y_{i-1}, y_i), with the time index centred so that 1 and t are orthogonal.
@inline function dfpair(y::Vector{Float64}, n::Int, model::Int)
    tbar = (n + 1) / 2; stt = n * (n * n - 1) / 12
    sa = 0.0; sb = 0.0; sta = 0.0; stb = 0.0
    @inbounds @simd for i in 1:n
        a = y[i]; b = y[i+1]; t = i - tbar
        sa += a; sb += b; sta += t * a; stb += t * b
    end
    ma = sa / n; mb = sb / n
    ga = model == 3 ? sta / stt : 0.0; gb = model == 3 ? stb / stt : 0.0
    saa = 0.0; sbb = 0.0; sab = 0.0
    @inbounds @simd for i in 1:n
        t = i - tbar
        a = y[i] - ma - ga * t; b = y[i+1] - mb - gb * t
        saa += a * a; sbb += b * b; sab += a * b
    end
    df = n - model
    pf = sab / saa; tf = (pf - 1) / sqrt((sbb - pf * sab) / df / saa)
    pr = sab / sbb; tr = (pr - 1) / sqrt((saa - pr * sab) / df / sbb)
    tf, tr
end
function simchunk!(TF::Vector{Float64}, TR::Vector{Float64}, rng::Xoshiro, y::Vector{Float64}, e::Vector{Float64}, n::Int,
                   model::Int, c::Float64, stat::Bool, r1::Int, r2::Int)
    rho = 1 + c / n; s0 = stat ? 1 / sqrt(1 - rho * rho) : 0.0
    for r in r1:r2
        randn!(rng, e)
        y[1] = s0 * randn(rng)
        @inbounds for t in 1:n; y[t+1] = rho * y[t] + e[t]; end
        TF[r], TR[r] = dfpair(y, n, model)
    end
    nothing
end
# R replications in chunks of 10^4, each chunk with its own generator (reproducible whatever the number of threads)
function simulate(n::Int, R::Int, model::Int, c::Float64, stat::Bool; seed::Int = 1, chunk::Int = 10_000)
    TF = Vector{Float64}(undef, R); TR = similar(TF)
    nch = cld(R, chunk)
    Threads.@threads :dynamic for k in 1:nch
        rng = Xoshiro(1_000_003 * seed + k); y = Vector{Float64}(undef, n + 1); e = Vector{Float64}(undef, n)
        simchunk!(TF, TR, rng, y, e, n, model, c, stat, (k - 1) * chunk + 1, min(k * chunk, R))
    end
    TF, TR
end
ecdf_at(sorted::Vector{Float64}, x::Float64) = searchsortedlast(sorted, x) / length(sorted)

# The published asymptotic critical values: the intercepts (T -> infinity, lag order fixed) of the response surfaces of
# Otero and Smith (2012, Computational Statistics 27, 473-486) as distributed with the Stata command adfmaxur of Otero and
# Baum (2018, Stata Journal 18, 22-28): the matrix file adfmaxur.mtx of the SSC package (fmwww.bc.edu/repec/bocode/a/,
# downloaded 10 November 2026, SHA-256 0392017c...9da4537b), 221 x 91 doubles, column 1 the probabilities, columns
# fixb_c1 and fixb_ct1 the intercepts for the constant and the constant-and-trend case with the lag order fixed (the
# other lag-selection rules have intercepts within 0.015 of these), rows 13, 16, 21, 31 the levels 1, 2.5, 5, 10%; with
# each the coefficients of 1/T, ..., 1/T^4 (fixb_c2..5, fixb_ct2..5; T the number of observations less one, the
# lag terms (p/T)^k vanish at p = 0).  Leybourne (1995) itself tabulates finite samples T = 25, ..., 400 only (Otero and
# Baum 2018, p. 22); it was not accessible here.
const OTERO_SMITH = Dict(
    "DU" => ((-3.021979328347456, -3.1047376833059843, -135.55705997293626, 3514.1775951558375, -33102.84717447683),
             (-2.701327771271952, -1.5458981260230757, -117.4656995357982, 3246.780369904911, -30673.232393339742),
             (-2.4309652855572104, -0.29702939825835983, -121.78004083703854, 3460.646479184208, -32407.54236994963),
             (-2.1243348434283575, 0.6960996665803805, -115.55973083782919, 3304.1682427481865, -30825.742271705065)),
    "DT" => ((-3.6728014591352434, -4.805021308070536, -261.57601506322317, 6416.350970728177, -58477.46181068197),
             (-3.3689069522036377, -3.1017395547619344, -191.99854449926715, 5007.993985214096, -46541.852991039865),
             (-3.1130093024423076, -1.680092993412245, -166.70739407074507, 4726.036624439272, -45575.9066889761),
             (-2.8242192449570567, -0.3092378083974836, -144.27482500122642, 4304.052969703225, -41978.37986538699)))

qfin = Dict{Tuple{String,Int},Vector{Float64}}()                    # simulated quantiles at LEVELS, c = 0
const R = 4_000_000
if !("nomc" in ARGS)
    PS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    println("\n(4a) Monte Carlo: Gaussian AR(1), n = 250 and 1000, $R replications each; the simulated distribution function at")
    println("    the exact quantiles x_p of the limit; d = F_n(x_p) - p, se = sqrt(p(1-p)/R), z = d/se; ext = (4 d_1000 - d_250)/3")
    println("    (the limit with an O(1/n) term removed), its z with se_ext = sqrt(17) se/3; the forward t-ratio alone as a")
    println("    calibration (its d at the same points against the engine's P(tau <= x_p), and the z of its extrapolation, whose limit")
    println("    is known: residuals of that size are finite-sample terms beyond O(1/n)); for the zero start at c < 0 the")
    println("    column 'sym' is the z of the simulation against the symmetric formula, which does not apply there")
    mcok = true; tmc = 0.0
    for (cas, c, lab) in (("DU", 0.0, "zero"), ("DU", -10.0, "zero"), ("SU", -10.0, "stationary"), ("DT", 0.0, "zero"), ("DT", -10.0, "zero"), ("ST", -10.0, "stationary"))
        model, stat = D.casemodel(cas)
        xq = adfmax_icdf(collect(PS), cas, c)
        Ftq = ltratio_cdf(xq, cas, c)
        sym = (!stat && c < 0) ? [x >= 0 ? 2ltratio_cdf(x, cas, c) - 1 : 2D.tr_adfmax(x, cas, c; refl = false, trunc = true)[1] - 1 for x in xq] : fill(NaN, length(xq))
        res = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
        for n in (250, 1000)
            global tmc += @elapsed TF, TR = simulate(n, R, model, c, stat; seed = n + 7 * model + round(Int, -c) + (stat ? 100 : 0))
            M = sort!(max.(TF, TR)); sort!(TF)
            res[n] = ([ecdf_at(M, x) for x in xq], [ecdf_at(TF, x) for x in xq])
            c == 0 && (qfin[(cas, n)] = [M[ceil(Int, a * R)] for a in LEVELS])
        end
        @printf("\n  %s, c = %g, %s start%s\n", cas[2] == 'U' ? "constant" : "constant and trend", c, lab, (c == 0 ? " (the start does not matter at c = 0)" : ""))
        @printf("  %6s %9s | %9s %6s | %9s %6s | %9s %6s | %8s %8s %6s | %8s\n", "p", "x_p", "d_250", "z", "d_1000", "z", "ext", "z", "tau d250", "d1000", "ext z", "sym z")
        se = [sqrt(p * (1 - p) / R) for p in PS]
        zmax = 0.0; zmax1000 = 0.0
        for (k, p) in enumerate(PS)
            d250 = res[250][1][k] - p; d1000 = res[1000][1][k] - p; ext = (4d1000 - d250) / 3
            zext = ext / (sqrt(17) * se[k] / 3)
            zmax = max(zmax, abs(zext)); zmax1000 = max(zmax1000, abs(d1000 / se[k]))
            t250 = res[250][2][k] - Ftq[k]; t1000 = res[1000][2][k] - Ftq[k]
            zt = (4t1000 - t250) / 3 / (sqrt(17 * Ftq[k] * (1 - Ftq[k]) / R) / 3)
            zs = isnan(sym[k]) ? NaN : (res[1000][1][k] - sym[k]) / se[k]
            @printf("  %6.3f %9.5f | %+9.5f %6.1f | %+9.5f %6.1f | %+9.5f %6.1f | %+8.5f %+8.5f %6.1f | %8.0f\n", p, xq[k], d250, d250 / se[k], d1000, d1000 / se[k], ext, zext, t250, t1000, zt, zs)
        end
        ok = zmax <= 4
        global mcok &= ok
        @printf("  max |z| at n = 1000: %.1f; max |z| of the extrapolated difference: %.1f  %s\n", zmax1000, zmax, note(ok, "Monte Carlo $cas $c"))
    end
    @printf("\n  (4a) the Monte Carlo agrees with the limit (extrapolated differences within 4 se everywhere): %s  (simulation %.0fs)\n", mcok ? "PASS" : "FAIL", tmc)
end

println("\n(4c) published asymptotic critical values: the T -> infinity intercepts of the response surfaces of Otero and Smith")
println("    (2012) in the file adfmaxur.mtx of the Stata command adfmaxur (Otero and Baum 2018), lag order fixed; Leybourne's")
println("    (1995) own table (T = 25 to 400) was not accessible")
println("    and their finite-sample values at T = n pairs with no lags, cv(T) = sum_k theta_k T^{-k}, beside the quantiles of the")
println("    simulation (4a; not with the argument nomc), whose standard errors are sqrt(a(1 - a)/R)/f(x_a)")
for cas in ("DU", "DT")
    @printf("  %s  se of a simulated quantile %s\n", cas, join([@sprintf("%9.5f", sqrt(a * (1 - a) / R) / adfmax_pdf(cvmax[cas][k], cas, 0.0)) for (k, a) in enumerate(LEVELS)], " "))
    @printf("  %s  level             %s\n", cas, join([@sprintf("%9.3f", a) for a in LEVELS], " "))
    @printf("      exact limit       %s\n", join([@sprintf("%9.5f", q) for q in cvmax[cas]], " "))
    @printf("      O-S intercept     %s\n", join([@sprintf("%9.5f", th[1]) for th in OTERO_SMITH[cas]], " "))
    @printf("      difference        %s\n", join([@sprintf("%+9.5f", OTERO_SMITH[cas][k][1] - cvmax[cas][k]) for k in 1:4], " "))
    @printf("      exact size at O-S %s\n", join([@sprintf("%9.5f", adfmax_cdf(th[1], cas, 0.0)) for th in OTERO_SMITH[cas]], " "))
    for n in (250, 1000)
        @printf("      n = %4d: O-S     %s\n", n, join([@sprintf("%9.5f", sum(th[k] / n^(k - 1) for k in 1:5)) for th in OTERO_SMITH[cas]], " "))
        haskey(qfin, (cas, n)) && @printf("              simulated %s\n", join([@sprintf("%9.5f", q) for q in qfin[(cas, n)]], " "))
    end
end

@printf("\nrun time %.0f s\n", time() - T0)
isempty(fails) || error("check_applications failed: " * join(fails, "; "))
println("all checks passed")
