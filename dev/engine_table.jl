# The measurements behind the choice of engine (DFLimit.autoengine): each cell of the table in the README, "Which engine",
# by the classical engine (DFLimit.DFExact, engine = :classical) and the general engine (DFLimit.DFDet, engine = :general):
# the warm time of one call (BenchmarkTools minimum of 2 after a warm-up; the general engine's outer path cached, its
# time given separately), the largest difference of the two, and where they differ by more than 1e-8 the checked values of
# both (verified = true) to say which is right.  Writes dev/engine_table.txt.
#
#   jlcap --project=<an environment with DFLimit and BenchmarkTools> dev/engine_table.jl
using DFLimit, BenchmarkTools, Printf, Logging, Statistics
const DX = DFLimit.DFExact; const DG = DFLimit.DFDet
const OUT = joinpath(@__DIR__, "engine_table.txt")
io = open(OUT, "w")
pr(args...) = (print(io, args...); flush(io); print(args...))
pr(@sprintf("The choice of engine: warm times in ms (BenchmarkTools minimum of 2), %d threads; dF, df the largest differences of the two\n", Threads.nthreads()))
pr("engines; auto: the engine of DFLimit.autoengine in the cell; ref: where they differ by more than 1e-8, the distance of each to\n")
pr("the other's checked value (verified = true).\n\n")
const MD = DFLimit
quiet(f) = with_logger(NullLogger()) do; f(); end
tm(f) = (quiet(f); quiet(() -> @belapsed $f() samples = 2 evals = 1))
mean_sd(cas, c; kw...) = (m = ltratio_mom(2, cas, c; kw...); (m[1], sqrt(max(m[2] - m[1]^2, 1e-12))))
grid(cas, c; ks = (-2, -1, 0, 1, 2), kw...) = (ms = mean_sd(cas, c; kw...); [ms[1] + k * ms[2] for k in ks])
function row(label, fc, fg; fcv = nothing, fgv = nothing, note = "", auto = :classical)
    vc = quiet(fc); vg = quiet(fg)
    d = maximum(abs, vcat(vc) .- vcat(vg))
    tc = tm(fc); tg = tm(fg)
    ref = ""
    if d > 1e-8 && fcv !== nothing && fgv !== nothing
        rc = quiet(fcv); rg = quiet(fgv)
        ref = @sprintf("  ref: classical - its checked %.1e, general - its checked %.1e, classical - general checked %.1e, general - classical checked %.1e",
                       maximum(abs, vcat(vc) .- vcat(rc)), maximum(abs, vcat(vg) .- vcat(rg)), maximum(abs, vcat(vc) .- vcat(rg)), maximum(abs, vcat(vg) .- vcat(rc)))
    end
    pr(@sprintf("  %-62s classical %9.2f  general %9.2f  ratio g/c %7.2f  diff %8.1e  auto %-9s%s%s\n", label, 1e3tc, 1e3tg, tg / tc, d, auto, note, ref))
end
F(cas, x, c; kw...) = ltratio_cdf(x, cas, c; engine = :classical, kw...)
Fg(cas, x, c; kw...) = ltratio_cdf(x, cas, c; engine = :general, kw...)
fF(cas, x, c; kw...) = ltratio_pdf(x, cas, c; engine = :classical, kw...)
fFg(cas, x, c; kw...) = ltratio_pdf(x, cas, c; engine = :general, kw...)
function tcell(label, cas, c; c0 = 0.0, g0 = 0.0, ks = (-2, -1, 0, 1, 2), xs = nothing)
    xs = xs === nothing ? grid(cas, c; ks, c0, g0) : xs
    au = MD.autoengine(:dist, MD.case_spec(cas, g0), Float64(c), Float64(c0), xs[cld(length(xs), 2)])
    row(label * @sprintf(" (%s, c = %g%s%s)", cas, c, c0 != 0 ? ", c0 = $c0" : "", g0 != 0 ? ", g0 = $g0" : ""),
        () -> vcat(F(cas, xs, c; c0, g0), fF(cas, xs, c; c0, g0)), () -> vcat(Fg(cas, xs, c; c0, g0), fFg(cas, xs, c; c0, g0));
        fcv = () -> vcat(F(cas, xs, c; c0, g0, verified = true), fF(cas, xs, c; c0, g0, verified = true)),
        fgv = () -> vcat(Fg(cas, xs, c; c0, g0, verified = true), fFg(cas, xs, c; c0, g0, verified = true)), auto = au)
end

pr("(1) the t-ratio, F and f at x = mean + (-2, -1, 0, 1, 2) sd\n")
for cas in ("DK", "DU", "DT"), c in (0.0, -10.0, -50.0); tcell("zero start, c0 = 0, -100 <= c <= 5", cas, c); end
tcell("zero start, c0 = 0, -100 <= c <= 5", "DG", 0.0; xs = [-3.5, -2.8, -2.0, -1.2, -0.5]); tcell("zero start, c0 = 0, -100 <= c <= 5", "DG", -10.0; xs = [-4.5, -3.5, -2.5, -1.5, -0.5])
for cas in ("DT",), c in (3.0, 5.0); tcell("zero start, c0 = 0, 0 < c <= 5", cas, c); end
for cas in ("DK", "DU", "DT"); tcell("zero start, c0 = 0, c < -100", cas, -300.0); end
for (cas, c, c0) in (("DU", -10.0, -10.0), ("DK", -5.0, -5.0), ("DT", -5.0, -5.0), ("DU", -50.0, -50.0), ("DK", -20.0, -20.0), ("DT", -100.0, -100.0)); tcell("zero start, c0 < 0, -100 <= c <= 0", cas, c; c0); end
for (cas, c, c0) in (("DT", -20.0, 3.0), ("DU", -5.0, 2.0), ("DT", -50.0, 5.0), ("DK", -10.0, 2.0), ("DK", -2.0, 1.0), ("DK", 0.0, 2.0), ("DU", 0.0, 2.0), ("DT", 1.0, 3.0)); tcell("zero start, c0 > 0, -100 < c <= 1", cas, c; c0); end
for (cas, c, c0) in (("DU", -100.0, 3.0), ("DU", 2.0, 1.0), ("DK", 2.0, 2.0)); tcell("zero start, c0 > 0, otherwise", cas, c; c0); end
for cas in ("DK", "DU", "DT"); tcell("zero start, c0 != 0, c < -100", cas, -300.0; c0 = -300.0); end
tcell("zero start, c0 != 0, c < -100", "DU", -480.0; c0 = -480.0)
for (cas, c) in (("SU", -10.0), ("ST", -50.0)); tcell("stationary start, c0 = 0", cas, c); end
for (cas, c) in (("SK", -5.0), ("SU", -10.0), ("ST", -50.0), ("SU", -100.0), ("SK", -300.0)); tcell("stationary start, c0 = c, c <= -5", cas, c; c0 = c); end
for (cas, c) in (("SK", -1.0), ("SU", -0.5), ("SK", -0.1), ("SU", -0.1), ("SU", -3e-3), ("ST", -3e-3)); tcell("stationary start, c0 = c, -5 < c <= -3e-3", cas, c; c0 = c); end
for (cas, c) in (("SU", -1e-3), ("ST", -1e-3), ("SK", -1e-4)); tcell("stationary start, c0 = c, c > -3e-3", cas, c; c0 = c); end
tcell("stationary start, c0 > 0", "ST", -1.0; c0 = 2.0)
for (cas, c, c0) in (("DK", 5.0, -3.0), ("DU", 5.0, -3.0), ("DT", 5.0, -3.0), ("DK", 4.0, -0.5)); tcell("above the unit root, c0 < 0, c >= 4", cas, c; c0, ks = (-1, 0, 1)); end
for (cas, c) in (("DT", 7.0), ("DK", 7.0), ("DU", 10.0)); tcell("above the unit root, c >= 7, the bulk", cas, c; ks = (-1, 0, 1)); end
for (cas, c, g0) in (("DK", 0.0, 1.0), ("DK", -0.5, 0.5), ("DK", -5.0, 2.0), ("DK", -50.0, 0.5), ("DK", 1.0, 1.0)); tcell("fixed start, case DK, c <= 1, (c g0)^2/2 <= 400", cas, c; g0); end
tcell("fixed start, case DK, c <= 1, (c g0)^2/2 <= 400", "DK", -10.0; c0 = -2.0, g0 = 1.0)
for (cas, c, g0) in (("DK", -100.0, 0.5), ("DK", -80.0, 0.3)); tcell("fixed start, case DK, c >= -100, g0 <= 0.5", cas, c; g0); end
for (cas, c, g0) in (("DK", 2.0, 1.0), ("DK", -20.0, 2.0), ("DK", -100.0, 1.0)); tcell("fixed start, case DK, otherwise", cas, c; g0); end
for (cas, c, g0) in (("DU", -10.0, 1.0), ("DT", 0.0, 2.0), ("DU", -50.0, 1.5)); tcell("fixed start, cases DU, DT", cas, c; g0); end

pr("\n(2) the moments E t^k, k = 1..4 (diff: relative)\n")
function mrow(label, cas, c; c0 = 0.0, g0 = 0.0)
    fc = () -> ltratio_mom(4, cas, c; c0, g0, engine = :classical); fg = () -> ltratio_mom(4, cas, c; c0, g0, engine = :general)
    vc = quiet(fc); vg = quiet(fg); d = maximum(abs.(vg ./ vc .- 1)); tc = tm(fc); tg = tm(fg)
    au = MD.autoengine(:mom, MD.case_spec(cas, g0), Float64(c), Float64(c0), 0.0)
    pr(@sprintf("  %-62s classical %9.2f  general %9.2f  ratio g/c %7.2f  diff %8.1e  auto %-9s\n", label * " ($cas, c = $c$(c0 != 0 ? ", c0 = $c0" : "")$(g0 != 0 ? ", g0 = $g0" : ""))", 1e3tc, 1e3tg, tg / tc, d, au))
end
for (cas, c) in (("DK", -5.0), ("DU", -5.0), ("DT", -5.0), ("DT", 0.0), ("SK", -5.0), ("SU", -5.0), ("ST", -5.0)); mrow("moments", cas, c); end
mrow("moments, fixed start", "DU", -5.0; g0 = 1.0); mrow("moments", "DU", -50.0); mrow("moments", "DU", -100.0); mrow("moments", "DK", -480.0)
mrow("moments, stationary, c0 != 0", "SK", -300.0; c0 = -300.0); mrow("moments, stationary, c0 != 0", "ST", -200.0; c0 = -200.0); mrow("moments, stationary, c0 != 0", "SU", -5.0; c0 = -5.0)
mrow("moments, zero start, c0 != 0", "DK", -100.0; c0 = -100.0); mrow("moments, zero start, c0 != 0", "DU", -480.0; c0 = -480.0); mrow("moments, zero start, c0 != 0", "DT", -300.0; c0 = -300.0); mrow("moments, zero start, c0 != 0", "DU", -5.0; c0 = -5.0)

pr("\n(3) quantiles and the coefficient statistic\n")
for (cas, c, g0) in (("DU", 0.0, 0.0), ("DU", -10.0, 0.0), ("DT", -5.0, 0.0), ("ST", -10.0, 0.0), ("DK", -5.0, 1.0), ("DU", -5.0, 1.0), ("DK", 0.0, 0.0), ("DU", -50.0, 0.0), ("SU", -5.0, 0.0), ("DT", 3.0, 0.0), ("DG", 0.0, 0.0))
    au = MD.autoengine(:icdf, MD.case_spec(cas, g0), c, 0.0, 0.0)
    row("ltratio_icdf at p = 0.01, 0.05, 0.5 ($cas, c = $c$(g0 != 0 ? ", g0 = $g0" : ""))", () -> ltratio_icdf([0.01, 0.05, 0.5], cas, c; g0, engine = :classical),
        () -> ltratio_icdf([0.01, 0.05, 0.5], cas, c; g0, engine = :general); auto = au)
end
for (cas, c) in (("DU", 0.0), ("SU", -5.0), ("DT", -10.0))
    au = MD.autoengine(:dist, MD.case_spec(cas), c, -10.0, 0.0)
    row("coef_cdf at y = -20, -10, -5 ($cas, c = $c)", () -> coef_cdf([-20.0, -10.0, -5.0], cas, c; engine = :classical), () -> coef_cdf([-20.0, -10.0, -5.0], cas, c; engine = :general); auto = au)
end
row("coef_icdf at p = 0.05 (DU, c = 0)", () -> coef_icdf(0.05, "DU"; engine = :classical), () -> coef_icdf(0.05, "DU"; engine = :general))

pr("\n(4) the linear form P(U - aV <= b) and the law of the denominator P(V <= b)\n")
for (cas, a, bs, c) in (("DK", -3.5, [-0.6, -0.2639, 0.3], 0.0), ("DK", -3.5, [-0.6, -0.2639, 0.3], -7.0), ("DK", -1.0, [-0.4, 0.0, 0.5], 0.0), ("DK", 2.0, [-2.0, 0.0, 0.5], -5.0),
                        ("DK", 0.0, [-0.499, -0.4, 0.0, 1.0], 0.0), ("DK", -0.1, [-0.49, -0.4, 0.0, 0.5], 0.0), ("DK", 0.0, [-0.4, 0.5, 3.0], 3.0), ("DG", -7.3, [-0.9, -0.5, 0.0], 0.0),
                        ("DG", 0.0, [-0.49, -0.3, 0.5], 0.0), ("DK", NaN, [0.05, 0.2, 0.6], -5.0), ("DK", NaN, [0.05, 0.2, 0.6], 2.0), ("DK", NaN, [0.02, 0.06, 0.1], 5.0),
                        ("DK", NaN, [0.02, 0.06, 0.1], 7.0), ("DK", -1.0, [-0.3, 0.0, 50.0], 5.0), ("DK", 2.5, [-160.0, -0.49, 0.0], 5.0))
    isnan(a) && (bs = DFLimit.DFExact.EVdk(c) .* [0.3, 1.0, 2.5])
    au = MD.autoengine(:lincdf, MD.lin_spec(cas, -13.5), c, 0.0, a)
    row("ltratio_lincdf, a = $a$(isnan(a) ? " (P(V <= b))" : "") ($cas, c = $c)", () -> ltratio_lincdf(bs, a, cas, c; engine = :classical), () -> ltratio_lincdf(bs, a, cas, c; engine = :general); auto = au)
end
# case DG with GLS detrending at small |cbar| (the vertical line of the general engine)
for (cb, c, a) in ((-5.0, 0.0, -2.0), (-2.0, -5.0, -7.3), (-1.0, 0.0, -2.0), (-1.0, 2.0, -2.0))
    bs = [-0.9, -0.5, 0.0, 0.5]
    au = MD.autoengine(:lincdf, MD.lin_spec("DG", cb), c, 0.0, a)
    row("ltratio_lincdf, a = $a (DG, cbar = $cb, c = $c)", () -> ltratio_lincdf(bs, a, "DG", c; cbar = cb, engine = :classical), () -> ltratio_lincdf(bs, a, "DG", c; cbar = cb, engine = :general); auto = au)
end
pr("\n(5) the power envelope, ADFmax, regions\n")
for (cas, cs) in (("DK", [-5.0, -10.0, -20.0]), ("DK", [-0.01, -0.5]), ("DG", [-5.0, -10.0, -20.0]), ("DG", [-1.0, -2.0]))
    au = MD.autoengine(:envelope, MD.case_spec(cas), cs[1], 0.0, 0.0)
    row("power_envelope at c = $(join(cs, ", ")) ($cas)", () -> power_envelope(cs, cas; engine = :classical), () -> power_envelope(cs, cas; engine = :general); auto = au)
end
for (cas, c) in (("DU", 0.0), ("DT", -10.0), ("SU", -10.0), ("ST", -5.0), ("SU", -0.1), ("ST", -50.0))
    au = MD.autoengine(:adfmax, MD.case_spec(cas), c, 0.0, 0.0)
    row("adfmax_cdf, adfmax_pdf at x = -3, -2.4, -1.5 ($cas, c = $c)", () -> vcat(adfmax_cdf([-3.0, -2.4, -1.5], cas, c; engine = :classical), adfmax_pdf([-3.0, -2.4, -1.5], cas, c; engine = :classical)),
        () -> vcat(adfmax_cdf([-3.0, -2.4, -1.5], cas, c; engine = :general), adfmax_pdf([-3.0, -2.4, -1.5], cas, c; engine = :general)); auto = au)
end
row("ltratio_region, line a = -3.5, b = -0.2639 (DK, c = -5)", () -> ltratio_region((type = "line", a = -3.5, b = -0.2639), "DK", -5.0; engine = :classical), () -> ltratio_region((type = "line", a = -3.5, b = -0.2639), "DK", -5.0; engine = :general))
row("ltratio_region, parabola x = -1.94087 (DK, c = -7)", () -> ltratio_region((type = "parabola", x = -1.94087), "DK", -7.0; engine = :classical), () -> ltratio_region((type = "parabola", x = -1.94087), "DK", -7.0; engine = :general))
close(io)
