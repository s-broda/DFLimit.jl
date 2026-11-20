# Reference values at large negative c for the tests of the three engines (test_dfexact.jl, quickcheck.jl, c/test, the
# MATLAB ltratio_test): the line integral (method = :single) in cases DK, DU, DT, SK, SU, ST at c = -60, -100, -200, -480
# with c0 = 0 and c0 = c, at x = mean + z sd, z = -5, -2, 0, 2, 5 (mean and sd from ltratio_mom); the moments; and the 1%,
# 5% and 50% quantiles of cases DU and SU at c = -100 and -480.  Written to test/engine/largec_ref.txt, one value per line:
#   val  case c c0 z x F f       (the distribution function and the density at x)
#   mom  case c c0 m1 m2 m3 m4
#   icdf case c c0 p q
# On the way the values are checked: node doubling (NODEFAC = 2, the Filon panels split in two, the step of the trapezoidal
# rule halved, the fixed layouts with twice the nodes per panel), a second line Re t = 1/2
# for the zero start at c0 = 0 (the default is 1), and for each (case, c, c0) the density integrated by 96-point
# Gauss-Legendre over [mean - 12 sd, mean + 16 sd] against one and the four moments.  The largest changes are printed.
# Run: julia -t auto --project=dev dev/largec_ref.jl   (about an hour on 6 threads)
using Printf
using DFLimit; const DFExact = DFLimit.DFExact; const D = DFExact
const CS = (-60.0, -100.0, -200.0, -480.0)
const ZS = (-5.0, -2.0, 0.0, 2.0, 5.0)
out = open(joinpath(@__DIR__, "..", "test", "engine", "largec_ref.txt"), "w")
wdF = wdf = wtF = wtf = wmass = wmom = 0.0
for cas in ("DK", "DU", "DT", "SK", "SU", "ST"), c in CS, c0 in (0.0, c)
    m = ltratio_mom(4, cas, c; c0); mu = m[1]; sd = sqrt(m[2] - mu^2)
    @printf(out, "mom %s %g %g %.17g %.17g %.17g %.17g\n", cas, c, c0, m...)
    for z in ZS
        x = mu + z * sd
        F = ltratio_cdf(x, cas, c; c0, method = :single); f = ltratio_pdf(x, cas, c; c0, method = :single)
        @printf(out, "val %s %g %g %g %.17g %.17g %.17g\n", cas, c, c0, z, x, F, f)
        D.NODEFAC[] = 2; D.NPANK[] = 96; D.NPANR[] = 64
        F2 = ltratio_cdf(x, cas, c; c0, method = :single, pfac = 2, hu = 0.0375); f2 = ltratio_pdf(x, cas, c; c0, method = :single, pfac = 2, hu = 0.0375)
        D.NODEFAC[] = 1; D.NPANK[] = 48; D.NPANR[] = 32
        global wdF = max(wdF, abs(F2 - F)); global wdf = max(wdf, abs(f2 - f))
        if c0 == 0 && cas[1] == 'D'
            Ft = ltratio_cdf(x, cas, c; method = :single, tau = 0.5); ft = ltratio_pdf(x, cas, c; method = :single, tau = 0.5)
            global wtF = max(wtF, abs(Ft - F)); global wtf = max(wtf, abs(ft - f))
        end
    end
    flush(out)
    u, w = D.rule(96); a, b = mu - 12sd, mu + 16sd
    xs = a .+ (b - a) .* u; ws = (b - a) .* w
    fx = ltratio_pdf(xs, cas, c; c0, method = :single)
    M = [sum(ws .* fx .* xs .^ k) for k in 0:4]
    global wmass = max(wmass, abs(M[1] - 1)); global wmom = max(wmom, maximum(abs(M[k+1] - m[k]) / abs(m[k]) for k in 1:4))
    @printf("%s c=%g c0=%g: mass - 1 = %.1e, moments (rel.) %s\n", cas, c, c0, M[1] - 1, join([@sprintf("%.1e", (M[k+1] - m[k]) / abs(m[k])) for k in 1:4], " "))
    flush(stdout)
end
for cas in ("DU", "SU"), c in (-100.0, -480.0), p in (0.01, 0.05, 0.5)
    q = ltratio_icdf(p, cas, c; method = :single)
    @printf(out, "icdf %s %g 0 %g %.17g\n", cas, c, p, q)
end
close(out)
@printf("node doubling: |dF| %.1e, |df| %.1e; line Re t = 1/2: |dF| %.1e, |df| %.1e; density: |mass - 1| %.1e, moments %.1e (relative)\n",
        wdF, wdf, wtF, wtf, wmass, wmom)
