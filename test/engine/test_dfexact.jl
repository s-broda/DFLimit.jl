using JSON, Printf
using DFLimit; const DFExact = DFLimit.DFExact

println("Threads: ", Threads.nthreads())
ref = JSON.parsefile(joinpath(@__DIR__, "python_exact_reference.json"))
DFExact.cdf(-1.0, 1); DFExact.density(-1.0, 1); DFExact.cdf(-1.0, 2); DFExact.density(-1.0, 3)   # compile
println("Against the stored reference values (python_exact_reference.json, computed by this code on 2026-10-31):")
worstF = worstf = 0.0
for r in ref
    model = r["model"]; c = Float64(r["c"]); q = Float64(r["q"])
    tF = @elapsed F = DFExact.cdf(q, model; c = c)
    tf = @elapsed f = DFExact.density(q, model; c = c)
    global worstF = max(worstF, abs(F - r["cdf"])); global worstf = max(worstf, abs(f / r["density"] - 1))
    @printf("M%d c=%g q=%6.3f: F = %.10f (reference %.10f, %5.1f ms, reference run %5.1f ms)  f = %.10f (reference %.10f, %5.1f ms, reference run %5.1f ms)\n",
            model, c, q, F, r["cdf"], 1e3tF, 1e3r["t_cdf"], f, r["density"], 1e3tf, 1e3r["t_density"])
end
@printf("worst |cdf diff| = %.1e, worst density rel diff = %.1e\n", worstF, worstf)

isfile(joinpath(@__DIR__, "python_reference.json")) && begin
q64 = JSON.parsefile(joinpath(@__DIR__, "python_reference.json"))
println("\nAt the 64 exact quantiles of df_quantiles.json: cdf(q) - prob")
worst = 0.0; ttot = 0.0
for r in q64
    t0 = time(); F = DFExact.cdf(Float64(r["q"]), r["model"]; c = Float64(r["c"])); global ttot += time() - t0
    global worst = max(worst, abs(F - r["prob"]))
end
@printf("worst |cdf(q) - prob| = %.1e over %d points, %.0f ms per point on average\n", worst, length(q64), 1e3ttot / length(q64))
end

tq = @elapsed qq = DFExact.quantile(0.05, 2)
@printf("\nM2 5%% quantile by Newton: %.9f in %.2f s\n", qq, tq)

# Far from the unit root: the regression points of largec_ref.txt (written by largec_ref.jl, which checks them by node
# doubling, a second line Re t and the moments of the density): the line integral and the automatic method at c = -60,
# -100, -200, -480 in the six cases with c0 = 0 and c0 = c, x = mean + (-5, -2, 0, 2, 5) sd; the moments; the quantiles.
println("\nFar from the unit root (largec_ref.txt):")
LR = [split(l) for l in readlines(joinpath(@__DIR__, "largec_ref.txt"))]
let wS = 0.0, wA = 0.0, wV = 0.0, wm = 0.0, wq = 0.0, nv = 0, ts = 0.0, ta = 0.0
    for p in LR
        p[1] == "val" || continue
        cas = String(p[2]); c, c0, z, x, F, f = parse.(Float64, p[3:8])
        ts += @elapsed (Fs = ltratio_cdf(x, cas, c; c0, method = :single); fs = ltratio_pdf(x, cas, c; c0, method = :single))
        ta += @elapsed (Fa = ltratio_cdf(x, cas, c; c0); fa = ltratio_pdf(x, cas, c; c0))         # (the default: one evaluation)
        Fv = ltratio_cdf(x, cas, c; c0, verified = true); fv = ltratio_pdf(x, cas, c; c0, verified = true)
        wS = max(wS, abs(Fs - F), abs(fs - f)); wA = max(wA, abs(Fa - F), abs(fa - f)); wV = max(wV, abs(Fv - F), abs(fv - f)); nv += 1
    end
    for p in LR
        p[1] == "mom" || continue
        cas = String(p[2]); c, c0 = parse.(Float64, p[3:4]); mr = parse.(Float64, p[5:8])
        m = ltratio_mom(4, cas, c; c0); wm = max(wm, maximum(abs.(m .- mr) ./ abs.(mr)))
    end
    for p in LR
        p[1] == "icdf" || continue
        cas = String(p[2]); c, c0, pr, qr = parse.(Float64, p[3:6])
        q = ltratio_icdf(pr, cas, c; c0); wq = max(wq, abs(q - qr))
    end
    @printf("%d points: single |dF|, |df| %.1e (%.1f ms per pair), auto (one evaluation) %.1e (%.1f ms per pair), auto verified %.1e; moments %.1e (relative); quantiles %.1e\n",
            nv, wS, 1e3ts / nv, wA, 1e3ta / nv, wV, wm, wq)
    (wS < 1e-11 && wA < 2e-9 && wV < 1e-10 && wm < 1e-10 && wq < 1e-9) || error("test_dfexact: the large-|c| points are off")
end

# The contour integral representation (method :contour, the choice of :auto with a fixed start in cases DU, DT; November
# 2026): the stored values of c/test/contour_ref.txt (checked against the line and the conditional integral), each through
# ltratio_cdf/ltratio_pdf with method :auto (the points inside the region of a fixed start, fixed_ok), and the line
# integral at the first six of them
println("\nThe contour integral (c/test/contour_ref.txt):")
CR = [parse.(Float64, split(l)) for l in readlines(joinpath(@__DIR__, "contour_ref.txt"))]
let wa = 0.0, wl = 0.0, w1 = 0.0, ta = 0.0, t1 = 0.0, n = 0
    for (i, r) in enumerate(CR)
        model = Int(r[1]); c, c0, g0, x, F, f = r[2:7]; cas = ("DK", "DU", "DT")[model]
        DFExact.fixed_ok(model, c, g0) || continue                  # (the engine-level points outside the user-level region)
        ta += @elapsed (Fa = ltratio_cdf(x, cas, c; c0, g0, verified = true); fa = ltratio_pdf(x, cas, c; c0, g0, verified = true)); n += 1
        wa = max(wa, abs(Fa - F), abs(fa - f))
        t1 += @elapsed (F1 = ltratio_cdf(x, cas, c; c0, g0); f1 = ltratio_pdf(x, cas, c; c0, g0))       # (the default: one evaluation)
        w1 = max(w1, abs(F1 - F), abs(f1 - f))
        i <= 6 && (wl = max(wl, abs(ltratio_cdf(x, cas, c; c0, g0, method = :single) - F), abs(ltratio_pdf(x, cas, c; c0, g0, method = :single) - f)))
    end
    @printf("%d points: auto, verified |dF|, |df| %.1e (%.1f ms per pair, values below 1e-3 by the conditional integral); the default (one evaluation) %.1e (%.1f ms per pair); the line integral at six of them %.1e\n",
            n, wa, 1e3ta / n, w1, 1e3t1 / n, wl)
    (wa < 1e-10 && w1 < 5e-9 && wl < 1e-10) || error("test_dfexact: the contour integral points are off")
end
