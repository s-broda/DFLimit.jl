# check_contour.jl: randomized comparison of the contour integral representation (method = :contour, `tr_contour`) with
# the line integral (:single, the noncentral kernel) and the conditional integral (:double) over the region of a fixed
# start in cases DU, DT (fixed_ok): c in [-100, 2], g0 = xi/sqrt(-2c) with xi in [0.2, 5] for c < -0.5 (|g0| in [0.1, 1.5]
# in case DU, [0.1, 3] in case DT otherwise), c0 = 0, c0 = c or c0 in [-5, 3], x = mean + sd z with z ~ N(0, 2^2), x = 0,
# x near the double root -sqrt(2 c0) (c0 > 0) and the tails mean -/+ 6 sd; density and distribution function.
#
#   julia -t 8 --project=dev dev/check_contour.jl [npoints] [seed] [out.csv]
#
# Each row: the three values, the contour integral's error estimate and status, the conditional integral's status, and
# the times; the line integral (1-40 s a value with a fixed start) only where the two others differ by more than 1e-10 or
# the conditional integral flags the point (NaN otherwise; run_check(...; alllines = true) computes it everywhere).  Summarize with the function `summarize` below (julia --project=dev -e 'include("dev/check_contour.jl"); summarize("out.csv")').
using DFLimit; const DFExact = DFLimit.DFExact
using Printf, Random

function draw_point(rng)
    while true
        model = rand(rng, (2, 3))
        u = rand(rng)
        c = u < 0.65 ? 2 - 32rand(rng) : u < 0.85 ? -30 - 70rand(rng) : 1 + rand(rng)
        if c < -0.5
            g0 = (0.2 + 4.8rand(rng)) / sqrt(-2c)
        else
            g0 = (0.1 + (model == 2 ? 1.4 : 2.9) * rand(rng)) * (rand(rng, Bool) ? 1 : -1)
        end
        DFExact.fixed_ok(model, c, g0) || continue
        v = rand(rng)
        c0 = v < 0.3 ? 0.0 : v < 0.6 ? c : -5 + 8rand(rng)
        return model, c, g0, c0
    end
end
function draw_x(rng, model, c, g0, c0)
    m = DFExact.ltratio_mom(2, ("DK", "DU", "DT")[model], c; c0, g0)
    mu = m[1]; sd = sqrt(max(m[2] - m[1]^2, 1e-8))
    r = rand(rng)
    r < 0.05 && return 0.0
    (r < 0.12 && c0 > 0) && return -sqrt(2c0) * (1 + (rand(rng) - 0.5) * 2e-3)
    r < 0.17 && return mu - 6sd
    r < 0.22 && return mu + 6sd
    mu + 2sd * randn(rng)
end
function run_check(n::Int, seed::Int, out::String; alllines::Bool = false)
    rng = MersenneTwister(seed)
    io = open(out, "w")
    println(io, "case,c,g0,c0,x,kind,contour,cerr,cstatus,line,double,dstatus,t_contour,t_line,t_double")
    DFExact.tr_contour([-2.0], 2, -5.0; g0 = 1.0)                         # (compilation)
    for i in 1:n
        model, c, g0, c0 = draw_point(rng)
        x = draw_x(rng, model, c, g0, c0)
        for kind in (1, 2)
            tc = @elapsed (Fc, ic) = DFExact.tr_contour([x], model, c; c0, g0, pdf = kind == 1, warn = false)
            td = @elapsed (Fd, id) = try DFExact.tr_limdist([x], model, c; c0, g0, pdf = kind == 1, warn = false) catch; ([NaN], (status = [-1],)) end
            # the line integral (slow: 1-40 s with a fixed start) only where the two others disagree or the conditional flags
            tl = 0.0; Fl = NaN
            if !(abs(Fc[1] - Fd[1]) <= 1e-10) || id.status[1] != 0 || alllines
                tl = @elapsed Fl = try kind == 1 ? DFExact.density(x, model; c, c0, g0) : DFExact.cdf(x, model; c, c0, g0) catch; NaN end
            end
            @printf(io, "%s,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.3e,%d,%.17g,%.17g,%d,%.4f,%.4f,%.4f\n", ("DK", "DU", "DT")[model], c, g0, c0, x, kind,
                    Fc[1], ic.err[1], ic.status[1], Fl, Fd[1], id.status[1], tc, tl, td)
            flush(io)
        end
    end
    close(io)
end

"""error statistics of a check file: the contour integral against the line integral, against the conditional integral,
and against the nearer of the two; the disagreements above tol with both"""
function summarize(file::String; tol::Float64 = 1e-9)
    R = [split(l, ",") for l in readlines(file)[2:end]]
    P(r, k) = parse(Float64, r[k])
    el = Float64[]; ed = Float64[]; en = Float64[]; bad = String[]; nflag = 0
    for r in R
        c = P(r, 7); l = P(r, 10); d = P(r, 11)
        a = abs(c - l); b = abs(c - d)
        isfinite(a) && push!(el, a); isfinite(b) && push!(ed, b)
        m = min(isfinite(a) ? a : Inf, isfinite(b) ? b : Inf); push!(en, m)
        r[9] != "0" && (nflag += 1)
        m > tol && push!(bad, join(r[1:9], ",") * @sprintf("  -line %.1e  -double %.1e (double status %s)", c - l, c - d, r[12]))
    end
    q(v, p) = (s = sort(v); s[clamp(ceil(Int, p * length(s)), 1, length(s))])
    @printf("%d values; contour status != 0 at %d\n", length(R), nflag)
    for (lab, v) in (("vs line", el), ("vs conditional", ed), ("vs the nearer", en))
        @printf("  %-15s max %.2e  99%% %.2e  median %.2e\n", lab, maximum(v), q(v, 0.99), q(v, 0.5))
    end
    println("disagreements above $tol with both references: $(length(bad))")
    foreach(println, bad)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
    seed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    out = length(ARGS) >= 3 ? ARGS[3] : "check_contour_$(seed).csv"
    run_check(n, seed, out)
    summarize(out)
end
