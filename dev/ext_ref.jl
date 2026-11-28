# Reference values of the three extensions of the limit (src/DFDet_vol.jl, src/DFDet_cov.jl, src/DFDet_seas.jl) for the
# ports (the C engine's tests, in the format of c/test/det_ref.txt of dflimit-matlab) and for the Julia test
# test/extensions/ref.jl: writes test/ref/ext_ref.txt.  Run from the repository root:
#   JLCAP_GB=4 ~/bin/jlcap --project=dev dev/ext_ref.jl
# Each line starts with a tag; values are printed with 17 digits and every value line ends with its tolerance.
#   design id spec...                 a design (trend p | perron M q lam...), as in det_ref.txt
#   vol id nb b_1..b_nb om_1..om_nb+1  a volatility path: the break dates and the standard deviations as given (unscaled)
#   vdval did vid nc c_1..c_nc s t D tol           the volatility transform D(s,t) at complex s, t (re im each)
#   vroots did vid nc c.. t ldre r mu.. tol        log D_V(t) (real part) and the r reciprocal roots mu (any order)
#   vtstar did vid nc c.. tstar tol
#   vval func did vid nc c.. c0 n x.. n v.. tol    func = cdf | pdf (dist_cdf, dist_pdf with vol, one call for the points)
#   vquant did vid nc c.. c0 p x tol               dist_quantile with vol (x0 = -2)
#   cval func did start g0 c c0 rho method n x.. n v.. tol   func = cdf | pdf with rho (start 0 zero, 1 fixed, 2 stationary)
#   cquant did start g0 c c0 rho p x tol           dist_quantile with rho (x0 = -2)
#   sval func did c c0 vid n x.. n v.. tol         func = t3cdf | t3pdf (vid 0: no volatility path)
#   squant func did c p x tol                      func = t3
# (until November 27, 2026 also the tags sdval, the seasonal transform D(s,u,t), and the functions t4cdf, t4pdf, fcdf,
# t4, f of the second t-ratio and the F statistic, which are not part of the release 0.5.0)
#   and the regime variants (GLS detrending, the statistic that detrends first, the stationary start of the first regime), with
#   <var> = did vid nc c_1..c_nc start gls first: start 0 zero | 2 stationary, gls = cbar or NaN (least squares), first 1
#   for detrend_first of the design:
#   rdval <var> s t D tol                          the transform D(s,t) (vol_dval of vol_model(D, V, c; start, gls))
#   rroots <var> t ldre r mu.. tol                 as vroots
#   rtstar <var> tstar tol
#   rval func <var> c0 n x.. n v.. tol             func = cdf | pdf (dist_cdf, dist_pdf with vol and start, gls)
#   rquant <var> c0 p x tol                        dist_quantile (x0 = -2)
#   rcase func case vid nc c.. c0 n x.. n v.. tol  func = cdf | pdf | icdf of the case string (ltratio_cdf, ltratio_pdf,
#                                                   ltratio_icdf with vol; for icdf x holds the probabilities)
#   rwarn <var> c0 x v tol | fragment              dist_cdf at one point, which warns with a message that holds the fragment
#   err func | fragment                            an ArgumentError of func whose message contains the fragment (see the
#                                                   lines for the arguments, in words)
using DFLimit, Printf, Logging, Test
const DG = DFLimit.DFDet
const OUT = joinpath(@__DIR__, "..", "test", "ref", "ext_ref.txt")
io = open(OUT, "w")
fmt(x::Real) = @sprintf("%.17g", x)
fmt(z::Complex) = @sprintf("%.17g %.17g", real(z), imag(z))
line(args...) = println(io, join([a isa AbstractString ? a : a isa Symbol ? String(a) : fmt(a) for a in args], " "))
println(io, "# Reference values of the extensions of DFLimit.jl (dev/ext_ref.jl); see the header of that file for the tags")

# (designs 7 and 8 and volatility paths 8 and 9 for the regime variants: the quadratic trend, and the break in the trend and
# the three regimes and the one break of the Treasury bill rate in the lead paper's application)
const TB = 0.50220264317180618
designs = Dict(1 => DG.trend(-1), 2 => DG.trend(0), 3 => DG.trend(1), 4 => DG.perron(:A, 0.3), 5 => DG.perron(:C, 0.4), 6 => DG.perron(:C, 0.6),
               7 => DG.trend(2), 8 => DG.perron(:B, TB))
specs = Dict(1 => "trend -1", 2 => "trend 0", 3 => "trend 1", 4 => "perron A 1 0.3", 5 => "perron C 1 0.4", 6 => "perron C 1 0.6",
             7 => "trend 2", 8 => "perron B 1 $(fmt(TB))")
for id in 1:8; println(io, "design $id $(specs[id])"); end
vols = Dict(1 => ([0.3], [1.0, 3.0]), 2 => ([0.2, 0.7], [2.0, 1.0, 0.5]), 3 => ([0.1], [3.0, 1.0]), 4 => ([0.5], [1.0, 1.0]),
            5 => ([0.6], [3.0, 1.0]), 6 => ([0.1], [1.0, 5.0]), 7 => ([0.4], [1.0, 1.0]),
            8 => ([TB, 0.55506607929515417], [0.30792719352889125, 1.0998629891034262, 0.22195568485892345]),
            9 => ([TB], [0.30792719352889125, 0.41532671932520926]))
for id in 1:9
    b, om = vols[id]
    line("vol", string(id), string(length(b)), b..., om...)
end
vp(id) = volpath(vols[id]...)
cs(v) = (string(length(v)), v...)

# the volatility transform
for (did, vid, c) in ((2, 1, [0.0, 0.0]), (3, 2, [-3.0, 0.0, -10.0]), (1, 2, [-3.0, 0.0, -10.0]), (5, 5, [-5.0, -5.0]), (4, 4, [0.0, -20.0]))
    VM = DG.vol_model(designs[did], vp(vid), c)
    for (s, t) in ((0.3 + 0.0im, -2.0 + 0.0im), (0.2 + 0.7im, 0.3 - 4.0im), (0.1 + 0.0im, -50.0 + 20.0im))
        d = DG.vol_dval(VM, s, t)
        line("vdval", string(did), string(vid), cs(c)..., s, t, d, 1e-10 * abs(d))
    end
    W = DG.MWork(VM)
    for t in (0.5 + 0.0im, 2.0 - 7.0im, -3.0 + 40.0im)
        _, ld, mu, _ = DG.mgfparts(VM, 0.0, t, W)
        line("vroots", string(did), string(vid), cs(c)..., t, real(ld), string(length(mu)), mu..., 1e-10)
    end
    ts, _ = DG.tstar(VM, minimum(c))
    line("vtstar", string(did), string(vid), cs(c)..., ts, 1e-10 * ts)
end
# distribution functions and densities with vol
xs5 = [-4.5, -3.5, -2.5, -1.5, 0.0]
for (did, vid, c, c0) in ((1, 1, [0.0, 0.0], 0.0), (2, 1, [0.0, 0.0], 0.0), (3, 1, [-5.0, -5.0], 0.0), (2, 3, [0.0, 0.0], 0.0), (2, 6, [0.0, 0.0], 0.0),
                          (3, 2, [-3.0, 0.0, -10.0], 0.0), (2, 4, [0.0, -20.0], 0.0), (5, 5, [-5.0, -5.0], 0.0), (2, 1, [-5.0, -5.0], -2.0), (4, 1, [-10.0, -10.0], 0.0))
    D = designs[did]; V = vp(vid)
    F = dist_cdf(D, xs5, c; vol = V, c0); f = dist_pdf(D, xs5, c; vol = V, c0)
    line("vval", "cdf", string(did), string(vid), cs(c)..., c0, string(length(xs5)), xs5..., string(length(F)), F..., 1e-9)
    line("vval", "pdf", string(did), string(vid), cs(c)..., c0, string(length(xs5)), xs5..., string(length(f)), f..., 1e-9)
end
for (did, vid, c) in ((2, 1, [0.0, 0.0]), (3, 2, [-3.0, 0.0, -10.0]), (2, 6, [0.0, 0.0])), p in (0.05, 0.5)
    line("vquant", string(did), string(vid), cs(c)..., 0.0, p, dist_quantile(designs[did], p, c; vol = vp(vid)), 1e-7)
end

# a normal component
xs = [-4.0, -2.5, -1.0, 0.0, 1.5]
for (did, st, g0, c, c0, rho, m) in ((2, 0, 0.0, 0.0, 0.0, sqrt(0.5), :lines), (1, 0, 0.0, 0.0, 0.0, sqrt(0.3), :lines), (3, 0, 0.0, -5.0, cadf_c0(-5.0, sqrt(0.6), sqrt(0.4)), sqrt(0.6), :lines),
                                     (4, 0, 0.0, 0.0, 0.0, sqrt(0.2), :lines), (2, 2, 0.0, -5.0, 5.0, sqrt(0.5), :lines), (2, 0, 0.0, -5.0, predictive_c0(-5.0, -0.95), -0.95, :lines),
                                     (2, 0, 0.0, 0.0, 0.0, sqrt(0.995), :mixture), (2, 1, 1.0, -5.0, 0.0, sqrt(0.5), :mixture), (1, 0, 0.0, 0.0, 0.0, sqrt(0.5), :mixture),
                                     (3, 0, 0.0, -10.0, -2.0, 0.0, :lines))
    D = designs[did]; start = st == 0 ? :zero : st == 1 ? :fixed : :stationary
    k = (c0, start, gamma0 = g0, rho, method = rho == 0 ? :auto : m)
    F = dist_cdf(D, xs, c; k...); f = dist_pdf(D, xs, c; k...)
    line("cval", "cdf", string(did), string(st), g0, c, c0, rho, m, string(length(xs)), xs..., string(length(F)), F..., 1e-9)
    line("cval", "pdf", string(did), string(st), g0, c, c0, rho, m, string(length(xs)), xs..., string(length(f)), f..., 1e-9)
end
for (did, c, c0, rho, p) in ((2, 0.0, 0.0, sqrt(0.5), 0.05), (3, 0.0, 0.0, sqrt(0.9), 0.01), (2, -5.0, predictive_c0(-5.0, -0.95), -0.95, 0.95))
    line("cquant", string(did), "0", 0.0, c, c0, rho, p, dist_quantile(designs[did], p, c; c0, rho), 1e-7)
end

# the first seasonal t-ratio
for (did, c, c0, vid) in ((1, 0.0, 0.0, 0), (2, 0.0, 0.0, 0), (3, -5.0, 0.0, 0), (4, 0.0, 0.0, 0), (2, -5.0, -2.0, 0), (2, 0.0, 0.0, 1))
    D = designs[did]; k = vid == 0 ? (; c0) : (; c0, vol = vp(vid))
    c_ = vid == 0 ? c : [c, c]
    F = hegy_t3_cdf(D, xs, c_; k...); f = hegy_t3_pdf(D, xs, c_; k...)
    line("sval", "t3cdf", string(did), c, c0, string(vid), string(length(xs)), xs..., string(length(F)), F..., 1e-10)
    line("sval", "t3pdf", string(did), c, c0, string(vid), string(length(xs)), xs..., string(length(f)), f..., 1e-10)
end
for (did, c) in ((2, 0.0), (3, 0.0))
    D = designs[did]
    line("squant", "t3", string(did), c, 0.05, hegy_t3_quantile(D, 0.05, c), 1e-7)
end

# the regime variants: GLS detrending, the statistic that detrends first, the stationary start of the first regime
vmodel(did, vid, c, st, g, fi) = DG.vol_model(fi == 1 ? DG.detrend_first(designs[did]) : designs[did], vp(vid), c;
                                              start = st == 2 ? :stationary : :zero, gls = isnan(g) ? nothing : g)
vkw(st, g) = (; start = st == 2 ? :stationary : :zero, gls = isnan(g) ? nothing : g)
vdes(did, fi) = fi == 1 ? DG.detrend_first(designs[did]) : designs[did]
var(did, vid, c, st, g, fi) = (string(did), string(vid), cs(c)..., string(st), isnan(g) ? "NaN" : g, string(fi))
variants = ((3, 1, [0.0, 0.0], 0, -13.5, 0), (2, 2, [-3.0, 0.0, -10.0], 0, -7.0, 0), (5, 5, [-5.0, -5.0], 0, -13.5, 0),
            (3, 1, [0.0, 0.0], 0, NaN, 1), (7, 2, [-3.0, 0.0, -10.0], 0, NaN, 1), (5, 5, [-5.0, -5.0], 0, NaN, 1),
            (2, 1, [-4.0, -4.0], 2, NaN, 0), (3, 2, [-3.0, 0.0, -10.0], 2, NaN, 0), (1, 1, [-5.0, -5.0], 2, NaN, 0), (4, 4, [-2.0, -20.0], 2, NaN, 0))
for (did, vid, c, st, g, fi) in variants
    VM = vmodel(did, vid, c, st, g, fi)
    for (s, t) in ((0.3 + 0.0im, -2.0 + 0.0im), (0.2 + 0.7im, 0.3 - 4.0im), (0.1 + 0.0im, -50.0 + 20.0im))
        d = DG.vol_dval(VM, s, t)
        line("rdval", var(did, vid, c, st, g, fi)..., s, t, d, 1e-10 * abs(d))
    end
    W = DG.MWork(VM)
    for t in (0.5 + 0.0im, 2.0 - 7.0im, -3.0 + 40.0im)
        _, ld, mu, _ = DG.mgfparts(VM, 0.0, t, W)
        line("rroots", var(did, vid, c, st, g, fi)..., t, real(ld), string(length(mu)), mu..., 1e-10)
    end
    ts, _ = DG.tstar(VM, minimum(c))
    line("rtstar", var(did, vid, c, st, g, fi)..., ts, 1e-10 * ts)
end
# (the last cases: the sizes at the 5% values of a constant variance under one break in it, as in the lead paper's table,
# and the Treasury bill rate of its application: the statistic and the 5% value of a constant variance)
rvals = ((v..., 0.0, xs5) for v in variants)
rvals = (rvals..., (3, 1, [-5.0, -5.0], 0, -13.5, 0, -2.0, xs5), (2, 1, [-4.0, -4.0], 2, NaN, 0, -1.5, xs5),
         (3, 6, [0.0, 0.0], 0, -13.5, 0, 0.0, [-2.8456308261599799]), (3, 3, [0.0, 0.0], 0, -13.5, 0, 0.0, [-2.8456308261599799]),
         (2, 5, [0.0, 0.0], 0, -7.0, 0, 0.0, [-1.9408732523381735]), (2, 1, [0.0, 0.0], 0, -7.0, 0, 0.0, [-1.9408732523381735]),
         (8, 8, [0.0, 0.0, 0.0], 0, NaN, 1, 0.0, [-4.410078431125541, -3.8595374255581674]), (8, 9, [0.0, 0.0], 0, NaN, 1, 0.0, [-4.410078431125541]),
         (8, 8, [0.0, 0.0, 0.0], 0, -17.786784140969164, 0, 0.0, [-4.1094158565465664, -3.4160276971974133]))
for (did, vid, c, st, g, fi, c0, x) in rvals
    D = vdes(did, fi); V = vp(vid)
    F = dist_cdf(D, x, c; vol = V, c0, vkw(st, g)...); f = dist_pdf(D, x, c; vol = V, c0, vkw(st, g)...)
    line("rval", "cdf", var(did, vid, c, st, g, fi)..., c0, string(length(x)), x..., string(length(F)), F..., 1e-9)
    line("rval", "pdf", var(did, vid, c, st, g, fi)..., c0, string(length(x)), x..., string(length(f)), f..., 1e-9)
end
for (did, vid, c, st, g, fi) in (variants[1], variants[2], variants[4], variants[6], variants[7], variants[8]), p in (0.05, 0.5)
    q = dist_quantile(vdes(did, fi), p, c; vol = vp(vid), vkw(st, g)...)
    line("rquant", var(did, vid, c, st, g, fi)..., 0.0, p, q, 1e-7)
end
# the case strings with a volatility path: a stationary start (SU, ST) and GLS detrending (DG)
xs3 = [-3.5, -2.5, -1.0]; ps3 = [0.01, 0.05, 0.5]
for (case, vid, c, c0) in (("SU", 1, [-4.0, -4.0], 0.0), ("ST", 2, [-3.0, 0.0, -10.0], 0.0), ("DG", 1, [0.0, 0.0], 0.0), ("DG", 5, [-5.0, -5.0], -2.0))
    F = ltratio_cdf(xs3, case, c; vol = vp(vid), c0); f = ltratio_pdf(xs3, case, c; vol = vp(vid), c0)
    q = [ltratio_icdf(p, case, c; vol = vp(vid), c0) for p in ps3]
    line("rcase", "cdf", case, string(vid), cs(c)..., c0, "3", xs3..., "3", F..., 1e-9)
    line("rcase", "pdf", case, string(vid), cs(c)..., c0, "3", xs3..., "3", f..., 1e-9)
    line("rcase", "icdf", case, string(vid), cs(c)..., c0, "3", ps3..., "3", q..., 1e-7)
end
# a brittle spot, where the engine warns: GLS detrending with cbar = -2 puts the outer contour on the vertical line
let (did, vid, c, st, g, fi) = (3, 1, [0.0, 0.0], 0, -2.0, 0), x = -2.0, frag = "vertical line"
    logger = Test.TestLogger(min_level = Logging.Warn)
    F = Logging.with_logger(logger) do; dist_cdf(designs[did], [x], c; vol = vp(vid), gls = g)[1]; end
    any(r -> occursin(frag, r.message), logger.logs) || error("ext_ref: no warning with \"$frag\" at the brittle spot")
    println(io, join([a isa AbstractString ? a : fmt(a) for a in ("rwarn", var(did, vid, c, st, g, fi)..., 0.0, x, F, 1e-6)], " "), " | ", frag)
end

# argument errors (the fragment of the message; the arguments in words)
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other"; end
for (fn, words, f, frag) in (
        ("volpath", "a break date at 1.2", () -> volpath([1.2]), "must lie in (0, 1)"),
        ("volpath", "break dates 0.5, 0.3", () -> volpath([0.5, 0.3]), "strictly increasing"),
        ("volpath", "one omega for two regimes", () -> volpath([0.5], [1.0]), "one standard deviation omega per regime"),
        ("volpath", "omega -1", () -> volpath([0.5], [1.0, -1.0]), "finite and positive"),
        ("dist_cdf", "vol, c = 2", () -> dist_cdf(designs[2], [-2.0], 2.0; vol = vp(1)), "must be at most 0"),
        ("dist_cdf", "vol, fixed start gamma0 = 1", () -> dist_cdf(designs[2], [-2.0], -5.0; vol = vp(1), start = :fixed, gamma0 = 1.0), "fixed start (gamma0 != 0) is not computed"),
        ("dist_cdf", "vol, stationary start, c = 0 on the first regime", () -> dist_cdf(designs[2], [-2.0], [0.0, -5.0]; vol = vp(1), start = :stationary), "needs c < 0 on the first regime"),
        ("dist_cdf", "vol, detrend_first and gls = -13.5", () -> dist_cdf(DG.detrend_first(designs[3]), [-2.0], -5.0; vol = vp(1), gls = -13.5), "detrends first already"),
        ("dist_cdf", "vol, gls = -13.5 and a stationary start", () -> dist_cdf(designs[3], [-2.0], -5.0; vol = vp(1), gls = -13.5, start = :stationary), "zero start only"),
        ("dist_cdf", "vol, detrend_first and a stationary start", () -> dist_cdf(DG.detrend_first(designs[3]), [-2.0], -5.0; vol = vp(1), start = :stationary), "zero start only"),
        ("dist_cdf", "vol, three c for two regimes", () -> dist_cdf(designs[2], [-2.0], [0.0, -1.0, 0.0]; vol = vp(1)), "one value per regime"),
        ("dist_cdf", "c a vector without vol", () -> dist_cdf(designs[2], [-2.0], [0.0, -1.0]), "needs a volatility path"),
        ("dist_cdf", "vol and rho = 0.5", () -> dist_cdf(designs[2], [-2.0], -5.0; vol = vp(1), rho = 0.5), "rho and vol together"),
        ("dist_cdf", "rho = 1.5", () -> dist_cdf(designs[2], [-2.0], -5.0; rho = 1.5), "rho must lie in [-1, 1]"),
        ("dist_cdf", "rho = 0.5, gls = -13.5", () -> dist_cdf(designs[3], [-2.0], -5.0; rho = 0.5, gls = -13.5), "GLS detrending"),
        ("dist_cdf", "rho = 0.5, method lines, fixed start", () -> dist_cdf(designs[2], [-2.0], -5.0; rho = 0.5, start = :fixed, gamma0 = 1.0, method = :lines), "zero or a stationary start"),
        ("cadf_c0", "rho = 0", () -> cadf_c0(-5.0, 0.0), "not be 0"),
        ("hegy_t3_cdf", "c = 1", () -> hegy_t3_cdf(designs[2], [1.0], 1.0), "must be at most 0"))
    m = errmsg(f)
    occursin(frag, m) || error("ext_ref: $fn ($words) gave \"$m\", without \"$frag\"")
    println(io, "err $fn $words | $frag")
end
close(io)
println("wrote ", OUT)
