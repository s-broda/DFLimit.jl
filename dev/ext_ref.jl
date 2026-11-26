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
#   sdval did c s u t D tol                        the seasonal transform D(s,u,t) at complex points
#   sval func did c c0 vid n x.. n v.. tol         func = t3cdf | t3pdf | t4cdf | t4pdf | fcdf (vid 0: no volatility path)
#   squant func did c p x tol                      func = t3 | t4 | f
#   err func | fragment                            an ArgumentError of func whose message contains the fragment (see the
#                                                   lines for the arguments, in words)
using DFLimit, Printf, Logging
const DG = DFLimit.DFDet
const OUT = joinpath(@__DIR__, "..", "test", "ref", "ext_ref.txt")
io = open(OUT, "w")
fmt(x::Real) = @sprintf("%.17g", x)
fmt(z::Complex) = @sprintf("%.17g %.17g", real(z), imag(z))
line(args...) = println(io, join([a isa AbstractString ? a : a isa Symbol ? String(a) : fmt(a) for a in args], " "))
println(io, "# Reference values of the extensions of DFLimit.jl (dev/ext_ref.jl); see the header of that file for the tags")

designs = Dict(1 => DG.trend(-1), 2 => DG.trend(0), 3 => DG.trend(1), 4 => DG.perron(:A, 0.3), 5 => DG.perron(:C, 0.4), 6 => DG.perron(:C, 0.6))
specs = Dict(1 => "trend -1", 2 => "trend 0", 3 => "trend 1", 4 => "perron A 1 0.3", 5 => "perron C 1 0.4", 6 => "perron C 1 0.6")
for id in 1:6; println(io, "design $id $(specs[id])"); end
vols = Dict(1 => ([0.3], [1.0, 3.0]), 2 => ([0.2, 0.7], [2.0, 1.0, 0.5]), 3 => ([0.1], [3.0, 1.0]), 4 => ([0.5], [1.0, 1.0]),
            5 => ([0.6], [3.0, 1.0]), 6 => ([0.1], [1.0, 5.0]), 7 => ([0.4], [1.0, 1.0]))
for id in 1:7
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

# the seasonal statistics
for (did, c) in ((1, 0.0), (2, 0.0), (3, -3.0), (5, -2.0))
    SM = DG.SeasModel(DG.prep(designs[did]))
    for (s, u, t) in ((0.3 + 0.0im, 0.0im, 0.5 + 0.0im), (0.2 + 0.0im, 0.4 + 0.0im, 0.3 + 0.0im), (0.2 + 1.0im, 0.3im, 0.4 + 2.0im), (0.0im, 2.0im, 1.0 + 30.0im))
        d = DG.seas_dval(SM, s, u, t; c)
        line("sdval", string(did), c, s, u, t, d, 1e-10 * abs(d))
    end
end
for (did, c, c0, vid) in ((1, 0.0, 0.0, 0), (2, 0.0, 0.0, 0), (3, -5.0, 0.0, 0), (4, 0.0, 0.0, 0), (2, -5.0, -2.0, 0), (2, 0.0, 0.0, 1))
    D = designs[did]; k = vid == 0 ? (; c0) : (; c0, vol = vp(vid))
    c_ = vid == 0 ? c : [c, c]
    F = hegy_t3_cdf(D, xs, c_; k...); f = hegy_t3_pdf(D, xs, c_; k...)
    line("sval", "t3cdf", string(did), c, c0, string(vid), string(length(xs)), xs..., string(length(F)), F..., 1e-10)
    line("sval", "t3pdf", string(did), c, c0, string(vid), string(length(xs)), xs..., string(length(f)), f..., 1e-10)
end
q4 = [0.5, 1.0, 1.645, 2.5]; xF = [1.0, 3.0, 6.0, 10.0]
for (did, c) in ((1, 0.0), (2, 0.0), (3, 0.0), (2, -5.0), (3, -10.0))
    D = designs[did]
    F4 = hegy_t4_cdf(D, q4, c); f4 = hegy_t4_pdf(D, q4, c); FF = hegy_f_cdf(D, xF, c)
    line("sval", "t4cdf", string(did), c, 0.0, "0", string(length(q4)), q4..., string(length(F4)), F4..., 1e-9)
    line("sval", "t4pdf", string(did), c, 0.0, "0", string(length(q4)), q4..., string(length(f4)), f4..., 1e-9)
    line("sval", "fcdf", string(did), c, 0.0, "0", string(length(xF)), xF..., string(length(FF)), FF..., 1e-9)
end
for (did, c) in ((2, 0.0), (3, 0.0))
    D = designs[did]
    line("squant", "t3", string(did), c, 0.05, hegy_t3_quantile(D, 0.05, c), 1e-7)
    line("squant", "t4", string(did), c, 0.975, hegy_t4_quantile(D, 0.975, c), 1e-7)
    line("squant", "f", string(did), c, 0.95, hegy_f_quantile(D, 0.95, c), 1e-7)
end

# argument errors (the fragment of the message; the arguments in words)
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other"; end
for (fn, words, f, frag) in (
        ("volpath", "a break date at 1.2", () -> volpath([1.2]), "must lie in (0, 1)"),
        ("volpath", "break dates 0.5, 0.3", () -> volpath([0.5, 0.3]), "strictly increasing"),
        ("volpath", "one omega for two regimes", () -> volpath([0.5], [1.0]), "one standard deviation omega per regime"),
        ("volpath", "omega -1", () -> volpath([0.5], [1.0, -1.0]), "finite and positive"),
        ("dist_cdf", "vol, c = 2", () -> dist_cdf(designs[2], [-2.0], 2.0; vol = vp(1)), "must be at most 0"),
        ("dist_cdf", "vol, stationary start", () -> dist_cdf(designs[2], [-2.0], -5.0; vol = vp(1), start = :stationary), "zero start only"),
        ("dist_cdf", "vol, gls = -13.5", () -> dist_cdf(designs[3], [-2.0], -5.0; vol = vp(1), gls = -13.5), "GLS detrending"),
        ("dist_cdf", "vol, three c for two regimes", () -> dist_cdf(designs[2], [-2.0], [0.0, -1.0, 0.0]; vol = vp(1)), "one value per regime"),
        ("dist_cdf", "c a vector without vol", () -> dist_cdf(designs[2], [-2.0], [0.0, -1.0]), "needs a volatility path"),
        ("dist_cdf", "vol and rho = 0.5", () -> dist_cdf(designs[2], [-2.0], -5.0; vol = vp(1), rho = 0.5), "rho and vol together"),
        ("dist_cdf", "rho = 1.5", () -> dist_cdf(designs[2], [-2.0], -5.0; rho = 1.5), "rho must lie in [-1, 1]"),
        ("dist_cdf", "rho = 0.5, gls = -13.5", () -> dist_cdf(designs[3], [-2.0], -5.0; rho = 0.5, gls = -13.5), "GLS detrending"),
        ("dist_cdf", "rho = 0.5, method lines, fixed start", () -> dist_cdf(designs[2], [-2.0], -5.0; rho = 0.5, start = :fixed, gamma0 = 1.0, method = :lines), "zero or a stationary start"),
        ("cadf_c0", "rho = 0", () -> cadf_c0(-5.0, 0.0), "not be 0"),
        ("hegy_t4_cdf", "perron A 0.5", () -> hegy_t4_cdf(designs[4], [1.0]), "polynomial seasonal trends"),
        ("hegy_t3_cdf", "c = 1", () -> hegy_t3_cdf(designs[2], [1.0], 1.0), "must be at most 0"),
        ("hegy_f_cdf", "c = -30", () -> hegy_f_cdf(designs[2], [1.0], -30.0), "must be at least -20"))
    m = errmsg(f)
    occursin(frag, m) || error("ext_ref: $fn ($words) gave \"$m\", without \"$frag\"")
    println(io, "err $fn $words | $frag")
end
close(io)
println("wrote ", OUT)
