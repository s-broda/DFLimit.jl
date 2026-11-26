# DFLimit.jl

[![CI](https://github.com/s-broda/DFLimit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/s-broda/DFLimit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/s-broda/DFLimit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/s-broda/DFLimit.jl)

The exact limiting distribution of the Dickey-Fuller t-ratio in Julia.

`DFLimit` computes the distribution function, density, quantiles and moments of the limits of the Dickey-Fuller
t-ratios under local alternatives, phi = 1 + c/n, by the exact representations of two papers. The lead paper,

> Bao, Y., Broda, S. A. and Kan, R., "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio with General
> Deterministic Terms", working paper, 2026, available at <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7745921>,

gives the inversion formula on paths around the singularities of the joint moment generating function and with it the
limit for any piecewise smooth deterministic terms; its companion,

> Bao, Y., Broda, S. A. and Kan, R., "Low-Rank Representations of the Classical Dickey-Fuller Limits, with an Extension
> to Predictive Regressions", working paper, 2026, available at <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7725619>,

gives the representations of the classical cases, in which the numerator has rank one or two (the line integral and the
conditional integral), and the probabilities of regions of the plane of the numerator and the denominator.

The package covers the regression without deterministic terms, with an intercept and with an intercept and a trend;
a zero, a fixed or a stationary initial observation; the t-ratio centred at the unit root or at a local null value c0;
and the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996). It also has the ADFmax test of
Leybourne (1995), the coefficient statistic, the probabilities of regions {U <= g(V)} of the limit experiment
(point-optimal and weighted-average-power tests), the distribution of the linear form U - aV and the asymptotic power
envelope. A second engine, that of the lead paper, computes the same quantities for any deterministic terms: polynomial
trends of any order, breaks in level and slope at known dates (Perron's models), regressors given by functions, the
statistic that detrends first and GLS detrending, and it has the point-optimal statistic, the power envelope and the LM
statistic for any design ("Two engines" below). The classical engine is the multithreaded Julia engine of the companion
paper, `DFExact.jl`, whose home is this package (`src/`); the same functions, with the same names and arguments, exist
in MATLAB and, all but the boundary type `"fun"` of `ltratio_region`, in R and Python.

| function | computes |
|---|---|
| `ltratio_cdf(x, cas, c = 0; c0, g0, method)` | distribution function P[t <= x] of the limit |
| `ltratio_pdf(x, cas, c = 0; c0, g0, method)` | its density |
| `ltratio_icdf(p, cas, c = 0; c0, g0, method, tol)` | quantiles (asymptotic critical values) |
| `ltratio_mom(p, cas, c = 0; c0, g0)` | moments E[t^j], j = 1..p |
| `coef_cdf(y, cas, c = 0; c0, ...)`, `coef_icdf(p, cas, c = 0; tol, ...)` | the coefficient statistic n(phi^ - phi0) |
| `coef_pdf(y, cas, c = 0; c0, g0)`, `coef_mom(p, cas, c = 0; c0, g0)` | its density and moments |
| `adfmax_cdf(x, cas, c = 0)`, `adfmax_pdf`, `adfmax_icdf(p, cas, c = 0; tol, x0)` | the ADFmax statistic (cases DU, DT, SU, ST) |
| `ltratio_region(bnd, cas = "DK", c = 0)` | P{U <= g(V)} for a line, the parabola of the t-ratio, the WAP boundary or a function g |
| `ltratio_lincdf(b, a, cas, c = 0; cbar, N, umax)` | the distribution of U - aV (a = NaN: of V), case DK or DG |
| `power_envelope(c, cas = "DK"; alpha)` | the asymptotic power envelope of Elliott, Rothenberg and Stock (1996), case DK (= DU) or DG |
| `ltratio_bisect(f, lo, hi, tol = 1e-9)` | bisection, as used by `power_envelope` |

The case `cas` is a two-letter code: `D` (fixed start, `g0`, default 0) or `S` (stationary start, c <= -1e-4),
followed by `K` (no deterministic terms, tau), `U` (intercept, tau_mu) or `T` (intercept and trend, tau_tau);
`"DG"` is the GLS-detrended trend case (DF-GLS, cbar = -13.5); in place of the case a design of the general engine
may be given (see "Two engines" below). `method` is `:auto` (at each point the fastest
representation that one evaluation gives to well below 1e-5), `:single` or `:line` (the line integral representation of
the companion paper), `:double` or `:conditional` (the conditional integral representation, also of the companion
paper) or `:contour` (the contour integral representation: the inversion formula on paths of the lead paper, around
the singularities of the moment generating function, with the
integral over v in closed form and the paths in s and t deformed, a double integral with an explicit integrand; a fixed
start in cases DU and DT only, and there the choice of `:auto`). By default (`verified = false`) every value is one
evaluation by one method, with a warning, once per call, where the parameters lie in a regime in which one evaluation is
known to be brittle; `verified = true` gives the checked computation (error estimates and fallbacks, at 1 to 4 times
the cost of the default, about 1.8 with the contour integral, more in the tails below 1e-3, which it recomputes). `x`, `y` and the probabilities `p` may be arrays. The boundary `bnd` of `ltratio_region` is a NamedTuple (or a
Dict): `(type = "line", a, b)`, `(type = "parabola", x, c0)`, `(type = "wap", k, omega)` or `(type = "fun", g)` with
`g` a function of a scalar, optionally with `v0` and `brks`.

The functions return the values only (the MATLAB functions also return an `info` structure); every one has a
docstring (`?ltratio_cdf`). Below c = -550, and with a fixed start g0 outside the region in which the programs are
checked, they raise an `ArgumentError` that says so. The engine's functions that are not exported are reached through
the module `DFLimit.DFExact`, among them the representations, which take the model number 1, 2, 3, 4 for K, U, T,
G: the line integral `DFLimit.DFExact.cdf(q, model; c, c0, g0, start = :zero)` (`start = :stationary` for a
stationary start), `DFLimit.DFExact.density` and `DFLimit.DFExact.moments(model; c, c0, g0, start)`, the
conditional integral `DFLimit.DFExact.tr_limdist(x, model, c; c0, g0, start, pdf)` and the contour integral
`DFLimit.DFExact.tr_contour(x, model, c; c0, g0, pdf)` (the comments of `src/DFExact.jl` describe them, and
`dev/README.md` the range of c in which the rules are set).

## Two engines

The package holds two engines. The classical engine, `DFLimit.DFExact` (`src/DFExact.jl`), is that of the companion
paper: the line and conditional integral representations for the three classical cases and case DG, and for a fixed
start in cases DU, DT the contour integral (the inversion on paths of the lead paper, the integral over v in closed
form). The
general engine, `DFLimit.DFDet` (`src/DFDet.jl`, the file of the repository df-deterministics, with its outer paths of
the classical cases at the unit root in `data/paths`), is that of the lead paper: it computes the same limits for any
deterministic terms (piecewise polynomial or smooth functions of r = t/n with known break dates, the statistic that
detrends first, GLS detrending) by the inversion formula on paths around the singularities of the joint moment
generating function. They stay two submodules, each its own file as at its home (the statistics of the lead paper
beyond the t-ratio are a file of their own, `src/DFDet_paper2.jl`, evaluated in the module `DFDet`): the engines were
written separately and share many
internal names, and kept apart each can be brought up to date by copying its file. The one file that knows both is
`src/merged.jl`: the interface, the router that the classical user-level functions consult after their own checks of
the arguments (`DFExact.ROUTE`), and the rule `DFLimit.autoengine`.

### One interface

| call | computes |
|---|---|
| `ltratio_cdf(x, D, c = 0; c0, g0, start, gls, method, engine)`, `ltratio_pdf` | the t-ratio of any design D |
| `ltratio_icdf(p, D, c = 0; ...)`, `ltratio_mom(p, D, c = 0; ...)` | its quantiles and moments |
| `coef_cdf(y, D, c = 0; ...)`, `coef_icdf(p, D, c = 0; ...)` | the coefficient statistic |
| `adfmax_cdf(x, D, c = 0; start)`, `adfmax_pdf`, `adfmax_icdf` | ADFmax |
| `ltratio_lincdf(b, a, D, c = 0; gls)` | P(U - aV <= b) (a = NaN: P(V <= b)) |
| `ltratio_region(bnd, D, c = 0)` | P{U <= g(V)}: a line or the parabola of the t-ratio |
| `power_envelope(c, D; alpha)` | the Gaussian power envelope of the design |

The design `D` is `trend(p)` (p = -1: no deterministic terms, 0: an intercept, 1: a linear trend, and any p),
`perron(:A, lambda)`, `perron(:B, lambda)`, `perron(:C, lambda)` (lambda a break date or a vector of them),
`fdesign(f, breaks)` (regressors given by a function), `ltrend(p)` (an orthonormal polynomial trend),
`design_from_matrix(X)` (from the regressor matrix of a regression) or `detrend_first(D)` (the statistic that detrends
first); the start is `g0` (a fixed start J(0) = g0) or `start = :stationary`, and `gls = cbar` GLS-detrends at abar =
1 + cbar/n. A design that is a classical case is that case: `ltratio_cdf(x, trend(0), c)` is `ltratio_cdf(x, "DU",
c)`, `trend(1)` with `start = :stationary` is case ST and with `gls = -13.5` case DG. The case strings keep the classical
engine's checks of the arguments (the range of c, the region of the fixed start), as in 0.5.0; a design beyond them goes
to the general engine. Every call takes the engine of `autoengine` (`engine = :auto`); `engine = :classical` forces the
classical engine (an `ArgumentError` for a design it does not have) and `engine = :general` the general one (without the
classical checks; `method` then names the general engine's representation, `:auto`, `:line` or `:paths`). A `method`
other than `:auto`, `verified = true` and the classical engine's own keywords keep the classical engine for a case
string. Where the general engine computes (a design the classical engine does not have, or `engine = :general`),
`method` is the general engine's own: `:auto`, `:line` (the line integral with the explicit kernel) or `:paths` (the
inversion on paths around the roots), see "Which engine"; for a design that is a classical case `method = :line` is the
classical line integral and `method = :paths` the general engine's paths, so that `ltratio_cdf(x, trend(2), c; method =
:paths)` and `ltratio_icdf(p, trend(0), c; method = :paths)` force the paths and `ltratio_cdf(x, "DU", c; engine =
:general, method = :line)` the general engine's line integral. The general engine is checked from c = -480 to 10, and
above the unit root, for the coefficient statistic, the moments and the t-ratio centred at the true value (c0 = c), up to c = 16 with
deterministic terms (10 with them and a fixed start) and up to c = 20 without them (`DFLimit.DFDet.gen_cmax`): beyond, one
root of its transform is of the
order 2c e^{-2c} and the others are no longer resolved in double precision, and every function of the general engine
raises an `ArgumentError` that names the statistic, c and the bound; the rule takes the classical engine there.

The general engine's own functions are exported as well: `dist_cdf(D, xs, c; c0, start, gamma0, gls, method)`, `dist_pdf`,
`dist_quantile`, `dist_moments`, `coef_pdf`, `coef_quantile`, `coef_cdf_gp`, `lin_cdf`, `lin_pdf`, `lin_quantile`,
`v_cdf`, `v_pdf`, `adfmax_quantile`, `region_prob`, `local_power`, `simulate_tratio`, `simulate_gls`, `simulate_first`
and the designs; its `coef_cdf`, `adfmax_cdf`, `adfmax_pdf` and `power_envelope`, whose names are the classical
functions', are `DFLimit.DFDet.coef_cdf` and so on. The statistics of the lead paper beyond the t-ratio are
functions of the general engine: the point-optimal statistic of Elliott, Rothenberg and Stock and the power envelope
for any design (`pt_cdf(D, qs, cbar, c)`, `pt_pdf`, `pt_quantile(D, p, cbar)`, `pt_power(D, cbar, c)`,
`pt_envelope(D, cs)`, also through `power_envelope(c, D)`; for the intercept and the linear trend it is the envelope of
the classical construction to 1e-8), the LM statistic of Schmidt and Phillips and of Amsler and Lee (`lm_cdf(D, xs, c)`,
`lm_quantile(D, p)`, `lm_power(D, cs)`; 5% value -3.0230 for the linear trend) and the companions of the GLS-detrended
t-ratio, MZ_alpha (`mza_cdf(D, ys, c; gls)`, `mza_quantile`) and MSB (`msb_cdf(D, xs, c; gls)`, `msb_quantile`).

### Which engine

`dev/engine_table.jl` (`dev/engine_table.txt`) times both engines in each cell (4 threads, the warm time of one call,
BenchmarkTools minimum of 2; F and f at x = mean + (-2, -1, 0, 1, 2) sd for the t-ratio) and compares their values; where
they differ by more than 1e-8 it compares each with the other's checked computation (`verified = true`). The rule,
`autoengine` in `src/merged.jl`, takes the faster engine at the same accuracy, and the right one where they disagree.

Within the general engine, `dist_cdf`, `dist_pdf` and `dist_quantile` (and the classical functions with a design, through
them) choose between two representations by the keyword `method`: `:auto` (the default) takes the line integral with
the explicit kernel where it covers the design and c0 = 0, and the inversion on paths around the roots otherwise;
`:line` forces the line integral (an `ArgumentError` with the reason where it does not cover the design, the start or
the options) and `:paths` the inversion on paths. The line integral covers the designs whose numerator has rank one or
two: no deterministic terms, the continuous designs that do not vanish at the origin and whose span is closed under
differentiation (polynomial trends of any order, an intercept with sines and cosines), and `detrend_first` of a
continuous design that does not vanish at the origin, with a zero or a stationary start and least squares detrending
(not GLS detrending, a fixed start or a design with a jump). It is the classical engine's integrator with P(t) and the
two roots supplied by the general engine's transform (`src/DFDet_line.jl`). On warm calls at c0 = 0 it is 3 to 5 times
faster than the paths below the unit root (at least 1.3 times for every covered design, start and c from -50 to 2) and
on par with the classical engine for the intercept and the trend; with c0 != 0 it is 2 to 5 times slower, hence the
rule. A case string or a classical design with `engine = :general` at c0 = 0 therefore takes the general engine's line
integral; with `method = :paths` it takes the paths (as the general column of the table below at c0 = 0, measured before
the line integral was added).

| cell | engine | classical (ms) | general (ms) | largest difference | why |
|---|---|---|---|---|---|
| t-ratio, c0 = 0, zero or stationary start (any c), cases DK, DU, DT, SK, SU, ST, DG | classical | 3.1 to 59 | 21 to 158 | 1.3e-12 | 1.4 to 18 times faster |
| zero start, c0 < 0, -100 <= c <= 0 | classical | 7.7 to 54 | 25 to 103 | 7.4e-10 (c = -100; the classical engine's one evaluation, the general engine within 1.4e-12 of the classical checked value since its rules far from the unit root start at c = -60) | up to 3.3 times faster in case DK; in cases DU, DT the general engine is at most 1.1 times faster |
| zero start, c0 > 0, -100 < c <= 1 | general | 69 to 368 | 23 to 86 | 7.7e-12 | 1.8 to 4.7 times faster |
| the same, case DK, within 1e-3 of the tangency x* = -sqrt(2 c0) above it | classical | | | | the density is infinite at x*; just above it the general engine's density is off by up to 1.5e-3 relative (x* + 1e-9), the classical by 5e-10; below x* the general engine is the right one (the classical density off by 10% at x* - 1e-9 at c = -20, c0 = 0.5) |
| zero start, c0 > 0 otherwise (c <= -100 or c > 1) | classical | 19 to 370 | 59 to 434 | 2.0e-12 | 1.2 to 3.1 times faster |
| zero start, c0 != 0, c < -100 | general | 193 to 502 | 76 to 133 | 1.7e-10 | 2.5 to 3.8 times faster |
| stationary start, c0 != 0, c <= -3e-3 | general | 153 to 8430 | 35 to 116 | 1.6e-10 (case SK at c = -0.1, the classical engine's one evaluation) | 2.7 to 190 times faster (case SK at c = c0 = -0.1: 8.4 s against 45 ms); values below 1e-3 go back to the classical conditional integral, whose error is relative |
| stationary start, c0 != 0, c > -3e-3 | classical | | | 8.1e-10 (c = -1e-3) | next to zero the general engine is off by 1e-10 to 7e-8 in cases SU, ST |
| above the unit root, c0 < 0 (c >= 4) | classical | 43 to 412 | 157 to 658 | 2.1e-13 | 1.2 to 15 times faster; the general engine, within 2.1e-13 since its repair (Filon's rule on the vertical line; before, off by 8.5e-5 in F and 1e-2 in f), is 2.3 times faster in case DK at c = 4, c0 = -0.5 |
| above the unit root, the bulk at c >= 7 | classical | 218 to 916 | 535 to 19775 | 1.4e-14 | 1.6 to 22 times faster |
| fixed start, case DK, c <= 1, (c g0)^2/2 <= 400 or g0 <= 0.5 with c >= -100, except c0 < 0 with c > -1 | general | 74 to 277 | 39 to 164 | 2.2e-11 | 1.4 to 3.2 times faster |
| the same with c0 < 0 and c > -1 | classical | | | | the general engine's one evaluation of before November 2026 was off in the upper tail by up to 2.7e-6 at the 99.99% point and 2.9e-2 beyond (c = 0.5, c0 = -5 and -10, g0 = 1), the classical engine within 1e-12 of its checked computation and the conditional integral |
| fixed start, cases DU, DT, c < -10: c0 > 0 below c = -20 or where \|c g0\| > 20, any c0 below c = -20 where \|c g0\| > 60 | general | | | 1e-10 in F; the density the derivative of F to 1e-10 of the peak | the classical contour integral's one evaluation was off (November 2026): in the density by up to 2.2e-2 of the peak (case DU, c = -60, g0 = 1.5, c0 = 5), in F by 3.9e-5 (c0 = -2); its checked computation in the density by up to 8.4e-6; the general engine 0.23 to 0.9 times the time down to c = -45, up to 5 times below c = -75 |
| fixed start, case DK otherwise, cases DU, DT otherwise | classical | 87 to 331 | 82 to 755 | 2.5e-11 | up to 2.3 times faster (the general engine at most 1.1 times), and its warnings of the brittle regimes |
| moments, zero or fixed start, c0 = 0 (c from -480 to 0) | general | 14 to 346 | 8.4 to 15 | 4.2e-14 relative | 1.2 to 28 times faster |
| moments, stationary start: case SK at c <= -3e-3, cases SU (c0 = 0 only), ST at c <= -1 | general | 42 to 282 | 8.7 to 222 | 1.6e-9 relative (c = c0 = -300) | 1.3 to 30 times faster; far from the unit root with c0 = c neither engine is converged (both move by 1e-8 to 5e-7 relative under finer rules) |
| moments, stationary start: case SU with c0 != 0, and nearer zero | classical | 45 | 115 | 2e-12 relative | 2.6 times faster in case SU; next to zero the general engine's moments lose accuracy in cases SU, ST (1e-10 at c = -0.1 to 4e-7 at c = -3e-3, where they fail) |
| moments, zero start, c0 != 0 | classical in cases DK, DU, general in case DT | 17 to 338 | 46 to 241 | 1.8e-10 relative | 1.6 to 2.8 times faster in cases DK, DU; 1.4 times faster in case DT |
| quantiles at c0 = 0: a fixed start, cases DT (c <= 0), ST (c <= -1), DU at c = 0 | general | 246 to 650 | 107 to 272 | 9.6e-11 | 1.8 to 4.7 times faster (one path, F and f in one pass) |
| quantiles otherwise | classical | 37 to 254 | 66 to 346 | 2.8e-11 | equal to 9.4 times faster (case DT at c = 3: the general engine 1.2 times faster); each evaluation of the search routed, and where the rule gives the point to the general engine, F and f from one pass of its paths (`route_dist2`, November 2026: four quantiles 0.54 to 0.77 times as long, median 0.59, the quantiles bit for bit) |
| the coefficient statistic | as the t-ratio at x = 0 with the null value c0 + y, except with a zero start (cases DK, DU, DT, -100 < c <= 1): c0 + y > 0 classical, c0 + y < 0 in case DK for c > -5 general | 8.5 to 132 | 17 | 3.9e-12 | (case SU at c = -5: 8 times faster by the general engine); one point per call at x = 0: with c0 + y > 0 the classical line integral 2 to 12 times faster, with c0 + y < 0 in case DK for c > -5 the general engine 1.4 to 2.7 times (c from 1 to -80, y at the 1 to 99% quantiles) |
| P(V <= b) (`ltratio_lincdf` with a = NaN), case DK | general | 5.5 to 310 | 0.04 to 0.17 | 4.8e-15 below c = 4; above, the classical inversion is off by 5.9e-10 (c = 4) to 1.8e-7 (c = 7) | 33 to 4400 times faster, and above the unit root the more accurate (the general engine within 4.4e-12 of the closed form of M_V inverted on hyperbolas of its own) |
| P(U - aV <= b), case DK | classical | 5.3 to 18 | 5.0 to 274 | 9.6e-13 | up to 30 times faster (above the unit root 12 to 30 times); the general engine 1.1 times faster at a = 2, c = -5 and at a = 0, c = 0 |
| P(U - aV <= b), case DG, a != 0, cbar <= -2 (cbar <= -1 above the unit root), and every c > 4 | general | 86 to 11265 | 21 to 141 | 7.1e-12 | 1.4 to 80 times faster (at cbar = -7 and -10 below the unit root equal); beyond c = 4 the classical code refuses |
| P(U - aV <= b), case DG otherwise | classical | 86 to 88 | 29 to 101 | 3.2e-11 | the closed form at a = 0 (the general engine, within 7.2e-14 of it, is 1.2 to 3 times faster); at cbar = -1 below the unit root 1.1 to 1.3 times faster |
| the power envelope, case DK at c <= -0.01, case DG at c <= -1 | general | 78 to 1771 | 86 to 692 | 2.1e-9 | the critical value to 1e-10 (the classical engine bisects it to 1e-9; in case DG the general engine is within 1.4e-12 of a bisection to 1e-14); 1.1 times slower to 2.6 times faster |
| the power envelope nearer zero | classical | | | | case DK at c = -1e-4: the general engine is off by 1.1e-9 (the critical value at the edge of the support of U - aV); case DG from c = -0.5 on the general engine is the slower, and at c = -0.01 it exhausts the memory |
| ADFmax, stationary start, c <= -0.1 | general | 110 to 196 | 50 to 58 | 3e-12 | 2.2 to 3.4 times faster; next to zero (c = -1e-4) it is off by 2e-9 |
| ADFmax, zero start | classical | 22, 34 | 50, 52 | 2.6e-12 | 1.6 to 2.3 times faster |
| regions (`ltratio_region`) | classical | 1.1, 1.4 | 3.4, 3.8 | 1e-12 | 2.7 to 3.1 times faster, and the only engine for the WAP boundary and a function g |
| above the range of the general engine (c > 16 with deterministic terms, c > 10 with them and a fixed start, c > 20 without) | classical | | | | the general engine refuses there |
| every design beyond the classical cases or beyond the classical engine's checks | general | | | | the only engine |

Where the classical engine computes, its values are those of 0.5.0 bit for bit; where the rule takes the general engine
for a case string, the values move within the differences above. The borders were swept: the stationary start at c =
-3e-3 (cases SU, ST: the general engine within 4e-11 of the classical at c = -3e-3, off by 8e-10 at c = -1e-3), the
moments next to zero (case SK down to c = -3e-3, cases SU, ST down to c = -1), the power envelope (case DK within 3e-12
of the classical down to c = -0.01) and the tangency of case DK (the window 1e-3 above x*). The tests follow the rule
(`test/runtests.jl`, `test/engine`): one stored value changed, the power envelope of case DK at c = -7, now
0.49940198281562 (it was 0.49940198260314117, the classical engine's bisection of the critical value to 1e-9; the same
construction with the critical value bisected to 1e-14 is within 8e-13 of the new value), and the checks of the
classical engine's own computations (ADFmax with a stationary start, the moments, quantiles and default values of its
test grid far from the unit root, the power envelope) force `engine = :classical`, so that they keep their stored
values. Three defects of the general engine are repaired in `src/DFDet.jl` (see NEWS): the law of V on the vertical
line (off by 2e-3 at c = 5, and with GLS detrending at small |cbar| by up to 1.1e-3 in the upper tail), the law of U at
the edge of its support (4.4e-7), and the parts at the root of the
parabola on the vertical line, which now go to Filon's rule (the linear form was off by up to 4e-3 above the unit root
and by 1e-4 with GLS detrending at small |cbar|, the t-ratio with c0 < 0 by 8.5e-5 and its density by 1e-2). On the
line ADFmax and parabolas with two roots (c0 > 0 with x < 0) keep the Gauss-Legendre rule and its warning.

Above the unit root the t-ratio with c0 != c has its upper tail at x of the order (c - c0) e^c/(2c). There the line
integral's inner rule exceeds the bounds of the engine (from x ~ 700), and in case DK the classical engine takes the
conditional integral with its checks whatever its flag (its one evaluation is off by up to 14 there, and before
November 23, 2026 the default route returned it where the checked value was flagged: values outside [0, 1]); for c > 4 the
three cases take the conditional integral at x > 300 too, where the line integral's cost grows like x^4 (in cases DU, DT
at c = 16 it exhausted the memory at the median), and case DK with c0 > c at x < -20, where the line integral is off by up
to 1.5e-3. Against an independent computation (the law of V given J(1), `scratch/fut/bridge_ref.jl`) the values of case DK
are within 1.3e-8 up to c = 14, those of cases DU, DT within 5e-7 of the large-c law at c = 13 and 14; above c = 14 the
classical engine refuses such points with an `ArgumentError` (the conditional integral is off by 7.7e-5 in case DK at
c = 16 and gives 1e44 in case DT, and the general engine's line exceeds its bounds). The general engine cuts its line to
the oscillation with c0 (it was off by 1.1e-3 at c = 10, c0 = 9). Every distribution function (`ltratio_cdf`, `coef_cdf`,
`dist_cdf`, `DFDet.coef_cdf`) ends with a check of its values: below -1e-8 or above 1 + 1e-8 an error that names the call.

### The rules of one evaluation

One evaluation of the general engine's inversion on paths (`dist_cdf`, `dist_pdf`, `dist_quantile` and, through them,
the classical functions where the rule `autoengine` takes the general engine) takes a thinner rule than before November
2026 where it holds (`DFDet.PATH_RULES`, `DFDet.path_region`; keyword `rule`, :auto by default): the outer hyperbola
with 1.5 times the step and the U-shapes with 16 nodes per panel and 32 on the tails of their rays for 24 and 40 (32 and
48 for 48 and 64 with a fixed start), 0.50 times the time of one evaluation (the median over 564 configurations, 0.30 to
0.87; the thin inner rule on the present hyperbola 0.73). The regions, a rule in (the start, whether the design has
deterministic terms, c, c0) that the C engine and the MATLAB programs can take over:

| region | rule | why |
|---|---|---|
| c > -1, c0 < 0, zero or stationary start | the thin inner rule on the present hyperbola | the thinner outer step lost 1.3e-8 of the peak of the density (case DK at c = 0, c0 = -5) |
| c > -1, c0 < 0, fixed start, deterministic terms | the thin inner rule on the present hyperbola with half its step | the thinner outer step lost up to 3.1e-6 (case DU at c = 2, c0 = -5) |
| c > -1, c0 < 0, fixed start, no deterministic terms | the present inner rule on the present hyperbola, the step min(1/2, 2.5/(g0 \|c0\|)) times its own where it is flatter than 70 degrees (2 to 11 times the cost) | the present step was off by up to 0.86 in the upper tail (c = 0.5, g0 = 1, c0 = -20; 1.8e-3 at c0 = -5), this one within 7e-12 of the classical engine (g0 from 0.5 to 2, c from -0.5 to 1, c0 from -1 to -20) |
| c > -1, c0 = 0, fixed start, no deterministic terms | the thin inner rule on the present hyperbola | the thinner outer step lost 1.4e-8 of the peak of the density at mean + 5.75 sd (c = 0.5) |
| c > 2 with a fixed start, c > 5; c < -60 with c0 other than 0 and c | the present rule | beyond the range of the sweep (at c = 5 with a fixed start the thin rule differed from the present one by 5.6e-6 of the peak of the density, Perron's model C with two breaks; at c = -480, case DK, c0 = 2, by 7.7e-9) |
| elsewhere | the thin rule | |

On the vertical line (above the unit root, where no hyperbola keeps the positive roots right of the origin) the line
keeps its panels and only the inner rule thins. `rule = :present` gives the rule of before, `verified = true` takes it
(and its variants), and `hfac`, `n`, `ntail` override the rule. A sweep of 822 configurations (no deterministic terms,
an intercept, linear and quadratic trends, Perron's model B with one break and model C with two, the linear trend
GLS-detrended at -13.5 and detrending first; zero, fixed (g0 = 1) and stationary starts; c from 2 to -60 and c0 in {0,
2, -2, 5, -5, c}; F and f at the 1e-5 to 1 - 1e-5 points; `scratch/rules/sweep1.jl` of the branch kan-integration) set
the regions, and a dense one checked them (`scratch/rules/dense3.jl`: 736 configurations, x from the mean - 6 sd to the
mean + 6 sd in steps of 0.5 sd, c from 2 to -60 and at 5 and -480; the rule within 7.6e-10 of the present rule in F and
4e-9 of the peak in f wherever it is not the present one or a finer one). The criterion: within max(1e-9, twice the
present rule's own error) in F and max(1e-8 of the largest density, twice its own) in f, against a reference with half
the outer step and twice the inner nodes. A first thin rule with 12 and 24 nodes held at the sweep's points but lost up
to 9e-8 of the peak of the density between them near the unit root (case DK at c = 0.5 near the mode), which 16 and 32
nodes do not. Far from the unit root the reference is not the more accurate: in case DK at c = -60, c0 = 2 the present
rule is within 1.7e-10 of the classical engine's checked computation, the reference 6.7e-10. On the line with c0 > 0 at
x < 0 (two roots of the parabola) the line keeps its Gauss-Legendre rule and its warning, and the present rule is off by
up to 2.7e-5 there (case DK at c = 5, c0 = 2), as is the thin one.

The classical engine's contour integral (a fixed start in cases DU, DT; `DFExact.ctr_rule`, keyword `rule` of
`tr_contour`) takes, for one evaluation, 32 nodes per panel and ray of the U-shapes for 64 and the outer step 1.5 times
as long where they hold: 0.42 times the time near the unit root (median; 0.28 to 0.70), the 32 nodes on the present
step above the unit root (c > 1, 0.59 times), 64 nodes on the longer step with c0 > 0 below c = -2.5 and everywhere
below c = -10, and the present rule where |c g0| > 20 (there the longer step lost up to 2.4e-2 in F, case DT at c =
-30, g0 = 3). Against the checked computation on a dense grid of x (129 configurations: cases DU with g0 0.5 and 1.5,
DT with 1 and 3, c from 2 to -60, c0 in {0, 2, -2, -5, c}) every configuration holds the criterion, within 6.2e-11 in
F. Far from the unit root the present rule's own one evaluation was off: with c0 > 0 in the density by up to 5e-6 of
the peak (case DU at c = -30, g0 = 0.5, c0 = 2), 4.2e-3 (c = -75, c0 = 5) and 2.2e-2 (c = -60, g0 = 1.5, c0 = 5), and
where |c g0| > 60 with any c0 in the distribution function by up to 3.9e-5 (case DU, c = -60, g0 = 1.5, c0 = -2) and
6.3e-6 (c = -100, g0 = 0.9); its checked computation was off in the density by up to 8.4e-6 of the peak. The rule of
the engines (`autoengine`) gives these cells to the general engine: with c0 > 0 below c = -20 (below c = -10 where |c
g0| > 20), and with any c0 below c = -20 where |c g0| > 60. There the general engine's density is the derivative of the
distribution function on which the two engines agree (to 1e-10 of the peak, by Richardson's central difference;
`scratch/rules/probe_ctr_far5.jl`), and it takes 0.23 to 0.9 times the time of the classical one evaluation down to c =
-45, up to 1.8 times at c = -60 and 1.2 to 5 times below c = -75.

The classical engine's conditional integral (one evaluation, `DFExact.cond_rule`) takes 48 nodes on the path of each
inner inversion and of the integral over l for 96 with a stationary start for c <= -2, with a zero start in cases DU, DT
for c >= -5 and in case DK for c <= -13.5 (medians of warm calls, F and f at 13 points: 0.60 times the time with a
stationary start, 0.81 and 0.87 in cases DU, DT with a zero start, 0.86 in case DK, where the conditional integral takes
20 to 50 ms), and 64 with a stationary start for c > -2; a fixed start (measured at one configuration only) and c > 2
keep 96. Against the rule with 192 nodes and the outer tolerance 1e-14 on a dense grid of x (150 configurations: cases
DK, DU, DT with a zero start, c from 2 to -100, and SK, SU, ST, c from -0.5 to -50, c0 in {2, -2, -5, c}) the regions
hold the criterion. Elsewhere 48 nodes lost up to 1e-3 far from the unit root with a zero start in cases DU, DT (1.8e-8
of the peak of the density at c = -13.5), 7e-8 of the peak in case SK at c = -0.5, and 7e-7 with a fixed start (case DU,
c = 2, c0 = -2, g0 = 0.5). The outer tolerance 1e-10 in place of 1e-12 held as well but gained nothing measurable (0.85
against 0.90 with a zero start) and is not taken.

## The devices of Bao and Kan

The half-plane statistics of the six classical cases, the coefficient statistic (`coef_cdf`, `coef_pdf`, `coef_icdf`,
`coef_mom`), the linear form U - aV (`ltratio_lincdf`) and the power envelope (`power_envelope`), and the moments of
the t-ratio (`ltratio_mom`) are computed by default with the numerical devices of Yong Bao and Raymond Kan, "Exact
Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1)
Algorithms", ported to the limits of this package in the module `DFLimit.DFKan` (`src/DFKan*.jl`). The joint
characteristic function of the limit pair comes in closed forms that are free of cancellation on both sides of the
unit root, with the branch of the square root fixed at each point from factors whose arguments are known, and in
Taylor-mode arithmetic (jets), so that its derivatives come out of the same assembly. A statistic that rejects on a
half-plane is then one inversion along a ray, by adaptive quadrature over decades with the tail beyond S = 400/|w| (w
the slope of the dominant phase) completed by repeated integration by parts, the derivatives of the slowly varying
factor read off a jet of order eight. The moments of the t-ratio are one-dimensional integrals over s of the j-th
derivative of the moment generating function along its affine path, which one jet of order j gives at each node of an
adaptive quadrature in log s (`method = :jets`; before, Cauchy integrals on a circle); the quantile searches of the
t-ratio (`ltratio_icdf`, also where the rule gives them to the general engine) start from the normal approximation
with these moments (`DFLimit.mom_start`; before, the engines' circles; `DFLimit.MOM_START[] = false` restores that).
Every value comes with an error estimate: `err = true` returns `(values, errors)`, `tol` (1e-12) is the absolute
tolerance, and a warning names the values whose estimate exceeds it. `method = :engine` is the computation of before
(the engines, by the rule of `autoengine`; the t-ratio at x = 0 for the coefficient statistic), which reproduces those
values, and engine = :classical or :general, the engines' own keywords and case DG take it as well; `method =
:inversion` forces the single inversion; `verified = true` recomputes by the engines' checked computation the values
whose estimate exceeds the tolerance.

| function | the single inversion by default | largest difference from the engines | which is right where they differ | warm time (ms), single inversion against `method = :engine` |
|---|---|---|---|---|
| `coef_cdf`, `coef_icdf` | the six cases with zero, fixed and stationary starts, -550 <= c <= 20 | 2.4e-11 in cases SK, SU, ST; in cases DK, DU, DT 3.1e-10 (c = -550), 1.8e-9 (case DU, c = -1e-3, the 1% point) and 4e-9 at c = 20, where the law lies within 1e-6 of y = c and the rounding of y moves F (1,836 points at nine quantiles; at each the single inversion with the tolerances 1e-10 and 1e-12 agrees to 1e-15) | case DU, c = -60, g0 = 1.5: the engines' one evaluation off by up to 1.5e-6, the single inversion within 4.3e-12 of their checked computation | 0.04 to 0.3 against 0.3 to 59 (30 to 1600 times faster; 0.8 times in the upper tail of case DK, where the line integral has no root); quantiles 1.6 to 2.5 against 55 to 144 |
| `coef_pdf`, `coef_mom` | the same cases (new functions) | the density within 2.1e-13 of the general engine's (`DFDet.coef_pdf`); the first two moments within 1.3e-12 of the integrals of the density | | 0.05 to 1 |
| `ltratio_lincdf` | case DK with a zero start for -100 <= c <= 4, the six cases through their designs (fixed start included) for -100 <= c <= 2, a != 0 | 5.5e-11 against the classical code (2.2e-11 at the tolerance 1e-12), 1e-12 against the general engine | with a fixed start above the unit root the general engine's one evaluation is off by up to 6e-2 (case DK, c = 1, a = -7, the 99% point) and 2.9e-3 (case DU, c = 2): the single inversion agrees with its checked computation and with the plain quadrature without the tail to 1e-14 | 0.08 to 0.5 against 4.8 to 12 (11 to 165 times faster) |
| `ltratio_mom` (jets) | the six cases, any c0 and fixed start, c <= 20 | 1.4e-11 relative near the unit root (8.2e-12 to 1.4e-11 at c = 0, -3, -5), 3.2e-10 against the stored values far from it at c0 = 0, 1e-8 to 1.3e-7 with c0 = c far from it (2e-7 at c = c0 = -480 with a stationary start) | with c0 = c far from the unit root the engines' circles are not converged: they move towards the jets under refinement, and at c = c0 = -60 the integral of x^j times the density is within 1.5e-9 of the jets and 5.2e-9 of the circles; with a fixed start above the unit root the engines' one evaluation is off by up to 0.65 relative (case DK, c = 2, c0 = -2, g0 = 1), and their circles with four times the nodes and a quarter of the step agree with the jets to 3.4e-13; at c = 10, 20 with c0 = c by up to 33 relative (case DT) | 0.3 to 1.3 against 12 to 158 for four moments (16 to 290 times faster) |
| `ltratio_icdf` (the start) | the six cases with zero, fixed and stationary starts, c <= 20 | 2.4e-11 (the search tolerance 1e-10), except case DK with c0 = 2 far from the unit root: 4.3e-10 at c = -20 and 5.5e-9 at c = -60, where the distribution function the search inverts is known to about 1e-9 | | whole calls (four quantiles) 0.27 to 1.0 times as long (median 0.85; case DK at c = 0: 46 to 13.8 ms) |
| `power_envelope` | case DK (DU), c <= -1e-4 | 2.7e-12 against the general engine (critical value to 1e-10) from c = -1e-4 to -500 | the classical bisection of the critical value to 1e-9 is off by up to 1.1e-9 | 1.2 to 13 against 35 to 161 (10 to 40 times faster) |

The general engine has the same inversion for the statistics of a design whose transform it evaluates point by point:
the point-optimal statistic (`pt_cdf`, `pt_pdf`, `pt_quantile`, `pt_power`, `pt_envelope`, and through them
`power_envelope` for designs beyond the classical cases) and `coef_cdf_gp` (the coefficient statistic by Gil-Pelaez, the
independent check of the contour integral), with the branch of the square root of the determinant fixed at each point
from the pivots of the precision matrix, Bao and Kan's lemma on accretive matrices (`DFDet.DetCF`, `src/DFKan_det.jl`),
in place of its continuation along the fixed grid of 60,000 points. `method = :grid` (or the keywords N, umax) gives the
grid of before with its refusals. The point-optimal statistic agrees with the contour integral of the general engine on
its quadratic form (the paths around the roots, an independent evaluation) to 3.3e-12 from c = -30 to 4 (an intercept at
cbar = -7, a linear trend at -13.5, Perron's model B at -17.8, a quadratic trend at -20; the 1 to 99% points), where the
grid is off by up to 3.8e-10 at c = 3 and refuses at c = 4 and 5; the single inversion takes 1.2 to 65 ms a value against
77 to 640 ms for the grid. Above c = 4 it can take seconds a value (11 s for Perron's model B at c = 5), and the grid with
its refusal stays the default there; the single inversion has no limit on q (the grid refused |q| > 1000). The quantile
searches keep the transform values along the ray (5% points 13 to 77 ms against 63 to 478, envelopes 15 to 86 ms against
126 to 876). `coef_cdf_gp` agrees with the contour integral (`DFDet.coef_cdf`) to 9.6e-13 from c = -30 to 5 for polynomial
trends, Perron's model C with one and two breaks and a stationary start (2.5e-11 with GLS detrending, where the grid and the
contour integral differ by 1.6e-10), in 3 to 58 ms a value against 120 to 160 for the grid; above c = 5 (7 s a value at c =
10) and with a fixed start the grid stays.

The same devices fix the branches inside the engines: the classical engine's P(t), M_0(t) and the labels of the roots of
its line and contour integrals (`DFExact.tparts_point`) and the argument of log P(t) in the provider of the general
engine's line integral are determined at each point instead of being continued along the grids, to rounding (5.7e-14) the
values of before, at the same cost.

At a = 0 (the law of U alone, whose transform decays like a power, with the edge of its support where the tail start runs
away) the linear form stays with the engines' closed form; at c = 7 in case DK the classical code differs from the single
inversion by 2e-8 at the 99.9% point with a = 2, which no third computation settled, so the default there is unchanged.
Times are warm BenchmarkTools minima on four threads of a shared Apple M4 (November 2026, `scratch/int` of the branch
kan-integration), against `method = :engine` in the same session.

## Example

```julia
julia> using DFLimit

julia> ltratio_icdf([0.01, 0.05, 0.10], "DU")         # asymptotic critical values of tau_mu
3-element Vector{Float64}:
 -3.430353000605524
 -2.8615688636519616
 -2.566791069677308

julia> ltratio_cdf(-2.86157, "DU", -5)                # limiting power of the 5% test at c = -5
0.11651356209338504

julia> ltratio_cdf(-2.86157, "SU", -5)                # the same with a stationary start
0.12496412271269264

julia> ltratio_cdf(0, "DK", 2; c0 = 2)                # t-ratio centred at the true value
0.6504355619776843

julia> ltratio_cdf(-2, "DU", -5; g0 = 1.5)            # fixed start
0.864977240077059

julia> ltratio_pdf(-2, "DT", -5)                      # density
0.5088008223323366

julia> ltratio_mom(4, "DT")                           # first four moments of tau_tau
4-element Vector{Float64}:
  -2.181355818598303
   5.320675920705155
 -14.035800103555014
  39.55443049617824

julia> coef_cdf(-14.0936, "DU", -10)                  # power of the 5% coefficient test at c = -10
0.45925256320763047

julia> adfmax_icdf(0.05, "DU")                        # 5% critical value of ADFmax with a constant
-2.435416499432509

julia> ltratio_region((type = "wap", k = 3.6658, omega = 10), "DK")   # size of the WAP test of the companion paper
0.049999622834651807

julia> power_envelope(-7)                             # the power envelope at c = -7
0.4994019828155684
```

The engine is multithreaded: start Julia with `julia -t auto` (or set the environment variable
`JULIA_NUM_THREADS=auto`). It runs its loops on `DFLimit.DFExact.NWORKERS[]` tasks, by default the smaller of the
number of threads and the number of cores; set it to change that.

The batched sums of the line integral take the exponentials of all their nodes at once from a vector library where one
is at hand. On macOS that is Apple's Accelerate framework, part of the system, which the package uses without being
asked. On x86-64 Linux and Windows it is IntelVectorMath (MKL), an optional dependency: install it in your environment
(`pkg> add IntelVectorMath`) and load it in the session, `using IntelVectorMath` before or after `using DFLimit`, and
Julia loads the package extension `DFLimitIntelVectorMathExt`, which makes the engine use it. Without either the engine
uses plain loops. `DFLimit.DFExact.vecmath()` says which is in use: `"Accelerate"`, `"IntelVectorMath"` or `"none"`.

## Installation

The package needs Julia 1.10 or later. It is registered in Julia's General registry; install it from the package
prompt (press `]` at the `julia>` prompt) with

```
pkg> add DFLimit
```

or, equivalently, `using Pkg; Pkg.add("DFLimit")`. The development version, which may hold changes that are not yet
released, comes from the GitHub repository: `pkg> add DFLimit#main` (or
`Pkg.add(url = "https://github.com/s-broda/DFLimit.jl")`), or, from a local clone,
`Pkg.develop(path = "/path/to/DFLimit.jl")`. The dependencies (SpecialFunctions, QuadGK, StaticArrays, PrecompileTools
and standard libraries) are installed with it. The first `using DFLimit` precompiles the package, which takes about half
a minute: it runs a few of the functions once, so that their first call in a session does not wait several seconds for
compilation. `Pkg.test("DFLimit")` runs the tests (about five minutes on four threads; they compare the values with
values stored by the companion paper's code, and end with the engine's own checks in `test/engine/`). The longer checks
of the engine and the scripts that write its stored values are in `dev/` (see `dev/README.md`). The documentation is
built with Documenter from `docs/`: `julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'`,
then `julia --project=docs docs/make.jl`, which writes the pages to `docs/build`.

## Other implementations

The same functions exist as MATLAB programs in the package dflimit-matlab, which also holds the C engine, a port of
this engine checked against its values, with a MEX gateway for MATLAB. The R package
[`dflimit`](https://github.com/s-broda/dflimit) and the Python package dflimit-py are built on that C engine. The
companion paper's repository has the scripts that reproduce its tables and figures (`julia/replication/`, on this
package).

## Licence and citation

MIT (see `LICENSE`). Please cite the lead paper when using the package, and the companion paper as well where you use
the representations of the classical cases (`CITATION.bib` has both).

The engine in `src/DFExact.jl` and `src/df_series.jl` was developed in the directory `julia/` of the companion paper's
repository up to DF-Limit commit b484d4f; this package is its home since.
