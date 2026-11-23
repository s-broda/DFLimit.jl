# DFLimit.jl

[![CI](https://github.com/s-broda/DFLimit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/s-broda/DFLimit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/s-broda/DFLimit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/s-broda/DFLimit.jl)

The exact limiting distribution of the Dickey-Fuller t-ratio in Julia.

`DFLimit` computes the distribution function, density, quantiles and moments of the limits of the Dickey-Fuller
t-ratios under local alternatives, phi = 1 + c/n, by the exact representations of

> Bao, Y., Broda, S. A. and Kan, R., "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio and Some Related
> Statistics", working paper, 2026, available at <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7725619>.

for the regression without deterministic terms, with an intercept and with an intercept and a trend; with a zero,
a fixed or a stationary initial observation; for the t-ratio centred at the unit root or at a local null value c0;
and for the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996). It also has the ADFmax test of
Leybourne (1995), the coefficient statistic, the probabilities of regions {U <= g(V)} of the limit experiment
(point-optimal and weighted-average-power tests), the distribution of the linear form U - aV and the asymptotic power
envelope. A second engine, that of the companion paper on general deterministic terms (in preparation), computes the
same quantities for any deterministic terms: polynomial trends of any order, breaks in level and slope at known dates
(Perron's models), regressors given by functions, the statistic that detrends first and GLS detrending, and it has the
point-optimal statistic, the power envelope and the LM statistic for any design ("Two engines" below). The numbers of
the classical cases come from the multithreaded Julia engine of the paper, `DFExact.jl`, whose home is this package
(`src/`); the same functions, with the same names and arguments, exist in MATLAB and, all but `ltratio_lincdf`,
`power_envelope` and the boundary type `"fun"` of `ltratio_region`, in R.

| function | computes |
|---|---|
| `ltratio_cdf(x, cas, c = 0; c0, g0, method)` | distribution function P[t <= x] of the limit |
| `ltratio_pdf(x, cas, c = 0; c0, g0, method)` | its density |
| `ltratio_icdf(p, cas, c = 0; c0, g0, method, tol)` | quantiles (asymptotic critical values) |
| `ltratio_mom(p, cas, c = 0; c0, g0)` | moments E[t^j], j = 1..p |
| `coef_cdf(y, cas, c = 0; c0, ...)`, `coef_icdf(p, cas, c = 0; tol, ...)` | the coefficient statistic n(phi^ - phi0) |
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
the paper), `:double` or `:conditional` (the conditional integral representation) or `:contour` (the contour integral
representation: the inversion formula on paths around the singularities of the moment generating function, with the
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

The package holds two engines. The classical engine, `DFLimit.DFExact` (`src/DFExact.jl`), is that of the paper above:
the line, conditional and contour integral representations for the three classical cases and case DG. The general
engine, `DFLimit.DFDet` (`src/DFDet.jl`, the file of the repository df-deterministics, with its outer paths of the
classical cases at the unit root in `data/paths`), computes the same limits for any deterministic terms (piecewise
polynomial or smooth functions of r = t/n with known break dates, the statistic that detrends first, GLS detrending)
by the inversion formula on paths around the singularities of the joint moment generating function. They stay two
submodules, each its own file as at its home (the additions for the paper on general deterministic terms are a file of
their own, `src/DFDet_paper2.jl`, evaluated in the module `DFDet`): the engines were written separately and share many
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
functions', are `DFLimit.DFDet.coef_cdf` and so on. The statistics of the paper on general deterministic terms are
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
| fixed start, case DK, c <= 1, (c g0)^2/2 <= 400 or g0 <= 0.5 with c >= -100 | general | 74 to 277 | 39 to 164 | 2.2e-11 | 1.4 to 3.2 times faster |
| fixed start, case DK otherwise, cases DU, DT | classical | 87 to 331 | 82 to 755 | 2.5e-11 | up to 2.3 times faster (the general engine at most 1.1 times), and its warnings of the brittle regimes |
| moments, zero or fixed start, c0 = 0 (c from -480 to 0) | general | 14 to 346 | 8.4 to 15 | 4.2e-14 relative | 1.2 to 28 times faster |
| moments, stationary start: case SK at c <= -3e-3, cases SU (c0 = 0 only), ST at c <= -1 | general | 42 to 282 | 8.7 to 222 | 1.6e-9 relative (c = c0 = -300) | 1.3 to 30 times faster; far from the unit root with c0 = c neither engine is converged (both move by 1e-8 to 5e-7 relative under finer rules) |
| moments, stationary start: case SU with c0 != 0, and nearer zero | classical | 45 | 115 | 2e-12 relative | 2.6 times faster in case SU; next to zero the general engine's moments lose accuracy in cases SU, ST (1e-10 at c = -0.1 to 4e-7 at c = -3e-3, where they fail) |
| moments, zero start, c0 != 0 | classical in cases DK, DU, general in case DT | 17 to 338 | 46 to 241 | 1.8e-10 relative | 1.6 to 2.8 times faster in cases DK, DU; 1.4 times faster in case DT |
| quantiles at c0 = 0: a fixed start, cases DT (c <= 0), ST (c <= -1), DU at c = 0 | general | 246 to 650 | 107 to 272 | 9.6e-11 | 1.8 to 4.7 times faster (one path, F and f in one pass) |
| quantiles otherwise | classical | 37 to 254 | 66 to 346 | 2.8e-11 | equal to 9.4 times faster (case DT at c = 3: the general engine 1.2 times faster); each evaluation of the search routed |
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

## Example

```julia
julia> using DFLimit

julia> ltratio_icdf([0.01, 0.05, 0.10], "DU")         # asymptotic critical values of tau_mu
3-element Vector{Float64}:
 -3.4303530006055194
 -2.861568863651965
 -2.566791069677309

julia> ltratio_cdf(-2.86157, "DU", -5)                # limiting power of the 5% test at c = -5
0.11651356209338504

julia> ltratio_cdf(-2.86157, "SU", -5)                # the same with a stationary start
0.12496412271269275

julia> ltratio_cdf(0, "DK", 2; c0 = 2)                # t-ratio centred at the true value
0.6504355619776843

julia> ltratio_cdf(-2, "DU", -5; g0 = 1.5)            # fixed start
0.8649772400771694

julia> ltratio_pdf(-2, "DT", -5)                      # density
0.5088008223323365

julia> ltratio_mom(4, "DT")                           # first four moments of tau_tau
4-element Vector{Float64}:
  -2.1813558185984525
   5.320675920705032
 -14.035800103551088
  39.55443049607968

julia> coef_cdf(-14.0936, "DU", -10)                  # power of the 5% coefficient test at c = -10
0.45925256320730923

julia> adfmax_icdf(0.05, "DU")                        # 5% critical value of ADFmax with a constant
-2.435416499432509

julia> ltratio_region((type = "wap", k = 3.6658, omega = 10), "DK")   # size of the WAP test of the paper
0.049999622834651807

julia> power_envelope(-7)                             # the power envelope at c = -7
0.49940198260314117
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
`Pkg.develop(path = "/path/to/DFLimit.jl")`. The dependencies (SpecialFunctions, QuadGK, PrecompileTools) are
installed with it. The first `using DFLimit` precompiles the package, which takes about half a minute: it runs a few of
the functions once, so that their first call in a session does not wait several seconds for compilation.
`Pkg.test("DFLimit")` runs the tests (about six minutes; they compare the values with values stored by the paper's
code, and end with the engine's own checks in `test/engine/`). The longer checks of the engine and the scripts that
write its stored values are in `dev/` (see `dev/README.md`).

## Other implementations

The same functions exist as MATLAB programs in the package dflimit-matlab, which also holds the C engine, a port of
this engine checked against its values, with a MEX gateway for MATLAB. The R package
[`dflimit`](https://github.com/s-broda/dflimit) and the Python package dflimit-py are built on that C engine. The paper's
repository has the scripts that reproduce the tables and figures of the paper (`julia/replication/`, on this package).

## Licence and citation

MIT (see `LICENSE`). Please cite the paper when using the package.

The engine in `src/DFExact.jl` and `src/df_series.jl` was developed in the directory `julia/` of the paper's repository up
to DF-Limit commit b484d4f; this package is its home since.
