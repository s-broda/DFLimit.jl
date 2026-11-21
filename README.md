# DFLimit.jl

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
| `ltratio_cdf(x, D, c = 0; c0, g0, start, gls, engine)`, `ltratio_pdf` | the t-ratio of any design D |
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
classical checks; `method`, which names a representation of the classical engine, then must be `:auto`). A `method`
other than `:auto`, `verified = true` and the classical engine's own keywords keep the classical engine for a case
string. The general engine is checked from c = -480 to 10.

The general engine's own functions are exported as well: `dist_cdf(D, xs, c; c0, start, gamma0, gls)`, `dist_pdf`,
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

| cell | engine | classical (ms) | general (ms) | largest difference | why |
|---|---|---|---|---|---|
| t-ratio, zero or stationary start, c0 = 0 (any c), cases DK, DU, DT, SK, SU, ST, DG | classical | 3 to 81 | 34 to 255 | 4e-11 | 1.7 to 28 times faster |
| zero start, c0 < 0, -100 <= c <= 0 | classical | 10 to 77 | 40 to 162 | 1.4e-9 (c = -100) | equal to 4 times faster |
| zero start, c0 > 0, -100 < c < 0, cases DU, DT | general | 286 to 547 | 73 to 136 | 8e-12 | 2.9 to 4.8 times faster |
| the same, case DK | classical | 117, 132 | 45, 36 | 2e-12 | the density is infinite at the tangency x = -sqrt(2 c0), where the stored checks hold the classical line integral |
| zero start, c0 != 0, c < -100 | general | 287 to 728 | 120 to 216 | 1.7e-10 | 2.4 to 3.3 times faster |
| stationary start, c0 != 0, c <= -0.1 | general | 203 to 10338 | 52 to 190 | 1e-9 | 2.4 to 140 times faster (case SK at c = c0 = -0.1: 10.3 s against 75 ms); values below 1e-3 go back to the classical conditional integral, whose error is relative |
| stationary start, c > -0.1 | classical | | | | the general engine is off by 2.5e-8 at c = -1e-4 |
| above the unit root, c0 < 0 (c >= 4) | classical | 604 | 734 | 6.5e-3 | the general engine is wrong (7e-3 from its checked value; the classical 8e-15 from its own) |
| above the unit root, the bulk at c >= 7 | classical | 288 to 1089 | 1198 to 37551 | 1.4e-14 | 2.7 to 34 times faster |
| fixed start, case DK, c <= 1, (c g0)^2/2 <= 400 | general | 121 to 404 | 76 to 157 | 5e-11 | 1.04 to 2.6 times faster |
| fixed start, case DK otherwise, cases DU, DT | classical | 150 to 620 | 155 to 1149 | 1.4e-10 | mostly faster, and its warnings of the brittle regimes |
| moments, c >= -50 | general | 14 to 444 | 13 to 27 | 8e-13 relative | equal to 25 times faster |
| moments, c < -50 | classical | 91 to 519 | 29 to 343 | 1.6e-9 relative | the stored checks of the classical engine (the classical third and fourth moments are documented to 1e-8 to 2e-7 at c = c0 = -300 with a stationary start) |
| quantiles at c0 = 0: a fixed start, cases DT and ST for c <= 0, case DU at c = 0 | general | 408 to 1142 | 214 to 434 | 1e-10 | 1.7 to 3.8 times faster (one path, F and f in one pass) |
| quantiles otherwise | classical | 48 to 388 | 145 to 725 | 3e-11 | 1.1 to 15 times faster; each evaluation of the search routed |
| the coefficient statistic | as the t-ratio at x = 0 with the null value c0 + y | 14 to 182 | 24 to 137 | 8e-11 | (case SU at c = -5: 7.5 times faster by the general engine) |
| P(V <= b) (`ltratio_lincdf` with a = NaN) | general | 6 | 0.03 | 3e-15 | 200 times faster |
| P(U - aV <= b), abs(a) < 1 (case DK), and case DG | general | 3 to 93 | 8 to 35 | 1.2e-5 (a = 0) | the classical inversion is off at small abs(a) (by 4e-5 against the closed form at a = 0); case DG 2.6 times faster |
| P(U - aV <= b), abs(a) >= 1, case DK | classical | 5 to 6 | 8 to 15 | 1e-12 | 1.3 to 2.7 times faster |
| the power envelope | classical | 79, 627 | 135, 835 | 2.1e-9 | 1.3 to 1.7 times faster |
| ADFmax | classical | 30 to 214 | 80 to 83 | 3e-12 | faster with a zero start; with a stationary start the general engine is 2.2 to 2.6 times faster, but the stored checks hold the classical values to 1e-12 |
| regions (`ltratio_region`) | classical | 2 | 6 | 1e-12 | 3.3 times faster, and the only engine for the WAP boundary and a function g |
| every design beyond the classical cases or beyond the classical engine's checks | general | | | | the only engine |

Where the classical engine computes, its values are those of 0.5.0 bit for bit. The linear form P(U - aV <= b) is being
repaired in the classical engine (it is inaccurate near the edge of the support, by up to 7e-2, and at small |a|); the
general engine's `lin_cdf` is off by 1.2e-4 at that edge too. Once the repair lands, the rule for the linear form should
be measured again: where the repaired classical inversion is as accurate, case DK with |a| < 1 goes back to it (2.5 to
4 times faster), P(V <= b) and case DG stay with the general engine unless the repair changes their cost.

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

## Installation

The package needs Julia 1.10 or later. It is not registered; install it from its GitHub repository with

```julia
using Pkg
Pkg.add(url = "https://github.com/s-broda/DFLimit.jl")
```

or, from a local clone, `Pkg.develop(path = "/path/to/DFLimit.jl")`. The dependencies (SpecialFunctions, QuadGK,
PrecompileTools) are installed with it. The first `using DFLimit` precompiles the package, which takes about half a
minute: it runs a few of the functions once, so that their first call in a session does not wait several seconds for
compilation. `Pkg.test("DFLimit")` runs the tests (about six minutes; they compare the values with values stored by the
paper's code, and end with the engine's own checks in `test/engine/`). The longer checks of the engine and the scripts that
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
