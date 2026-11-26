# Two engines

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

## One interface

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
inversion on paths around the roots), see [Which engine](@ref); for a design that is a classical case `method = :line` is the
classical line integral and `method = :paths` the general engine's paths, so that `ltratio_cdf(x, trend(2), c; method =
:paths)` and `ltratio_icdf(p, trend(0), c; method = :paths)` force the paths and `ltratio_cdf(x, "DU", c; engine =
:general, method = :line)` the general engine's line integral. The general engine is checked from c = -480 to 10, and
above the unit root, for the coefficient statistic, the moments and the t-ratio centred at the true value (c0 = c), up to c = 16 with
deterministic terms (10 with them and a fixed start) and up to c = 20 without them (`DFLimit.DFDet.gen_cmax`): beyond, one
root of its transform is of the
order 2c e^{-2c} and the others are no longer resolved in double precision, and every function of the general engine
raises an `ArgumentError` that names the statistic, c and the bound; the rule takes the classical engine there. Far above
the unit root, for c > 5.5, the inversion on paths by one evaluation may be off by more than 1e-5 in the upper tail for
a design beyond the classical cases and for a fixed start with deterministic terms (Perron's model C at c = 7 by 1.5e-5
in F, a linear trend with a fixed start at c = 7 by 8.7e-5), and the general engine warns there (`verified = true`
checks it).

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
