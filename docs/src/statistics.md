# Statistics beyond the t-ratio

Besides the t-ratio the package has the other statistics of the limit experiment of the Dickey-Fuller regression, the
pair (U, V) of the numerator and the denominator: the classical engine for the classical cases, the general engine for
any design (see [Two engines](@ref)). Every function below takes a case code or a design in place of `cas`, and its
docstring (`?coef_cdf`) has the arguments; how the half-plane statistics and the moments are computed is in
[The devices of Bao and Kan](@ref).

## The classical functions

| function | computes |
|---|---|
| `coef_cdf(y, cas, c = 0; c0, ...)`, `coef_icdf(p, cas, c = 0; tol, ...)` | the coefficient statistic n(phi^ - phi0), whose limit is (U - c0 V)/V |
| `coef_pdf(y, cas, c = 0; c0, g0)`, `coef_mom(p, cas, c = 0; c0, g0)` | its density and moments |
| `adfmax_cdf(x, cas, c = 0)`, `adfmax_pdf`, `adfmax_icdf(p, cas, c = 0; tol, x0)` | the ADFmax statistic of Leybourne (1995), the larger of the t-ratios of the forward and the reverse regression (cases DU, DT, SU, ST) |
| `ltratio_region(bnd, cas = "DK", c = 0)` | P{U <= g(V)} for a line, the parabola of the t-ratio, the boundary of the weighted-average-power test of the companion paper or a function g |
| `ltratio_lincdf(b, a, cas, c = 0; cbar, N, umax)` | the distribution of the linear form U - aV (a = NaN: of V), case DK or DG; the point-optimal statistics are of this form |
| `power_envelope(c, cas = "DK"; alpha)` | the asymptotic power envelope of Elliott, Rothenberg and Stock (1996), case DK (= DU) or DG |

Every test of the unit root in the limit experiment of case DK rejects on a region of the plane of (U, V), and the
first part of the conditional integral representation of the companion paper, the probability of a region of the
plane, gives its probability for every boundary g by one outer integral over V; the ADFmax statistic rejects on the
region between two curves of the plane. The boundary `bnd` of `ltratio_region` is a NamedTuple (or a Dict):
`(type = "line", a, b)`, `(type = "parabola", x, c0)`, `(type = "wap", k, omega)` or `(type = "fun", g)` with `g` a
function of a scalar, optionally with `v0` and `brks`.

```julia
julia> coef_cdf(-14.0936, "DU", -10)                  # power of the 5% coefficient test at c = -10
0.45925256320763047

julia> coef_mom(1, "DK")                              # the mean of n(phi^ - 1) at the unit root
1-element Vector{Float64}:
 -1.781430171277838

julia> adfmax_icdf(0.05, "DU")                        # 5% critical value of ADFmax with a constant
-2.435416499432509

julia> ltratio_region((type = "wap", k = 3.6658, omega = 10), "DK")   # size of the WAP test of the companion paper
0.049999622834651807

julia> ltratio_lincdf(-0.2639, -3.5, "DK", -7)        # power of the point-optimal test against cbar = -7 at c = -7
0.49938157210469375

julia> power_envelope(-7)                             # the power envelope at c = -7
0.4994019828155684
```

## The general engine

The general engine's own functions are exported as well: `dist_cdf(D, xs, c; c0, start, gamma0, gls, method)`,
`dist_pdf`, `dist_quantile`, `dist_moments` (the t-ratio), `coef_pdf`, `coef_quantile`, `coef_cdf_gp` (the coefficient
statistic), `lin_cdf`, `lin_pdf`, `lin_quantile`, `v_cdf`, `v_pdf` (the linear form U - aV and the law of V),
`adfmax_quantile`, `region_prob`, `local_power`, `simulate_tratio`, `simulate_gls`, `simulate_first` and the designs;
its `coef_cdf`, `adfmax_cdf`, `adfmax_pdf` and `power_envelope`, whose names are the classical functions', are
`DFLimit.DFDet.coef_cdf` and so on. Note the order of the arguments: the general engine's own functions take the
design first (`dist_cdf(D, xs, c)`), the classical functions the points first (`ltratio_cdf(x, D, c)`).

## The statistics of the lead paper beyond the t-ratio

The statistics of the lead paper beyond the t-ratio are functions of the general engine: the point-optimal statistic of
Elliott, Rothenberg and Stock and the power envelope for any design (`pt_cdf(D, qs, cbar, c)`, `pt_pdf`,
`pt_quantile(D, p, cbar)`, `pt_power(D, cbar, c)`, `pt_envelope(D, cs)`, also through `power_envelope(c, D)`; for the
intercept and the linear trend it is the envelope of the classical construction to 1e-8), the LM statistic of Schmidt
and Phillips and of Amsler and Lee (`lm_cdf(D, xs, c)`, `lm_quantile(D, p)`, `lm_power(D, cs)`; 5% value -3.0230 for
the linear trend; for trend functions whose span contains the constant and a linear trend, an `ArgumentError` otherwise,
`DFDet.span_has_constant`) and the companions of the GLS-detrended t-ratio, MZ_alpha (`mza_cdf(D, ys, c; gls)`,
`mza_quantile`) and MSB (`msb_cdf(D, xs, c; gls)`, `msb_quantile`).

```julia
julia> D = perron(:C, 0.5);                           # a break in level and slope at the middle of the sample

julia> local_power(D, [-5.0, -10.0, -20.0])           # the power of the 5% test
3-element Vector{Float64}:
 0.06689363562246417
 0.11488561464598873
 0.34595668565582766

julia> pt_envelope(D, [-5.0, -10.0, -20.0])           # the power envelope of the design
3-element Vector{Float64}:
 0.08062917914287143
 0.17888853618084333
 0.606153452406375

julia> lm_quantile(perron(:B, 0.5), 0.05)             # the LM test with a slope break
-3.6474783394296186
```
