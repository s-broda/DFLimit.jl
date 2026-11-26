# DFLimit.jl

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
statistic for any design ([Two engines](@ref)). The classical engine is the multithreaded Julia engine of the companion
paper, `DFExact.jl`, whose home is this package (`src/`); the same functions, with the same names and arguments, exist
in MATLAB and, all but the boundary type `"fun"` of `ltratio_region`, in R and Python.

## The functions

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

The case codes, the representations and the options are described in [Case codes and arguments](@ref), the two engines
and the choice between them in [Two engines](@ref), the statistics other than the t-ratio in
[Statistics beyond the t-ratio](@ref), and every function in the [API reference](@ref).

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

```bibtex
@unpublished{BaoBrodaKan2026main,
  author = {Bao, Yong and Broda, Simon A. and Kan, Raymond},
  title  = {The Exact Limiting Distribution of the {Dickey}--{Fuller} $t$-Ratio with General Deterministic Terms},
  year   = {2026},
  note   = {Working paper, available at SSRN},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7745921}
}

@unpublished{BaoBrodaKan2026lowrank,
  author = {Bao, Yong and Broda, Simon A. and Kan, Raymond},
  title  = {Low-Rank Representations of the Classical {Dickey}--{Fuller} Limits, with an Extension to Predictive Regressions},
  year   = {2026},
  note   = {Working paper, available at SSRN},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7725619}
}
```
