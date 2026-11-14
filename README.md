# DFLimit.jl

The exact limiting distribution of the Dickey-Fuller t-ratio in Julia.

`DFLimit` computes the distribution function, density, quantiles and moments of the limits of the Dickey-Fuller
t-ratios under local alternatives, phi = 1 + c/n, by the exact representations of

> Bao, Y., Broda, S. A. and Kan, R., "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio and Some Related
> Statistics", working paper.

for the regression without deterministic terms, with an intercept and with an intercept and a trend; with a zero,
a fixed or a stationary initial observation; for the t-ratio centred at the unit root or at a local null value c0;
and for the GLS-detrended trend case of Elliott, Rothenberg and Stock (1996). It also has the ADFmax test of
Leybourne (1995), the coefficient statistic, the probabilities of regions {U <= g(V)} of the limit experiment
(point-optimal and weighted-average-power tests), the distribution of the linear form U - aV and the asymptotic power
envelope. The numbers come from the multithreaded Julia engine of the paper, `DFExact.jl`, which the package holds
unchanged; the same functions, with the same names and arguments, exist in MATLAB and, all but `ltratio_lincdf`,
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

The case `cas` is a two-letter code: `D` (fixed start, `g0`, default 0) or `S` (stationary start, c < 0),
followed by `K` (no deterministic terms, tau), `U` (intercept, tau_mu) or `T` (intercept and trend, tau_tau);
`"DG"` is the GLS-detrended trend case (DF-GLS, cbar = -13.5). `method` is `:auto` (the faster representation at
each point), `:single` or `:line` (the line integral, Theorem 2) or `:double` or `:conditional` (the conditional
integral, Theorem S1). `x`, `y` and the probabilities `p` may be arrays. The boundary `bnd` of `ltratio_region` is a NamedTuple (or a
Dict): `(type = "line", a, b)`, `(type = "parabola", x, c0)`, `(type = "wap", k, omega)` or `(type = "fun", g)` with
`g` a function of a scalar, optionally with `v0` and `brks`.

The functions return the values only (the MATLAB functions also return an `info` structure); every one has a
docstring (`?ltratio_cdf`). Below c = -550, and with a fixed start g0 outside the region in which the programs are
checked, they raise an `ArgumentError` that says so. The engine's functions that are not exported are reached through
the module `DFLimit.DFExact`, among them the two representations, which take the model number 1, 2, 3, 4 for K, U, T,
G: the line integral `DFLimit.DFExact.cdf(q, model; c, c0, g0, start = :zero)` (`start = :stationary` for a
stationary start), `DFLimit.DFExact.density` and `DFLimit.DFExact.moments(model; c, c0, g0, start)`, and the
conditional integral `DFLimit.DFExact.tr_limdist(x, model, c; c0, g0, start, pdf)` (see `julia/README.md` of the
paper's repository).

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

julia> ltratio_region((type = "wap", k = 3.6658, omega = 10), "DK")   # size of the WAP test of Section 7
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
compilation. `Pkg.test("DFLimit")` runs the tests (a few minutes; they compare the values with values stored by the
paper's code).

## Other implementations

The paper's repository has the same functions as MATLAB programs (`public/`), the C engine (`c/`, with a MEX gateway
for MATLAB) and the Julia engine of this package (`julia/`, with its checks and the scripts that reproduce the tables
and figures of the paper). The R package [`dflimit`](https://github.com/s-broda/dflimit) is built on the C engine.

## Licence and citation

MIT (see `LICENSE`). Please cite the paper when using the package.

The engine in `src/DFExact.jl` and `src/df_series.jl` is copied from the directory `julia/` of the paper's repository
by `tools/sync-engine.sh` (the upstream commit is in `UPSTREAM`); do not edit those files here.
