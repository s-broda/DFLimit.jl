# Volatility and persistence regimes

The Dickey-Fuller t-ratio when the standard deviation of the innovations changes at known dates, and, if wanted, the
local-to-unity parameter with it. The data are y_t = d(t/n)'beta + u_t with u_t = phi_t u_{t-1} + omega(t/n) e_t, u_0 =
0, e_t independent with unit variance, phi_t = 1 + c(t/n)/n, and omega and c constant on the regimes between the break
dates (fractions of the sample); the statistic is the t-ratio of the lagged level in the least squares regression on
d(t/n), with the usual variance estimate. Its limit is (U - c0 V)/sqrt(V), U and V the functionals of the general engine
of the process dJ = c(r) J dr + omega(r) dW, J(0) = 0, with omega scaled so that the integral of omega^2 is one (the
variance estimate converges to the average of omega^2, and the statistic does not depend on the scale of omega).

The variants of the engine carry over to the regimes: GLS detrending (`gls = cbar`), the statistic that detrends first
(`detrend_first`), and a stationary initial observation (`start = :stationary`). Neither the detrending nor the variance
estimate uses the variance path; the regimes change the law of the path alone. For a fixed initial observation the
transform is computed and the distribution is not (below).

## The transform

On the pieces of the design refined at the regime boundaries let theta_j = sqrt(c_j^2 - 2t omega_j^2) and g_j = (c_j +
theta_j)/(2 omega_j^2). Girsanov's theorem, piece by piece, turns the tilt by exp(t int J^2) into the process dJ =
-theta(r) J dr + omega(r) dW, further tilted by exp{g_last J(1)^2 + sum_j (g_j - g_{j+1}) J(b_j)^2} at the right ends b_j
of the pieces, and E exp(sU + tV) = e^{-s/2} D(s,t)^{-1/2} with

    D(s,t) = exp{sum_j (c_j + theta_j) L_j} det{I - 2 (s A + t B + Gamma) Sigma},

A and B the design's matrices padded with zeros for J at the regime boundaries, Gamma the diagonal matrix of the tilts and
Sigma the covariance of the functionals under the tilted process (the engine's transfer functions with the exponent and
the variance of each piece). This is the engine's transform with a diagonal tilt in place of gamma E11, and everything
downstream of the determinant is the engine's code: the outer contour (the hyperbola, steepened while a group of roots
leaves its side of the imaginary axis), the continuation of the roots and of log P along it, the inner integrals around
the roots, the rules of one evaluation, the checked computation and the quantile search. (The steepening while either
group leaves its side, where the engine looked at the positive group only, changes no path of the engine's own designs:
464 configurations with zero, fixed and stationary starts and GLS detrending, c from -480 to 14.) Two constants differ
from the engine's: under regimes the hyperbola starts at 70 degrees, where the engine's designs start at 45, and the
thin outer step of the default rule is twice the step of that hyperbola, where it is 1.5 times for the engine's designs
(`DFDet.VOL_PHI`, `DFDet.VOL_HFAC`; the reason and the measurements are under "Accuracy" below). The calls with a
volatility path take the general engine whatever the rule of the engines (`autoengine`), and `engine = :classical`
refuses them. Without deterministic terms D is
the first entry of a product of two-by-two transfer matrices of the regimes, which the tests use as an independent
check.

## Detrending and the start

Each variant enters through the one part of the transform that it changes.

GLS detrending and the statistic that detrends first change the matrices A and B alone. The design is refined at the
regime boundaries and prepared as the engine prepares it (`prep_gls` at cbar, or the numerator of `detrend_first`): the
functionals are the path at one and at the dates of the prepared design and its integrals against the basis of that
design, and the law of the path enters as above, through the exponent, Gamma and Sigma. With GLS detrending the numerator
has rank one, one positive root, as at a constant variance.

A normal start, J(0) centred normal with variance v0 and independent of W, makes J(0) one more functional. With h the
mean of the functionals under the tilted process started at one, the covariance of (ell, J(0)) is [Sigma + v0 h h', v0 h;
v0 h', v0], the matrices are those of the engine's stationary start (A bordered by b/2 and -1/2, B by zeros), and Gamma
gains -g_1 at the start. `start = :stationary` is the stationary start of the first regime, v0 = omega_1^2/(-2 c_1), which
needs c < 0 there.

A fixed start J(0) = gamma0 shifts the mean of the functionals to gamma0 h, and log M = log M_0 + gamma0^2 Q(s,t) with M_0
the transform of the zero start and Q the exponent of the engine's fixed start at the tilt t B + Gamma, with g_1 in place
of gamma. J(0) is in the units of the path, whose innovation variance averages one over the sample: y_0 = gamma0 sigma
sqrt(n) with sigma^2 the average innovation variance. A normal start is the average of the fixed one over gamma0, D_N =
D_0 (1 - 2 v0 Q), which the tests use in both directions. The exponent is `DFLimit.DFDet.vol_qf`. The distribution with a
fixed start is not computed under regimes, and `dist_cdf`, `dist_pdf`, `dist_quantile` and the classical functions raise
an `ArgumentError` for `start = :fixed` with a nonzero `gamma0` and more than one regime: the engine lays the paths of a
fixed start by rules made for a constant variance and persistence, and under regimes they lose accuracy or fail for starts
of moderate size (the start counts in the units of the first regime's standard deviation; with an intercept and a
persistence that changes, or far from the unit root, no angle of the outer contour gave a value in November 2026).

A start is combined with least squares detrending in the one-step regression, as in the engine: with GLS detrending or
with the statistic that detrends first the limit is derived for a zero start, and a start raises an `ArgumentError`.

## Calls

| call | computes |
|---|---|
| `volpath(breaks, omega)` | the regimes: break dates in (0, 1), the standard deviations (any positive scale) |
| `dist_cdf(D, xs, c; vol)`, `dist_pdf`, `dist_quantile(D, p, c; vol)` | the t-ratio of any design D under the regimes |
| the same with `gls = cbar` | GLS detrending at 1 + cbar/n |
| the same with `detrend_first(D)` | the statistic that detrends first |
| the same with `start = :stationary` | the stationary start of the first regime, J(0) ~ N(0, omega_1^2/(-2 c_1)) |
| `ltratio_cdf(x, cas_or_D, c; vol)`, `ltratio_pdf`, `ltratio_icdf` | the same through the classical functions (a case string or a design; `"SU"`, `"ST"` the stationary start, `"DG"` GLS detrending) |
| `hegy_t3_cdf(D, xs, c; vol)`, `hegy_t3_pdf`, `hegy_t3_quantile` | the seasonal t-ratio of Dickey-Fuller type under the regimes ([Seasonal unit roots](@ref)) |

The local-to-unity parameter c is a number, or a vector with one value per regime (changing persistence); c0, the local
null value of the t-ratio, is a number as always. A volatility path with one regime and a scalar c is the engine itself.
The extension covers least squares detrending with a zero or a stationary start, and with a zero start GLS detrending and
the statistic that detrends first, every c at most 0 (above the unit root it is
not implemented) and at least -480, and the inversion on paths (`method = :auto` or `:paths`; the line integral is not
derived for this transform). `verified = true` is the engine's checked computation, with the variant contours built for
the regimes.

The parts of the transform are functions of the general engine, `DFLimit.DFDet`: `vol_model(D, V, c; start, g0, gls)`
builds the model of a design under the regimes (and takes `v0`, the variance of a normal start other than the stationary
one), `vol_dval(VM, s, t)` is the determinant, `vol_qf(VM, s, t)` the exponent of the fixed start, `vol_roots(VM, t)` the
roots in s at a real t, `tstar(VM, 0)` the first zero of D(0, .), and `vol_path(VM)` the outer contour with the roots
continued along it.

## Examples

```julia
julia> V = volpath([0.1], [1.0, 3.0])               # the standard deviation triples after a tenth of the sample
volpath([0.1], [0.349215, 1.04765])

julia> cv = ltratio_icdf(0.05, "DU")                # the 5% critical value at a constant variance
-2.8615688636519616

julia> ltratio_cdf(cv, "DU"; vol = V)               # the size of that test under the variance break
0.02278491904449314

julia> ltratio_icdf(0.05, "DU"; vol = V)            # the 5% critical value under the break
-2.579981541980405

julia> ltratio_cdf(cv, "DU", [0.0, -20.0]; vol = volpath([0.5]))   # power when c falls from 0 to -20 at mid-sample
0.13186507163798253

julia> dist_quantile(perron(:C, 0.4), [0.01, 0.05], -5.0; vol = volpath([0.6], [3.0, 1.0]))
2-element Vector{Float64}:
 -5.272473428374128
 -4.674747457616461
```

The last line is a break in level and slope at 0.4 with the standard deviation falling to a third at 0.6, under the local
alternative c = -5.

The variants, with the standard deviation tripling at 30% of the sample:

```julia
julia> V = volpath([0.3], [1.0, 3.0]);

julia> cvg = ltratio_icdf(0.05, "DG")               # GLS detrending, linear trend: the 5% value at a constant variance
-2.845630826159995

julia> ltratio_cdf(cvg, "DG"; vol = V)              # the size of that test under the variance break
0.07124632629844938

julia> ltratio_icdf(0.05, "DG"; vol = V)            # its 5% critical value under the break
-2.996681074104641

julia> dist_quantile(detrend_first(perron(:B, 0.4)), 0.05, 0.0; vol = V)   # a slope break at 0.4, detrended first
-3.71083413593991

julia> ltratio_cdf(-2.8615688636519616, "SU", -5; vol = V)   # the stationary start of the first regime, c = -5
0.11821415344519763

julia> DFLimit.DFDet.vol_roots(DFLimit.DFDet.vol_model(trend(0), V, -4.0; start = :stationary), 0.5)   # the roots of D(., 1/2)
2-element Vector{Float64}:
 -22.778950831346375
   7.769016921838749
```

## Accuracy and cost

At a constant variance with a dummy boundary the distribution function and the density are those of the engine's paths to
1e-15, at c = 0, -10 and -100; the transform equals the engine's determinant to 1e-10 relative and, without deterministic
terms, the product of the transfer matrices to 1e-11; extrapolated in 1/n, the exact moment generating function of the
regression in samples of 200, 400 and 800 agrees with it to 2e-5. The values of the prototype (the table of sizes at
eight decimals, its quantiles at four, the power under changing persistence), which were checked there against
simulations of the regression and of the limit functional, are reproduced to their printed digits. The rule of one
evaluation is the engine's (`path_region`); on the 36 cells of the table it is within 4e-12 of the present rule.

GLS detrending, the statistic that detrends first and the determinant of a normal start were computed for the lead paper
by a prototype among its replication code before the package had them; its values are kept in `test/ref/regime_ref.txt`.
On the prototype's contour, the engine's 45-degree hyperbola, and with its nodes the package reproduced the distribution
function and the density to 2e-15 (291 evaluations, among them the 216 cells of the paper's table of sizes and critical
values under one variance break and the p-values of its illustration); on the 70-degree hyperbola it now takes under
regimes, `rule = :present` is within 5e-12 of the prototype and the default rule of one evaluation within 1e-11 (on the
45-degree hyperbola the default rule was off by 6.5e-9 for the quadratic trend detrended first under three regimes with
their own c). The determinant, the roots and t_* agree to 1e-13, and the 369 quantiles to 2e-10, the tolerance of the
searches. The
prototype's determinant of a normal start and its determinant of the zero start yield the exponent of the fixed start,
which `vol_qf` reproduces to 1.4e-10.

The stationary start, which the prototype did not invert, agrees with the engine's at a constant variance with a dummy
boundary to 1e-14; its transform agrees with the exact moment generating function of the regression in samples of 200,
400 and 800, extrapolated in 1/n, to 2e-5; and its distribution function equals the average over a normal start of the
distribution function with a fixed start, computed from the zero start's roots and the exponent of the start, to 5e-11.

On a wider grid (140 cases: five designs with a stationary start, five with GLS detrending and four detrended first,
under ten paths with ratios of the standard deviations up to 9 and c down to -150, November 27, 2026) the default rule is within 1e-6 of
the checked computation in the distribution function, with two exceptions, both announced. Where the computed value in
the upper tail exceeded one by more than 1e-8 the call failed with the engine's range error (one case, a stationary start
with a break in level and slope; the steeper hyperbola of the next paragraph computes it). And with GLS detrending at a
cbar near zero (-2), or beside a standard deviation that rises ninefold late in the sample, no hyperbola keeps the positive
root right of the origin and the outer contour is the vertical line: the engine warns there, and the values differed by up
to 5e-4 between rules or the call failed.

The outer hyperbola under regimes starts at 70 degrees. On the engine's 45-degree hyperbola a root of the transform can
come close to the imaginary axis far out (an early regime of high variance together with a break in the deterministic
terms or a persistence that changes: with a break in the trend at 0.5, standard deviations 2, 1, 0.5 on regimes ending at
0.2 and 0.7 and c = -5, 0, -10 the smallest real part of the positive roots falls from 15 at the vertex to 1.2 at
|t| = 1200). The part of the outer integrand that belongs to that root is then not damped, and its phase turns faster
than the outer step resolves, so the trapezoidal rule loses the same amount at every x: 1.4e-6 in that case, where the
99.9% quantile is -0.67 and the call failed with the range error from x = 1 on. On 1456 cases (14 designs, 13 paths of two
or three regimes with ratios of the standard deviations up to 9, c constant and changing, zero and stationary start;
November 28, 2026, against the 80-degree hyperbola with half the step and the inner nodes of the present rule) the
45-degree hyperbola with the thin step was off by more than 1e-5 in 17 cases and by more than 1e-8 in 71 (3.8e-4 at
most), and with the present step by more than 1e-5 in 2 (6.8e-5). Starting at 70 degrees with twice the hyperbola's step
the largest error is 1.7e-8, on 195 nodes on average against 140 of the thin step and 209 of the present step on the
45-degree hyperbola; 2.5 times the step lost 2.6e-6 in one case. Without regimes the thin step on the 45-degree hyperbola
is within 1e-13 of the same reference on these designs, and the engine's own contour is unchanged.

Times on warm calls (four threads of an M4 laptop on mains power with nothing else running, November 26, 2026): the
outer contour with its roots 3 to 19 ms, one value of F and f 2 to 5 ms, a 5% quantile 9 to 45 ms (no deterministic
terms, an intercept, a linear trend with the variance tripling at mid-sample; a break in level and slope at 0.4 with the
variance falling at 0.6), 1.4 to 3.6 times faster than the prototype in the same session: the thin rule of one
evaluation, the engine's static arrays for the small matrices of the transform, its continuation of the roots between the
nodes only where a step is not safe, and t_* from the Nystrom estimate rather than from two discretized regressions. On the 70-degree
hyperbola (November 28, 2026, the same machine, the minimum of 60 to 200 repetitions after a warm-up, the two versions in
alternating processes): one value of F 2.4 to 7.7 ms and a 5% quantile 13 to 63 ms, 1.2 to 1.4 times the 1.7 to 5.4 ms
and 9 to 45 ms of the 45-degree hyperbola with the thin step; a first call from empty caches 6 to 28 ms, 0.95 to 1.2
times.
