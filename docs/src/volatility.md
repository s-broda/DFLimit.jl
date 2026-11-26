# Volatility and persistence regimes

The Dickey-Fuller t-ratio when the standard deviation of the innovations changes at known dates, and, if wanted, the
local-to-unity parameter with it. The data are y_t = d(t/n)'beta + u_t with u_t = phi_t u_{t-1} + omega(t/n) e_t, u_0 =
0, e_t independent with unit variance, phi_t = 1 + c(t/n)/n, and omega and c constant on the regimes between the break
dates (fractions of the sample); the statistic is the t-ratio of the lagged level in the least squares regression on
d(t/n), with the usual variance estimate. Its limit is (U - c0 V)/sqrt(V), U and V the functionals of the general engine
of the process dJ = c(r) J dr + omega(r) dW, J(0) = 0, with omega scaled so that the integral of omega^2 is one (the
variance estimate converges to the average of omega^2, and the statistic does not depend on the scale of omega).

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
the roots, the rules of one evaluation, the checked computation and the quantile search. Without deterministic terms D is
the first entry of a product of two-by-two transfer matrices of the regimes, which the tests use as an independent
check.

## Calls

| call | computes |
|---|---|
| `volpath(breaks, omega)` | the regimes: break dates in (0, 1), the standard deviations (any positive scale) |
| `dist_cdf(D, xs, c; vol)`, `dist_pdf`, `dist_quantile(D, p, c; vol)` | the t-ratio of any design D under the regimes |
| `ltratio_cdf(x, cas_or_D, c; vol)`, `ltratio_pdf`, `ltratio_icdf` | the same through the classical functions (a case string or a design) |
| `hegy_t3_cdf(D, xs, c; vol)`, `hegy_t3_pdf`, `hegy_t3_quantile` | the seasonal t-ratio of Dickey-Fuller type under the regimes ([Seasonal unit roots](@ref)) |

The local-to-unity parameter c is a number, or a vector with one value per regime (changing persistence); c0, the local
null value of the t-ratio, is a number as always. A volatility path with one regime and a scalar c is the engine itself.
The extension covers the zero start, least squares detrending and the one-step regression (a fixed or stationary start,
`gls` and `detrend_first` raise an `ArgumentError`), every c at most 0 (above the unit root it is not implemented) and at
least -480, and the inversion on paths (`method = :auto` or `:paths`; the line integral is not derived for this
transform). `verified = true` is the engine's checked computation, with the variant contours built for the regimes.

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

## Accuracy and cost

At a constant variance with a dummy boundary the distribution function and the density are those of the engine's paths to
1e-15, at c = 0, -10 and -100; the transform equals the engine's determinant to 1e-10 relative and, without deterministic
terms, the product of the transfer matrices to 1e-11; extrapolated in 1/n, the exact moment generating function of the
regression in samples of 200, 400 and 800 agrees with it to 2e-5. The values of the prototype (the table of sizes at
eight decimals, its quantiles at four, the power under changing persistence), which were checked there against
simulations of the regression and of the limit functional, are reproduced to their printed digits. The rule of one
evaluation is the engine's (`path_region`); on the 36 cells of the table it is within 4e-12 of the present rule.

@@TIMINGS@@
