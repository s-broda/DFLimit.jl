# A normal component: covariates and the predictive regression

Two statistics have the same limit, rho times the Dickey-Fuller limit at a shifted local null value plus an independent
normal variable,

    T = rho tau_{c0} + sqrt(1 - rho^2) Z,    tau_{c0} = (U - c0 V)/sqrt(V),

Z standard normal and independent of (U, V). The covariate-augmented t-ratio of Hansen (1995) adds stationary covariates
to the Dickey-Fuller regression; rho is the long-run correlation of the innovation of the series and the error of the
augmented regression, R the ratio of their long-run standard deviations (Hansen's rho^2 and R^2 are the squares; R =
|rho| when the regression is well specified), and c0 = c (1 - 1/(rho R)), zero at the unit root (`cadf_c0`; c is the
package's local-to-unity parameter, Hansen's c with the opposite sign). The t-ratio of the slope beta in the predictive
regression y_t = mu + beta x_{t-1} + u_t, x_t = phi x_{t-1} + v_t, with rho the correlation of the innovations and the
local alternative beta = b sigma_u/(sigma_v n), has this limit with the intercept's tau and c0 = c - b/rho (the companion
paper's section on the predictive regression; `predictive_c0`). One keyword serves both: `rho` of the t-ratio functions,
with c0 as always.

## Two computations

The inversion on two lines with the normal component in the kernel (`method = :lines`): the double inversion of the joint
moment generating function of (U, V), as the engine has it on its outer contour in t, against the kernels of the lead
paper at changed arguments,

    P(T > x) = (2 pi i)^{-2} int int M(s,t) (2/s) E_1(x s/rho, p) dt ds,    f_T(x) = (2 pi i)^{-2} int int M(s,t) (2/rho) E_2(x s/rho, p) dt ds,

with p = t + c0 s - (1 - rho^2) s^2/(2 rho^2) and E_m(b, p) = int_0^inf v^m e^{-bv - pv^2} dv in closed form (the
incomplete Gaussian integrals). The inner integral runs over a line Re s = sigma between the origin and the nearest root
on the side given by the sign of x, its two ends turned into the right half-plane. The normal average (`method =
:mixture`): P(T <= x) = int F((x - rb z)/rho) phi(z) dz with F the t-ratio's distribution function, the trapezoidal rule
in z (graded toward the kink of F when c0 > 0), the density by an integration by parts. `method = :auto` takes the lines
for 0.1 <= rho^2 <= 0.99 with a zero or a stationary start at c <= 0, the normal average otherwise (near |rho| = 1 the
kernel of the lines oscillates and the normal average is the cheaper; the lines do not cover a fixed start or the
vertical line above the unit root). `verified = true` computes both and returns the normal average on the checked
distribution function, with a warning, where they differ by more than `vtol`.

The normal average calls the general engine's distribution function of the t-ratio, at the shifted local null value c0
and at all the points of its rule in one call, with the engine's `method = :auto`: the line integral at c0 = 0 where it
covers the design (the paths above the unit root while the outer hyperbola keeps its angle of 45 degrees), the paths at
c0 != 0, and with c0 > 0 on the vertical line above the unit root the line integral at the points x < 0. The rule of the
engines (`autoengine`) is not consulted there: its cells were measured on calls of five points, and the general engine
shares its outer contour over the hundreds of points of the average. `rho = -1` with a case string or a classical design
is the t-ratio's own law at -x, by the engine of the rule. The engine's warnings of a brittle computation (far above the
unit root, the vertical line's parts at a root) come once per call, not once per point or per step of the quantile
search.

## Calls

| call | computes |
|---|---|
| `ltratio_cdf(x, cas_or_D, c; rho, c0)`, `ltratio_pdf`, `ltratio_icdf(p, cas_or_D, c; rho, c0)` | the limit T of any design, start and c |
| `dist_cdf(D, xs, c; rho, c0)`, `dist_pdf`, `dist_quantile(D, p, c; rho, c0)` | the same, the general engine's argument order |
| `cadf_c0(c, rho, R = abs(rho))` | Hansen's local null value c (1 - 1/(rho R)) |
| `predictive_c0(c, rho, b = 0)` | the predictive regression's c - b/rho |

rho lies in [-1, 1]: rho = 1 is the t-ratio itself (the same call without the keyword), rho = 0 the standard normal law,
and a negative rho the reflection P(T <= x) = 1 - P(|rho| tau + rb Z <= -x). Least squares detrending and the one-step
regression (`gls` and `detrend_first` raise an `ArgumentError`, as does a volatility path together with rho).

## Examples

```julia
julia> ltratio_icdf([0.01, 0.05, 0.1], "DU"; rho = sqrt(0.5))     # Hansen's test with an intercept, rho^2 = 0.5
3-element Vector{Float64}:
 -3.2062789931174214
 -2.5867469344813205
 -2.2575791153746296

julia> c0 = cadf_c0(-5, sqrt(0.5))                  # the local null value at c = -5 (R = rho)
4.999999999999998

julia> ltratio_cdf(-2.58675, "DU", -5; rho = sqrt(0.5), c0)       # the power of the 5% test at c = -5
0.36828922771661277

julia> 1 - ltratio_cdf(1.6448536269514722, "DU", 0; rho = -0.95, c0 = predictive_c0(0, -0.95))
0.42125879087918017

julia> ltratio_icdf(0.95, "DU", -5; rho = -0.95, c0 = predictive_c0(-5, -0.95))
2.3775199949214865

julia> 1 - ltratio_cdf(1.645, "DU", -5; rho = -0.95, c0 = predictive_c0(-5, -0.95, 10))
0.9987165058779714
```

The last three lines are the predictive regression with rho = -0.95: the size of the right-tailed t-test at the normal
5% value when the regressor has a unit root (the companion paper's 0.4213), the exact 95% point of the null limit at c =
-5 (2.378), and the power against the local alternative b = 10 at the normal value.

## Accuracy and cost

The two computations share the engine's outer contour, its roots and log P, and nothing else: they agree to 1e-9 and
mostly to 1e-13 (designs with and without deterministic terms and with breaks, rho^2 from 0.2 to 0.95, local
alternatives with c0 of either sign, a stationary start, c = -50). The critical values and powers of the prototype's
table (Hansen's designs and two with breaks, checked there against simulations of the regression and of the limit and
against Hansen's simulated table) are reproduced to their five and four printed decimals, and the companion paper's
table of the sizes of the predictive regression's t-tests, a normal average of the classical engine's distribution
function, to its four.

Times on warm calls (four threads of an M4 laptop, November 26, 2026, measured while another job shared the machine):
the inversion on two lines 25 to 95 ms a value (rho^2 = 0.5 to 0.9; no deterministic terms, an intercept, a linear trend,
a level break), a 5% quantile 0.1 to 0.8 s, the normal average 0.1 to 0.35 s a value at c0 = 0 and about 4 s with c0 > 0
(its rule graded toward the kink), the same as the prototype within the scatter of the load (the algorithms are the
prototype's).
