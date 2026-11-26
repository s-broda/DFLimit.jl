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

@@EXAMPLES@@

## Accuracy and cost

The two computations share the engine's outer contour, its roots and log P, and nothing else: they agree to 1e-9 and
mostly to 1e-13 (designs with and without deterministic terms and with breaks, rho^2 from 0.2 to 0.95, local
alternatives with c0 of either sign, a stationary start, c = -50). The critical values and powers of the prototype's
table (Hansen's designs and two with breaks, checked there against simulations of the regression and of the limit and
against Hansen's simulated table) are reproduced to their five and four printed decimals, and the companion paper's
table of the sizes of the predictive regression's t-tests, a normal average of the classical engine's distribution
function, to its four.

@@TIMINGS@@
