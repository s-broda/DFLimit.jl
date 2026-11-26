# Seasonal unit roots

At a frequency strictly between 0 and pi (pi/2 for quarterly data; each pair of harmonic frequencies of monthly data) a
pair of complex unit roots gives the regression of Hylleberg, Engle, Granger and Yoo two t-ratios and an F statistic.
Their limit process is complex: Z = W_a + i W_b with independent standard Brownian motions, dZ = c Z dr + dW, Z(0) = 0
(c in the units of the roots' modulus 1 + c/n, n the number of years), and Z~ its residual on the deterministic terms. The
three functionals

    R = Re int Z~* dZ,   L = Im int Z~* dZ,   V = int |Z~|^2

give t3 = (R - c0 V)/sqrt(V), the t-ratio of pi_3 in the quarterly numbering (the coefficient of y_{3,t-2}), of
Dickey-Fuller shape; t4 = L/sqrt(V), the t-ratio of pi_4 (the coefficient of y_{3,t-1}), symmetric about zero; and F =
(R^2 + L^2)/(2V), the F statistic for pi_3 = pi_4 = 0. The deterministic terms are the real trend functions of a design
of the general engine, entering with complex coefficients: `trend(-1)` for a regression without seasonal terms (a
constant or a linear trend leave the harmonic statistics alone), `trend(0)` for seasonal dummies (with or without a
trend), `trend(1)` for seasonal dummies and seasonal trends. At the frequencies 0 and pi the roots are real and the
statistics are the Dickey-Fuller t-ratios of the classical functions (with seasonal dummies the intercept's).

## The computations

t3. (R, V) is the sum of two independent copies of the lead paper's (U, V), so its transform is e^{-s}/D(s, t) with the
general engine's determinant D, whose singularities in s are poles: the inner integral of the inversion on paths is the
sum of the residues at the roots, and the density and the distribution function are single integrals over the engine's
outer contour of closed-form functions (the engine's kernels at the roots). This holds for every design of the engine and
for a volatility path (`vol`). Where two roots of a group come within 1e-4 (relative) of each other along the contour,
the residues are replaced by the engine's inversion on paths of the same transform with every root doubled.

t4 and F. The transform of the three functionals, E exp(sR + uL + tV) = e^{-s}/D(s, u, t), is the engine's determinant of
the complex process: by Girsanov's theorem the tilt by exp(uL + t int |Z|^2) turns Z into the complex Ornstein-Uhlenbeck
process with parameter -theta + iu, theta^2 = c^2 - 2t - u^2, and D = e^{c + theta} det{I - 2 (sA + uA_L + tB + gamma E11)
S}, with the engine's A and B, the area matrix A_L of the design (the antisymmetric part of the matrix whose symmetric part
is in A) and S the covariance with the engine's transfer functions at the two exponents theta -+ iu (at u = 0 the
engine's). The inversion on the lines Re u = 0 and Re t = tau, after the residues in s, gives double integrals of
closed-form functions of D(0, iy, tau + iw) and of its roots, on a product rule in y and sqrt(w) that is built once per
design and c and cached (closed forms of D without seasonal terms and with seasonal dummies at the unit root). Without
deterministic terms t4 is exactly standard normal, and the functions return the normal law.

## Calls

| call | computes |
|---|---|
| `hegy_t3_cdf(D, xs, c = 0; c0, vol)`, `hegy_t3_pdf`, `hegy_t3_quantile(D, p, c; x0, tol, c0, vol)` | t3, any design, a volatility path |
| `hegy_t4_cdf(D, qs, c = 0)`, `hegy_t4_pdf`, `hegy_t4_quantile(D, p, c; x0, tol)` | t4, the polynomial seasonal trends of degree -1, 0, 1 |
| `hegy_f_cdf(D, xs, c = 0)`, `hegy_f_quantile(D, p, c; tol)` | F, the same designs (there is no density) |

All take the zero start and c <= 0; t4 and F c >= -20 (the range in which their grids are checked). `verified = true`
gives t3 by both computations (a warning where they differ by more than `vtol`) and t4 and F on a second grid with another
tau and a shorter range.

## Examples

@@EXAMPLES@@

## Accuracy and cost

The residues agree with the engine's inversion on paths of the doubled transform to 2.4e-13 (designs with and without
deterministic terms, with breaks, c from 0 to -20 and -50, c0 of either sign), with the exponential law of |Z(1)|^2/2
without deterministic terms (P(t3 <= 0) = 1 - 1/e) to 2e-16, and with the elementary form at the unit root in spherical
Bessel functions to 1e-12. The transform equals its closed forms (without seasonal terms; seasonal dummies at the unit
root; the transforms of Nabeya (2001) with seasonal dummies and with seasonal trends) to 1e-10 and the engine's
determinant at u = 0 to 1e-11. Without deterministic terms the lines give the normal law for t4 to 1e-11 and the
convolution (t3^2 + chi^2_1)/2 for F to 1e-11; the moments of the densities of t3 and t4 agree with Nabeya's (2001) table
to its eight decimals; the critical values and powers of the prototype's table (checked there against simulations of the
regression) are reproduced to their four.

@@TIMINGS@@
