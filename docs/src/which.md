# Which engine

`dev/engine_table.jl` (`dev/engine_table.txt`) times both engines in each cell (4 threads, the warm time of one call,
BenchmarkTools minimum of 2; F and f at x = mean + (-2, -1, 0, 1, 2) sd for the t-ratio) and compares their values; where
they differ by more than 1e-8 it compares each with the other's checked computation (`verified = true`). The rule,
`autoengine` in `src/merged.jl`, takes the faster engine at the same accuracy, and the right one where they disagree.
It was re-derived on November 26, 2026, after the thinner rules of one evaluation had halved the time of the general
engine's paths: from interleaved timings of the two engines in each cell (the two choices alternated in one process,
warm, the minimum of 12 rounds each; two runs, a cell moved only where the other choice was at least 1.3 times faster in
both) and from checks of the new choice's one evaluation against the other engine's checked computation on 41 points
from the 0.01% to the 99.99% point over the cell's region (within 1e-7 in F, in f of the peak and in the quantiles'
probabilities). The times in the table below are those of `dev/engine_table.txt`.

Within the general engine, `dist_cdf`, `dist_pdf` and `dist_quantile` (and the classical functions with a design, through
them) choose between two representations by the keyword `method`: `:auto` (the default, `DFDet.line_auto`) takes the
line integral with the explicit kernel where it covers the design and c0 = 0, except above the unit root for 0 < c <= 1.5
with one regressor (the intercept) and 0 < c <= 2 with two or more, and with c0 > 0 at the points x < 0 where the outer
contour of the paths is the vertical line (`DFDet.line_vline`; the quantiles keep the paths there); the inversion on paths
around the roots otherwise; `:line` forces the line
integral (an `ArgumentError` with the reason where it does not cover the design, the start or the options) and `:paths`
the inversion on paths. The line integral covers the designs whose numerator has rank one or two: no deterministic terms,
the continuous designs that do not vanish at the origin and whose span is closed under differentiation (polynomial
trends of any order, an intercept with sines and cosines), and `detrend_first` of a continuous design that does not
vanish at the origin, with a zero or a stationary start and least squares detrending (not GLS detrending, a fixed start
or a design with a jump). It is the classical engine's integrator with P(t) and the two roots supplied by the general
engine's transform (`src/DFDet_line.jl`). On warm calls at c0 = 0 (November 26, 2026; F and f at five points and the 5%
quantile, four threads) it is 1.03 to 4.1 times faster than the paths below the unit root (median 1.8). Above the unit
root the paths are 1.3 to 2.3 times faster than the line integral (and than the classical engine for the intercept and
the trend) while their outer hyperbola keeps its angle of 45 degrees: for the intercept up to c = 1.5, for the trends,
the trend detrended first and the intercept with a sine and a cosine up to c = 2 or 2.5, for the quadratic trend up to
c = 3; beyond, as the hyperbola steepens, the line integral is the faster again (1.5 times for the intercept at c = 2, 2
to 9 times for every design at c = 3 except the quadratic trend), and without deterministic terms it stays the faster
from c = 1 on. With c0 != 0 the line integral is 4 to 10 times slower, hence the paths there, except with c0 > 0 on the
vertical line: there the parabola X(v) of a point x < 0 has two positive roots, whose parts the line's rule does not
resolve, and the paths' one evaluation was off by up to 1e-3 in the lower tail (an intercept with a sine and a cosine at
c = 5, c0 = 2), where the line integral is accurate. A case string or a classical design with `engine = :general`
therefore takes the general engine's line integral at c0 = 0 below the unit root (as the general column of the table
below, except case DG, which the line integral does not cover) and its paths above it as described; with `method =
:paths` it takes the paths.

| cell | engine | classical (ms) | general (ms) | largest difference | why |
|---|---|---|---|---|---|
| t-ratio, c0 = 0, zero or stationary start below the unit root, cases DK, DU, DT, SK, SU, ST, DG; above it case DK, case DU for c > 1.5, case DT for c > 2 | classical | 2.8 to 52 | 2.7 to 80 | 1.7e-13 | in case DG 11 to 12 times faster (the general engine has no line integral with GLS detrending); elsewhere below the unit root the general engine's line integral is 1.0 to 1.24 times faster, below the threshold of 1.3 |
| t-ratio, c0 = 0, zero start, case DU for 0 < c <= 1.5, case DT for 0 < c <= 2 | general (its paths) | | | 1.4e-11 | 1.6 to 2.3 times faster (November 26, 2026) |
| zero start, c0 < 0, case DK for -5 < c <= 2 | general | | | 2.1e-10 | 5.6 to 28 times faster (the classical engine's line integral with its general kernel) |
| zero start, c0 < 0, case DT for -50 <= c <= -10 | general | | | 8.6e-12 | 1.3 to 2.7 times faster, at x = 0 too |
| zero start, c0 < 0 otherwise (c >= -100) | classical | 7.7 to 55 | 13 to 46 | 7.4e-10 (c = -100) | in case DK from c = -5 on (its conditional integral) 1.05 to 2.6 times faster; in case DU and in case DT nearer the unit root the gains of either engine vary with c and c0 |
| zero start, c0 > 0, c <= 1 (case DT: c <= 2) | general | 60 to 365 | 12 to 43 | 9.5e-11 | 2.4 to 9.5 times faster (at c = -100, which the rule gave the classical engine until November 26, 2026, 2.4 to 5 times; in case DT at c = 1.5, 2 3.6 to 7 times) |
| the same, case DK, within 1e-3 of the tangency x* = -sqrt(2 c0) above it | classical | | | | the density is infinite at x*; just above it the general engine's density is off by up to 1.5e-3 relative (x* + 1e-9), the classical by 5e-10; below x* the general engine is the right one (the classical density off by 10% at x* - 1e-9 at c = -20, c0 = 0.5) |
| zero start, c0 > 0 otherwise (c > 1, case DT c > 2) | classical | 15 to 372 | 31 to 89 | 3.5e-11 | 2.1 times faster in case DU at c = 2 |
| zero start, c0 != 0, c < -100 | general | 196 to 495 | 36 to 74 | 2.6e-11 | 5.4 to 6.8 times faster |
| stationary start, c0 != 0, c <= -3e-3 | general | 102 to 8732 | 18 to 56 | 1.6e-10 (case SK at c = -0.1, the classical engine's one evaluation) | 4.3 to 230 times faster (case SK at c = c0 = -0.1: 8.7 s against 38 ms); values below 1e-3 go back to the classical conditional integral, whose error is relative |
| stationary start, c0 != 0, c > -3e-3 | classical | | | 8.1e-10 (c = -1e-3) | next to zero the general engine is off by 1e-10 to 7e-8 in cases SU, ST |
| above the unit root, c0 < 0 (c > 2) | classical | 43 to 420 | 114 to 465 | 2.2e-13 | 4.8 to 11 times faster in cases DU, DT (c = 5) |
| above the unit root, the bulk at c >= 7 | classical | 199 to 672 | 380 to 21195 | 1.4e-13 | 1.1 to 32 times faster |
| fixed start, case DK, c <= 1, (c g0)^2/2 <= 400 or \|g0\| <= 1.5 with \|c g0\| <= 50 and c >= -100, except c0 < 0 with c > -1 | general | 78 to 277 | 28 to 78 | 3.7e-11 | 2.0 to 5.8 times faster; beyond (c g0)^2/2 <= 400 1.4 to 6 times (November 26, 2026; the rule tested g0 <= 0.5 with c >= -100 there and did not take the absolute value of g0) |
| the same with c0 < 0 and c > -1 | classical | | | | the general engine's one evaluation of before November 2026 was off in the upper tail by up to 2.7e-6 at the 99.99% point and 2.9e-2 beyond (c = 0.5, c0 = -5 and -10, g0 = 1), the classical engine within 1e-12 of its checked computation and the conditional integral |
| fixed start, cases DU, DT, c < -10: c0 > 0 below c = -20 or where \|c g0\| > 20, any c0 below c = -20 where \|c g0\| > 60 | general | | | 1e-10 in F; the density the derivative of F to 1e-10 of the peak | the classical contour integral's one evaluation was off (November 2026): in the density by up to 2.2e-2 of the peak (case DU, c = -60, g0 = 1.5, c0 = 5), in F by 3.9e-5 (c0 = -2); its checked computation in the density by up to 8.4e-6; the general engine 0.23 to 0.9 times the time down to c = -45, up to 5 times below c = -75 |
| fixed start, case DK otherwise, cases DU, DT otherwise | classical | 35 to 215 | 40 to 127 | 5.4e-11 | 1.15 to 1.2 times faster in cases DU, DT and in case DK at c = 2, and its warnings of the brittle regimes; with \|g0\| = 2 or \|c g0\| > 50 in case DK the gains vary |
| moments (`method = :engine`), zero or fixed start, c0 = 0 | general | 13 to 195 | 9.2 to 14 | 4.2e-14 relative | 1.4 to 15 times faster |
| moments, stationary start, c0 = 0: case SK at c <= -3e-3, cases SU, ST at c <= -1 | general | 8.1 to 21 | 9.3 to 15 | 1.7e-13 relative | 1.2 to 1.3 times faster in cases SU, ST at c = -5; in case SK the classical engine is 1.2 times faster, below the threshold |
| moments, c0 != 0: stationary start for c <= -5, case SU, zero start in cases DK, DU, and in case DT for c < -5 | classical | 9 to 79 | 30 to 247 | 1.6e-9 relative (c = c0 = -300) | 1.8 to 10 times faster (November 26, 2026; far from the unit root neither engine is converged: both move by 1e-8 to 5e-7 relative under finer rules) |
| moments, c0 != 0 nearer the unit root: stationary start for -5 < c <= -3e-3 (cases SK; ST from c = -1 on), case DT for c >= -5 | general | | | | in case ST at c = c0 = -1 25 times faster, in case DT at c = 0, c0 = -2 2.8 times; next to zero the general engine's moments lose accuracy in cases SU, ST (1e-10 at c = -0.1 to 4e-7 at c = -3e-3), the classical engine there |
| quantiles at c0 = 0: a fixed start, case DT for c <= 2, case DU for 0 <= c <= 1.5, cases SU, ST for c <= -0.5 | general | | | 1.0e-10 | 1.1 to 6.3 times faster (one path, F and f in one pass); above the unit root by the paths 2.3 to 2.5 times; with a stationary start 1.3 to 1.7 times (next to zero the general search is off, by 2.6e-7 in case ST at c = -1e-3) |
| quantiles otherwise | classical | 13 to 101 | 11 to 192 | 5.0e-12 | 7.3 times faster in case DG; in cases DK, DU (below the unit root), SK the gains of the general search vary from 1.1 to 1.6; each evaluation of the search routed, and where the rule gives the point to the general engine, F and f from one pass of its paths (`route_dist2`) |
| the coefficient statistic (`method = :engine`) | as the t-ratio at x = 0 with the null value c0 + y, except with a zero start (cases DK, DU, DT, -100 < c <= 2): c0 + y > 0 classical, c0 + y < 0 in case DK for c > -5 general | 5.9 to 85 | 8.1 to 76 | 3.9e-12 (7.6e-11 for the quantile) | one point per call at x = 0: with c0 + y > 0 the classical line integral 2 to 12 times faster, with c0 + y < 0 in case DK for c > -5 the general engine 1.4 to 2.7 times; in case DT for -50 <= c <= -10 the general engine 1.6 to 3.1 times |
| P(V <= b) (`ltratio_lincdf` with a = NaN), case DK | general | 5.5 to 320 | 0.03 to 0.17 | 4.8e-15 below c = 4; above, the classical inversion is off by 5.9e-10 (c = 4) to 1.8e-7 (c = 7) | 34 to 4000 times faster, and above the unit root the more accurate |
| P(U - aV <= b), case DK | classical | 5.4 to 20 | 5.3 to 276 | 9.6e-13 | up to 29 times faster (above the unit root 11 to 29 times) |
| P(U - aV <= b), case DG, a != 0, c > 0 with cbar <= -1, and every c > 4 | general | | | 7.1e-12 | 2.2 times faster above the unit root (cbar = -1, c = 2); beyond c = 4 the classical code refuses |
| P(U - aV <= b), case DG otherwise (at and below the unit root at every cbar; a = 0) | classical | 14 to 25 | 22 to 155 | 7.7e-12 | 2 to 11.5 times faster below the unit root (November 26, 2026; the general engine took cbar <= -2 there until then); the closed form at a = 0 |
| the power envelope, case DK at c <= -0.01 | general | 79 to 170 | 84 to 111 | 2.8e-12 | the critical value to 1e-10 (the classical engine bisects it to 1e-9); 1.07 times slower to 1.5 times faster; by default the single inversion of Bao and Kan computes |
| the power envelope, case DG, and case DK nearer zero | classical | | | 2.1e-9 | in case DG 4.7 to 21 times faster at c = -2 to -20 (November 26, 2026; the general engine took c <= -1 until then); case DK at c = -1e-4: the general engine is off by 1.1e-9 |
| ADFmax, stationary start, c <= -0.1 | general | 112 to 197 | 50 to 59 | 3e-12 | 2.3 to 3.4 times faster; next to zero (c = -1e-4) it is off by 2e-9 |
| ADFmax, zero start | classical | 22, 33 | 50, 53 | 2.6e-12 | 1.6 to 2.3 times faster |
| regions (`ltratio_region`): the parabola of the t-ratio at c0 = 0, c <= 0 | general | | | 1.2e-13 | 2.5 to 27 times faster (its line integral; November 26, 2026) |
| regions otherwise: the line, the parabola with c0 != 0 or above the unit root, the WAP boundary and a function g | classical | | | 1e-12 | 3.1 times faster for the line, 1.3 to 2.6 times for the parabola with c0 != 0; the classical engine is the only one for the WAP boundary and a function g |
| above the range of the general engine (c > 16 with deterministic terms, c > 10 with them and a fixed start, c > 20 without) | classical | | | | the general engine refuses there |
| every design beyond the classical cases or beyond the classical engine's checks | general | | | | the only engine |

Where the classical engine computes, its values are those of 0.5.0 bit for bit; where the rule takes the general engine
for a case string, the values move within the differences above (in the cells moved on November 26, 2026 within 1e-7). The borders were swept: the stationary start at c =
-3e-3 (cases SU, ST: the general engine within 4e-11 of the classical at c = -3e-3, off by 8e-10 at c = -1e-3), the
moments next to zero (case SK down to c = -3e-3, cases SU, ST down to c = -1), the power envelope (case DK within 3e-12
of the classical down to c = -0.01) and the tangency of case DK (the window 1e-3 above x*). The tests follow the rule
(`test/runtests.jl`, `test/engine`): one stored value changed, the power envelope of case DK at c = -7, now
0.49940198281562 (it was 0.49940198260314117, the classical engine's bisection of the critical value to 1e-9; the same
construction with the critical value bisected to 1e-14 is within 8e-13 of the new value), and the checks of the
classical engine's own computations (ADFmax with a stationary start, the moments, quantiles and default values of its
test grid far from the unit root, the power envelope) force `engine = :classical`, so that they keep their stored
values. Three defects of the general engine are repaired in `src/DFDet.jl` (see NEWS): the law of V on the vertical
line (off by 2e-3 at c = 5, and with GLS detrending at small |cbar| by up to 1.1e-3 in the upper tail), the law of U at
the edge of its support (4.4e-7), and the parts at the root of the
parabola on the vertical line, which now go to Filon's rule (the linear form was off by up to 4e-3 above the unit root
and by 1e-4 with GLS detrending at small |cbar|, the t-ratio with c0 < 0 by 8.5e-5 and its density by 1e-2). On the
line ADFmax and parabolas with two roots (c0 > 0 with x < 0) keep the Gauss-Legendre rule and its warning.

Above the unit root the t-ratio with c0 != c has its upper tail at x of the order (c - c0) e^c/(2c). There the line
integral's inner rule exceeds the bounds of the engine (from x ~ 700), and in case DK the classical engine takes the
conditional integral with its checks whatever its flag (its one evaluation is off by up to 14 there, and before
November 23, 2026 the default route returned it where the checked value was flagged: values outside [0, 1]); for c > 4 the
three cases take the conditional integral at x > 300 too, where the line integral's cost grows like x^4 (in cases DU, DT
at c = 16 it exhausted the memory at the median), and case DK with c0 > c at x < -20, where the line integral is off by up
to 1.5e-3. Against an independent computation (the law of V given J(1), `scratch/fut/bridge_ref.jl`) the values of case DK
are within 1.3e-8 up to c = 14, those of cases DU, DT within 5e-7 of the large-c law at c = 13 and 14; above c = 14 the
classical engine refuses such points with an `ArgumentError` (the conditional integral is off by 7.7e-5 in case DK at
c = 16 and gives 1e44 in case DT, and the general engine's line exceeds its bounds). The general engine cuts its line to
the oscillation with c0 (it was off by 1.1e-3 at c = 10, c0 = 9). Every distribution function (`ltratio_cdf`, `coef_cdf`,
`dist_cdf`, `DFDet.coef_cdf`) ends with a check of its values: below -1e-8 or above 1 + 1e-8 an error that names the call.
