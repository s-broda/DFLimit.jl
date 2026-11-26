# Which engine

`dev/engine_table.jl` (`dev/engine_table.txt`) times both engines in each cell (4 threads, the warm time of one call,
BenchmarkTools minimum of 2; F and f at x = mean + (-2, -1, 0, 1, 2) sd for the t-ratio) and compares their values; where
they differ by more than 1e-8 it compares each with the other's checked computation (`verified = true`). The rule,
`autoengine` in `src/merged.jl`, was set to take the faster engine at the same accuracy, and the right one where they
disagree. The times in the table below are those of the run of November 26, 2026 (version 0.5.0, the thinner rules of
one evaluation included, `dev/engine_table.txt`; an M4 laptop on battery power, about five minutes for the whole
table): the general engine's paths take about half the time they took when the rule was set, so that in some cells the
rule now takes the slower engine, which the last column says ("now"); the rule is unchanged in this release.

Within the general engine, `dist_cdf`, `dist_pdf` and `dist_quantile` (and the classical functions with a design, through
them) choose between two representations by the keyword `method`: `:auto` (the default) takes the line integral with
the explicit kernel where it covers the design and c0 = 0, and the inversion on paths around the roots otherwise;
`:line` forces the line integral (an `ArgumentError` with the reason where it does not cover the design, the start or
the options) and `:paths` the inversion on paths. The line integral covers the designs whose numerator has rank one or
two: no deterministic terms, the continuous designs that do not vanish at the origin and whose span is closed under
differentiation (polynomial trends of any order, an intercept with sines and cosines), and `detrend_first` of a
continuous design that does not vanish at the origin, with a zero or a stationary start and least squares detrending
(not GLS detrending, a fixed start or a design with a jump). It is the classical engine's integrator with P(t) and the
two roots supplied by the general engine's transform (`src/DFDet_line.jl`). On warm calls at c0 = 0 (November 26,
2026, after the thinner rules of the paths; F and f at five points, four threads) it is 1.03 to 4.1 times faster than
the paths below the unit root (median 1.8; 2.8 to 4.1 times without deterministic terms and a zero start, 1.05 to 1.3
times at c = -50 with deterministic terms; no deterministic terms, an intercept, the linear and the quadratic trend and
the trend detrended first, zero and stationary starts, c = 0, -10, -50) and 1.06 to 1.23 times faster than the classical
engine for the intercept and the trend; above the unit root, at c = 2, it is 1.3 to 3.8 times faster without
deterministic terms and with an intercept but 1.6 to 1.8 times slower than the paths for the linear and the quadratic
trend and the trend detrended first (the rule takes it there all the same); with c0 != 0 it is 4 to 10 times slower,
hence the rule. A case string or a classical design with `engine = :general` at c0 = 0 therefore takes the general
engine's line integral (as the general column of the table below at c0 = 0, except case DG, which the line integral does
not cover); with `method = :paths` it takes the paths.

| cell | engine | classical (ms) | general (ms) | largest difference | why |
|---|---|---|---|---|---|
| t-ratio, c0 = 0, zero or stationary start (any c), cases DK, DU, DT, SK, SU, ST, DG | classical | 2.8 to 52 | 2.7 to 80 | 1.7e-13 | in case DG 11 to 12 times faster (the general engine has no line integral with GLS detrending) and above the unit root 1.5 times (case DT, c = 3, 5); the general engine's line integral is now 1.04 to 1.23 times faster in the other cells below the unit root (c from 0 to -300; the classical 1.24 times faster in case DT at c = -300) |
| zero start, c0 < 0, -100 <= c <= 0 | classical | 7.7 to 55 | 13 to 46 | 7.4e-10 (c = -100; the classical engine's one evaluation, the general engine within 1.4e-12 of the classical checked value since its rules far from the unit root start at c = -60) | 1.25 to 1.7 times faster in case DK (1.1 times in case DU at c = -50); in cases DU, DT at c = -10, -5, -100 the general engine is now 1.6 to 2.2 times faster |
| zero start, c0 > 0, -100 < c <= 1 | general | 60 to 365 | 12 to 43 | 9.5e-11 | 3.1 to 9.5 times faster |
| the same, case DK, within 1e-3 of the tangency x* = -sqrt(2 c0) above it | classical | | | | the density is infinite at x*; just above it the general engine's density is off by up to 1.5e-3 relative (x* + 1e-9), the classical by 5e-10; below x* the general engine is the right one (the classical density off by 10% at x* - 1e-9 at c = -20, c0 = 0.5) |
| zero start, c0 > 0 otherwise (c <= -100 or c > 1) | classical | 15 to 372 | 31 to 89 | 3.5e-11 | 2.1 times faster in case DU at c = 2; the general engine is now 1.3 times faster in case DK at c = 2 (c0 = 2) and 4.2 times in case DU at c = -100 (c0 = 3) |
| zero start, c0 != 0, c < -100 | general | 196 to 495 | 36 to 74 | 2.6e-11 | 5.4 to 6.8 times faster |
| stationary start, c0 != 0, c <= -3e-3 | general | 102 to 8732 | 18 to 56 | 1.6e-10 (case SK at c = -0.1, the classical engine's one evaluation) | 4.3 to 230 times faster (case SK at c = c0 = -0.1: 8.7 s against 38 ms); values below 1e-3 go back to the classical conditional integral, whose error is relative |
| stationary start, c0 != 0, c > -3e-3 | classical | | | 8.1e-10 (c = -1e-3) | next to zero the general engine is off by 1e-10 to 7e-8 in cases SU, ST |
| above the unit root, c0 < 0 (c >= 4) | classical | 43 to 420 | 114 to 465 | 2.2e-13 | 4.8 to 11 times faster in cases DU, DT (c = 5); in case DK the general engine, within 2.2e-13 since its repair (Filon's rule on the vertical line; before, off by 8.5e-5 in F and 1e-2 in f), is 1.2 times faster at c = 5, c0 = -3 and 3.4 times at c = 4, c0 = -0.5 |
| above the unit root, the bulk at c >= 7 | classical | 199 to 672 | 380 to 21195 | 1.4e-13 | 1.1 to 32 times faster |
| fixed start, case DK, c <= 1, (c g0)^2/2 <= 400 or g0 <= 0.5 with c >= -100, except c0 < 0 with c > -1 | general | 78 to 277 | 28 to 78 | 3.7e-11 | 2.0 to 5.8 times faster |
| the same with c0 < 0 and c > -1 | classical | | | | the general engine's one evaluation of before November 2026 was off in the upper tail by up to 2.7e-6 at the 99.99% point and 2.9e-2 beyond (c = 0.5, c0 = -5 and -10, g0 = 1), the classical engine within 1e-12 of its checked computation and the conditional integral |
| fixed start, cases DU, DT, c < -10: c0 > 0 below c = -20 or where \|c g0\| > 20, any c0 below c = -20 where \|c g0\| > 60 | general | | | 1e-10 in F; the density the derivative of F to 1e-10 of the peak | the classical contour integral's one evaluation was off (November 2026): in the density by up to 2.2e-2 of the peak (case DU, c = -60, g0 = 1.5, c0 = 5), in F by 3.9e-5 (c0 = -2); its checked computation in the density by up to 8.4e-6; the general engine 0.23 to 0.9 times the time down to c = -45, up to 5 times below c = -75 |
| fixed start, case DK otherwise, cases DU, DT otherwise | classical | 35 to 215 | 40 to 127 | 5.4e-11 | 1.15 to 1.2 times faster in cases DU, DT and in case DK at c = 2, and its warnings of the brittle regimes; in case DK at c = -20 and -100 the general engine is now 1.35 and 1.7 times faster |
| moments, zero or fixed start, c0 = 0 (c from -480 to 0) | general | 13 to 195 | 9.2 to 14 | 4.2e-14 relative | 1.4 to 15 times faster |
| moments, stationary start: case SK at c <= -3e-3, cases SU (c0 = 0 only), ST at c <= -1 | general | 8.1 to 21 | 9.3 to 225 | 1.6e-9 relative (c = c0 = -300) | 1.2 to 1.3 times faster in cases SU, ST at c = -5; the classical engine is now 1.1 times faster in case SK at c = -5 and 5.6 to 10.5 times with c0 = c far from the unit root (c = -300, -200); there neither engine is converged (both move by 1e-8 to 5e-7 relative under finer rules) |
| moments, stationary start: case SU with c0 != 0, and nearer zero | classical | 19 | 115 | 2e-12 relative | 6 times faster in case SU; next to zero the general engine's moments lose accuracy in cases SU, ST (1e-10 at c = -0.1 to 4e-7 at c = -3e-3, where they fail) |
| moments, zero start, c0 != 0 | classical in cases DK, DU, general in case DT | 16 to 79 | 45 to 247 | 1.8e-10 relative | 2.4 to 4.1 times faster in cases DK, DU; in case DT the classical engine is now 3.1 times faster |
| quantiles at c0 = 0: a fixed start, cases DT (c <= 0), ST (c <= -1), DU at c = 0 | general | 39 to 459 | 30 to 133 | 1.0e-10 | 1.1 to 6.3 times faster (one path, F and f in one pass) |
| quantiles otherwise | classical | 13 to 101 | 11 to 192 | 5.0e-12 | 1.2 times faster in case DT at c = 3 and 7.3 times in case DG; at c0 = 0 in cases DK, DU, SU the general engine is now 1.2 to 1.4 times faster; each evaluation of the search routed, and where the rule gives the point to the general engine, F and f from one pass of its paths (`route_dist2`, November 2026: four quantiles 0.54 to 0.77 times as long, median 0.59, the quantiles bit for bit) |
| the coefficient statistic | as the t-ratio at x = 0 with the null value c0 + y, except with a zero start (cases DK, DU, DT, -100 < c <= 1): c0 + y > 0 classical, c0 + y < 0 in case DK for c > -5 general | 5.9 to 85 | 8.1 to 76 | 3.9e-12 (7.6e-11 for the quantile) | (case SU at c = -5: 10.5 times faster by the general engine; in case DU at c = 0 the classical engine 2 to 2.9 times faster, in case DT at c = -10 now the general engine 2.2 times); one point per call at x = 0: with c0 + y > 0 the classical line integral 2 to 12 times faster, with c0 + y < 0 in case DK for c > -5 the general engine 1.4 to 2.7 times (c from 1 to -80, y at the 1 to 99% quantiles) |
| P(V <= b) (`ltratio_lincdf` with a = NaN), case DK | general | 5.5 to 320 | 0.03 to 0.17 | 4.8e-15 below c = 4; above, the classical inversion is off by 5.9e-10 (c = 4) to 1.8e-7 (c = 7) | 34 to 4000 times faster, and above the unit root the more accurate (the general engine within 4.4e-12 of the closed form of M_V inverted on hyperbolas of its own) |
| P(U - aV <= b), case DK | classical | 5.4 to 20 | 5.3 to 276 | 9.6e-13 | up to 29 times faster (above the unit root 11 to 29 times); the general engine 1.04 times faster at a = 2, c = -5 and 1.1 times at a = 0, c = 0 |
| P(U - aV <= b), case DG, a != 0, cbar <= -2 (cbar <= -1 above the unit root), and every c > 4 | general | 14 to 241 | 22 to 110 | 7.1e-12 | 2.2 times faster above the unit root (cbar = -1, c = 2); below it the classical engine is now 1.6 to 2.8 times faster (cbar = -13.5, -5, -2); beyond c = 4 the classical code refuses |
| P(U - aV <= b), case DG otherwise | classical | 15 to 25 | 30 to 100 | 3.2e-11 | the closed form at a = 0 (1.2 times faster; the general engine within 7.2e-14 of it); at cbar = -1 below the unit root 6.9 times faster |
| the power envelope, case DK at c <= -0.01, case DG at c <= -1 | general | 79 to 1555 | 84 to 684 | 2.1e-9 | the critical value to 1e-10 (the classical engine bisects it to 1e-9; in case DG the general engine is within 1.4e-12 of a bisection to 1e-14); 1.07 times slower to 2.3 times faster in case DK and at c = -1, -2 in case DG; at c = -5 to -20 in case DG the classical engine is now 4.1 times faster |
| the power envelope nearer zero | classical | | | | case DK at c = -1e-4: the general engine is off by 1.1e-9 (the critical value at the edge of the support of U - aV); case DG from c = -0.5 on the general engine is the slower, and at c = -0.01 it exhausts the memory |
| ADFmax, stationary start, c <= -0.1 | general | 112 to 197 | 50 to 59 | 3e-12 | 2.3 to 3.4 times faster; next to zero (c = -1e-4) it is off by 2e-9 |
| ADFmax, zero start | classical | 22, 33 | 50, 53 | 2.6e-12 | 1.6 to 2.3 times faster |
| regions (`ltratio_region`) | classical | 1.1, 1.4 | 3.3, 0.41 | 1e-12 | 3.1 times faster for the line; for the parabola of the t-ratio the general engine is now 3.5 times faster; the classical engine is the only one for the WAP boundary and a function g |
| above the range of the general engine (c > 16 with deterministic terms, c > 10 with them and a fixed start, c > 20 without) | classical | | | | the general engine refuses there |
| every design beyond the classical cases or beyond the classical engine's checks | general | | | | the only engine |

Where the classical engine computes, its values are those of 0.5.0 bit for bit; where the rule takes the general engine
for a case string, the values move within the differences above. The borders were swept: the stationary start at c =
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
