# The devices of Bao and Kan

The half-plane statistics of the six classical cases, the coefficient statistic (`coef_cdf`, `coef_pdf`, `coef_icdf`,
`coef_mom`), the linear form U - aV (`ltratio_lincdf`) and the power envelope (`power_envelope`), and the moments of
the t-ratio (`ltratio_mom`) are computed by default with the numerical devices of Yong Bao and Raymond Kan, "Exact
Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1)
Algorithms", ported to the limits of this package in the module `DFLimit.DFKan` (`src/DFKan*.jl`). The joint
characteristic function of the limit pair comes in closed forms that are free of cancellation on both sides of the
unit root, with the branch of the square root fixed at each point from factors whose arguments are known, and in
Taylor-mode arithmetic (jets), so that its derivatives come out of the same assembly. A statistic that rejects on a
half-plane is then one inversion along a ray, by adaptive quadrature over decades with the tail beyond S = 400/|w| (w
the slope of the dominant phase) completed by repeated integration by parts, the derivatives of the slowly varying
factor read off a jet of order eight. The moments of the t-ratio are one-dimensional integrals over s of the j-th
derivative of the moment generating function along its affine path, which one jet of order j gives at each node of an
adaptive quadrature in log s (`method = :jets`; before, Cauchy integrals on a circle); the quantile searches of the
t-ratio (`ltratio_icdf`, also where the rule gives them to the general engine) start from the normal approximation
with these moments (`DFLimit.mom_start`; before, the engines' circles; `DFLimit.MOM_START[] = false` restores that).
Every value comes with an error estimate: `err = true` returns `(values, errors)`, `tol` (1e-12) is the absolute
tolerance, and a warning names the values whose estimate exceeds it. `method = :engine` is the computation of before
(the engines, by the rule of `autoengine`; the t-ratio at x = 0 for the coefficient statistic), which reproduces those
values, and engine = :classical or :general, the engines' own keywords and case DG take it as well; `method =
:inversion` forces the single inversion; `verified = true` recomputes by the engines' checked computation the values
whose estimate exceeds the tolerance.

| function | the single inversion by default | largest difference from the engines | which is right where they differ | warm time (ms), single inversion against `method = :engine` |
|---|---|---|---|---|
| `coef_cdf`, `coef_icdf` | the six cases with zero, fixed and stationary starts, -550 <= c <= 20 | 2.4e-11 in cases SK, SU, ST; in cases DK, DU, DT 3.1e-10 (c = -550), 1.8e-9 (case DU, c = -1e-3, the 1% point) and 4e-9 at c = 20, where the law lies within 1e-6 of y = c and the rounding of y moves F (1,836 points at nine quantiles; at each the single inversion with the tolerances 1e-10 and 1e-12 agrees to 1e-15) | case DU, c = -60, g0 = 1.5: the engines' one evaluation off by up to 1.5e-6, the single inversion within 4.3e-12 of their checked computation | 0.03 to 1 against 0.2 to 11 (up to 420 times faster, median 24; about as fast, 0.7 to 1.1 times, in the upper tail near the unit root); quantiles 0.3 to 2.8 against 18 to 320 |
| `coef_pdf`, `coef_mom` | the same cases (new functions) | the density within 2.1e-13 of the general engine's (`DFDet.coef_pdf`); the first two moments within 1.3e-12 of the integrals of the density | | 0.04 to 1.6 |
| `ltratio_lincdf` | case DK with a zero start for -100 <= c <= 4, the six cases through their designs (fixed start included) for -100 <= c <= 2, a != 0 | 5.5e-11 against the classical code (2.2e-11 at the tolerance 1e-12), 1e-12 against the general engine | with a fixed start above the unit root the general engine's one evaluation is off by up to 6e-2 (case DK, c = 1, a = -7, the 99% point) and 2.9e-3 (case DU, c = 2): the single inversion agrees with its checked computation and with the plain quadrature without the tail to 1e-14 | 0.05 to 1.1 against 3.3 to 7.1 (6 to 100 times faster) |
| `ltratio_mom` (jets) | the six cases, any c0 and fixed start, c <= 20 | 1.4e-11 relative near the unit root (8.2e-12 to 1.4e-11 at c = 0, -3, -5), 3.2e-10 against the stored values far from it at c0 = 0, 1e-8 to 1.3e-7 with c0 = c far from it (2e-7 at c = c0 = -480 with a stationary start) | with c0 = c far from the unit root the engines' circles are not converged: they move towards the jets under refinement, and at c = c0 = -60 the integral of x^j times the density is within 1.5e-9 of the jets and 5.2e-9 of the circles; with a fixed start above the unit root the engines' one evaluation is off by up to 0.65 relative (case DK, c = 2, c0 = -2, g0 = 1), and their circles with four times the nodes and a quarter of the step agree with the jets to 3.4e-13; at c = 10, 20 with c0 = c by up to 33 relative (case DT) | 0.3 to 1.2 against 8 to 45 for four moments (11 to 100 times faster) |
| `ltratio_icdf` (the start) | the six cases with zero, fixed and stationary starts, c <= 20 | 2.4e-11 (the search tolerance 1e-10), except case DK with c0 = 2 far from the unit root: 4.3e-10 at c = -20 and 5.5e-9 at c = -60, where the distribution function the search inverts is known to about 1e-9 | | whole calls (four quantiles) 0.32 to 0.93 times as long (median 0.74; case DK at c = 0: 47 to 15 ms) |
| `power_envelope` | case DK (DU), c <= -1e-4 | 2.7e-12 against the general engine (critical value to 1e-10) from c = -1e-4 to -500 | the classical bisection of the critical value to 1e-9 is off by up to 1.1e-9 | 1.3 to 13 against 25 to 123 (9 to 57 times faster) |

The general engine has the same inversion for the statistics of a design whose transform it evaluates point by point:
the point-optimal statistic (`pt_cdf`, `pt_pdf`, `pt_quantile`, `pt_power`, `pt_envelope`, and through them
`power_envelope` for designs beyond the classical cases) and `coef_cdf_gp` (the coefficient statistic by Gil-Pelaez, the
independent check of the contour integral), with the branch of the square root of the determinant fixed at each point
from the pivots of the precision matrix, Bao and Kan's lemma on accretive matrices (`DFDet.DetCF`, `src/DFKan_det.jl`),
in place of its continuation along the fixed grid of 60,000 points. `method = :grid` (or the keywords N, umax) gives the
grid of before with its refusals. The point-optimal statistic agrees with the contour integral of the general engine on
its quadratic form (the paths around the roots, an independent evaluation) to 3.3e-12 from c = -30 to 4 (an intercept at
cbar = -7, a linear trend at -13.5, Perron's model B at -17.8, a quadratic trend at -20; the 1 to 99% points), where the
grid is off by up to 3.8e-10 at c = 3 and refuses at c = 4 and 5; the single inversion takes 1.5 to 11 ms a value against
46 to 460 ms for the grid. Above c = 4 it can take seconds a value (Perron's model B: 0.01 to 0.04 s at c = 5, 1.1 to 2.2 s at c = 6), and the grid with
its refusal stays the default there; the single inversion has no limit on q (the grid refused |q| > 1000). The quantile
searches keep the transform values along the ray (5% points 13 to 68 ms against 47 to 458, envelopes at three values of c 39 to 244 ms
against 302 to 2457). `coef_cdf_gp` agrees with the contour integral (`DFDet.coef_cdf`) to 9.6e-13 from c = -30 to 5 for polynomial
trends, Perron's model C with one and two breaks and a stationary start (2.5e-11 with GLS detrending, where the grid and the
contour integral differ by 1.6e-10), in 3 to 38 ms a value against 133 to 813 for the grid; above c = 5 (6.5 s a value at c =
10) and with a fixed start the grid stays.

The same devices fix the branches inside the engines: the classical engine's P(t), M_0(t) and the labels of the roots of
its line and contour integrals (`DFExact.tparts_point`) and the argument of log P(t) in the provider of the general
engine's line integral are determined at each point instead of being continued along the grids, to rounding (5.7e-14) the
values of before, at the same cost.

At a = 0 (the law of U alone, whose transform decays like a power, with the edge of its support where the tail start runs
away) the linear form stays with the engines' closed form; at c = 7 in case DK the classical code differs from the single
inversion by 2e-8 at the 99.9% point with a = 2, which no third computation settled, so the default there is unchanged.
Times are warm minima (7 timed calls after a warm-up, 3 from 1 s on) on four threads of an M4 laptop on mains power with
nothing else running (November 26, 2026), against `method = :engine` or the grid in the same session.
