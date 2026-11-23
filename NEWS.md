# News

## Unreleased

- **The upper tail of the fixed start near the unit root (general engine).** The placement of the U-shape G+ of a fixed
  start (`ufixed_place`) counted the growth e^{sigma^2 x^2/(4 Re(t + c0 s))} of the v-kernels of the upper tail to the
  left of the origin above the unit root only.  In case DK near it (c = 0, -0.5) G+ went left of the origin in the upper
  tail as well, and one evaluation was off by 3e-7 at x = 9.7 (g0 = 1, c0 = 0, where F is 1 to 1e-12), 7e-3 at x = 12
  and 1e39 at x = 20, with c0 = 2 by 55 at x = 4 and with c0 = -2 by 9e-5 at x = 20, without a warning below the range
  check; the default route takes the general engine for these arguments (`ltratio_cdf(9.732, "DK", 0; g0 = 1)` returned
  0.9999997).  The growth is now counted at every c (with the largest positive c0 in it): these points are within 1e-12 of
  the classical engine, the bulk is unchanged, and cases DU, DT (whose G+ stays right of the origin) are unchanged bit for
  bit.
- **Case DK with a fixed start and c0 < 0 near the unit root goes to the classical engine.** The rule `autoengine` took the
  general engine for the fixed start of case DK (c <= 1, (c g0)^2/2 <= 400) at every c0; with c0 < 0 near the unit root
  the outer hyperbola of the general engine does not resolve the upper tail, and its one evaluation was off by 2.7e-6 at
  the 99.99% point (c = 0.5, c0 = -5, g0 = 1), by 6.7e-6 at c = 0 with g0 = 1.5, and beyond the 99.99% point by up to
  2.9e-2 (c0 = -10), where the range check also refused values of 1.23 (`ltratio_cdf(45.11, "DK", 0.5; c0 = -10, g0 =
  1)` raised an error).  These points (c0 < 0, c > -1) now take the classical engine, whose line integral agrees with its
  checked computation and with the conditional integral to 1e-12 there.  The general engine itself takes a finer outer
  step there (see the rules of one evaluation below).
- **Thinner rules of one evaluation of the general engine's paths.** `dist_cdf`, `dist_pdf` and `dist_quantile` (and the
  classical functions where the rule takes the general engine) take the outer hyperbola with 1.5 times the step and the
  U-shapes with 16 nodes per panel and 32 on the tails of the rays (24 and 40 before; 32 and 48 for 48 and 64 with a
  fixed start), 0.50 times the time of one evaluation (median; 0.30 to 0.87), and near the unit root with c0 < 0 the
  present step (or a finer one) with the thin inner rule, by regions of the start, the deterministic terms, c and c0
  (`DFDet.path_region`, `DFDet.PATH_RULES`; the README, "The rules of one evaluation"), measured against a reference
  with half the step and twice the nodes to within max(1e-9, twice the present rule's own error) in F and max(1e-8,
  twice its own) of the largest density in f.  The values move by up to 7.6e-10 in F and 4e-9 of the peak in f (a dense
  grid of x, 736 configurations); beyond the range measured (c > 2 with a fixed start, c > 5, c < -60 with c0 other than
  0 and c) the rule is the present one.  With a fixed start, no deterministic terms, c0 < 0 and c > -1 the outer step is
  finer than before: the present one was off by up to 0.86 in the upper tail (c = 0.5, g0 = 1, c0 = -20), now within
  7e-12 of the classical engine, at 2 to 11 times the cost.  New keyword `rule` (:auto; :present the rule of before,
  also taken by `verified = true`; :thin, :inner, :outer, :fine, :finestep); `hfac` is `nothing` (by the rule) unless
  given.
- **The branches fixed at each point.** The classical engine's P(t), M_0(t) and the labels of the roots of the line and
  contour integrals (`DFExact.tparts`) and the argument of log P(t) in the provider of the general engine's line integral
  are fixed at each point, after Bao and Kan (the factors whose arguments are known; the pivots of the precision matrix,
  one small elimination per point in the general engine), in place of their continuation along the grids, which is
  removed with the switches of the prototypes. On the engines' grids the values are those of before to 5.7e-14 relative
  (whole calls of `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `dist_cdf`, `dist_pdf`, `dist_quantile` to 1.7e-14); on
  the hyperbolas of case DK at c = -480 the continuation gave non-finite values far out, the branch at the point does not.
  Warm whole calls on one thread take 0.94 to 1.01 times as long as before (the quantiles of cases DT, SK, ST 0.29 to 0.46
  times: a real point no longer builds the series of D; case DU at c = 2 1.25 times); the contour integral's stored
  value at the corner (c g0)^2/2 = 7200 of case DT moved by 5.9e-12 (to within 2e-12 of the general engine's checked
  value).
- **The point-optimal statistic and `coef_cdf_gp` by the single inversion.** `pt_cdf`, `pt_pdf`, `pt_quantile`, `pt_power`,
  `pt_envelope` (and `power_envelope` of designs beyond the classical cases) up to c = 4, and `coef_cdf_gp` (zero or
  stationary start) up to c = 5, take the single inversion of Bao and Kan with the branch of the square root of the
  determinant from the pivots of the precision matrix at each point (`DFDet.DetCF`, `src/DFKan_det.jl`) in place of the
  fixed grid of 60,000 points: 5 to 60 times faster (pt_cdf 1.2 to 65 ms against 77 to 640, quantiles and envelopes 13 to
  86 ms against 63 to 880), within 3.3e-12 of the contour integral on the statistic's quadratic form, and without the
  grid's refusals at c = 4 (an intercept, a linear trend) and for |q| > 1000. `method = :grid`, or N, umax, gives the grid
  of before (`coef_cdf_gp` with a fixed start keeps it); the grid's function of before is `DFDet.coef_cdf_grid`. The
  quantile searches keep the transform values along the ray across the search (`DFDet.MemoCF`). The values move by at
  most 1.8e-11 (the 5% point of the point-optimal statistic with an intercept, where the grid's Newton search stops)
  below the unit root and by up to 5e-10 above it (c = 3), where the grid is the less accurate one.
- **Thinner rules of one evaluation of the classical contour integral** (a fixed start in cases DU, DT;
  `DFExact.ctr_rule`): 32 nodes per panel and ray of the U-shapes for 64 and 1.5 times the outer step near the unit root
  (0.42 times the time, median), 32 nodes on the present step above the unit root (0.59 times), 64 nodes on the longer
  step with c0 > 0 below c = -2.5 and below c = -10, the present rule where |c g0| > 20; within 6.2e-11 of the checked
  computation in F on a dense grid of x (129 configurations), every configuration within max(1e-9, twice the present
  rule's own error) in F and max(1e-8, twice its own) of the peak in f. `tr_contour` takes the keyword `rule` (:auto,
  :present); `verified = true` is unchanged.
- **Thinner rules of one evaluation of the classical conditional integral** (`DFExact.cond_rule`): 48 nodes on the path
  of each inner inversion and of the integral over l for 96 with a stationary start for c <= -2, with a zero start in
  cases DU, DT for c >= -5 and in case DK for c <= -13.5, and 64 with a stationary start for c > -2 (0.60 times the time
  with a stationary start, 0.81 and 0.87 in cases DU, DT with a zero start; medians); every configuration of a dense
  grid of x (150: the six cases, c from 2 to -100, c0 in {2, -2, -5, c}) within the criterion against the rule with 192
  nodes and the outer tolerance 1e-14. A fixed start keeps 96 nodes. The options Nv, Nl of `tr_limdist` override the
  rule; `verified = true` is unchanged.
- **The fixed start of cases DU, DT far from the unit root goes to the general engine** (`autoengine`): with c0 > 0
  below c = -20 (below c = -10 where |c g0| > 20) and with any c0 below c = -20 where |c g0| > 60.  The classical
  contour integral's one evaluation was off there, in the density by up to 5e-6 of the peak (case DU, c = -30, g0 =
  0.5, c0 = 2), 4.2e-3 (c = -75, c0 = 5) and 2.2e-2 (g0 = 1.5, c = -60, c0 = 5), in the distribution function by up to
  3.9e-5 (case DU, g0 = 1.5, c = -60, c0 = -2) and 6.3e-6 (g0 = 0.9, c = -100); its checked computation by up to 8.4e-6
  of the peak in the density.  The general engine's density there is the derivative of the distribution function on
  which the two engines agree (to 1e-10 of the peak), at 0.23 to 0.9 times the time down to c = -45 and up to 5 times
  below c = -75.  The classical engine warns of the brittle one evaluation where |c g0| > 60 below c = -20 too.
- **The quantile searches start from the moments from jets.** `ltratio_icdf` (the classical Newton search, and the
  general engine's search where the rule gives it the quantiles) starts at the normal approximation with the first two
  moments from jets (`DFLimit.mom_start`, after Bao and Kan, 0.2 to 1 ms) in place of the circles of the engines (10 to
  150 ms): whole calls of four quantiles take 0.27 to 1.0 times as long (median 0.85; case DK at c = 0 13.8 ms for 46).
  The quantiles are those of before to 2.4e-11 (the search tolerance is 1e-10), except in case DK with c0 = 2 far from
  the unit root (4.3e-10 at c = -20, 5.5e-9 at c = -60), where the distribution function that the search inverts is
  itself known to about 1e-9. `DFLimit.MOM_START[] = false` restores the start of before.
- **The moments of the t-ratio from jets.** `ltratio_mom` takes the derivatives of the moment generating function from
  one Taylor-mode jet per node of an adaptive quadrature (after Bao and Kan) by default in the six classical cases, for any
  c0 and fixed start up to c = 20: 16 to 290 times faster (0.3 to 1.3 ms for four moments), within 1.4e-11 relative of
  before near the unit root. Far from the unit root with c0 = c the values move by 1e-8 to 2e-7 relative, where the circles
  of before were not converged; with a fixed start above the unit root (by up to 0.65 relative) and at c = 10, 20 with c0 = c
  the one evaluation of before was wrong. New keywords `method` (`:auto`, `:jets`, `:engine`, the circles of before),
  `tol`, `err`, `verified`.
- **The single inversion of Bao and Kan for the half-plane statistics.** The coefficient statistic (`coef_cdf`,
  `coef_icdf`), the linear form (`ltratio_lincdf`) and the power envelope (`power_envelope`) of the six classical cases are
  computed by default by one inversion of the closed-form joint characteristic function along the ray of the half-plane,
  with the tail completed by integration by parts from jets, the method of Bao and Kan ("Exact Finite-Sample Distribution
  and Moments of the Least Squares Estimator in First-Order Autoregressions: O(1) Algorithms"; the module `DFLimit.DFKan`),
  10 to 1600 times faster than before (see the README, "The devices of Bao and Kan"). New: `coef_pdf(y, cas, c)` (the density, also for designs) and `coef_mom(p, cas, c)` (the
  moments) of the coefficient statistic; the keywords `method` (`:auto`, `:inversion`, `:engine`), `tol` and `err` (the
  error estimates) of these functions. `method = :engine` reproduces the values of before. The values move by at most
  2.4e-11 to 1.8e-9 (the coefficient statistic; 4e-9 at c = 20, the rounding of y), 5.5e-11 (the linear form) and 2.7e-12
  against the general engine's envelope, except
  where the engines' one evaluation was wrong: the coefficient statistic of case DU with a fixed start far from the unit
  root (off by 1.5e-6 at c = -60, g0 = 1.5) and the linear form with a fixed start above the unit root (by up to 6e-2,
  through a design); the classical envelope's bisection to 1e-9 is replaced by a critical value to 1e-12. The linear form
  of the classical cases other than DK (through their designs) takes the single inversion too.
- **Two engines in one package.** The general engine of the paper on general deterministic terms (any regressors that
  are piecewise polynomial or smooth functions of r = t/n with known break dates; the inversion formula on paths around
  the singularities of the joint moment generating function) is now part of the package, as the submodule
  `DFLimit.DFDet` (the file of the repository df-deterministics), with its outer paths of the classical cases at the unit
  root (`data/paths`). New dependencies: StaticArrays, Random.
- **One interface.** The classical functions (`ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `ltratio_mom`, `coef_cdf`,
  `coef_icdf`, `adfmax_cdf`, `adfmax_pdf`, `adfmax_icdf`, `ltratio_lincdf`, `ltratio_region`, `power_envelope`) take a
  design in place of the case string: `trend(p)`, `perron(model, breaks)`, `fdesign(f)`, `ltrend(p)`,
  `design_from_matrix(X)`, `detrend_first(D)`, with the keywords `g0` (a fixed start), `start = :stationary` and
  `gls = cbar` (GLS detrending). The general engine's own functions (`dist_cdf`, `dist_pdf`, `dist_quantile`,
  `dist_moments`, `lin_cdf`, `v_cdf`, `region_prob`, `local_power`, ...) are exported too; its `coef_cdf`,
  `adfmax_cdf`, `adfmax_pdf` and `power_envelope` are `DFLimit.DFDet.coef_cdf` and so on.
- **The engine chosen automatically.** Every call, with a case string or a design, takes the engine of
  `DFLimit.autoengine` (`engine = :auto`), or the one forced by `engine = :classical` or `engine = :general`; the rule
  and its measurements are the table "Which engine" of the README. Where the classical engine keeps the computation its
  values are unchanged bit for bit, and its checks of the arguments (the range of c, the region of the fixed start) are
  those of 0.5.0 for case strings; a design beyond them goes to the general engine. Where the rule takes the general
  engine for a case string, the values move within the differences the table lists (at most 1.6e-9 relative in the
  moments far from the unit root, where neither engine is converged; 2.1e-9 in the power envelope, where the general
  engine's critical value is the more accurate: case DK at c = -7 gives 0.49940198281562 in place of
  0.49940198260314117). The linear form of 0.5.0's repair stays with the classical engine in case DK (except `P(V <=
  b)`, by the general engine for every c, 33 to 4400 times faster) and goes to the general engine in case DG for `a !=
  0` with `cbar <= -2` (`cbar <= -1` above the unit root), 1.4 to 80 times faster and within 7e-12, and beyond c = 4,
  where the classical code refuses.
- **Repairs of the general engine** (`src/DFDet.jl`, each in a commit of its own, to be copied back to df-deterministics).
  The law of V above the unit root (`v_cdf`, `v_pdf`): from c = 4 on the path of case DK is the vertical line, on
  which M_V(t) e^{-t y} oscillates faster than its panels resolve, and P(V <= b) was off by 2e-3; the law of V now takes
  a hyperbola of its own (`v_path`, the logarithms of the roots continued along it in `vsum`), within 1e-13 at c = 4, 5
  and 4.4e-12 at c = 7 of the closed form of M_V inverted on hyperbolas of its own (where the classical inversion is off
  by 5.9e-10 to 1.8e-7; with GLS detrending at small |cbar|, also on the line, P(V <= b) was off by up to 2.6e-6 at
  F = 0.99 and 1.1e-3 further out, and three other hyperbolas and a line with panels 16 times finer now agree with it to
  4e-11), and `ltratio_lincdf` with a = NaN goes to the general engine for every c. The law of U at the
  edge of its support (`tnode_specs!`): the rays of the kernels stopped at the roots, while their factor e^{-A s} decays
  only on the scale 1/|A|; P(U <= b) was off by 4.4e-7 at b = -0.4999 (P(U - aV <= b) at a = 0.1 by 1.9e-8); the rays
  now reach 1/|A|, within 1e-14 of the closed form down to b = -1/2 + 1e-7. Filon's rule on the vertical line (above
  the unit root, and with GLS detrending at small |cbar|): the kernels at a root b of the parabola carry e^{-t b^2},
  which the panels of the line did not resolve, and the linear form was off by up to 4e-3 in case DK at c = 4 to 7 (in
  case DG at cbar = -5 to -1 by 2e-5 to 1e-4), the t-ratio with c0 < 0 by up to 8.5e-5 and its density by up to 1e-2;
  those parts now go to Filon's rule (`filon_roots`,
  `filon_roots_x`, `contour_sum_filon`), and the line is cut to the oscillation of the factor e^{-A s_k(t)} of the
  kernels as well (`line_refine`); the linear form is within 1e-13 of the classical code at c = 4 and 5, the t-ratio
  within 1.4e-12 and its density within 2e-13 at c = 4 to 7 with c0 from -0.5 to -5. ADFmax and parabolas with two
  roots (c0 > 0 with x < 0) keep the line's Gauss-Legendre rule and its warning. A design whose roots run off along the
  outer contour, as (1 + r, 1{r <= 0.4}) on the vertical line at every c, needs U-shapes of millions of nodes: it is
  refused with an error before any inner integral is evaluated (`ushape_guard`, from the roots the outer path carries,
  at the nodes where the transform is not negligible), in 0.2 s and a few MB, where the refusal came after 3 to 4 s and
  0.8 GB (`USHAPE_MAXNODES`). The outer hyperbola refuses more than 1e5 nodes (`PATH_MAXNODES`: a factor `hfac` of its
  step far below one, or not positive) and `dist_moments` a grid of more than 1e6 points (c beyond 2.5e4).
  Non-finite arguments, as in the C engine and MATLAB: every function gives at +-Inf and NaN either the limit or an
  error that names the argument. `coef_cdf_gp` did not return at y = +-Inf (now 1 and 0 without the sum, NaN at NaN);
  the searches from a start value of +-Inf (`dist_quantile`, `coef_quantile`, `lin_quantile`, `adfmax_quantile`,
  `mza_quantile`) did not return or returned +-Inf, and a tolerance of +Inf stopped them after one step (now errors);
  `lin_quantile` and `adfmax_quantile` refuse p outside (0, 1) (at p = NaN they returned 1 and -1); the point-optimal
  statistic did not return at c = +-Inf (an error, also for umax = Inf and cbar = -Inf); a non-finite c gave "matrix
  contains Infs or NaNs", a non-finite c0 or gamma0 NaN or an InexactError (now "c must be a finite real scalar" and so
  on); the linear form at a = +-Inf gave NaN (an error; a = NaN is the law of V), the region below a line needs a
  finite slope; the law of V is 0 at y <= 0 and at -Inf, where it gave NaN; the envelopes (`DFDet.power_envelope`,
  `pt_envelope`) are NaN at c = NaN, where they gave the level, and 1 at c = -Inf, and refuse a level outside (0, 1);
  `local_power` checks its level and c0. The rules far from the unit root start at c = -60 (`FAR_CMIN`; they started at
  c = -100): on -100 <= c < -60 the outer contour was the 45-degree hyperbola, on which the terms grow far from the
  unit root, and the distribution function was off by 1e-11 at c = -70 and by up to 1.4e-9 at c = -100 against the
  line integral (cases DU, DT, SK, SU at c0 = 0, case DU at c0 = 2), the linear form of the envelope by up to 9e-10;
  with the angle of `fixed_phi` for every start and the caps 20/min(A, 40/|sigma|) of the segments from c = -60 on,
  and the contour's end at the larger of 1e5 and 20 c^2 for every c, within 5e-13 there (case DK 3e-11, as below c =
  -100), at 0.2 to 0.95 times the former time in cases DU, DT and 1.3 to 1.5 times in cases DK, SU; elsewhere the
  values are unchanged bit for bit.
- **The line integral for the general designs of rank one or two.** `dist_cdf`, `dist_pdf` and `dist_quantile` take
  `method = :auto` (the default), `:line` or `:paths`: the line integral with the explicit kernel where it covers the
  design and c0 = 0, the inversion on paths around the roots otherwise. The line integral is the classical engine's
  integrator with P(t) and the two roots supplied by the general engine's transform (`src/DFDet_line.jl`). It covers the
  designs whose numerator has rank one or two: no deterministic terms, the continuous designs that do not vanish at the
  origin and whose span is closed under differentiation (polynomial trends of any order, an intercept with sines and
  cosines) and `detrend_first` of a continuous design that does not vanish at the origin, with a zero or a stationary
  start and least squares detrending; `method = :line` refuses the rest (GLS detrending, a fixed start, a design with a
  jump or a numerator of another rank) with the reason, `:auto` takes the paths there. On warm calls at c0 = 0 it is 3
  to 5 times faster than the paths below the unit root (at least 1.3 times for every covered design, start and c from
  -50 to 2) and on par with the classical engine for the intercept and the trend; with c0 != 0 it is 2 to 5 times
  slower, hence the rule. Its values are within 1.7e-11 of the paths' (trends of order 3, 4, 6 and the intercept with
  the first sine and cosine at c = 0, -13.5, -20, -50). Through the one interface `method = :line` or `:paths` forces
  the general engine's representation for a design the classical engine does not have and with `engine = :general`; for
  a design that is a classical case `:line` is the classical line integral and `:paths` the general engine's paths.
  `empty_caches!` drops the line integral's providers too. With `engine = :general` a classical case at c0 = 0 now takes
  the general engine's line integral (`method = :paths` for the paths).
- **The coefficient statistic's engine.** `coef_cdf` and `coef_icdf` evaluate the t-ratio's distribution function at
  x = 0 with the null value c0 + y, one point per call, where the rule of the t-ratio (measured with five points) took
  the general engine for c0 + y > 0. At x = 0 with a zero start (cases DK, DU, DT, -100 < c <= 1) the rule now takes the
  classical line integral for c0 + y > 0 (2 to 12 times faster there) and the general engine for c0 + y < 0 in case DK
  with c > -5 (1.4 to 2.7 times faster); elsewhere the rule of the t-ratio. The values move within the differences of
  the two engines (within 3e-11 in these cells).
- **The general engine above the unit root** (`src/DFDet.jl`, `src/merged.jl`; the same repairs in the C engine and the
  MATLAB programs of dflimit-matlab). Four repairs, a guard of `coef_cdf_gp`, and the range of c made explicit.
    - The coefficient statistic far above the unit root was silently wrong. The panels of the paths around the roots
      merged every point within 1e-12 max(B - A, 1) of an end with it (`graded_panels`); far above the unit root the
      segment of the path around the positive root is shorter than that (the root lies within about 2c e^{-2c} of the
      branch point -t/c0 of the coefficient statistic, 1e-16 at c = 20) and got no panel at all. From c = 16 on the
      coefficient statistic of the general engine was off by up to 0.2 without a warning (case DK at c = 20: 0.3128 for
      the 30% point; through `DFDet.coef_cdf`, `coef_cdf(y, "DK", c; engine = :general)`, `coef_cdf(y, trend(-1), c;
      engine = :general)`, `ltratio_cdf(0, "DK", c; c0, engine = :general)` and `dist_cdf`; the default route takes the
      classical engine there and was right). The tolerance is now relative to the length of the interval. Against the
      classical engine and the single inversion of DFKan (seven levels from 0.1% to 99.9%) the distribution function and
      the density are within 2.4e-9 and 4.2e-9 relative up to c = 20 without deterministic terms, the distribution
      function within 1.5e-8 up to c = 16 for an intercept, a linear and a quadratic trend and Perron's models A and C
      (the density of cases DU, DT within 4.4e-9 relative); the quantiles (`coef_quantile`) within 1.3e-13 of the
      classical engine's.
    - The moments far above the unit root were wrong on the default route. The (1,1) entry of the matrix of the
      transform was formed as 1 plus the product term minus 2 gamma Sigma_11 plus (m011 - (1 - 2 gamma Sigma_11)), with
      an absolute rounding error of 1e-16; above the unit root the entry is of the order e^{-2c} where the moments and
      the inversion of Gil-Pelaez evaluate the transform (s != 0), and the moments of the general engine, which the rule
      takes at c0 = 0 for every c, were off by 1.5e-9 to 7e-8 relative at c = 10, 6e-4 at c = 15 and 0.95 at c = 20 in
      cases DK, DU, DT (`ltratio_mom(4, "DK", 20)` among them). The entry is now m011 plus the product term (`buildM!`,
      `buildM_s`): within 1.1e-14 relative of the classical engine at c = 2 to 20, c0 in {-2, 0, 2}.
    - The moments with a local null value c0 != 0 refused where the circle of the Cauchy integrals reaches beyond t =
      c^2/2: the logarithm continued around the circle was that of the determinant alone, whose phase at the real point
      is -Im theta there, and the check of the real start failed on every circle (`ltratio_mom(4, "DT", 0.0; c0 = -2.0)`
      threw a TaskFailedException on the default route; case DT at c = 0 with c0 < 0). The logarithm is now that of the
      entire function e^{c + theta} det(...) where the circle reaches t = c^2/2 (`dist_moments`; elsewhere unchanged bit
      for bit).
    - The density of the coefficient statistic with a fixed start above the unit root was off without a warning: the
      abscissa of the path G+ around the positive roots is chosen by an objective (`place`) that weighs the size of the
      transform and the grading toward the origin and the other group, but not the branch point -Re(t)/c0 of the
      v-kernels, the left end of its admissible interval, where the kernel of the density of the coefficient statistic
      has a double pole, (t + c0 s)^{-2}. With a fixed start (cases DK, DU, g0 in {0.5, 1, 3}) one evaluation was off by
      more than 1e-5 at 42 of 270 points of a map from c = 1 to 5 (nine levels from 1% to 99%), by up to 0.12 (case DU
      at c = 5, g0 = 0.5, at the 90% point, the density 1.29) and 5.7e-2 (case DK at c = 5, g0 = 0.5, at the 5% point),
      and by 5.9e-5 in case DU at c = 2, g0 = 1 at the 95% point; the distribution function by up to 1.1e-5. The branch
      point is now a focus of the grading cost: within 1.3e-8 of DFKan on the map, extended to case DT (405 points). At
      c <= 0 nothing changes (the map was within 3e-10 before): the change applies above the unit root with c0 > 0 only.
    - `coef_cdf_gp` refuses where its grid does not resolve the transform (`gp_edge`, `GP_EDGE`): the error of its
      trapezoidal rule is about 0.16 exp(-2 pi d/h), d the distance of the nearest singularity of the transform from the
      real line, h the step of the grid near the origin, and d shrinks like 2c e^{-c} above the unit root; it was off by
      1.1e-2 at c = 12, by 1.7 at c = 15 and by 15 at c = 20 for the 30% point of case DK. Below d/h = 3.3 it now raises
      an ArgumentError that names the N that serves (with the default N from about c = 10 in the bulk and c = 5 in the
      lower tail).
    - The range of c. Above the unit root one root of the transform is of the order 2c e^{-2c}, and the reciprocal
      roots, the eigenvalues of a small matrix, span e^{2c}: with deterministic terms the others are lost in double
      precision from c = 17 (the second reciprocal root of case DU is -0.0051 at c = 14, -0.0040 at c = 16, -0.0023 at c
      = 17 and zero at c = 19, and the paths fail with "rootpath: ... roots at the start" from c = 17 in case DU, c = 18
      in case DT; with a fixed start the coefficient statistic of case DU was off by 3e-5 at c = 12 and that of case DT
      by 1e-4 at c = 14, without a warning). Every function of the general engine now refuses above c = 20 without
      deterministic terms, above c = 16 with them and above c = 10 with them and a fixed start (`gen_cmax`, `GEN_CMAX0`,
      `GEN_CMAX`, `GEN_CMAX_FIXED`), with an ArgumentError that names the statistic, c and the bound and points to the
      classical engine; the rule `autoengine` takes the classical engine there. Without deterministic terms the
      coefficient statistic with a fixed start (g0 = 1, 3) is within 5.4e-8 of DFKan up to c = 20, with them (cases DU,
      DT) within 3.2e-6 up to c = 10.
- **The upper tail above the unit root, and the final check of a distribution function** (`src/DFExact.jl`, `src/DFDet.jl`,
  `src/DFDet_line.jl`; the same changes in the C engine and the MATLAB programs of dflimit-matlab). Above the unit root the
  t-ratio with a null value c0 != c has its upper tail at x of the order (c - c0) e^c/(2c) (V is of the order e^{2c}), and
  the default route returned values of the distribution function outside [0, 1] there without a warning: in case DK at
  c = 10, c0 = 2 0.8415 for 0.7436 at x = 1e4 and -1.3674 for 0.9993 at x = 3e4, at c = 7, c0 = 2 1.3804 for 1.0 at x = 3e3,
  at c = 10, c0 = 9 1.0479 for 0.9936 at x = 3e3 (also with `verified = true` and `engine = :classical`). The cause: beyond
  x ~ 700 the rule of the line integral's inner integrals needs more than RULEMAX nodes (the phase of their saddle factor
  grows like x^2), the point goes to the conditional integral with its checks (`route_double`), and where that flagged the
  point its one evaluation (one path) was returned, which in case DK is off by up to 14 (it was set on case DT, where the
  checked value is off by 3e-2 with the flag and one path is right). In case DK the flag is a false alarm: against an
  independent computation (the law of V given J(1) from Girsanov's theorem, inverted on a line through its saddle point
  and integrated over J(1), `scratch/fut/bridge_ref.jl`, within 1e-12 of both engines where they agree) the checked value
  is within 1.3e-8 up to c = 14. Case DK now takes the checked value whatever its flag; cases DU, DT are unchanged (one
  path where the checked value is flagged, within 1e-14 of the general engine). Far above the unit root (c > 4) the
  three cases take the conditional integral, as at the points beyond the bound, at x > 300 too, and case DK with c0 > c at
  x < -20 (`far_tail`, before the rule's representations): at large x the line integral's cost grows like x^4 (10 s at
  x = 700 and more than three minutes at x = 2350 in case DK at c = 10, c0 = -2; in cases DU, DT at c = 16, c0 = 0 some
  2 GB at x = 1.3e6 and more than 8 GB at the median, x = 4.4e6), the conditional integral's is 0.1 to 0.4 s; and in case
  DK with c0 > c at x < -20 the far root |x|/c0 of X(v) lies inside the range of sqrt(V) and the line integral was off
  by up to 9.8e-6 at c = 5, 5.5e-5 at c = 7 (a negative value at c0 = 9, x = -627), 1.5e-3 at c = 10 (c0 = 10.5,
  x = -400) and 2.9e-4 at c = 13 without a warning (the conditional integral within 3e-10 there). In cases DU, DT these
  values are within 1e-14 of the general engine up to c = 13 (x up to 3e4) and within 5e-7 of the large-c law (t ~ (c -
  c0) |J(1)| sqrt(k), k the limit of V/J(1)^2; `scratch/fut/check_asym.jl`) at c = 13 and 14. Above c = 14 these points
  are refused in the three cases with an `ArgumentError` that names the statistic, c, c0, x and the region computed
  (`FAR_CMAX`; before anything is computed, with `verified = true` too): the checked conditional integral of case DK is
  off by up to 7.7e-5 at c = 16 from the 99.7% point on, 5.5e-3 at c = 18 and by orders of magnitude at c = 20, one path
  of case DT gives values of 1e44 and more at c = 16, the line integral of cases DU, DT exhausts the memory there, and the
  general engine's line exceeds its bounds. With `verified = true` these points no longer fall back to the line
  integral. The
  general engine cut its vertical line to the oscillation of the outer integrand as if c0 were 0 (`line_refine`): with
  c0 != 0 the stationary value of the exponent, (s_k x)^2/(4 (t + c0 s_k)), oscillates faster, by 1/|1 - c0/c| in case DK,
  and the distribution function was off by 1.1e-3 at c = 10, c0 = 9, x = 1000, 3.8e-3 at x = 3000, and 1.0000000112 at
  x = 1e4 (`engine = :general`); for c0 > 0 the line is now cut to the larger of the two rates (`line_refined(CP, xs;
  c0)`), within 2e-13 of the reference (with c0 < 0 the rate is below that of c0 = 0, and the cut is unchanged). Every distribution function of the two engines (`ltratio_cdf`, `coef_cdf`, `dist_cdf`,
  `DFDet.coef_cdf`) ends with a check of its values (`cdf_range_check`): a value below -1e-8 or above 1 + 1e-8 raises an
  error that names the call, its arguments, the point and the value; where the representations hold their errors are far
  below 1e-8, so such a value is a computation that failed (the general engine in case DK with c0 > 0 at x < -sqrt(2 c0)
  above the unit root, where its line keeps the Gauss-Legendre rule and its warning: -1.7e-4 for 7.5e-6 at c = 5, c0 = 1,
  x = -4, now an error). The map of the default route in case DK with a zero start (c in {1, 2, 3, 5, 7, 10, 13, 16},
  c0 in {-2, 0, 1, 2, c - 1, c + 2}, x from about the 0.1% to the 99.994% point and into both tails, `scratch/fut/map2.jl`
  against `bridge_ref.jl` for c >= 5 and the general engine below): within 1.1e-8 of the reference at every point
  computed (5.2e-7 at c = 7, c0 = 9, x = -15.7, where the line integral keeps the point), no value outside [0, 1], and at
  c = 16 the points beyond x = 300 (and below x = -20 with c0 > c) refused; cases DU, DT (against the general engine
  where it computes, and the large-c law at c >= 13) within 8.2e-6 of the law at c = 13 and 1.3e-7 at c = 16 (its own
  error at c0 = c - 1), the two engines differing by up to 1.7e-4 only where the general engine's line keeps its
  Gauss-Legendre rule (c0 > 0 at x < -sqrt(2 c0)).
- **Case DG faster.** The transform behind the linear form and the power envelope of case DG (`cf_gls`) formed 3 x 3
  complex matrices on the heap with a matrix solve and a determinant at each of the 60001 points of its grid; written
  out, it is 6 times faster (82 to 14 ms; the classical envelope of case DG at c = -5, -10, -20 from 623 to 151 ms),
  within 2e-15 of before on the grid and 1.1e-16 in the linear form.
- **The statistics of the second paper as functions:** the point-optimal statistic and the Gaussian power envelope for
  any design (`pt_cdf`, `pt_pdf`, `pt_quantile`, `pt_power`, `pt_envelope`; `power_envelope(c, D)`), the LM statistic
  of Schmidt and Phillips and of Amsler and Lee (`lm_cdf`, `lm_quantile`, `lm_power`) and the companions of the
  GLS-detrended t-ratio, MZ_alpha (`mza_cdf`, `mza_quantile`) and MSB (`msb_cdf`, `msb_quantile`).

## 0.5.0

Four errors of the engine are repaired. Values outside the regions named below are unchanged bit for bit.

- **Wrong values with a fixed start.** With a fixed start (`g0 != 0`), a negative local null value close to zero
  (`c0` of -0.01 or -0.001), `x > 0` and `c >= 0`, the line integral returned wrong values, from errors of 1e-7 and
  2e-3 that look plausible to values of 1e74 and beyond: at `c = 0`, `c0 = -0.01`, `g0 = 1`, `x = 1.5` in case DK it
  returned 5.6e74 where the value is 0.960992. Case DK was affected by default, cases DU and DT with
  `method = :single`. A piece of the path that should descend climbed and cancelled against its neighbour; such a
  piece is now cut back.
- **Unbounded memory for a negative local null value near zero.** With `c0 < 0` close to zero and `x > 0`, in every
  case and with every start, the paths of the line integral grow like `(x/c0)^2` and every node was stored: a call
  with a stationary start at `c = c0 = -1e-4` took more than 5 GB. Paths are now summed as they are built and are
  limited to two million nodes. Points beyond the limit go to the conditional integral under `method = :auto`, with a
  warning where a value is flagged; with `method = :single`, and in case DG, they raise an `ArgumentError` that names
  the arguments.
- **Unbounded memory at large `x` above the unit root.** In the bulk of the distribution at `c = 7` and beyond (`x`
  in the hundreds) every new Gauss-Legendre rule was built with its full matrix of eigenvectors: case DT at `c = 7`,
  `x = 403.7` took 3 GB. Rules above 1024 nodes are now computed by Newton's method in memory proportional to their
  size, and are limited to 4096 nodes; points beyond that go to the conditional integral as above.
- **Wrong values of the linear form for small `|a|`.** `ltratio_lincdf` inverted the characteristic function of
  `U - aV` by the formula of Gil-Pelaez on a grid that ends at `u = 4000`. At `a = 0` (the law of `U`, whose density
  is infinite at the edge `b = -1/2` of its support) the transform decays only like `u^(-1/2)`, and the values were off
  by up to 7e-2 within 1e-3 of the edge, 1e-2 within 1e-2 of it and 1e-4 in the bulk, in cases DK and DG alike; below
  the support they were as low as -1.5e-2 instead of 0, and above 1 by up to 5e-5 (2380 of 12541 values of a map over
  `a`, `b` and `c` lay outside [0, 1]). The error stays at that size for `|a|` up to about 3e-3 and falls to 1e-3 in
  case DK (4e-5 in case DG) at `|a| = 0.3` and to 3e-8 at `|a| = 1`; it is larger where `U` and `V` are concentrated
  (1.3e-3 at `c = -50` for small `|a|`; at `c = -200` and `-500` even for `|a|` of 3 to 10) and where `U - aV` is wide
  (up to 1.2e-2 in case DK at `c = 5`). `P(V <= b)` (`a = NaN`) was off by 8e-8 at `c = -50`, 4e-2 at `c = -200`, 8e-2
  at `c = -500` and 1.2e-5 at `c = 5`. Where the transform has not fallen below 5e-9 by the end of the grid, or
  `c > 0`, the value now comes from a conditional integral, in case DK that of the region `{U <= aV + b}` (as
  `ltratio_region`), in case DG the conditional integral given `U` (the law of `U + 1/2` is a gamma law of shape 1/2 under
  every tilt in `V`): to about 1e-10, within [0, 1] and 0 below the support. `P(V <= b)` uses a longer grid for
  `c <= -50` and a finer one for `c > 2`. `power_envelope` near `c = 0`, where its weight `a` is small, takes the
  conditional integral too (in case DG it fell below `alpha` there; now `alpha + O(c^2)`, 0.0500000034 at `c = -1e-3`;
  at `c = -1` 0.0531924225 for 0.0531924279).
  `ltratio_lincdf` now requires `c >= -550` (as the other functions), case DG `c` up to about 3.5 at `cbar = -13.5`
  (less for `cbar` nearer 0) and `a = NaN` `c` up to about 7, with an `ArgumentError` beyond, and gives 0 and 1 at
  `b = -Inf` and `Inf`. `ltratio_region` with `v0 > 0` changes with `P(V < v0)` at `c <= -50` and `c > 2`.
- The caches kept per point and per `c` hold at most 64 entries.

## 0.4.0

- The documentation and the comments refer to the representations by name (the line integral, the contour integral
  and the conditional integral representation) instead of by the paper's theorem and equation numbers. The results are
  unchanged.
- The package is now the home of the Julia engine: `src/DFExact.jl` and `src/df_series.jl` are no longer copied from the
  paper's repository (`tools/sync-engine.sh` and `UPSTREAM` are gone). The engine's checks moved here with it:
  `test/engine/quickcheck.jl` and `test/engine/test_dfexact.jl` run with `Pkg.test()` (JSON and Random are test
  dependencies), the longer checks and the writers of stored values are in `dev/`. The engine and the results are
  unchanged.
- IntelVectorMath is a weak dependency (`[weakdeps]`, compat 0.5): its use, on x86-64 Linux and Windows, is in the
  package extension `DFLimitIntelVectorMathExt`, which Julia loads when IntelVectorMath is loaded in the session; the
  engine no longer loads it from the active environment itself. Without it the values are unchanged, bit for bit.
- The tests run on GitHub Actions on every push to `main`, on tags and on pull requests (Julia 1.10 and the current
  release on Linux, the current release on macOS and Windows, and one job with IntelVectorMath), with coverage on
  Codecov; CompatHelper keeps the compatibility bounds current.

## 0.2.0

- Every value is now one evaluation by one method by default: no second evaluation, no refinement driven by an error
  estimate, no fallback to another representation and no recomputation of the tails. `:auto` takes at each point the
  fastest representation that one evaluation gives to well below 1e-5 (`DFExact.automethod`). The new keyword
  `verified = true` of `ltratio_cdf`, `ltratio_pdf`, `ltratio_icdf`, `coef_cdf` and `coef_icdf` gives the checked
  computation of before (error estimates, the agreement of two evaluations, the fallbacks and the recomputation of the
  tails), and reproduces the values of before. Where the parameters lie in a regime in which one evaluation is known to
  be brittle (with the conditional integral: a zero start with `c0 > 0` below the unit root, case DK with `c0 != 0` for
  `c > -5`, case SK for `c > -0.5`, `c < -100` with a zero start and `c < -50` with a stationary one, a fixed start below
  `c = -20` in case DK and `c = -30` in cases DU, DT; with the contour integral: `c0 > 0` for `c < -15`, `c0 < 0` for
  `c > 1`, and in case DU `c0 > 3` for `c > 1.5`) a warning names the regime, once per call (`DFExact.brittle_msg`). Away
  from these regimes the default values agree with the verified ones to well below 1e-5 (to 5e-9 at the reference
  points of the contour integral).
- A third representation, the contour integral (`method = :contour`, `DFExact.tr_contour`), for a fixed start `g0 != 0`
  in cases DU and DT: a double integral from the inversion formula of the paper on paths around the singularities of
  the moment generating function, with the integral over v in closed form, the paths in s around the roots of the determinant and a hyperbola in t, an explicit integrand
  where the line integral needs the noncentral kernel, itself an integral. `:auto` takes it there (one evaluation takes 7 to 15 ms a value on ten threads, against 26 to 360 ms for the line integral with its noncentral kernel and 0.24 to 0.80 s for the conditional integral), except for `c > 1` with `c0 < 0` and, in
  case DU, for `c > 1.5` with `c0 > 3`, where it fails and the conditional integral serves. Where the line integral was
  accurate the values change by at most 4e-13 (3.5e-10 at the corner `(c g0)^2/2 = 7200` of case DT, where the line
  integral is the less accurate); with `c0 > 0` at `c` below about -60 the line integral, which `:auto` took before, is
  wrong (`ltratio_cdf` gave values up to 1e83 there), and the values are now right.
- The moments of case DK with a fixed start (`ltratio_mom`) were NaN below `c` of about -280 (a factor of the moment
  generating function overflowed); it is now computed in a form that stays finite.
- The engine re-synced from DF-Limit (November 2026). The power envelope of the trend case,
  `power_envelope(c, "DG")`, uses the weight `a = c^2/(2 (c - 1))` of the point-optimal invariant test of Elliott,
  Rothenberg and Stock (it used `c/2`, the weight of case DK; the curve rises by at most 3e-5). With a stationary
  start the functions throw an `ArgumentError` for `c > -1e-4`, where the variance `-1/(2c)` of the start is too
  large for the quadrature. Two errors near the unit root are fixed: in case SK the distribution function by the line
  integral took fixed node layouts whose constant part is wrong for `c > -1.2` (off by 2e-8 at `c = -0.5`, 1e-4 at
  `-0.1`, the same at every `x`; the density was right), and the conditional integral cut off the mass of `V` near
  zero (the coefficient statistic of case SK was off by up to 1e-3 for `c` between -0.01 and 0).
