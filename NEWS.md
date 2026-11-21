# News

## Unreleased

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
  roots (c0 > 0 with x < 0) keep the line's Gauss-Legendre rule and its warning.
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
