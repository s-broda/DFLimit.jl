# News

## 0.5.0

Three errors of the engine are repaired. Values outside the regions named below are unchanged bit for bit.

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
