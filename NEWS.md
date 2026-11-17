# News

## Unreleased

- The engine re-synced from DF-Limit (November 2026). The power envelope of the trend case,
  `power_envelope(c, "DG")`, uses the weight `a = c^2/(2 (c - 1))` of the point-optimal invariant test of Elliott,
  Rothenberg and Stock (it used `c/2`, the weight of case DK; the curve rises by at most 3e-5). With a stationary
  start the functions throw an `ArgumentError` for `c > -1e-4`, where the variance `-1/(2c)` of the start is too
  large for the quadrature. Two errors near the unit root are fixed: in case SK the distribution function by the line
  integral took fixed node layouts whose constant part is wrong for `c > -1.2` (off by 2e-8 at `c = -0.5`, 1e-4 at
  `-0.1`, the same at every `x`; the density was right), and the conditional integral cut off the mass of `V` near
  zero (the coefficient statistic of case SK was off by up to 1e-3 for `c` between -0.01 and 0).
