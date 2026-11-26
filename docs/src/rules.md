# The rules of one evaluation

One evaluation of the general engine's inversion on paths (`dist_cdf`, `dist_pdf`, `dist_quantile` and, through them,
the classical functions where the rule `autoengine` takes the general engine) takes a thinner rule than before November
2026 where it holds (`DFDet.PATH_RULES`, `DFDet.path_region`; keyword `rule`, :auto by default): the outer hyperbola
with 1.5 times the step and the U-shapes with 16 nodes per panel and 32 on the tails of their rays for 24 and 40 (32 and
48 for 48 and 64 with a fixed start), 0.50 times the time of one evaluation (the median over 564 configurations, 0.30 to
0.87; the thin inner rule on the present hyperbola 0.73). The regions, a rule in (the start, whether the design has
deterministic terms, c, c0) that the C engine and the MATLAB programs can take over:

| region | rule | why |
|---|---|---|
| c > -1, c0 < 0, zero or stationary start | the thin inner rule on the present hyperbola | the thinner outer step lost 1.3e-8 of the peak of the density (case DK at c = 0, c0 = -5) |
| c > -1, c0 < 0, fixed start, deterministic terms | the thin inner rule on the present hyperbola with half its step | the thinner outer step lost up to 3.1e-6 (case DU at c = 2, c0 = -5) |
| c > -1, c0 < 0, fixed start, no deterministic terms | the present inner rule on the present hyperbola, the step min(1/2, 2.5/(g0 \|c0\|)) times its own where it is flatter than 70 degrees (2 to 11 times the cost) | the present step was off by up to 0.86 in the upper tail (c = 0.5, g0 = 1, c0 = -20; 1.8e-3 at c0 = -5), this one within 7e-12 of the classical engine (g0 from 0.5 to 2, c from -0.5 to 1, c0 from -1 to -20) |
| c > -1, c0 = 0, fixed start, no deterministic terms | the thin inner rule on the present hyperbola | the thinner outer step lost 1.4e-8 of the peak of the density at mean + 5.75 sd (c = 0.5) |
| c > 2 with a fixed start, c > 5; c < -60 with c0 other than 0 and c | the present rule | beyond the range of the sweep (at c = 5 with a fixed start the thin rule differed from the present one by 5.6e-6 of the peak of the density, Perron's model C with two breaks; at c = -480, case DK, c0 = 2, by 7.7e-9) |
| elsewhere | the thin rule | |

On the vertical line (above the unit root, where no hyperbola keeps the positive roots right of the origin) the line
keeps its panels and only the inner rule thins. `rule = :present` gives the rule of before, `verified = true` takes it
(and its variants), and `hfac`, `n`, `ntail` override the rule. A sweep of 822 configurations (no deterministic terms,
an intercept, linear and quadratic trends, Perron's model B with one break and model C with two, the linear trend
GLS-detrended at -13.5 and detrending first; zero, fixed (g0 = 1) and stationary starts; c from 2 to -60 and c0 in {0,
2, -2, 5, -5, c}; F and f at the 1e-5 to 1 - 1e-5 points; `scratch/rules/sweep1.jl` of the branch kan-integration) set
the regions, and a dense one checked them (`scratch/rules/dense3.jl`: 736 configurations, x from the mean - 6 sd to the
mean + 6 sd in steps of 0.5 sd, c from 2 to -60 and at 5 and -480; the rule within 7.6e-10 of the present rule in F and
4e-9 of the peak in f wherever it is not the present one or a finer one). The criterion: within max(1e-9, twice the
present rule's own error) in F and max(1e-8 of the largest density, twice its own) in f, against a reference with half
the outer step and twice the inner nodes. A first thin rule with 12 and 24 nodes held at the sweep's points but lost up
to 9e-8 of the peak of the density between them near the unit root (case DK at c = 0.5 near the mode), which 16 and 32
nodes do not. Far from the unit root the reference is not the more accurate: in case DK at c = -60, c0 = 2 the present
rule is within 1.7e-10 of the classical engine's checked computation, the reference 6.7e-10. On the line with c0 > 0 at
x < 0 (two roots of the parabola) the line keeps its Gauss-Legendre rule and its warning, and the present rule is off by
up to 2.7e-5 there (case DK at c = 5, c0 = 2), as is the thin one; `method = :auto` takes the line integral at these
points where it covers the design (`DFDet.line_vline`). The extensions take the same rules: the volatility transform
that of `path_region` for the zero-start model of its design at the smallest c of its regimes, the normal average of
`rho` that of each of its calls of the distribution function.

The classical engine's contour integral (a fixed start in cases DU, DT; `DFExact.ctr_rule`, keyword `rule` of
`tr_contour`) takes, for one evaluation, 32 nodes per panel and ray of the U-shapes for 64 and the outer step 1.5 times
as long where they hold: 0.42 times the time near the unit root (median; 0.28 to 0.70), the 32 nodes on the present
step above the unit root (c > 1, 0.59 times), 64 nodes on the longer step with c0 > 0 below c = -2.5 and everywhere
below c = -10, and the present rule where |c g0| > 20 (there the longer step lost up to 2.4e-2 in F, case DT at c =
-30, g0 = 3). Against the checked computation on a dense grid of x (129 configurations: cases DU with g0 0.5 and 1.5,
DT with 1 and 3, c from 2 to -60, c0 in {0, 2, -2, -5, c}) every configuration holds the criterion, within 6.2e-11 in
F. Far from the unit root the present rule's own one evaluation was off: with c0 > 0 in the density by up to 5e-6 of
the peak (case DU at c = -30, g0 = 0.5, c0 = 2), 4.2e-3 (c = -75, c0 = 5) and 2.2e-2 (c = -60, g0 = 1.5, c0 = 5), and
where |c g0| > 60 with any c0 in the distribution function by up to 3.9e-5 (case DU, c = -60, g0 = 1.5, c0 = -2) and
6.3e-6 (c = -100, g0 = 0.9); its checked computation was off in the density by up to 8.4e-6 of the peak. The rule of
the engines (`autoengine`) gives these cells to the general engine: with c0 > 0 below c = -20 (below c = -10 where |c
g0| > 20), and with any c0 below c = -20 where |c g0| > 60. There the general engine's density is the derivative of the
distribution function on which the two engines agree (to 1e-10 of the peak, by Richardson's central difference;
`scratch/rules/probe_ctr_far5.jl`), and it takes 0.23 to 0.9 times the time of the classical one evaluation down to c =
-45, up to 1.8 times at c = -60 and 1.2 to 5 times below c = -75.

The classical engine's conditional integral (one evaluation, `DFExact.cond_rule`) takes 48 nodes on the path of each
inner inversion and of the integral over l for 96 with a stationary start for c <= -2, with a zero start in cases DU, DT
for c >= -5 and in case DK for c <= -13.5 (medians of warm calls, F and f at 13 points: 0.60 times the time with a
stationary start, 0.81 and 0.87 in cases DU, DT with a zero start, 0.86 in case DK, where the conditional integral takes
20 to 50 ms), and 64 with a stationary start for c > -2; a fixed start (measured at one configuration only) and c > 2
keep 96. Against the rule with 192 nodes and the outer tolerance 1e-14 on a dense grid of x (150 configurations: cases
DK, DU, DT with a zero start, c from 2 to -100, and SK, SU, ST, c from -0.5 to -50, c0 in {2, -2, -5, c}) the regions
hold the criterion. Elsewhere 48 nodes lost up to 1e-3 far from the unit root with a zero start in cases DU, DT (1.8e-8
of the peak of the density at c = -13.5), 7e-8 of the peak in case SK at c = -0.5, and 7e-7 with a fixed start (case DU,
c = 2, c0 = -2, g0 = 0.5). The outer tolerance 1e-10 in place of 1e-12 held as well but gained nothing measurable (0.85
against 0.90 with a zero start) and is not taken.
