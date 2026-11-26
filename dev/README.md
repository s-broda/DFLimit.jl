# Development checks of the engine

The engine, `src/DFExact.jl` with `src/df_series.jl`, was developed in the directory `julia/` of the companion paper's
repository (DF-Limit) up to commit b484d4f; its checks came with it. The two that act as tests, `quickcheck.jl` and
`test_dfexact.jl`, are in `test/engine/` with the stored values they read and run with `Pkg.test()`. The scripts here are
longer checks and the writers of stored values, run by hand when the engine changes. They use the environment of this
folder, which loads the package from the parent folder (Julia 1.11 or later, for the `[sources]` entry of
`dev/Project.toml`); once, from the package root:

```sh
julia --project=dev -e 'using Pkg; Pkg.instantiate()'
```

`engine_table.jl` measures the two engines of the package (the classical one, `DFLimit.DFExact`, and the general one,
`DFLimit.DFDet`) in each cell of the rule `DFLimit.autoengine`: the warm time of one call by each, their largest
difference, the engine the rule takes, and where they differ by more than 1e-8 each against the other's checked
computation; it writes `engine_table.txt`, from which the table "Which engine" of the README is taken:
`julia -t 4 --project=dev dev/engine_table.jl`, about half an hour.

`certify_dfexact.jl` certifies the line integral by node doubling (every adaptive node count and every nodes-per-panel
count of the fixed layouts), contour independence and the uniform rule, for c > 0 up to the far upper tail and for
c <= 0: `julia -t auto --project=dev dev/certify_dfexact.jl`, about 20 minutes.

`check_applications.jl` checks the application functions against each other and against closed forms, by node doubling,
and by a Monte Carlo of the finite-sample ADFmax statistic with the response surfaces of Otero and Smith (2012):
`julia -t auto --project=dev dev/check_applications.jl`, about five minutes; the argument `nomc` leaves out the Monte
Carlo.

`check_contour.jl` compares the contour integral representation with the line and the conditional integral at random
points of the region of a fixed start in cases DU, DT and writes one row per point to a CSV file:
`julia -t 8 --project=dev dev/check_contour.jl [npoints] [seed] [out.csv]`; `summarize("out.csv")` (after
`include("dev/check_contour.jl")`) prints the summary.

`largec_ref.jl` writes the reference values far from the unit root, `test/engine/largec_ref.txt` (c = -60, -100, -200,
-480; the six cases, c0 = 0 and c0 = c, x from the mean - 5 sd to the mean + 5 sd; moments; quantiles), which node
doubling, a second line and the moments of the density confirm to 1e-11: `julia -t auto --project=dev dev/largec_ref.jl`,
about an hour on six threads. `test/engine/test_dfexact.jl`, `test/engine/quickcheck.jl`, the C engine's tests and the
MATLAB `ltratio_test` of dflimit-matlab check against them.

`abadir/` compares case DK at c = 0 with the formulas of Abadir (1995): `python3 abadir_series.py` (mpmath) writes
`abadir_series.json`, then, from `dev/abadir/`, `julia -t auto --project=.. abadir_check.jl`.

The C engine of dflimit-matlab (`c/`) is a port of this engine; the reference values of its tests (`c/test/*_ref.txt`)
and its tables (`c/tables.c`) are written from this package by the scripts `c/test/*_ref.jl` and `c/gen_tables.jl`
there.

## Range of c

The rules are set for c from about -480 to 10. For c < -20 the end and the step of the outer integral follow c (`wmaxc`,
`hufac`), the Filon panels of the edge parts are at most half a period, 2 pi |c|, long (`filonpanels`), the first panels
of the K_0 layouts and of the adaptive edge segment follow the scales 1/b and 1/|s_1| of the kernel near X = 0
(`k0first`, `edgehead`), t_* is bracketed in phi = sqrt(2t - c^2) (`tstar`), the moments take circles of radius at most 2
and s up to 80|c| (`momrcap`, `momthi`), and the conditional integral of cases DU, DT starts at E V e^-3 (`dzlo`) and
keeps |a(v)| at least 2^-54 away from zero (`rfloor`); nothing changes for c >= -20 (bit for bit). Below c = -550 the
user-level functions raise an error (`CMIN`; the moments of cases DT, ST fail from c = -600, the line integral from
c = -650; at -550 and -600 the line integral is stable to 1e-12 in the six cases, case DG to 3e-11). Known limits: with
c0 = c the third and fourth moments are accurate to about 1e-9 with a zero start and to 1e-8 to 2e-7 with a stationary
start at c = -100 to -480; with a fixed start g0 != 0 the user-level functions raise an error outside the region in
which the noncentral kernel is checked to 1e-10 (`fixed_ok`): in case DK c <= 2, |g0| <= 2 (|g0| <= 1 for c > 0), in case
DU -100 <= c <= 2, |g0| <= 1.5 and (c g0)^2/2 <= 4050, in case DT -100 <= c <= 2, |g0| <= 3 and (c g0)^2/2 <= 7200 (the
MATLAB programs, which have only the conditional integral there, also require c >= -20, or c >= -40 with
(c g0)^2/2 <= 100).
