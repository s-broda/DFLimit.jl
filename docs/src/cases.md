# Case codes and arguments

The case `cas` is a two-letter code: `D` (fixed start, `g0`, default 0) or `S` (stationary start, c <= -1e-4),
followed by `K` (no deterministic terms, tau), `U` (intercept, tau_mu) or `T` (intercept and trend, tau_tau);
`"DG"` is the GLS-detrended trend case (DF-GLS, cbar = -13.5); in place of the case a design of the general engine
may be given (see [Two engines](@ref)). `method` is `:auto` (at each point the fastest
representation that one evaluation gives to well below 1e-5), `:single` or `:line` (the line integral representation of
the companion paper), `:double` or `:conditional` (the conditional integral representation, also of the companion
paper) or `:contour` (the contour integral representation: the inversion formula on paths of the lead paper, around
the singularities of the moment generating function, with the
integral over v in closed form and the paths in s and t deformed, a double integral with an explicit integrand; a fixed
start in cases DU and DT only, and there the choice of `:auto`). By default (`verified = false`) every value is one
evaluation by one method, with a warning, once per call, where the parameters lie in a regime in which one evaluation is
known to be brittle; `verified = true` gives the checked computation (error estimates and fallbacks, at 1 to 4 times
the cost of the default, about 1.8 with the contour integral, more in the tails below 1e-3, which it recomputes). `x`, `y` and the probabilities `p` may be arrays. The boundary `bnd` of `ltratio_region` is a NamedTuple (or a
Dict): `(type = "line", a, b)`, `(type = "parabola", x, c0)`, `(type = "wap", k, omega)` or `(type = "fun", g)` with
`g` a function of a scalar, optionally with `v0` and `brks`.

The functions return the values only (the MATLAB functions also return an `info` structure); every one has a
docstring (`?ltratio_cdf`). Below c = -550, and with a fixed start g0 outside the region in which the programs are
checked, they raise an `ArgumentError` that says so. The engine's functions that are not exported are reached through
the module `DFLimit.DFExact`, among them the representations, which take the model number 1, 2, 3, 4 for K, U, T,
G: the line integral `DFLimit.DFExact.cdf(q, model; c, c0, g0, start = :zero)` (`start = :stationary` for a
stationary start), `DFLimit.DFExact.density` and `DFLimit.DFExact.moments(model; c, c0, g0, start)`, the
conditional integral `DFLimit.DFExact.tr_limdist(x, model, c; c0, g0, start, pdf)` and the contour integral
`DFLimit.DFExact.tr_contour(x, model, c; c0, g0, pdf)` (the comments of `src/DFExact.jl` describe them, and
`dev/README.md` the range of c in which the rules are set).
