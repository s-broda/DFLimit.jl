"""
    DFLimit

The exact limiting distribution of the Dickey-Fuller t-ratio: the distribution function, density, quantiles and
moments of the limits of the Dickey-Fuller t-ratios under local alternatives, by the exact representations of Bao,
Broda and Kan, "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio with General Deterministic Terms" (the
lead paper, SSRN 7745921), and, for the classical cases, of its companion paper, "Low-Rank Representations of the
Classical Dickey-Fuller Limits, with an Extension to Predictive Regressions" (SSRN 7725619).

The classical engine, that of the companion paper, is the module `DFLimit.DFExact` (the file `src/DFExact.jl`, with
`src/df_series.jl`); `DFLimit` re-exports its exported functions (`ltratio_cdf`,
`ltratio_pdf`, `ltratio_icdf`, `ltratio_mom`, `adfmax_cdf`, `adfmax_pdf`, `adfmax_icdf`, `coef_cdf`, `coef_icdf`,
`ltratio_region`, `ltratio_lincdf`, `power_envelope`, `ltratio_bisect`), and the functions it does not export are
reached as `DFLimit.DFExact.cdf` and so on.  The engine is multithreaded: start Julia with `-t auto`.

The general engine, that of the lead paper (any piecewise smooth deterministic terms), is the module `DFLimit.DFDet`
(the file `src/DFDet.jl`); the classical functions take its designs (`trend`, `perron`, `fdesign`, ...) in place of the
case, and `DFLimit.autoengine` chooses the engine of every call (`src/merged.jl`; the README, "Two engines").

Three extensions of the limit, computed by the general engine: volatility and persistence regimes (`volpath` and the
keyword `vol` of the t-ratio functions), a normal component (the keyword `rho`: the covariate-augmented t-ratio, the
predictive regression; `cadf_c0`, `predictive_c0`) and seasonal unit roots at a harmonic frequency (`hegy_t3_cdf`,
`hegy_t4_cdf`, `hegy_f_cdf` and their densities and quantiles).
"""
module DFLimit

include("DFExact.jl")
using .DFExact

# the engine's exported names, exported from here too (the list follows the engine file)
for name in names(DFExact)
    name === :DFExact || @eval export $name
end

# The general engine (any deterministic terms, the inversion on paths around the singularities of the joint moment
# generating function): the module DFLimit.DFDet, the file src/DFDet.jl as in the repository df-deterministics, with its
# outer paths of the three classical cases at the unit root in data/paths.  Its user-level names are exported from here
# where they do not clash with the classical engine's; the four that do (coef_cdf, adfmax_cdf, adfmax_pdf,
# power_envelope) are the classical functions, which take a design of the general engine as well (src/merged.jl), and
# the general engine's own are DFLimit.DFDet.coef_cdf and so on.
# The devices of Bao and Kan ("Exact Finite-Sample Distribution and Moments of the Least Squares Estimator in First-Order
# Autoregressions: O(1) Algorithms"): the closed-form characteristic function with the branch fixed at each point, jets,
# the single inversion with tail completion; the module DFLimit.DFKan, which the interface below calls for the
# half-plane statistics and the moments of the classical cases (nothing of it is exported), and whose generic inversion
# the general engine's file src/DFKan_det.jl uses
include("DFKan.jl")

include("DFDet.jl")
using .DFDet: Determ, trend, perron, fdesign, ltrend, design_from_matrix, detrend_first, dist_cdf, dist_pdf, dist_quantile,
              dist_moments, coef_pdf, coef_quantile, local_power, lin_cdf, lin_pdf, lin_quantile, v_cdf, v_pdf,
              adfmax_quantile, region_prob, simulate_tratio, simulate_gls, simulate_first, empty_caches!
export Determ, trend, perron, fdesign, ltrend, design_from_matrix, detrend_first, dist_cdf, dist_pdf, dist_quantile,
       dist_moments, coef_pdf, coef_quantile, local_power, lin_cdf, lin_pdf, lin_quantile, v_cdf, v_pdf,
       adfmax_quantile, region_prob, simulate_tratio, simulate_gls, simulate_first, empty_caches!

# the statistics of the lead paper (general deterministic terms): the point-optimal statistic and the envelope for any design,
# the LM statistic, MZ_alpha and MSB (in the module DFDet, whose file stays that of df-deterministics)
# the general engine's determinant with the branch from the pivots and the single inversion (Bao and Kan), for the
# point-optimal statistic and coef_cdf_gp
Base.include(DFDet, joinpath(@__DIR__, "DFKan_det.jl"))
using .DFDet: coef_cdf_gp
export coef_cdf_gp
Base.include(DFDet, joinpath(@__DIR__, "DFDet_paper2.jl"))
using .DFDet: pt_cdf, pt_pdf, pt_quantile, pt_power, pt_envelope, lm_cdf, lm_quantile, lm_power, mza_cdf, mza_quantile, msb_cdf, msb_quantile
export pt_cdf, pt_pdf, pt_quantile, pt_power, pt_envelope, lm_cdf, lm_quantile, lm_power, mza_cdf, mza_quantile, msb_cdf, msb_quantile
Base.include(DFDet, joinpath(@__DIR__, "DFDet_line.jl"))   # the line integral for the designs of rank one or two

# The three extensions of the limit (in the module DFDet, on its machinery: the outer contour of a transform model, the
# inner integrals, the rules of one evaluation): volatility and persistence regimes (the keyword vol of the t-ratio
# functions), a normal component (rho: the covariate-augmented t-ratio and the predictive regression), and the statistics
# of a pair of complex unit roots at a harmonic frequency (hegy_*)
Base.include(DFDet, joinpath(@__DIR__, "DFDet_vol.jl"))
Base.include(DFDet, joinpath(@__DIR__, "DFDet_cov.jl"))
Base.include(DFDet, joinpath(@__DIR__, "DFDet_seas.jl"))
using .DFDet: VolPath, volpath, cadf_c0, predictive_c0, hegy_t3_cdf, hegy_t3_pdf, hegy_t3_quantile, hegy_t4_cdf, hegy_t4_pdf,
              hegy_t4_quantile, hegy_f_cdf, hegy_f_quantile
export VolPath, volpath, cadf_c0, predictive_c0, hegy_t3_cdf, hegy_t3_pdf, hegy_t3_quantile, hegy_t4_cdf, hegy_t4_pdf,
       hegy_t4_quantile, hegy_f_cdf, hegy_f_quantile

# one interface over the two engines and the single inversion, and the choice between the engines (autoengine)
using SpecialFunctions: erfinv
include("merged.jl")
export coef_mom
DFExact.ROUTE[] = router
__init__() = (DFExact.ROUTE[] = router; nothing)

# A precompile workload: without it the first call of each function waits for compilation, about 6 s for most of them
# on an Apple M4 (1.4 to 8.7 s), with it a fraction of a second; it adds about 20 s to the precompilation (4.5 s to 22
# s) and 46 MB to the cache file.  The engine's __init__ does not run while the package
# is precompiled, so the workload builds the tables of K_0 first (k0tables!, the part of __init__ that the values depend
# on; the rest sets the number of tasks and looks for Apple's Accelerate), and the values computed here and the caches
# they fill are those of a loaded package (a check: the package and the engine file included directly give the same
# values bit for bit).  __init__ itself is not called here: the number of tasks and the vector library belong to the
# session that loads the package.  IntelVectorMath, where it is used, comes in through the package extension
# DFLimitIntelVectorMathExt (ext/), which this workload does not involve.
import PrecompileTools
PrecompileTools.@compile_workload begin
    DFExact.k0tables!()
    for cas in ("DK", "DU", "DT", "SU", "DG")
        ltratio_cdf(-2.0, cas, -5); ltratio_pdf(-2.0, cas, -5)
    end
    ltratio_cdf(0.0, "DK", 2; c0 = 2)
    ltratio_cdf(-2.0, "DU", -5; method = :double)
    ltratio_cdf(-2.0, "DU", -5; g0 = 1.5)
    ltratio_icdf(0.05, "DU")
    ltratio_mom(2, "DU")
    adfmax_cdf(-2.5, "DU"); adfmax_pdf(-2.5, "DU")
    ltratio_region((type = "wap", k = 3.6658, omega = 10), "DK", 0)
    # the general engine: a design beyond the classical cases, and a classical cell it takes
    ltratio_cdf(-2.0, perron(:C, 0.3), -5); ltratio_pdf(-2.0, perron(:C, 0.3), -5)
    ltratio_cdf(-1.0, "SU", -10; c0 = -10)
end

end # module
