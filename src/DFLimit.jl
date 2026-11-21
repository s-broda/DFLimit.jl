"""
    DFLimit

The exact limiting distribution of the Dickey-Fuller t-ratio: the distribution function, density, quantiles and
moments of the limits of the Dickey-Fuller t-ratios under local alternatives, by the exact representations of Bao,
Broda and Kan, "The Exact Limiting Distribution of the Dickey-Fuller t-Ratio and Some Related Statistics".

The numbers come from the Julia engine of the paper, the module `DFLimit.DFExact` (the file `src/DFExact.jl`, with
`src/df_series.jl`); `DFLimit` re-exports its exported functions (`ltratio_cdf`,
`ltratio_pdf`, `ltratio_icdf`, `ltratio_mom`, `adfmax_cdf`, `adfmax_pdf`, `adfmax_icdf`, `coef_cdf`, `coef_icdf`,
`ltratio_region`, `ltratio_lincdf`, `power_envelope`, `ltratio_bisect`), and the functions it does not export are
reached as `DFLimit.DFExact.cdf` and so on.  The engine is multithreaded: start Julia with `-t auto`.

The general engine of the companion paper on general deterministic terms is the module `DFLimit.DFDet` (the file
`src/DFDet.jl`); the classical functions take its designs (`trend`, `perron`, `fdesign`, ...) in place of the case, and
`DFLimit.autoengine` chooses the engine of every call (`src/merged.jl`; the README, "Two engines").
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
include("DFDet.jl")
using .DFDet: Determ, trend, perron, fdesign, ltrend, design_from_matrix, detrend_first, dist_cdf, dist_pdf, dist_quantile,
              dist_moments, coef_pdf, coef_quantile, coef_cdf_gp, local_power, lin_cdf, lin_pdf, lin_quantile, v_cdf, v_pdf,
              adfmax_quantile, region_prob, simulate_tratio, simulate_gls, simulate_first, empty_caches!
export Determ, trend, perron, fdesign, ltrend, design_from_matrix, detrend_first, dist_cdf, dist_pdf, dist_quantile,
       dist_moments, coef_pdf, coef_quantile, coef_cdf_gp, local_power, lin_cdf, lin_pdf, lin_quantile, v_cdf, v_pdf,
       adfmax_quantile, region_prob, simulate_tratio, simulate_gls, simulate_first, empty_caches!

# the statistics of the paper on general deterministic terms: the point-optimal statistic and the envelope for any design,
# the LM statistic, MZ_alpha and MSB (in the module DFDet, whose file stays that of df-deterministics)
Base.include(DFDet, joinpath(@__DIR__, "DFDet_paper2.jl"))
using .DFDet: pt_cdf, pt_pdf, pt_quantile, pt_power, pt_envelope, lm_cdf, lm_quantile, lm_power, mza_cdf, mza_quantile, msb_cdf, msb_quantile
export pt_cdf, pt_pdf, pt_quantile, pt_power, pt_envelope, lm_cdf, lm_quantile, lm_power, mza_cdf, mza_quantile, msb_cdf, msb_quantile

# one interface over the two engines, and the choice between them (autoengine)
using SpecialFunctions: erfinv
include("merged.jl")
DFExact.ROUTE[] = router
__init__() = (DFExact.ROUTE[] = router; nothing)

# A precompile workload: without it the first call of each function waits for compilation, about 6 s for most of them
# on an Apple M4 (1.4 to 8.7 s), with it a fraction of a second; it adds about 20 s to the precompilation (4.5 s to 22
# s) and 46 MB to the cache file.  The engine's __init__ does not run while the package
# is precompiled, so the workload builds the tables of K_0 first (k0tables!, the part of __init__ that the values depend
# on; the rest sets the number of tasks and the vector library), and the values computed here and the caches they fill
# are those of a loaded package (a check: the package and the engine file included directly give the same values bit for
# bit).  __init__ itself is not called here: on x86-64 Linux and Windows it loads IntelVectorMath from the active
# environment if it is there, which a precompilation should not do.
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
