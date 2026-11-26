# API reference

The exported functions, by topic, and the parts of the engines that the guide names. The functions of the engines
that are not exported are reached through their modules (`DFLimit.DFExact`, `DFLimit.DFDet`, `DFLimit.DFKan`) and
described by the comments of their files.

## The t-ratio

```@docs
ltratio_cdf
ltratio_pdf
ltratio_icdf
ltratio_mom
```

## The coefficient statistic

```@docs
coef_cdf
coef_icdf
coef_pdf
coef_mom
```

## ADFmax

```@docs
adfmax_cdf
adfmax_pdf
adfmax_icdf
```

## Regions, linear forms and the power envelope

```@docs
ltratio_region
ltratio_lincdf
power_envelope
ltratio_bisect
```

## Designs

```@docs
Determ
trend
perron
fdesign
ltrend
design_from_matrix
detrend_first
```

## The general engine

```@docs
dist_cdf
dist_pdf
dist_quantile
dist_moments
coef_quantile
coef_cdf_gp
lin_cdf
lin_pdf
lin_quantile
v_cdf
v_pdf
adfmax_quantile
region_prob
local_power
simulate_tratio
simulate_gls
simulate_first
empty_caches!
DFLimit.DFDet.coef_cdf
DFLimit.DFDet.adfmax_cdf
DFLimit.DFDet.adfmax_pdf
DFLimit.DFDet.power_envelope
```

## The statistics of the lead paper beyond the t-ratio

```@docs
pt_cdf
pt_pdf
pt_quantile
pt_power
pt_envelope
lm_cdf
lm_quantile
lm_power
mza_cdf
mza_quantile
msb_cdf
msb_quantile
```

## The extensions of the limit

Volatility and persistence regimes (the keyword `vol` of the t-ratio functions, [Volatility and persistence
regimes](@ref)), a normal component (the keyword `rho`, [A normal component: covariates and the predictive
regression](@ref)) and the statistics of a pair of complex unit roots ([Seasonal unit roots](@ref)).

```@docs
volpath
VolPath
cadf_c0
predictive_c0
hegy_t3_cdf
hegy_t3_pdf
hegy_t3_quantile
hegy_t4_cdf
hegy_t4_pdf
hegy_t4_quantile
hegy_f_cdf
hegy_f_quantile
DFLimit.DFDet.TransformModel
DFLimit.DFDet.VolModel
```

## The choice of engine

```@docs
DFLimit.autoengine
DFLimit.mom_start
DFLimit.route_dist2
```

## Modules and representations

```@docs
DFLimit
DFLimit.DFExact
DFLimit.DFExact.tr_limdist
DFLimit.DFExact.tr_contour
DFLimit.DFDet
DFLimit.DFKan
```

## The building blocks of the general engine

The module `DFLimit.DFDet` exports, besides the functions above, the parts from which they are built: the prepared
design and its model with a start, the joint moment generating function and its determinant, the roots and the outer
contour, and the evaluation of the inversion on paths on a given contour.

```@docs
DFLimit.DFDet.Prep
DFLimit.DFDet.Model
DFLimit.DFDet.model
DFLimit.DFDet.prep_gls
DFLimit.DFDet.regmat
DFLimit.DFDet.Dval
DFLimit.DFDet.Qf
DFLimit.DFDet.logmgf
DFLimit.DFDet.logMtilde
DFLimit.DFDet.mgfparts
DFLimit.DFDet.tparts
DFLimit.DFDet.tstar
DFLimit.DFDet.rootinfo
DFLimit.DFDet.rootpath
DFLimit.DFDet.CPath
DFLimit.DFDet.cpath
DFLimit.DFDet.linepath
DFLimit.DFDet.XSpec
DFLimit.DFDet.contour_eval
DFLimit.DFDet.contour_eval_specs
```
