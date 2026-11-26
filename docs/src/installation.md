# Installation

The package needs Julia 1.10 or later. It is registered in Julia's General registry; install it from the package
prompt (press `]` at the `julia>` prompt) with

```
pkg> add DFLimit
```

or, equivalently, `using Pkg; Pkg.add("DFLimit")`. The development version, which may hold changes that are not yet
released, comes from the GitHub repository: `pkg> add DFLimit#main` (or
`Pkg.add(url = "https://github.com/s-broda/DFLimit.jl")`), or, from a local clone,
`Pkg.develop(path = "/path/to/DFLimit.jl")`. The dependencies (SpecialFunctions, QuadGK, StaticArrays, PrecompileTools and standard
libraries) are installed with it. The first `using DFLimit` precompiles the package, which takes about half a minute: it runs a few of
the functions once, so that their first call in a session does not wait several seconds for compilation.
`Pkg.test("DFLimit")` runs the tests (about six minutes; they compare the values with values stored by the companion
paper's code, and end with the engine's own checks in `test/engine/`). The longer checks of the engine and the scripts that
write its stored values are in `dev/` (see `dev/README.md`). The documentation is built with Documenter from `docs/`:
`julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'`, then
`julia --project=docs docs/make.jl`, which writes the pages to `docs/build`.

## Threads and the vector library

The engine is multithreaded: start Julia with `julia -t auto` (or set the environment variable
`JULIA_NUM_THREADS=auto`). It runs its loops on `DFLimit.DFExact.NWORKERS[]` tasks, by default the smaller of the
number of threads and the number of cores; set it to change that.

The batched sums of the line integral take the exponentials of all their nodes at once from a vector library where one
is at hand. On macOS that is Apple's Accelerate framework, part of the system, which the package uses without being
asked. On x86-64 Linux and Windows it is IntelVectorMath (MKL), an optional dependency: install it in your environment
(`pkg> add IntelVectorMath`) and load it in the session, `using IntelVectorMath` before or after `using DFLimit`, and
Julia loads the package extension `DFLimitIntelVectorMathExt`, which makes the engine use it. Without either the engine
uses plain loops. `DFLimit.DFExact.vecmath()` says which is in use: `"Accelerate"`, `"IntelVectorMath"` or `"none"`.
