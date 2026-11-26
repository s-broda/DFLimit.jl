# The documentation of DFLimit.jl (Documenter).  Locally, from the root of the repository:
#     julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
#     julia --project=docs docs/make.jl
# The site is written to docs/build.  On GitHub Actions (.github/workflows/Documentation.yml) the same script also
# deploys it, from the branch main and from tags, to the branch gh-pages.
using Documenter
using DFLimit

makedocs(
    sitename = "DFLimit.jl",
    authors = "Yong Bao, Simon A. Broda and Raymond Kan",
    modules = [DFLimit],
    checkdocs = :exports,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", edit_link = "main",
                             size_threshold_warn = 200 * 2^10, size_threshold = 400 * 2^10),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Guide" => [
            "Case codes and arguments" => "cases.md",
            "Two engines" => "engines.md",
            "Which engine" => "which.md",
            "The rules of one evaluation" => "rules.md",
            "The devices of Bao and Kan" => "devices.md",
        ],
        "Statistics beyond the t-ratio" => "statistics.md",
        "API reference" => "api.md",
    ],
)

deploydocs(repo = "github.com/s-broda/DFLimit.jl.git", devbranch = "main", push_preview = false)
