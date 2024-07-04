using Documenter
#using DocumenterMermaid
using OptimalControl

makedocs(;
    warnonly = [:cross_references, :autodocs_block],
    sitename = "spin.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Bisaturation" => "bisaturation.md",
    ],
    checkdocs=:none,
)

deploydocs(
    repo = "github.com/control-toolbox/spin.jl.git",
    devbranch = "main"
)
