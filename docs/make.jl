using Documenter

makedocs(;
    warnonly = [:cross_references, :autodocs_block],
    sitename = "spin",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md",
        "Bi-saturation (direct)" => "bisaturation.md",
        "Bi-saturation (indirect)" => "tir_saturation.md",
    ],
    checkdocs=:none,
)

deploydocs(
    repo = "github.com/control-toolbox/spin.git",
    devbranch = "main"
)