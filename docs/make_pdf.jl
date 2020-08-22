using JuLDL
using Documenter
using DocumenterLaTeX

makedocs(;
    modules=[JuLDL],
    authors="Xuefeng Luo",
    repo="https://github.com/MegamindHenry/JuLDL.jl/blob/{commit}{path}#L{line}",
    sitename="JuLDL.jl",
    format=LaTeX(platform = "none"),
    pages=[
        "Home" => "index.md",
        "Manual" => Any[
            "Make Cue Matrix" => "man/make_cue_matrix.md",
        ],
    ],
)