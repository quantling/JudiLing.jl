using JudiLing
using Documenter
using DocumenterLaTeX

makedocs(;
    modules=[JudiLing],
    authors="Xuefeng Luo",
    repo="https://github.com/MegamindHenry/JudiLing.jl/blob/{commit}{path}#L{line}",
    sitename="JudiLing.jl",
    format=LaTeX(platform = "none"),
    pages=[
        "Home" => "index.md",
        "Manual" => Any[
            "Make Cue Matrix" => "man/make_cue_matrix.md",
        ],
    ],
)