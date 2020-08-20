using JuLDL
using Documenter

makedocs(;
    modules=[JuLDL],
    authors="Xuefeng Luo",
    repo="https://github.com/MegamindHenry/JuLDL.jl/blob/{commit}{path}#L{line}",
    sitename="JuLDL.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MegamindHenry.github.io/JuLDL.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MegamindHenry/JuLDL.jl",
)
