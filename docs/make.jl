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
        "Manual" => Any[
            "Make Cue Matrix" => "man/make_cue_matrix.md",
            "Make Semantic Matrix" => "man/make_semantic_matrix.md",
            "Cholesky" => "man/cholesky.md",
            "Make Adjacency Matrix" => "man/make_adjacency_matrix.md",
            "Make Yt Matrix" => "man/make_yt_matrix.md",
            "Find Paths" => "man/find_path.md"
            "Utils" => "man/utils.md"
        ],
        "All Manual index" => "man/all_manual.md"
    ],
)

deploydocs(;
    repo="github.com/MegamindHenry/JuLDL.jl",
)
