using PyCall
using Flux
using JudiLing
using Documenter

makedocs(;
    modules=[JudiLing],
    authors="Xuefeng Luo",
    repo="https://github.com/MegamindHenry/JudiLing.jl/blob/{commit}{path}#L{line}",
    sitename="JudiLing.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MegamindHenry.github.io/JudiLing.jl",
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
            "Find Paths" => "man/find_path.md",
            "Evaluation" => "man/eval.md",
            "Output" => "man/output.md",
            "Test Combo" => "man/test_combo.md",
            "Display" => "man/display.md",
            "Utils" => "man/utils.md",
            "Pickle" => "man/pickle.md",
            "Pyndl" => "man/pyndl.md",
            "Widrow-Hoff Learning" => "man/wh.md",
            "Deep learning" => "man/deep_learning.md",
            "Measures function" => "man/measures_func.md"
        ],
        "All Manual index" => "man/all_manual.md"
    ],
)

deploydocs(;
    repo="github.com/MegamindHenry/JudiLing.jl",
)
