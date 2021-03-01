using JudiLing
using Test
using CSV
using DataFrames

@testset "make cue matrix for latin" begin
    try
        latin = DataFrame(CSV.File(joinpath("data", "latin_mini.csv")))
        cue_obj = JudiLing.make_cue_matrix(
            latin,
            grams = 3,
            target_col = :Word,
            tokenized = false,
            keep_sep = false,
        )
        JudiLing.make_Yt_matrix(2, latin, cue_obj.f2i, target_col = :Word)
        @test true
    catch
        @test false
    end
end
