using JudiLing
using CSV
using Test

@testset "make full adjacency matrix" begin
    try
        i2f = Dict([(1, "#ab"), (2, "abc"), (3, "bc#"), (4, "#bc"), (5, "ab#")])
        JudiLing.make_adjacency_matrix(i2f)
        @test true
    catch e
        @test e == false
    end
end

@testset "make combined adjacency matrix" begin
    try
        latin_full = CSV.DataFrame!(CSV.File(joinpath(
            @__DIR__,
            "data",
            "latin_mini.csv",
        )))

        latin_train = latin_full[1:3, :]
        latin_val = latin_full[10:15, :]

        A = JudiLing.make_combined_adjacency_matrix(
            latin_train,
            latin_val,
            grams = 3,
            target_col = :Word,
            tokenized = false,
            keep_sep = false,
        )

        @test A[1, 2] == 1
        @test A[2, 3] == 1
        @test A[3, 4] == 1
        @test A[4, 5] == 1
        @test A[2, 6] == 1
        @test A[6, 7] == 1
    catch e
        @test false
    end
end
