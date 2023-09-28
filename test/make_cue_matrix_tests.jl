using JudiLing
using CSV
using Test
using DataFrames

@testset "make cue matrix for latin" begin
    latin = DataFrame(CSV.File(joinpath("data", "latin_mini.csv")))
    latin_cue_obj_train = JudiLing.make_cue_matrix(
        latin,
        grams = 3,
        target_col = :Word,
        tokenized = false,
        keep_sep = false,
    )

    latin_val = latin[101:150, :]
    latin_cue_obj_val = JudiLing.make_cue_matrix(
        latin_val,
        latin_cue_obj_train,
        grams = 3,
        target_col = :Word,
        tokenized = false,
        keep_sep = false,
    )
end

@testset "make cue matrix for french" begin
    french = DataFrame(CSV.File(joinpath("data", "french_mini.csv")))
    french_cue_obj_train = JudiLing.make_cue_matrix(
        french,
        grams = 2,
        target_col = :Syllables,
        tokenized = true,
        sep_token = "-",
        keep_sep = true,
    )

    french_val = french[101:150, :]
    french_cue_obj_val = JudiLing.make_cue_matrix(
        french_val,
        french_cue_obj_train,
        grams = 2,
        target_col = :Syllables,
        tokenized = true,
        sep_token = "-",
        keep_sep = true,
    )
end

@testset "make cue matrix for utterance" begin
    utterance =
        DataFrame(CSV.File(joinpath("data", "utterance_mini.csv")))
    utterance_cue_obj_train = JudiLing.make_cue_matrix(
        utterance,
        grams = 3,
        target_col = :Words,
        tokenized = true,
        sep_token = ".",
        keep_sep = true,
    )

    utterance_val = utterance[101:150, :]
    utterance_cue_obj_val = JudiLing.make_cue_matrix(
        utterance_val,
        utterance_cue_obj_train,
        grams = 3,
        target_col = :Words,
        tokenized = true,
        sep_token = ".",
        keep_sep = true,
    )
end

@testset "make combined cue matrix" begin
    latin_full = DataFrame(CSV.File(joinpath(
        @__DIR__,
        "data",
        "latin_mini.csv",
    )))

    latin_train = latin_full[1:3, :]
    latin_val = latin_full[10:15, :]

    cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
        latin_train,
        latin_val,
        grams = 3,
        target_col = :Word,
        tokenized = false,
        keep_sep = false,
    )

    @test cue_obj_train.C[1, 3] == 1
    @test cue_obj_val.C[1, 3] == 0
    @test cue_obj_train.i2f[3] == "oco"
    @test cue_obj_val.i2f[3] == "oco"
    @test cue_obj_train.f2i["oco"] == 3
    @test cue_obj_val.f2i["oco"] == 3
    @test cue_obj_train.A[1, 3] == 0
    @test cue_obj_val.A[1, 3] == 0

    latin_train = DataFrame(CSV.File(joinpath(
        @__DIR__,
        "data",
        "latin_train.csv",
    )))
    latin_val = DataFrame(CSV.File(joinpath(
        @__DIR__,
        "data",
        "latin_val.csv",
    )))

    # check that indeed there are columns with differing string types
    println(typeof(latin_train.Word) != typeof(latin_val.Word))

    # but make combined cue matrix still runs
    cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
        latin_train,
        latin_val,
        grams = 3,
        target_col = :Word,
        tokenized = false,
        keep_sep = false,
    )
end
