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


@testset "CFBS" begin

    features = [[0., 0.1, 1.0, 3.2, 3.1], [-1., 0.5], [-0.]]

    C = JudiLing.make_cue_matrix_from_CFBS(features)
    target_C = [[0. 0.1 1.0 3.2 3.1]
                [-1. 0.5 0. 0. 0.]
                [-0. 0. 0. 0. 0.]]
    @test C == target_C

    C = JudiLing.make_cue_matrix_from_CFBS(features, ncol = 6)
    target_C = [[0. 0.1 1.0 3.2 3.1 0.]
                [-1. 0.5 0. 0. 0. 0.]
                [-0. 0. 0. 0. 0. 0.]]
    @test C == target_C

    C = JudiLing.make_cue_matrix_from_CFBS(features, pad_val = 1.)
    target_C = [[0. 0.1 1.0 3.2 3.1]
                [-1. 0.5 1. 1. 1.]
                [-0. 1. 1. 1. 1.]]
    @test C == target_C

    @test_throws TypeError JudiLing.make_cue_matrix_from_CFBS(features, pad_val = 1)

    @test_throws ErrorException JudiLing.make_cue_matrix_from_CFBS(features, ncol=4)

    features2 = [[0., 0.1, 1.0, 3.2, 3.1, 4.05]]

    C1, C2 = JudiLing.make_combined_cue_matrix_from_CFBS(features, features2)
    target_C1 = [[0. 0.1 1.0 3.2 3.1 0.]
                [-1. 0.5 0. 0. 0. 0.]
                [-0. 0. 0. 0. 0. 0.]]
    target_C2 = [0. 0.1 1.0 3.2 3.1 4.05]
    @test C1 == target_C1
    @test C2 == target_C2

    C1, C2 = JudiLing.make_combined_cue_matrix_from_CFBS(features, features2, ncol = 7)
    target_C1 = [[0. 0.1 1.0 3.2 3.1 0. 0.]
                [-1. 0.5 0. 0. 0. 0. 0.]
                [-0. 0. 0. 0. 0. 0. 0.]]
    target_C2 = [0. 0.1 1.0 3.2 3.1 4.05 0.]
    @test C1 == target_C1
    @test C2 == target_C2

    C1, C2 = JudiLing.make_combined_cue_matrix_from_CFBS(features, features2, pad_val = 1.)
    target_C1 = [[0. 0.1 1.0 3.2 3.1 1.]
                [-1. 0.5 1. 1. 1. 1.]
                [-0. 1. 1. 1. 1. 1.]]
    target_C2 = [0. 0.1 1.0 3.2 3.1 4.05]
    @test C1 == target_C1
    @test C2 == target_C2

    @test_throws TypeError JudiLing.make_combined_cue_matrix_from_CFBS(features, features2, pad_val = 1)

    @test_throws ErrorException JudiLing.make_combined_cue_matrix_from_CFBS(features, features2, ncol=4)
end
