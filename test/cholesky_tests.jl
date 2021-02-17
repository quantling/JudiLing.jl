using JudiLing
using CSV
using Test
using SparseArrays

@testset "cholesky transformation matrix" begin
    try
        C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
        S = [1 0 1 0; 1 1 0 0; 0 0 1 1]

        JudiLing.make_transform_matrix(C, S)

        C = sparse(C)
        S = Matrix(S)
        JudiLing.make_transform_matrix(C, S)
        JudiLing.make_transform_matrix(S, C)

        C = sparse(C)
        S = sparse(S)
        JudiLing.make_transform_matrix(C, S)

        fac_C = JudiLing.make_transform_fac(C)
        JudiLing.make_transform_matrix(fac_C, C, S)

        C = Matrix(C)
        fac_C = JudiLing.make_transform_fac(C)
        JudiLing.make_transform_matrix(fac_C, C, S)

        @test true
    catch e
        @test e == false
    end
end

@testset "cholesky transformation matrix for latin" begin
    try
        latin_train =
            CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
        cue_obj_train = JudiLing.make_cue_matrix(
            latin_train,
            grams = 3,
            target_col = :Word,
            tokenized = false,
            keep_sep = false,
        )

        latin_val = latin_train[101:150, :]
        cue_obj_val = JudiLing.make_cue_matrix(
            latin_val,
            cue_obj_train,
            grams = 3,
            target_col = :Word,
            tokenized = false,
            keep_sep = false,
        )

        n_features = size(cue_obj_train.C, 2)

        S_train, S_val = JudiLing.make_S_matrix(
            latin_train,
            latin_val,
            ["Lexeme"],
            ["Person", "Number", "Tense", "Voice", "Mood"],
            ncol = n_features,
        )

        G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)

        F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

        @test true
    catch e
        @test e == false
    end
end
