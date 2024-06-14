using JudiLing
using CSV
using Test
using DataFrames

@testset "display" begin
    latin =
        DataFrame(CSV.File(joinpath("data", "latin_mini.csv")))
    cue_obj = JudiLing.make_cue_matrix(
        latin,
        grams = 3,
        target_col = :Word,
        tokenized = false,
        keep_sep = false,
    )
    JudiLing.display_matrix(latin, :Word, cue_obj, cue_obj.C, :C)

    n_features = size(cue_obj.C, 2)
    S = JudiLing.make_S_matrix(
        latin,
        ["Lexeme"],
        ["Person", "Number", "Tense", "Voice", "Mood"],
        ncol = n_features,
    )
    JudiLing.display_matrix(latin, :Word, cue_obj, S, :S)

    G = JudiLing.make_transform_matrix(S, cue_obj.C)
    JudiLing.display_matrix(latin, :Word, cue_obj, G, :G)

    Chat = S * G
    JudiLing.display_matrix(latin, :Word, cue_obj, Chat, :Chat)

    F = JudiLing.make_transform_matrix(cue_obj.C, S)
    JudiLing.display_matrix(latin, :Word, cue_obj, F, :F)

    Shat = cue_obj.C * F
    JudiLing.display_matrix(latin, :Word, cue_obj, Shat, :Shat)

    A = cue_obj.A
    JudiLing.display_matrix(latin, :Word, cue_obj, A, :A)

    acc, R = JudiLing.eval_SC(Chat, cue_obj.C, R = true)
    JudiLing.display_matrix(latin, :Word, cue_obj, R, :R)

    pS_obj = JudiLing.make_pS_matrix(latin, features_col="Word")
    JudiLing.display_matrix(latin, :Word, pS_obj, pS_obj.pS, :pS)
end
