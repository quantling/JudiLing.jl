using JudiLing
using CSV
using Test
using DataFrames

@testset "eval_SC" begin
    A = [
        1 0 0
        0 1 0
        0 0 1
        1 0 0
        0 1 0
        0 0 1
        1 0 0
        0 1 0
        0 0 1
        0 0 1
    ]

    B = [
        1 0 0
        0 1 0
        0 0 1
        1 0 0
        1 0 0
        0 0 1
        1 0 0
        0 1 0
        0 0 1
        0 0 1
    ]

    C = [
        1 0 0
        0 1 0
        0 0 1
        1 0 0
        1 0 0
        1 0 0
        1 0 0
        1 0 0
        1 0 0
        0 0 1
    ]

    @test JudiLing.eval_SC(A, B) == 0.9
    @test JudiLing.eval_SC(A, C) == 0.6
    @test JudiLing.eval_SC(C, B) == 0.7
    @test JudiLing.eval_SC(A, B, 2) == 0.9
    @test JudiLing.eval_SC(A, B, 3) == 0.9
    @test JudiLing.eval_SC(A, B, 4) == 0.9
    @test JudiLing.eval_SC(A, B, 4) == 0.9
    @test JudiLing.eval_SC(A, C, 2) == 0.6
    @test JudiLing.eval_SC(C, B, 2) == 0.7
end

@testset "eval_SC homophone" begin
    data = DataFrame(
        A = ["aba", "aba", "abc", "abd"],
        B = ["M", "F", "F", "M"],
        C = ["S", "L", "S", "L"],
    )

    cue_obj = JudiLing.make_cue_matrix(
        data,
        grams = 3,
        target_col = :A,
        tokenized = false,
        keep_sep = false,
    )

    n_features = size(cue_obj.C, 2)

    S = JudiLing.make_S_matrix(data, ["B"], ["C"], add_noise = true)

    G = JudiLing.make_transform_matrix(S, cue_obj.C)

    Chat = S * G

    @test JudiLing.eval_SC(Chat, cue_obj.C, data, :A) == 1
    @test JudiLing.eval_SC(Chat, cue_obj.C, data, :A, 2) == 1
    @test JudiLing.eval_SC(Chat, cue_obj.C, data, :A, 3) == 1
end

@testset "eval_SC_loose" begin
    latin = DataFrame(
        Word = ["ABC", "BCD", "CDE", "BCD"],
        Lexeme = ["A", "B", "C", "B"],
        Person = ["B", "C", "D", "D"],
    )

    cue_obj = JudiLing.make_cue_matrix(
        latin,
        grams = 3,
        target_col = :Word,
        tokenized = false,
        keep_sep = false,
    )

    n_features = size(cue_obj.C, 2)
    S = JudiLing.make_S_matrix(
        latin,
        [:Lexeme],
        [:Person],
        ncol = n_features,
    )

    G = JudiLing.make_transform_matrix(S, cue_obj.C)
    Chat = S * G
    F = JudiLing.make_transform_matrix(cue_obj.C, S)
    Shat = cue_obj.C * F

    @test JudiLing.eval_SC_loose(Chat, cue_obj.C, 1) == 0.75
    @test JudiLing.eval_SC_loose(Shat, S, 1) == 0.75
    @test JudiLing.eval_SC_loose(Chat, cue_obj.C, 1, latin, :Word) == 1
    @test JudiLing.eval_SC_loose(Shat, S, 1, latin, :Word) == 1

    for k = 2:4
        @test JudiLing.eval_SC_loose(Chat, cue_obj.C, k) == 1
        @test JudiLing.eval_SC_loose(Shat, S, k) == 1
        @test JudiLing.eval_SC_loose(Chat, cue_obj.C, k, latin, :Word) == 1
        @test JudiLing.eval_SC_loose(Shat, S, k, latin, :Word) == 1
    end

    latin_train = DataFrame(
        Word = ["ABC", "BCD", "CDE", "BCD"],
        Lexeme = ["A", "B", "C", "B"],
        Person = ["B", "C", "D", "D"],
    )

    latin_val = DataFrame(
        Word = ["ABC", "BCD"],
        Lexeme = ["A", "B"],
        Person = ["B", "C"],
    )

    cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
        latin_train,
        latin_val,
        grams = 3,
        target_col = :Word,
        tokenized = false,
        keep_sep = false,
    )

    n_features = size(cue_obj_train.C, 2)
    S_train, S_val = JudiLing.make_combined_S_matrix(
        latin_train,
        latin_val,
        [:Lexeme],
        [:Person],
        ncol = n_features,
        add_noise = false
    )

    G = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)
    Chat_val = S_val * G
    Chat_train = S_train * G
    F = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)
    Shat_val = cue_obj_val.C * F
    Shat_train = cue_obj_train.C * F

    @test JudiLing.eval_SC_loose(Chat_val, cue_obj_val.C, cue_obj_train.C, 1) >= 0.5
    @test JudiLing.eval_SC_loose(Chat_val, cue_obj_val.C, cue_obj_train.C, 2) == 1
    @test JudiLing.eval_SC_loose(Shat_val, S_val, S_train, 1) >= 0.5
    @test JudiLing.eval_SC_loose(Shat_val, S_val, S_train, 2) == 1
    @test JudiLing.eval_SC_loose(Chat_val, cue_obj_val.C, cue_obj_train.C, 1, latin_val, latin_train, :Word) == 1
    @test JudiLing.eval_SC_loose(Chat_val, cue_obj_val.C, cue_obj_train.C, 2, latin_val, latin_train, :Word) == 1
    @test JudiLing.eval_SC_loose(Shat_val, S_val, S_train, 1, latin_val, latin_train, :Word) == 1
    @test JudiLing.eval_SC_loose(Shat_val, S_val, S_train, 2, latin_val, latin_train, :Word) == 1
end

@testset "accuracy_comprehension" begin
    latin = DataFrame(
        Word = ["ABC", "BCD", "CDE", "BCD"],
        Lexeme = ["A", "B", "C", "B"],
        Person = ["B", "C", "D", "D"],
    )

    cue_obj = JudiLing.make_cue_matrix(
        latin,
        grams = 3,
        target_col = :Word,
        tokenized = false,
        keep_sep = false,
    )

    n_features = size(cue_obj.C, 2)
    S = JudiLing.make_S_matrix(
        latin,
        [:Lexeme],
        [:Person],
        ncol = n_features,
    )

    G = JudiLing.make_transform_matrix(S, cue_obj.C)
    Chat = S * G
    F = JudiLing.make_transform_matrix(cue_obj.C, S)
    Shat = cue_obj.C * F

    acc_comp = JudiLing.accuracy_comprehension(
        S,
        Shat,
        latin;
        target_col = :Word,
        base = [:Lexeme],
        inflections = [:Person],
    )
end

@testset "eval_acc tests" begin
    latin_train =
        DataFrame(CSV.File(joinpath("data", "latin_mini.csv")))
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

    Chat_train = S_train * G_train
    Chat_val = S_val * G_train

    F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

    Shat_train = cue_obj_train.C * F_train
    Shat_val = cue_obj_val.C * F_train

    A = cue_obj_train.A

    max_t = JudiLing.cal_max_timestep(latin_train, latin_val, :Word)

    res_train, gpi_train = JudiLing.learn_paths(
        latin_train,
        latin_train,
        cue_obj_train.C,
        S_train,
        F_train,
        Chat_train,
        A,
        cue_obj_train.i2f,
        cue_obj_train.f2i,
        gold_ind = cue_obj_train.gold_ind,
        Shat_val = Shat_train,
        check_gold_path = true,
        max_t = max_t,
        max_can = 10,
        grams = 3,
        threshold = 0.1,
        tokenized = false,
        sep_token = "_",
        keep_sep = false,
        target_col = :Word,
        issparse = :dense,
        verbose = false,
    )

    res_val, gpi_val = JudiLing.learn_paths(
        latin_train,
        latin_val,
        cue_obj_train.C,
        S_val,
        F_train,
        Chat_val,
        A,
        cue_obj_train.i2f,
        cue_obj_train.f2i,
        gold_ind = cue_obj_val.gold_ind,
        Shat_val = Shat_val,
        check_gold_path = true,
        max_t = max_t,
        max_can = 10,
        grams = 3,
        threshold = 0.1,
        is_tolerant = true,
        tolerance = 0.1,
        max_tolerance = 0,
        tokenized = false,
        sep_token = "-",
        keep_sep = false,
        target_col = :Word,
        issparse = :dense,
        verbose = false,
    )

    JudiLing.eval_acc(res_train, cue_obj_train.gold_ind)
    JudiLing.eval_acc(res_train, cue_obj_train)

    JudiLing.eval_acc(res_val, cue_obj_val.gold_ind)
    JudiLing.eval_acc(res_val, cue_obj_val)

    res_train = JudiLing.learn_paths(
        latin_train,
        cue_obj_train,
        S_train,
        F_train,
        Chat_train,
        verbose = false,
    )

    JudiLing.eval_acc(res_train, cue_obj_train.gold_ind)
    JudiLing.eval_acc(res_train, cue_obj_train)
end

@testset "eval_SC train and val tests" begin
    Chat_train = [
    1 0 0
    0 1 0
    0 0 1
    ]

    C_train = [
    1 0 0
    0 1 0
    0 0 1
    ]

    Chat_val = [
    1 1 0
    0 1 1
    1 0 1
    ]

    C_val = [
    1 1 0
    0 1 1
    1 0 1
    ]

    @test JudiLing.eval_SC(Chat_train, C_train) == 1
    @test JudiLing.eval_SC(Chat_val, C_val) == 1
    @test JudiLing.eval_SC(Chat_train, C_train, C_val) == 1
    @test JudiLing.eval_SC(Chat_val, C_val, C_train) == 1

    Chat_train = [
    1 0 0
    0 1 0
    0 0 1
    ]

    C_train = [
    1 0 0
    0 1 0
    0 0 1
    ]

    Chat_val = [
    1 0 0
    0 1 1
    1 0 1
    ]

    C_val = [
    0.9 0.1 0
    0 1 1
    1 0 1
    ]

    @test JudiLing.eval_SC(Chat_train, C_train) == 1
    @test JudiLing.eval_SC(Chat_val, C_val) == 1
    @test JudiLing.eval_SC(Chat_train, C_train, C_val) == 1
    @test JudiLing.eval_SC(Chat_val, C_val, C_train) == round(2/3, digits=4)

    latin_train =
        DataFrame(CSV.File(joinpath("data", "latin_mini.csv")))
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

    Chat_train = S_train * G_train
    Chat_val = S_val * G_train

    F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

    Shat_train = cue_obj_train.C * F_train
    Shat_val = cue_obj_val.C * F_train

    JudiLing.eval_SC(Chat_train, cue_obj_train.C, cue_obj_val.C)
    JudiLing.eval_SC(Chat_val, cue_obj_val.C, cue_obj_train.C)

    @test JudiLing.eval_SC(Chat_train, cue_obj_train.C, cue_obj_val.C, latin_train, latin_val, :Word) ≈ 1
    @test JudiLing.eval_SC(Chat_val, cue_obj_val.C, cue_obj_train.C, latin_val, latin_train, :Word) > 0.15
end
