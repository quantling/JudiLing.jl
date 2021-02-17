using JudiLing
using CSV
using Test

@testset "output tests for latin" begin
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

        JudiLing.write2csv(
            res_train,
            latin_train,
            cue_obj_train,
            cue_obj_train,
            "res_latin_train.csv",
            grams = 3,
            tokenized = false,
            sep_token = nothing,
            start_end_token = "#",
            output_sep_token = "",
            path_sep_token = ":",
            target_col = :Word,
            root_dir = ".",
            output_dir = "test_out",
        )

        JudiLing.write2csv(
            res_val,
            latin_val,
            cue_obj_train,
            cue_obj_val,
            "res_latin_val.csv",
            grams = 3,
            tokenized = false,
            sep_token = nothing,
            start_end_token = "#",
            output_sep_token = "",
            path_sep_token = ":",
            target_col = :Word,
            root_dir = ".",
            output_dir = "test_out",
        )

        JudiLing.write2csv(
            gpi_train,
            "gpi_latin_train.csv",
            root_dir = ".",
            output_dir = "test_out",
        )

        JudiLing.write2csv(
            gpi_val,
            "gpi_latin_val.csv",
            root_dir = ".",
            output_dir = "test_out",
        )

        JudiLing.write2df(
            res_train,
            latin_train,
            cue_obj_train,
            cue_obj_train,
            grams = 3,
            tokenized = false,
            sep_token = nothing,
            start_end_token = "#",
            output_sep_token = "",
            path_sep_token = ":",
            target_col = :Word,
        )

        JudiLing.write2df(
            res_val,
            latin_val,
            cue_obj_train,
            cue_obj_val,
            grams = 3,
            tokenized = false,
            sep_token = nothing,
            start_end_token = "#",
            output_sep_token = "",
            path_sep_token = ":",
            target_col = :Word,
        )

        JudiLing.write2df(gpi_train)

        JudiLing.write2df(gpi_val)

        @test true
    catch e
        @test e == false
    finally
        path = joinpath(".", "test_out")
        rm(path, force = true, recursive = true)
    end
end

@testset "output tests for lexome matrix" begin
    try
        mkpath(joinpath(@__DIR__, "test_out"))
        latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))

        L1 = JudiLing.make_L_matrix(
            latin,
            ["Lexeme"],
            ["Person", "Number", "Tense", "Voice", "Mood"],
            ncol = 20,
        )

        L2 = JudiLing.make_L_matrix(latin, ["Lexeme"], ncol = 20)

        JudiLing.save_L_matrix(L1, joinpath(@__DIR__, "test_out", "L1.csv"))
        JudiLing.save_L_matrix(L2, joinpath(@__DIR__, "test_out", "L2.csv"))


        L3 = JudiLing.load_L_matrix(joinpath(@__DIR__, "test_out", "L1.csv"))
        L4 = JudiLing.load_L_matrix(joinpath(@__DIR__, "test_out", "L2.csv"))

        @test L3.L == L1.L
        @test L3.i2f == L1.i2f
        @test L3.f2i == L1.f2i
        @test L3.ncol == L1.ncol

        @test L4.L == L2.L
        @test L4.i2f == L2.i2f
        @test L4.f2i == L2.f2i
        @test L4.ncol == L2.ncol
    catch
        @test false
    finally
        path = joinpath(@__DIR__, "test_out")
        rm(path, force = true, recursive = true)
    end
end
