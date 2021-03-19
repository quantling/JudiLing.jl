using JudiLing
using Test

@testset "test_combo tests" begin
    data_path = joinpath(@__DIR__, "data", "latin_mini.csv")
    data_output_dir = joinpath(@__DIR__, "out_data")
    output_dir = joinpath(@__DIR__, "out")
    JudiLing.test_combo(
        :train_only,
        val_sample_size = 100,
        data_path = data_path,
        data_prefix = "latin",
        data_output_dir = data_output_dir,
        n_features_columns = ["Lexeme","Person","Number","Tense","Voice","Mood"],
        n_grams_target_col = :Word,
        n_grams_tokenized = false,
        grams = 3,
        n_features_base = ["Lexeme"],
        n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
        output_dir = output_dir,
        verbose = false
        )

    JudiLing.test_combo(
        :random_split,
        val_sample_size = 100,
        data_path = data_path,
        data_prefix = "latin",
        data_output_dir = data_output_dir,
        n_features_columns = ["Lexeme","Person","Number","Tense","Voice","Mood"],
        n_grams_target_col = :Word,
        n_grams_tokenized = false,
        if_combined = true,
        grams = 3,
        n_features_base = ["Lexeme"],
        n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
        output_dir = output_dir,
        verbose = false
        )

    JudiLing.test_combo(
        :carefully_split,
        val_sample_size = 100,
        data_path = data_path,
        data_prefix = "latin",
        data_output_dir = data_output_dir,
        n_features_columns = ["Lexeme","Person","Number","Tense","Voice","Mood"],
        n_grams_target_col = :Word,
        n_grams_tokenized = false,
        grams = 3,
        n_features_base = ["Lexeme"],
        n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
        output_dir = output_dir,
        verbose = false
        )

    rm(data_output_dir, force = true, recursive = true)
    rm(output_dir, force = true, recursive = true)
end
