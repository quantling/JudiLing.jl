using JudiLing
using CSV
using Test
using DataFrames

@testset "output tests for latin" begin
    data_path = joinpath(@__DIR__, "data", "latin_mini.csv")
    output_dir_path = joinpath(@__DIR__, "out_data", "latin_mini.csv")

    JudiLing.lpo_cv_split(10, data_path)
    JudiLing.loo_cv_split(data_path)

    JudiLing.train_val_random_split(
        data_path,
        output_dir_path,
        "latin",
        train_sample_size = 100,
        val_sample_size = 10,
        )

    JudiLing.train_val_random_split(
        data_path,
        output_dir_path,
        "latin",
        val_ratio = 0.1,
        )

    JudiLing.train_val_careful_split(
        data_path,
        output_dir_path,
        "latin",
        ["Lexeme", "Person", "Number", "Tense", "Voice", "Mood"],
        train_sample_size = 100,
        val_sample_size = 10,
        n_grams_target_col = :Word,
        n_grams_tokenized = false,
        n_grams_sep_token = nothing,
        grams = 3,
        n_grams_keep_sep = false,
        start_end_token = "#",
        )

    JudiLing.train_val_careful_split(
        data_path,
        output_dir_path,
        "latin",
        ["Lexeme", "Person", "Number", "Tense", "Voice", "Mood"],
        val_ratio = 0.1,
        n_grams_target_col = :Word,
        n_grams_tokenized = false,
        n_grams_sep_token = nothing,
        grams = 3,
        n_grams_keep_sep = false,
        start_end_token = "#",
        )

    rm(output_dir_path, force = true, recursive = true)
end