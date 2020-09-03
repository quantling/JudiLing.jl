using JudiLing
using Test

@testset "test_combo tests for french" begin
  try
    mkpath(joinpath("french_out"))
    test_io = open(joinpath("french_out", "out.log"), "w")

    learning_modes = [:cholesky, :pyndl, :wh]
    path_methods = [:learn_paths, :build_paths]

    for (lm, pm) in Iterators.product(learning_modes, path_methods)
      JudiLing.test_combo(
        joinpath("data", "french_mini.csv"),
        joinpath("french_out"),
        ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
        ["Lexeme"],
        ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
        data_prefix="french",
        max_test_data=nothing,
        split_max_ratio=0.1,
        is_full_A=false,
        n_grams_target_col=:Syllables,
        n_grams_tokenized=true,
        n_grams_sep_token="-",
        n_grams_keep_sep=true,
        grams=3,
        start_end_token="#",
        path_sep_token=":",
        learning_mode=lm,
        alpha=0.1,
        betas=(0.1,0.1),
        eta=0.1,
        n_epochs=nothing,
        path_method=pm,
        max_t=nothing,
        max_can=10,
        train_threshold=0.1,
        val_is_tolerant=false,
        val_threshold=(-100.0),
        val_tolerance=(-1000.0),
        val_max_tolerance=4,
        train_n_neighbors=2,
        val_n_neighbors=10,
        root_dir=@__DIR__,
        csv_dir="french_out",
        csv_prefix="french",
        random_seed=314,
        log_io=test_io,
        verbose=false)
    end

    close(test_io)

    @test true
  catch e
    @test e == false
  finally
    path = joinpath(".", "french_out")
    rm(path, force=true, recursive=true)
  end
end

@testset "test_combo tests for latin" begin
  try
    mkpath(joinpath("latin_out"))
    test_io = open(joinpath("latin_out", "out.log"), "w")

    learning_modes = [:cholesky, :pyndl, :wh]
    path_methods = [:learn_paths, :build_paths]

    for (lm, pm) in Iterators.product(learning_modes, path_methods)
      JudiLing.test_combo(
        joinpath("data", "latin_mini.csv"),
        joinpath("latin_out"),
        ["Lexeme","Person","Number","Tense","Voice","Mood"],
        ["Lexeme"],
        ["Person","Number","Tense","Voice","Mood"],
        data_prefix="latin",
        max_test_data=nothing,
        split_max_ratio=0.1,
        is_full_A=false,
        n_grams_target_col=:Word,
        n_grams_tokenized=false,
        n_grams_sep_token=nothing,
        n_grams_keep_sep=false,
        grams=3,
        start_end_token="#",
        path_sep_token=":",
        learning_mode=lm,
        alpha=0.1,
        betas=(0.1,0.1),
        eta=0.1,
        n_epochs=nothing,
        path_method=pm,
        max_t=nothing,
        max_can=10,
        train_threshold=0.1,
        val_is_tolerant=false,
        val_threshold=(-100.0),
        val_tolerance=(-1000.0),
        val_max_tolerance=4,
        train_n_neighbors=2,
        val_n_neighbors=10,
        root_dir=@__DIR__,
        csv_dir="latin_out",
        csv_prefix="latin",
        random_seed=314,
        log_io=test_io,
        verbose=false)
    end

    close(test_io)

    @test true
  catch e
    @test e == false
  finally
    path = joinpath(".", "french_out")
    rm(path, force=true, recursive=true)
  end
end