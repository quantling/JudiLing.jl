using JudiLing
using Test

@testset "load dataset" begin
    data = JudiLing.load_dataset("data/latin_mini.csv")

    @test size(data,1) == 200

    @test typeof(data[!, "Word"]) == Vector{String}

    data = JudiLing.load_dataset("data/latin_train.csv", limit=2)

    @test size(data,1) == 2
end

@testset "random_split" begin
    # testing aliases
    data_train1, data_val1 = JudiLing.loading_data_randomly_split("data/latin_mini.csv", "data/random_test", "latin",
    val_ratio=0.1)
    data_train2, data_val2 = JudiLing.loading_data_random_split("data/latin_mini.csv", "data/random_test", "latin",
    val_ratio=0.1)
    data_train3, data_val3 = JudiLing.load_data_randomly_split("data/latin_mini.csv", "data/random_test", "latin",
    val_ratio=0.1)
    data_train4, data_val4 = JudiLing.load_data_random_split("data/latin_mini.csv", "data/random_test", "latin",
    val_ratio=0.1)

    @test data_train1 == data_train2
    @test data_train1 == data_train3
    @test data_train1 == data_train4
    @test data_val1 == data_val2
    @test data_val1 == data_val3
    @test data_val1 == data_val4

    # testing sizes
    data = JudiLing.load_dataset("data/latin_mini.csv")
    target_val_size = round(Int64, size(data,1) * 0.1)
    @test size(data_val1, 1) == target_val_size
    @test size(data_train1, 1) == size(data,1) - target_val_size

    # testing non-overlap (this only works because the words in latin_mini are unique)
    @test length(intersect(Set(data_train1.Word), Set(data_val1.Word))) == 0

    # clean up
    rm("data/random_test", recursive=true)
end


@testset "careful_split" begin
    # testing aliases
    data_train1, data_val1 = JudiLing.loading_data_careful_split("data/latin_mini.csv", "latin", "data/careful_test",
    ["Lexeme","Person","Number","Tense","Voice","Mood"],
    val_ratio=0.1, n_grams_target_col = "Word")
    data_train2, data_val2 = JudiLing.loading_data_careful_split("data/latin_mini.csv", "latin", "data/careful_test",
    ["Lexeme","Person","Number","Tense","Voice","Mood"],
    val_ratio=0.1, n_grams_target_col = "Word")
    data_train3, data_val3 = JudiLing.load_data_carefully_split("data/latin_mini.csv", "latin", "data/careful_test",
    ["Lexeme","Person","Number","Tense","Voice","Mood"],
    val_ratio=0.1, n_grams_target_col = "Word")
    data_train4, data_val4 = JudiLing.load_data_carefully_split("data/latin_mini.csv", "latin", "data/careful_test",
    ["Lexeme","Person","Number","Tense","Voice","Mood"],
    val_ratio=0.1, n_grams_target_col = "Word")

    @test data_train1 == data_train2
    @test data_train1 == data_train3
    @test data_train1 == data_train4
    @test data_val1 == data_val2
    @test data_val1 == data_val3
    @test data_val1 == data_val4

    # testing sizes
    data = JudiLing.load_dataset("data/latin_mini.csv")
    target_val_size = round(Int64, size(data, 1) * 0.1)
    @test size(data_val1, 1) == target_val_size
    @test size(data_train1, 1) == size(data,1) - target_val_size

    # testing non-overlap (this only works because the words in latin_mini are unique)
    @test length(intersect(Set(data_train1.Word), Set(data_val1.Word))) == 0

    # testing that all the unique features in the validation data occur in the training data
    for col in ["Lexeme","Person","Number","Tense","Voice","Mood"]
        @test length(setdiff(Set(data_val1[:, col]), Set(data_train1[:, col]))) == 0
    end

    # clean up
    rm("data/careful_test", recursive=true)
end
