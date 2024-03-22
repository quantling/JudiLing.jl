using Test
using Flux
using JudiLing
using CSV, DataFrames

train = DataFrame(CSV.File(joinpath("data", "latin_train.csv")))
val = DataFrame(CSV.File(joinpath("data", "latin_val.csv")))

cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(train, val, target_col="Word", grams=3)
S_train, S_val = JudiLing.make_combined_S_matrix(train, val, ["Lexeme"],
["Person", "Number", "Tense", "Voice", "Mood"])

@testset "basic setup" begin

    res = JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                "test.bson",
                                batchsize=3)


    model = res.model

    @test model isa Chain

    Shat_train = JudiLing.predict_from_deep_model(model, cue_obj_train.C)

    @test JudiLing.eval_SC(Shat_train, S_train) ≈ 1.0

    res = JudiLing.get_and_train_model(S_train, cue_obj_train.C,
                                "test.bson",
                                batchsize=3)

    model = res.model

    @test model isa Chain

    Chat_train = JudiLing.predict_from_deep_model(model, S_train)

    @test JudiLing.eval_SC(Chat_train, cue_obj_train.C) ≈ 1.0

end

@testset "validation data" begin

    res = JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                cue_obj_val.C,
                                S_val,
                                train, val,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                batchsize=3)

    model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

    @test model isa Chain
    @test length(losses_train) == length(losses_val) == length(accs_val) == 100

    Shat_train = JudiLing.predict_from_deep_model(model, cue_obj_train.C)
    Shat_val = JudiLing.predict_from_deep_model(model, cue_obj_val.C)

    @test JudiLing.eval_SC(Shat_train, S_train) ≈ 1.0
    @test JudiLing.eval_SC(Shat_val, S_val) < 1.0
    @test JudiLing.eval_SC(Shat_val, S_val) >= 0.0
    @test Flux.mse(Shat_val', S_val') ≈ findmin(losses_val)[1]

    res = JudiLing.get_and_train_model(S_train,
                                cue_obj_train.C,
                                S_val,
                                cue_obj_val.C,
                                train, val,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                batchsize=3)

    model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

    @test model isa Chain

    Chat_train = JudiLing.predict_from_deep_model(model, S_train)
    Chat_val = JudiLing.predict_from_deep_model(model, S_val)

    @test JudiLing.eval_SC(Chat_train, cue_obj_train.C) ≈ 1.0
    @test JudiLing.eval_SC(Chat_val, cue_obj_val.C) < 1.0
    @test JudiLing.eval_SC(Chat_val, cue_obj_val.C) >= 0.0
    @test Flux.mse(Chat_val', cue_obj_val.C')  ≈ findmin(losses_val)[1]

end

@testset "early stopping" begin

    train_es = train[1:2,:]
    val_es = val

    cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(train_es, val_es, target_col="Word", grams=3)
    S_train, S_val = JudiLing.make_combined_S_matrix(train_es, val_es, ["Lexeme"],
    ["Person", "Number", "Tense", "Voice", "Mood"])


    res = JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                cue_obj_val.C,
                                S_val,
                                train_es, val_es,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                early_stopping=20,
                                n_epochs=1000,
                                batchsize=2)

    model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

    @test model isa Chain

    @test length(losses_train) == length(losses_val) == length(accs_val) < 1000

    Shat_train = JudiLing.predict_from_deep_model(model, cue_obj_train.C)
    Shat_val = JudiLing.predict_from_deep_model(model, cue_obj_val.C)

    @test JudiLing.eval_SC(Shat_train, S_train) ≈ 1.0
    @test Flux.mse(Shat_val', S_val') ≈ findmin(losses_val)[1]

    res = JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                cue_obj_val.C,
                                S_val,
                                train_es, val_es,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                early_stopping=20,
                                optimise_for_acc = true,
                                batchsize=2)

    model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

    Shat_train = JudiLing.predict_from_deep_model(model, cue_obj_train.C)
    Shat_val = JudiLing.predict_from_deep_model(model, cue_obj_val.C)

    @test JudiLing.eval_SC(Shat_train, S_train) ≈ 1.0
    @test JudiLing.eval_SC(Shat_val, S_val, S_train, val_es, train_es, :Word) ≈ findmax(accs_val)[1]

end

@testset "changing hyperparameters" begin

    cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(train, val, target_col="Word", grams=3)
    S_train, S_val = JudiLing.make_combined_S_matrix(train, val, ["Lexeme"],
    ["Person", "Number", "Tense", "Voice", "Mood"])

    @testset "batchsize" begin
        res = JudiLing.get_and_train_model(S_train,
                                    cue_obj_train.C,
                                    S_val,
                                    cue_obj_val.C,
                                    train, val,
                                    :Word,
                                    "test.bson",
                                    return_losses=true,
                                    batchsize=2)

        model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val
    end

    @testset "number of epochs" begin
        res = JudiLing.get_and_train_model(S_train,
                                    cue_obj_train.C,
                                    S_val,
                                    cue_obj_val.C,
                                    train, val,
                                    :Word,
                                    "test.bson",
                                    return_losses=true,
                                    batchsize=3,
                                    n_epochs=500)

        model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

        @test length(losses_train) == length(losses_val) == length(accs_val) == 500
    end

    @testset "optimizer" begin

    res = JudiLing.get_and_train_model(S_train,
                                cue_obj_train.C,
                                S_val,
                                cue_obj_val.C,
                                train, val,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                batchsize=3,
                                optimizer=Flux.Adam(0.00001))

    model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

    res2 = JudiLing.get_and_train_model(S_train,
                                cue_obj_train.C,
                                S_val,
                                cue_obj_val.C,
                                train, val,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                batchsize=3,
                                optimizer=Flux.Adam(0.1))

    model2, losses_train2, losses_val2, accs_val2 = res2.model, res2.losses_train, res2.losses_val, res2.accs_val

    @test losses_train[end] - losses_train2[end] > 0.1
    end

    @testset "hidden dim" begin
        res = JudiLing.get_and_train_model(S_train,
                                    cue_obj_train.C,
                                    S_val,
                                    cue_obj_val.C,
                                    train, val,
                                    :Word,
                                    "test.bson",
                                    return_losses=true,
                                    batchsize=3,
                                    hidden_dim=200)

        model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

        @test size(Flux.params(model[1])[1],1) == 200
    end

    @testset "supplying model" begin
        model = Chain(Dense(size(S_train,2)=>500), Dense(500=>500), Dense(500=>size(cue_obj_train.C, 2)))

        res = JudiLing.get_and_train_model(S_train,
                                    cue_obj_train.C,
                                    S_val,
                                    cue_obj_val.C,
                                    train, val,
                                    :Word,
                                    "test.bson",
                                    return_losses=true,
                                    batchsize=3,
                                    model=model)

        model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

        @test size(Flux.params(model[1])[1],1) == 500
        @test size(Flux.params(model[2])[1],1) == 500
        @test size(Flux.params(model[3])[1],1) == size(cue_obj_train.C, 2)
        @test length(model) == 3

    end

    @testset "loss function" begin
        model_prod = Chain(
            Dense(size(S_train, 2) => 1000, relu),   # activation function inside layer
            Dense(1000 => size(cue_obj_train.C, 2)),
            sigmoid) |> gpu
        res = JudiLing.get_and_train_model(S_train,
                                    cue_obj_train.C,
                                    S_val,
                                    cue_obj_val.C,
                                    train,
                                    val,
                                    :Word,
                                    "test.bson",
                                    batchsize=3,
                                    loss_func=Flux.binarycrossentropy,
                                    model=model_prod,
                                    return_losses=true)

        model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val

        Chat_train = JudiLing.predict_from_deep_model(model, S_train)
        Chat_val = JudiLing.predict_from_deep_model(model, S_val)

        # train and validation data are too small for this test
        #@test JudiLing.eval_SC(Chat_train, cue_obj_train.C) ≈ 1.0
        @test Flux.binarycrossentropy(Chat_val', Matrix(cue_obj_val.C)') ≈ findmin(losses_val)[1]
        @test Flux.mse(Chat_val', cue_obj_val.C') != findmin(losses_val)[1]
    end

end

@testset "learn paths" begin
    cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(train, val, target_col="Word", grams=3)
    S_train, S_val = JudiLing.make_combined_S_matrix(train, val, ["Lexeme"],
    ["Person", "Number", "Tense", "Voice", "Mood"])

    res = JudiLing.get_and_train_model(cue_obj_train.C, S_train,
                                "test.bson",
                                batchsize=3)

    model_comp = res.model

    res = JudiLing.get_and_train_model(S_train, cue_obj_train.C,
                                "test.bson",
                                batchsize=3)

    model_prod = res.model

    Chat_train = JudiLing.predict_from_deep_model(model_prod, S_train)

    res_learn = JudiLing.learn_paths(train, cue_obj_train, S_train, model_comp, Chat_train)
end
