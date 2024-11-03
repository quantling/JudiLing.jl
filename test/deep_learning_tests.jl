using Test
using Flux
using JudiLing
using CSV, DataFrames
using LinearAlgebra: diag
using DataLoaders

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

@testset "training acc" begin

    @test_throws ArgumentError JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                "test.bson",
                                batchsize=3,
                                return_train_acc = true)

    res = JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                "test.bson",
                                batchsize=3,
                                return_train_acc = true,
                                data_train=train,
                                target_col = :Word)


    accs_train = res.accs_train

    @test !ismissing(accs_train)
    @test length(accs_train) == 100
    @test accs_train[end] ≈ 1.0

    res = JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                cue_obj_val.C,
                                S_val,
                                train, val,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                batchsize=3,
                                return_train_acc = true)

    accs_train = res.accs_train
    accs_val = res.accs_val

    @test !ismissing(accs_train)
    @test length(accs_train) == length(accs_val) == 100
    @test accs_train[end] ≈ 1.0
    @test accs_train[5] > accs_val[5]
end

@testset "measures_func" begin

    @testset "training data only" begin

        function compute_target_corr(X_train, Y_train, X_val, Y_val,
                                        Yhat_train, Yhat_val, data_train,
                                        data_val, target_col, model, epoch)
            _, corr = JudiLing.eval_SC(Yhat_train, Y_train, R=true)
            data_train[!, string("target_corr_", epoch)] = diag(corr)
            return(data_train, missing)
        end

        res = JudiLing.get_and_train_model(cue_obj_train.C,
                                    S_train,
                                    "test.bson",
                                    batchsize=3,
                                    return_train_acc = true,
                                    data_train = train,
                                    target_col = :Word,
                                    measures_func=compute_target_corr)

        data_train = res.data_train

        @test size(data_train, 1) == size(train,1)
        @test size(data_train, 2) - size(train,2) == 101
        expected_cols = [string("target_corr_", epoch) for epoch in 1:100]
        @test all(expected_cols .∈ [names(data_train)])
        @test "target_corr_final" ∈ names(data_train)

        @test data_train[1, "target_corr_1"] < data_train[1, "target_corr_50"] < data_train[1, "target_corr_100"]
        @test data_train[1, "target_corr_100"] == data_train[1, "target_corr_final"]

    end

    @testset "Passing kargs to measures_func" begin

        function compute_target_corr(X_train, Y_train, X_val, Y_val,
                                        Yhat_train, Yhat_val, data_train,
                                        data_val, target_col, model, epoch; dummy_string="x")
            _, corr = JudiLing.eval_SC(Yhat_train, Y_train, R=true)
            data_train[!, string("target_corr_", epoch, "_", dummy_string)] = diag(corr)
            return(data_train, missing)
        end

        res = JudiLing.get_and_train_model(cue_obj_train.C,
                                    S_train,
                                    "test.bson",
                                    batchsize=3,
                                    return_train_acc = true,
                                    data_train = train,
                                    target_col = :Word,
                                    measures_func=compute_target_corr,
                                    dummy_string="y")

        data_train = res.data_train

        expected_cols = [string("target_corr_", epoch, "_y") for epoch in 1:100]
        @test all(expected_cols .∈ [names(data_train)])
        @test "target_corr_final_y" ∈ names(data_train)

    end

    @testset "with training and validation data" begin
        function compute_target_corr(X_train, Y_train, X_val, Y_val,
                                        Yhat_train, Yhat_val, data_train,
                                        data_val, target_col, model, epoch)
            _, corr = JudiLing.eval_SC(Yhat_train, Y_train, R=true)
            data_train[!, string("target_corr_", epoch)] = diag(corr)
            _, corr = JudiLing.eval_SC(Yhat_val, Y_val, R=true)
            data_val[!, string("target_corr_", epoch)] = diag(corr)
            return(data_train, data_val)
        end

        res = JudiLing.get_and_train_model(cue_obj_train.C,
                                    S_train,
                                    cue_obj_val.C,
                                    S_val,
                                    train, val,
                                    :Word,
                                    "test.bson",
                                    return_losses=true,
                                    batchsize=3,
                                    measures_func=compute_target_corr)

        data_train = res.data_train
        data_val = res.data_val

        @test size(data_train, 1) == size(train,1)
        @test size(data_val, 1) == size(val,1)
        @test size(data_train, 2) - size(train,2) == 101
        @test size(data_val, 2) - size(val,2) == 101
        expected_cols = [string("target_corr_", epoch) for epoch in 1:100]
        @test all(expected_cols .∈ [names(data_train)])
        @test all(expected_cols .∈ [names(data_val)])

        @test "target_corr_final" ∈ names(data_train)
        @test "target_corr_final" ∈ names(data_val)

        @test data_train[1, "target_corr_1"] < data_train[1, "target_corr_50"] < data_train[1, "target_corr_100"]
        @test data_train[1, string("target_corr_", findmin(res.losses_val)[2])]  == data_train[1, "target_corr_final"]

        @test data_val[1, "target_corr_1"] < data_val[1, "target_corr_50"] < data_val[1, "target_corr_100"]
        @test data_val[1, string("target_corr_", findmin(res.losses_val)[2])]  == data_val[1, "target_corr_final"]

    end
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
    @test findmin(losses_val)[2] + 20 == length(losses_val)

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
    @test findmax(accs_val)[2] + 20 == length(accs_val)

    res = JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                cue_obj_val.C,
                                S_val,
                                train_es, val_es,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                early_stopping=10,
                                optimise_for_acc = true,
                                batchsize=2)

    model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val
    @test findmax(accs_val)[2] + 10 == length(accs_val)

    res = JudiLing.get_and_train_model(cue_obj_train.C,
                                S_train,
                                cue_obj_val.C,
                                S_val,
                                train_es, val_es,
                                :Word,
                                "test.bson",
                                return_losses=true,
                                early_stopping=10,
                                n_epochs=1000,
                                batchsize=2)

    model, losses_train, losses_val, accs_val = res.model, res.losses_train, res.losses_val, res.accs_val
    @test findmin(losses_val)[2] + 10 == length(losses_val)

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

@testset "fiddl" begin

    @testset "dataloader" begin
        X1 = [[1 2 3]
              [4 5 6]
              [7 8 9]]
        Y1 = [[1 0]
              [2 1]
              [3 2]]

        learn_seq = [3,2,1,2,3]
        data = JudiLing.FIDDLDataset(X1', Y1', learn_seq)

        x_expected = [X1[3:3,:]', X1[2:2,:]', X1[1:1,:]', X1[2:2,:]', X1[3:3,:]']
        x_batches = []
        y_expected = [Y1[3:3,:]', Y1[2:2,:]', Y1[1:1,:]', Y1[2:2,:]', Y1[3:3,:]']
        y_batches = []
        for (i, batch) in enumerate(DataLoaders.DataLoader(data, 1))
            push!(x_batches, batch[1])
            push!(y_batches, batch[2])
        end
        @test x_batches == x_expected
        @test y_batches == y_expected

        x_expected = [X1[[3,2],:]', X1[[1,2],:]', X1[3:3,:]']
        x_batches = []
        y_expected = [Y1[[3,2],:]', Y1[[1,2],:]', Y1[3:3,:]']
        y_batches = []
        for (i, batch) in enumerate(DataLoaders.DataLoader(data, 2))
            push!(x_batches, batch[1])
            push!(y_batches, batch[2])
        end
        @test x_batches == x_expected
        @test y_batches == y_expected
    end

    @testset "basic setup" begin
        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=1,
                        n_batch_eval=1)

        @test length(res.accs) == length(res.losses) == length(res.losses_train) == 3
        @test size(Flux.params(res.model[1])[1],1) == 1000

    end

    @testset "learn_seq" begin

        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,1,2,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=1,
                        n_batch_eval=1)

        @test length(res.accs) == length(res.losses) == length(res.losses_train) == 6

        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,3,3,3,3,3,3,3,3,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=1,
                        n_batch_eval=1)

        @test length(res.accs) == length(res.losses) == length(res.losses_train) == 12
        Shat = JudiLing.predict_from_deep_model(res.model, cue_obj_train.C)
        _, corr = JudiLing.eval_SC(Shat, S_train, R=true)
        target_corr = diag(corr)
        @test target_corr[1] < target_corr[3]
        @test target_corr[2] < target_corr[3]

        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,1,1,1,1,1,1],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=1,
                        n_batch_eval=1)

        @test length(res.accs) == length(res.losses) == length(res.losses_train) == 9
        Shat = JudiLing.predict_from_deep_model(res.model, cue_obj_train.C)
        _, corr = JudiLing.eval_SC(Shat, S_train, R=true)
        target_corr = diag(corr)
        @test target_corr[1] > target_corr[3]
        @test target_corr[1] > target_corr[2]
    end

    @testset "n_batch_eval" begin
        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,1,2,3,1,2,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=1,
                        n_batch_eval=2)

        @test length(res.accs) == length(res.losses) == length(res.losses_train) == 5
    end

    @testset "batchsize" begin
        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,1,2,3,1,2,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=3,
                        n_batch_eval=1)

        @test length(res.accs) == length(res.losses) == length(res.losses_train) == 3
    end

    @testset "measures_func" begin

        # standard setup
        function compute_target_corr(X_train, Y_train,
                                        Yhat_train, data, target_col, model, step)
            _, corr = JudiLing.eval_SC(Yhat_train, Y_train, R=true)
            data[!, string("target_corr_", step)] = diag(corr)
            return(data)
        end

        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,1,2,3,1,2,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=1,
                        n_batch_eval=1,
                        measures_func=compute_target_corr)

        @test size(res.data, 1) == size(train,1)
        @test size(res.data, 2) - size(train,2) == 10

        expected_cols = [string("target_corr_", step) for step in 1:9]
        @test all(expected_cols .∈ [names(res.data)])
        @test "target_corr_final" ∈ names(res.data)

        @test res.data[1, "target_corr_1"] < res.data[1, "target_corr_3"] < res.data[1, "target_corr_final"]

        # see if correlation acc improves as expected
        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,3,3,3,3,3,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=1,
                        n_batch_eval=1,
                        measures_func=compute_target_corr)

        @test size(res.data, 1) == size(train,1)
        @test size(res.data, 2) - size(train,2) == 10

        expected_cols = [string("target_corr_", step) for step in 1:9]
        @test all(expected_cols .∈ [names(res.data)])
        @test "target_corr_final" ∈ names(res.data)

        @test res.data[3, "target_corr_1"] < res.data[3, "target_corr_3"] < res.data[3, "target_corr_final"]

        # change batch size
        function compute_target_corr(X_train, Y_train,
                                        Yhat_train, data, target_col, model, step)
            _, corr = JudiLing.eval_SC(Yhat_train, Y_train, R=true)
            data[!, string("target_corr_", step)] = diag(corr)
            return(data)
        end

        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,1,2,3,1,2,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=2,
                        n_batch_eval=1,
                        measures_func=compute_target_corr)

        @test size(res.data, 1) == size(train,1)
        @test size(res.data, 2) - size(train,2) == 6

        expected_cols = [string("target_corr_", step) for step in [2,4,6,8,9]]
        @test all(expected_cols .∈ [names(res.data)])
        @test "target_corr_final" ∈ names(res.data)

        # pass karg
        function compute_target_corr(X_train, Y_train,
                                        Yhat_train, data, target_col, model, step;
                                        dummy_string="x")
            _, corr = JudiLing.eval_SC(Yhat_train, Y_train, R=true)
            data[!, string("target_corr_", step, "_", dummy_string)] = diag(corr)
            return(data)
        end

        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,1,2,3,1,2,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=1,
                        n_batch_eval=1,
                        measures_func=compute_target_corr,
                        dummy_string="y")

        @test size(res.data, 1) == size(train,1)
        @test size(res.data, 2) - size(train,2) == 10

        expected_cols = [string("target_corr_", step, "_y") for step in 1:9]
        @test all(expected_cols .∈ [names(res.data)])
        @test "target_corr_final_y" ∈ names(res.data)
    end

    @testset "compute accuracy" begin
        res = JudiLing.fiddl(cue_obj_train.C,
                        S_train,
                        [1,2,3,1,2,3,1,2,3],
                        train,
                        :Word,
                        "test.bson";
                        batchsize=3,
                        n_batch_eval=1,
                        compute_accuracy=false)
        @test length(res.accs) == 0
    end
end
