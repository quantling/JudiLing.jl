using .Flux
using BSON: @save, @load

"""
    get_and_train_model(X_train::Union{SparseMatrixCSC,Matrix},
                        Y_train::Union{SparseMatrixCSC,Matrix},
                        X_val::Union{SparseMatrixCSC,Matrix,Missing},
                        Y_val::Union{SparseMatrixCSC,Matrix,Missing},
                        data_train::Union{DataFrame,Missing},
                        data_val::Union{DataFrame,Missing},
                        target_col::Union{Symbol,String,Missing},
                        model_outpath::String;
                        hidden_dim::Int=1000,
                        n_epochs::Int=100,
                        batchsize::Int=64,
                        loss_func::Function=Flux.mse,
                        optimizer=Flux.Adam(0.001)
                        model::Union{Missing, Chain}=missing,
                        early_stopping::Union{Missing, Int}=missing,
                        optimise_for_acc::Bool=false
                        return_losses::Bool=false,
                        verbose::Bool=true,
                        )

Trains a deep learning model from X_train to Y_train, saving the model with either the highest
validation accuracy or lowest validation loss (depending on `optimise_for_acc`) to `outpath`.

The default model looks like this:
```
inp_dim = size(X_train, 2)
out_dim = size(Y_train, 2)
Chain(Dense(inp_dim => hidden_dim, relu), Dense(hidden_dim => out_dim))
```
Any other model with the same input and output dimensions can be provided to the function with the `model` argument.
The default loss function is mean squared error, but any other loss function can be provded, as long as it fits with the model architecture.

By default the adam optimizer (Kingma and Ba, 2015) with learning rate 0.001 is used. You can provide any other optimizer. If you want to use a different learning rate, e.g. 0.01, provide `optimizer=Flux.Adam(0.01)`. If you do not want to use an optimizer at all, and simply use normal gradient descent, provide `optimizer=Descent(0.001)`, again replacing the learning rate with the learning rate of your preference.

# Obligatory arguments
- `X_train::Union{SparseMatrixCSC,Matrix}`: training input matrix of dimension m x n
- `Y_train::Union{SparseMatrixCSC,Matrix}`: training output/target matrix of dimension m x k
- `X_train::Union{SparseMatrixCSC,Matrix}`: validation input matrix of dimension l x n
- `Y_train::Union{SparseMatrixCSC,Matrix}`: validation output/target matrix of dimension l x k
- `data_train::DataFrame`: training data
- `data_val::DataFrame`: validation data
- `target_col::Union{Symbol, String}`: column with target wordforms in data_train and data_val
- `model_outpath::String`: filepath to where final model should be stored (in .bson format)

# Optional arguments
- `hidden_dim::Int=1000`: hidden dimension of the model
- `n_epochs::Int=100`: number of epochs for which the model should be trained
- `batchsize::Int=64`: batchsize during training
- `loss_func::Function=Flux.mse`: Loss function. Per default this is the mse loss, but other options might be a crossentropy loss (`Flux.crossentropy`). Make sure the model makes sense with the loss function!
- `optimizer=Flux.Adam(0.001)`: optimizer to use for training
- `model::Union{Missing, Chain} = missing`: A custom model can be provided for training. Its requirements are that it has to correspond to the input and output size of the training and validation data
- `early_stopping::Union{Missing, Int}=missing`: If `missing`, no early stopping is used. Otherwise `early_stopping` indicates how many epochs have to pass without improvement in validation accuracy before the training is stopped.
- `optimise_for_acc::Bool=false`: if true, keep model with highest validation *accuracy*. If false, keep model with lowest validation *loss*.
- `return_losses::Bool=false`: whether additional to the model per-epoch losses for the training and test data as well as per-epoch accuracy on the validation data should be returned
- `verbose::Bool=true`: Turn on verbose mode
"""
function get_and_train_model(X_train::Union{SparseMatrixCSC,Matrix},
                            Y_train::Union{SparseMatrixCSC,Matrix},
                            X_val::Union{SparseMatrixCSC,Matrix,Missing},
                            Y_val::Union{SparseMatrixCSC,Matrix,Missing},
                            data_train::Union{DataFrame,Missing},
                            data_val::Union{DataFrame,Missing},
                            target_col::Union{Symbol, String,Missing},
                            model_outpath::String;
                            hidden_dim::Int=1000,
                            n_epochs::Int=100,
                            batchsize::Int=64,
                            loss_func::Function=Flux.mse,
                            optimizer=Flux.Adam(0.001),
                            model::Union{Missing, Chain} = missing,
                            early_stopping::Union{Missing, Int}=missing,
                            optimise_for_acc::Bool=false,
                            return_losses::Bool=false,
                            verbose::Bool=true)

    # set up early stopping and saving of best models
    min_loss = typemax(Float64)
    max_acc = 0

    function id_func(x)
        return (x)
    end

    if !ismissing(early_stopping)
        if optimise_for_acc
            init_score = max_acc
        else
            init_score = min_loss
        end
        es = Flux.early_stopping(id_func, early_stopping, init_score=init_score)
    end

    # Set up the model if not provided
    verbose && println("Setting up model...")
    flush(stdout)
    if ismissing(model)
        model = Chain(
            Dense(size(X_train, 2) => hidden_dim, relu),   # activation function inside layer
            Dense(hidden_dim => size(Y_train, 2))) |> gpu        # move model to GPU, if available
    end

    verbose && @show model
    flush(stdout)

    # create data loader for training data
    verbose && println("Setting up data structures...")
    flush(stdout)
    loader_train = Flux.DataLoader((X_train', Y_train') , batchsize=batchsize, shuffle=true);

    if !ismissing(X_val) & !ismissing(Y_val)
        loader_val = Flux.DataLoader((X_val', Y_val'), batchsize=batchsize, shuffle=false);
    end

    # Set up optimizer
    verbose && println("Setting up optimizer...")
    flush(stdout)
    optim = Flux.setup(optimizer, model)  # will store optimiser momentum, etc.

    verbose && println("Training...")
    flush(stdout)

    # Some lists for storing losses and accuracies
    losses_train = []
    losses_val = []
    accs_val = []

    # Setting up progress bar
    p = Progress(n_epochs)

    # training for n_epochs epochs
    for epoch in 1:n_epochs
        all_losses_epoch_train = []
        for (x_cpu, y_cpu) in loader_train
            x = x_cpu |> gpu
            y = y_cpu |> gpu
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                loss_func(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
            push!(all_losses_epoch_train, loss)  # logging, outside gradient context
        end
        # store mean loss of epoch
        mean_train_loss = mean(all_losses_epoch_train)
        push!(losses_train, mean_train_loss)

        # Compute validation loss
        if !ismissing(X_val) & !ismissing(Y_val)
            all_losses_epoch_val = []
            #create array object to append predictions to (we'll skip the first line of ones, but it needs the right shape)
            #preds_val = ones(1,size(Y_val,2))
            preds_val = Matrix{Float64}(undef, 0, size(Y_val,2))

            for (x_cpu, y_cpu) in loader_val
                x = x_cpu |> gpu
                y = y_cpu |> gpu

                Yhat_val = model(x)|> cpu
                preds_val = vcat(preds_val,Yhat_val')
                # for some reason, I have to make sure y_cpu is not sparse here.
                # I have no idea why
                loss_val = loss_func(Yhat_val, Matrix(y_cpu))
                push!(all_losses_epoch_val,loss_val)
            end
            mean_val_loss = mean(all_losses_epoch_val)
            push!(losses_val, mean_val_loss)

            # Compute validation accuracy
            acc = JudiLing.eval_SC(preds_val, Y_val, Y_train, data_val, data_train, target_col)
            push!(accs_val, acc)


            # update progress bar with training and validation losses and accuracy
            ProgressMeter.next!(p; showvalues = [("Training loss",mean_train_loss),
                                                 ("Validation loss",mean_val_loss),
                                                 ("Validation accuracy", acc)])

             # Save if model with highest accuracy
             if (optimise_for_acc && (acc > max_acc)) || (!optimise_for_acc && (mean_val_loss < min_loss))
                model_cpu = model |> cpu
                 @save model_outpath model_cpu
                 max_acc = acc
                 min_loss = mean_val_loss
             end

             # early stopping
             if optimise_for_acc
                 !ismissing(early_stopping) && es(-acc) && break
             else
                 !ismissing(early_stopping) && es(mean_val_loss) && break
             end
        else
            model_cpu = model |> cpu
            @save model_outpath model_cpu

            # update progress bar with training and validation losses and accuracy
            ProgressMeter.next!(p; showvalues = [("Training loss",mean_train_loss)])

        end
    end

    @load model_outpath model_cpu

    return_losses && return(model_cpu, losses_train, losses_val, accs_val)
    return(model_cpu)

end

"""
    get_and_train_model(X_train::Union{SparseMatrixCSC,Matrix},
                        Y_train::Union{SparseMatrixCSC,Matrix},
                        model_outpath::String;
                        hidden_dim::Int=1000,
                        n_epochs::Int=100,
                        batchsize::Int=64,
                        loss_func::Function=Flux.mse,
                        optimizer=Flux.Adam(0.001),
                        model::Union{Missing, Chain} = missing,
                        return_losses::Bool=false,
                        verbose::Bool=true)

Trains a deep learning model from X_train to Y_train, saving the model after n_epochs epochs.
The default model looks like this:

```
inp_dim = size(X_train, 2)
out_dim = size(Y_train, 2)
Chain(Dense(inp_dim => hidden_dim, relu), Dense(hidden_dim => out_dim))
```

Any other model with the same input and output dimensions can be provided to the function with the `model` argument.
The default loss function is mean squared error, but any other loss function can be provded, as long as it fits with the model architecture.

By default the adam optimizer (Kingma and Ba, 2015) with learning rate 0.001 is used. You can provide any other optimizer. If you want to use a different learning rate, e.g. 0.01, provide `optimizer=Flux.Adam(0.01)`. If you do not want to use an optimizer at all, and simply use normal gradient descent, provide `optimizer=Descent(0.001)`, again replacing the learning rate with the learning rate of your preference.

# Obligatory arguments
- `X_train::Union{SparseMatrixCSC,Matrix}`: training input matrix of dimension m x n
- `Y_train::Union{SparseMatrixCSC,Matrix}`: training output/target matrix of dimension m x k
- `model_outpath::String`: filepath to where final model should be stored (in .bson format)

# Optional arguments
- `hidden_dim::Int=1000`: hidden dimension of the model
- `n_epochs::Int=100`: number of epochs for which the model should be trained
- `batchsize::Int=64`: batchsize during training
- `loss_func::Function=Flux.mse`: Loss function. Per default this is the mse loss, but other options might be a crossentropy loss (`Flux.crossentropy`). Make sure the model makes sense with the loss function!
- `optimizer=Flux.Adam(0.001)`: optimizer to use for training
- `model::Union{Missing, Chain} = missing`: A custom model can be provided for training. Its requirements are that it has to correspond to the input and output size of the training and validation data
- `return_losses::Bool=false`: whether additional to the model per-epoch losses for the training and test data as well as per-epoch accuracy on the validation data should be returned
- `verbose::Bool=true`: Turn on verbose mode
"""
function get_and_train_model(X_train::Union{SparseMatrixCSC,Matrix},
                            Y_train::Union{SparseMatrixCSC,Matrix},
                            model_outpath::String;
                            hidden_dim::Int=1000,
                            n_epochs::Int=100,
                            batchsize::Int=64,
                            loss_func::Function=Flux.mse,
                            optimizer=Flux.Adam(0.001),
                            model::Union{Missing, Chain} = missing,
                            return_losses::Bool=false,
                            verbose::Bool=true)

    get_and_train_model(X_train,
                        Y_train,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        model_outpath;
                        hidden_dim=hidden_dim,
                        n_epochs=n_epochs,
                        return_losses=return_losses,
                        verbose=verbose,
                        model=model,
                        early_stopping=missing,
                        loss_func=loss_func,
                        batchsize=batchsize,
                        optimise_for_acc=false,
                        optimizer=optimizer)
end


function fiddl(X_train::Union{SparseMatrixCSC,Matrix},
                Y_train::Union{SparseMatrixCSC,Matrix},
                learn_seq::Vector,
                data::DataFrame,
                target_col::Union{Symbol, String},
                model_outpath::String;
                hidden_dim::Int=1000,
                batchsize::Int=64,
                loss_func::Function=Flux.mse,
                optimizer=Flux.Adam(0.001),
                model::Union{Missing, Chain} = missing,
                return_losses::Bool=false,
                verbose::Bool=true,
                n_batch_eval::Int=100,
                measures_func::Union{Function, Missing}=missing)

    data = deepcopy(data)

    # Set up the model if not provided
    verbose && println("Setting up model...")
    flush(stdout)
    if ismissing(model)
        model = Chain(
            Dense(size(X_train, 2) => hidden_dim, relu),   # activation function inside layer
            Dense(hidden_dim => size(Y_train, 2))) |> gpu        # move model to GPU, if available
    else
        model = model |> gpu
    end

    verbose && @show model
    flush(stdout)

    # Set up optimizer
    verbose && println("Setting up optimizer...")
    flush(stdout)
    optim = Flux.setup(optimizer, model)  # will store optimiser momentum, etc.

    verbose && println("Setting up data for evaluation...")
    flush(stdout)
    loader_data = Flux.DataLoader((X_train', Y_train') , batchsize=batchsize, shuffle=false); # has to be unshuffled, otherwise the accuracy calculation will be wrong

    verbose && println("Training...")
    flush(stdout)

    # Some lists for storing losses and accuracies
    losses_train = []
    losses = []
    accs = []

    # Setting up progress bar
    p = Progress(Int(ceil(length(learn_seq)/(batchsize * n_batch_eval))))

    step = 0
    for subseq in Iterators.partition(learn_seq, batchsize * n_batch_eval)
        if batchsize > length(subseq)
            batchsize = length(subseq)
        end

        step += batchsize * n_batch_eval

        # set up loader for current step
        loader_train = Flux.DataLoader((X_train[subseq,:]', Y_train[subseq,:]') , batchsize=batchsize, shuffle=false);

        # train current step
        all_losses_epoch_train = []
        for (x_cpu, y_cpu) in loader_train
            x = x_cpu |> gpu
            y = y_cpu |> gpu
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x)
                loss_func(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
            push!(all_losses_epoch_train, loss)  # logging, outside gradient context
        end
        # store mean loss of epoch
        mean_train_loss = mean(all_losses_epoch_train)
        push!(losses_train, mean_train_loss)

        all_losses_epoch = []
        preds = Matrix{Float64}(undef, 0, size(Y_train,2))

        for (x_cpu, y_cpu) in loader_data
            x = x_cpu |> gpu
            y = y_cpu |> gpu

            Yhat = model(x)|> cpu
            preds = vcat(preds,Yhat')
            # for some reason, I have to make sure y_cpu is not sparse here.
            # I have no idea why
            loss = loss_func(Yhat, Matrix(y_cpu))
            push!(all_losses_epoch,loss)
        end
        mean_loss = mean(all_losses_epoch)
        push!(losses, mean_loss)

        # Compute validation accuracy
        acc = JudiLing.eval_SC(preds, Y_train, data, target_col)
        push!(accs, acc)

        model_cpu = model |> cpu
        @save model_outpath model_cpu

        # this will need to be implemented properly with all possible arguments
        # a measures function may need
        if !ismissing(measures_func)
            data = measures_func(X_train, Y_train, preds, data, target_col, step)
        end

        # update progress bar with training and validation losses and accuracy
        ProgressMeter.next!(p; showvalues = [("Step loss", mean_train_loss),
                                             ("Overall loss", mean_loss),
                                             ("Overall accuracy", acc)])
    end

    if !ismissing(measures_func)
        data = measures_func(X_train, Y_train, preds, data, target_col, "final")
    end

    res = Vector{Any}([model |> cpu])
    if !ismissing(measures_func)
        append!(res, [data])
    end
    if return_losses
        append!(res, [losses_train, losses, accs])
    end
    return(res)

end

"""
    predict_from_deep_model(model::Chain,
                            X::Union{SparseMatrixCSC,Matrix})

Generates output of a model given input `X`.

# Obligatory arguments
- `model::Chain`: Model of type Flux.Chain, as generated by `get_and_train_model`
- `X::Union{SparseMatrixCSC,Matrix}`: Input matrix of size (number_of_samples, inp_dim) where inp_dim is the input dimension of `model`

"""
function predict_from_deep_model(model::Chain,
                                 X::Union{SparseMatrixCSC,Matrix})
    model_cpu = model |> cpu
    Yhat = model_cpu(X' |> cpu)
    return(convert(Matrix, Yhat'))
end


"""
    predict_shat(model::Chain,
                 ci::Vector{Int})
Predicts semantic vector shat given a deep learning comprehension model `model` and a
list of indices of ngrams `ci`.

# Obligatory arguments
- `model::Chain`: Deep learning comprehension model as generated by `get_and_train_model`
- `ci::Vector{Int}`: Vector of indices of ngrams in c vector. Essentially, this is a vector indicating which ngrams in a c vector are absent and which are present.

"""
function predict_shat(model::Chain,
                     ci::Vector{Int})
    dim = size(Flux.params(model[1])[1])[2]
    c = zeros(1, dim)
    c[:, ci] .= 1
    shat = predict_from_deep_model(model, c)
    return(shat)
end
