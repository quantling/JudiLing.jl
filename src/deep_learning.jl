"""
    get_and_train_model(X_train,
                        Y_train,
                        X_val,
                        Y_val,
                        data_train,
                        data_val,
                        target_col,
                        model_outpath;
                        hidden_dim = 1000,
                        n_epochs = 100,
                        return_losses = false,
                        verbose = true,
                        model = missing,
                        early_stopping = missing,
                        loss_func = Flux.mse,
                        batchsize = 64,
                        optimise_for_acc = false)

Trains a deep learning model from X_train to Y_train, saving the model with either the highest
validation accuracy or lowest validation loss (depending on `optimise_for_acc`) to `outpath`.
The default model looks like this:
```
inp_dim = size(X_train, 2)
out_dim = size(Y_train, 2)
Chain(Dense(inp_dim => hidden_dim, tanh), BatchNorm(hidden_dim), Dense(hidden_dim => out_dim))
```
Any other model with the same input and output dimensions can be provided to the function with the `model` argument.
The default loss function is mean squared error, but any other loss function can be provded, as long as it fits with the model architecture.

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
- `return_losses::Bool=false`: whether additional to the model per-epoch losses for the training and test data as well as per-epoch accuracy on the validation data should be returned
- `verbose::Bool=true`: Turn on verbose mode
- `model::Union{Missing, Chain} = missing`: A custom model can be provided for training. Its requirements are that it has to correspond to the input and output size of the training and validation data
- `early_stopping::Union{Missing, Int}=missing`: If `missing`, no early stopping is used. Otherwise `early_stopping` indicates how many epochs have to pass without improvement in validation accuracy before the training is stopped.
- `loss_func::Function=Flux.mse`: Loss function. Per default this is the mse loss, but other options might be a crossentropy loss (`Flux.crossentropy`). Make sure the model makes sense with the loss function!
- `batchsize::Int=64`: batchsize during training
- `optimise_for_acc::Bool=false`: if true, keep model with highest validation *accuracy*. If false, keep model with lowest validation *loss*.
"""
function get_and_train_model(X_train::Union{SparseMatrixCSC,Matrix},
                            Y_train::Union{SparseMatrixCSC,Matrix},
                            X_val::Union{SparseMatrixCSC,Matrix},
                            Y_val::Union{SparseMatrixCSC,Matrix},
                            data_train::DataFrame,
                            data_val::DataFrame,
                            target_col::Union{Symbol, String},
                            model_outpath::String;
                            hidden_dim::Int=1000,
                            n_epochs::Int=100,
                            return_losses::Bool=false,
                            verbose::Bool=true,
                            model::Union{Missing, Chain} = missing,
                            early_stopping::Union{Missing, Int}=missing,
                            loss_func::Function=Flux.mse,
                            batchsize::Int=64,
                            optimise_for_acc::Bool=false)

    # set up early stopping and saving of best models
    min_loss = 100000000
    max_acc = 0

    function id_func(x)
        return (x)
    end

    if !ismissing(early_stopping)
        es = Flux.early_stopping(id_func, early_stopping, init_score=min_loss)
    end

    # Set up the model if not provided
    verbose && println("Setting up model...")
    flush(stdout)
    if ismissing(model)
        model = Chain(
            Dense(size(X_train, 2) => hidden_dim, tanh),   # activation function inside layer
            BatchNorm(hidden_dim),
            Dense(hidden_dim => size(Y_train, 2))) |> gpu        # move model to GPU, if available
    end

    verbose && @show model
    flush(stdout)

    # create data loader for training data
    verbose && println("Setting up data structures...")
    flush(stdout)
    loader_train = Flux.DataLoader((X_train', Y_train') |> gpu, batchsize=batchsize, shuffle=true);

    # Set up optimizer
    verbose && println("Setting up optimizer...")
    flush(stdout)
    optim = Flux.setup(Flux.Adam(0.001), model)  # will store optimiser momentum, etc.

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
        for (x, y) in loader_train
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
        Yhat_val = model(X_val')
        mean_val_loss = loss_func(Yhat_val, Y_val')
        push!(losses_val, mean_val_loss)

        # Compute validation accuracy
        acc = JudiLing.eval_SC(Yhat_val', Y_val, Y_train, data_val, data_train, target_col)
        push!(accs_val, acc)

        # update progress bar with training and validation losses and accuracy
        ProgressMeter.next!(p; showvalues = [("Training loss",mean_train_loss),
                                             ("Validation loss",mean_val_loss),
                                             ("Validation accuracy", acc)])

        # Save if model with highest accuracy
        if (optimise_for_acc && (acc > max_acc)) || (!optimise_for_acc && (mean_val_loss < min_loss))
            @save model_outpath model
            max_acc = acc
            min_loss = mean_val_loss
        end

        # early stopping
        !ismissing(early_stopping) && es(mean_val_loss) && break
    end

    @load model_outpath model

    return_losses && return(model, losses_train, losses_val, accs_val)
    return(model)
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
    Yhat = model(X' |> gpu) |> cpu
    return(convert(Matrix, Yhat'))
end
