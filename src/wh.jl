"""
    wh_learn(
        X,
        Y;
        eta = 0.01,
        n_epochs = 1,
        weights = nothing,
        learn_seq = nothing,
        save_history = false,
        history_cols = nothing,
        history_rows = nothing,
        verbose = false,
        )

Widrow-Hoff Learning.

# Obligatory Arguments
- `test_mode::Symbol`: which test mode, currently supports :train_only, :pre_split, :careful_split and :random_split.

# Optional Arguments
- `eta::Float64=0.1`: the learning rate
- `n_epochs::Int64=1`: the number of epochs to be trained
- `weights::Matrix=nothing`: the initial weights
- `learn_seq::Vector=nothing`: the learning sequence
- `save_history::Bool=false`: if true, a partical training history will be saved
- `history_cols::Vector=nothing`: the list of column indices you want to saved in history, e.g. `[1,32,42]` or `[2]`
- `history_rows::Vector=nothing`: the list of row indices you want to saved in history, e.g. `[1,32,42]` or `[2]`
- `verbose::Bool = false`: if true, more information will be printed out
"""
function wh_learn(
    X,
    Y;
    eta = 0.01,
    n_epochs = 1,
    weights = nothing,
    learn_seq = nothing,
    save_history = false,
    history_cols = nothing,
    history_rows = nothing,
    verbose = false,
    )

    X = Array(X)
    Y = Array(Y)

    if size(X, 1) != size(Y, 1)
        throw(ArgumentError("X($(size(inputs,1))) and Y($(size(outputs,1))) length doesn't match"))
    end

    if isnothing(weights)
        W = zeros(Float64, (size(X, 2), size(Y, 2)))
    else
        W = weights
    end

    # construct learn_seq if nothing
    if isnothing(learn_seq)
        learn_seq = 1:size(X, 1)
    end

    if save_history
        history = zeros(Float64, n_epochs, length(history_rows), length(history_cols))
    end

    inputT = Matrix{Float64}(undef, (size(X, 2), 1))
    pred = Matrix{Float64}(undef, (1, size(Y, 2)))
    deltaW = Matrix{Float64}(undef, (size(X, 2), size(Y, 2)))
    if verbose
        if isnothing(learn_seq)
            pb = Progress(size(X, 1) * n_epochs)
        else
            pb = Progress(size(learn_seq, 1) * n_epochs)
        end
    end
    for j = 1:n_epochs # 100 epochs
        for i in learn_seq # for each events
            # pred = X[i:i, :]*W
            mul!(pred, view(X, i:i, :), W)
            # obsv = Y[i:i, :]-pred
            broadcast!(-, pred, view(Y, i:i, :), pred)
            # inputT = X[i:i, :]'
            transpose!(inputT, view(X, i:i, :))
            # update = inputT*obsv
            mul!(deltaW, inputT, pred)
            # deltaW = eta*update
            rmul!(deltaW, eta)
            # W += deltaW
            broadcast!(+, W, W, deltaW)

            verbose && ProgressMeter.next!(pb)
        end
        # push history
        if save_history
            history[j,:,:] = copy(W[history_rows, history_cols])
        end
    end
    if save_history
        return W, history
    end
    W
end

"""
    make_learn_seq(freq; random_seed = 314)

Make Widrow-Hoff learning sequence from frequencies.
Creates a randomly ordered sequences of indices where each index appears according to its frequncy.

!!! note
    Though the generation of the learning sequence is controlled by a random seed, it may change across Julia versions, see here: https://docs.julialang.org/en/v1/stdlib/Random/

# Obligatory arguments
- `freq`: Vector with frequencies.

# Optional arguments
- `random_seed = 314`: Random seed to control randomness.

# Example
```julia
learn_seq = JudiLing.make_learn_seq(data.frequency)
```
"""
function make_learn_seq(freq; random_seed = 314)
    if isnothing(freq)
        return nothing
    end

    learn_seq = [repeat([i], n) for (i, n) in enumerate(freq)]
    learn_seq = collect(Iterators.flatten(learn_seq))
    rng = MersenneTwister(random_seed)
    shuffle(rng, learn_seq)
end
