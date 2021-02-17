function wh_learn(
    X::Union{Matrix,SparseMatrixCSC},
    Y::Union{Matrix,SparseMatrixCSC};
    eta = 0.01::Float64,
    n_epochs = 1::Int64,
    weights = nothing::Union{Matrix{Float64},Nothing},
    learn_seq = nothing::Union{Nothing,Vector{Int64}},
    verbose = false::Bool,
)::Matrix{Float64}

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

    inputT = Matrix{Float64}(undef, (size(X, 2), 1))
    pred = Matrix{Float64}(undef, (1, size(Y, 2)))
    deltaW = Matrix{Float64}(undef, (size(X, 2), size(Y, 2)))
    verbose && begin
        pb = Progress(size(X, 1) * n_epochs)
    end
    for j = 1:n_epochs # 100 epochs
        for i in learn_seq # for each events
            # pred = X[i:i, :]*W
            mul!(pred, X[i:i, :], W)
            # obsv = Y[i:i, :]-pred
            broadcast!(-, pred, Y[i:i, :], pred)
            # inputT = X[i:i, :]'
            transpose!(inputT, X[i:i, :])
            # update = inputT*obsv
            mul!(deltaW, inputT, pred)
            # deltaW = eta*update
            rmul!(deltaW, eta)
            # W += deltaW
            broadcast!(+, W, W, deltaW)
            verbose && ProgressMeter.next!(pb)
        end
    end
    W
end

function make_learn_seq(
    freq::Vector{Int64};
    random_seed = 314::Int64,
)::Vector{Int64}

    learn_seq = [repeat([i], n) for (i, n) in enumerate(freq)]
    learn_seq = collect(Iterators.flatten(learn_seq))
    rng = MersenneTwister(random_seed)
    shuffle(rng, learn_seq)
end
