function wh_learn(
  X::Union{Matrix, SparseMatrixCSC},
  Y::Union{Matrix, SparseMatrixCSC};
  eta=0.01::Float64,
  n_epochs=1::Int64,
  weights=nothing::Union{Matrix{Float64},Nothing},
  verbose=false::Bool
  )::Matrix{Float64}

  X = Array(X)
  Y = Array(Y)

  if size(X, 1) != size(Y, 1)
    throw(ArgumentError("X($(size(inputs,1))) and Y($(size(outputs,1))) length doesn't match"))
  end

  if isnothing(weights)
    W = zeros(Float64, (size(X,2), size(Y,2)))
  else
    W = weights
  end
  
  inputT = Matrix{Float64}(undef, (size(X,2), 1))
  pred = Matrix{Float64}(undef, (1, size(Y,2)))
  deltaW = Matrix{Float64}(undef, (size(X,2), size(Y,2)))
  verbose && begin pb = Progress(size(X,1)*n_epochs) end
  for j in 1:n_epochs # 100 epochs
    for i in 1:size(X,1) # for each events
      # pred = X[i:i, :]*W
      mul!(pred, X[i:i, :], W)
      # obsv = Y[i:i, :]-pred
      broadcast!(-, pred, Y[i:i, :], pred)
      # inputT = X[i:i, :]'
      transpose!(inputT,X[i:i, :])
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