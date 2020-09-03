function wh_learn(
  inputs::Union{Matrix, SparseMatrixCSC},
  outputs::Union{Matrix, SparseMatrixCSC};
  eta=0.1::Float64,
  n_epochs=nothing::Union{Int64, Nothing},
  weights=nothing::Union{Matrix{Float64},Nothing},
  verbose=false::Bool
  )
  
  inputs = convert(Matrix{Float64}, inputs)
  outputs = convert(Matrix{Float64}, outputs)

  n_row = size(inputs, 2)
  n_col = size(outputs, 2)
  n_events = size(outputs, 1)

  # check length of C and S
  if n_events != size(inputs, 1)
    throw(ArgumentError("inputs($(size(inputs,1))) and outputs($(size(outputs,1))) length doesn't match"))
  end

  # if no weights pass, using os
  if isnothing(weights)
    weights = zeros(Float64, n_row, n_col)
  else
    weights = weights
  end

  iter = 1:n_events
  # iter = 1:10
  if !isnothing(n_epochs)
    iter = collect(repeat(iter, outer=n_epochs))
    rng = MersenneTwister(1234)
    iter = shuffle(rng, iter)
  end
  verbose && begin iter=tqdm(iter) end
  for i in iter
    weights = learn_inplace(inputs[i:i,:], outputs[i:i,:], weights, eta)
  end
  weights
end

function learn_inplace(
  input::Matrix,
  output::Matrix,
  W::Matrix{Float64},
  eta::Float64
  )
  deltaW = eta*(input'*(output-input*W))
  W += deltaW
end