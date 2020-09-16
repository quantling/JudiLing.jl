"""
Make full adjacency matrix.
"""
function make_adjacency_matrix end

"""
    make_adjacency_matrix(::Dict) -> ::SparseMatrixCSC

Make full adjacency matrix based only on the form of n-grams regardless whether 
they are seen in the training data. This usually takes hours for large datasets.

...
# Obligatory Arguments
- `i2f::Dict`: the dictionary return features given indices

# Optional Arguments
- `tokenized::Bool=false`:if true, the dataset target is assumed tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separate token
- `verbose::Bool=false`: if true, more information prints out

# Examples
```julia
# without tokenization
i2f = Dict([(1, "#ab"), (2, "abc"), (3, "bc#"), (4, "#bc"), (5, "ab#")])
JudiLing.make_adjacency_matrix(i2f)

# with tokenization
i2f = Dict([(1, "#-a-b"), (2, "a-b-c"), (3, "b-c-#"), (4, "#-b-c"), (5, "a-b-#")])
JudiLing.make_adjacency_matrix(
  i2f,
  tokenized=true,
  sep_token="-")
```
...
"""
function make_adjacency_matrix(
  i2f::Dict;
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  verbose=false::Bool
  )::SparseMatrixCSC

  ngrams = [i2f[i] for i in 1:length(i2f)]

  if tokenized && !isnothing(sep_token)
    words = split.(ngrams, sep_token)
  else
    words = split.(ngrams, "")
  end

  n_ngrams = length(ngrams)

  I = Int64[]
  J = Int64[]
  V = Int64[]

  iter = 1:n_ngrams
  if verbose
    pb = Progress(n_ngrams)
  end

  for i in iter
    for j in 1:n_ngrams
      if isattachable(words[i], words[j])
        push!(I, i)
        push!(J, j)
        push!(V, 1)
      end
    end
    if verbose
      ProgressMeter.next!(pb)
    end
  end

  sparse(I, J, V, n_ngrams, n_ngrams, *)
end