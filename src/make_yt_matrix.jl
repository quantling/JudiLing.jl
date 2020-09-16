"""
Make Yt matrix given timestep t.
"""
function make_Yt_matrix end

"""
    make_Yt_matrix(::Int64, ::DataFrame) -> ::SparseMatrixCSC

Make Yt matrix given timestep t. Each row of Yt matrix indicates the supports 
of each n-gram predicted in timestep t.

...
# Obligatory Arguments
- `t::Int64`: the timestep t
- `data::DataFrame`: the dataset

# Optional Arguments
- `tokenized::Bool=false`: whether n-grams are tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: what is the sepertate token
- `verbose::Bool=false`: if verbose, more information prints out

# Examples
```julia
latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
JudiLing.make_Yt_matrix(2, latin)
```
...
"""
function make_Yt_matrix(
  t::Int64,
  data::DataFrame;
  grams=3::Int64,
  target_col="Words"::String,
  tokenized=false::Bool,
  keep_sep=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  start_end_token="#"::Union{String, Char}
  )::SparseMatrixCSC

  # split tokens from words or other columns
  if tokenized && !isnothing(sep_token)
    tokens = split.(data[:, target_col], sep_token)
  else
    tokens = split.(data[:, target_col], "")
  end

  # making ngrams from tokens
  # make_ngrams function are below
  ngrams = make_ngrams.(tokens, grams, keep_sep, sep_token, start_end_token)

  # find all unique ngrams features
  ngrams_features = unique(vcat(ngrams...))

  f2i = Dict(v => i for (i, v) in enumerate(ngrams_features))
  i2f = Dict(i => v for (i, v) in enumerate(ngrams_features))

  ngrams_l = length.(ngrams)

  # the number of non-zero values is equal to the number of utterances have
  # larger length than timestep t
  n_f = length(ngrams_l[ngrams_l .>= t])

  m = size(data, 1)
  n = length(ngrams_features)
  I = zeros(Int64, n_f)
  J = zeros(Int64, n_f)
  V = ones(Int64, n_f)

  cnt = 0
  for (i, v) in enumerate(ngrams)
    if t <= length(v)
      cnt += 1
      I[cnt] = i
      J[cnt] = f2i[v[t]]
    end
  end

  sparse(I, J, V, m, n)
end