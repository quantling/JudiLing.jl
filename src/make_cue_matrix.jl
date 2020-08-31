"""
a struct that store info after make_cue_matrix
C is the cue matrix
f2i is the dictionary return indices giving features
i2f is in another hand return features when giving indices
gold_ind stores gold paths within a list of indices
A is the adjacency matrix
"""
struct Cue_Matrix_Struct
  C::Union{Matrix, SparseMatrixCSC}
  f2i::Dict
  i2f::Dict
  gold_ind::Vector{Vector{Integer}}
  A::SparseMatrixCSC
end

"""
Construct cue matrix.
"""
function make_cue_matrix end

"""
Given tokens make n-grams.
"""
function make_ngrams end

"""
  make_cue_matrix(::DataFrame)

This function makes cue matrix and corresponding indices given dataset as csv file.

...
# Arguments
- `grams::Integer=3`: the number of grams for cues 
- `target_col::Union{String, Symbol}=:Words`: the column name for target
- `tokenized::Bool=false`: whether the word is tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: what is the seperate token
- `keep_sep::Bool=false`: whether to keep seperater in cues
- `start_end_token::Union{String, Char}="#"`: start and end token
- `verbose::Bool=false`: if verbose, more information prints out

# Examples
```julia
latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
latin_cue_obj_train = JudiLing.make_cue_matrix(
  latin,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )
```
...
"""
function make_cue_matrix(
  data::DataFrame;
  grams=3::Integer,
  target_col=:Words::Union{String, Symbol},
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  keep_sep=false::Bool,
  start_end_token="#"::Union{String, Char},
  verbose=false::Bool
  )::Cue_Matrix_Struct

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

  n_f = sum([length(v) for v in ngrams])

  m = size(data, 1)
  n = length(ngrams_features)
  I = zeros(Int64, n_f)
  J = zeros(Int64, n_f)
  V = ones(Int64, n_f)

  A = [Int64[] for i in 1:length(ngrams_features)]

  cnt = 0
  for (i, v) in enumerate(ngrams)
    last = 0
    for (j, f) in enumerate(v)
      cnt += 1
      I[cnt] = i
      fi = f2i[f]
      J[cnt] = fi
      if j == 1
        last = fi
      else
        push!(A[last], fi)
        last = fi
      end
    end
  end

  cue = sparse(I, J, V, m, n, *)

  ngrams_ind = [[f2i[x] for x in y] for y in ngrams]

  verbose && println("making adjacency matrix...")
  A = [sort(unique(i)) for i in A]
  n_adj = sum(length.(A))
  I = zeros(Int64, n_adj)
  J = zeros(Int64, n_adj)
  V = ones(Int64, n_adj)

  cnt = 0
  iter = enumerate(A)
  verbose && begin iter = tqdm(iter) end
  for (i,v) in iter
    for j in v
      cnt += 1
      I[cnt] = i
      J[cnt] = j
    end
  end

  A = sparse(I, J, V, length(f2i), length(f2i))

  Cue_Matrix_Struct(cue, f2i, i2f, ngrams_ind, A)
end

"""
  make_cue_matrix(::DataFrame,::Cue_Matrix_Struct)

This function make cue matrix and corresponding indices giving dataset as csv file and
train dataset cue obj. This is often used to construct val_cue_obj, in order to maintain
the same indices.

...
# Arguments
- `grams::Integer=3`: the number of grams for cues 
- `target_col::Union{String, Symbol}=:Words`: the column name for target
- `tokenized::Bool=false`: whether the word is tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: what is the seperate token
- `keep_sep::Bool=false`: whether to keep seperater in cues
- `start_end_token::Union{String, Char}="#"`: start and end token
- `verbose::Bool=false`: if verbose, more information prints out

# Examples
```julia
latin = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
latin_cue_obj_train = JudiLing.make_cue_matrix(
  latin,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )
# simulate the val dataset. Notice here that latin_val is part of training dataset to make
# sure all features and n-grams covered by training dataset.
latin_val = latin[101:150,:]
latin_cue_obj_val = JudiLing.make_cue_matrix(
  latin_val,
  latin_cue_obj_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )
```
...
"""
function make_cue_matrix(
  data::DataFrame,
  cue_obj::Cue_Matrix_Struct;
  grams=3::Integer,
  target_col="Words"::String,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  keep_sep=false::Bool,
  start_end_token="#"::Union{String, Char},
  verbose=false::Bool
  )::Cue_Matrix_Struct

  # split tokens from words or other columns
  if tokenized && !isnothing(sep_token)
    tokens = split.(data[:, target_col], sep_token)
  else
    tokens = split.(data[:, target_col], "")
  end

  # making ngrams from tokens
  # make_ngrams function are below
  ngrams = make_ngrams.(tokens, grams, keep_sep, sep_token, start_end_token)

  f2i = cue_obj.f2i
  i2f = cue_obj.i2f

  n_f = sum([length(v) for v in ngrams])

  m = size(data, 1)
  n = length(f2i)
  I = zeros(Int64, n_f)
  J = zeros(Int64, n_f)
  V = ones(Int64, n_f)


  cnt = 0
  for (i, v) in enumerate(ngrams)
    for (j, f) in enumerate(v)
      cnt += 1
      I[cnt] = i
      J[cnt] = f2i[f]
    end
  end

  cue = sparse(I, J, V, m, n, *)
  ngrams_ind = [[f2i[x] for x in y] for y in ngrams]

  Cue_Matrix_Struct(cue, f2i, i2f, ngrams_ind, cue_obj.A)
end

"""
  make_ngrams(::Array,::Integer,::Bool,
  ::Union{Nothing, String, Char},::Union{String, Char}

given a list of tokens, return all ngrams in a list
"""
function make_ngrams(
  tokens::Array,
  grams=3::Integer,
  keep_sep=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  start_end_token="#"::Union{String, Char}
  )::Array

  push!(pushfirst!(tokens, start_end_token), start_end_token)
  if keep_sep
    # collect ngrams
    ngrams = join.(collect(zip((
      Iterators.drop(tokens, k) for k = 0:grams-1)...)), sep_token)
  else
    ngrams = join.(collect(zip((
      Iterators.drop(tokens, k) for k = 0:grams-1)...)), "")
  end

  ngrams
end