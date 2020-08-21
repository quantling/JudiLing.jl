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
  gold_ind::Vector{Vector{Int64}}
  A::SparseMatrixCSC
end

"""
  f()

This is a test
"""
function f() end

"""
  make_cue_matrix()

JuLDL.make_cue_matrix(
data::DataFrame;
grams=3::Int64,
words_column=:Words::Union{String, Symbol},
tokenized=false::Bool,
sep_token=nothing::Union{Nothing, String, Char},
keep_sep=false::Bool,
start_end_token="#"::Union{String, Char},
verbose=false::Bool
)::Cue_Matrix_Struct

This function make cue matrix and corresponding indices giving dataset as csv file.
"""
function make_cue_matrix() end

function make_cue_matrix(
  data::DataFrame;
  grams=3::Int64,
  words_column=:Words::Union{String, Symbol},
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  keep_sep=false::Bool,
  start_end_token="#"::Union{String, Char},
  verbose=false::Bool
  )::Cue_Matrix_Struct

  # split tokens from words or other columns
  if tokenized && !isnothing(sep_token)
    tokens = split.(data[:, words_column], sep_token)
  else
    tokens = split.(data[:, words_column], "")
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
  make_ngrams(
  tokens::Array,
  grams=3::Int64,
  keep_sep=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  start_end_token="#"::Union{String, Char}
  )::Array

given a list of tokens, return all ngrams in a list
"""
function make_ngrams(
  tokens::Array,
  grams=3::Int64,
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