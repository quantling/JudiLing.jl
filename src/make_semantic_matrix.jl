"""
This a struct that store all information about prelinguistic and their feature
indices.
pS is the cue matrix
f2i is the dictionary return indices giving features
i2f is in another hand return features when giving indices
"""
struct PS_Matrix_Struct
  pS::Union{Matrix, SparseMatrixCSC}
  f2i::Dict
  i2f::Dict
end

"""
  make_pS_matrix(::DataFrame)

This is a function that create prelinguistic matrix giving a csv file.

```julia
utterance = CSV.DataFrame!(CSV.File(joinpath("data", "utterance_mini.csv")))
s_obj_train = JuLDL.make_pS_matrix(utterance)
```
"""
function make_pS_matrix(
  utterances::DataFrame;
  col_name=:CommunicativeIntention::Symbol,
  sep_token="_"::String
  )::PS_Matrix_Struct

  # find out all possible features in this dataset
  features = unique(vcat(
    split.(utterances[:, col_name], sep_token)...))

  # using dict to store feature names
  f2i = Dict(v => i for (i, v) in enumerate(features))
  i2f = Dict(i => v for (i, v) in enumerate(features))

  # find out features for each utterance
  vs = unique.(split.(utterances[:, col_name], sep_token))

  # total number of feature in the entire dataset
  # to initialize a sparse matrix
  n_f = sum([length(v) for v in vs])

  # initialize sparse matrix components
  m = size(utterances, 1)
  n = length(i2f)
  I = zeros(Int64, n_f)
  J = zeros(Int64, n_f)
  V = ones(Int64, n_f)

  # calculate each cell in sparse matrix
  cnt = 0
  for (i, v) in enumerate(vs)
    for (j, f) in enumerate(v)
      cnt += 1
      I[cnt] = i
      J[cnt] = f2i[f]
    end
  end

  # create sparse matrix
  pS = sparse(I, J, V, m, n, *)

  PS_Matrix_Struct(pS, f2i, i2f)
end

"""
  make_pS_matrix(::DataFrame, PS_Matrix_Struct)

This is a function that construct prelinguistic matrix giving utterances and
training s_obj. The feature indices should maintain the same as thoes in s_obj.

```julia
utterance = CSV.DataFrame!(CSV.File(joinpath("data", "utterance_mini.csv")))
s_obj_train = JuLDL.make_pS_matrix(utterance)
s_obj_val = JuLDL.make_pS_matrix(utterance_val, s_obj_train)
```
"""
function make_pS_matrix(
  utterances::DataFrame,
  utterances_train::PS_Matrix_Struct;
  col_name=:CommunicativeIntention::Symbol,
  sep_token="_"::String
  )::PS_Matrix_Struct

  # find out all possible features in this dataset
  features = unique(vcat(
    split.(utterances[:, col_name], sep_token)...))

  # using dict to store feature names
  f2i = utterances_train.f2i
  i2f = utterances_train.i2f

  # find out features for each utterance
  vs = unique.(split.(utterances[:, col_name], sep_token))

  # total number of feature in the entire dataset
  # to initialize a sparse matrix
  n_f = sum([length(v) for v in vs])

  # initialize sparse matrix components
  m = size(utterances, 1)
  n = length(i2f)
  I = zeros(Int64, n_f)
  J = zeros(Int64, n_f)
  V = ones(Int64, n_f)

  # calculate each cell in sparse matrix
  cnt = 0
  for (i, v) in enumerate(vs)
    for (j, f) in enumerate(v)
      cnt += 1
      I[cnt] = i
      J[cnt] = f2i[f]
    end
  end

  # create sparse matrix
  pS = sparse(I, J, V, m, n, *)

  PS_Matrix_Struct(pS, f2i, i2f)
end