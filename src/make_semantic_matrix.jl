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
This is the function that make prelinguistic semantic matrix.
"""
function make_pS_matrix end

"""
This is the function that make simulated semantic matrix.
"""
function make_S_matrix end

"""
  make_pS_matrix(::DataFrame)

This is a function that create prelinguistic matrix giving a csv file.

...
# Arguments
- `features_col::Symbol=:CommunicativeIntention`: the column name for communicative intention
- `sep_token::String="_"`: the seperated token in the communicative intention column

# Examples
```julia
utterance = CSV.DataFrame!(CSV.File(joinpath("data", "utterance_mini.csv")))
s_obj_train = JudiLing.make_pS_matrix(utterance)
```
...
"""
function make_pS_matrix(
  utterances::DataFrame;
  features_col=:CommunicativeIntention::Symbol,
  sep_token="_"::String
  )::PS_Matrix_Struct

  # find out all possible features in this dataset
  features = unique(vcat(
    split.(utterances[:, features_col], sep_token)...))

  # using dict to store feature names
  f2i = Dict(v => i for (i, v) in enumerate(features))
  i2f = Dict(i => v for (i, v) in enumerate(features))

  # find out features for each utterance
  vs = unique.(split.(utterances[:, features_col], sep_token))

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

...
# Arguments
- `features_col::Symbol=:CommunicativeIntention`: the column name for communicative intention
- `sep_token::String="_"`: the seperated token in the communicative intention column

# Examples
```julia
utterance = CSV.DataFrame!(CSV.File(joinpath("data", "utterance_mini.csv")))
s_obj_train = JudiLing.make_pS_matrix(utterance)
s_obj_val = JudiLing.make_pS_matrix(utterance_val, s_obj_train)
```
...
"""
function make_pS_matrix(
  utterances::DataFrame,
  utterances_train::PS_Matrix_Struct;
  features_col=:CommunicativeIntention::Symbol,
  sep_token="_"::String
  )::PS_Matrix_Struct

  # find out all possible features in this dataset
  features = unique(vcat(
    split.(utterances[:, features_col], sep_token)...))

  # using dict to store feature names
  f2i = utterances_train.f2i
  i2f = utterances_train.i2f

  # find out features for each utterance
  vs = unique.(split.(utterances[:, features_col], sep_token))

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
  make_S_matrix(::DataFrame, ::Vector, ::Vector)

This is a function that create simulated semantic matrix. Give each feature a
random semantic vector, and sum up all features to compose the semantic vector.

...
# Arguments
- `ncol::Integer=200`: the dimension size of vectors, usually the same as cue vectors
- `sd_base_mean::Integer=1`: the sd mean of base features
- `sd_inflection_mean::Integer=1`: the sd mean of inflectional features
- `sd_base::Integer=4`: the sd of base features
- `sd_inflection::Integer=4`: the sd of inflectional features
- `seed::Integer=314`: the random seed
- `isdeep::Bool=true`: if in deep mode, mean of each feature is also randomized 
- `add_noise::Bool=true`: whether to add noise at the end of construction
- `sd_noise::Integer=1`: the sd of the noise matrix
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may exceeds 1 or -1 depending on the sd

# Examples
```julia
# Examples
french = CSV.DataFrame!(CSV.File(joinpath("data", "french_mini.csv")))
S_train = JudiLing.make_S_matrix(
  french,
  ["Lexeme"],
  ["Tense","Aspect","Person","Number","Gender","Class","Mood"])
```
...
"""
function make_S_matrix(
  data::DataFrame,
  base=["Lexeme"]::Vector,
  inflections=["Person", "Number", "Tense", "Voice", "Mood"]::Vector;
  ncol=200::Integer,
  sd_base_mean=1::Integer,
  sd_inflection_mean=1::Integer,
  sd_base=4::Integer,
  sd_inflection=4::Integer,
  seed=314::Integer,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Integer,
  normalized=false::Bool
  )::Matrix

  # collect all infl_features
  base_f = [f for b in base for f in unique(data[:,b])]
  infl_f = [f for i in inflections for f in unique(data[:,i])]

  # maps features to indices
  base_f2i = Dict(v=>i for (i,v) in enumerate(base_f))
  infl_f2i = Dict(v=>i for (i,v) in enumerate(infl_f))

  Random.seed!(seed)
  if isdeep # deep mode random means for each feature
    base_means = rand(Normal(0, sd_base_mean), length(base_f))
    infl_means = rand(Normal(0, sd_inflection_mean), length(infl_f))

    base_m = [rand(Normal(base_means[i], sd_base), ncol) for i in 1:length(base_f)]
    infl_m = [rand(Normal(infl_means[i], sd_inflection), ncol) for i in 1:length(infl_f)]
  else # otherwise use mean=0 for all features
    base_m = [rand(Normal(0, sd_base), ncol) for i in 1:length(base_f)]
    infl_m = [rand(Normal(0, sd_inflection), ncol) for i in 1:length(infl_f)]
  end

  # julia is column-wise language
  # assign St first then do transpose is faster
  St = Array{AbstractFloat, 2}(undef, ncol, size(data, 1))
  for i in 1:size(data, 1)
    s_base = sum([base_m[base_f2i[f]] for f in data[i, base]])
    s_infl = sum([infl_m[infl_f2i[f]] for f in data[i, inflections]])
    s = s_base + s_infl
    St[:,i] = s
  end

  # add random var to S
  if add_noise
      noise = rand(Normal(0, sd_noise), size(St, 1), size(St, 2))
      St += noise
  end
  if normalized
    n_features = length(base) + length(inflections)
    return St'./n_features
  end
  St'
end

"""
  make_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector)

This is a function that create validation simulated semantic matrix. Give each feature a
random semantic vector, and sum up all features to compose the semantic vector.

...
# Arguments
- `ncol::Integer=200`: the dimension size of vectors, usually the same as cue vectors
- `sd_base_mean::Integer=1`: the sd mean of base features
- `sd_inflection_mean::Integer=1`: the sd mean of inflectional features
- `sd_base::Integer=4`: the sd of base features
- `sd_inflection::Integer=4`: the sd of inflectional features
- `seed::Integer=314`: the random seed
- `isdeep::Bool=true`: if in deep mode, mean of each feature is also randomized 
- `add_noise::Bool=true`: whether to add noise at the end of construction
- `sd_noise::Integer=1`: the sd of the noise matrix
- `normalized::Bool=false`: if normalized, values of matrix maintain close between 1 and -1
# Examples

```julia
# Examples
french = CSV.DataFrame!(CSV.File(joinpath("data", "french_mini.csv")))
french_val = french[100:end,:]
S_train, S_val = JudiLing.make_S_matrix(
    french,
    french_val,
    ["Lexeme"],
    ["Tense","Aspect","Person","Number","Gender","Class","Mood"])
```
...
"""
function make_S_matrix(
  data_train::DataFrame,
  data_val::DataFrame,
  base=["Lexeme"]::Vector,
  inflections=["Person", "Number", "Tense", "Voice", "Mood"]::Vector;
  ncol=200::Integer,
  sd_base_mean=1::Integer,
  sd_inflection_mean=1::Integer,
  sd_base=4::Integer,
  sd_inflection=4::Integer,
  seed=314::Integer,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Integer,
  normalized=false::Bool
  )::Tuple{Matrix, Matrix}

  # collect all infl_features
  base_f = [f for b in base for f in unique(data_train[:,b])]
  infl_f = [f for i in inflections for f in unique(data_train[:,i])]

  # maps features to indices
  base_f2i = Dict(v=>i for (i,v) in enumerate(base_f))
  infl_f2i = Dict(v=>i for (i,v) in enumerate(infl_f))

  Random.seed!(seed)
  if isdeep # deep mode random means for each feature
    base_means = rand(Normal(0, sd_base_mean), length(base_f))
    infl_means = rand(Normal(0, sd_inflection_mean), length(infl_f))

    base_m = [rand(Normal(base_means[i], sd_base), ncol) for i in 1:length(base_f)]
    infl_m = [rand(Normal(infl_means[i], sd_inflection), ncol) for i in 1:length(infl_f)]
  else # otherwise use mean=0 for all features
    base_m = [rand(Normal(0, sd_base), ncol) for i in 1:length(base_f)]
    infl_m = [rand(Normal(0, sd_inflection), ncol) for i in 1:length(infl_f)]
  end

  # julia is column-wise language
  # assign St first then do transpose is faster
  St_train = Array{AbstractFloat, 2}(undef, ncol, size(data_train, 1))
  for i in 1:size(data_train, 1)
    s_base = sum([base_m[base_f2i[f]] for f in data_train[i, base]])
    s_infl = sum([infl_m[infl_f2i[f]] for f in data_train[i, inflections]])
    s = s_base + s_infl
    St_train[:,i] = s
  end

  # add random var to S
  if add_noise
      noise = rand(Normal(0, sd_noise), size(St_train, 1), size(St_train, 2))
      St_train += noise
  end

  # julia is column-wise language
  # assign St first then do transpose is faster
  St_val = Array{AbstractFloat, 2}(undef, ncol, size(data_val, 1))
  for i in 1:size(data_val, 1)
    s_base = sum([base_m[base_f2i[f]] for f in data_val[i, base]])
    s_infl = sum([infl_m[infl_f2i[f]] for f in data_val[i, inflections]])
    s = s_base + s_infl
    St_val[:,i] = s
  end

  # add random var to S
  if add_noise
      noise = rand(Normal(0, sd_noise), size(St_val, 1), size(St_val, 2))
      St_val += noise
  end

  if normalized
    n_features = length(base) + length(inflections)
    return St_train'./n_features, St_val'./n_features
  end
  St_train', St_val'
end

"""
  make_S_matrix(::DataFrame)

This is a function that create simulated semantic matrix, provided
for dataset that only have base features. Give each feature a random semantic
vector, and sum up all features to compose the semantic vector.

...
# Arguments
- `base::Vector=["Lexeme"]`: the base features 
- `ncol::Integer=200`: the dimension size of vectors, usually the same as cue vectors
- `sd_base_mean::Integer=1`: the sd mean of base features
- `sd_base::Integer=4`: the sd of base features
- `seed::Integer=314`: the random seed
- `isdeep::Bool=true`: if in deep mode, mean of each feature is also randomized 
- `add_noise::Bool=true`: whether to add noise at the end of construction
- `sd_noise::Integer=1`: the sd of the noise matrix

# Examples
```julia
french = CSV.DataFrame!(CSV.File(joinpath("data", "french_mini.csv")))

S_train = JudiLing.make_S_matrix(
  french,
  base=["Lexeme"])=
```
...
"""
function make_S_matrix(
  data::DataFrame;
  base=["Lexeme"]::Vector,
  ncol=200::Integer,
  sd_base_mean=1::Integer,
  sd_base=4::Integer,
  seed=314::Integer,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Integer
  )::Matrix

  # collect all infl_features
  base_f = [f for b in base for f in unique(data[:,b])]

  # maps features to indices
  base_f2i = Dict(v=>i for (i,v) in enumerate(base_f))

  if isdeep # deep mode random means for each feature
    base_means = rand(Normal(0, sd_base_mean), length(base_f))
    base_m = [rand(Normal(base_means[i], sd_base), ncol) for i in 1:length(base_f)]
  else # otherwise use mean=0 for all features
    base_m = [rand(Normal(0, sd_base), ncol) for i in 1:length(base_f)]
  end

  # julia is column-wise language
  # assign St first then do transpose is faster
  St = Array{AbstractFloat, 2}(undef, ncol, size(data, 1))
  for i in 1:size(data, 1)
    s_base = sum([base_m[base_f2i[f]] for f in data[i, base]])
    s = s_base
    St[:,i] = s
  end

  # add random var to S
  if add_noise
      noise = rand(Normal(0, sd_noise), size(St, 1), size(St, 2))
      St += noise
  end

  St'
end

"""
  make_S_matrix(::DataFrame, ::DataFrame)

This is a function that create validation simulated semantic matrix, provided
for dataset that only have base features. Give each feature a random semantic
vector, and sum up all features to compose the semantic vector.

...
# Arguments
- `base::Vector=["Lexeme"]`: the base features 
- `ncol::Integer=200`: the dimension size of vectors, usually the same as cue vectors
- `sd_base_mean::Integer=1`: the sd mean of base features
- `sd_base::Integer=4`: the sd of base features
- `seed::Integer=314`: the random seed
- `isdeep::Bool=true`: if in deep mode, mean of each feature is also randomized 
- `add_noise::Bool=true`: whether to add noise at the end of construction
- `sd_noise::Integer=1`: the sd of the noise matrix

# Examples
```julia
french = CSV.DataFrame!(CSV.File(joinpath("data", "french_mini.csv")))
french_val = french[100:end,:]
S_train, S_val = JudiLing.make_S_matrix(
    french,
    french_val,
    base=["Lexeme"])
```
...
"""
function make_S_matrix(
  data_train::DataFrame,
  data_val::DataFrame;
  base=["Lexeme"]::Array,
  ncol=200::Integer,
  sd_base_mean=1::Integer,
  sd_base=4::Integer,
  seed=314::Integer,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Integer
  )::Tuple{Matrix, Matrix}

  # collect all infl_features
  base_f = [f for b in base for f in unique(data_train[:,b])]

  # maps features to indices
  base_f2i = Dict(v=>i for (i,v) in enumerate(base_f))

  if isdeep # deep mode random means for each feature
    base_means = rand(Normal(0, sd_base_mean), length(base_f))
    base_m = [rand(Normal(base_means[i], sd_base), ncol) for i in 1:length(base_f)]
  else # otherwise use mean=0 for all features
    base_m = [rand(Normal(0, sd_base), ncol) for i in 1:length(base_f)]
  end

  # julia is column-wise language
  # assign St first then do transpose is faster
  St_train = Array{AbstractFloat, 2}(undef, ncol, size(data_train, 1))
  for i in 1:size(data_train, 1)
    s_base = sum([base_m[base_f2i[f]] for f in data_train[i, base]])
    s = s_base
    St_train[:,i] = s
  end

  # add random var to S
  if add_noise
      noise = rand(Normal(0, sd_noise), size(St_train, 1), size(St_train, 2))
      St_train += noise
  end

  # julia is column-wise language
  # assign St first then do transpose is faster
  St_val = Array{AbstractFloat, 2}(undef, ncol, size(data_val, 1))
  for i in 1:size(data_val, 1)
    s_base = sum([base_m[base_f2i[f]] for f in data_val[i, base]])
    s = s_base
    St_val[:,i] = s
  end

  # add random var to S
  if add_noise
      noise = rand(Normal(0, sd_noise), size(St_val, 1), size(St_val, 2))
      St_val += noise
  end

  St_train', St_val'
end
