"""
A structure that stores the discrete semantic vectors:
pS is the discrete semantic matrix;
f2i is a dictionary returning the indices for features;
i2f is a dictionary returning the features for indices.
"""
struct PS_Matrix_Struct
  pS::Union{Matrix, SparseMatrixCSC}
  f2i::Dict
  i2f::Dict
end

"""
A structure that stores Lexome semantic vectors:
L is Lexome semantic matrix;
f2i is a dictionary returning the indices for features;
i2f is a dictionary returning the features for indices.
"""
struct L_Matrix_Struct
  L::Matrix{Float64}
  f2i::Dict{AbstractString,Int64}
  i2f::Vector{AbstractString}
  ncol::Int64
end

"""
Make discrete semantic matrix.
"""
function make_pS_matrix end

"""
Make simulated semantic matrix.
"""
function make_S_matrix end

"""
Make simulated lexome matrix.
"""
function make_L_matrix end

"""
Make combined simulated S matrices, where combined features from both training datasets and validation datasets
"""
function make_combined_S_matrix end

"""
Make combined simulated Lexome matrix, where combined features from both training datasets and validation datasets
"""
function make_combined_L_matrix end

"""
    make_pS_matrix(::DataFrame) -> ::PS_Matrix_Struct

Create a discrete semantic matrix given a dataframe.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset

# Optional Arguments
- `features_col::Symbol=:CommunicativeIntention`: the column name for target
- `sep_token::String="_"`: separator

# Examples
```julia
s_obj_train = JudiLing.make_pS_matrix(
  utterance,
  features_col=:CommunicativeIntention,
  sep_token="_")
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
    make_pS_matrix(::DataFrame, ::PS_Matrix_Struct) -> ::PS_Matrix_Struct

Construct discrete semantic matrix for the validation datasets given by the 
exemplar in the dataframe, and given the S matrix for the training datasets.

...
# Obligatory Arguments
- `utterances::DataFrame`: the dataset
- `utterances_train::PS_Matrix_Struct`: training PS object

# Optional Arguments
- `features_col::Symbol=:CommunicativeIntention`: the column name for target
- `sep_token::String="_"`: separator

# Examples
```julia
s_obj_val = JudiLing.make_pS_matrix(
  utterance_val,
  s_obj_train,
  features_col=:CommunicativeIntention,
  sep_token="_")
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
    make_S_matrix(::DataFrame, ::Vector, ::Vector) -> ::Matrix

Create simulated semantic matrix for the training datasets, given the input
data of a vector specified contex lexemes and a vector specified gramatic 
lexemes. The semantic vector of a word form is constructed summing semantic 
vectors of content and gramatic lexemes.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset
- `base::Vector`: context lexemes
- `inflections::Vector`: grammatic lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_inflection_mean::Int64=1`: the sd mean of inflectional features
- `sd_base::Int64=4`: the sd of base features
- `sd_inflection::Int64=4`: the sd of inflectional features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized 
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S_train = JudiLing.make_S_matrix(
  french,
  ["Lexeme"],
  ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
  ncol=200)

# deep mode
S_train = JudiLing.make_S_matrix(
  ...
  sd_base_mean=1,
  sd_inflection_mean=1,
  isdeep=true,
  ...)

# non-deep mode
S_train = JudiLing.make_S_matrix(
  ...
  isdeep=false,
  ...)

# add additional Gaussian noise
S_train = JudiLing.make_S_matrix(
  ...
  add_noise=true,
  sd_noise=1,
  ...)

# further control of means and standard deviations
S_train = JudiLing.make_S_matrix(
  ...
  sd_base_mean=1,
  sd_inflection_mean=1,
  sd_base=4,
  sd_inflection=4,
  sd_noise=1,
  ...)
```
...
"""
function make_S_matrix(
  data::DataFrame,
  base::Vector,
  inflections::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_inflection_mean=1::Int64,
  sd_base=4::Int64,
  sd_inflection=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Int64,
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
  St = Array{Float64, 2}(undef, ncol, size(data, 1))
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
    make_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector) -> ::Tuple{Matrix, Matrix}

Create simulated semantic matrix for the validation datasets, given the input
data of a vector specified contex lexemes and a vector specified gramatic 
lexemes. The semantic vector of a word form is constructed summing semantic 
vectors of content and gramatic lexemes.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes
- `inflections::Vector`: grammatic lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_inflection_mean::Int64=1`: the sd mean of inflectional features
- `sd_base::Int64=4`: the sd of base features
- `sd_inflection::Int64=4`: the sd of inflectional features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized 
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S_train, S_val = JudiLing.make_S_matrix(
  french,
  french_val,
  ["Lexeme"],
  ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
  ncol=200)

# deep mode
S_train, S_val = JudiLing.make_S_matrix(
  ...
  sd_base_mean=1,
  sd_inflection_mean=1,
  isdeep=true,
  ...)

# non-deep mode
S_train, S_val = JudiLing.make_S_matrix(
  ...
  isdeep=false,
  ...)

# add additional Gaussian noise
S_train, S_val = JudiLing.make_S_matrix(
  ...
  add_noise=true,
  sd_noise=1,
  ...)

# further control of means and standard deviations
S_train, S_val = JudiLing.make_S_matrix(
  ...
  sd_base_mean=1,
  sd_inflection_mean=1,
  sd_base=4,
  sd_inflection=4,
  sd_noise=1,
  ...)
```
...
"""
function make_S_matrix(
  data_train::DataFrame,
  data_val::DataFrame,
  base::Vector,
  inflections::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_inflection_mean=1::Int64,
  sd_base=4::Int64,
  sd_inflection=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Int64,
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
  St_train = Array{Float64, 2}(undef, ncol, size(data_train, 1))
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
  St_val = Array{Float64, 2}(undef, ncol, size(data_val, 1))
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
    make_S_matrix(::DataFrame, ::Vector) -> ::Matrix

Create simulated semantic matrix for the training datasets with only base 
features, given the input data of a vector specified contex lexemes and a 
vector specified gramatic lexemes. The semantic vector of a word form is 
constructed summing semantic vectors of content and gramatic lexemes.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset
- `base::Vector`: context lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_base::Int64=4`: the sd of base features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized 
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd


# Examples
```julia
# basic usage
S_train = JudiLing.make_S_matrix(
  french,
  ["Lexeme"],
  ncol=200)

# deep mode
S_train = JudiLing.make_S_matrix(
  ...
  sd_base_mean=1,
  sd_inflection_mean=1,
  isdeep=true,
  ...)

# non-deep mode
S_train = JudiLing.make_S_matrix(
  ...
  isdeep=false,
  ...)

# add additional Gaussian noise
S_train = JudiLing.make_S_matrix(
  ...
  add_noise=true,
  sd_noise=1,
  ...)

# further control of means and standard deviations
S_train = JudiLing.make_S_matrix(
  ...
  sd_base_mean=1,
  sd_inflection_mean=1,
  sd_base=4,
  sd_inflection=4,
  sd_noise=1,
  ...)
```
...
"""
function make_S_matrix(
  data::DataFrame,
  base::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_base=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Int64
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
  St = Array{Float64, 2}(undef, ncol, size(data, 1))
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
    make_S_matrix(::DataFrame, ::DataFrame, ::Vector) -> ::Tuple{Matrix, Matrix}

Create simulated semantic matrix for the validation datasets with only base 
features, given the input data of a vector specified contex lexemes and a 
vector specified gramatic lexemes. The semantic vector of a word form is 
constructed summing semantic vectors of content and gramatic lexemes.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_base::Int64=4`: the sd of base features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized 
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S_train, S_val = JudiLing.make_S_matrix(
  french,
  french_val,
  ["Lexeme"],
  ncol=200)

# deep mode
S_train, S_val = JudiLing.make_S_matrix(
  ...
  sd_base_mean=1,
  sd_inflection_mean=1,
  isdeep=true,
  ...)

# non-deep mode
S_train, S_val = JudiLing.make_S_matrix(
  ...
  isdeep=false,
  ...)

# add additional Gaussian noise
S_train, S_val = JudiLing.make_S_matrix(
  ...
  add_noise=true,
  sd_noise=1,
  ...)

# further control of means and standard deviations
S_train, S_val = JudiLing.make_S_matrix(
  ...
  sd_base_mean=1,
  sd_inflection_mean=1,
  sd_base=4,
  sd_inflection=4,
  sd_noise=1,
  ...)
```
...
"""
function make_S_matrix(
  data_train::DataFrame,
  data_val::DataFrame,
  base::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_base=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Int64
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
  St_train = Array{Float64, 2}(undef, ncol, size(data_train, 1))
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
  St_val = Array{Float64, 2}(undef, ncol, size(data_val, 1))
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

"""
    make_S_matrix(::DataFrame, ::DataFrame, ::Pyndl_Weight_Struct, ::Vector{String}) -> ::Tuple{Matrix, Matrix}

Create semantic matrix for pyndl mode
"""
function make_S_matrix(
  data_train::DataFrame,
  data_val::DataFrame,
  pyndl_weights::Pyndl_Weight_Struct,
  n_features_columns::Vector{String}
  )::Tuple{Matrix, Matrix}
  
  f2i = Dict(v => i for (i, v) in enumerate(pyndl_weights.outcomes))
  i2f = Dict(i => v for (i, v) in enumerate(pyndl_weights.outcomes))

  n_f = length(pyndl_weights.outcomes)

  St_train = zeros(Float64, n_f, size(data_train, 1))
  for i in 1:size(data_train, 1)
    for f in data_train[i, n_features_columns]
      St_train[f2i[f],i] = 1
    end
  end

  St_val = zeros(Float64, n_f, size(data_val, 1))
  for i in 1:size(data_val, 1)
    for f in data_val[i, n_features_columns]
      St_val[f2i[f],i] = 1
    end
  end

  St_train', St_val'
end

"""
    make_S_matrix(::DataFrame, ::Vector, ::Vector, ::L_Matrix_Struct) -> ::Matrix

Create simulated semantic matrix where lexome matrix is available.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset
- `base::Vector`: context lexemes
- `inflections::Vector`: grammatic lexemes
- `L::L_Matrix_Struct`: the lexome matrix

# Optional Arguments
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S1 = JudiLing.make_S_matrix(
  latin,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  L1,
  add_noise=true,
  sd_noise=1,
  normalized=false
  )
```
...
"""
function make_S_matrix(
  data::DataFrame,
  base::Vector,
  inflections::Vector,
  L::L_Matrix_Struct;
  add_noise=true::Bool,
  sd_noise=1::Int64,
  normalized=false::Bool
  )::Matrix
  
  JudiLing.make_S_matrix(
    data,
    nothing,
    base,
    inflections,
    L,
    add_noise=add_noise,
    sd_noise=sd_noise,
    normalized=normalized
    )
end

"""
    make_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::L_Matrix_Struct) -> ::Union{Matrix, Tuple{Matrix, Matrix}}

Create simulated semantic matrix where lexome matrix is available.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes
- `L::L_Matrix_Struct`: the lexome matrix

# Optional Arguments
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S1, S2 = JudiLing.make_S_matrix(
  latin,
  latin_val,
  ["Lexeme"],
  L1,
  add_noise=true,
  sd_noise=1,
  normalized=false
  )
```
...
"""
function make_S_matrix(
  data_train::DataFrame,
  data_val::Union{DataFrame, Nothing},
  base::Vector,
  L::L_Matrix_Struct;
  add_noise=true::Bool,
  sd_noise=1::Int64,
  normalized=false::Bool
  )::Union{Matrix, Tuple{Matrix, Matrix}}
  
  JudiLing.make_S_matrix(
    data_train,
    data_val,
    base,
    nothing,
    L,
    add_noise=add_noise,
    sd_noise=sd_noise,
    normalized=normalized
    )

end

"""
    make_S_matrix(::DataFrame, ::Vector, ::L_Matrix_Struct) -> ::Matrix

Create simulated semantic matrix where lexome matrix is available.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset
- `base::Vector`: context lexemes
- `L::L_Matrix_Struct`: the lexome matrix

# Optional Arguments
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S1 = JudiLing.make_S_matrix(
  latin,
  ["Lexeme"],
  L1,
  add_noise=true,
  sd_noise=1,
  normalized=false
  )
```
...
"""
function make_S_matrix(
  data::DataFrame,
  base::Vector,
  L::L_Matrix_Struct;
  add_noise=true::Bool,
  sd_noise=1::Int64,
  normalized=false::Bool
  )::Matrix

  JudiLing.make_S_matrix(
    data,
    nothing,
    base,
    nothing,
    L,
    add_noise=add_noise,
    sd_noise=sd_noise,
    normalized=normalized
    )

end

"""
    make_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector, ::L_Matrix_Struct) -> ::Union{Matrix, Tuple{Matrix, Matrix}}

Create simulated semantic matrix where lexome matrix is available.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes
- `inflections::Vector`: grammatic lexemes
- `L::L_Matrix_Struct`: the lexome matrix

# Optional Arguments
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S1, S2 = JudiLing.make_S_matrix(
  latin,
  latin_val,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  L1,
  add_noise=true,
  sd_noise=1,
  normalized=false
  )
```
...
"""
function make_S_matrix(
  data_train::DataFrame,
  data_val::Union{DataFrame, Nothing},
  base::Vector,
  inflections::Union{Vector, Nothing},
  L::L_Matrix_Struct;
  add_noise=true::Bool,
  sd_noise=1::Int64,
  normalized=false::Bool
  )::Union{Matrix, Tuple{Matrix, Matrix}}
  
  St_train = Array{Float64, 2}(undef, L.ncol, size(data_train, 1))

  for i in 1:size(data_train, 1)
    if !isnothing(inflections)
      s_base = sum([L.L[L.f2i[f],:] for f in skipmissing(data_train[i, base])])
      s_infl = sum([L.L[L.f2i[f],:] for f in skipmissing(data_train[i, inflections])])
      s = s_base + s_infl
    else
      s_base = sum([L.L[L.f2i[f],:] for f in skipmissing(data_train[i, base])])
      s = s_base
    end
    St_train[:,i] = s
  end

  if add_noise
      noise = rand(Normal(0, sd_noise), size(St_train, 1), size(St_train, 2))
      St_train += noise
  end

  if normalized
    n_features = length(base) + length(inflections)
    St_train = St_train./n_features
  end

  if !isnothing(data_val)
    St_val = Array{Float64, 2}(undef, L.ncol, size(data_val, 1))

    for i in 1:size(data_val, 1)
      if !isnothing(inflections)
        s_base = sum([L.L[L.f2i[f],:] for f in data_val[i, base]])
        s_infl = sum([L.L[L.f2i[f],:] for f in data_val[i, inflections]])
        s = s_base + s_infl
      else
        s_base = sum([L.L[L.f2i[f],:] for f in data_val[i, base]])
        s = s_base
      end
      St_val[:,i] = s
    end

    if add_noise
        noise = rand(Normal(0, sd_noise), size(St_val, 1), size(St_val, 2))
        St_val += noise
    end

    if normalized
      if !isnothing(inflections)
        n_features = length(base) + length(inflections)
      else
        n_features = length(base)
      end
      St_val = St_val./n_features
    end

    return Array(St_train'), Array(St_val')
  end

  Array(St_train')
end

"""
    make_L_matrix(::DataFrame, ::Vector, ::Vector) -> ::L_Matrix_Struct

Create Lexome Matrix with simulated semantic vectors.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset
- `base::Vector`: context lexemes
- `inflections::Vector`: grammatic lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_inflection_mean::Int64=1`: the sd mean of inflectional features
- `sd_base::Int64=4`: the sd of base features
- `sd_inflection::Int64=4`: the sd of inflectional features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized

# Examples
```julia
# basic usage
L = JudiLing.make_L_matrix(
  latin,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=200)
```
...
"""
function make_L_matrix(
  data::DataFrame,
  base::Vector,
  inflections::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_inflection_mean=1::Int64,
  sd_base=4::Int64,
  sd_inflection=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool
  )::L_Matrix_Struct
  
  # collect all features and f2i mappings
  base_f, base_f2i = process_features(data, base)
  infl_f, infl_f2i = process_features(data, inflections)

  # setup seed
  Random.seed!(seed)

  n_base_f = length(base_f)
  n_infl_f = length(infl_f)

  # pre-allocated L matrix
  L = Matrix{Float64}(undef, (n_base_f+n_infl_f ,ncol))

  if isdeep # deep mode random means for each feature
    return L_Matrix_Struct(L, sd_base, sd_base_mean, 
      sd_inflection, sd_inflection_mean, base_f, infl_f, 
      base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)
  else # otherwise use mean=0 for all features
    return L_Matrix_Struct(L, sd_base, sd_inflection, base_f, infl_f, 
      base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)
  end
end

"""
    make_L_matrix(::DataFrame, ::Vector) -> ::L_Matrix_Struct

Create Lexome Matrix with simulated semantic vectors where there are only base features.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset
- `base::Vector`: context lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_base::Int64=4`: the sd of base features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized

# Examples
```julia
# basic usage
L = JudiLing.make_L_matrix(
  latin,
  ["Lexeme"],
  ncol=200)
```
...
"""
function make_L_matrix(
  data::DataFrame,
  base::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_base=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool
  )::L_Matrix_Struct
  
  make_L_matrix(
    data,
    base,
    [];
    ncol=ncol,
    sd_base_mean=sd_base_mean,
    sd_base=sd_base,
    seed=seed,
    isdeep=isdeep
    )
end

"""
    make_combined_L_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector) -> ::L_Matrix_Struct

Create Lexome Matrix with simulated semantic vectors, where features are 
combined from both training datasets and validation datasets.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes
- `inflections::Vector`: grammatic lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_inflection_mean::Int64=1`: the sd mean of inflectional features
- `sd_base::Int64=4`: the sd of base features
- `sd_inflection::Int64=4`: the sd of inflectional features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized

# Examples
```julia
# basic usage
L = JudiLing.make_combined_L_matrix(
  latin_train,
  latin_val,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features)
```
...
"""
function make_combined_L_matrix(
  data_train::DataFrame,
  data_val::DataFrame,
  base::Vector,
  inflections::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_inflection_mean=1::Int64,
  sd_base=4::Int64,
  sd_inflection=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool
  )::L_Matrix_Struct

  data_combined = copy(data_train)
  append!(data_combined, data_val)

  make_L_matrix(
    data_combined,
    base,
    inflections;
    ncol=ncol,
    sd_base_mean=sd_base_mean,
    sd_base=sd_base,
    seed=seed,
    isdeep=isdeep
    )

end

"""
    make_combined_L_matrix(::DataFrame, ::DataFrame, ::Vector) -> ::L_Matrix_Struct

Create Lexome Matrix with simulated semantic vectors, where features are 
combined from both training datasets and validation datasets.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_inflection_mean::Int64=1`: the sd mean of inflectional features
- `sd_base::Int64=4`: the sd of base features
- `sd_inflection::Int64=4`: the sd of inflectional features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized

# Examples
```julia
# basic usage
L = JudiLing.make_combined_L_matrix(
  latin_train,
  latin_val,
  ["Lexeme"],
  ncol=n_features)
```
...
"""
function make_combined_L_matrix(
  data_train::DataFrame,
  data_val::DataFrame,
  base::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_inflection_mean=1::Int64,
  sd_base=4::Int64,
  sd_inflection=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool
  )::L_Matrix_Struct
  
  make_combined_L_matrix(
    data_train,
    data_val,
    base,
    [];
    ncol=ncol,
    sd_base_mean=sd_base_mean,
    sd_base=sd_base,
    seed=seed,
    isdeep=isdeep
    )
end

"""
    make_combined_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector, ::L_Matrix_Struct) -> ::Tuple{Matrix, Matrix}

Create simulated semantic matrix for the training datasets and validation 
datasets with existing Lexome matrix, where features are combined from both 
training datasets and validation datasets.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes
- `inflections::Vector`: grammatic lexemes
- `L::L_Matrix_Struct`: the Lexome Matrix

# Optional Arguments
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S_train, S_val = JudiLing.make_combined_S_matrix(
  latin_train,
  latin_val,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  L)
```
...
"""
function make_combined_S_matrix(
  data_train::DataFrame,
  data_val::Union{DataFrame, Nothing},
  base::Vector,
  inflections::Union{Vector, Nothing},
  L::L_Matrix_Struct;
  add_noise=true::Bool,
  sd_noise=1::Int64,
  normalized=false::Bool
  )::Tuple{Matrix, Matrix}

  make_S_matrix(
    data_train,
    data_val,
    base,
    inflections,
    L;
    add_noise=add_noise,
    sd_noise=sd_noise,
    normalized=normalized
    )
end

"""
    make_combined_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector) -> ::Tuple{Matrix, Matrix}

Create simulated semantic matrix for the training datasets and validation 
datasets, where features are combined from both training datasets and 
validation datasets.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes
- `inflections::Vector`: grammatic lexemes

# Optional Arguments
- `ncol::Int64=200`: dimension of semantic vectors, usually the same as that of cue vectors
- `sd_base_mean::Int64=1`: the sd mean of base features
- `sd_inflection_mean::Int64=1`: the sd mean of inflectional features
- `sd_base::Int64=4`: the sd of base features
- `sd_inflection::Int64=4`: the sd of inflectional features
- `seed::Int64=314`: the random seed
- `isdeep::Bool=true`: if true, mean of each feature is also randomized 
- `add_noise::Bool=true`: if true, add additional Gaussian noise
- `sd_noise::Int64=1`: the sd of the Gaussian noise
- `normalized::Bool=false`: if true, most of the values range between 1 and -1, it may slightly exceed between 1 or -1 depending on the sd

# Examples
```julia
# basic usage
S_train, S_val = JudiLing.make_combined_S_matrix(
  latin_train,
  latin_val,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features)
```
...
"""
function make_combined_S_matrix(
  data_train::DataFrame,
  data_val::DataFrame,
  base::Vector,
  inflections::Vector;
  ncol=200::Int64,
  sd_base_mean=1::Int64,
  sd_inflection_mean=1::Int64,
  sd_base=4::Int64,
  sd_inflection=4::Int64,
  seed=314::Int64,
  isdeep=true::Bool,
  add_noise=true::Bool,
  sd_noise=1::Int64,
  normalized=false::Bool
  )::Tuple{Matrix, Matrix}

  L = make_combined_L_matrix(
    data_train,
    data_val,
    base,
    inflections;
    ncol=ncol,
    sd_base_mean=sd_base_mean,
    sd_base=sd_base,
    seed=seed,
    isdeep=isdeep
    )

  make_S_matrix(
    data_train,
    data_val,
    base,
    inflections,
    L;
    add_noise=add_noise,
    sd_noise=sd_noise,
    normalized=normalized
    )
end

"""
    L_Matrix_Struct(L, sd_base, sd_base_mean, sd_inflection, sd_inflection_mean, base_f, infl_f, base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)

Construct L_Matrix_Struct with deep mode.
"""
function L_Matrix_Struct(L, sd_base, sd_base_mean, 
  sd_inflection, sd_inflection_mean, base_f, infl_f, 
  base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)
  comp_f_M!(L, sd_base, sd_base_mean, n_base_f, ncol, 0)
  comp_f_M!(L, sd_inflection, sd_inflection_mean, n_infl_f, ncol, n_base_f)

  L_Matrix_Struct(
    L,
    merge_f2i(base_f2i, infl_f2i, n_base_f, n_infl_f),
    vcat(base_f, infl_f),
    ncol)
end

"""
    L_Matrix_Struct(L, sd_base, sd_inflection, base_f, infl_f, base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)

Construct L_Matrix_Struct without deep mode.
"""
function L_Matrix_Struct(L, sd_base, sd_inflection, base_f, infl_f, 
  base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)
  comp_f_M!(L, sd_base, n_base_f, ncol, 0)
  comp_f_M!(L, sd_inflection, n_infl_f, ncol, n_base_f)

  L_Matrix_Struct(
    L,
    merge_f2i(base_f2i, infl_f2i, n_base_f, n_infl_f),
    vcat(base_f, infl_f),
    ncol)
end

"""
    process_features(data, feature_cols)

Collect all features given datasets and feature column names.
"""
function process_features(data, feature_cols)
  features = [f for fc in feature_cols for f in skipmissing(unique(data[:,fc]))]
  base_f2i = Dict(v=>i for (i,v) in enumerate(features))

  features, base_f2i
end

"""
    comp_f_M!(L, sd, sd_mean, n_f, ncol, n_b)

Compose feature Matrix with deep mode.
"""
function comp_f_M!(L, sd, sd_mean, n_f, ncol, n_b)
  means = rand(Normal(0, sd_mean), n_f)
  for i in 1:n_f
    L[i+n_b,:] = rand(Normal(means[i], sd), ncol)
  end
end

"""
    comp_f_M!(L, sd, n_f, ncol, n_b)

Compose feature Matrix without deep mode.
"""
function comp_f_M!(L, sd, n_f, ncol, n_b)
  for i in 1:n_f
    L[i+n_b,:] = rand(Normal(0, sd), ncol)
  end
end

"""
    merge_f2i(base_f2i, infl_f2i, n_base_f, n_infl_f)

Merge base f2i dictionary and inflectional f2i dictionary.
"""
function merge_f2i(base_f2i, infl_f2i, n_base_f, n_infl_f)
  # add infl_f2i into base_f2i
  # infl_f2i need shift n_base_f
  f2i = copy(base_f2i)
  for (k,v) in infl_f2i
    f2i[k] = v+n_base_f
  end
  f2i
end