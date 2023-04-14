"""
A structure that stores the discrete semantic vectors:
pS is the discrete semantic matrix;
f2i is a dictionary returning the indices for features;
i2f is a dictionary returning the features for indices.
"""
struct PS_Matrix_Struct
    pS::Union{Matrix,SparseMatrixCSC}
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
    make_pS_matrix(utterances)

Create a discrete semantic matrix given a dataframe.

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
"""
function make_pS_matrix(
    utterances;
    features_col = :CommunicativeIntention,
    sep_token = "_",
)

    # find out all possible features in this dataset
    features = unique(vcat(split.(utterances[:, features_col], sep_token)...))

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
    make_pS_matrix(utterances, utterances_train)

Construct discrete semantic matrix for the validation datasets given by the
exemplar in the dataframe, and given the S matrix for the training datasets.

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
"""
function make_pS_matrix(
    utterances,
    utterances_train;
    features_col = :CommunicativeIntention,
    sep_token = "_",
)

    # find out all possible features in this dataset
    features = unique(vcat(split.(utterances[:, features_col], sep_token)...))

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
    make_S_matrix(data::DataFrame, base::Vector, inflections::Vector)

Create simulated semantic matrix for the training datasets, given the input
data of a vector specified contex lexemes and a vector specified gramatic
lexemes. The semantic vector of a word form is constructed summing semantic
vectors of content and gramatic lexemes.

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
"""
function make_S_matrix(
    data::DataFrame,
    base::Vector,
    inflections::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_inflection_mean = 1,
    sd_base = 4,
    sd_inflection = 4,
    seed = 314,
    isdeep = true,
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    L = make_L_matrix(
        data,
        base,
        inflections,
        sd_base_mean = sd_base_mean,
        sd_inflection_mean = sd_inflection_mean,
        sd_base = sd_base,
        sd_inflection = sd_inflection,
        ncol = ncol,
        seed = seed,
        isdeep = isdeep,
    )

    make_S_matrix(
        data,
        base,
        inflections,
        L,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )
end

"""
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector)

Create simulated semantic matrix for the validation datasets, given the input
data of a vector specified contex lexemes and a vector specified gramatic
lexemes. The semantic vector of a word form is constructed summing semantic
vectors of content and gramatic lexemes.

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
"""
function make_S_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    base::Vector,
    inflections::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_inflection_mean = 1,
    sd_base = 4,
    sd_inflection = 4,
    seed = 314,
    isdeep = true,
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    L = make_L_matrix(
        data_train,
        base,
        inflections,
        sd_base_mean = sd_base_mean,
        sd_inflection_mean = sd_inflection_mean,
        sd_base = sd_base,
        sd_inflection = sd_inflection,
        ncol = ncol,
        seed = seed,
        isdeep = isdeep,
    )

    make_S_matrix(
        data_train,
        data_val,
        base,
        inflections,
        L,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )
end

"""
    make_S_matrix(data::DataFrame, base::Vector)

Create simulated semantic matrix for the training datasets with only base
features, given the input data of a vector specified contex lexemes and a
vector specified gramatic lexemes. The semantic vector of a word form is
constructed summing semantic vectors of content and gramatic lexemes.

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
"""
function make_S_matrix(
    data::DataFrame,
    base::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_base = 4,
    seed = 314,
    isdeep = true,
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    L = make_L_matrix(
        data,
        base,
        sd_base_mean = sd_base_mean,
        sd_base = sd_base,
        ncol = ncol,
        seed = seed,
        isdeep = isdeep,
    )

    make_S_matrix(
        data,
        base,
        L,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )
end

"""
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector)

Create simulated semantic matrix for the validation datasets with only base
features, given the input data of a vector specified contex lexemes and a
vector specified gramatic lexemes. The semantic vector of a word form is
constructed summing semantic vectors of content and gramatic lexemes.

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
"""
function make_S_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    base::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_base = 4,
    seed = 314,
    isdeep = true,
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    L = make_L_matrix(
        data_train,
        base,
        sd_base_mean = sd_base_mean,
        sd_base = sd_base,
        ncol = ncol,
        seed = seed,
        isdeep = isdeep,
    )

    make_S_matrix(
        data_train,
        data_val,
        base,
        L,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )
end

"""
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, pyndl_weights::Pyndl_Weight_Struct, n_features_columns::Vector)

Create semantic matrix for pyndl mode
"""
function make_S_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    pyndl_weights::Pyndl_Weight_Struct,
    n_features_columns::Vector,
)

    f2i = Dict(v => i for (i, v) in enumerate(pyndl_weights.outcomes))
    i2f = Dict(i => v for (i, v) in enumerate(pyndl_weights.outcomes))

    n_f = length(pyndl_weights.outcomes)

    St_train = zeros(Float64, n_f, size(data_train, 1))
    for i = 1:size(data_train, 1)
        for f in data_train[i, n_features_columns]
            St_train[f2i[f], i] = 1
        end
    end

    St_val = zeros(Float64, n_f, size(data_val, 1))
    for i = 1:size(data_val, 1)
        for f in data_val[i, n_features_columns]
            St_val[f2i[f], i] = 1
        end
    end

    St_train', St_val'
end

"""
    make_S_matrix(data_train::DataFrame, base::Vector, inflections::Vector, L::L_Matrix_Struct)

Create simulated semantic matrix where lexome matrix is available.

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
"""
function make_S_matrix(
    data_train::DataFrame,
    base::Vector,
    inflections::Vector,
    L::L_Matrix_Struct;
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    n_train = size(data_train, 1)
    n_base = length(base)
    n_infl = length(inflections)

    St_train = make_St(L, n_train, data_train, base, inflections)
    add_noise && add_St_noise!(St_train, sd_noise)
    normalized && normalize_St!(St_train, n_base, n_infl)

    Array(St_train')
end

"""
    make_S_matrix(data_train::DataFrame, data_val::Union{DataFrame, Nothing}, base::Vector, L::L_Matrix_Struct)

Create simulated semantic matrix where lexome matrix is available.

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
"""
function make_S_matrix(
    data_train::DataFrame,
    data_val::Union{DataFrame,Nothing},
    base::Vector,
    L::L_Matrix_Struct;
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    JudiLing.make_S_matrix(
        data_train,
        data_val,
        base,
        [],
        L,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )

end

"""
    make_S_matrix(data::DataFrame, base::Vector, L::L_Matrix_Struct)

Create simulated semantic matrix where lexome matrix is available.

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
"""
function make_S_matrix(
    data::DataFrame,
    base::Vector,
    L::L_Matrix_Struct;
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    JudiLing.make_S_matrix(
        data,
        base,
        [],
        L,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )

end

"""
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector, L::L_Matrix_Struct)

Create simulated semantic matrix where lexome matrix is available.

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
"""
function make_S_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    base::Vector,
    inflections::Vector,
    L::L_Matrix_Struct;
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    n_train = size(data_train, 1)
    n_val = size(data_val, 1)
    n_base = length(base)
    n_infl = length(inflections)

    St_train = make_St(L, n_train, data_train, base, inflections)
    add_noise && add_St_noise!(St_train, sd_noise)
    normalized && normalize_St!(St_train, n_base, n_infl)

    St_val = make_St(L, n_val, data_val, base, inflections)
    add_noise && add_St_noise!(St_val, sd_noise)
    normalized && normalize_St!(St_val, n_base, n_infl)

    Array(St_train'), Array(St_val')
end



"""
    make_L_matrix(data::DataFrame, base::Vector, inflections::Vector)

Create Lexome Matrix with simulated semantic vectors.

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
"""
function make_L_matrix(
    data::DataFrame,
    base::Vector,
    inflections::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_inflection_mean = 1,
    sd_base = 4,
    sd_inflection = 4,
    seed = 314,
    isdeep = true,
)

    # collect all features and f2i mappings
    base_f, base_f2i = process_features(data, base)
    infl_f, infl_f2i = process_features(data, inflections)

    # setup seed
    Random.seed!(seed)

    n_base_f = length(base_f)
    n_infl_f = length(infl_f)

    # pre-allocated L matrix
    L = Matrix{Float64}(undef, (n_base_f + n_infl_f, ncol))

    if isdeep # deep mode random means for each feature
        return L_Matrix_Struct(
            L,
            sd_base,
            sd_base_mean,
            sd_inflection,
            sd_inflection_mean,
            base_f,
            infl_f,
            base_f2i,
            infl_f2i,
            n_base_f,
            n_infl_f,
            ncol,
        )
    else # otherwise use mean=0 for all features
        return L_Matrix_Struct(
            L,
            sd_base,
            sd_inflection,
            base_f,
            infl_f,
            base_f2i,
            infl_f2i,
            n_base_f,
            n_infl_f,
            ncol,
        )
    end
end

"""
    make_L_matrix(data::DataFrame, base::Vector)

Create Lexome Matrix with simulated semantic vectors where there are only base features.

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
"""
function make_L_matrix(
    data::DataFrame,
    base::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_base = 4,
    seed = 314,
    isdeep = true,
)

    make_L_matrix(
        data,
        base,
        [];
        ncol = ncol,
        sd_base_mean = sd_base_mean,
        sd_base = sd_base,
        seed = seed,
        isdeep = isdeep,
    )
end

"""
    make_combined_L_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector)

Create Lexome Matrix with simulated semantic vectors, where features are
combined from both training datasets and validation datasets.

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
"""
function make_combined_L_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    base::Vector,
    inflections::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_inflection_mean = 1,
    sd_base = 4,
    sd_inflection = 4,
    seed = 314,
    isdeep = true,
)

    data_combined = copy(data_train)
    append!(data_combined, data_val)

    make_L_matrix(
        data_combined,
        base,
        inflections;
        ncol = ncol,
        sd_base_mean = sd_base_mean,
        sd_base = sd_base,
        seed = seed,
        isdeep = isdeep,
    )

end

"""
    make_combined_L_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector)

Create Lexome Matrix with simulated semantic vectors, where features are
combined from both training datasets and validation datasets.

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
"""
function make_combined_L_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    base::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_inflection_mean = 1,
    sd_base = 4,
    sd_inflection = 4,
    seed = 314,
    isdeep = true,
)

    make_combined_L_matrix(
        data_train,
        data_val,
        base,
        [];
        ncol = ncol,
        sd_base_mean = sd_base_mean,
        sd_base = sd_base,
        seed = seed,
        isdeep = isdeep,
    )
end

"""
    make_combined_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector, L::L_Matrix_Struct)

Create simulated semantic matrix for the training datasets and validation
datasets with existing Lexome matrix, where features are combined from both
training datasets and validation datasets.

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
"""
function make_combined_S_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    base::Vector,
    inflections::Vector,
    L::L_Matrix_Struct;
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    make_S_matrix(
        data_train,
        data_val,
        base,
        inflections,
        L;
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )
end

"""
    make_combined_S_matrix(data_train::DataFrame, data_val::Union{DataFrame, Nothing}, base::Vector, L::L_Matrix_Struct)

Create simulated semantic matrix for the training datasets and validation
datasets with existing Lexome matrix, where features are combined from both
training datasets and validation datasets.

# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `base::Vector`: context lexemes
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
"""
function make_combined_S_matrix(
    data_train::DataFrame,
    data_val::Union{DataFrame,Nothing},
    base::Vector,
    L::L_Matrix_Struct;
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    make_S_matrix(
        data_train,
        data_val,
        base,
        L;
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )
end

"""
    make_combined_S_matrix(  data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector)

Create simulated semantic matrix for the training datasets and validation
datasets, where features are combined from both training datasets and
validation datasets.

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
"""
function make_combined_S_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    base::Vector,
    inflections::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_inflection_mean = 1,
    sd_base = 4,
    sd_inflection = 4,
    seed = 314,
    isdeep = true,
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    L = make_combined_L_matrix(
        data_train,
        data_val,
        base,
        inflections;
        ncol = ncol,
        sd_base_mean = sd_base_mean,
        sd_base = sd_base,
        seed = seed,
        isdeep = isdeep,
    )

    make_S_matrix(
        data_train,
        data_val,
        base,
        inflections,
        L;
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )
end

"""
    make_combined_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector)

Create simulated semantic matrix for the training datasets and validation
datasets, where features are combined from both training datasets and
validation datasets.

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
"""
function make_combined_S_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    base::Vector;
    ncol = 200,
    sd_base_mean = 1,
    sd_inflection_mean = 1,
    sd_base = 4,
    sd_inflection = 4,
    seed = 314,
    isdeep = true,
    add_noise = true,
    sd_noise = 1,
    normalized = false,
)

    L = make_combined_L_matrix(
        data_train,
        data_val,
        base,
        ncol = ncol,
        sd_base_mean = sd_base_mean,
        sd_base = sd_base,
        seed = seed,
        isdeep = isdeep,
    )

    make_S_matrix(
        data_train,
        data_val,
        base,
        L,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized,
    )
end

"""
    load_S_matrix_from_fasttext(data::DataFrame,
                                language::Symbol;
                                target_col=:Word,
                                default_file::Int=1)

Load semantic matrix from fasttext, loaded using the Embeddings.jl package.
Subset fasttext vectors to include only words in `target_col` of `data`, and
subset data to only include words in `target_col` for which semantic vector
is available.

- `default_file=1` loads from https://fasttext.cc/docs/en/crawl-vectors.html,
  paper: E. Grave*, P. Bojanowski*, P. Gupta, A. Joulin, T. Mikolov,
         Learning Word Vectors for 157 Languages
  License: CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/
- `default_file=2` loads from https://fasttext.cc/docs/en/pretrained-vectors.html
  paper: P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov,
         Enriching Word Vectors with Subword Information
  License: CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/

# Obligatory Arguments
- `data::DataFrame`: the dataset
- `language::Symbol`: the language of the words in the dataset,
    offically ISO 639-2 (see https://github.com/JuliaText/Embeddings.jl/issues/34#issuecomment-782604523)
    but practically it seems more like ISO 639-1 to me with ISO 639-2 only being used
    if ISO 639-1 isn't available (see https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes)

# Optional Arguments
- `target_col=:Word`: column with orthographic representation of words in `data`
- `default_file::Int=1`: source of vectors, for more information see above and here: https://github.com/JuliaText/Embeddings.jl#loading-different-embeddings

# Examples
```julia
# basic usage
latin_small, S = JudiLing.load_S_matrix_from_fasttext(latin, :la, target_col=:Word)
```
"""
function load_S_matrix_from_fasttext(data::DataFrame,
                                     language::Symbol;
                                     target_col=:Word,
                                     default_file::Int=1)

    embtable = load_embeddings(FastText_Text{language}, default_file,
                                     keep_words=Set(data[!, target_col]));

    # code from https://github.com/JuliaText/Embeddings.jl#basic-example
    get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

     function get_embedding(word)
         ind = get_word_index[word]
         emb = embtable.embeddings[:,ind]
         return emb
     end

     function create_S(words)
     	S = zeros(length(words), size(get_embedding(embtable.vocab[1]),1))
     	for i in 1:size(S, 1)
     		S[i,:] = get_embedding(words[i])
     	end
     	S
     end

     data_small = filter(row -> haskey(get_word_index, row[target_col]), data)

     S_ft = create_S(data_small[!,target_col])

     return data_small, S_ft
end

"""
    load_S_matrix_from_fasttext(data_train::DataFrame,
                                data_val::DataFrame,
                                language::Symbol;
                                target_col=:Word,
                                default_file::Int=1)

Load semantic matrix from fasttext, loaded using the Embeddings.jl package.
Subset fasttext vectors to include only words in `target_col` of `data_train` and `data_val`, and
subset data to only include words in `target_col` for which semantic vector
is available.
Returns subsetted train and val data and train and val semantic matrices.

- `default_file=1` loads from https://fasttext.cc/docs/en/crawl-vectors.html,
  paper: E. Grave*, P. Bojanowski*, P. Gupta, A. Joulin, T. Mikolov,
         Learning Word Vectors for 157 Languages
  License: CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/
- `default_file=2` loads from https://fasttext.cc/docs/en/pretrained-vectors.html
  paper: P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov,
         Enriching Word Vectors with Subword Information
  License: CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/

# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `language::Symbol`: the language of the words in the dataset,
    offically ISO 639-2 (see https://github.com/JuliaText/Embeddings.jl/issues/34#issuecomment-782604523)
    but practically it seems more like ISO 639-1 to me with ISO 639-2 only being used
    if ISO 639-1 isn't available (see https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes)

# Optional Arguments
- `target_col=:Word`: column with orthographic representation of words in `data`
- `default_file::Int=1`: source of vectors, for more information see above and here: https://github.com/JuliaText/Embeddings.jl#loading-different-embeddings

# Examples
```julia
# basic usage
latin_small_train, latin_small_val, S_train, S_val = JudiLing.load_S_matrix_from_fasttext(latin_train,
                                                      latin_val,
                                                      :la,
                                                      target_col=:Word)
```
"""
function load_S_matrix_from_fasttext(data_train::DataFrame,
                                     data_val::DataFrame,
                                     language::Symbol;
                                     target_col=:Word,
                                     default_file::Int=1)

    data_combined = copy(data_train)
    append!(data_combined, data_val)

    embtable = load_embeddings(FastText_Text{language}, default_file,
                                     keep_words=Set(data_combined[!, target_col]));

    # code from https://github.com/JuliaText/Embeddings.jl#basic-example
    get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

     function get_embedding(word)
         ind = get_word_index[word]
         emb = embtable.embeddings[:,ind]
         return emb
     end

     function create_S(words)
     	S = zeros(length(words), size(get_embedding(embtable.vocab[1]),1))
     	for i in 1:size(S, 1)
     		S[i,:] = get_embedding(words[i])
     	end
     	S
     end

     data_train_small = filter(row -> haskey(get_word_index,
                                             row[target_col]),
                                             data_train)

     data_val_small = filter(row -> haskey(get_word_index,
                                             row[target_col]),
                                             data_val)

     S_ft_train = create_S(data_train_small[!,target_col])
     S_ft_val = create_S(data_val_small[!,target_col])

     return data_train_small, data_val_small, S_ft_train, S_ft_val
end

"""
    L_Matrix_Struct(L, sd_base, sd_base_mean, sd_inflection, sd_inflection_mean, base_f, infl_f, base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)

Construct L_Matrix_Struct with deep mode.
"""
function L_Matrix_Struct(
    L,
    sd_base,
    sd_base_mean,
    sd_inflection,
    sd_inflection_mean,
    base_f,
    infl_f,
    base_f2i,
    infl_f2i,
    n_base_f,
    n_infl_f,
    ncol,
)
    comp_f_M!(L, sd_base, sd_base_mean, n_base_f, ncol, 0)
    comp_f_M!(L, sd_inflection, sd_inflection_mean, n_infl_f, ncol, n_base_f)

    L_Matrix_Struct(
        L,
        merge_f2i(base_f2i, infl_f2i, n_base_f, n_infl_f),
        vcat(base_f, infl_f),
        ncol,
    )
end

"""
    L_Matrix_Struct(L, sd_base, sd_inflection, base_f, infl_f, base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)

Construct L_Matrix_Struct without deep mode.
"""
function L_Matrix_Struct(
    L,
    sd_base,
    sd_inflection,
    base_f,
    infl_f,
    base_f2i,
    infl_f2i,
    n_base_f,
    n_infl_f,
    ncol,
)
    comp_f_M!(L, sd_base, n_base_f, ncol, 0)
    comp_f_M!(L, sd_inflection, n_infl_f, ncol, n_base_f)

    L_Matrix_Struct(
        L,
        merge_f2i(base_f2i, infl_f2i, n_base_f, n_infl_f),
        vcat(base_f, infl_f),
        ncol,
    )
end

"""
    process_features(data, feature_cols)

Collect all features given datasets and feature column names.
"""
function process_features(data, feature_cols)
    features =
        [f for fc in feature_cols for f in skipmissing(unique(data[:, fc]))]
    base_f2i = Dict(v => i for (i, v) in enumerate(features))

    features, base_f2i
end

"""
    comp_f_M!(L, sd, sd_mean, n_f, ncol, n_b)

Compose feature Matrix with deep mode.
"""
function comp_f_M!(L, sd, sd_mean, n_f, ncol, n_b)
    means = rand(Normal(0, sd_mean), n_f)
    for i = 1:n_f
        L[i+n_b, :] = rand(Normal(means[i], sd), ncol)
    end
end

"""
    comp_f_M!(L, sd, n_f, ncol, n_b)

Compose feature Matrix without deep mode.
"""
function comp_f_M!(L, sd, n_f, ncol, n_b)
    for i = 1:n_f
        L[i+n_b, :] = rand(Normal(0, sd), ncol)
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
    for (k, v) in infl_f2i
        f2i[k] = v + n_base_f
    end
    f2i
end

"""
    lexome_sum(L, features)

Sum up semantic vector, given lexome vector.
"""
function lexome_sum(L, features)
    ls = [L.L[L.f2i[f], :] for f in skipmissing(features)]
    if length(ls) > 0
        return sum(ls)
    end
    zeros(Float64, L.ncol)
end

"""
    make_St(L, n, data, base, inflections)

Make S transpose matrix with inflections.
"""
function make_St(L, n, data, base, inflections)
    St = Array{Float64,2}(undef, L.ncol, n)
    for i = 1:n
        ls_base = lexome_sum(L, data[i, base])
        ls_infl = lexome_sum(L, data[i, inflections])
        ls = ls_base + ls_infl
        St[:, i] = ls
    end
    St
end

"""
    make_St(L, n, data, base)

Make S transpose matrix without inflections.
"""
function make_St(L, n, data, base)
    St = Array{Float64,2}(undef, L.ncol, n)
    for i = 1:n
        ls_base = lexome_sum(L, data[i, base])
        St[:, i] = ls_base
    end
    St
end

"""
    add_St_noise!(St, sd_noise)

Add noise.
"""
function add_St_noise!(St, sd_noise)
    Noise = rand(Normal(0, sd_noise), size(St))
    broadcast!(+, St, St, Noise)
end

"""
    normalize_St!(St, n_base, n_infl)

Normalize S transpose with inflections.
"""
function normalize_St!(St, n_base, n_infl)
    broadcast!(/, St, St, n_base + n_infl)
end

"""
    normalize_St!(St, n_base)

Normalize S transpose without inflections.
"""
function normalize_St!(St, n_base)
    St = St ./ n_base
end
