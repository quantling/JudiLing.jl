"""
A structure that stores information created by make_cue_matrix:
C is the cue matrix;
f2i is a dictionary returning the indices for features;
i2f is a dictionary returning the features for indices;
gold_ind is a list of indices of gold paths;
A is the adjacency matrix.
"""
struct Cue_Matrix_Struct
    C::Union{Matrix,SparseMatrixCSC}
    f2i::Dict
    i2f::Dict
    gold_ind::Vector{Vector{Int64}}
    A::SparseMatrixCSC
end

"""
Construct cue matrix.
"""
function make_cue_matrix end

"""
Construct cue matrix where combined features and adjacencies for both training datasets and validation datasets.
"""
function make_combined_cue_matrix end

"""
Given a list of string tokens, extract their n-grams.
"""
function make_ngrams end

"""
    make_cue_matrix(data::DataFrame)

Make the cue matrix for training datasets and corresponding indices as well as the adjacency matrix
and gold paths given a dataset in a form of dataframe.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset

# Optional Arguments
- `grams::Int64=3`: the number of grams for cues
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `tokenized::Bool=false`:if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `keep_sep::Bool=false`: if true, keep separators in cues
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# make cue matrix without tokenization
cue_obj_train = JudiLing.make_cue_matrix(
     latin_train,
    grams=3,
    target_col=:Word,
    tokenized=false,
    sep_token="-",
    start_end_token="#",
    keep_sep=false,
    verbose=false
    )

# make cue matrix with tokenization
cue_obj_train = JudiLing.make_cue_matrix(
    french_train,
    grams=3,
    target_col=:Syllables,
    tokenized=true,
    sep_token="-",
    start_end_token="#",
    keep_sep=true,
    verbose=false
    )
```
...
"""
function make_cue_matrix(
    data::DataFrame;
    grams = 3,
    target_col = :Words,
    tokenized = false,
    sep_token = nothing,
    keep_sep = false,
    start_end_token = "#",
    verbose = false,
)

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

    A = [Int64[] for i = 1:length(ngrams_features)]

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
    if verbose
        pb = Progress(length(A))
    end
    for (i, v) in iter
        for j in v
            cnt += 1
            I[cnt] = i
            J[cnt] = j
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end

    A = sparse(I, J, V, length(f2i), length(f2i))

    Cue_Matrix_Struct(cue, f2i, i2f, ngrams_ind, A)
end

"""
    make_cue_matrix(data::DataFrame, cue_obj::Cue_Matrix_Struct)

Make the cue matrix for validation datasets and corresponding indices as well as the adjacency matrix
and gold paths given a dataset in a form of dataframe.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset
- `cue_obj::Cue_Matrix_Struct`: training cue object

# Optional Arguments
- `grams::Int64=3`: the number of grams for cues
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `tokenized::Bool=false`:if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `keep_sep::Bool=false`: if true, keep separators in cues
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# make cue matrix without tokenization
cue_obj_val = JudiLing.make_cue_matrix(
  latin_val,
  cue_obj_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  sep_token="-",
  keep_sep=false,
  start_end_token="#",
  verbose=false
  )

# make cue matrix with tokenization
cue_obj_val = JudiLing.make_cue_matrix(
    french_val,
    cue_obj_train,
    grams=3,
    target_col=:Syllables,
    tokenized=true,
    sep_token="-",
    keep_sep=true,
    start_end_token="#",
    verbose=false
    )
```
...
"""
function make_cue_matrix(
    data::DataFrame,
    cue_obj::Cue_Matrix_Struct;
    grams = 3,
    target_col = "Words",
    tokenized = false,
    sep_token = nothing,
    keep_sep = false,
    start_end_token = "#",
    verbose = false,
)

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
    make_cue_matrix(data_train::DataFrame, data_val::DataFrame)

Make the cue matrix for traiing and validation datasets at the same time.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset

# Optional Arguments
- `grams::Int64=3`: the number of grams for cues
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `tokenized::Bool=false`:if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `keep_sep::Bool=false`: if true, keep separators in cues
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# make cue matrix without tokenization
cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
    latin_train,
    latin_val,
    grams=3,
    target_col=:Word,
    tokenized=false,
    keep_sep=false
    )

# make cue matrix with tokenization
cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
    french_train,
    french_val,
    grams=3,
    target_col=:Syllables,
    tokenized=true,
    sep_token="-",
    keep_sep=true,
    start_end_token="#",
    verbose=false
    )
```
...
"""
function make_cue_matrix(
    data_train::DataFrame,
    data_val::DataFrame;
    grams = 3,
    target_col = "Words",
    tokenized = false,
    sep_token = nothing,
    keep_sep = false,
    start_end_token = "#",
    verbose = false,
)

    cue_obj_train = make_cue_matrix(
        data_train,
        grams = grams,
        target_col = target_col,
        tokenized = tokenized,
        sep_token = sep_token,
        keep_sep = keep_sep,
        start_end_token = start_end_token,
        verbose = verbose,
    )

    cue_obj_val = make_cue_matrix(
        data_val,
        cue_obj_train,
        grams = grams,
        target_col = target_col,
        tokenized = tokenized,
        sep_token = sep_token,
        keep_sep = keep_sep,
        start_end_token = start_end_token,
        verbose = verbose,
    )

    cue_obj_train, cue_obj_val
end

"""
  make_cue_matrix(data::DataFrame, pyndl_weights::Pyndl_Weight_Struct)

Make the cue matrix for pyndl mode.
"""
function make_cue_matrix(
    data::DataFrame,
    pyndl_weights::Pyndl_Weight_Struct;
    grams = 3,
    target_col = "Words",
    tokenized = false,
    sep_token = nothing,
    keep_sep = false,
    start_end_token = "#",
    verbose = false,
)

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

    f2i = Dict(v => i for (i, v) in enumerate(pyndl_weights.cues))
    i2f = Dict(i => v for (i, v) in enumerate(pyndl_weights.cues))

    n_f = sum([length(v) for v in ngrams])

    m = size(data, 1)
    n = length(ngrams_features)
    I = zeros(Int64, n_f)
    J = zeros(Int64, n_f)
    V = ones(Int64, n_f)

    A = [Int64[] for i = 1:length(ngrams_features)]

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
    if verbose
        pb = Progress(length(A))
    end
    for (i, v) in iter
        for j in v
            cnt += 1
            I[cnt] = i
            J[cnt] = j
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end

    A = sparse(I, J, V, length(f2i), length(f2i))

    Cue_Matrix_Struct(cue, f2i, i2f, ngrams_ind, A)
end

"""
    make_combined_cue_matrix(data_train, data_val)

Make the cue matrix for training and validation datasets at the same time, where the features and adjacencies are combined.

...
# Obligatory Arguments
- `data_train::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset

# Optional Arguments
- `grams::Int64=3`: the number of grams for cues
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `tokenized::Bool=false`:if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `keep_sep::Bool=false`: if true, keep separators in cues
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# make cue matrix without tokenization
cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
    latin_train,
    latin_val,
    grams=3,
    target_col=:Word,
    tokenized=false,
    keep_sep=false
    )

# make cue matrix with tokenization
cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
    french_train,
    french_val,
    grams=3,
    target_col=:Syllables,
    tokenized=true,
    sep_token="-",
    keep_sep=true,
    start_end_token="#",
    verbose=false
    )
```
...
"""
function make_combined_cue_matrix(
    data_train,
    data_val;
    grams = 3,
    target_col = "Words",
    tokenized = false,
    sep_token = nothing,
    keep_sep = false,
    start_end_token = "#",
    verbose = false,
)

    data_combined = copy(data_train)
    append!(data_combined, data_val)

    cue_obj_combined = make_cue_matrix(
        data_combined,
        grams = grams,
        target_col = target_col,
        tokenized = tokenized,
        sep_token = sep_token,
        keep_sep = keep_sep,
        start_end_token = start_end_token,
        verbose = verbose,
    )

    cue_obj_train = make_cue_matrix(
        data_train,
        cue_obj_combined,
        grams = grams,
        target_col = target_col,
        tokenized = tokenized,
        sep_token = sep_token,
        keep_sep = keep_sep,
        start_end_token = start_end_token,
        verbose = verbose,
    )

    cue_obj_val = make_cue_matrix(
        data_val,
        cue_obj_combined,
        grams = grams,
        target_col = target_col,
        tokenized = tokenized,
        sep_token = sep_token,
        keep_sep = keep_sep,
        start_end_token = start_end_token,
        verbose = verbose,
    )

    cue_obj_train, cue_obj_val
end

"""
    make_ngrams(tokens, grams=3, keep_sep=false, sep_token=nothing, start_end_token="#")

Given a list of string tokens return a list of all n-grams for these tokens.
"""
function make_ngrams(
    tokens,
    grams = 3,
    keep_sep = false,
    sep_token = nothing,
    start_end_token = "#",
)

    push!(pushfirst!(tokens, start_end_token), start_end_token)
    if keep_sep
        # collect ngrams
        ngrams =
            join.(
                collect(zip((Iterators.drop(tokens, k) for k = 0:grams-1)...)),
                sep_token,
            )
    else
        ngrams =
            join.(
                collect(zip((Iterators.drop(tokens, k) for k = 0:grams-1)...)),
                "",
            )
    end

    ngrams
end
