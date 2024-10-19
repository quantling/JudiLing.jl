"""
A structure that stores information created by make_cue_matrix:
C is the cue matrix;
f2i is a dictionary returning the indices for features;
i2f is a dictionary returning the features for indices;
gold_ind is a list of indices of gold paths;
A is the adjacency matrix;
grams is the number of grams for cues;
target_col is the column name for target strings;
tokenized is whether the dataset target is tokenized;
sep_token is the separator;
keep_sep is whether to keep separators in cues;
start_end_token is the start and end token in boundary cues.
"""
struct Cue_Matrix_Struct
    C::Union{Matrix, SparseMatrixCSC}
    f2i::Dict
    i2f::Dict
    gold_ind::Vector{Vector{Int64}}
    A::SparseMatrixCSC
    grams::Union{Vector{Int64}, Int64}
    target_col::Union{Symbol, String}
    tokenized::Bool
    sep_token::Union{String, Nothing}
    keep_sep::Bool
    start_end_token::String
end

"""
Construct cue matrix.
"""
function make_cue_matrix end

"""
Given a list of string tokens, extract their n-grams.
"""
function make_ngrams end

"""
    make_cue_matrix(data::DataFrame)

Make the cue matrix for training datasets and corresponding indices as well as the adjacency matrix
and gold paths given a dataset in a form of dataframe.

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
"""
function make_cue_matrix(
    data;
    grams = [3],  # This is an array containing multiple values
    target_col = :Words,
    tokenized = false,
    sep_token = nothing,
    keep_sep = false,
    start_end_token = "#",
    verbose = false,
)

    # Process tokens
    if tokenized && !isnothing(sep_token)
        tokens = split.(data[:, target_col], sep_token)
    else
        tokens = split.(data[:, target_col], "")
    end

    # Ensure each element in tokens is of String type
    tokens = map(x -> map(string, x), tokens)

    ngrams_results = []  # Store results of all n-grams

    # Generate n-grams for each gram
    for g in grams
        for i in 1:length(tokens)
            push!(ngrams_results, make_ngrams(tokens[i], g, keep_sep, sep_token, start_end_token))
        end
    end

    # Find all unique n-gram features
    ngrams_features = unique(vcat(ngrams_results...))


    f2i = Dict(v => i for (i, v) in enumerate(ngrams_features))
    i2f = Dict(i => v for (i, v) in enumerate(ngrams_features))

    n = length(ngrams_features)
    
    n_f = sum(length.(ngrams_results)) 
    
    # Define m based on the length of grams
    if length(grams) == 1
        m = size(data, 1)  # When there's only one n-gram, m equals the number of rows in the data
    else
        m = size(data, 1) * length(grams)  # If grams contains multiple n-grams, m is the row count multiplied
    end

    # Initialize I, J, V
    I = zeros(Int64, n_f)  # Initialize I as a vector of length n_f
    J = zeros(Int64, n_f)
    V = ones(Int64, n_f)

    A = [Int64[] for _ in 1:length(ngrams_features)]

    cnt = 0 
    for (i, v) in enumerate(ngrams_results)
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

    cue = sparse(I, J, V, m, n)
    ngrams_ind = [[f2i[x] for x in y] for y in ngrams_results]

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

    return Cue_Matrix_Struct(cue, f2i, i2f, ngrams_ind, A, grams, target_col, tokenized, sep_token, keep_sep, start_end_token)
end



"""
    make_ngrams(tokens, grams, keep_sep, sep_token, start_end_token)

Given a list of string tokens return a list of all n-grams for these tokens.
"""
function make_ngrams(tokens, g, keep_sep, sep_token, start_end_token)
    ngrams = []

    tokens = collect(map(string, tokens))  # Ensure tokens are Strings
    new_tokens = push!(pushfirst!(copy(tokens), start_end_token), start_end_token)

    if keep_sep
        ngrams = join.(collect(zip((Iterators.drop(new_tokens, k) for k in 0:g-1)...)), sep_token)
    else
        ngrams = join.(collect(zip((Iterators.drop(new_tokens, k) for k in 0:g-1)...)), "")
    end

    return ngrams
end
