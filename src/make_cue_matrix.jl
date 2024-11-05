struct Cue_Matrix_Struct
    C::Union{Matrix, SparseMatrixCSC}
    f2i::Dict
    i2f::Dict
    gold_ind::Vector{Vector{Int64}}
    A::Union{SparseMatrixCSC, Nothing} 
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
"""

function make_cue_matrix(
    data;
    grams = [3],  
    target_col = :Words,
    tokenized = false,
    sep_token = nothing,
    keep_sep = false,
    start_end_token = "#",
    verbose = false,
)

    # Tokenize
    if tokenized && !isnothing(sep_token)
        tokens = split.(data[:, target_col], sep_token)
    else
        tokens = split.(data[:, target_col], "")
    end
    tokens = map(x -> map(string, x), tokens) 

    ngrams_results = []

    for i in 1:length(tokens)
        feat_buf = []
        for g in grams
            ngrams_x = make_ngrams(tokens[i], g, keep_sep, sep_token, start_end_token)
            feat_buf = vcat(feat_buf, ngrams_x)
        end
        push!(ngrams_results, feat_buf)
    end

    ngrams_features = unique(vcat(ngrams_results...))
    f2i = Dict(v => i for (i, v) in enumerate(ngrams_features))
    i2f = Dict(i => v for (i, v) in enumerate(ngrams_features))

    n_f = sum(length.(ngrams_results))
    m = size(data, 1)
    n = length(ngrams_features)
    I = zeros(Int64, n_f)
    J = zeros(Int64, n_f)
    V = ones(Int64, n_f)

    A = length(grams) > 1 ? nothing : [Int64[] for _ in 1:length(ngrams_features)]

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
                if A !== nothing 
                    push!(A[last], fi)
                end
                last = fi
            end
        end
    end    

    cue = sparse(I, J, V, m, n, *)

    ngrams_ind = [[f2i[x] for x in y] for y in ngrams_results]

    verbose && println("making adjacency matrix...")

    if A !== nothing
        A = [sort(unique(i)) for i in A]
        n_adj = sum(length.(A))

        I = zeros(Int64, n_adj)
        J = zeros(Int64, n_adj)
        V = ones(Int64, n_adj)

        cnt = 0
        for (i, v) in enumerate(A)
            for j in v
                cnt += 1
                I[cnt] = i
                J[cnt] = j
            end
        end

        A = sparse(I, J, V, length(f2i), length(f2i))
    end

    return Cue_Matrix_Struct(cue, f2i, i2f, ngrams_ind, A, grams, target_col, tokenized, sep_token, keep_sep, start_end_token)
end

"""
    make_cue_matrix(data::DataFrame, cue_obj::Cue_Matrix_Struct)

Make the cue matrix for validation datasets and corresponding indices as well as the adjacency matrix
and gold paths given a dataset in a form of dataframe.


"""

function make_cue_matrix(
    data::DataFrame,
    cue_obj::Cue_Matrix_Struct;
    grams = [3],
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

    # Ensure each element in tokens is of String type
    tokens = map(x -> map(string, x), tokens)

    ngrams_results = [] 

    for i in 1:length(tokens)
        feat_buf = []
        for g in grams
            ngrams_x = make_ngrams(tokens[i], g, keep_sep, sep_token, start_end_token)
            feat_buf = vcat(feat_buf, ngrams_x)
        end
        push!(ngrams_results, feat_buf)
    end
    

    f2i = cue_obj.f2i
    i2f = cue_obj.i2f

    n_f = sum(length.(ngrams_results))
    
    m = size(data, 1)
    n = length(f2i)
    I = zeros(Int64, n_f)   
    J = zeros(Int64, n_f)
    V = ones(Int64, n_f)

    cnt = 0
    for (i, v) in enumerate(ngrams_results)
        for (j, f) in enumerate(v)
            cnt += 1
            I[cnt] = i
            J[cnt] = f2i[f]
        end
    end

    cue = sparse(I, J, V, m, n, *)
    ngrams_ind = [[f2i[x] for x in y] for y in ngrams_results]

    Cue_Matrix_Struct(cue, f2i, i2f, ngrams_ind, cue_obj.A, grams, target_col,
        tokenized, sep_token, keep_sep, start_end_token)
end


"""
    make_cue_matrix(data_train::DataFrame, data_val::DataFrame)

Make the cue matrix for traiing and validation datasets at the same time.

"""
function make_cue_matrix(
    data_train::DataFrame,
    data_val::DataFrame;
    grams = [3],
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


function make_combined_cue_matrix(
    data_train,
    data_val;
    grams = [3],  
    target_col = :Words,
    tokenized = false,
    sep_token = nothing,
    keep_sep = false,
    start_end_token = "#",
    verbose = false,
)

    data_combined = copy(data_train)
    data_val = copy(data_val)

    for col in names(data_combined)
        data_combined[!, col] = inlinestring2string.(data_combined[!,col])
        data_val[!, col] = inlinestring2string.(data_val[!,col])
    end

    append!(data_combined, data_val, promote=true)

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
    make_ngrams(tokens, grams, keep_sep, sep_token, start_end_token)

Given a list of string tokens return a list of all n-grams for these tokens.
"""
function make_ngrams(
    tokens,
    grams,
    keep_sep,
    sep_token,
    start_end_token,
)

    tokens = collect(map(string, tokens))  
    new_tokens = push!(pushfirst!(tokens, start_end_token), start_end_token)
    
    if keep_sep
        # collect ngrams
        ngrams =
            join.(
                collect(zip((Iterators.drop(new_tokens, k) for k = 0:grams-1)...)),
                sep_token,
            )
    else
        ngrams =
            join.(
                collect(zip((Iterators.drop(new_tokens, k) for k = 0:grams-1)...)),
                "",
            )
    end 
    ngrams
end

 