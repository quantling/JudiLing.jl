"""
Pyndl object.
"""
struct Pyndl_Weight_Struct
    cues::Vector{String}
    outcomes::Vector{String}
    weight::Matrix{Float64}
end

"""
    pyndl(data_path)

Perform pyndl.
"""
function pyndl(
    data_path;
    alpha = 0.1,
    betas = (0.1, 0.1),
    method = "openmp"
)

    ndl = pyimport("pyndl.ndl")

    weights_py = ndl.ndl(
        events = data_path,
        alpha = alpha,
        betas = betas,
        method = method,
        remove_duplicates = true,
    )

    unwrap_xarray(weights_py)

end

function unwrap_xarray(weights)
    coords = weights.coords.to_dataset()
    cues = [i for i in coords.cues.data]
    outcomes = [i for i in coords.outcomes.data]
    weights = weights.data

    Pyndl_Weight_Struct(cues, outcomes, weights)
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
    n = length(pyndl_weights.cues)
    I = zeros(Int64, n_f)
    J = zeros(Int64, n_f)
    V = ones(Int64, n_f)

    A = [Int64[] for i = 1:length(pyndl_weights.cues)]

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

    Cue_Matrix_Struct(cue, f2i, i2f, ngrams_ind, A, grams, target_col,
        tokenized, sep_token, keep_sep, start_end_token)
end

"""
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, pyndl_weights::Pyndl_Weight_Struct, n_features_columns::Vector)

Create semantic matrix for pyndl mode
"""
function make_S_matrix(
    data_train::DataFrame,
    data_val::DataFrame,
    pyndl_weights::Pyndl_Weight_Struct,
    n_features_columns::Vector;
    tokenized=false,
    sep_token="_"
)

    f2i = Dict(v => i for (i, v) in enumerate(pyndl_weights.outcomes))
    i2f = Dict(i => v for (i, v) in enumerate(pyndl_weights.outcomes))

    n_f = length(pyndl_weights.outcomes)

    St_train = zeros(Float64, n_f, size(data_train, 1))
    for i = 1:size(data_train, 1)
        for f in data_train[i, n_features_columns]
            if tokenized
                for f_i in split(f, sep_token)
                    St_train[f2i[f_i], i] = 1
                end
            else
                St_train[f2i[f], i] = 1
            end
        end
    end

    St_val = zeros(Float64, n_f, size(data_val, 1))
    for i = 1:size(data_val, 1)
        for f in data_val[i, n_features_columns]
            if tokenized
                for f_i in split(f, sep_token)
                    St_val[f2i[f_i], i] = 1
                end
            else
                St_val[f2i[f], i] = 1
            end
        end
    end

    St_train', St_val'
end

"""
    make_S_matrix(data::DataFrame, pyndl_weights::Pyndl_Weight_Struct, n_features_columns::Vector)

Create semantic matrix for pyndl mode
"""
function make_S_matrix(
    data::DataFrame,
    pyndl_weights::Pyndl_Weight_Struct,
    n_features_columns::Vector;
    tokenized=false,
    sep_token="_"
)

    f2i = Dict(v => i for (i, v) in enumerate(pyndl_weights.outcomes))
    i2f = Dict(i => v for (i, v) in enumerate(pyndl_weights.outcomes))

    n_f = length(pyndl_weights.outcomes)

    St = zeros(Float64, n_f, size(data, 1))
    for i = 1:size(data, 1)
        for f in data[i, n_features_columns]
            if tokenized
                for f_i in split(f, sep_token)
                    St[f2i[f_i], i] = 1
                end
            else
                St[f2i[f], i] = 1
            end
        end
    end

    St'
end
