"""
Check whether the predictions are correct.
"""
function iscorrect(gold_ind, pred_ind)
    gold_ind == pred_ind
end

"""
Display prediction nicely.
"""
function display_pred(
    preds,
    i2f,
    grams,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
)

    if length(preds) == 0
        println("No prediction for this utterance")
    else
        for i = 1:length(preds)
            println("Predict $i")
            println("support: $(preds[i].support)")
            pred = translate(
                preds[i].ngrams_ind,
                i2f,
                grams,
                tokenized,
                sep_token,
                start_end_token,
                output_sep_token,
            )
            println("prediction: $pred")
        end
    end
end

"""
Translate indices into words or utterances
"""
function translate(
    ngrams_ind,
    i2f,
    grams,
    tokenized,
    sep_token,
    start_end_token,
    output_sep_token,
)

    if isnothing(output_sep_token)
        output_sep_token = ""
    end

    if tokenized && !isnothing(sep_token)
        s1 = join(
            [split(i2f[i], sep_token)[1] for i in ngrams_ind],
            output_sep_token,
        )
        s2 = join(
            split(i2f[ngrams_ind[end]], sep_token)[2:end],
            output_sep_token,
        )
        out = replace(
            s1 * output_sep_token * s2,
            start_end_token * sep_token => "",
        )
        out = replace(out, sep_token * start_end_token => "")
    else
        s1 = join([split(i2f[i], "")[1] for i in ngrams_ind], output_sep_token)
        s2 = join(split(i2f[ngrams_ind[end]], "")[2:end], output_sep_token)
        out = replace(s1 * output_sep_token * s2, start_end_token => "")
    end

    out
end

"""
Append indices together to form a path
"""
function translate_path(ngrams_ind, i2f; sep_token = ":")

    join([i2f[i] for i in ngrams_ind], sep_token)
end

"""
Check whether a matrix is truly sparse regardless its format, where M is originally a sparse matrix format.
"""
function is_truly_sparse(M::SparseMatrixCSC; threshold = 0.05, verbose = false)
    verbose && println("Sparsity: $(length(M.nzval)/M.m/M.n)")
    return threshold > (length(M.nzval) / M.m / M.n)
end

"""
Check whether a matrix is truly sparse regardless its format, where M is originally a dense matrix format.
"""
function is_truly_sparse(M::Matrix; threshold = 0.05, verbose = false)
    M = sparse(M)
    verbose && println("Sparsity: $(length(M.nzval)/M.m/M.n)")
    return threshold > (length(M.nzval) / M.m / M.n)
end

"""
Check whether a gram can attach to another gram.
"""
function isattachable(a, b)

    a[2:end] == b[1:end-1]
end

"""
Check whether a gram can attach to another gram.
"""
function isattachable(a, c, A)

    convert(Bool, A[a[end], c])
end

"""
Check whether a path is complete.
"""
function iscomplete(a, i2f; tokenized = false, sep_token = nothing, start_end_token = "#")

    ngram = i2f[a[end]]

    if tokenized && !isnothing(sep_token)
        last_w = split(ngram, sep_token)[end]
    else
        last_w = split(ngram, "")[end]
    end

    last_w == start_end_token
end

"""
Check whether a gram can start a path.
"""
function isstart(c, i2f; tokenized = false, sep_token = nothing, start_end_token = "#")

    ngram = i2f[c]

    if tokenized && !isnothing(sep_token)
        start_w = split(ngram, sep_token)[1]
    else
        start_w = split(ngram, "")[1]
    end

    start_w == start_end_token
end

"""
Check whether a predicted path is in training data.
"""
function isnovel(gold_ind, pred_ngrams_ind)

    !(pred_ngrams_ind in gold_ind)
end

"""
Check whether there are tokens already used in dataset as n-gram components.
"""
function check_used_token(data, target_col, token, token_name)
    data_columns = data[:, target_col]
    res = filter(x -> !isnothing(findfirst(token, x)), data_columns)

    if length(res) > 0
        throw(ArgumentError("$token_name \"$token\" is already used in the dataset"))
    end
end

"""
Calculate the max timestep given training and validation datasets.
"""
function cal_max_timestep(
    data_train,
    data_val,
    target_col;
    tokenized = false,
    sep_token = "",
)
    words_train = data_train[:, target_col]
    words_val = data_val[:, target_col]

    if tokenized && !isnothing(sep_token)
        max_l_words_train =
            maximum(x -> length(split(x, sep_token)), words_train)
        max_l_words_val = maximum(x -> length(split(x, sep_token)), words_val)
    else
        max_l_words_train = maximum(x -> length(split(x, "")), words_train)
        max_l_words_val = maximum(x -> length(split(x, "")), words_val)
    end

    maximum([max_l_words_train, max_l_words_val]) + 1
end

"""
Calculate the max timestep given training datasets only.
"""
function cal_max_timestep(data, target_col; tokenized = false, sep_token = "")

    words = data[:, target_col]

    if tokenized && !isnothing(sep_token)
        max_l_words = maximum(x -> length(split(x, sep_token)), words)
    else
        max_l_words = maximum(x -> length(split(x, "")), words)
    end

    max_l_words + 1
end
