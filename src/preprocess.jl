struct SplitDataException <: Exception
    msg::String
end

"""
Split dataset into training and validation datasets.
"""
function train_val_split end

"""
Leave p out cross-validation.
"""
function lpo_cv_split end

"""
Leave one out cross-validation.
"""
function loo_cv_split end

"""
    lpo_cv_split(p, data_path)

Leave p out cross-validation.
"""
function lpo_cv_split(p, data_path; random_seed = 314)
    # read csv
    data = DataFrame(CSV.File(data_path))

    # shuffle data
    rng = MersenneTwister(random_seed)
    data = data[shuffle(rng, 1:size(data, 1)), :]

    data[p+1:end, :], data[1:p, :]
end

"""
    loo_cv_split(data_path)

Leave one out cross-validation.
"""
function loo_cv_split(data_path; random_seed = 314)
    lpo_cv_split(1, data_path)
end

function train_val_random_split(
    data_path,
    test_sample_size,
    output_dir_path,
    data_prefix;
    max_val = 0,
    max_val_ratio = 0.0,
    random_seed = 314,
    verbose = false,
    )
    
    data = DataFrame(CSV.File(data_path))
    n_data = size(data, 1)

    if test_sample_size != 0
        data = data[1:test_sample_size,:]
        n_data = test_sample_size
    end

    rng = MersenneTwister(random_seed)
    data = data[shuffle(rng, 1:n_data), :]

    max_val = cal_max_val(n_data, max_val, max_val_ratio)

    data_train = data[max_val+1:end,:]
    data_val = data[1:max_val,:]
    
    write_split_data(output_dir_path, data_prefix, data_train, data_val)

    verbose && begin
        println("Successfully randomly split data into $(size(data_train, 1))" * 
            " training data and $(size(data_val, 1)) validation data")
    end

    nothing
end

function train_val_carefully_split(
    data_path,
    test_sample_size,
    output_dir_path,
    n_features_columns;
    data_prefix = "data",
    max_val = 0,
    max_val_ratio = 0.0,
    n_grams_target_col = :PhonWord,
    n_grams_tokenized = false,
    n_grams_sep_token = nothing,
    grams = 3,
    n_grams_keep_sep = false,
    start_end_token = "#",
    random_seed = 314,
    verbose = false,
    )
    
    data = DataFrame(CSV.File(data_path))
    n_data = size(data, 1)

    if test_sample_size != 0
        data = data[1:test_sample_size,:]
        n_data = test_sample_size
    end

    rng = MersenneTwister(random_seed)
    data = data[shuffle(rng, 1:n_data), :]

    max_val = cal_max_val(n_data, max_val, max_val_ratio)

    init_num_train = round(Int64, (n_data - max_val) * 0.5)
    data_train = data[1:init_num_train, :]

    if n_grams_tokenized && !isnothing(n_grams_sep_token)
        tokens =
            split.(data_train[:, n_grams_target_col], n_grams_sep_token)
    else
        tokens = split.(data_train[:, n_grams_target_col], "")
    end

    verbose && println("Calculating data_train_ngrams ...")
    data_train_ngrams = String[]

    for i = 1:init_num_train
        push!(
            data_train_ngrams,
            make_ngrams(
                tokens[i],
                grams,
                n_grams_keep_sep,
                n_grams_sep_token,
                start_end_token,
            )...,
        )
    end
    data_train_ngrams = unique(data_train_ngrams)
    data_train_features =
        collect_features(data[1:init_num_train, :], n_features_columns)
    data_val = DataFrame()

    perform_split(
        data[init_num_train+1:end, :],
        data_train_ngrams,
        data_train_features,
        data_train,
        data_val,
        max_val,
        grams,
        n_grams_target_col,
        n_grams_tokenized,
        n_grams_sep_token,
        n_grams_keep_sep,
        start_end_token,
        n_features_columns,
        verbose = verbose,
        )

    if size(data_train, 1) <= 0 || size(data_val, 1) <= 0
        throw(SplitDataException("Failed to split data automaticly"))
    end

    write_split_data(output_dir_path, data_prefix, data_train, data_val)

    verbose && begin
        println("Successfully carefully split data into $(size(data_train, 1))" * 
            " training data and $(size(data_val, 1)) validation data")
    end

    nothing
end

function train_val_split(
    data_path,
    output_dir_path,
    n_features_columns;
    data_prefix = "data",
    max_test_data = nothing,
    split_max_ratio = 0.2,
    n_grams_target_col = :PhonWord,
    n_grams_tokenized = false,
    n_grams_sep_token = nothing,
    grams = 3,
    n_grams_keep_sep = false,
    start_end_token = "#",
    random_seed = 314,
    verbose = false,
    )

    # read csv
    utterances = DataFrame(CSV.File(data_path))

    # shuffle data
    rng = MersenneTwister(random_seed)
    utterances = utterances[shuffle(rng, 1:size(utterances, 1)), :]

    if !isnothing(max_test_data)
        utterances = utterances[1:max_test_data, :]
    end

    num_utterances = size(utterances, 1)

    max_num_val = round(Int64, num_utterances * split_max_ratio)
    init_num_train = round(Int64, (num_utterances - max_num_val) * 0.5)

    utterances_train = utterances[1:init_num_train, :]

    if n_grams_tokenized && !isnothing(n_grams_sep_token)
        tokens =
            split.(utterances_train[:, n_grams_target_col], n_grams_sep_token)
    else
        tokens = split.(utterances_train[:, n_grams_target_col], "")
    end

    verbose && println("Calculating utterances_train_ngrams ...")
    utterances_train_ngrams = String[]

    for i = 1:init_num_train
        push!(
            utterances_train_ngrams,
            make_ngrams(
                tokens[i],
                grams,
                n_grams_keep_sep,
                n_grams_sep_token,
                start_end_token,
            )...,
        )
    end
    utterances_train_ngrams = unique(utterances_train_ngrams)
    utterances_train_features =
        collect_features(utterances[1:init_num_train, :], n_features_columns)
    utterances_val = DataFrame()

    perform_split(
        utterances[init_num_train+1:end, :],
        utterances_train_ngrams,
        utterances_train_features,
        utterances_train,
        utterances_val,
        max_num_val,
        grams,
        n_grams_target_col,
        n_grams_tokenized,
        n_grams_sep_token,
        n_grams_keep_sep,
        start_end_token,
        n_features_columns,
        verbose = verbose,
    )

    if size(utterances_train, 1) <= 0 || size(utterances_val, 1) <= 0
        throw(SplitDataException("Could not split data automaticly"))
    end

    write_split_data(output_dir_path, data_prefix, data_train, data_val)

    verbose && begin
        println("Successfully split data into $(size(utterances_train, 1)) training data and $(size(utterances_val, 1)) validation data")
    end

    nothing
end

function train_val_split(
    data_path,
    output_dir_path;
    data_prefix = "data",
    split_max_ratio = 0.2,
    n_grams_target_col = :Word_n_grams,
    n_grams_tokenized = false,
    n_grams_sep_token = nothing,
    n_features_col_name = :CommunicativeIntention,
    n_features_tokenized = false,
    n_features_sep_token = nothing,
    random_seed = 314,
    verbose = false,
)

    # read csv
    utterances = DataFrame(CSV.File(data_path))
    num_utterances = size(utterances, 1)

    # shuffle data
    rng = MersenneTwister(random_seed)
    utterances = utterances[shuffle(rng, 1:size(utterances, 1)), :]

    init_num_train = round(Int64, num_utterances * 0.4)
    max_num_val = round(Int64, num_utterances * split_max_ratio)
    utterances_train = utterances[1:init_num_train, :]
    utterances_train_ngrams = unique([
        ngram for i = 1:init_num_train
        for
        ngram in split_features(
            utterances[i, :],
            n_grams_target_col,
            n_grams_tokenized,
            n_grams_sep_token,
        )
    ])
    utterances_train_features = unique([
        feature for i = 1:init_num_train
        for
        feature in split_features(
            utterances[i, :],
            n_features_col_name,
            n_features_tokenized,
            n_grams_sep_token,
        )
    ])
    utterances_val = DataFrame()

    perform_split(
        utterances[init_num_train+1:end, :],
        utterances_train_ngrams,
        utterances_train_features,
        utterances_train,
        utterances_val,
        max_num_val,
        n_grams_target_col,
        n_grams_tokenized,
        n_grams_sep_token,
        n_features_col_name,
        n_features_tokenized,
        n_features_sep_token,
        verbose = verbose,
    )

    if size(utterances_train, 1) <= 0 || size(utterances_val, 1) <= 0
        throw(SplitDataException("Could not split data automaticly"))
    end

    write_split_data(output_dir_path, data_prefix, data_train, data_val)

    verbose && begin
        println("Successfully split data into $(size(utterances_train, 1)) training data and $(size(utterances_val, 1)) validation data")
        println()
    end
    nothing
end

function split_features(
    datarow,
    col_name,
    tokenized = false,
    sep_token = nothing,
)
    if tokenized && !isnothing(sep_token)
        split(datarow[col_name], sep_token)
    else
        split(datarow[col_name], "")
    end
end

function collect_features(data, n_features_columns)
    features = String[]
    for c in n_features_columns
        push!(features, unique(data[:, c])...)
    end
    unique(features)
end

function perform_split(
    utterances,
    utterances_train_ngrams,
    utterances_train_features,
    utterances_train,
    utterances_val,
    max_num_val,
    n_grams_target_col,
    n_grams_tokenized,
    n_grams_sep_token,
    n_features_col_name,
    n_features_tokenized,
    n_features_sep_token;
    verbose = false,
)

    iter = 1:size(utterances, 1)
    if verbose
        pb = Progress(size(utterances, 1))
    end
    for i in iter
        ngrams = split_features(
            utterances[i, :],
            n_grams_target_col,
            n_grams_tokenized,
            n_grams_sep_token,
        )
        features = split_features(
            utterances[i, :],
            n_features_col_name,
            n_features_tokenized,
            n_features_sep_token,
        )

        # to check whether
        if !any(x -> !any(y -> y == x, utterances_train_ngrams), ngrams) &&
           !any(x -> !any(y -> y == x, utterances_train_features), features) &&
           size(utterances_val, 1) < max_num_val
            if !isnothing(utterances_val)
                push!(utterances_val, utterances[i, :])
            else
                utterances_val = DataFrames.DataFrame(utterances[i, :])
            end
        else
            push!(utterances_train, utterances[i, :])
            utterances_train_ngrams =
                unique(push!(utterances_train_ngrams, ngrams...))
            utterances_train_features =
                unique(push!(utterances_train_features, features...))
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end
end

function perform_split(
    utterances,
    utterances_train_ngrams,
    utterances_train_features,
    utterances_train,
    utterances_val,
    max_num_val,
    grams,
    n_grams_target_col,
    n_grams_tokenized,
    n_grams_sep_token,
    n_grams_keep_sep,
    start_end_token,
    n_features_columns;
    verbose = false,
)

    if n_grams_tokenized && !isnothing(n_grams_sep_token)
        tokens = split.(utterances[:, n_grams_target_col], n_grams_sep_token)
    else
        tokens = split.(utterances[:, n_grams_target_col], "")
    end

    iter = 1:size(utterances, 1)
    if verbose
        pb = Progress(size(utterances, 1))
    end
    for i in iter
        ngrams = make_ngrams(
            tokens[i],
            grams,
            n_grams_keep_sep,
            n_grams_sep_token,
            start_end_token,
        )
        features = unique(utterances[i, n_features_columns])

        # to check whether
        if !any(x -> !any(y -> y == x, utterances_train_ngrams), ngrams) &&
           !any(x -> !any(y -> y == x, utterances_train_features), features) &&
           size(utterances_val, 1) < max_num_val
            if !isnothing(utterances_val)
                push!(utterances_val, utterances[i, :])
            else
                utterances_val = DataFrames.DataFrame(utterances[i, :])
            end
        else
            push!(utterances_train, utterances[i, :])
            utterances_train_ngrams =
                unique(push!(utterances_train_ngrams, ngrams...))
            utterances_train_features =
                unique(push!(utterances_train_features, features...))
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end
end

function preprocess_ndl(
    data_path,
    save_path;
    grams = 3,
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    n_grams_sep_token = "-",
    n_grams_keep_sep = false,
    n_features_columns = [
        "Lexeme",
        "Person",
        "Number",
        "Tense",
        "Voice",
        "Mood",
    ],
)

    # read csv
    data = DataFrame(CSV.File(data_path))

    io = GZip.open(joinpath(save_path), "w")

    cues, outcomes = make_cue_outcome(
        data,
        grams,
        n_grams_target_col,
        n_grams_tokenized,
        n_grams_sep_token,
        n_grams_keep_sep,
        n_features_columns,
    )

    # write header
    write(io, "cues\toutcomes\n")

    for i = 1:size(data, 1)
        write(io, "$(cues[i])\t$(outcomes[i])\n")
    end

    close(io)

    nothing
end

function make_cue_outcome(
    data,
    grams,
    n_grams_target_col,
    n_grams_tokenized,
    n_grams_sep_token,
    n_grams_keep_sep,
    n_features_columns,
)

    n_rows = size(data, 1)

    cues = Vector{String}(undef, n_rows)
    outcomes = Vector{String}(undef, n_rows)

    # split tokens from words or other columns
    if n_grams_tokenized && !isnothing(n_grams_sep_token)
        tokens = split.(data[:, n_grams_target_col], n_grams_sep_token)
    else
        tokens = split.(data[:, n_grams_target_col], "")
    end

    for i = 1:n_rows
        cues[i] = join(
            make_ngrams(
                tokens[i],
                grams,
                n_grams_keep_sep,
                n_grams_sep_token,
                "#",
            ),
            "_",
        )
        outcomes[i] = join(data[i, n_features_columns], "_")
    end

    cues, outcomes
end

function cal_max_val(n_data, max_val, max_val_ratio)
    if max_val == 0
        if max_val_ratio == 0.0
            throw(ArgumentError("You haven't specify :max_val or " * 
                " :max_val_ratio yet!"))
        end
        max_val = round(Int64, n_data * max_val_ratio)
    else
        if max_val_ratio != 0.0
            @warn "You have specified both :max_val and :max_val_ratio. Only" *
            ":max_val will be used!"
        end
    end
    return max_val
end

function write_split_data(output_dir_path, data_prefix, data_train, data_val)
    mkpath(output_dir_path)

    CSV.write(
        joinpath(output_dir_path, "$(data_prefix)_train.csv"),
        data_train,
        quotestrings = true,
    )
    CSV.write(
        joinpath(output_dir_path, "$(data_prefix)_val.csv"),
        data_val,
        quotestrings = true,
    )
end
