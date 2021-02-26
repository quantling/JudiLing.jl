using JudiLing
using CSV
using DataFrames
using Random

KWARGS_DEFAULT = Dict([
        (:train_sample_size, 0),
        (:val_sample_size, 0),
        (:extension, ".csv"),
        (:max_val, 0),
        (:max_val_ratio, 0.0),
        (:n_grams_target_col, :Word),
        (:n_grams_tokenized, false),
        (:n_grams_sep_token, nothing),
        (:grams, 3),
        (:n_grams_keep_sep, false),
        (:start_end_token, "#"),
        (:path_sep_token, ":"),
        (:random_seed, 314),
        (:sd_base_mean, 1),
        (:sd_inflection_mean, 1),
        (:sd_base, 4),
        (:sd_inflection, 4),
        (:seed, 314),
        (:isdeep, true),
        (:add_noise, true),
        (:sd_noise, 1),
        (:normalized, false),
        (:if_combined, false),
        (:learn_mode, :cholesky),
        (:method, :additive),
        (:shift, 0.02),
        (:multiplier, 1.01),
        (:output_format, :auto),
        (:sparse_ratio, 0.2),
        (:wh_freq, nothing),
        (:init_weights, nothing),
        (:eta, 0.1),
        (:n_epochs, 1),
        (:verbose, false),
        ])

function test_combo(test_mode;kwargs...)
    verbose = get_kwarg(kwargs, :verbose, required=false)
    train_sample_size = get_kwarg(kwargs, :train_sample_size, required=false)
    val_sample_size = get_kwarg(kwargs, :val_sample_size, required=false)
    n_grams_target_col = get_kwarg(kwargs, :n_grams_target_col, required=false)
    n_grams_tokenized = get_kwarg(kwargs, :n_grams_tokenized, required=false)
    n_grams_sep_token = get_kwarg(kwargs, :n_grams_sep_token, required=false)
    grams = get_kwarg(kwargs, :grams, required=false)
    n_grams_keep_sep = get_kwarg(kwargs, :n_grams_keep_sep, required=false)
    start_end_token = get_kwarg(kwargs, :start_end_token, required=false)
    path_sep_token = get_kwarg(kwargs, :path_sep_token, required=false)
    random_seed = get_kwarg(kwargs, :random_seed, required=false)
    if_combined = get_kwarg(kwargs, :if_combined, required=false)

    # split and load data
    if test_mode == :train_only
        data_path = get_kwarg(kwargs, :data_path, required=true)
        data_train, data_val = loading_data_train_only(data_path, 
            train_sample_size, val_sample_size)
    elseif test_mode == :pre_split
        data_dir_path = get_kwarg(kwargs, :data_dir_path, required=true)
        data_prefix = get_kwarg(kwargs, :data_prefix, required=true)
        extension = get_kwarg(kwargs, :extension, required=false)
        data_train, data_val = loading_data_pre_split(
            data_dir_path, data_prefix, extension=extension)
    elseif test_mode == :random_split
        data_path = get_kwarg(kwargs, :data_path, required=true)
        output_dir_path = get_kwarg(kwargs, :output_dir_path, required=true)
        data_prefix = get_kwarg(kwargs, :data_prefix, required=true)

        max_val = get_kwarg(kwargs, :max_val, required=false)
        max_val_ratio = get_kwarg(kwargs, :max_val_ratio, required=false)

        data_train, data_val = loading_data_randomly_split(
            data_path,
            test_sample_size,
            output_dir_path,
            data_prefix;
            max_val,
            max_val_ratio,
            random_seed,
            verbose)

    elseif test_mode == :carefully_split
        data_path = get_kwarg(kwargs, :data_path, required=true)
        data_prefix = get_kwarg(kwargs, :data_prefix, required=true)
        output_dir_path = get_kwarg(kwargs, :output_dir_path, required=true)
        n_features_columns = get_kwarg(kwargs, :n_features_columns, required=true)

        max_val = get_kwarg(kwargs, :max_val, required=false)
        max_val_ratio = get_kwarg(kwargs, :max_val_ratio, required=false)

        random_seed = get_kwarg(kwargs, :random_seed, required=false)

        data_train, data_val = loading_data_carefully_split(
            data_path,
            test_sample_size,
            data_prefix,
            output_dir_path,
            n_features_columns,
            max_val=max_val,
            max_val_ratio=max_val_ratio,
            n_grams_target_col=n_grams_target_col,
            n_grams_tokenized=n_grams_tokenized,
            n_grams_sep_token=n_grams_sep_token,
            grams = grams,
            n_grams_keep_sep= n_grams_keep_sep,
            start_end_token = start_end_token,
            random_seed=random_seed,
            verbose=verbose)
    else
        throw(ArgumentError("test_mode is incorrect, using :train_only," * 
            " :pre_split, :care_split or :random_split"))
    end

    JudiLing.check_used_token(
        data_train,
        n_grams_target_col,
        start_end_token,
        "start_end_token",
    )

    JudiLing.check_used_token(
        data_val,
        n_grams_target_col,
        path_sep_token,
        "path_sep_token",
    )

    JudiLing.check_used_token(
        data_train,
        n_grams_target_col,
        start_end_token,
        "start_end_token",
    )

    JudiLing.check_used_token(
        data_val,
        n_grams_target_col,
        path_sep_token,
        "path_sep_token",
    )

    n_features_base = get_kwarg(kwargs, :n_features_base, required=true)
    n_features_inflections = get_kwarg(kwargs, :n_features_inflections, required=true)

    sd_base_mean = get_kwarg(kwargs, :sd_base_mean, required=false)
    sd_inflection_mean = get_kwarg(kwargs, :sd_inflection_mean, required=false)
    sd_base = get_kwarg(kwargs, :sd_base, required=false)
    sd_inflection = get_kwarg(kwargs, :sd_inflection, required=false)
    isdeep = get_kwarg(kwargs, :isdeep, required=false)
    add_noise = get_kwarg(kwargs, :add_noise, required=false)
    sd_noise = get_kwarg(kwargs, :sd_noise, required=false)
    normalized = get_kwarg(kwargs, :normalized, required=false)

    # make cue matrix/matrices
    # make semantic matrix/matrices

    cue_obj_train, cue_obj_val = make_cue_train_val(
        data_train,
        data_val,
        grams,
        n_grams_target_col,
        n_grams_tokenized,
        n_grams_sep_token,
        n_grams_keep_sep,
        start_end_token,
        if_combined,
        verbose)

    n_features = size(cue_obj_train.C, 2)
    S_train, S_val = make_S_train_val(data_train, data_val,
        n_features_base, n_features_inflections,
        n_features, sd_base_mean, sd_inflection_mean, sd_base,
        sd_inflection, random_seed, isdeep, add_noise, sd_noise,
        normalized, verbose)


    learn_mode = get_kwarg(kwargs, :learn_mode, required=false)

    # cholesky params
    method = get_kwarg(kwargs, :method, required=false)
    shift = get_kwarg(kwargs, :shift, required=false)
    multiplier = get_kwarg(kwargs, :multiplier, required=false)
    output_format = get_kwarg(kwargs, :output_format, required=false)
    sparse_ratio = get_kwarg(kwargs, :sparse_ratio, required=false)

    # wh params
    wh_freq = get_kwarg(kwargs, :wh_freq, required=false)
    init_weights = get_kwarg(kwargs, :init_weights, required=false)
    eta = get_kwarg(kwargs, :eta, required=false)
    n_epochs = get_kwarg(kwargs, :n_epochs, required=false)

    if learn_mode == :cholesky
        F_train = JudiLing.make_transform_matrix(
            cue_obj_train.C,
            S_train,
            method = method,
            shift = shift,
            multiplier = multiplier,
            output_format = output_format,
            sparse_ratio = sparse_ratio,
            verbose = verbose,
        )

        G_train = JudiLing.make_transform_matrix(
            S_train,
            cue_obj_train.C,
            method = method,
            shift = shift,
            multiplier = multiplier,
            output_format = output_format,
            sparse_ratio = sparse_ratio,
            verbose = verbose,
        )
    elseif learn_mode == :wh
        learn_seq = JudiLing.make_learn_seq(wh_freq, random_seed=random_seed)

        F_train = JudiLing.wh_learn(
            cue_obj_train.C,
            S_train;
            eta = eta,
            n_epochs = n_epochs,
            weights = init_weights,
            learn_seq = learn_seq,
            verbose = verbose,
            )

        G_train = JudiLing.wh_learn(
            S_train,
            cue_obj_train.C;
            eta = eta,
            n_epochs = n_epochs,
            weights = init_weights,
            learn_seq = learn_seq,
            verbose = verbose,
            )
    end

    display(F_train)
    display(G_train)
    # display(S)
end

function loading_data_train_only(data_path, test_sample_size)
    data = DataFrame(CSV.File(data_path))
    if test_sample_size == 0
        test_sample_size = size(data, 2)
    end
    data, data[1:test_sample_size, :]
end

function loading_data_pre_split(
    data_dir_path,
    data_prefix;
    extension=".csv")

    train_path = joinpath(data_dir_path, data_prefix * "_train" * extension)
    val_path = joinpath(data_dir_path, data_prefix * "_val" * extension)
    data_train = DataFrame(CSV.File(train_path))
    data_val = DataFrame(CSV.File(val_path))

    data_train, data_val
end

function loading_data_randomly_split(
    data_path,
    test_sample_size,
    output_dir_path,
    data_prefix;
    max_val,
    max_val_ratio,
    random_seed,
    verbose)
    verbose && println("Spliting data...")

    JudiLing.train_val_random_split(
        data_path,
        test_sample_size,
        output_dir_path,
        data_prefix,
        max_val = max_val,
        max_val_ratio = max_val_ratio,
        random_seed = random_seed,
        verbose = verbose,
        )

    # load data
    verbose && println("Loading CSV...")
    loading_data_pre_split(output_dir_path, data_prefix)
end

function loading_data_carefully_split(
    data_path,
    test_sample_size,
    data_prefix,
    output_dir_path,
    n_features_columns;
    max_val=0,
    max_val_ratio=0.0,
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    n_grams_sep_token = nothing,
    grams = 3,
    n_grams_keep_sep= false,
    start_end_token = "#",
    random_seed=314,
    verbose=false)

    verbose && println("Spliting data...")
    JudiLing.train_val_carefully_split(
        data_path,
        test_sample_size,
        output_dir_path,
        n_features_columns,
        data_prefix = data_prefix,
        max_val = max_val,
        max_val_ratio = max_val_ratio,
        n_grams_target_col = n_grams_target_col,
        n_grams_tokenized = n_grams_tokenized,
        n_grams_sep_token = n_grams_sep_token,
        grams = grams,
        n_grams_keep_sep = n_grams_keep_sep,
        start_end_token = start_end_token,
        random_seed = random_seed,
        verbose = verbose,
    )

    # load data
    verbose && println("Loading CSV...")
    loading_data_pre_split(output_dir_path, data_prefix)
end

function make_cue_train_only(data, grams, target_col, tokenized, sep_token,
    keep_sep, start_end_token, verbose)

    JudiLing.make_cue_matrix(
        data,
        grams = grams,
        target_col = target_col,
        tokenized = tokenized,
        sep_token = sep_token,
        keep_sep = keep_sep,
        start_end_token = start_end_token,
        verbose = verbose
        )
end

function make_cue_train_val(data_train, data_val, grams, target_col, tokenized,
    sep_token, keep_sep, start_end_token, if_combined, verbose)

    if if_combined
        cue_obj_train, cue_obj_val = JudiLing.make_combined_cue_matrix(
            data_train,
            data_val,
            grams = grams,
            target_col = target_col,
            tokenized = tokenized,
            sep_token = sep_token,
            keep_sep = keep_sep,
            start_end_token = start_end_token,
            verbose = verbose
            )
    else
        cue_obj_train, cue_obj_val = JudiLing.make_cue_matrix(
            data_train,
            data_val,
            grams = grams,
            target_col = target_col,
            tokenized = tokenized,
            sep_token = sep_token,
            keep_sep = keep_sep,
            start_end_token = start_end_token,
            verbose = verbose
            )
    end
    cue_obj_train, cue_obj_val
end

function make_S_train_only(data, n_features_base, n_features_inflections,
    ncol, sd_base_mean, sd_inflection_mean, sd_base, sd_inflection, seed,
    isdeep, add_noise, sd_noise, normalized, verbose)
    verbose && println("Making S matrix...")
    JudiLing.make_S_matrix(
        data,
        n_features_base,
        n_features_inflections,
        ncol = ncol,
        sd_base_mean = sd_base_mean,
        sd_inflection_mean = sd_inflection_mean,
        sd_base = sd_base,
        sd_inflection = sd_inflection,
        seed = seed,
        isdeep = isdeep,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized
    )
end

function make_S_train_val(data_train, data_val,
    n_features_base, n_features_inflections,
    ncol, sd_base_mean, sd_inflection_mean, sd_base, sd_inflection, seed,
    isdeep, add_noise, sd_noise, normalized, verbose)
    verbose && println("Making S matrix...")
    JudiLing.make_S_matrix(
        data_train,
        data_val,
        n_features_base,
        n_features_inflections,
        ncol = ncol,
        sd_base_mean = sd_base_mean,
        sd_inflection_mean = sd_inflection_mean,
        sd_base = sd_base,
        sd_inflection = sd_inflection,
        seed = seed,
        isdeep = isdeep,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized
    )
end

function get_kwarg(kwargs, kw; required=false)
    if !haskey(kwargs, kw)
        if required
            throw(ArgumentError("$kw is not specified!"))
        else
            return get_default_kwargs(kw)
        end
    end
    kwargs[kw]
end

function get_default_kwargs(kw)
    if !haskey(KWARGS_DEFAULT, kw)
        throw(ArgumentError("$kw don't have default value!"))
    else
        return KWARGS_DEFAULT[kw]
    end
end

test_combo(
    :train_only,
    data_path = joinpath(@__DIR__, "data", "latin.csv"),
    # max_val = 200,
    # max_val_ratio = 0.2,
    data_prefix="latin",
    output_dir_path=joinpath(@__DIR__, "data"),
    n_features_columns=["Lexeme","Person","Number","Tense","Voice","Mood"],
    n_grams_target_col=:Word,
    n_grams_tokenized=true,
    n_grams_sep_token="",
    n_grams_keep_sep=true,
    grams=2,
    n_features_base = ["Lexeme"],
    n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
    if_combined = true,
    learn_mode = :cholesky,
    # learn_mode = :wh,
    # eta = 0.00001,
    # n_epochs = 100,
    verbose = true
    )

# test_combo(
#     :pre_split,
#     data_dir_path=joinpath(@__DIR__, "data"),
#     data_prefix="estonian",
#     n_features_columns=["Lexeme","Case","Number"],
#     n_grams_target_col=:Word,
#     n_grams_tokenized=false,
#     n_grams_sep_token="",
#     n_grams_keep_sep=false,
#     grams=2,
#     n_features_base = ["Lexeme"],
#     n_features_inflections = ["Lexeme","Case","Number"],
#     if_combined = false,
#     learn_mode = :cholesky,
#     # learn_mode = :wh,
#     # eta = 0.00001,
#     # n_epochs = 100,
#     verbose = true)

# test_combo(
#     :carefully_split,
#     data_path = joinpath(@__DIR__, "data", "latin.csv"),
#     max_val = 200,
#     # max_val_ratio = 0.2,
#     data_prefix="latin",
#     output_dir_path=joinpath(@__DIR__, "data"),
#     n_features_columns=["Lexeme","Person","Number","Tense","Voice","Mood"],
#     n_grams_target_col=:Word,
#     n_grams_tokenized=true,
#     n_grams_sep_token="",
#     n_grams_keep_sep=true,
#     grams=2,
#     n_features_base = ["Lexeme"],
#     n_features_inflections = ["Person","Number","Tense","Voice","Mood"],
#     if_combined = true,
#     learn_mode = :cholesky,
#     # learn_mode = :wh,
#     # eta = 0.00001,
#     # n_epochs = 100,
#     verbose = true
#     )

# test_combo(
#     :random_split,
#     test_sample_size = 1000,
#     data_path = joinpath(@__DIR__, "data", "french.csv"),
#     max_val = 200,
#     # max_val_ratio = 0.2,
#     data_prefix = "french",
#     output_dir_path = joinpath(@__DIR__, "data"),
#     n_features_columns = ["Lexeme","Tense","Aspect","Person","Number","Gender","Class","Mood"],
#     n_grams_target_col = :Syllables,
#     n_grams_tokenized = true,
#     n_grams_sep_token = "-",
#     n_grams_keep_sep = true,
#     grams = 2,
#     n_features_base = ["Lexeme"],
#     n_features_inflections = ["Tense","Aspect","Person","Number","Gender","Class","Mood"],
#     if_combined = true,
#     learn_mode = :cholesky,
#     # learn_mode = :wh,
#     # eta = 0.00001,
#     # n_epochs = 100,
#     verbose = true
#     )

;