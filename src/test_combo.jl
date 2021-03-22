KWARGS_DEFAULT = Dict([
        (:train_sample_size, 0),
        (:val_sample_size, 0),
        (:extension, ".csv"),
        (:val_ratio, 0.0),
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
        (:max_t, 0),
        (:A, nothing),
        (:A_mode, :combined),
        (:max_can, 10),
        (:threshold_train, 0.1),
        (:is_tolerant_train, false),
        (:tolerance_train, -0.1),
        (:max_tolerance_train, 2),
        (:threshold_val, 0.1),
        (:is_tolerant_val, false),
        (:tolerance_val, -0.1),
        (:max_tolerance_val, 2),
        (:issparse, false),
        (:sparse_ratio, 0.05),
        (:n_neighbors_train, 10),
        (:n_neighbors_val, 20),
        (:output_dir, "out"),
        (:verbose, false)
        ])

"""
    test_combo(test_mode;kwargs...)

A wrapper function for a full model for a specific combination of parameters. A detailed introduction is in [Test Combo Introduction](@ref)
```
"""
function test_combo(test_mode;kwargs...)
    verbose = get_kwarg(kwargs, :verbose, required=false)
    train_sample_size = get_kwarg(kwargs, :train_sample_size, required=false)
    val_sample_size = get_kwarg(kwargs, :val_sample_size, required=false)
    val_ratio = get_kwarg(kwargs, :val_ratio, required=false)
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
            train_sample_size = train_sample_size, 
            val_sample_size = val_sample_size)
    elseif test_mode == :pre_split
        data_path = get_kwarg(kwargs, :data_path, required=true)
        data_prefix = get_kwarg(kwargs, :data_prefix, required=true)
        extension = get_kwarg(kwargs, :extension, required=false)
        data_train, data_val = loading_data_pre_split(
            data_path, data_prefix, 
            train_sample_size = train_sample_size, 
            val_sample_size = val_sample_size, extension=extension)
    elseif test_mode == :random_split
        data_path = get_kwarg(kwargs, :data_path, required=true)
        data_output_dir = get_kwarg(kwargs, :data_output_dir, required=true)
        data_prefix = get_kwarg(kwargs, :data_prefix, required=true)

        data_train, data_val = loading_data_randomly_split(
            data_path,
            data_output_dir,
            data_prefix,
            train_sample_size = train_sample_size,
            val_sample_size = val_sample_size,
            val_ratio = val_ratio,
            verbose = verbose)

    elseif test_mode == :careful_split
        data_path = get_kwarg(kwargs, :data_path, required=true)
        data_prefix = get_kwarg(kwargs, :data_prefix, required=true)
        data_output_dir = get_kwarg(kwargs, :data_output_dir, required=true)
        n_features_columns = get_kwarg(kwargs, :n_features_columns, required=true)

        data_train, data_val = loading_data_careful_split(
            data_path,
            data_prefix,
            data_output_dir,
            n_features_columns,
            train_sample_size = train_sample_size,
            val_sample_size = val_sample_size,
            val_ratio=val_ratio,
            n_grams_target_col=n_grams_target_col,
            n_grams_tokenized=n_grams_tokenized,
            n_grams_sep_token=n_grams_sep_token,
            grams = grams,
            n_grams_keep_sep= n_grams_keep_sep,
            start_end_token = start_end_token,
            verbose=verbose)
    else
        throw(ArgumentError("test_mode is incorrect, using :train_only," * 
            " :pre_split, :care_split or :random_split"))
    end

    check_used_token(
        data_train,
        n_grams_target_col,
        start_end_token,
        "start_end_token",
    )

    check_used_token(
        data_val,
        n_grams_target_col,
        path_sep_token,
        "path_sep_token",
    )

    check_used_token(
        data_train,
        n_grams_target_col,
        start_end_token,
        "start_end_token",
    )

    check_used_token(
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
        sd_inflection, isdeep, add_noise, sd_noise,
        normalized, if_combined, verbose)

    # temporary fix, S_val is not valid using :train_only
    # add noise don't apply to both S
    if test_mode == :train_only
        if val_sample_size == 0
            val_sample_size = 2
        end
        S_val = S_train[1:val_sample_size, :]
    end

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
        F_train = make_transform_matrix(
            cue_obj_train.C,
            S_train,
            method = method,
            shift = shift,
            multiplier = multiplier,
            output_format = output_format,
            sparse_ratio = sparse_ratio,
            verbose = verbose,
        )

        G_train = make_transform_matrix(
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
        learn_seq = make_learn_seq(wh_freq)

        F_train = wh_learn(
            cue_obj_train.C,
            S_train;
            eta = eta,
            n_epochs = n_epochs,
            weights = init_weights,
            learn_seq = learn_seq,
            verbose = verbose,
        )

        G_train = wh_learn(
            S_train,
            cue_obj_train.C;
            eta = eta,
            n_epochs = n_epochs,
            weights = init_weights,
            learn_seq = learn_seq,
            verbose = verbose,
        )
    else
        throw(ArgumentError("learn_mode is incorrect, using :cholesky," * 
            ":wh"))
    end

    Shat_train = cue_obj_train.C * F_train
    Shat_val = cue_obj_val.C * F_train
    Chat_train = S_train * G_train
    Chat_val = S_val * G_train

    # calculate max timestep
    max_t = get_kwarg(kwargs, :max_t, required=false)

    if max_t == 0
        max_t = cal_max_timestep(data_train, data_val, 
            n_grams_target_col, tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token)
    end

    # choose A
    A = get_kwarg(kwargs, :A, required=false)
    A_mode = get_kwarg(kwargs, :A_mode, required=false)

    if isnothing(A)
        if A_mode == :combined
            A = make_combined_adjacency_matrix(
                data_train,
                data_val;
                grams = grams,
                target_col = n_grams_target_col,
                tokenized = n_grams_tokenized,
                sep_token = n_grams_sep_token,
                keep_sep = n_grams_keep_sep,
                start_end_token = start_end_token,
                verbose = verbose
            )
        elseif A_mode == :train_only
            A = cue_obj_train.A
        else
            throw(ArgumentError("A_mode $A_mode is not supported!" * 
                "Please choose from :combined or :train_only"))
        end
    end

    max_can = get_kwarg(kwargs, :max_can, required=false)
    threshold_train = get_kwarg(kwargs, :threshold_train, required=false)
    is_tolerant_train = get_kwarg(kwargs, :is_tolerant_train, required=false)
    tolerance_train = get_kwarg(kwargs, :tolerance_train, required=false)
    max_tolerance_train = get_kwarg(kwargs, :max_tolerance_train, required=false)
    threshold_val = get_kwarg(kwargs, :threshold_val, required=false)
    is_tolerant_val = get_kwarg(kwargs, :is_tolerant_val, required=false)
    tolerance_val = get_kwarg(kwargs, :tolerance_val, required=false)
    max_tolerance_val = get_kwarg(kwargs, :max_tolerance_val, required=false)
    issparse = get_kwarg(kwargs, :issparse, required=false)
    sparse_ratio = get_kwarg(kwargs, :sparse_ratio, required=false)

    res_learn_train, gpi_learn_train = learn_paths(
        data_train,
        data_train,
        cue_obj_train.C,
        S_train,
        F_train,
        Chat_train,
        A,
        cue_obj_train.i2f,
        cue_obj_train.f2i,
        gold_ind = cue_obj_train.gold_ind,
        Shat_val = Shat_train,
        check_gold_path = true,
        max_t = max_t,
        max_can = max_can,
        threshold = threshold_train,
        is_tolerant = is_tolerant_train,
        tolerance = tolerance_train,
        max_tolerance = max_tolerance_train,
        grams = grams,
        tokenized = n_grams_tokenized,
        sep_token = n_grams_sep_token,
        keep_sep = n_grams_keep_sep,
        target_col = n_grams_target_col,
        issparse = issparse,
        sparse_ratio = sparse_ratio,
        verbose = verbose)

    res_learn_val, gpi_learn_val = learn_paths(
        data_train,
        data_val,
        cue_obj_train.C,
        S_val,
        F_train,
        Chat_val,
        A,
        cue_obj_train.i2f,
        cue_obj_train.f2i,
        gold_ind = cue_obj_val.gold_ind,
        Shat_val = Shat_val,
        check_gold_path = true,
        max_t = max_t,
        max_can = max_can,
        threshold = threshold_val,
        is_tolerant = is_tolerant_val,
        tolerance = tolerance_val,
        max_tolerance = max_tolerance_val,
        grams = grams,
        tokenized = n_grams_tokenized,
        sep_token = n_grams_sep_token,
        keep_sep = n_grams_keep_sep,
        target_col = n_grams_target_col,
        issparse = issparse,
        sparse_ratio = sparse_ratio,
        verbose = verbose)

    n_neighbors_train = get_kwarg(kwargs, :n_neighbors_train, required=false)
    n_neighbors_val = get_kwarg(kwargs, :n_neighbors_val, required=false)

    rC_train = cor(Chat_train, Matrix(cue_obj_train.C), dims = 2)
    rC_val = cor(Chat_val, Matrix(cue_obj_train.C), dims = 2)

    res_build_train = build_paths(
        data_train,
        cue_obj_train.C,
        S_train,
        F_train,
        Chat_train,
        A,
        cue_obj_train.i2f,
        cue_obj_train.gold_ind;
        rC = rC_train,
        max_t = max_t,
        max_can = max_can,
        n_neighbors = n_neighbors_train,
        grams = grams,
        tokenized = n_grams_tokenized,
        sep_token = n_grams_sep_token,
        target_col = n_grams_target_col,
        verbose = verbose,
    )

    res_build_val = build_paths(
        data_val,
        cue_obj_train.C,
        S_val,
        F_train,
        Chat_val,
        A,
        cue_obj_val.i2f,
        cue_obj_train.gold_ind;
        rC = rC_val,
        max_t = max_t,
        max_can = max_can,
        n_neighbors = n_neighbors_val,
        grams = grams,
        tokenized = n_grams_tokenized,
        sep_token = n_grams_sep_token,
        target_col = n_grams_target_col,
        verbose = verbose,
    )

    acc_Chat_train = eval_SC(Chat_train, cue_obj_train.C)
    acc_Shat_train = eval_SC(Shat_train, S_train)
    acc_Shat_train_homo = eval_SC(Shat_train, S_train, data_train, n_grams_target_col)

    acc_Chat_val = eval_SC(Chat_val, cue_obj_val.C)
    acc_Shat_val = eval_SC(Shat_val, S_val)
    acc_Shat_val_homo = eval_SC(Shat_val, S_val, data_val, n_grams_target_col)

    acc_learn_train = JudiLing.eval_acc(
        res_learn_train,
        cue_obj_train.gold_ind,
        verbose=verbose
    )

    acc_learn_val = JudiLing.eval_acc(
        res_learn_val,
        cue_obj_val.gold_ind,
        verbose=verbose
    )

    acc_build_train = JudiLing.eval_acc(
        res_build_train,
        cue_obj_train.gold_ind,
        verbose=verbose
    )

    acc_build_val = JudiLing.eval_acc(
        res_build_val,
        cue_obj_val.gold_ind,
        verbose=verbose
    )

    output_dir = get_kwarg(kwargs, :output_dir, required=false)

    mkpath(output_dir)
    accio = open(joinpath(output_dir, "acc.out"), "w")
    println(accio, "Acc for Chat train: $acc_Chat_train")
    println(accio, "Acc for Shat train: $acc_Shat_train")
    println(accio, "Acc for Shat train homophones: $acc_Shat_train_homo")
    println(accio, "Acc for Chat val: $acc_Chat_val")
    println(accio, "Acc for Shat val: $acc_Shat_val")
    println(accio, "Acc for Shat val homophones: $acc_Shat_val_homo")
    println(accio, "Acc for learn_path train: $acc_learn_train")
    println(accio, "Acc for learn_path val: $acc_learn_val")
    println(accio, "Acc for build_path train: $acc_build_train")
    println(accio, "Acc for build_path val: $acc_build_val")
    close(accio)

    write2csv(
        res_learn_train,
        data_train,
        cue_obj_train,
        cue_obj_train,
        "learn_res_train.csv",
        grams=grams,
        tokenized=n_grams_tokenized,
        sep_token=n_grams_sep_token,
        start_end_token=start_end_token,
        output_sep_token=n_grams_sep_token,
        path_sep_token=path_sep_token,
        target_col=n_grams_target_col,
        root_dir=".",
        output_dir=output_dir
        )

    write2csv(
        gpi_learn_train,
        "learn_gpi_train.csv",
        root_dir=".",
        output_dir=output_dir
        )

    write2csv(
        res_learn_val,
        data_val,
        cue_obj_train,
        cue_obj_val,
        "learn_res_val.csv",
        grams=grams,
        tokenized=n_grams_tokenized,
        sep_token=n_grams_sep_token,
        start_end_token=start_end_token,
        output_sep_token=n_grams_sep_token,
        path_sep_token=path_sep_token,
        target_col=n_grams_target_col,
        root_dir=".",
        output_dir=output_dir
        )

    write2csv(
        gpi_learn_val,
        "learn_gpi_val.csv",
        root_dir=".",
        output_dir=output_dir
        )

    write2csv(
        res_build_train,
        data_train,
        cue_obj_train,
        cue_obj_train,
        "build_res_train.csv",
        grams=grams,
        tokenized=n_grams_tokenized,
        sep_token=n_grams_sep_token,
        start_end_token=start_end_token,
        output_sep_token=n_grams_sep_token,
        path_sep_token=path_sep_token,
        target_col=n_grams_target_col,
        root_dir=".",
        output_dir=output_dir
        )

    write2csv(
        res_build_val,
        data_val,
        cue_obj_train,
        cue_obj_val,
        "build_res_val.csv",
        grams=grams,
        tokenized=n_grams_tokenized,
        sep_token=n_grams_sep_token,
        start_end_token=start_end_token,
        output_sep_token=n_grams_sep_token,
        path_sep_token=path_sep_token,
        target_col=n_grams_target_col,
        root_dir=".",
        output_dir=output_dir
        )

end

function loading_data_train_only(
    data_path;
    train_sample_size = 0, 
    val_sample_size = 0)

    data = DataFrame(CSV.File(data_path))

    if train_sample_size != 0
        data = data[1:train_sample_size, :]
    end

    if val_sample_size == 0
        val_sample_size = 2
    end

    data, data[1:val_sample_size, :]
end

function loading_data_pre_split(
    data_output_dir,
    data_prefix;
    train_sample_size = 0,
    val_sample_size = 0,
    extension=".csv")

    train_path = joinpath(data_output_dir, data_prefix * "_train" * extension)
    val_path = joinpath(data_output_dir, data_prefix * "_val" * extension)
    data_train = DataFrame(CSV.File(train_path))
    data_val = DataFrame(CSV.File(val_path))

    if train_sample_size != 0
        data_train = data_train[1:train_sample_size, :]
    end

    if val_sample_size != 0
        data_val = data_val[1:val_sample_size, :]
    end

    data_train, data_val
end

function loading_data_randomly_split(
    data_path,
    output_dir_path,
    data_prefix;
    train_sample_size = 0,
    val_sample_size = 0,
    val_ratio = 0.0,
    verbose = false)
    verbose && println("Spliting data...")

    train_val_random_split(
        data_path,
        output_dir_path,
        data_prefix,
        train_sample_size = train_sample_size,
        val_sample_size = val_sample_size,
        val_ratio = val_ratio,
        verbose = verbose,
        )

    # load data
    verbose && println("Loading CSV...")
    loading_data_pre_split(output_dir_path, data_prefix)
end

function loading_data_careful_split(
    data_path,
    data_prefix,
    output_dir_path,
    n_features_columns;
    train_sample_size = 0,
    val_sample_size = 0,
    val_ratio = 0.0,
    n_grams_target_col = :Word,
    n_grams_tokenized = false,
    n_grams_sep_token = nothing,
    grams = 3,
    n_grams_keep_sep = false,
    start_end_token = "#",
    verbose = false)

    verbose && println("Spliting data...")
    train_val_careful_split(
        data_path,
        output_dir_path,
        data_prefix,
        n_features_columns,
        train_sample_size = train_sample_size,
        val_sample_size = val_sample_size,
        val_ratio = val_ratio,
        n_grams_target_col = n_grams_target_col,
        n_grams_tokenized = n_grams_tokenized,
        n_grams_sep_token = n_grams_sep_token,
        grams = grams,
        n_grams_keep_sep = n_grams_keep_sep,
        start_end_token = start_end_token,
        verbose = verbose,
    )

    # load data
    verbose && println("Loading CSV...")
    loading_data_pre_split(output_dir_path, data_prefix)
end

function make_cue_train_only(data, grams, target_col, tokenized, sep_token,
    keep_sep, start_end_token, verbose)

    make_cue_matrix(
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
        cue_obj_train, cue_obj_val = make_combined_cue_matrix(
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
        cue_obj_train, cue_obj_val = make_cue_matrix(
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
    ncol, sd_base_mean, sd_inflection_mean, sd_base, sd_inflection,
    isdeep, add_noise, sd_noise, normalized, verbose)
    verbose && println("Making S matrix...")
    make_S_matrix(
        data,
        n_features_base,
        n_features_inflections,
        ncol = ncol,
        sd_base_mean = sd_base_mean,
        sd_inflection_mean = sd_inflection_mean,
        sd_base = sd_base,
        sd_inflection = sd_inflection,
        isdeep = isdeep,
        add_noise = add_noise,
        sd_noise = sd_noise,
        normalized = normalized
    )
end

function make_S_train_val(data_train, data_val,
    n_features_base, n_features_inflections,
    ncol, sd_base_mean, sd_inflection_mean, sd_base, sd_inflection,
    isdeep, add_noise, sd_noise, normalized, if_combined, verbose)
    verbose && println("Making S matrix...")
    if if_combined
        make_combined_S_matrix(
            data_train,
            data_val,
            n_features_base,
            n_features_inflections,
            ncol = ncol,
            sd_base_mean = sd_base_mean,
            sd_inflection_mean = sd_inflection_mean,
            sd_base = sd_base,
            sd_inflection = sd_inflection,
            isdeep = isdeep,
            add_noise = add_noise,
            sd_noise = sd_noise,
            normalized = normalized
        )
    else
        make_S_matrix(
            data_train,
            data_val,
            n_features_base,
            n_features_inflections,
            ncol = ncol,
            sd_base_mean = sd_base_mean,
            sd_inflection_mean = sd_inflection_mean,
            sd_base = sd_base,
            sd_inflection = sd_inflection,
            isdeep = isdeep,
            add_noise = add_noise,
            sd_noise = sd_noise,
            normalized = normalized
        )
    end
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