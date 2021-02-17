"""
A wrapper function for a full model for a specific combination of parameters.
"""
function test_combo end

"""
    function test_combo(::String, ::Vector, ::Vector, ::Vector) -> ::Nothing

A wrapper function for a full model for a specific combination of parameters.

# Obligatory Arguments
- `data_path::String`: the path for the dataset
- `n_features_columns::Vector`: the list of all features
- `n_features_base::Vector`: the list of all base features
- `n_features_inflections::Vector`: the list of all other features

# Optional Arguments
- `val_only::Bool=false`: if true, then only validation datasets will be evaluate
- `output_dir_path="data"::String`: the path for storing training and validation datasets
- `data_prefix="data"::String`: the prefix for training and validation datasets
- `max_test_data=nothing::Union{Nothing, Int64}`: the maximum number of data in current testing
- `split_max_ratio=0.2::Float64`: proportion of the dataset to be held out for validation
- `issparse::Symbol=:auto`: force output format to dense(:dense), sparse(:sparse), or let the program decide (:auto)
- `sparse_ratio::Float64=0.2`: a matrix is considered sparse when the proportion of 0 cells is lower than `sparse_ratio`
- `is_full_A=false::Bool`: if true, a completed adjacency matrix is constructed using `make_adjacency_matrix` otherwise the adjacency matrix is constructed by `make_cue_matrix`
- `n_grams_target_col=:PhonWord::Symbol`: the column name for target strings
- `n_grams_tokenized=false::Bool`: if true, the dataset target is tokenized
- `n_grams_sep_token=""::Union{String, Char}`: separator
- `n_grams_keep_sep=false::Bool`: if true, keep separators in cues
- `grams::Int64=3`: the number n in n-gram cues
- `start_end_token="#"::Union{String, Char}`: start and end token in boundary cues
- `path_sep_token=":"::Union{String, Char}`: path separator
- `learning_mode=:cholesky::Symbol`: the mode for learning, currently supporting :cholesky, :pyndl and :wh
- `alpha=0.1::Float64`: the alpha value for pyndl learning mode
- `betas=(0.1,0.1)::Tuple{Float64,Float64}`: the beta values for pyndl learning mode
- `eta=0.1::Float64`: the eta learning rate for wh learning mode
- `n_epochs=nothing::Union{Int64, Nothing}`: the number of epochs for wh learning
- `path_method=:build_paths::Symbol`: the mode for constructing paths, :build_paths or :learn_paths
- `max_t::Int64=15`: maximum number of timesteps
- `max_can::Int64=10`: maximum number of candidates to include in the results
- `train_threshold=0.1::Float64`: the value set for the support such that if the support of a n-gram is higher than this value, select the n-gram anyway for training data
- `val_is_tolerant=false::Bool`: if true, select a specified number of n-grams whose supports are below threshold and above tolerance to be added to the path for validation data
- `val_threshold=(-100.0)::Float64`: the value set for the support such that if the support of a n-gram is higher than this value, select the n-gram anyway for validation data
- `val_tolerance=(-1000.0)::Float64`: the value set in tolerant mode such that if the support for a n-gram is inbetween this value and the threshold and the max_tolerance number has not been reached, then allow this n-gram to be added to the path for validation data
- `val_max_tolerance=4::Int64`: maximum number of nodes allowed in a path for validation data
- `train_n_neighbors=2::Int64`: find indices for only the top n neighbors for the training dataset
- `val_n_neighbors=10::Int64`: find indices for only the top n neighbors for the validation dataset
- `root_dir::String="."`: dir path for project root dir
- `csv_dir="out"::String`: output dir inside root dir for storing csv output
- `csv_prefix=""::String`: csv file prefix
- `seed::Int64=314`: the random seed for the whole analysis
- `log_io=stdout::IO`: the log IO
- `verbose::Bool=false`: if true, more information is printed

...
# Examples
```julia
# basic usage
JudiLing.test_combo(
  joinpath("data", "latin.csv"),
  ["Lexeme","Person","Number","Tense","Voice","Mood"],
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  n_grams_target_col=:Word,
  grams=3,
  ...)

# tokenized target
JudiLing.test_combo(
  ...
  n_grams_target_col=:PhonWord,
  n_grams_tokenized=true,
  n_grams_sep_token="-",
  n_grams_keep_sep=true,
  grams=3,
  ...)

# controls of all tokens
JudiLing.test_combo(
  ...
  start_end_token="#",
  path_sep_token=":",
  ...)

# learn_paths mode
JudiLing.test_combo(
  ...
  path_method=:learn_paths,
  train_threshold=0.1,
  val_threshold=0.01,
  ...)

# learn_paths mode with tolerance mode
JudiLing.test_combo(
  ...
  path_method=:learn_paths,
  train_threshold=0.1,
  val_is_tolerant=true,
  val_threshold=0.01,
  val_tolerance=-0.1,
  val_max_tolerance=4,
  ...)

# build_paths mode
JudiLing.test_combo(
  ...
  path_method=:build_paths,
  train_n_neighbors=2,
  val_n_neighbors=10,
  ...)

# control sparse matrix format
JudiLing.test_combo(
  ...
  issparse=:auto,
  sparse_ratio=0.2,
  ...)

# write output log into a File
mkpath(joinpath("french_out"))
test_io = open(joinpath("french_out", "out.log"), "w")

JudiLing.test_combo(
    ...
    log_io=test_io,
    ...)

close(test_io)
```
...
"""
function test_combo(
    data_path::String,
    n_features_columns::Vector,
    n_features_base::Vector,
    n_features_inflections::Vector;
    val_only::Bool = false,
    output_dir_path = "data"::String,
    data_prefix = "data"::String,
    max_test_data = nothing::Union{Nothing,Int64},
    split_max_ratio = 0.2::Float64,
    issparse = :auto::Symbol,
    sparse_ratio = 0.2::Float64,
    is_full_A = false::Bool,
    n_grams_target_col = :PhonWord::Symbol,
    n_grams_tokenized = false::Bool,
    n_grams_sep_token = ""::Union{String,Char},
    n_grams_keep_sep = false::Bool,
    grams = 3::Int64,
    start_end_token = "#"::Union{String,Char},
    path_sep_token = ":"::Union{String,Char},
    learning_mode = :cholesky::Symbol,
    alpha = 0.1::Float64,
    betas = (0.1, 0.1)::Tuple{Float64,Float64},
    eta = 0.1::Float64,
    n_epochs = nothing::Union{Int64,Nothing},
    path_method = :build_paths::Symbol,
    max_t = nothing::Union{Int64,Nothing},
    max_can = 10::Int64,
    train_threshold = 0.1::Float64,
    val_is_tolerant = false::Bool,
    val_threshold = (-100.0)::Float64,
    val_tolerance = (-1000.0)::Float64,
    val_max_tolerance = 4::Int64,
    train_n_neighbors = 2::Int64,
    val_n_neighbors = 10::Int64,
    root_dir = "."::String,
    csv_dir = "out"::String,
    csv_prefix = ""::String,
    random_seed = 314::Int64,
    log_io = stdout::IO,
    verbose = false::Bool,
)::Nothing

    # split data
    verbose && println("spliting data...")
    train_val_split(
        data_path,
        output_dir_path,
        n_features_columns,
        data_prefix = data_prefix,
        max_test_data = max_test_data,
        split_max_ratio = split_max_ratio,
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
    data_train = CSV.DataFrame!(CSV.File(joinpath(
        output_dir_path,
        "$(data_prefix)_train.csv",
    )))
    data_val = CSV.DataFrame!(CSV.File(joinpath(
        output_dir_path,
        "$(data_prefix)_val.csv",
    )))

    check_used_token(
        data_train,
        n_grams_target_col,
        start_end_token,
        "start_end_token",
    )
    check_used_token(
        data_train,
        n_grams_target_col,
        path_sep_token,
        "path_sep_token",
    )
    check_used_token(
        data_val,
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

    if learning_mode == :cholesky
        verbose && println("Making cue matrix...")
        cue_obj_train = make_cue_matrix(
            data_train,
            grams = grams,
            target_col = n_grams_target_col,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            keep_sep = n_grams_keep_sep,
            verbose = verbose,
        )
        cue_obj_val = make_cue_matrix(
            data_val,
            cue_obj_train,
            grams = grams,
            target_col = n_grams_target_col,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            keep_sep = n_grams_keep_sep,
            verbose = verbose,
        )
        verbose && println("training type: $(typeof(cue_obj_train.C))")
        verbose && println("val type: $(typeof(cue_obj_val.C))")
        verbose && println("training size: $(size(cue_obj_train.C))")
        verbose && println("val size: $(size(cue_obj_val.C))")
        verbose && println()

        n_features = size(cue_obj_train.C, 2)

        verbose && println("Making S matrix...")
        S_train, S_val = make_S_matrix(
            data_train,
            data_val,
            n_features_base,
            n_features_inflections,
            ncol = n_features,
        )
        verbose && println("training type: $(typeof(S_train))")
        verbose && println("val type: $(typeof(S_val))")
        verbose && println("training size: $(size(S_train))")
        verbose && println("val size: $(size(S_val))")
        verbose && println()

        verbose && println("Make G matrix...")
        G_train = make_transform_matrix(S_train, cue_obj_train.C)
        verbose && println("G type: $(typeof(G_train))")
        verbose && println("G size: $(size(G_train))")

        verbose && println("Make F matrix...")
        F_train = make_transform_matrix(cue_obj_train.C, S_train)
        verbose && println("F type: $(typeof(F_train))")
        verbose && println("F size: $(size(F_train))")

    elseif learning_mode == :pyndl
        verbose && println("Preprocessing pyndl text...")
        preprocess_ndl(
            joinpath(output_dir_path, "$(data_prefix)_train.csv"),
            joinpath(output_dir_path, "$(data_prefix)_train.tab.gz"),
            grams = grams,
            n_grams_target_col = n_grams_target_col,
            n_grams_tokenized = n_grams_tokenized,
            n_grams_sep_token = n_grams_sep_token,
            n_grams_keep_sep = n_grams_keep_sep,
            n_features_columns = n_features_columns,
        )

        verbose && println("Using pyndl make G matrix...")
        pws = pyndl(
            joinpath(output_dir_path, "$(data_prefix)_train.tab.gz"),
            alpha = alpha,
            betas = betas,
        )

        G_train = pws.weight
        verbose && println("G type: $(typeof(G_train))")
        verbose && println("G size: $(size(G_train))")

        verbose && println("Making cue matrix...")
        cue_obj_train = make_cue_matrix(
            data_train,
            pws,
            grams = grams,
            target_col = n_grams_target_col,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            keep_sep = n_grams_keep_sep,
            verbose = verbose,
        )
        cue_obj_val = make_cue_matrix(
            data_val,
            cue_obj_train,
            grams = grams,
            target_col = n_grams_target_col,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            keep_sep = n_grams_keep_sep,
            verbose = verbose,
        )
        verbose && println("training type: $(typeof(cue_obj_train.C))")
        verbose && println("val type: $(typeof(cue_obj_val.C))")
        verbose && println("training size: $(size(cue_obj_train.C))")
        verbose && println("val size: $(size(cue_obj_val.C))")
        verbose && println()

        n_features = size(cue_obj_train.C, 2)

        verbose && println("Making S matrix...")
        S_train, S_val =
            make_S_matrix(data_train, data_val, pws, n_features_columns)
        verbose && println("training type: $(typeof(S_train))")
        verbose && println("val type: $(typeof(S_val))")
        verbose && println("training size: $(size(S_train))")
        verbose && println("val size: $(size(S_val))")
        verbose && println()

        verbose && println("Make F matrix...")
        F_train = make_transform_matrix(cue_obj_train.C, S_train)
        verbose && println("F type: $(typeof(F_train))")
        verbose && println("F size: $(size(F_train))")
    elseif learning_mode == :wh
        verbose && println("Making cue matrix...")
        cue_obj_train = make_cue_matrix(
            data_train,
            grams = grams,
            target_col = n_grams_target_col,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            keep_sep = n_grams_keep_sep,
            verbose = verbose,
        )
        cue_obj_val = make_cue_matrix(
            data_val,
            cue_obj_train,
            grams = grams,
            target_col = n_grams_target_col,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            keep_sep = n_grams_keep_sep,
            verbose = verbose,
        )
        verbose && println("training type: $(typeof(cue_obj_train.C))")
        verbose && println("val type: $(typeof(cue_obj_val.C))")
        verbose && println("training size: $(size(cue_obj_train.C))")
        verbose && println("val size: $(size(cue_obj_val.C))")
        verbose && println()

        n_features = size(cue_obj_train.C, 2)

        verbose && println("Making S matrix...")
        S_train, S_val = make_S_matrix(
            data_train,
            data_val,
            n_features_base,
            n_features_inflections,
            ncol = n_features,
        )
        verbose && println("training type: $(typeof(S_train))")
        verbose && println("val type: $(typeof(S_val))")
        verbose && println("training size: $(size(S_train))")
        verbose && println("val size: $(size(S_val))")
        verbose && println()

        verbose && println("Using wh make G matrix...")
        G_train = wh_learn(
            cue_obj_train.C,
            S_train,
            eta = eta,
            n_epochs = n_epochs,
            verbose = verbose,
        )
        verbose && println("G type: $(typeof(G_train))")
        verbose && println("G size: $(size(G_train))")

        verbose && println("Make F matrix...")
        F_train = make_transform_matrix(cue_obj_train.C, S_train)
        verbose && println("F type: $(typeof(F_train))")
        verbose && println("F size: $(size(F_train))")

    else
        throw(ArgumentError("learning_mode is incorrect, using :cholesky, :wh or :pyndl"))
    end

    verbose && println("Calculating Chat...")
    Chat_train = convert(Matrix{Float64}, S_train) * G_train
    Chat_val = convert(Matrix{Float64}, S_val) * G_train

    verbose && println("Calculating Shat...")
    Shat_train = convert(Matrix{Float64}, cue_obj_train.C) * F_train
    Shat_val = convert(Matrix{Float64}, cue_obj_val.C) * F_train

    verbose && println("Calculating A...")
    if is_full_A
        A = make_adjacency_matrix(
            cue_obj_train.i2f,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            verbose = verbose,
        )
    else
        A = cue_obj_train.A
    end

    if isnothing(max_t)
        max_t = cal_max_timestep(
            data_train,
            data_val,
            n_grams_target_col,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
        )
    end

    verbose && println("Finding paths...")
    if path_method == :learn_paths
        if !val_only
            res_train, gpi_train = learn_paths(
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
                grams = grams,
                threshold = train_threshold,
                tokenized = n_grams_tokenized,
                sep_token = n_grams_sep_token,
                keep_sep = n_grams_keep_sep,
                target_col = n_grams_target_col,
                issparse = issparse,
                sparse_ratio = sparse_ratio,
                verbose = verbose,
            )
        end

        res_val, gpi_val = learn_paths(
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
            grams = grams,
            threshold = val_threshold,
            is_tolerant = val_is_tolerant,
            tolerance = val_tolerance,
            max_tolerance = val_max_tolerance,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            keep_sep = n_grams_keep_sep,
            target_col = n_grams_target_col,
            issparse = issparse,
            sparse_ratio = sparse_ratio,
            verbose = verbose,
        )
    else
        if !val_only
            res_train = build_paths(
                data_train,
                cue_obj_train.C,
                S_train,
                F_train,
                Chat_train,
                A,
                cue_obj_train.i2f,
                cue_obj_train.gold_ind,
                max_t = max_t,
                n_neighbors = train_n_neighbors,
                verbose = verbose,
            )
        end

        res_val = build_paths(
            data_val,
            cue_obj_train.C,
            S_val,
            F_train,
            Chat_val,
            A,
            cue_obj_train.i2f,
            cue_obj_train.gold_ind,
            max_t = max_t,
            n_neighbors = val_n_neighbors,
            verbose = verbose,
        )
    end

    verbose && println("Evaluate acc...")
    if !val_only
        acc_train =
            eval_acc(res_train, cue_obj_train.gold_ind, verbose = verbose)
    end
    acc_val = eval_acc(res_val, cue_obj_val.gold_ind, verbose = verbose)
    if !val_only
        acc_train_loose =
            eval_acc_loose(res_train, cue_obj_train.gold_ind, verbose = verbose)
    end
    acc_val_loose =
        eval_acc_loose(res_val, cue_obj_val.gold_ind, verbose = verbose)
    !val_only && println(log_io, "Acc for train: $acc_train")
    println(log_io, "Acc for val: $acc_val")
    !val_only && println(log_io, "Acc for train loose: $acc_train_loose")
    println(log_io, "Acc for val loose: $acc_val_loose")

    if !val_only
        write2csv(
            res_train,
            data_train,
            cue_obj_train,
            cue_obj_train,
            "res_$(csv_prefix)_train.csv",
            grams = grams,
            tokenized = n_grams_tokenized,
            sep_token = n_grams_sep_token,
            start_end_token = start_end_token,
            output_sep_token = n_grams_sep_token,
            path_sep_token = path_sep_token,
            root_dir = root_dir,
            output_dir = csv_dir,
            target_col = n_grams_target_col,
        )
    end

    write2csv(
        res_val,
        data_val,
        cue_obj_train,
        cue_obj_val,
        "res_$(csv_prefix)_val.csv",
        grams = grams,
        tokenized = n_grams_tokenized,
        sep_token = n_grams_sep_token,
        start_end_token = start_end_token,
        output_sep_token = n_grams_sep_token,
        path_sep_token = path_sep_token,
        root_dir = root_dir,
        output_dir = csv_dir,
        target_col = n_grams_target_col,
    )

    nothing
end
