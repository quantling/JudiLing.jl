struct SplitDataException <: Exception
  msg::String
end

function train_val_split(
  data_path::String,
  output_dir_path::String,
  n_features_columns::Vector;
  data_prefix="data"::String,
  max_test_data=nothing::Union{Nothing, Int64},
  split_max_ratio=0.2::Float64,
  n_grams_target_col=:PhonWord::Symbol,
  n_grams_tokenized=false::Bool,
  n_grams_sep_token=nothing::Union{Nothing, String, Char},
  grams=3::Int64,
  n_grams_keep_sep=false::Bool,
  start_end_token="#"::Union{String, Char},
  random_seed=314::Int64,
  verbose=false::Bool
  )::Nothing

  # read csv
  utterances = CSV.DataFrame!(CSV.File(data_path))

  # shuffle data
  rng = MersenneTwister(random_seed)
  utterances = utterances[shuffle(rng, 1:size(utterances, 1)),:]

  if !isnothing(max_test_data)
    utterances=utterances[1:max_test_data,:]
  end

  num_utterances = size(utterances, 1)

  init_num_train = round(Int64, num_utterances*0.4)
  max_num_val = round(Int64, num_utterances*split_max_ratio)
  utterances_train = utterances[1:init_num_train,:]

  if n_grams_tokenized && !isnothing(n_grams_sep_token)
    tokens = split.(utterances_train[:, n_grams_target_col], n_grams_sep_token)
  else
    tokens = split.(utterances_train[:, n_grams_target_col], "")
  end

  verbose && println("Calculating utterances_train_ngrams ...")
  utterances_train_ngrams = String[]

  for i in 1:init_num_train
    push!(utterances_train_ngrams, make_ngrams(
      tokens[i], grams, n_grams_keep_sep,
      n_grams_sep_token, start_end_token)...)
  end
  utterances_train_ngrams = unique(utterances_train_ngrams)
  utterances_train_features = collect_features(
    utterances[1:init_num_train,:],
    n_features_columns
  )
  utterances_val = DataFrame()

  perform_split(utterances[init_num_train+1:end,:],
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
    verbose=verbose
    )

  if size(utterances_train, 1) <= 0 || size(utterances_val, 1) <= 0
    throw(SplitDataException("Could not split data automaticly"))
  end

  mkpath(output_dir_path)

  CSV.write(joinpath(output_dir_path, "$(data_prefix)_train.csv"), utterances_train, quotestrings=true)
  CSV.write(joinpath(output_dir_path, "$(data_prefix)_val.csv"), utterances_val, quotestrings=true)

  verbose && begin
    println("Successfully split data into $(size(utterances_train, 1)) training data and $(size(utterances_val, 1)) validation data")
  end

  nothing
end

function train_val_split(
  data_path::String,
  output_dir_path::String;
  data_prefix="data"::String,
  split_max_ratio=0.2::Float64,
  n_grams_target_col=:Word_n_grams::Symbol,
  n_grams_tokenized=false::Bool,
  n_grams_sep_token=nothing::Union{Nothing, String, Char},
  n_features_col_name=:CommunicativeIntention::Symbol,
  n_features_tokenized=false::Bool,
  n_features_sep_token=nothing::Union{Nothing, String, Char},
  random_seed=314::Int64,
  verbose=false::Bool
  )::Nothing

  # read csv
  utterances = CSV.DataFrame!(CSV.File(data_path))
  num_utterances = size(utterances, 1)

  # shuffle data
  rng = MersenneTwister(random_seed)
  utterances = utterances[shuffle(rng, 1:size(utterances, 1)), :]
  
  init_num_train = round(Int64, num_utterances*0.4)
  max_num_val = round(Int64, num_utterances*split_max_ratio)
  utterances_train = utterances[1:init_num_train,:]
  utterances_train_ngrams = unique(
    [ngram for i in 1:init_num_train
      for ngram in split_features(
        utterances[i,:], n_grams_target_col, n_grams_tokenized, n_grams_sep_token)])
  utterances_train_features = unique(
    [feature for i in 1:init_num_train
      for feature in split_features(
        utterances[i,:], n_features_col_name, n_features_tokenized, n_grams_sep_token)])
  utterances_val = DataFrame()

  perform_split(utterances[init_num_train+1:end,:],
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
    verbose=verbose
    )

  if size(utterances_train, 1) <= 0 || size(utterances_val, 1) <= 0
    throw(SplitDataException("Could not split data automaticly"))
  end

  mkpath(output_dir_path)

  CSV.write(joinpath(output_dir_path, "$(data_prefix)_train.csv"), utterances_train)
  CSV.write(joinpath(output_dir_path, "$(data_prefix)_val.csv"), utterances_val)

  verbose && begin
    println("Successfully split data into $(size(utterances_train, 1)) training data and $(size(utterances_val, 1)) validation data")
    println()
  end
  nothing
end

function split_features(
  datarow::DataFrames.DataFrameRow,
  col_name::Symbol,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char}
  )::Vector
  if tokenized && !isnothing(sep_token)
    split(datarow[col_name], sep_token)
  else
    split(datarow[col_name], "")
  end
end

function collect_features(
  data::DataFrame,
  n_features_columns::Vector
  )
  features = String[]
  for c in n_features_columns
    push!(features, unique(data[:,c])...)
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
  verbose=false::Bool
  )

  iter = 1:size(utterances, 1)
  verbose && begin iter = tqdm(iter) end
  for i in iter
    ngrams = split_features(
      utterances[i,:],
      n_grams_target_col,
      n_grams_tokenized,
      n_grams_sep_token)
    features = split_features(
      utterances[i,:],
      n_features_col_name,
      n_features_tokenized,
      n_features_sep_token)

    # to check whether
    if !any(x->!any(y->y==x, utterances_train_ngrams), ngrams) &&
      !any(x->!any(y->y==x, utterances_train_features), features) &&
      size(utterances_val, 1) < max_num_val
      if !isnothing(utterances_val)
        push!(utterances_val, utterances[i,:])
      else
        utterances_val = DataFrames.DataFrame(utterances[i,:])
      end
    else
      push!(utterances_train, utterances[i,:])
      utterances_train_ngrams = unique(push!(utterances_train_ngrams, ngrams...))
      utterances_train_features = unique(push!(utterances_train_features, features...))
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
  verbose=false::Bool
  )

  if n_grams_tokenized && !isnothing(n_grams_sep_token)
    tokens = split.(utterances[:, n_grams_target_col], n_grams_sep_token)
  else
    tokens = split.(utterances[:, n_grams_target_col], "")
  end

  iter = 1:size(utterances, 1)
  verbose && begin iter = tqdm(iter) end
  for i in iter
    ngrams = make_ngrams(
      tokens[i], grams, n_grams_keep_sep,
      n_grams_sep_token, start_end_token)
    features = unique(utterances[i,n_features_columns])

    # to check whether
    if !any(x->!any(y->y==x, utterances_train_ngrams), ngrams) &&
      !any(x->!any(y->y==x, utterances_train_features), features) &&
      size(utterances_val, 1) < max_num_val
      if !isnothing(utterances_val)
        push!(utterances_val, utterances[i,:])
      else
        utterances_val = DataFrames.DataFrame(utterances[i,:])
      end
    else
      push!(utterances_train, utterances[i,:])
      utterances_train_ngrams = unique(push!(utterances_train_ngrams, ngrams...))
      utterances_train_features = unique(push!(utterances_train_features, features...))
    end
  end
end

function preprocess_ndl end

function preprocess_ndl(
  data_path::String,
  save_path::String;
  grams=3::Integer,
  n_grams_target_col=:Word::Union{String, Symbol},
  n_grams_tokenized=false::Bool,
  n_grams_sep_token="-"::String,
  n_grams_keep_sep=false::Bool,
  n_features_columns=["Lexeme","Person","Number","Tense","Voice","Mood"]::Vector
  )::Nothing

  # read csv
  data = CSV.DataFrame!(CSV.File(data_path))

  io = GZip.open(joinpath(save_path), "w")

  cues, outcomes = make_cue_outcome(
    data,
    grams,
    n_grams_target_col,
    n_grams_tokenized,
    n_grams_sep_token,
    n_grams_keep_sep,
    n_features_columns)

  # write header
  write(io, "cues\toutcomes\n")

  for i in 1:size(data, 1)
    write(io, "$(cues[i])\t$(outcomes[i])\n")
  end

  close(io)

  nothing
end

function make_cue_outcome(
  data::DataFrame,
  grams::Int64,
  n_grams_target_col::Union{String, Symbol},
  n_grams_tokenized::Bool,
  n_grams_sep_token::String,
  n_grams_keep_sep::Bool,
  n_features_columns::Vector
  )::Tuple{Vector{String}, Vector{String}}

  n_rows = size(data, 1)

  cues = Vector{String}(undef, n_rows)
  outcomes = Vector{String}(undef, n_rows)

  # split tokens from words or other columns
    if n_grams_tokenized && !isnothing(n_grams_sep_token)
      tokens = split.(data[:, n_grams_target_col], n_grams_sep_token)
    else
      tokens = split.(data[:, n_grams_target_col], "")
    end

  for i in 1:n_rows
    cues[i] = join(make_ngrams(tokens[i], grams, n_grams_keep_sep, n_grams_sep_token, "#"), "_")
    outcomes[i] = join(data[i,n_features_columns], "_")
  end

  cues, outcomes
end
