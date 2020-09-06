"""
check whether the prediction is correct
"""
function iscorrect(
  gold_ind::Vector,
  pred_ind::Vector
  )
  gold_ind == pred_ind
end

"""
display prediction nicely
"""
function display_pred(
  preds::Array,
  i2f::Dict,
  grams::Int64,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  start_end_token="#"::Union{String, Char},
  output_sep_token=""::Union{String, Char}
  )::Nothing

  if length(preds) == 0
    println("No prediction for this utterance")
  else
    for i in 1:length(preds)
      println("Predict $i")
      println("support: $(preds[i].support)")
      pred = translate(preds[i].ngrams_ind, i2f, grams, tokenized, sep_token, start_end_token, output_sep_token)
      println("prediction: $pred")
    end
  end
end

"""
translate indices into words or utterances
"""
function translate(
  ngrams_ind::Vector,
  i2f::Dict,
  grams::Int64,
  tokenized::Bool,
  sep_token::Union{Nothing, String, Char},
  start_end_token::Union{String, Char},
  output_sep_token::Union{String, Char, Nothing}
  )::String

  if isnothing(output_sep_token)
    output_sep_token = ""
  end

  if tokenized && !isnothing(sep_token)
    s1 = join([split(i2f[i], sep_token)[1] for i in ngrams_ind], output_sep_token)
    s2 = join(split(i2f[ngrams_ind[end]], sep_token)[2:end], output_sep_token)
    out = replace(s1*output_sep_token*s2, start_end_token*sep_token => "")
    out = replace(out, sep_token*start_end_token => "")
  else
    s1 = join([split(i2f[i], "")[1] for i in ngrams_ind], output_sep_token)
    s2 = join(split(i2f[ngrams_ind[end]], "")[2:end], output_sep_token)
    out = replace(s1*output_sep_token*s2, start_end_token => "")
  end

  out
end

"""
just append indices together
"""
function translate_path(
  ngrams_ind,
  i2f;
  sep_token=":"
  )::String

  join([i2f[i] for i in ngrams_ind], sep_token)
end

"""
check whether a matrix is truly sparse regardless its format
"""
function is_truly_sparse(
  M::SparseMatrixCSC;
  threshold=0.2::AbstractFloat,
  verbose=false::Bool
  )::Bool
  verbose && println("Sparsity: $(length(M.nzval)/M.m/M.n)")
  return threshold > (length(M.nzval)/M.m/M.n)
end

"""
check whether a matrix is truly sparse regardless its format
"""
function is_truly_sparse(
  M::Matrix;
  threshold=0.2::AbstractFloat,
  verbose=false::Bool
  )::Bool
  M = sparse(M)
  verbose && println("Sparsity: $(length(M.nzval)/M.m/M.n)")
  return threshold > (length(M.nzval)/M.m/M.n)
end

"""
check a gram is attach to another gram
"""
function isattachable(
  a::Array,
  b::Array
  )::Bool

  a[2:end] == b[1:end-1]
end

"""
check a gram is attach to another gram
"""
function isattachable(
  a::Array,
  c::Int64,
  Al::SparseMatrixCSC
  )::Bool

  convert(Bool, Al[a[end],c])
end

"""
check a gram could complete a path
"""
function iscomplete(
  a::Array,
  i2f::Dict;
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char}
  )::Bool

  ngram = i2f[a[end]]

  if tokenized && !isnothing(sep_token)
    last_w = split(ngram, sep_token)[end]
  else
    last_w = split(ngram, "")[end]
  end

  last_w == "#"
end

"""
check a gram could start a path
"""
function isstart(
  c::Int64,
  i2f::Dict;
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char}
  )::Bool

  ngram = i2f[c]

  if tokenized && !isnothing(sep_token)
    start_w = split(ngram, sep_token)[1]
  else
    start_w = split(ngram, "")[1]
  end

  start_w == "#"
end

"""
check wheter a path is in training data or nor
"""
function isnovel(
  gold_ind::Vector,
  pred_ngrams_ind::Array
  )::Bool

  !(pred_ngrams_ind in gold_ind)
end

"""
check whether there are token used in dataset
"""
function check_used_token(
  data::DataFrame,
  target_col::Symbol,
  token::Union{String, Char},
  token_name::String
  )::Nothing
  data_columns = data[:, target_col]
  res = filter(x->!isnothing(findfirst(token, x)) , data_columns)

  if length(res) > 0
    throw(ArgumentError("$token_name \"$token\" is already used in the dataset"))
  end
end

"""
calculate max timestep given training and validation datasets
"""
function cal_max_timestep(
  data_train::DataFrame,
  data_val::DataFrame,
  target_col::Union{Symbol, String};
  tokenized=false::Bool,
  sep_token=""::Union{String, Char, Nothing}
  )::Int64
  words_train = data_train[:, target_col]
  words_val = data_val[:, target_col]

  if tokenized && !isnothing(sep_token)
    max_l_words_train = maximum(x->length(split(x, sep_token)), words_train)
    max_l_words_val = maximum(x->length(split(x, sep_token)), words_val)
  else
    max_l_words_train = maximum(x->length(split(x, "")), words_train)
    max_l_words_val = maximum(x->length(split(x, "")), words_val)
  end

  maximum([max_l_words_train, max_l_words_val])+1
end

"""
calculate max timestep given training dataset
"""
function cal_max_timestep(
  data::DataFrame,
  target_col::Union{Symbol, String};
  tokenized=false::Bool,
  sep_token=""::Union{String, Char}
  )::Int64

  words = data[:, target_col]

  if tokenized && !isnothing(sep_token)
    max_l_words = maximum(x->length(split(x, sep_token)), words)
  else
    max_l_words = maximum(x->length(split(x, "")), words)
  end

  max_l_words+1
end