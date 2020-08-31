"""
a struct that store info about comprehension accuracy
"""
struct Comp_Acc_Struct
  dfr::DataFrame
  acc::Float64
  err::Array
end

"""
  accuracy_comprehension(::Matrix, ::Matrix)

Evaluate the comprehension accuracy

...
# Arguments
- `target_col::Union{String, Symbol}=:Words`: target column name
- `base::Vector=["Lexeme"]`: base features
- `inflections::Union{Nothing, Vector}=nothing`: inflective features

# Examples
```julia
latin_train = CSV.DataFrame!(CSV.File(joinpath("data", "latin_mini.csv")))
cue_obj_train = JudiLing.make_cue_matrix(
  latin_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

latin_val = latin_train[101:150,:]
cue_obj_val = JudiLing.make_cue_matrix(
  latin_val,
  cue_obj_train,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )

n_features = size(cue_obj_train.C, 2)

S_train, S_val = JudiLing.make_S_matrix(
  latin_train,
  latin_val,
  ["Lexeme"],
  ["Person","Number","Tense","Voice","Mood"],
  ncol=n_features)

G_train = JudiLing.make_transform_matrix(S_train, cue_obj_train.C)

Chat_train = S_train * G_train
Chat_val = S_val * G_train
JudiLing.eval_SC(cue_obj_train.C, Chat_train)
JudiLing.eval_SC(cue_obj_val.C, Chat_val)

F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train
JudiLing.eval_SC(S_train, Shat_train)
JudiLing.eval_SC(S_val, Shat_val)

accuracy_comprehension(
  S_train,
  Shat_train,
  latin_val,
  target_col=:Words,
  base=["Lexeme"],
  inflections=["Person","Number","Tense","Voice","Mood"]
  )

accuracy_comprehension(
  S_val,
  Shat_val,
  latin_train,
  target_col=:Words,
  base=["Lexeme"],
  inflections=["Person","Number","Tense","Voice","Mood"]
  )
```
...
"""
function accuracy_comprehension(
  S::Matrix,
  Shat::Matrix,
  data::DataFrame;
  target_col=:Words::Union{String, Symbol},
  base=["Lexeme"]::Vector,
  inflections=nothing::Union{Nothing, Vector}
  )::Comp_Acc_Struct

  corMat = cor(Shat, S, dims=2)
  top_index = [i[2] for i in argmax(corMat, dims=2)]

  dfr = DataFrame()
  dfr.target = data[:,target_col]
  dfr.form = vec([data[i,target_col] for i in top_index])
  dfr.r = vec([corMat[index, value] for (index, value) in enumerate(top_index)])
  dfr.r_target = corMat[diagind(corMat)]
  dfr.correct = [dfr.target[i]==dfr.form[i] for i in 1:size(dfr, 1)]

  if !isnothing(inflections)
    all_features = vcat(base, inflections)
  else
    all_features = base
  end

  for f in all_features
    dfr.tmp = vec([data[index, f]==data[value, f] for (index, value) in enumerate(top_index)])
    rename!(dfr, "tmp" => f)
  end

  acc = sum(dfr[:,"correct"])/size(dfr,1)
  err = findall(x->x!=1, dfr[:,"correct"])

  Comp_Acc_Struct(dfr, acc, err)
end

"""
  eval_SC(Union{SparseMatrixCSC, Matrix}, Union{SparseMatrixCSC, Matrix})

evaluate S and Shat or C and Chat

...
# Examples
```julia
#after you had Shat and Chat
eval_SC(cue_obj_train.C, Chat_train)
eval_SC(cue_obj_val.C, Chat_val)
eval_SC(S_train, Shat_train)
eval_SC(S_val, Shat_val)
```
...
"""
function eval_SC(
  SChat::Union{SparseMatrixCSC, Matrix},
  SC::Union{SparseMatrixCSC, Matrix}
  )::Float64

  rSC = cor(convert(Matrix{Float64}, SChat), convert(Matrix{Float64}, SC), dims=2)
  v = [i[1]==i[2] ? 1 : 0 for i in argmax(rSC, dims=2)]
  sum(v)/length(v)
end

"""
Manually check the results
"""
function eval_manual(
  res::Array,
  data::DataFrame,
  i2f::Dict;
  s=1::Int64,
  e=nothing::Union{Nothing, Int64},
  grams=3::Int64,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  start_end_token="#"::Union{String, Char},
  verbose=false::Bool,
  identifier="Identifier"::String,
  output_sep_token=""::Union{String, Char}
  )::Nothing

  # is end is not specified then it is the end of dataset
  if isnothing(e)
    e = length(res)
  end

  for i in s:e
    verbose && println("="^20)
    verbose && println("utterance $i: ")
    verbose && println(data[i, identifier])
    verbose && println("predictions: ")
    verbose && begin display_pred(
        res[i],
        i2f,
        grams,
        tokenized,
        sep_token,
        start_end_token,
        output_sep_token)
      end
  end
  return
end


"""
  eval_acc(::Array, ::Array)

Evaluate the outputs from learn_paths function or build_paths function

...
# Arguments
- `verbose::Bool=false`: if verbose, more information prints out

# Examples
```julia
#after you had results from learn_paths or build_paths
acc_train = JudiLing.eval_acc(
  res_train,
  cue_obj_train.gold_ind,
  verbose=false
)
acc_val = JudiLing.eval_acc(
  res_val,
  cue_obj_val.gold_ind,
  verbose=false
)
```
...
"""
function eval_acc(
    res::Array,
    gold_inds::Array;
    verbose=false::Bool
  )::Float64

  total = length(res)
  acc = 0

  iter = 1:total
  verbose && begin iter = tqdm(iter) end

  for i in iter
    if isassigned(res[i], 1) && iscorrect(gold_inds[i], res[i][1].ngrams_ind)
      acc += 1
    end
  end

  acc/total
end

"""
  eval_acc_loose(::Array, ::Array)

Evaluate the outputs from learn_paths function or build_paths function, if one of the candidates
are correct, then we take it correct. This reflects how many paths we could found
but we could not recogni

...
# Arguments
- `verbose::Bool=false`: if verbose, more information prints out

# Examples
```julia
#after you had results from learn_paths or build_paths
acc_train_loose = JudiLing.eval_acc_loose(
  res_train,
  cue_obj_train.gold_ind,
  verbose=false
)
acc_val_loose = JudiLing.eval_acc_loose(
  res_val,
  cue_obj_val.gold_ind,
  verbose=false
)
```
...
"""
function eval_acc_loose(
    res::Array,
    gold_inds::Array;
    verbose=false::Bool
  )::Float64

  total = length(res)
  acc = 0

  iter = 1:total
  verbose && begin iter = tqdm(iter) end

  for i in iter
    is_correct = false
    for j in 1:length(res[i])
      if isassigned(res[i], 1) && iscorrect(gold_inds[i], res[i][j].ngrams_ind)
        is_correct = true
      end
    end
    if is_correct
      acc += 1
    end
  end

  acc/total
end

"""
Evaluate gold path info
"""
function eval_gpi(
  gpi::Vector{Gold_Path_Info_Struct},
  threshold=0.1::Float64,
  tolerance=(-1000.0)::Float64
)
  c = [count(x->threshold>=x>tolerance, g.ngrams_ind_support) for g in gpi]

  t_c = length(c)

  summary_c = [(x,count(cnt->cnt==x, c),count(cnt->cnt==x, c)/t_c) for x in sort(unique(c))]

  summary_c
end