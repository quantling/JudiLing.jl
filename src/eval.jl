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

Evaluate the outputs from shuo function or hua function

...
# Arguments
- `verbose::Bool=false`: if verbose, more information prints out

# Examples
```julia
#after you had results from shuo or hua
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

Evaluate the outputs from shuo function or hua function, if one of the candidates
are correct, then we take it correct. This reflects how many paths we could found
but we could not recogni

...
# Arguments
- `verbose::Bool=false`: if verbose, more information prints out

# Examples
```julia
#after you had results from shuo or hua
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