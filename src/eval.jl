"""
A structure that stores information about comprehension accuracy.
"""
struct Comp_Acc_Struct
  dfr::DataFrame
  acc::Float64
  err::Array
end

"""
    accuracy_comprehension(::Matrix, ::Matrix) -> ::Comp_Acc_Struct

Evaluate the comprehension accuracy.

...
# Obligatory Arguments
- `S::Matrix`: the S matrix
- `Shat::Matrix`: the Shat matrix
- `data::DataFrame`: the dataset

# Optional Arguments
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `base::Vector=["Lexeme"]`: base features
- `inflections::Union{Nothing, Vector}=nothing`: inflectional features

# Examples
```julia
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
    eval_SC(Union{SparseMatrixCSC, Matrix}, Union{SparseMatrixCSC, Matrix}) -> ::Float64

Quickly evaluate Shat and Chat matrices.

...
# Obligatory Arguments
- `SChat::Union{SparseMatrixCSC, Matrix}`: the Chat or Shat matrix
- `SC::Union{SparseMatrixCSC, Matrix}`: the C or S matrix

# Optional Arguments
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
  v = [(i[1]==i[2] || rSC[i[1],i[1]]==rSC[i]) ? 1 : 0 for i in argmax(rSC, dims=2)]
  sum(v)/length(v)
end

"""
    eval_manual(::Array, ::DataFrame, ::Dict) -> ::Nothing

Evaluate the results from `build_paths` and `learn_paths` manually.
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
    eval_acc(::Array, ::Array) -> ::Float64

Evaluate the accuracy of the results from `learn_paths` or `build_paths`.

...
# Obligatory Arguments
- `res::Array`: the results from `learn_paths` or `build_paths`
- `gold_inds::Array`: the gold paths' indices

# Optional Arguments
- `verbose::Bool=false`: if true, more information is printed

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
  if verbose
    pb = Progress(total)
  end

  for i in iter
    if isassigned(res[i], 1) && iscorrect(gold_inds[i], res[i][1].ngrams_ind)
      acc += 1
    end
    if verbose
      ProgressMeter.next!(pb)
    end
  end

  acc/total
end

"""
    eval_acc_loose(::Array, ::Array) -> ::Float64

Evaluate the accuracy of the results from `learn_paths` or `build_paths`, if 
one of the candidates is correct, then we take it as correct. This reflects how 
many paths were found but could not be recognized as the best path.

...
# Obligatory Arguments
- `res::Array`: the results from `learn_paths` or `build_paths`
- `gold_inds::Array`: the gold paths' indices

# Optional Arguments
- `verbose::Bool=false`: if true, more information is printed

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
  if verbose
    pb = Progress(total)
  end

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
    if verbose
      ProgressMeter.next!(pb)
    end
  end

  acc/total
end

"""
    extract_gpi(::Vector{Gold_Path_Info_Struct}, ::Float64, ::Float64) -> ::Array

Extract, using gold paths' information, how many n-grams for a gold 
path are below the threshold but above the tolerance.
"""
function extract_gpi(
  gpi::Vector{Gold_Path_Info_Struct},
  threshold=0.1::Float64,
  tolerance=(-1000.0)::Float64
)::Array
  c = [count(x->threshold>=x>tolerance, g.ngrams_ind_support) for g in gpi]

  t_c = length(c)

  summary_c = [(x,count(cnt->cnt==x, c),count(cnt->cnt==x, c)/t_c) for x in sort(unique(c))]

  summary_c
end