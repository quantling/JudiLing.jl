"""
store paths information found by learn_paths or build_paths function
"""
struct Result_Path_Info_Struct
  ngrams_ind::Array
  num_tolerance::Integer
  support::AbstractFloat
end

"""
store gold paths information including indices and indices support and total support
it can be used to evaluate how low the threshold is set in order to find the correct paths
"""
struct Gold_Path_Info_Struct
  ngrams_ind::Array
  ngrams_ind_support::Array
  support::AbstractFloat
end

"""
learn_paths function takes each timestep individually and calculate Yt_hat respectively,
"""
function learn_paths end

"""
build_paths function is shortcut algorithms that only takes n-grams that closed to the
validation data
"""
function build_paths end

"""
  learn_paths(::DataFrame,::DataFrame,::SparseMatrixCSC,::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Matrix,::SparseMatrixCSC,::Dict)

learn_paths function takes each timestep individually and calculate Yt_hat respectively,

...
# Arguments
- `gold_ind::Union{Nothing, Vector}=nothing`: for in gold_path_info mode
- `Shat_val::Union{Nothing, Matrix}=nothing`: for gold_path_info mode
- `check_gold_path::Bool=false`: if turn on gold_path_info mode
- `max_t::Integer=15`: maximum timestep
- `max_can::Integer=10`: maximum candidates when output
- `threshold::AbstractFloat=0.1`: for each timestep, only grams greater than threshold will be selected
- `is_tolerant::Bool=false`: if in tolerant mode, path allows limited nodes under threshold but greater than tolerance
- `tolerance::AbstractFloat=(-1000.0)`: in tolerant mode, only nodes greater than tolerance and lesser than threshold will be selected
- `max_tolerance::Integer=4`: maximum numbers of nodes allowed in a path
- `grams::Integer=3`: n-grams
- `tokenized::Bool=false`: whether tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: seperate token
- `keep_sep::Bool=false`: whether keep seperaters in grams
- `target_col::Union{String, :Symbol}=:Words`: word column names
- `issparse::Symbol=:auto`: mt matrix output format mode
- `verbose::Bool=false`: if verbose, more information will be printed out

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

F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

A = cue_obj_train.A

max_t = JudiLing.cal_max_timestep(latin_train, latin_val, :Word)

res_train, gpi_train = JudiLing.learn_paths(
  latin_train,
  latin_train,
  cue_obj_train.C,
  S_train,
  F_train,
  Chat_train,
  A,
  cue_obj_train.i2f,
  gold_ind=cue_obj_train.gold_ind,
  Shat_val=Shat_train,
  check_gold_path=true,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.1,
  tokenized=false,
  sep_token="_",
  keep_sep=false,
  target_col=:Word,
  issparse=:dense,
  verbose=false)

res_val, gpi_val = JudiLing.learn_paths(
  latin_train,
  latin_val,
  cue_obj_train.C,
  S_val,
  F_train,
  Chat_val,
  A,
  cue_obj_train.i2f,
  gold_ind=cue_obj_val.gold_ind,
  Shat_val=Shat_val,
  check_gold_path=true,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.1,
  is_tolerant=true,
  tolerance=0.1,
  max_tolerance=0,
  tokenized=false,
  sep_token="-",
  keep_sep=false,
  target_col=:Word,
  issparse=:dense,
  verbose=false)
```
...
"""
function learn_paths(
  data_train::DataFrame,
  data_val::DataFrame,
  C_train::SparseMatrixCSC,
  S_val::Union{SparseMatrixCSC, Matrix},
  F_train::Union{SparseMatrixCSC, Matrix},
  Chat_val::Matrix,
  Al::SparseMatrixCSC,
  i2f::Dict;
  gold_ind=nothing::Union{Nothing, Vector},
  Shat_val=nothing::Union{Nothing, Matrix},
  check_gold_path=false::Bool,
  max_t=15::Integer,
  max_can=10::Integer,
  threshold=0.1::AbstractFloat,
  is_tolerant=false::Bool,
  tolerance=(-1000.0)::AbstractFloat,
  max_tolerance=4::Integer,
  grams=3::Integer,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  keep_sep=false::Bool,
  target_col="Words"::String,
  issparse=:auto::Symbol,
  verbose=false::Bool
  )::Union{Tuple{Vector{Vector{Result_Path_Info_Struct}}, Vector{Gold_Path_Info_Struct}}, Vector{Vector{Result_Path_Info_Struct}}}

  # initialize queues for storing paths
  n_val = size(data_val, 1)
  # working_q = Array{Queue{Array{Integer,1}},1}(undef, n_val)
  working_q = Vector{Queue{Tuple{Vector{Integer}, Integer}}}(undef, n_val)
  # res = Array{Array{Array{Integer}},1}(undef, n_val)
  res = Vector{Vector{Tuple{Vector{Integer}, Integer}}}(undef, n_val)
  for j in 1:n_val
    res[j] = Tuple{Vector{Integer}, Integer}[]
  end

  # # initialize gold_path_info supports
  if check_gold_path && !isnothing(gold_ind)
    gold_path_info_supports = Vector{Vector{AbstractFloat}}(undef, size(data_val, 1))
    for i in 1:n_val
      gi = gold_ind[i]
      gold_path_info_supports[i] = Vector{AbstractFloat}(undef, length(gi))
    end
  end

  verbose && println("Making fac C")
  fac_C_train = make_transform_fac(C_train)

  for i in 1:max_t
    verbose && println("="^10)
    verbose && println("Timestep $i")
    verbose && i>=2 && println("average $(mean(length.(working_q))) of paths currently")

    verbose && println("Calculating Yt...")
    Yt_train = make_Yt_matrix(
      i,
      data_train,
      grams=grams,
      target_col=target_col,
      tokenized=tokenized,
      sep_token=sep_token,
      keep_sep=keep_sep)

    verbose && println("Calculating Mt...")
    Mt_train = make_transform_matrix(fac_C_train, C_train, Yt_train, output_format=issparse, verbose=verbose)
    # Mt = sparse(Mt)
    # verbose && println("Sparsity of Mt: $(length(Mt.nzval)/Mt.m/Mt.n)")

    verbose && println("Calculating Ythat...")
    # Ythat_val = Matrix(Chat_val)*Matrix(Mt_train)
    Ythat_val = Chat_val*Mt_train
    if is_truly_sparse(Ythat_val, verbose=verbose)
      Ythat_val = sparse(Ythat_val)
    end
    # Ythat = sparse(Ythat)
    # verbose && println("Sparsity of Ythat: $(length(Ythat.nzval)/Ythat.m/Ythat.n)")

    # collect supports for gold path each timestep
    if check_gold_path && !isnothing(gold_ind)
      for j in 1:size(data_val, 1)
        gi = gold_ind[j]
        if i <= length(gi)
          gold_path_info_supports[j][i] = Ythat_val[j,gi[i]]
        end
      end
    end

    verbose && println("Finding paths...")
    iter = 1:n_val
    verbose && begin iter = tqdm(iter) end
    for j in iter
      # collect all n-grams which has greater support than the threshold
      candidates_t = findall(x->x>threshold, Ythat_val[j,:])
      candidates_t_tlr = findall(x->x>tolerance&&x<=threshold, Ythat_val[j,:])

      # for timestep 2 and after 2
      if isassigned(working_q, j)
        tmp_working_q = Queue{Tuple{Vector{Integer},Integer}}()
        while !isempty(working_q[j])
          a = dequeue!(working_q[j]) ## a = [11] Al[11,5] == 1 # candidates = [1, 5, 7]

          for c in candidates_t ## c = 5 # a = [11, 1, 5, 7] # a = [11, 1] [11, 5] [11, 7]
            # if a n-grams is attachable then append it
            if isattachable(a[1], c, Al)
              a_copy = deepcopy(a[1])
              push!(a_copy, c)
              # if the path is complete then move it to result list
              if iscomplete(a_copy, i2f, tokenized=tokenized, sep_token=sep_token)
                push!(res[j], (a_copy,a[2]))
              else
                # otherwise enqueue it to the next timestep
                enqueue!(tmp_working_q, (a_copy,a[2]))
              end
            end
          end

          if is_tolerant && a[2] < max_tolerance
            for c in candidates_t_tlr ## c = 5 # a = [11, 1, 5, 7] # a = [11, 1] [11, 5] [11, 7]
              # if a n-grams is attachable then append it
              if isattachable(a[1], c, Al)
                a_copy = deepcopy(a[1])
                push!(a_copy, c)
                # if the path is complete then move it to result list
                if iscomplete(a_copy, i2f, tokenized=tokenized, sep_token=sep_token)
                  push!(res[j], (a_copy,a[2]+1))
                else
                  # otherwise enqueue it to the next timestep
                  enqueue!(tmp_working_q, (a_copy,a[2]+1))
                end
              end
            end
          end

        end

        # refresh queue for the next timestep
        working_q[j] = tmp_working_q
      # for timestep 1
      else
        working_q[j] = Queue{Tuple{Vector{Integer},Integer}}()
        for c in candidates_t
          # check whether a n-gram is a start n-gram
          if isstart(c, i2f, tokenized=tokenized, sep_token=sep_token)
            a = Integer[]
            push!(a, c)
            # check whether this n-gram is both start and complete
            if iscomplete(a, i2f, tokenized=tokenized, sep_token=sep_token)
              push!(res[j], (a,0))
            else
              enqueue!(working_q[j], (a,0))
            end
          end
        end

        if is_tolerant && 0 < max_tolerance
          for c in candidates_t_tlr
            # check whether a n-gram is a start n-gram
            if isstart(c, i2f, tokenized=tokenized, sep_token=sep_token)
              a = Integer[]
              push!(a, c)
              # check whether this n-gram is both start and complete
              if iscomplete(a, i2f, tokenized=tokenized, sep_token=sep_token)
                push!(res[j], (a,1))
              else
                enqueue!(working_q[j], (a,1))
              end
            end
          end
        end

      end
    end
  end

  verbose && println("Evaluating paths...")
  res = eval_can(res, S_val, F_train, i2f, max_can, verbose)

  # initialize gold_path_infos
  if check_gold_path && !isnothing(gold_ind)
    gold_path_infos = Vector{Gold_Path_Info_Struct}(undef, size(data_val, 1))

    # calculate all shat correlation with S
    Scors = [cor(Shat_val[i, :], S_val[i, :]) for i in 1:n_val]

    for i in 1:size(data_val, 1)
      gold_path_infos[i] = Gold_Path_Info_Struct(gold_ind[i], gold_path_info_supports[i], Scors[i])
    end
    return res, gold_path_infos
  end

  res
end

"""
  build_paths(::DataFrame,::SparseMatrixCSC,::Union{SparseMatrixCSC, Matrix},::::Union{SparseMatrixCSC, Matrix},::Matrix,::SparseMatrixCSC,::Dict,::Array)

build_paths function is shortcut algorithms that only takes n-grams that closed to the
validation data

...
# Arguments
- `rC::Union{Nothing, Matrix}=nothing`: correlation Matrix of C and Chat, passing it to save computing time
- `max_t::Integer=15`: maximum timestep
- `max_can::Integer=10`: maximum candidates when output
- `n_neighbors::Integer=10`: find indices only in top n neighbors
- `grams::Integer=3`: n-grams
- `tokenized::Bool=false`: whether tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: seperate token
- `target_col::Union{String, :Symbol}=:Words`: word column names
- `verbose::Bool=false`: if verbose, more information will be printed out

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

F_train = JudiLing.make_transform_matrix(cue_obj_train.C, S_train)

Shat_train = cue_obj_train.C * F_train
Shat_val = cue_obj_val.C * F_train

A = cue_obj_train.A

max_t = JudiLing.cal_max_timestep(latin_train, latin_val, :Word)

JudiLing.build_paths(
  latin_train,
  cue_obj_train.C,
  S_train,
  F_train,
  Chat_train,
  A,
  cue_obj_train.i2f,
  cue_obj_train.gold_ind,
  max_t=max_t,
  n_neighbors=10,
  verbose=false
  )

JudiLing.build_paths(
  latin_val,
  cue_obj_train.C,
  S_val,
  F_train,
  Chat_val,
  A,
  cue_obj_train.i2f,
  cue_obj_train.gold_ind,
  max_t=max_t,
  n_neighbors=10,
  verbose=false
  )
```
...
"""
function build_paths(
  data_val::DataFrame,
  C_train::SparseMatrixCSC,
  S_val::Union{SparseMatrixCSC, Matrix},
  F_train::Union{SparseMatrixCSC, Matrix},
  Chat_val::Matrix,
  Al::SparseMatrixCSC,
  i2f::Dict,
  C_train_ind::Array;
  rC=nothing::Union{Nothing, Matrix},
  max_t=15::Integer,
  max_can=10::Integer,
  n_neighbors=10::Integer,
  grams=3::Integer,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  target_col=:Words::Union{String, Symbol},
  verbose=false::Bool
  )::Vector{Vector{Result_Path_Info_Struct}}
  # initialize queues for storing paths
  n_val = size(data_val, 1)
  # working_q = Array{Queue{Array{Integer,1}},1}(undef, n_val)
  # res = Array{Array{Array{Integer}},1}(undef, n_val)
  res = Vector{Vector{Tuple{Vector{Integer}, Integer}}}(undef, n_val)

  for j in 1:n_val
    res[j] = Tuple{Vector{Integer}, Integer}[]
  end

  verbose && println("Finding all top features..")
  # findall features indices for all utterances
  isnothing(rC) && begin rC = cor(Chat_val, Matrix(C_train), dims=2) end
  top_indices = find_top_feature_indices(
    # C_train,
    # Chat_val,
    rC,
    C_train_ind,
    n_neighbors=n_neighbors,
    verbose=verbose)

  # verbose && println("="^10)
  # verbose && println("Timestep $i")

  verbose && println("Finding paths...")
  iter = 1:n_val
  verbose && begin iter = tqdm(iter) end
  for j in iter
    candidates_t = top_indices[j]

    # timestep 1
    working_q = Queue{Array{Integer, 1}}()
    for c in candidates_t
      # check whether a n-gram is a start n-gram
      if isstart(c, i2f, tokenized=tokenized, sep_token=sep_token)
        a = Integer[]
        push!(a, c)
        # check whether this n-gram is both start and complete
        if iscomplete(a, i2f, tokenized=tokenized, sep_token=sep_token)
          push!(res[j], (a,0))
        else
          enqueue!(working_q, a)
        end
      end
    end

    for i in 2:max_t
      tmp_working_q = Queue{Array{Integer, 1}}()
      while !isempty(working_q)
        a = dequeue!(working_q) ## a = [11] Al[11,5] == 1 # candidates = [1, 5, 7]
        for c in candidates_t ## c = 5 # a = [11, 1, 5, 7] # a = [11, 1] [11, 5] [11, 7]
          # if a n-grams is attachable then append it
          if isattachable(a, c, Al)
            a_copy = deepcopy(a)
            push!(a_copy, c)
            # if the path is complete then move it to result list
            if iscomplete(a_copy, i2f, tokenized=tokenized, sep_token=sep_token)
              push!(res[j], (a_copy,0))
            else
              # otherwise enqueue it to the next timestep
              enqueue!(tmp_working_q, a_copy)
            end
          end
        end
      end

      # refresh queue for the next timestep
      working_q = tmp_working_q
    end
  end

  verbose && println("Evaluating paths...")
  eval_can(res, S_val, F_train, i2f, max_can, verbose)
end

"""
  eval_can(::Vector{Vector{Tuple{Vector{Integer}, Integer}}},::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Dict,::Integer,::Bool)

at the end of finding path algorithms, each candidates need to be evaluated
regarding their predicted semantic vectors
"""
function eval_can(
  candidates::Vector{Vector{Tuple{Vector{Integer}, Integer}}},
  S::Union{SparseMatrixCSC, Matrix},
  F::Union{SparseMatrixCSC, Matrix},
  i2f::Dict,
  max_can::Integer,
  verbose=false::Bool
  )::Array{Array{Result_Path_Info_Struct,1},1}

  verbose && println("average $(mean(length.(candidates))) of paths to evaluate")

  res_l = Array{Array{Result_Path_Info_Struct,1},1}(undef, size(S, 1))
  iter = 1:size(S, 1)
  verbose && begin iter = tqdm(iter) end
  for i in iter
    res = Result_Path_Info_Struct[]
    if size(candidates[i], 1) > 0
      for (ci,n) in candidates[i] # ci = [1,3,4]
        Chat = zeros(Integer, length(i2f))
        Chat[ci] .= 1
        Shat = Chat'*F
        Scor = cor(Shat[1, :], S[i, :])
        push!(res, Result_Path_Info_Struct(ci, n, Scor))
      end
    end
    # we collect only top x candidates from the top
    res_l[i] = collect(Iterators.take(sort!(res, by=x->x.support, rev=true), max_can))
  end

  res_l
end

"""
  find_top_feature_indices(::Matrix, ::Array)

find out all indices within the closed top n datarow for a given validation datarow
"""
function find_top_feature_indices(
  # C_train::SparseMatrixCSC,
  # Chat_val::Union{SparseMatrixCSC, Matrix},
  rC::Matrix,
  C_train_ind::Array;
  n_neighbors=10::Integer,
  verbose=false::Bool
  )::Vector{Vector{Integer}}

  # collect num of val data
  n_val = size(rC, 1)

  # calculate correlation matrix
  # rC = cor(Chat_val, Matrix(C_train), dims=2)
  # display(rC)

  # initialize features list for all candidates
  features_all = Vector{Vector{Integer}}(undef, n_val)

  # create iter for tqdm
  verbose && println("finding all n_neighbors features...")
  iter = 1:n_val
  verbose && begin iter = tqdm(iter) end

  # find all features of n_neighbors
  for i in iter
    features = [C_train_ind[j] for j in sortperm(rC[i,:], rev=true)[1:n_neighbors]]
    features_all[i] = unique(collect(Iterators.flatten(features)))
  end

  features_all
end