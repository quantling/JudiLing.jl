"""
Store path information build by learn_paths() or build_paths()
"""
struct Result_Path_Info_Struct
  ngrams_ind::Array
  num_tolerance::Int64
  support::AbstractFloat
end

"""
Store gold path information including indices and indices' support and total 
support. It can be used to evaluate how low the threshold needs to be set in 
order to find the correct paths.
"""
struct Gold_Path_Info_Struct
  ngrams_ind::Array
  ngrams_ind_support::Array
  support::AbstractFloat
end

"""
    learn_paths(::DataFrame, ::DataFrame, ::SparseMatrixCSC, ::Union{SparseMatrixCSC, Matrix}, ::Union{SparseMatrixCSC, Matrix}, ::Matrix, ::SparseMatrixCSC, ::Dict) -> ::Union{Tuple{Vector{Vector{Result_Path_Info_Struct}}, Vector{Gold_Path_Info_Struct}}, Vector{Vector{Result_Path_Info_Struct}}}

One sequence finding algorithm used discrimination learning for the position of triphones.

...
# Obligatory Arguments
- `data::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `C_train::SparseMatrixCSC`: the C matrix for training dataset
- `S_val::Union{SparseMatrixCSC, Matrix}`: the S matrix for validation dataset
- `F_train::Union{SparseMatrixCSC, Matrix}`: the F matrix for training dataset
- `Chat_val::Matrix`: the Chat matrix for validation dataset
- `A::SparseMatrixCSC`: the adjacency matrix
- `i2f::Dict`: the dictionary return features given indices

# Optional Arguments
- `gold_ind::Union{Nothing, Vector}=nothing`: gold paths' indices
- `Shat_val::Union{Nothing, Matrix}=nothing`: the Shat matrix for validation dataset
- `check_gold_path::Bool=false`: if true, return a list of gold paths' information as second output
- `max_t::Int64=15`: maximum timestep
- `max_can::Int64=10`: maximum candidates to keep in the results
- `threshold::AbstractFloat=0.1`:the value set for the support such that if the support of a n-gram is higher than this value, select the n-gram anyway
- `is_tolerant::Bool=false`: if true, select a specified number of n-grams whose supports are below threshold and above tolerance to be added to the path
- `tolerance::AbstractFloat=(-1000.0)`: the value set in tolerant mode such that if the support for a n-gram is inbetween this value and the threshold and the max_tolerance number has not been reached, then allow this n-gram to be added to the path
- `max_tolerance::Int64=4`: maximum number of nodes allowed in a path
- `grams::Int64=3`: the number of grams for cues
- `tokenized::Bool=false`: if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `keep_sep::Bool=false`:if true, keep separators in cues
- `target_col::Union{String, :Symbol}=:Words`: the column name for target strings
- `issparse::Symbol=:auto`: control of whether output of Mt matrix is a dense matrix or a sparse matrix
- `sparse_ratio::Float64=0.2`: the ratio to decide whether a matrix is sparse
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# basic usage with non-tokenized data
res = JudiLing.learn_paths(
  latin,
  latin,
  cue_obj.C,
  S,
  F,
  Chat,
  A,
  cue_obj.i2f,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.1,
  tokenized=false,
  keep_sep=false,
  target_col=:Word,
  verbose=true)

# basic usage with tokenized data
res = JudiLing.learn_paths(
  french,
  french,
  cue_obj.C,
  S,
  F,
  Chat,
  A,
  cue_obj.i2f,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.1,
  tokenized=true,
  sep_token="-",
  keep_sep=true,
  target_col=:Syllables,
  verbose=true)

# basic usage for val data
res_val = JudiLing.learn_paths(
  latin_train,
  latin_val,
  cue_obj_train.C,
  S_val,
  F_train,
  Chat_val,
  A,
  cue_obj_train.i2f,
  max_t=max_t,
  max_can=10,
  grams=3,
  threshold=0.1,
  tokenized=false,
  keep_sep=false,
  target_col=:Word,
  verbose=true)

# turn on tolerance mode
res_val = JudiLing.learn_paths(
  ...
  threshold=0.1,
  is_tolerant=true,
  tolerance=-0.1,
  max_tolerance=4,
  ...)

# control over Sparsity
res_val = JudiLing.learn_paths(
  ...
  issparse=:auto,
  sparse_ratio=0.2,
  ...)

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
  A::SparseMatrixCSC,
  i2f::Dict;
  gold_ind=nothing::Union{Nothing, Vector},
  Shat_val=nothing::Union{Nothing, Matrix},
  check_gold_path=false::Bool,
  max_t=15::Int64,
  max_can=10::Int64,
  threshold=0.1::AbstractFloat,
  is_tolerant=false::Bool,
  tolerance=(-1000.0)::AbstractFloat,
  max_tolerance=4::Int64,
  grams=3::Int64,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  keep_sep=false::Bool,
  target_col="Words"::String,
  issparse=:auto::Symbol,
  sparse_ratio=0.2::Float64,
  verbose=false::Bool
  )::Union{Tuple{Vector{Vector{Result_Path_Info_Struct}}, Vector{Gold_Path_Info_Struct}}, Vector{Vector{Result_Path_Info_Struct}}}

  # initialize queues for storing paths
  n_val = size(data_val, 1)
  # working_q = Array{Queue{Array{Int64,1}},1}(undef, n_val)
  working_q = Vector{Queue{Tuple{Vector{Int64}, Int64}}}(undef, n_val)
  # res = Array{Array{Array{Int64}},1}(undef, n_val)
  res = Vector{Vector{Tuple{Vector{Int64}, Int64}}}(undef, n_val)
  for j in 1:n_val
    res[j] = Tuple{Vector{Int64}, Int64}[]
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
    Mt_train = make_transform_matrix(fac_C_train, C_train, Yt_train,
      output_format=issparse, sparse_ratio=sparse_ratio, verbose=verbose)
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
    if verbose
      pb = Progress(n_val)
    end
    for j in iter
      # collect all n-grams which has greater support than the threshold
      candidates_t = findall(x->x>threshold, Ythat_val[j,:])
      candidates_t_tlr = findall(x->x>tolerance&&x<=threshold, Ythat_val[j,:])

      # for timestep 2 and after 2
      if isassigned(working_q, j)
        tmp_working_q = Queue{Tuple{Vector{Int64},Int64}}()
        while !isempty(working_q[j])
          a = dequeue!(working_q[j]) ## a = [11] A[11,5] == 1 # candidates = [1, 5, 7]

          for c in candidates_t ## c = 5 # a = [11, 1, 5, 7] # a = [11, 1] [11, 5] [11, 7]
            # if a n-grams is attachable then append it
            if isattachable(a[1], c, A)
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
              if isattachable(a[1], c, A)
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
        working_q[j] = Queue{Tuple{Vector{Int64},Int64}}()
        for c in candidates_t
          # check whether a n-gram is a start n-gram
          if isstart(c, i2f, tokenized=tokenized, sep_token=sep_token)
            a = Int64[]
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
              a = Int64[]
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
      if verbose
        ProgressMeter.next!(pb)
      end
    end
  end

  verbose && println("Evaluating paths...")
  res = eval_can(res, S_val, F_train, i2f, max_can, verbose)

  # initialize gold_path_infos
  if check_gold_path
    if isnothing(gold_ind)
      throw(ErrorException("gold_ind is nothing! Perhaps you forgot to pass gold_ind as an argument."))
    end

    if isnothing(Shat_val)
      throw(ErrorException("Shat_val is nothing! Perhaps you forgot to pass Shat_val as an argument."))
    end
    
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
    build_paths(::DataFrame,::SparseMatrixCSC,::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Matrix,::SparseMatrixCSC,::Dict,::Array) -> ::Vector{Vector{Result_Path_Info_Struct}}

build_paths function is a shortcut algorithm for finding paths that only takes
the n-grams of that which are close to the targets.

...
# Obligatory Arguments
- `data::DataFrame`: the training dataset
- `data_val::DataFrame`: the validation dataset
- `C_train::SparseMatrixCSC`: the C matrix for training dataset
- `S_val::Union{SparseMatrixCSC, Matrix}`: the S matrix for validation dataset
- `F_train::Union{SparseMatrixCSC, Matrix}`: the F matrix for training dataset
- `Chat_val::Matrix`: the Chat matrix for validation dataset
- `A::SparseMatrixCSC`: the adjacency matrix
- `i2f::Dict`: the dictionary return features given indices
- `C_train_ind::Array`: the gold paths' indices for training dataset

# Optional Arguments
- `rC::Union{Nothing, Matrix}=nothing`: correlation Matrix of C and Chat, pass it to save computing time
- `max_t::Int64=15`: maximum timestep
- `max_can::Int64=10`: maximum candidates to keep in the results
- `n_neighbors::Int64=10`: find indices only in top n neighbors
- `grams::Int64=3`: the number of grams for cues
- `tokenized::Bool=false`: if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `target_col::Union{String, :Symbol}=:Words`: the column name for target strings
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# training dataset
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

# validation dataset
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
  A::SparseMatrixCSC,
  i2f::Dict,
  C_train_ind::Array;
  rC=nothing::Union{Nothing, Matrix},
  max_t=15::Int64,
  max_can=10::Int64,
  n_neighbors=10::Int64,
  grams=3::Int64,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  target_col=:Words::Union{String, Symbol},
  verbose=false::Bool
  )::Vector{Vector{Result_Path_Info_Struct}}
  # initialize queues for storing paths
  n_val = size(data_val, 1)
  # working_q = Array{Queue{Array{Int64,1}},1}(undef, n_val)
  # res = Array{Array{Array{Int64}},1}(undef, n_val)
  res = Vector{Vector{Tuple{Vector{Int64}, Int64}}}(undef, n_val)

  for j in 1:n_val
    res[j] = Tuple{Vector{Int64}, Int64}[]
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
  if verbose
    pb = Progress(n_val)
  end
  for j in iter
    candidates_t = top_indices[j]

    # timestep 1
    working_q = Queue{Array{Int64, 1}}()
    for c in candidates_t
      # check whether a n-gram is a start n-gram
      if isstart(c, i2f, tokenized=tokenized, sep_token=sep_token)
        a = Int64[]
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
      tmp_working_q = Queue{Array{Int64, 1}}()
      while !isempty(working_q)
        a = dequeue!(working_q) ## a = [11] A[11,5] == 1 # candidates = [1, 5, 7]
        for c in candidates_t ## c = 5 # a = [11, 1, 5, 7] # a = [11, 1] [11, 5] [11, 7]
          # if a n-grams is attachable then append it
          if isattachable(a, c, A)
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
    if verbose
      ProgressMeter.next!(pb)
    end
  end

  verbose && println("Evaluating paths...")
  eval_can(res, S_val, F_train, i2f, max_can, verbose)
end

"""
    eval_can(::Vector{Vector{Tuple{Vector{Int64}, Int64}}},::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Dict,::Int64,::Bool) -> ::Array{Array{Result_Path_Info_Struct,1},1}

Evaluate each candidate regarding the correlation between predicted semantic 
vector and original semantic vector.
"""
function eval_can(
  candidates::Vector{Vector{Tuple{Vector{Int64}, Int64}}},
  S::Union{SparseMatrixCSC, Matrix},
  F::Union{SparseMatrixCSC, Matrix},
  i2f::Dict,
  max_can::Int64,
  verbose=false::Bool
  )::Array{Array{Result_Path_Info_Struct,1},1}

  verbose && println("average $(mean(length.(candidates))) of paths to evaluate")

  res_l = Array{Array{Result_Path_Info_Struct,1},1}(undef, size(S, 1))
  iter = 1:size(S, 1)
  if verbose
    pb = Progress(size(S, 1))
  end
  for i in iter
    res = Result_Path_Info_Struct[]
    if size(candidates[i], 1) > 0
      for (ci,n) in candidates[i] # ci = [1,3,4]
        Chat = zeros(Int64, length(i2f))
        Chat[ci] .= 1
        Shat = Chat'*F
        Scor = cor(Shat[1, :], S[i, :])
        push!(res, Result_Path_Info_Struct(ci, n, Scor))
      end
    end
    # we collect only top x candidates from the top
    res_l[i] = collect(Iterators.take(sort!(res, by=x->x.support, rev=true), max_can))
    if verbose
      ProgressMeter.next!(pb)
    end
  end

  res_l
end

"""
    find_top_feature_indices(::Matrix, ::Array) -> ::Vector{Vector{Int64}}

Find all indices of the top n closest neighbors for a given target.
"""
function find_top_feature_indices(
  # C_train::SparseMatrixCSC,
  # Chat_val::Union{SparseMatrixCSC, Matrix},
  rC::Matrix,
  C_train_ind::Array;
  n_neighbors=10::Int64,
  verbose=false::Bool
  )::Vector{Vector{Int64}}

  # collect num of val data
  n_val = size(rC, 1)

  # calculate correlation matrix
  # rC = cor(Chat_val, Matrix(C_train), dims=2)
  # display(rC)

  # initialize features list for all candidates
  features_all = Vector{Vector{Int64}}(undef, n_val)

  # create iter for progress bar
  verbose && println("finding all n_neighbors features...")
  iter = 1:n_val
  if verbose
    pb = Progress(n_val)
  end

  # find all features of n_neighbors
  for i in iter
    features = [C_train_ind[j] for j in sortperm(rC[i,:], rev=true)[1:n_neighbors]]
    features_all[i] = unique(collect(Iterators.flatten(features)))
    if verbose
      ProgressMeter.next!(pb)
    end
  end

  features_all
end