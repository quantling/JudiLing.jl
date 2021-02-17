"""
Make full adjacency matrix.
"""
function make_adjacency_matrix end

"""
    make_adjacency_matrix(i2f)

Make full adjacency matrix based only on the form of n-grams regardless of whether 
they are seen in the training data. This usually takes hours for large datasets, 
as all possible combinations are considered.

...
# Obligatory Arguments
- `i2f::Dict`: the dictionary returning features given indices

# Optional Arguments
- `tokenized::Bool=false`:if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator token
- `verbose::Bool=false`: if true, more information will be printed

# Examples
```julia
# without tokenization
i2f = Dict([(1, "#ab"), (2, "abc"), (3, "bc#"), (4, "#bc"), (5, "ab#")])
JudiLing.make_adjacency_matrix(i2f)

# with tokenization
i2f = Dict([(1, "#-a-b"), (2, "a-b-c"), (3, "b-c-#"), (4, "#-b-c"), (5, "a-b-#")])
JudiLing.make_adjacency_matrix(
  i2f,
  tokenized=true,
  sep_token="-")
```
...
"""
function make_adjacency_matrix(
  i2f;
  tokenized=false,
  sep_token=nothing,
  verbose=false
  )

  ngrams = [i2f[i] for i in 1:length(i2f)]

  if tokenized && !isnothing(sep_token)
    words = split.(ngrams, sep_token)
  else
    words = split.(ngrams, "")
  end

  n_ngrams = length(ngrams)

  I = Int64[]
  J = Int64[]
  V = Int64[]

  iter = 1:n_ngrams
  if verbose
    pb = Progress(n_ngrams)
  end

  for i in iter
    for j in 1:n_ngrams
      if isattachable(words[i], words[j])
        push!(I, i)
        push!(J, j)
        push!(V, 1)
      end
    end
    if verbose
      ProgressMeter.next!(pb)
    end
  end

  sparse(I, J, V, n_ngrams, n_ngrams, *)
end

"""
    make_combined_adjacency_matrix(data_train, data_val)

Make combined adjacency matrix.

...
# Obligatory Arguments
- `data_train::DataFrame`: training dataset
- `data_val::DataFrame`: validation dataset

# Optional Arguments
- `grams=3`: the number of grams for cues 
- `target_col=:Words`: the column name for target strings
- `tokenized=false`:if true, the dataset target is assumed to be tokenized
- `sep_token=nothing`: separator
- `keep_sep=false`: if true, keep separators in cues
- `start_end_token="#"`: start and end token in boundary cues
- `verbose=false`: if true, more information is printed

# Examples
```julia
JudiLing.make_combined_adjacency_matrix(
  latin_train,
  latin_val,
  grams=3,
  target_col=:Word,
  tokenized=false,
  keep_sep=false
  )
```
...
"""
function make_combined_adjacency_matrix(
  data_train,
  data_val;
  grams=3,
  target_col=:Words,
  tokenized=false,
  sep_token=nothing,
  keep_sep=false,
  start_end_token="#",
  verbose=false
  )

  t, v = make_combined_cue_matrix(
    data_train,
    data_val;
    grams=grams,
    target_col=target_col,
    tokenized=tokenized,
    sep_token=sep_token,
    keep_sep=keep_sep,
    start_end_token=start_end_token,
    verbose=verbose)

  t.A
end