"""
Make Yt matrix for timestep t.
"""
function make_Yt_matrix end

"""
    make_Yt_matrix(t, data, f2i)

Make Yt matrix for timestep t. A given column of the Yt matrix specifies the support
for the corresponding n-gram predicted for timestep t for each of the observations (rows of Yt).

...
# Obligatory Arguments
- `t::Int64`: the timestep t
- `data::DataFrame`: the dataset
- `f2i::Dict`: the dictionary returning indices given features

# Optional Arguments
- `tokenized::Bool=false`: if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator token
- `verbose::Bool=false`: if verbose, more information will be printed

# Examples
```julia
latin = DataFrame(CSV.File(joinpath("data", "latin_mini.csv")))
JudiLing.make_Yt_matrix(2, latin)
```
...
"""
function make_Yt_matrix(
    t,
    data,
    f2i;
    grams = 3,
    target_col = "Words",
    tokenized = false,
    keep_sep = false,
    sep_token = nothing,
    start_end_token = "#",
)

    # split tokens from words or other columns
    if tokenized && !isnothing(sep_token)
        tokens = split.(data[:, target_col], sep_token)
    else
        tokens = split.(data[:, target_col], "")
    end

    # making ngrams from tokens
    # make_ngrams function are below
    ngrams = make_ngrams.(tokens, grams, keep_sep, sep_token, start_end_token)

    ngrams_l = length.(ngrams)

    # the number of non-zero values is equal to the number of utterances have
    # larger length than timestep t
    n_f = length(ngrams_l[ngrams_l.>=t])

    m = size(data, 1)
    n = length(f2i)
    I = zeros(Int64, n_f)
    J = zeros(Int64, n_f)
    V = ones(Int64, n_f)

    cnt = 0
    for (i, v) in enumerate(ngrams)
        if t <= length(v)
            cnt += 1
            I[cnt] = i
            J[cnt] = f2i[v[t]]
        end
    end

    sparse(I, J, V, m, n)

end
