"""
A structure that stores information about comprehension accuracy.
"""
struct Comp_Acc_Struct
    dfr::DataFrame
    acc::Float64
    err::Array
end

"""
Assess model accuracy on the basis of the correlations of row vectors of Chat and
C or Shat and S. Ideally the target words have highest correlations on the diagonal
of the pertinent correlation matrices. Homophones support option is implemented.
"""
function eval_SC end


"""
Assess model accuracy on the basis of the correlations of row vectors of Chat and
C or Shat and S. Count it as correct if one of the top k candidates is correct.
Homophones support option is implemented.
"""
function eval_SC_loose end

"""
accuracy_comprehension(S, Shat, data)

Evaluate comprehension accuracy.

...
# Obligatory Arguments
- `S::Matrix`: the (gold standard) S matrix
- `Shat::Matrix`: the (predicted) Shat matrix
- `data::DataFrame`: the dataset

# Optional Arguments
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `base::Vector=nothing`: base features (typically a lexeme)
- `inflections::Union{Nothing, Vector}=nothing`: other features (typically in inflectional features)

# Examples
```julia
accuracy_comprehension(
    S_train,
    Shat_train,
    latin_val,
    target_col=:Words,
    base=[:Lexeme],
    inflections=[:Person, :Number, :Tense, :Voice, :Mood]
    )

accuracy_comprehension(
    S_val,
    Shat_val,
    latin_train,
    target_col=:Words,
    base=["Lexeme"],
    inflections=[:Person, :Number, :Tense, :Voice, :Mood]
    )
```
...
"""
function accuracy_comprehension(
    S,
    Shat,
    data;
    target_col = :Words,
    base = nothing,
    inflections = nothing,
)

    corMat = cor(Shat, S, dims = 2)
    top_index = [i[2] for i in argmax(corMat, dims = 2)]

    dfr = DataFrame()
    dfr.target = data[:, target_col]
    dfr.form = vec([data[i, target_col] for i in top_index])
    dfr.r =
        vec([corMat[index, value] for (index, value) in enumerate(top_index)])
    dfr.r_target = corMat[diagind(corMat)]
    dfr.correct = [dfr.target[i] == dfr.form[i] for i = 1:size(dfr, 1)]

    if !isnothing(inflections)
        all_features = vcat(base, inflections)
    else
        all_features = base
    end

    for f in all_features
        dfr.tmp = vec([
            data[index, f] == data[value, f]
            for (index, value) in enumerate(top_index)
        ])
        rename!(dfr, "tmp" => f)
    end

    acc = sum(dfr[:, "correct"]) / size(dfr, 1)
    err = findall(x -> x != 1, dfr[:, "correct"])

    Comp_Acc_Struct(dfr, acc, err)
end

"""
eval_SC(SChat, SC)

Assess model accuracy on the basis of the correlations of row vectors of Chat and
C or Shat and S. Ideally the target words have highest correlations on the diagonal
of the pertinent correlation matrices.

...
# Obligatory Arguments
- `SChat::Union{SparseMatrixCSC, Matrix}`: the Chat or Shat matrix
- `SC::Union{SparseMatrixCSC, Matrix}`: the C or S matrix

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)
- `R::Bool=false`: if true, pairwise correlation matrix R is return

```julia
eval_SC(Chat_train, cue_obj_train.C)
eval_SC(Chat_val, cue_obj_val.C)
eval_SC(Shat_train, S_train)
eval_SC(Shat_val, S_val)
```
...
"""
function eval_SC(SChat, SC; digits=4, R=false)
    rSC = cor(
        convert(Matrix{Float64}, SChat),
        convert(Matrix{Float64}, SC),
        dims = 2,
    )
    v = [rSC[i[1], i[1]] == rSC[i] ? 1 : 0 for i in argmax(rSC, dims = 2)]
    acc = round(sum(v) / length(v), digits=digits)
    if R
        return acc, rSC
    else
        return acc
    end
end

"""
eval_SC(SChat, SC, data, target_col)

Assess model accuracy on the basis of the correlations of row vectors of Chat and
C or Shat and S. Ideally the target words have highest correlations on the diagonal
of the pertinent correlation matrices. Support for homophones.

...
# Obligatory Arguments
- `SChat::Union{SparseMatrixCSC, Matrix}`: the Chat or Shat matrix
- `SC::Union{SparseMatrixCSC, Matrix}`: the C or S matrix

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)
- `R::Bool=false`: if true, pairwise correlation matrix R is return

```julia
eval_SC(Chat_train, cue_obj_train.C)
eval_SC(Chat_val, cue_obj_val.C)
eval_SC(Shat_train, S_train)
eval_SC(Shat_val, S_val)
```
...
"""
function eval_SC(SChat, SC, data, target_col; digits=4, R=false)
    rSC = cor(
        convert(Matrix{Float64}, SChat),
        convert(Matrix{Float64}, SC),
        dims = 2,
    )
    v = [
        data[i[1], target_col] == data[i[2], target_col] ? 1 : 0
        for i in argmax(rSC, dims = 2)
    ]
    acc = round(sum(v) / length(v), digits=digits)
    if R
        return acc, rSC
    else
        return acc
    end
end

"""
eval_SC(SChat, SC, batch_size)

Assess model accuracy on the basis of the correlations of row vectors of Chat and
C or Shat and S. Ideally the target words have highest correlations on the diagonal
of the pertinent correlation matrices. For large datasets, pass batch_size to
process evaluation in chucks.

...
# Obligatory Arguments
- `SChat`: the Chat or Shat matrix
- `SC`: the C or S matrix
- `batch_size`: batch size
- `data`: datasets
- `target_col`: target column name

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)
- `verbose::Bool=false`: if true, more information is printed

```julia
eval_SC(Chat_train, cue_obj_train.C, latin, :Word)
eval_SC(Chat_val, cue_obj_val.C, latin, :Word)
eval_SC(Shat_train, S_train, latin, :Word)
eval_SC(Shat_val, S_val, latin, :Word)
```
...
"""
function eval_SC(SChat, SC, batch_size; digits=4, verbose = false)
    l = size(SChat, 1)
    num_chucks = ceil(Int64, l / batch_size)
    verbose && begin
        pb = Progress(num_chucks)
    end
    correct = 0

    SChat_d = convert(Matrix{Float64}, SChat)
    SC_d = convert(Matrix{Float64}, SC)

    # for first parts
    for j = 1:num_chucks-1
        correct += eval_SC_chucks(
            SChat_d,
            SC_d,
            (j - 1) * batch_size + 1,
            j * batch_size,
            batch_size,
        )
        verbose && ProgressMeter.next!(pb)
    end
    # for last part
    correct += eval_SC_chucks(
        SChat_d,
        SC_d,
        (num_chucks - 1) * batch_size + 1,
        batch_size,
    )
    verbose && ProgressMeter.next!(pb)

    round(correct / l, digits=digits)
end

"""
eval_SC(SChat, SC, data, target_col, batch_size)

Assess model accuracy on the basis of the correlations of row vectors of Chat and
C or Shat and S. Ideally the target words have highest correlations on the diagonal
of the pertinent correlation matrices. For large datasets, pass batch_size to
process evaluation in chucks. Support homophones.

...
# Obligatory Arguments
- `SChat`: the Chat or Shat matrix
- `SC`: the C or S matrix
- `batch_size`: batch size
- `data`: datasets
- `target_col`: target column name

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)
- `verbose::Bool=false`: if true, more information is printed

```julia
eval_SC(Chat_train, cue_obj_train.C, latin, :Word, 5000)
eval_SC(Chat_val, cue_obj_val.C, latin, :Word, 5000)
eval_SC(Shat_train, S_train, latin, :Word, 5000)
eval_SC(Shat_val, S_val, latin, :Word, 5000)
```
...
"""
function eval_SC(SChat, SC, data, target_col, batch_size; digits=4, verbose = false)

    l = size(SChat, 1)
    num_chucks = ceil(Int64, l / batch_size)
    verbose && begin
        pb = Progress(num_chucks)
    end
    correct = 0

    SChat_d = convert(Matrix{Float64}, SChat)
    SC_d = convert(Matrix{Float64}, SC)

    # for first parts
    for j = 1:num_chucks-1
        correct += eval_SC_chucks(
            SChat_d,
            SC_d,
            (j - 1) * batch_size + 1,
            j * batch_size,
            batch_size,
            data,
            target_col,
        )
        verbose && ProgressMeter.next!(pb)
    end
    # for last part
    correct += eval_SC_chucks(
        SChat_d,
        SC_d,
        (num_chucks - 1) * batch_size + 1,
        batch_size,
        data,
        target_col,
    )
    verbose && ProgressMeter.next!(pb)

    round(correct / l, digits=digits)
end

function eval_SC_chucks(SChat, SC, s, e, batch_size)
    rSC = cor(SChat[s:e, :], SC, dims = 2)
    v = [(rSC[i[1], i[1]+s-1] == rSC[i]) ? 1 : 0 for i in argmax(rSC, dims = 2)]
    sum(v)
end

function eval_SC_chucks(SChat, SC, s, e, batch_size, data, target_col)
    rSC = cor(SChat[s:e, :], SC, dims = 2)
    v = [
        data[i[1]+s-1, target_col] == data[i[2], target_col] ? 1 : 0
        for i in argmax(rSC, dims = 2)
    ]
    sum(v)
end

function eval_SC_chucks(SChat, SC, s, batch_size)
    rSC = cor(SChat[s:end, :], SC, dims = 2)
    v = [(rSC[i[1], i[1]+s-1] == rSC[i]) ? 1 : 0 for i in argmax(rSC, dims = 2)]
    sum(v)
end

function eval_SC_chucks(SChat, SC, s, batch_size, data, target_col)
    rSC = cor(SChat[s:end, :], SC, dims = 2)
    v = [
        data[i[1]+s-1, target_col] == data[i[2], target_col] ? 1 : 0
        for i in argmax(rSC, dims = 2)
    ]
    sum(v)
end

"""
eval_SC_loose(SChat, SC, k)

Assess model accuracy on the basis of the correlations of row vectors of Chat and
C or Shat and S. Count it as correct if one of the top k candidates is correct.

...
# Obligatory Arguments
- `SChat::Union{SparseMatrixCSC, Matrix}`: the Chat or Shat matrix
- `SC::Union{SparseMatrixCSC, Matrix}`: the C or S matrix
- `k`: top k candidates

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)

```julia
eval_SC_loose(Chat, cue_obj.C, k)
eval_SC_loose(Shat, S, k)
```
...
"""
function eval_SC_loose(SChat, SC, k; digits=4)
    total = size(SChat, 1)
    correct = 0
    rSC = cor(
        convert(Matrix{Float64}, SChat),
        convert(Matrix{Float64}, SC),
        dims = 2,
    )

    for i = 1:total
        p = sortperm(rSC[i, :], rev = true)
        p = p[1:k, :]
        if i in p
            correct += 1
        end
    end
    round(correct / total, digits=digits)
end

"""
eval_SC_loose(SChat, SC, k, data, target_col)

Assess model accuracy on the basis of the correlations of row vectors of Chat and
C or Shat and S. Count it as correct if one of the top k candidates is correct.
Support for homophones.

...
# Obligatory Arguments
- `SChat::Union{SparseMatrixCSC, Matrix}`: the Chat or Shat matrix
- `SC::Union{SparseMatrixCSC, Matrix}`: the C or S matrix
- `k`: top k candidates
- `data`: datasets
- `target_col`: target column name

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)

```julia
eval_SC_loose(Chat, cue_obj.C, k, latin, :Word)
eval_SC_loose(Shat, S, k, latin, :Word)
```
...
"""
function eval_SC_loose(SChat, SC, k, data, target_col; digits=4)
    total = size(SChat, 1)
    correct = 0
    rSC = cor(
        convert(Matrix{Float64}, SChat),
        convert(Matrix{Float64}, SC),
        dims = 2,
    )

    for i = 1:total
        p = sortperm(rSC[i, :], rev = true)
        p = p[1:k]
        if i in p
            correct += 1
        else
            if data[i, target_col] in data[p, target_col]
                correct += 1
            end
        end
    end
    round(correct / total, digits=digits)
end

"""
eval_manual(res, data, i2f)

Create extensive reports for the outputs from `build_paths` and `learn_paths`.
"""
function eval_manual(
    res,
    data,
    i2f;
    s = 1,
    e = nothing,
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    verbose = false,
    identifier = "Identifier",
    output_sep_token = "",
)

    # is end is not specified then it is the end of dataset
    if isnothing(e)
        e = length(res)
    end

    for i = s:e
        verbose && println("="^20)
        verbose && println("utterance $i: ")
        verbose && println(data[i, identifier])
        verbose && println("predictions: ")
        verbose && begin
            display_pred(
                res[i],
                i2f,
                grams,
                tokenized,
                sep_token,
                start_end_token,
                output_sep_token,
            )
        end
    end
    return
end


"""
eval_acc(res, gold_inds::Array)

Evaluate the accuracy of the results from `learn_paths` or `build_paths`.

...
# Obligatory Arguments
- `res::Array`: the results from `learn_paths` or `build_paths`
- `gold_inds::Array`: the gold paths' indices

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# evaluation on training data
acc_train = JudiLing.eval_acc(
    res_train,
    cue_obj_train.gold_ind,
    verbose=false
)

# evaluation on validation data
acc_val = JudiLing.eval_acc(
    res_val,
    cue_obj_val.gold_ind,
    verbose=false
)
```
...
"""
function eval_acc(res, gold_inds::Array; digits=4, verbose = false)

    total = length(res)
    acc = 0

    iter = 1:total
    if verbose
        pb = Progress(total)
    end

    for i in iter
        if isassigned(res[i], 1) &&
           iscorrect(gold_inds[i], res[i][1].ngrams_ind)
            acc += 1
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end

    round(acc / total, digits=digits)
end

"""
eval_acc(res, cue_obj::Cue_Matrix_Struct)

Evaluate the accuracy of the results from `learn_paths` or `build_paths`.

...
# Obligatory Arguments
- `res::Array`: the results from `learn_paths` or `build_paths`
- `cue_obj::Cue_Matrix_Struct`: the C matrix object

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
acc = JudiLing.eval_acc(res, cue_obj)
```
...
"""
function eval_acc(res, cue_obj::Cue_Matrix_Struct; digits = 4, verbose = false)
    eval_acc(res, cue_obj.gold_ind, digits = digits, verbose = verbose)
end

"""
eval_acc_loose(res, gold_inds)

Lenient evaluation of the accuracy of the results from `learn_paths` or `build_paths`,
counting a prediction as correct when the correlation of the predicted and gold
standard semantic vectors is among the n top correlations, where n is equal to
`max_can` in the 'learn_paths' or `build_paths` function.

...
# Obligatory Arguments
- `res::Array`: the results from `learn_paths` or `build_paths`
- `gold_inds::Array`: the gold paths' indices

# Optional Arguments
- `digits`: the specified number of digits after the decimal place (or before if negative)
- `verbose::Bool=false`: if true, more information is printed

# Examples
```julia
# evaluation on training data
acc_train_loose = JudiLing.eval_acc_loose(
    res_train,
    cue_obj_train.gold_ind,
    verbose=false
)

# evaluation on validation data
acc_val_loose = JudiLing.eval_acc_loose(
    res_val,
    cue_obj_val.gold_ind,
    verbose=false
)
```
...
"""
function eval_acc_loose(res, gold_inds; digits=4, verbose = false)

    total = length(res)
    acc = 0

    iter = 1:total
    if verbose
        pb = Progress(total)
    end

    for i in iter
        is_correct = false
        for j = 1:length(res[i])
            if isassigned(res[i], 1) &&
               iscorrect(gold_inds[i], res[i][j].ngrams_ind)
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

    round(acc / total, digits=4)
end

"""
extract_gpi(gpi, threshold=0.1, tolerance=(-1000.0))

Extract, using gold paths' information, how many n-grams for a gold
path are below the threshold but above the tolerance.
"""
function extract_gpi(gpi, threshold = 0.1, tolerance = (-1000.0))
    c = [
        count(x -> threshold >= x > tolerance, g.ngrams_ind_support)
        for g in gpi
    ]

    t_c = length(c)

    summary_c = [
        (x, count(cnt -> cnt == x, c), count(cnt -> cnt == x, c) / t_c)
        for x in sort(unique(c))
    ]

    summary_c
end
