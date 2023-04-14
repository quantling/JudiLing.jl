"""
Write results into a csv file. This function takes as input the results from the
`learn_paths` and `build_paths` functions, including the information on gold paths
that is optionally returned as second output result.
"""
function write2csv end

"""
Reformat results into a dataframe. This function takes as input the results from the
`learn_paths` and `build_paths` functions, including the information on gold paths
that is optionally returned as second output result.
"""
function write2df end

"""
Write comprehension evaluation into a CSV file, include target and predicted
ids and indentifiers and their correlations.
"""
function write_comprehension_eval end

"""
    write2csv(res, data, cue_obj_train, cue_obj_val, filename)

Write results into csv file for the results from `learn_paths` and `build_paths`.

# Obligatory Arguments
- `res::Array{Array{Result_Path_Info_Struct,1},1}`: the results from `learn_paths` or `build_paths`
- `data::DataFrame`: the dataset
- `cue_obj_train::Cue_Matrix_Struct`: the cue object for training dataset
- `cue_obj_val::Cue_Matrix_Struct`: the cue object for validation dataset
- `filename::String`: the filename

# Optional Arguments
- `grams::Int64=3`: the number n in n-gram cues
- `tokenized::Bool=false`: if true, the dataset target is tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `output_sep_token::Union{String, Char}=""`: output separator
- `path_sep_token::Union{String, Char}=":"`: path separator
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `root_dir::String="."`: dir path for project root dir
- `output_dir::String="."`: output dir inside root dir

# Examples
```julia
# writing results for training data
JudiLing.write2csv(
    res_train,
    latin_train,
     cue_obj_train,
    cue_obj_train,
    "res_latin_train.csv",
    grams=3,
    tokenized=false,
    sep_token=nothing,
    start_end_token="#",
    output_sep_token="",
    path_sep_token=":",
    target_col=:Word,
    root_dir=".",
    output_dir="test_out")

# writing results for validation data
JudiLing.write2csv(
    res_val,
    latin_val,
    cue_obj_train,
    cue_obj_val,
    "res_latin_val.csv",
    grams=3,
    tokenized=false,
    sep_token=nothing,
    start_end_token="#",
    output_sep_token="",
    path_sep_token=":",
    target_col=:Word,
    root_dir=".",
    output_dir="test_out")
```
"""
function write2csv(
    res,
    data,
    cue_obj_train,
    cue_obj_val,
    filename;
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :Words,
    root_dir = ".",
    output_dir = ".",
)

    output_path = joinpath(root_dir, output_dir)
    # create path if not exist
    mkpath(output_path)

    if isnothing(output_sep_token)
        output_sep_token = ""
    end

    df = JudiLing.write2df(
        res,
        data,
        cue_obj_train,
        cue_obj_val,
        grams = grams,
        tokenized = tokenized,
        sep_token = sep_token,
        start_end_token = start_end_token,
        output_sep_token = output_sep_token,
        path_sep_token = path_sep_token,
        target_col = target_col,
    )

    CSV.write(joinpath(output_path, filename), df, quotestrings = true)
    nothing
end

"""
    write2csv(gpi::Vector{Gold_Path_Info_Struct}, filename)

Write results into csv file for the gold paths' information optionally returned by
`learn_paths` and `build_paths`.

# Obligatory Arguments
- `gpi::Vector{Gold_Path_Info_Struct}`: the gold paths' information
- `filename::String`: the filename

# Optional Arguments
- `root_dir::String="."`: dir path for project root dir
- `output_dir::String="."`: output dir inside root dir

# Examples
```julia
# write gold standard paths to csv for training data
JudiLing.write2csv(
    gpi_train,
    "gpi_latin_train.csv",
    root_dir=".",
    output_dir="test_out"
    )

# write gold standard paths to csv for validation data
JudiLing.write2csv(
    gpi_val,
    "gpi_latin_val.csv",
    root_dir=".",
    output_dir="test_out"
    )
```
"""
function write2csv(gpi::Vector{Gold_Path_Info_Struct}, filename; root_dir = ".", output_dir = ".")
    output_path = joinpath(root_dir, output_dir)
    # create path if not exist
    mkpath(output_path)
    # open a file
    io = open(joinpath(output_path, filename), "w")

    # write header
    write(
        io,
        "\"utterance\",\"weakest_support\",\"weakest_support_timestep\",\"support\",\"gold_path\",\"timestep_support\"\n",
    )

    for (i, g) in enumerate(gpi)
        if isempty(g.ngrams_ind_support)
            ws = missing
            wst = missing
        else
            ws, wst = findmin(g.ngrams_ind_support)
        end
        write(
            io,
            "\"$i\",\"$(ws)\",\"$(wst)\",\"$(g.support)\",\"$(g.ngrams_ind)\",\"$(g.ngrams_ind_support)\"\n",
        )
    end
    # close file
    close(io)
end

"""
    write2csv(ts::Threshold_Stat_Struct, filename)

Write results into csv file for threshold and tolerance proportion for each timestep.

# Obligatory Arguments
- `gpi::Vector{Gold_Path_Info_Struct}`: the gold paths' information
- `filename::String`: the filename

# Optional Arguments
- `root_dir::String="."`: dir path for project root dir
- `output_dir::String="."`: output dir inside root dir

# Examples
```julia
JudiLing.write2csv(ts, "ts.csv", root_dir = @__DIR__, output_dir="out")
```
"""
function write2csv(ts::Threshold_Stat_Struct, filename; root_dir = ".", output_dir = ".")
    output_path = joinpath(root_dir, output_dir)

    # create path if not exist
    mkpath(output_path)

    df = JudiLing.write2df(ts)
    CSV.write(joinpath(output_path, filename), df, quotestrings = true)
    nothing
end

"""
    write2df(res, data, cue_obj_train, cue_obj_val)

Reformat results into a dataframe for the results form `learn_paths` and `build_paths`
functions.

# Obligatory Arguments
- `res`: output of `learn_paths` or `build_paths`
- `data::DataFrame`: the dataset
- `cue_obj_train`: cue object of the training data set
- `cue_obj_val`: cue object of the validation data set

# Optional Arguments
- `grams::Int64=3`: the number n in n-gram cues
- `tokenized::Bool=false`: if true, the dataset target is tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `output_sep_token::Union{String, Char}=""`: output separator
- `path_sep_token::Union{String, Char}=":"`: path separator
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings

# Examples
```julia
# writing results for training data
JudiLing.write2df(
    res_train,
    latin_train,
    cue_obj_train,
    cue_obj_train,
    grams=3,
    tokenized=false,
    sep_token=nothing,
    start_end_token="#",
    output_sep_token="",
    path_sep_token=":",
    target_col=:Word)

# writing results for validation data
JudiLing.write2df(
    res_val,
    latin_val,
    cue_obj_train,
    cue_obj_val,
    grams=3,
    tokenized=false,
    sep_token=nothing,
    start_end_token="#",
    output_sep_token="",
    path_sep_token=":",
    target_col=:Word)
```
"""
function write2df(
    res,
    data,
    cue_obj_train,
    cue_obj_val;
    grams = 3,
    tokenized = false,
    sep_token = nothing,
    start_end_token = "#",
    output_sep_token = "",
    path_sep_token = ":",
    target_col = :Words,
)

    utterance_vec = Union{Int64,Missing}[]
    identifier_vec = Union{String,Missing}[]
    path_vec = Union{String,Missing}[]
    pred_vec = Union{String,Missing}[]
    num_tolerance_vec = Union{Int64,Missing}[]
    support_vec = Union{Float64,Missing}[]
    isbest_vec = Union{Bool,Missing}[]
    iscorrect_vec = Union{Bool,Missing}[]
    isnovel_vec = Union{Bool,Missing}[]


    i2f = cue_obj_train.i2f
    for (i, r) in enumerate(res)
        is_best = true
        if length(r) == 0
            utterance = i
            identifier = data[i, target_col]
            path = missing
            pred = missing
            num_tolerance = missing
            support = missing
            is_correct = missing
            is_novel = missing
            is_best = missing
            push!(utterance_vec, utterance)
            push!(identifier_vec, identifier)
            push!(path_vec, path)
            push!(pred_vec, pred)
            push!(num_tolerance_vec, num_tolerance)
            push!(support_vec, support)
            push!(isbest_vec, is_best)
            push!(iscorrect_vec, is_correct)
            push!(isnovel_vec, is_novel)
        end

        for p in r
            utterance = i
            identifier = data[i, target_col]
            path = translate_path(p.ngrams_ind, i2f, sep_token = path_sep_token)
            pred = translate(
                p.ngrams_ind,
                i2f,
                grams,
                tokenized,
                sep_token,
                start_end_token,
                output_sep_token,
            )
            num_tolerance = p.num_tolerance
            support = p.support
            is_correct = iscorrect(cue_obj_val.gold_ind[i], p.ngrams_ind)
            is_novel = isnovel(cue_obj_train.gold_ind, p.ngrams_ind)
            push!(utterance_vec, utterance)
            push!(identifier_vec, identifier)
            push!(path_vec, path)
            push!(pred_vec, pred)
            push!(num_tolerance_vec, num_tolerance)
            push!(support_vec, support)
            push!(isbest_vec, is_best)
            push!(iscorrect_vec, is_correct)
            push!(isnovel_vec, is_novel)
            is_best = false
        end
    end

    df = DataFrame()
    df.utterance = utterance_vec
    df.identifier = identifier_vec
    df.path = path_vec
    df.pred = pred_vec
    df.num_tolerance = num_tolerance_vec
    df.support = support_vec
    df.isbest = isbest_vec
    df.iscorrect = iscorrect_vec
    df.isnovel = isnovel_vec

    df
end

"""
    write2df(gpi::Vector{Gold_Path_Info_Struct})

Write results into a dataframe for the gold paths' information optionally returned by
`learn_paths` and `build_paths`.

# Obligatory Arguments
- `gpi::Vector{Gold_Path_Info_Struct}`: the gold paths' information

# Examples
```julia
# write gold standard paths to df for training data
JudiLing.write2csv(gpi_train)

# write gold standard paths to df for validation data
JudiLing.write2csv(gpi_val)
```
"""
function write2df(gpi::Vector{Gold_Path_Info_Struct})

    utterance_vec = Union{Int64,Missing}[]
    weakest_support_vec = Union{Float64,Missing}[]
    weakest_support_timestep_vec = Union{Int64,Missing}[]
    support_vec = Union{Float64,Missing}[]
    gold_path_vec = Union{Vector,Missing}[]
    timestep_support_vec = Union{Vector,Missing}[]

    for (i, g) in enumerate(gpi)
        ws, wst = missing, missing
        try
            ws, wst = findmin(g.ngrams_ind_support)
        catch e
            ;
        end
        push!(utterance_vec, i)
        push!(weakest_support_vec, ws)
        push!(weakest_support_timestep_vec, wst)
        push!(support_vec, g.support)
        push!(gold_path_vec, g.ngrams_ind)
        push!(timestep_support_vec, g.ngrams_ind_support)
    end

    df = DataFrame()
    df.utterance = utterance_vec
    df.weakest_support = weakest_support_vec
    df.weakest_support_timestep = weakest_support_timestep_vec
    df.support = support_vec
    df.gold_path = gold_path_vec
    df.timestep_support = timestep_support_vec

    df
end


"""
    write2df(ts::Threshold_Stat_Struct)

Write results into a dataframe for threshold and tolerance proportion for each timestep.

# Obligatory Arguments
- `ts::Threshold_Stat_Struct`: the threshold and tolerance proportion

# Examples
```julia
JudiLing.write2df(ts)
```
"""
function write2df(ts::Threshold_Stat_Struct)
    n = ts.timestep

    df = DataFrame(vcat([Int64, Float64, Float64], [Float64 for k in 1:2n]),
        vcat([:utterance, :threshold, :tolerance], [Symbol("thr_prop_$k") for k in 1:n], [Symbol("tlr_prop_$k") for k in 1:n]),
        0)

    for i in 1:ts.n_rows
        push!(df, vcat([i, ts.threshold, ts.tolerance], ts.threshold_prop[i,:], ts.tolerance_prop[i,:]))
    end

    df
end

"""
    save_L_matrix(L, filename)

Save lexome matrix into csv file.

# Obligatory Arguments
- `L::L_Matrix_Struct`: the lexome matrix struct
- `filename::String`: the filename/filepath

# Examples
```julia
JudiLing.save_L_matrix(L, joinpath(@__DIR__, "L.csv"))
```
"""
function save_L_matrix(L, filename)

    L_df = DataFrame(L.L)
    insertcols!(L_df, 1, :col_names => L.i2f)
    CSV.write(filename, L_df, quotestrings = true, header = false)
    nothing
end

"""
    load_L_matrix(filename)

Load lexome matrix from csv file.

# Obligatory Arguments
- `filename::String`: the filename/filepath

# Optional Arguments
- `header::Bool=false`: header in csv

# Examples
```julia
L_load = JudiLing.load_L_matrix(joinpath(@__DIR__, "L.csv"))
```
"""
function load_L_matrix(filename; header = false)

    L_df = DataFrame(CSV.File(filename, header = header))
    i2f = L_df[:, 1]
    f2i = Dict(v => i for (i, v) in enumerate(i2f))
    ncol = size(L_df, 2) - 1
    L = Array(select(L_df, Not(1)))

    L_Matrix_Struct(L, f2i, i2f, ncol)
end

"""
    save_S_matrix(S, filename, data, target_col)

Save S matrix into a csv file.

# Obligatory Arguments
- `S::Matrix`: the S matrix
- `filename::String`: the filename/filepath
- `data::DataFrame`: the data
- `target_col::Symbol`: the name of target column

# Optional Arguments
- `sep::Bool=" "`: separator in CSV file

# Examples
```julia
JudiLing.save_S_matrix(S, joinpath(@__DIR__, "S.csv"), latin, :Word)
```
"""
function save_S_matrix(S, filename, data, target_col; sep = " ")

    S_df = DataFrame(S)
    insertcols!(S_df, 1, :col_names => data[:,target_col])
    CSV.write(filename, S_df, quotestrings = true, header = false, delim=sep)
    nothing
end

"""
    load_S_matrix(filename)

Load S matrix from a csv file.

# Obligatory Arguments
- `filename::String`: the filename/filepath

# Optional Arguments
- `header::Bool=false`: header in csv
- `sep::Bool=" "`: separator in CSV file

# Examples
```julia
JudiLing.load_S_matrix(joinpath(@__DIR__, "S.csv"))
```
"""
function load_S_matrix(filename; header = false, sep = " ")

    S_df = DataFrame(CSV.File(filename, header = header, delim = sep))
    words = S_df[:, 1]
    S = Array(select(S_df, Not(1)))

    S, words
end

"""
    write_comprehension_eval(SChat, SC, data, target_col, filename)

Write comprehension evaluation into a CSV file, include target and predicted
ids and indentifiers and their correlations.

# Obligatory Arguments
- `SChat::Matrix`: the Shat/Chat matrix
- `SC::Matrix`: the S/C matrix
- `data::DataFrame`: the data
- `target_col::Symbol`: the name of target column
- `filename::String`: the filename/filepath

# Optional Arguments
- `k`: top k candidates
- `root_dir::String="."`: dir path for project root dir
- `output_dir::String="."`: output dir inside root dir

# Examples
```julia
JudiLing.write_comprehension_eval(Chat, cue_obj.C, latin, :Word, "output.csv",
    k=10, root_dir=@__DIR__, output_dir="out")
```
"""
function write_comprehension_eval(
    SChat,
    SC,
    data,
    target_col,
    filename;
    k = 10,
    root_dir = ".",
    output_dir = "."
    )

    output_path = joinpath(root_dir, output_dir)
    mkpath(output_path)
    total = size(SChat, 1)

    io = open(joinpath(output_path, filename), "w")

    write(
        io,
        "\"target_utterance\",\"target_identifier\",\"predicted_utterance\",\"predicted_identifier\",\"support\"\n",
    )

    rSC = cor(
        convert(Matrix{Float64}, SChat),
        convert(Matrix{Float64}, SC),
        dims = 2,
    )

    for i = 1:total
        p = sortperm(rSC[i, :], rev = true)
        p = p[1:k]
        for j in p
            write(
                io,
                "\"$i\",\"$(data[i,target_col])\",\"$j\",\"$(data[j,target_col])\",\"$(rSC[i,j])\"\n",
            )
        end
    end

    close(io)
end

"""
    write_comprehension_eval(SChat, SC, SC_rest, data, data_rest, target_col, filename)

Write comprehension evaluation into a CSV file for both training and validation
datasets, include target and predicted ids and indentifiers and their
correlations.

# Obligatory Arguments
- `SChat::Matrix`: the Shat/Chat matrix
- `SC::Matrix`: the S/C matrix
- `SC_rest::Matrix`: the rest S/C matrix
- `data::DataFrame`: the data
- `data_rest::DataFrame`: the rest data
- `target_col::Symbol`: the name of target column
- `filename::String`: the filename/filepath

# Optional Arguments
- `k`: top k candidates
- `root_dir::String="."`: dir path for project root dir
- `output_dir::String="."`: output dir inside root dir

# Examples
```julia
JudiLing.write_comprehension_eval(Shat_val, S_val, S_train, latin_val, latin_train,
    :Word, "all_output.csv", k=10, root_dir=@__DIR__, output_dir="out")
```
"""
function write_comprehension_eval(
    SChat,
    SC,
    SC_rest,
    data,
    data_rest,
    target_col,
    filename;
    k = 10,
    root_dir = ".",
    output_dir = "."
    )

    data_combined = copy(data)
    append!(data_combined, data_rest)

    write_comprehension_eval(
        SChat,
        vcat(SC, SC_rest),
        data_combined,
        target_col,
        filename,
        k = k,
        root_dir = root_dir,
        output_dir = output_dir
        )

end
