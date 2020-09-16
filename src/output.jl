"""
Write results into csv file, support for both results and gold paths' information.
"""
function write2csv end

"""
Create a datafrate for results
"""
function write2df end

"""
    write2csv(::Array{Array{Result_Path_Info_Struct,1},1}, ::DataFrame, ::Cue_Matrix_Struct, ::Cue_Matrix_Struct, ::String) -> ::Nothing

Write results into csv file.

...
# Obligatory Arguments
- `res::Array{Array{Result_Path_Info_Struct,1},1}`: the results from `learn_paths` or `build_paths`
- `data::DataFrame`: the dataset
- `cue_obj_train::Cue_Matrix_Struct`: the cue object for training dataset
- `cue_obj_val::Cue_Matrix_Struct`: the cue object for validation dataset
- `filename::String`: the filename

# Optional Arguments
- `grams::Int64=3`: the number of grams for cues 
- `tokenized::Bool=false`: if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `output_sep_token::Union{String, Char}=""`: output separator
- `path_sep_token::Union{String, Char}=":"`: path separator
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings
- `root_dir::String="."`: dir path for project root dir
- `output_dir::String="."`: output dir inside root dir

# Examples
```julia
#after you had results from learn_paths or build_paths
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
...
"""
function write2csv(
  res::Array{Array{Result_Path_Info_Struct,1},1},
  data::DataFrame,
  cue_obj_train::Cue_Matrix_Struct,
  cue_obj_val::Cue_Matrix_Struct,
  filename::String;
  grams=3::Int64,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  start_end_token="#"::Union{String, Char},
  output_sep_token=""::Union{String, Char, Nothing},
  path_sep_token=":"::Union{String, Char},
  target_col=:Words::Union{String, Symbol},
  root_dir="."::String,
  output_dir="."::String
  )::Nothing
  
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
    grams=grams,
    tokenized=tokenized,
    sep_token=sep_token,
    start_end_token=start_end_token,
    output_sep_token=output_sep_token,
    path_sep_token=path_sep_token,
    target_col=target_col,)

  CSV.write(joinpath(output_path, filename), df, quotestrings=true)
  nothing
end

"""
    write2csv(::Vector{Gold_Path_Info_Struct}, ::String) -> ::Nothing

Write gold paths' information into csv file.

...
# Obligatory Arguments
- `gpi::Vector{Gold_Path_Info_Struct}`: the gold paths' information
- `filename::String`: the filename

# Optional Arguments
- `root_dir::String="."`: dir path for project root dir
- `output_dir::String="."`: output dir inside root dir

# Examples
```julia
#after you had results from learn_paths or build_paths
JudiLing.write2csv(
  gpi_train,
  "gpi_latin_train.csv",
  root_dir=".",
  output_dir="test_out"
  )

JudiLing.write2csv(
  gpi_val,
  "gpi_latin_val.csv",
  root_dir=".",
  output_dir="test_out"
  )
```
...
"""
function write2csv(
  gpi::Vector{Gold_Path_Info_Struct},
  filename::String;
  root_dir="."::String,
  output_dir="."::String
  )::Nothing
  output_path = joinpath(root_dir, output_dir)
  # create path if not exist
  mkpath(output_path)
  # open a file
  io = open(joinpath(output_path, filename), "w")

  # write header
  write(io, "\"utterance\",\"weakest_support\",\"weakest_support_timestep\",\"support\",\"gold_path\",\"timestep_support\"\n")

  for (i,g) in enumerate(gpi)
    ws, wst = findmin(g.ngrams_ind_support)
    write(io, "\"$i\",\"$(ws)\",\"$(wst)\",\"$(g.support)\",\"$(g.ngrams_ind)\",\"$(g.ngrams_ind_support)\"\n")
  end
  # close file
  close(io)
end

"""
    write2df(::Array{Array{Result_Path_Info_Struct,1},1}, ::DataFrame, ::Cue_Matrix_Struct, ::Cue_Matrix_Struct) -> ::DataFrame

Write results into dataframe.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset

# Optional Arguments
- `grams::Int64=3`: the number of grams for cues 
- `tokenized::Bool=false`: if true, the dataset target is assumed to be tokenized
- `sep_token::Union{Nothing, String, Char}=nothing`: separator
- `start_end_token::Union{String, Char}="#"`: start and end token in boundary cues
- `output_sep_token::Union{String, Char}=""`: output separator
- `path_sep_token::Union{String, Char}=":"`: path separator
- `target_col::Union{String, Symbol}=:Words`: the column name for target strings

# Examples
```julia
#after you had results from learn_paths or build_paths
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
...
"""
function write2df(
  res::Array{Array{Result_Path_Info_Struct,1},1},
  data::DataFrame,
  cue_obj_train::Cue_Matrix_Struct,
  cue_obj_val::Cue_Matrix_Struct;
  grams=3::Int64,
  tokenized=false::Bool,
  sep_token=nothing::Union{Nothing, String, Char},
  start_end_token="#"::Union{String, Char},
  output_sep_token=""::Union{String, Char},
  path_sep_token=":"::Union{String, Char},
  target_col=:Words::Union{String, Symbol}
  )::DataFrame

  utterance_vec = Union{Int64,Missing}[]
  identifier_vec = Union{String,Missing}[]
  path_vec = Union{String,Missing}[]
  pred_vec = Union{String,Missing}[]
  num_tolerance_vec = Union{Int64,Missing}[]
  support_vec = Union{AbstractFloat,Missing}[]
  isbest_vec = Union{Bool,Missing}[]
  iscorrect_vec = Union{Bool,Missing}[]
  isnovel_vec = Union{Bool,Missing}[]


  i2f = cue_obj_train.i2f
  for (i,r) in enumerate(res)
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
      path = translate_path(p.ngrams_ind, i2f, sep_token=path_sep_token)
      pred = translate(p.ngrams_ind,
        i2f,
        grams,
        tokenized,
        sep_token,
        start_end_token,
        output_sep_token)
      num_tolerance=p.num_tolerance
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