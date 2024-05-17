"""
    load_dataset(filepath::String;
                delim::String=",",
                kargs...)

Load a dataset from file, usually comma- or tab-separated.
Returns a DataFrame.

# Obligatory arguments
- `filepath::String`: Path to file to be loaded.

# Optional arguments
- `delim::String=","`: Delimiter in the file (usually either `","` or `"\\t"`).
- `kargs...`: Further keyword arguments are passed to `CSV.File()`.

# Example
```julia
latin = JudiLing.load_dataset("latin.csv")
first(latin, 10)
```
"""
function load_dataset(filepath::String;
                        delim::String=",",
                        kargs...)
    return(DataFrame(CSV.File(filepath, stringtype=String, delim=delim; kargs...)))
end


"""
    load_data_random_split(args...; kargs...)

Alias for loading_data_randomly_split
"""
function load_data_random_split(
    args...; kargs...)
    JudiLing.loading_data_randomly_split(args...; kargs...)
end

"""
    loading_data_random_split(args...; kargs...)

Alias for loading_data_randomly_split
"""
function loading_data_random_split(
    args...; kargs...)
    JudiLing.loading_data_randomly_split(args...; kargs...)
end

"""
    load_data_randomly_split(args...; kargs...)

Alias for loading_data_randomly_split
"""
function load_data_randomly_split(
    args...; kargs...)
    JudiLing.loading_data_randomly_split(args...; kargs...)
end

"""
    loading_data_randomly_split(
        data_path::String,
        output_dir_path::String,
        data_prefix::String;
        val_sample_size::Int = 0,
        val_ratio::Float64 = 0.0,
        random_seed::Int = 314)

Read in a dataframe, splitting the dataframe into a training and validation dataset. The two are also written to `output_dir_path` at the same time.

!!! note
    The order of `data_prefix` and `output_dir_path` is exactly reversed compared to `loading_data_careful_split`.


# Obligatory arguments
- `data_path::String`: Path to where the dataset is stored.
- `output_dir_path::String`: Path to where the new dataframes should be stored.
- `data_prefix::String`: Prefix of the two new files, will be called `data_prefix_train.csv` and `data_prefix_val.csv`.

# Optional arguments
- `val_sample_size::Int = 0`: Size of the validation dataset (only `val_sample_size` or `val_ratio` may be used).
- `val_ratio::Float64 = 0.0`: Fraction of the data that should be in the validation dataset (only `val_sample_size` or `val_ratio` may be used).
- `random_seed::Int = 314`: Random seed for controlling random split.

# Example
```
data_train, data_val = JudiLing.loading_data_randomly_split(
    "latin.csv",
    "careful",
    "latin",
    ["Lexeme","Person","Number","Tense","Voice","Mood"]
)
```
"""
function loading_data_randomly_split(
    data_path::String,
    output_dir_path::String,
    data_prefix::String;
    train_sample_size::Int = 0,
    val_sample_size::Int = 0,
    val_ratio::Float64 = 0.0,
    random_seed::Int = 314,
    verbose::Bool = false)
    verbose && println("Spliting data...")

    train_val_random_split(
        data_path,
        output_dir_path,
        data_prefix,
        train_sample_size = train_sample_size,
        val_sample_size = val_sample_size,
        val_ratio = val_ratio,
        random_seed = random_seed,
        verbose = verbose,
        )

    # load data
    verbose && println("Loading CSV...")
    loading_data_pre_split(output_dir_path, data_prefix)
end

"""
    load_data_careful_split(args...; kargs...)

Alias for load_data_carefully_split
"""
function load_data_careful_split(
    args...; kargs...)
    JudiLing.loading_data_careful_split(args...; kargs...)
end

"""
    load_data_carefully_split(args...; kargs...)

Alias for load_data_carefully_split
"""
function load_data_carefully_split(
    args...; kargs...)
    JudiLing.loading_data_careful_split(args...; kargs...)
end

"""
    loading_data_carefully_split(args...; kargs...)

Alias for load_data_carefully_split
"""
function loading_data_carefully_split(
    args...; kargs...)
    JudiLing.loading_data_careful_split(args...; kargs...)
end

"""
    loading_data_careful_split(
        data_path::String,
        data_prefix::String,
        output_dir_path::String,
        n_features_columns::Union{Vector{Symbol},Vector{String}};
        train_sample_size::Int = 0,
        val_sample_size::Int = 0,
        val_ratio::Float64 = 0.0,
        n_grams_target_col::Union{Symbol, String} = :Word,
        n_grams_tokenized::Bool = false,
        n_grams_sep_token::Union{Nothing, String} = nothing,
        grams::Int = 3,
        n_grams_keep_sep::Bool = false,
        start_end_token::String = "#",
        random_seed::Int = 314,
        verbose::Bool = false)

Read in a dataframe, splitting the dataframe into a training and validation dataset. The split is done such that all features in the columns specified
in `n_features_columns` occur both in the training and validation data. It is also ensured that the unique grams resulting from splitting the strings in column
`n_grams_target_col` into `grams`-grams occur in both datasets.
The two are also written to `output_dir_path` at the same time.

!!! note
    The order of `data_prefix` and `output_dir_path` is exactly reversed compared to `loading_data_randomly_split`.

# Obligatory arguments
- `data_path::String`: Path to where the dataset is stored.
- `output_dir_path::String`: Path to where the new dataframes should be stored.
- `data_prefix::String`: Prefix of the two new files, will be called `data_prefix_train.csv` and `data_prefix_val.csv`.
- `n_features_columns::Vector{Union{Symbol, String}}`: Vector with columns whose features have to occur in both the training and validation data.

# Optional arguments
- `val_sample_size::Int = 0`: Size of the validation dataset (only `val_sample_size` or `val_ratio` may be used).
- `val_ratio::Float64 = 0.0`: Fraction of the data that should be in the validation dataset (only `val_sample_size` or `val_ratio` may be used).
- `n_grams_target_col::Union{Symbol, String} = :Word`: Column with target words.
- `n_grams_tokenized::Bool = false`: Whether the words in `n_grams_target_col` are already tokenized.
- `n_grams_sep_token::Union{Nothing, String} = nothing`: String with which tokens in `n_grams_target_col` are separated (only used if `n_grams_tokenized=true`).
- `grams::Int = 3`: Granularity of the n-grams.
- `n_grams_keep_sep::Bool = false`: Whether the token separators should be kept in the ngrams (this is useful e.g. when working with syllables).
- `start_end_token::String = "#"`: Token with which the start and end of words should be marked.
- `random_seed::Int = 314`: Random seed for controlling random split.

# Example
```
data_train, data_val = JudiLing.loading_data_careful_split(
    "latin.csv",
    "latin",
    "careful",
    ["Lexeme","Person","Number","Tense","Voice","Mood"]
)
```
"""
function loading_data_careful_split(
    data_path::String,
    data_prefix::String,
    output_dir_path::String,
    n_features_columns::Union{Vector{Symbol},Vector{String}};
    train_sample_size::Int = 0,
    val_sample_size::Int = 0,
    val_ratio::Float64 = 0.0,
    n_grams_target_col::Union{Symbol, String} = :Word,
    n_grams_tokenized::Bool = false,
    n_grams_sep_token::Union{Nothing, String} = nothing,
    grams::Int = 3,
    n_grams_keep_sep::Bool = false,
    start_end_token::String = "#",
    random_seed::Int = 314,
    verbose::Bool = false)

    verbose && println("Splitting data...")
    train_val_careful_split(
        data_path,
        output_dir_path,
        data_prefix,
        n_features_columns,
        train_sample_size = train_sample_size,
        val_sample_size = val_sample_size,
        val_ratio = val_ratio,
        n_grams_target_col = n_grams_target_col,
        n_grams_tokenized = n_grams_tokenized,
        n_grams_sep_token = n_grams_sep_token,
        grams = grams,
        n_grams_keep_sep = n_grams_keep_sep,
        start_end_token = start_end_token,
        random_seed = random_seed,
        verbose = verbose,
    )

    # load data
    verbose && println("Loading CSV...")
    loading_data_pre_split(output_dir_path, data_prefix)
end
