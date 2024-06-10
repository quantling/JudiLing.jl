```@meta
CurrentModule = JudiLing
```

# Loading data

```@docs
load_dataset(filepath::String;
            delim::String=",",
            kargs...)
loading_data_randomly_split(
      data_path::String,
      output_dir_path::String,
      data_prefix::String;
      val_sample_size::Int = 0,
      val_ratio::Float = 0.0,
      random_seed::Int = 314)
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
```
