```@meta
CurrentModule = JudiLing
```

# Make Semantic Matrix

## Make binary semantic vectors

```@docs
    PS_Matrix_Struct
    make_pS_matrix
    make_pS_matrix(utterances)
    make_pS_matrix(utterances, utterances_train)
```

## Simulate semantic vectors

```@docs
    L_Matrix_Struct
    make_S_matrix
    make_L_matrix
    make_combined_S_matrix
    make_combined_L_matrix
    make_S_matrix(data::DataFrame, base::Vector, inflections::Vector)
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector)
    make_S_matrix(data::DataFrame, base::Vector)
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector)
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, pyndl_weights::Pyndl_Weight_Struct, n_features_columns::Vector)
    make_S_matrix(data_train::DataFrame, base::Vector, inflections::Vector, L::L_Matrix_Struct)
    make_S_matrix(data_train::DataFrame, data_val::Union{DataFrame, Nothing}, base::Vector, L::L_Matrix_Struct)
    make_S_matrix(data::DataFrame, base::Vector, L::L_Matrix_Struct)
    make_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector, L::L_Matrix_Struct)
    make_L_matrix(data::DataFrame, base::Vector)
    make_combined_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector, L::L_Matrix_Struct)
    make_combined_S_matrix(data_train::DataFrame, data_val::Union{DataFrame, Nothing}, base::Vector, L::L_Matrix_Struct)
    make_combined_S_matrix(  data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector)
    make_combined_S_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector)
    make_combined_L_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector, inflections::Vector)
    make_combined_L_matrix(data_train::DataFrame, data_val::DataFrame, base::Vector)
    L_Matrix_Struct(L, sd_base, sd_base_mean, sd_inflection, sd_inflection_mean, base_f, infl_f, base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)
    L_Matrix_Struct(L, sd_base, sd_inflection, base_f, infl_f, base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)
```

## Load from word2vec, fasttext or similar

```@docs
load_S_matrix_from_fasttext(data::DataFrame,
                                     language::Symbol;
                                     target_col=:Word,
                                     default_file::Int=1)
 load_S_matrix_from_fasttext(data_train::DataFrame,
                                  data_val::DataFrame,
                                  language::Symbol;
                                  target_col=:Word,
                                  default_file::Int=1)
  load_S_matrix_from_word2vec_file(data::DataFrame,
                              filepath::String;
                              target_col=:Word)
  load_S_matrix_from_word2vec_file(data_train::DataFrame,
                              data_val::DataFrame,
                              filepath::String;
                              target_col=:Word)
  load_S_matrix_from_fasttext_file(data::DataFrame,
                              filepath::String;
                              target_col=:Word)
  load_S_matrix_from_fasttext_file(data_train::DataFrame,
                              data_val::DataFrame,
                              filepath::String;
                              target_col=:Word)
```

## Utility functions

```@docs
    process_features(data, feature_cols)
    comp_f_M!(L, sd, sd_mean, n_f, ncol, n_b)
    comp_f_M!(L, sd, n_f, ncol, n_b)
    merge_f2i(base_f2i, infl_f2i, n_base_f, n_infl_f)
    lexome_sum(L, features)
    make_St(L, n, data, base, inflections)
    make_St(L, n, data, base)
    add_St_noise!(St, sd_noise)
    normalize_St!(St, n_base, n_infl)
    normalize_St!(St, n_base)
```
