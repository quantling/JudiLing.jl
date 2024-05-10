```@meta
CurrentModule = JudiLing
```

# Make Cue Matrix

```@docs
    Cue_Matrix_Struct
    make_cue_matrix
    make_combined_cue_matrix
    make_ngrams
    make_cue_matrix(data::DataFrame)
    make_cue_matrix(data::DataFrame, cue_obj::Cue_Matrix_Struct)
    make_cue_matrix(data_train::DataFrame, data_val::DataFrame)
    make_cue_matrix(data::DataFrame, pyndl_weights::Pyndl_Weight_Struct)
    make_combined_cue_matrix(data_train, data_val)
    make_cue_matrix_from_CFBS(features::Vector{Vector{T}};
                                        pad_val::T = 0.,
                                        ncol::Union{Missing,Int}=missing) where {T}
    make_combined_cue_matrix_from_CFBS(features_train::Vector{Vector{T}},
                                                features_test::Vector{Vector{T}};
                                                pad_val::T = 0.,
                                                ncol::Union{Missing,Int}=missing) where {T}
    make_ngrams(tokens, grams, keep_sep, sep_token, start_end_token)
```
