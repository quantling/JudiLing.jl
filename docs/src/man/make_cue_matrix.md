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
    make_ngrams(tokens, grams=3, keep_sep=false, sep_token=nothing, start_end_token="#")
```