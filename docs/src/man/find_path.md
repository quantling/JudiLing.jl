```@meta
CurrentModule = JudiLing
```

# Find Paths

## Structures

```@docs
    Result_Path_Info_Struct
    Gold_Path_Info_Struct
    Threshold_Stat_Struct
```

## Build paths

```@docs
    build_paths
    build_paths(
        data_val,
        C_train,
        S_val,
        F_train,
        Chat_val,
        A,
        i2f,
        C_train_ind;
        rC = nothing,
        max_t = 15,
        max_can = 10,
        n_neighbors = 10,
        grams = 3,
        tokenized = false,
        sep_token = nothing,
        target_col = :Words,
        start_end_token = "#",
        if_pca = false,
        pca_eval_M = nothing,
        ignore_nan = true,
        verbose = false,
    )
```

## Learn paths

```@docs
    learn_paths
    learn_paths(
        data::DataFrame,
        cue_obj::Cue_Matrix_Struct,
        S_val::Union{SparseMatrixCSC, Matrix},
        F_train,
        Chat_val::Union{SparseMatrixCSC, Matrix};
        Shat_val::Union{Nothing, Matrix} = nothing,
        check_gold_path::Bool = false,
        threshold::Float64 = 0.1,
        is_tolerant::Bool = false,
        tolerance::Float64 = (-1000.0),
        max_tolerance::Int = 3,
        activation::Union{Nothing, Function} = nothing,
        ignore_nan::Bool = true,
        verbose::Bool = true)
    learn_paths(
        data_train::DataFrame,
        data_val::DataFrame,
        C_train::Union{Matrix, SparseMatrixCSC},
        S_val::Union{Matrix, SparseMatrixCSC},
        F_train,
        Chat_val::Union{Matrix, SparseMatrixCSC},
        A::SparseMatrixCSC,
        i2f::Dict,
        f2i::Dict;
        gold_ind::Union{Nothing, Vector} = nothing,
        Shat_val::Union{Nothing, Matrix} = nothing,
        check_gold_path::Bool = false,
        max_t::Int = 15,
        max_can::Int = 10,
        threshold::Float64 = 0.1,
        is_tolerant::Bool = false,
        tolerance::Float64 = (-1000.0),
        max_tolerance::Int = 3,
        grams::Int = 3,
        tokenized::Bool = false,
        sep_token::Union{Nothing, String} = nothing,
        keep_sep::Bool = false,
        target_col::Union{Symbol, String} = "Words",
        start_end_token::String = "#",
        issparse::Union{Symbol, Bool} = :auto,
        sparse_ratio::Float64 = 0.05,
        if_pca::Bool = false,
        pca_eval_M::Union{Nothing, Matrix} = nothing,
        activation::Union{Nothing, Function} = nothing,
        ignore_nan::Bool = true,
        check_threshold_stat::Bool = false,
        verbose::Bool = false
    )
    learn_paths_rpi(
        data_train::DataFrame,
        data_val::DataFrame,
        C_train::Union{Matrix, SparseMatrixCSC},
        S_val::Union{Matrix, SparseMatrixCSC},
        F_train,
        Chat_val::Union{Matrix, SparseMatrixCSC},
        A::SparseMatrixCSC,
        i2f::Dict,
        f2i::Dict;
        gold_ind::Union{Nothing, Vector} = nothing,
        Shat_val::Union{Nothing, Matrix} = nothing,
        check_gold_path::Bool = false,
        max_t::Int = 15,
        max_can::Int = 10,
        threshold::Float64 = 0.1,
        is_tolerant::Bool = false,
        tolerance::Float64 = (-1000.0),
        max_tolerance::Int = 3,
        grams::Int = 3,
        tokenized::Bool = false,
        sep_token::Union{Nothing, String} = nothing,
        keep_sep::Bool = false,
        target_col::Union{Symbol, String} = "Words",
        start_end_token::String = "#",
        issparse::Union{Symbol, Bool} = :auto,
        sparse_ratio::Float64 = 0.05,
        if_pca::Bool = false,
        pca_eval_M::Union{Nothing, Matrix} = nothing,
        activation::Union{Nothing, Function} = nothing,
        ignore_nan::Bool = true,
        check_threshold_stat::Bool = false,
        verbose::Bool = false
    )
```

## Utility functions

```@docs
    eval_can(candidates, S, F, i2f, max_can, if_pca, pca_eval_M)
    find_top_feature_indices(rC, C_train_ind)
    make_ngrams_ind(res, n)
    predict_shat(F::Union{Matrix, SparseMatrixCSC},
                          ci::Vector{Int})
```
