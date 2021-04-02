```@meta
CurrentModule = JudiLing
```

# Evaluation

```@docs
    Comp_Acc_Struct
    eval_SC
    eval_SC_loose
    accuracy_comprehension(S, Shat, data)
    eval_SC(SChat::AbstractArray, SC::AbstractArray)
    eval_SC(SChat::AbstractArray, SC::AbstractArray, SC_rest::AbstractArray)
    eval_SC(SChat::AbstractArray, SC::AbstractArray, data::DataFrame, target_col::Union{String, Symbol})
    eval_SC(SChat::AbstractArray, SC::AbstractArray, SC_rest::AbstractArray, data::DataFrame, data_rest::DataFrame, target_col::Union{String, Symbol})
    eval_SC(SChat::AbstractArray, SC::AbstractArray, batch_size::Int64)
    eval_SC(SChat::AbstractArray, SC::AbstractArray, data::DataFrame, target_col::Union{String, Symbol}, batch_size::Int64)
    eval_SC_loose(SChat, SC, k)
    eval_SC_loose(SChat, SC, k, data, target_col)
    eval_manual(res, data, i2f)
    eval_acc(res, gold_inds::Array)
    eval_acc(res, cue_obj::Cue_Matrix_Struct)
    eval_acc_loose(res, gold_inds)
    extract_gpi(gpi, threshold=0.1, tolerance=(-1000.0))
```