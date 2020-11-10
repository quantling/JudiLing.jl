```@meta
CurrentModule = JudiLing
```

# Find Paths

```@docs
  Comp_Acc_Struct
  accuracy_comprehension
  eval_SC
  eval_SC(Union{SparseMatrixCSC, Matrix}, Union{SparseMatrixCSC, Matrix})
  eval_SC(SChat,SC,data,target_col)
  eval_SC(SChat,SC,batch_size;verbose=false)
  eval_SC(SChat,SC,data,target_col,batch_size;verbose=false)
  eval_acc(::Array, ::Array)
  eval_acc_loose(::Array, ::Array)
  extract_gpi
  eval_manual
```