```@meta
CurrentModule = JudiLing
```

# Find Paths

```@docs
  Result_Path_Info_Struct
  Gold_Path_Info_Struct
  shuo
  hua
  shuo(::DataFrame,::DataFrame,::SparseMatrixCSC,::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Matrix,::SparseMatrixCSC,::Dict)
  hua(::DataFrame,::SparseMatrixCSC,::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Matrix,::SparseMatrixCSC,::Dict,::Array)
  eval_can(::Vector{Vector{Tuple{Vector{Integer}, Integer}}},::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Dict,::Integer,::Bool)
  find_top_feature_indices(::Matrix, ::Array)
```