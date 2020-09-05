```@meta
CurrentModule = JudiLing
```

# Find Paths

```@docs
  Result_Path_Info_Struct
  Gold_Path_Info_Struct
  learn_paths
  build_paths
  learn_paths(::DataFrame,::DataFrame,::SparseMatrixCSC,::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Matrix,::SparseMatrixCSC,::Dict)
  build_paths(::DataFrame,::SparseMatrixCSC,::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Matrix,::SparseMatrixCSC,::Dict,::Array)
  eval_can(::Vector{Vector{Tuple{Vector{Int64}, Int64}}},::Union{SparseMatrixCSC, Matrix},::Union{SparseMatrixCSC, Matrix},::Dict,::Int64,::Bool)
  find_top_feature_indices(::Matrix, ::Array)
```