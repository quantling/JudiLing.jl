```@meta
CurrentModule = JuLDL
```

# Cholesky

```@docs
  make_transform_fac
  make_transform_matrix
  make_transform_fac(::SparseMatrixCSC)
  make_transform_fac(::Matrix)
  make_transform_matrix(::Union{LinearAlgebra.Cholesky, SuiteSparse.CHOLMOD.Factor}, ::Union{SparseMatrixCSC, Matrix}, ::Union{SparseMatrixCSC, Matrix})
  make_transform_matrix(::SparseMatrixCSC, ::Matrix)
  make_transform_matrix(::Matrix, ::Union{SparseMatrixCSC, Matrix})
  make_transform_matrix(::SparseMatrixCSC, ::SparseMatrixCSC)
  format_matrix(::Union{SparseMatrixCSC, Matrix}, ::Symbol, ::Bool)
```