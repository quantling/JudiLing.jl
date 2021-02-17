```@meta
CurrentModule = JudiLing
```

# Cholesky

```@docs
    make_transform_fac
    make_transform_matrix
    make_transform_fac(X::SparseMatrixCSC)
    make_transform_fac(X::Matrix)
    make_transform_matrix(fac::Union{LinearAlgebra.Cholesky, SuiteSparse.CHOLMOD.Factor}, X::Union{SparseMatrixCSC, Matrix}, Y::Union{SparseMatrixCSC, Matrix})
    make_transform_matrix(X::SparseMatrixCSC, Y::Matrix)
    make_transform_matrix(X::Matrix, Y::Union{SparseMatrixCSC, Matrix})
    make_transform_matrix(X::SparseMatrixCSC, Y::SparseMatrixCSC)
    format_matrix(M::Union{SparseMatrixCSC, Matrix}, output_format=:auto)
```