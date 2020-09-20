"""
The first part of make transform matrix, usually in the learn_paths function to 
same time and computing resources.
"""
function make_transform_fac end

"""
Using Cholesky decomposition to calculate transformation matrix from S to C or
from C to S.
"""
function make_transform_matrix end

"""
    make_transform_fac(::SparseMatrixCSC) -> ::SuiteSparse.CHOLMOD.Factor

Calculate first part of Cholesky decomposition for sparse matrix.
"""
function make_transform_fac(
  X::SparseMatrixCSC;
  method=:additive::Symbol,
  shift=0.02::Float64,
  multiplier=1.01::Float64
  )::SuiteSparse.CHOLMOD.Factor

  XtX = X'X

  if method == :additive
    fac = cholesky(XtX, shift=shift)
  else
    # convert value to Float64
    # otherwise multiplier would raise error
    XtX = convert(SparseMatrixCSC{Float64,Int64}, XtX)
    for i in 1:size(XtX,2)
      XtX[i,i] *= multiplier
    end
    fac = cholesky(XtX)
  end

  fac
end

"""
    make_transform_fac(::Matrix) -> ::LinearAlgebra.Cholesky

Calculate first part of Cholesky decomposition for dense matrix.
"""
function make_transform_fac(
  X::Matrix;
  method=:additive::Symbol,
  shift=0.02::Float64,
  multiplier=1.01::Float64
  )::LinearAlgebra.Cholesky

  XtX = X'X

  if method == :additive
    fac = cholesky(XtX + shift * I)
  else
    for i in 1:size(XtX,2)
      XtX[i,i] *= multiplier
    end
    fac = cholesky(XtX)
  end

  fac
end

"""
    make_transform_matrix(::Union{LinearAlgebra.Cholesky, SuiteSparse.CHOLMOD.Factor}, ::Union{SparseMatrixCSC, Matrix}, ::Union{SparseMatrixCSC, Matrix}) -> ::Union{Matrix, SparseMatrixCSC}

Second part of calculate Cholesky decomposition transformation matrix.
"""
function make_transform_matrix(
  fac::Union{LinearAlgebra.Cholesky, SuiteSparse.CHOLMOD.Factor},
  X::Union{SparseMatrixCSC, Matrix},
  Y::Union{SparseMatrixCSC, Matrix};
  output_format=:auto::Symbol,
  sparse_ratio=0.2::Float64,
  verbose=false::Bool
  )::Union{Matrix, SparseMatrixCSC}

  M = fac\(X'Y)
  format_matrix(M, output_format, sparse_ratio=sparse_ratio, verbose=verbose)
end

"""
    make_transform_matrix(::SparseMatrixCSC, ::Matrix) -> ::Union{SparseMatrixCSC, Matrix}

Using Cholesky decomposition to calculate transformation matrix from X to Y,
where X is a sparse matrix and Y is a dense matrix.

...
# Obligatory Arguments
- `X::SparseMatrixCSC`: the X matrix, where X is a sparse matrix
- `Y::Matrix`: the Y matrix, where Y is a dense matrix

# Optional Arguments
- `method::Symbol=:additive`: shift mode whether :additive or :multiplicative
- `shift::Float64=0.02`: shift value
- `multiplier::Float64=1.01`: multiplier value
- `output_format::Symbol=:auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64=0.2`: the ratio to decide whether a matrix is sparse
- `verbose::Bool=false`: if true, more information will be printed out

# Examples
```julia
# additive mode
JudiLing.make_transform_matrxi(
  C,
  S,
  method=:additive,
  shift=0.02,
  verbose=false)

# multiplicative mode
JudiLing.make_transform_matrix(
  C,
  S,
  method=:multiplicative,
  multiplier=1.01,
  verbose=false)
```

# further control of sparsity ratio
JudiLing.make_transform_matrix(
  ...
  output_format=:auto,
  sparse_ratio=0.2,
  ...)
...
"""
function make_transform_matrix(
  X::SparseMatrixCSC,
  Y::Matrix;
  method=:additive::Symbol,
  shift=0.02::Float64,
  multiplier=1.01::Float64,
  output_format=:auto::Symbol,
  sparse_ratio=0.2::Float64,
  verbose=false::Bool
  )::Union{SparseMatrixCSC, Matrix}

  XtX = X'X

  if method == :additive
    fac = cholesky(XtX, shift=shift)
  else
    XtX = convert(SparseMatrixCSC{Float64,Int64}, XtX)
    for i in 1:size(XtX,2)
      XtX[i,i] *= multiplier
    end
    fac = cholesky(XtX)
  end

  # M is in sparse format
  # but sometimes it is actually a dense matrix
  M = fac\(X'Y)
  format_matrix(M, output_format, sparse_ratio=sparse_ratio, verbose=verbose)
end

"""
    make_transform_matrix(::Matrix, ::Union{SparseMatrixCSC, Matrix}) -> ::Union{SparseMatrixCSC, Matrix}

Using Cholesky decomposition to calculate transformation matrix from X to Y,
where X is a dense matrix and Y is either a dense matrix or a sparse matrix.

...
# Obligatory Arguments
- `X::Matrix`: the X matrix, where X is a dense matrix
- `Y::Union{SparseMatrixCSC, Matrix}`: the Y matrix, where Y is either a sparse or a dense matrix

# Optional Arguments
- `method::Symbol=:additive`: shift mode whether :additive or :multiplicative
- `shift::Float64=0.02`: shift value
- `multiplier::Float64=1.01`: multiplier value
- `output_format::Symbol=:auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64=0.2`: the ratio to decide whether a matrix is sparse
- `verbose::Bool=false`: if true, more information will be printed out

# Examples
```julia
# additive mode
JudiLing.make_transform_matrix(
  C,
  S,
  method=:additive,
  shift=0.02,
  verbose=false)

# multiplicative mode
JudiLing.make_transform_matrix(
  C,
  S,
  method=:multiplicative,
  multiplier=1.01,
  verbose=false)

# further control of sparsity ratio
JudiLing.make_transform_matrix(
  ...
  output_format=:auto,
  sparse_ratio=0.2,
  ...)
```
...
"""
function make_transform_matrix(
  X::Matrix,
  Y::Union{SparseMatrixCSC, Matrix};
  method=:additive::Symbol,
  shift=0.02::Float64,
  multiplier=1.01::Float64,
  output_format=:auto::Symbol,
  sparse_ratio=0.2::Float64,
  verbose=false::Bool
  )::Union{SparseMatrixCSC, Matrix}

  XtX = X'X

  if method == :additive
    fac = cholesky(XtX + shift * I)
  else
    for i in 1:size(XtX,2)
      XtX[i,i] *= multiplier
    end
    fac = cholesky(XtX)
  end

  # M is in sparse format
  # but sometimes it is actually a dense matrix
  M = fac\(X'Y)
  format_matrix(M, output_format, sparse_ratio=sparse_ratio, verbose=verbose)
end

"""
    make_transform_matrix(::SparseMatrixCSC, ::SparseMatrixCSC) -> ::Union{SparseMatrixCSC, Matrix}

Using Cholesky decomposition to calculate transformation matrix from X to Y,
where X is a sparse matrix and Y is a sparse matrix.

...
# Obligatory Arguments
- `X::SparseMatrixCSC`: the X matrix, where X is a sparse matrix
- `Y::SparseMatrixCSC`: the Y matrix, where Y is a sparse matrix

# Optional Arguments
- `method::Symbol=:additive`: shift mode whether :additive or :multiplicative
- `shift::Float64=0.02`: shift value
- `multiplier::Float64=1.01`: multiplier value
- `output_format::Symbol=:auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64=0.2`: the ratio to decide whether a matrix is sparse
- `verbose::Bool=false`: if true, more information will be printed out

# Examples
```julia
# additive mode
JudiLing.make_transform_matrix(
  C,
  S,
  method=:additive,
  shift=0.02,
  verbose=false)

# multiplicative mode
JudiLing.make_transform_matrix(
  C,
  S,
  method=:multiplicative,
  multiplier=1.01,
  verbose=false)

# further control of sparsity ratio
JudiLing.make_transform_matrix(
  ...
  output_format=:auto,
  sparse_ratio=0.2,
  ...)
```
...
"""
function make_transform_matrix(
  X::SparseMatrixCSC,
  Y::SparseMatrixCSC;
  method=:additive::Symbol,
  shift=0.02::Float64,
  multiplier=1.01::Float64,
  output_format=:auto::Symbol,
  sparse_ratio=0.2::Float64,
  verbose=false::Bool
  )::Union{SparseMatrixCSC, Matrix}

  XtX = X'X

  if method == :additive
    fac = cholesky(XtX, shift=shift)
  else
    # convert value to Float64
    # otherwise multiplier would raise error
    XtX = convert(SparseMatrixCSC{Float64,Int64}, XtX)
    for i in 1:size(XtX,2)
      XtX[i,i] *= multiplier
    end
    fac = cholesky(XtX)
  end

  # M is in sparse format
  # but sometimes it is actually a dense matrix
  M = fac\(X'Y)
  format_matrix(M, output_format, sparse_ratio=sparse_ratio, verbose=verbose)
end

"""
    format_matrix(::Union{SparseMatrixCSC, Matrix}, ::Symbol) -> ::Union{SparseMatrixCSC, Matrix}

Convert output matrix format to either a dense matrix or a sparse matrix.
"""
function format_matrix(
  M::Union{SparseMatrixCSC, Matrix},
  output_format=:auto::Symbol;
  sparse_ratio=0.2::Float64,
  verbose=false::Bool
  )::Union{SparseMatrixCSC, Matrix}

  if output_format == :dense
    verbose && println("Returning a dense matrix format")
    Array(M)
  elseif output_format == :sparse
    verbose && println("Returning a sparse matrix format")
    sparse(M)
  else
    verbose && print("Auto mode: ")
    if is_truly_sparse(M, threshold=sparse_ratio, verbose=verbose)
      verbose && println("Returning a sparse matrix format")
      return M
    else
      verbose && println("Returning a dense matrix format")
      return Array(M)
    end
  end
end