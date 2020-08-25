"""
the first part of make transform matrix, usually in the shuo function to same
time and computing resources
"""
function make_transform_fac end

"""
using cholesky decomposition to calculate transformation matrix from S to C or
from C to S
"""
function make_transform_matrix end

"""
  make_transform_fac(::SparseMatrixCSC)

calculate first part of cholesky decomposition for sparse matrix
"""
function make_transform_fac(
  X::SparseMatrixCSC;
  method=:additive::Symbol,
  shift=0.02::AbstractFloat,
  multiplier=1.01::AbstractFloat
  )::SuiteSparse.CHOLMOD.Factor

  XtX = X'X

  if method == :additive
    fac = cholesky(XtX, shift=shift)
  else
    # convert value to AbstractFloat
    # otherwise multiplier would raise error
    XtX = convert(SparseMatrixCSC{AbstractFloat,Integer}, XtX)
    for i in 1:size(XtX,2)
      XtX[i,i] *= multiplier
    end
    fac = cholesky(XtX)
  end

  fac
end

"""
  make_transform_fac(::Matrix)

calculate first part of cholesky decomposition for dense matrix
"""
function make_transform_fac(
  X::Matrix;
  method=:additive::Symbol,
  shift=0.02::AbstractFloat,
  multiplier=1.01::AbstractFloat
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
  make_transform_matrix(::Union{LinearAlgebra.Cholesky, SuiteSparse.CHOLMOD.Factor}, ::Union{SparseMatrixCSC, Matrix}, ::Union{SparseMatrixCSC, Matrix})

second part of calculate cholesky decomposition transformation matrix

...
# Arguments
- `output_format::Symbol=:auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `verbose::Bool=false`: if verbose, more information will be printed out

# Examples
```julia
  C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
  S = [1 0 1 0; 1 1 0 0; 0 0 1 1]

  JuLDL.make_transform_matrix(C, S)
```
...
"""
function make_transform_matrix(
  fac::Union{LinearAlgebra.Cholesky, SuiteSparse.CHOLMOD.Factor},
  X::Union{SparseMatrixCSC, Matrix},
  Y::Union{SparseMatrixCSC, Matrix};
  output_format=:auto::Symbol,
  verbose=false::Bool
  )::Union{Matrix, SparseMatrixCSC}

  M = fac\(X'Y)
  format_matrix(M, output_format, verbose)
end

"""
  make_transform_matrix(::SparseMatrixCSC, ::Matrix)

using cholesky decomposition to calculate transformation matrix from X to Y,
where X is a sparse matrix and Y is a dense matrix

...
# Arguments
- `method::Symbol=:additive`: shift mode whether :additive or others
- `shift::AbstractFloat=0.02`: shift value
- `multiplier::AbstractFloat=1.01`: multiplier value
- `output_format::Symbol=:auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `verbose::Bool=false`: if verbose, more information will be printed out

# Examples
```julia
  C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
  S = [1 0 1 0; 1 1 0 0; 0 0 1 1]

  JuLDL.make_transform_matrix(C, S)
```
...
"""
function make_transform_matrix(
  X::SparseMatrixCSC,
  Y::Matrix;
  method=:additive::Symbol,
  shift=0.02::AbstractFloat,
  multiplier=1.01::AbstractFloat,
  output_format=:auto::Symbol,
  verbose=false::Bool
  )::Union{SparseMatrixCSC, Matrix}

  XtX = X'X

  if method == :additive
    fac = cholesky(XtX, shift=shift)
  else
    XtX = convert(SparseMatrixCSC{AbstractFloat,Integer}, XtX)
    for i in 1:size(XtX,2)
      XtX[i,i] *= multiplier
    end
    fac = cholesky(XtX)
  end

  # M is in sparse format
  # but sometimes it is actually a dense matrix
  M = fac\(X'Y)
  format_matrix(M, output_format, verbose)
end

"""
  make_transform_matrix(::Matrix, ::Union{SparseMatrixCSC, Matrix})

using cholesky decomposition to calculate transformation matrix from X to Y,
where X is a dense matrix and Y is either a dense matrix or a sparse matrix

...
# Arguments
- `method::Symbol=:additive`: shift mode whether :additive or others
- `shift::AbstractFloat=0.02`: shift value
- `multiplier::AbstractFloat=1.01`: multiplier value
- `output_format::Symbol=:auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `verbose::Bool=false`: if verbose, more information will be printed out

# Examples
```julia
  C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
  S = [1 0 1 0; 1 1 0 0; 0 0 1 1]

  JuLDL.make_transform_matrix(C, S)
```
...
"""
function make_transform_matrix(
  X::Matrix,
  Y::Union{SparseMatrixCSC, Matrix};
  method=:additive::Symbol,
  shift=0.02::AbstractFloat,
  multiplier=1.01::AbstractFloat,
  output_format=:auto::Symbol,
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
  format_matrix(M, output_format, verbose)
end

"""
  make_transform_matrix(::SparseMatrixCSC, ::SparseMatrixCSC)

using cholesky decomposition to calculate transformation matrix from X to Y,
where X is a sparse matrix and Y is a sparse matrix

...
# Arguments
- `method::Symbol=:additive`: shift mode whether :additive or others
- `shift::AbstractFloat=0.02`: shift value
- `multiplier::AbstractFloat=1.01`: multiplier value
- `output_format::Symbol=:auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `verbose::Bool=false`: if verbose, more information will be printed out

# Examples
```julia
  C = [1 1 1 1 0 0 0 0; 1 0 1 0 1 0 1 0; 0 0 0 0 1 1 1 1]
  S = [1 0 1 0; 1 1 0 0; 0 0 1 1]

  JuLDL.make_transform_matrix(C, S)
```
...
"""
function make_transform_matrix(
  X::SparseMatrixCSC,
  Y::SparseMatrixCSC;
  method=:additive::Symbol,
  shift=0.02::AbstractFloat,
  multiplier=1.01::AbstractFloat,
  output_format=:auto::Symbol,
  verbose=false::Bool
  )::Union{SparseMatrixCSC, Matrix}

  XtX = X'X

  if method == :additive
    fac = cholesky(XtX, shift=shift)
  else
    # convert value to AbstractFloat
    # otherwise multiplier would raise error
    XtX = convert(SparseMatrixCSC{AbstractFloat,Integer}, XtX)
    for i in 1:size(XtX,2)
      XtX[i,i] *= multiplier
    end
    fac = cholesky(XtX)
  end

  # M is in sparse format
  # but sometimes it is actually a dense matrix
  M = fac\(X'Y)
  format_matrix(M, output_format, verbose)
end

"""
  format_matrix(::Union{SparseMatrixCSC, Matrix}, ::Symbol, ::Bool)

convert ourput matrix to a dense matrix or sparse matrix
"""
function format_matrix(
  M::Union{SparseMatrixCSC, Matrix},
  output_format=:auto::Symbol,
  verbose=false::Bool
  )::Union{SparseMatrixCSC, Matrix}

  if output_format == :dense
    verbose && println("Returning a dense matrix format")
    Array(M)
  elseif output_format == :sparse
    verbose && println("Returning a sparse matrix format")
    sparse(M)
  else
    if is_truly_sparse(M, verbose=verbose)
      verbose && println("Returning a sparse matrix format")
      return M
    else
      verbose && println("Returning a dense matrix format")
      return Array(M)
    end
  end
end