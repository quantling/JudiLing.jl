"""
The first part of make transform matrix, usually used by the `learn_paths` function to
save time and computing resources.
"""
function make_transform_fac end

"""
Using Cholesky decomposition to calculate the transformation matrix from S to C
or from C to S.
"""
function make_transform_matrix end

"""
    make_transform_fac(X::SparseMatrixCSC)

Calculate the first step of Cholesky decomposition for sparse matrices.
"""
function make_transform_fac(
    X::SparseMatrixCSC;
    method = :additive,
    shift = 0.02,
    multiplier = 1.01,
)

    XtX = X'X

    if method == :additive
        fac = cholesky(XtX, shift = shift)
    else
        # convert value to Float64
        # otherwise multiplier would raise error
        XtX = convert(SparseMatrixCSC{Float64,Int64}, XtX)
        for i = 1:size(XtX, 2)
            XtX[i, i] *= multiplier
        end
        fac = cholesky(XtX)
    end

    fac
end

"""
    make_transform_fac(X::Matrix)

Calculate the first step of Cholesky decomposition for dense matrices.
"""
function make_transform_fac(
    X::Matrix;
    method = :additive,
    shift = 0.02,
    multiplier = 1.01,
)

    XtX = X'X

    if method == :additive
        fac = cholesky(XtX + shift * I)
    else
        for i = 1:size(XtX, 2)
            XtX[i, i] *= multiplier
        end
        fac = cholesky(XtX)
    end

    fac
end

"""
    make_transform_matrix(fac::Union{LinearAlgebra.Cholesky, SuiteSparse.CHOLMOD.Factor}, X::Union{SparseMatrixCSC, Matrix}, Y::Union{SparseMatrixCSC, Matrix})

Second step in calculating the Cholesky decomposition for the transformation matrix.
"""
function make_transform_matrix(
    fac::Union{LinearAlgebra.Cholesky,SuiteSparse.CHOLMOD.Factor},
    X::Union{SparseMatrixCSC,Matrix},
    Y::Union{SparseMatrixCSC,Matrix};
    output_format = :auto,
    sparse_ratio = 0.05,
    verbose = false,
)

    M = fac \ (X'Y)
    format_matrix(
        M,
        output_format,
        sparse_ratio = sparse_ratio,
        verbose = verbose,
    )
end

"""
    make_transform_matrix(X::SparseMatrixCSC, Y::Matrix)

Use Cholesky decomposition to calculate the transformation matrix from X to Y,
where X is a sparse matrix and Y is a dense matrix.

# Obligatory Arguments
- `X::SparseMatrixCSC`: the X matrix, where X is a sparse matrix
- `Y::Matrix`: the Y matrix, where Y is a dense matrix

# Optional Arguments
- `method::Symbol = :additive`: whether :additive or :multiplicative decomposition is required
- `shift::Float64 = 0.02`: shift value for :additive decomposition
- `multiplier::Float64 = 1.01`: multiplier value for :multiplicative decomposition
- `output_format::Symbol = :auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64 = 0.05`: the ratio to decide whether a matrix is sparse
- `verbose::Bool = false`: if true, more information will be printed out

# Examples
```julia
# additive mode
JudiLing.make_transform_matrix(
    C,
    S,
    method = :additive,
    shift = 0.02,
    verbose = false)

# multiplicative mode
JudiLing.make_transform_matrix(
    C,
    S,
    method = :multiplicative,
    multiplier = 1.01,
    verbose = false)

# further control of sparsity ratio
JudiLing.make_transform_matrix(
  ...
    output_format = :auto,
    sparse_ratio = 0.05,
  ...)
```
"""
function make_transform_matrix(
    X::SparseMatrixCSC,
    Y::Matrix;
    method = :additive,
    shift = 0.02,
    multiplier = 1.01,
    output_format = :auto,
    sparse_ratio = 0.05,
    verbose = false,
)

    XtX = X'X

    if method == :additive
        fac = cholesky(XtX, shift = shift)
    else
        XtX = convert(SparseMatrixCSC{Float64,Int64}, XtX)
        for i = 1:size(XtX, 2)
            XtX[i, i] *= multiplier
        end
        fac = cholesky(XtX)
    end

    # M is in sparse format
    # but sometimes it is actually a dense matrix
    M = fac \ (X'Y)
    format_matrix(
        M,
        output_format,
        sparse_ratio = sparse_ratio,
        verbose = verbose,
    )
end

"""
    make_transform_matrix(X::Matrix, Y::Union{SparseMatrixCSC, Matrix})

Use the Cholesky decomposition to calculate the transformation matrix from X to Y,
where X is a dense matrix and Y is either a dense matrix or a sparse matrix.

# Obligatory Arguments
- `X::Matrix`: the X matrix, where X is a dense matrix
- `Y::Union{SparseMatrixCSC, Matrix}`: the Y matrix, where Y is either a sparse or a dense matrix

# Optional Arguments
- `method::Symbol = :additive`: whether :additive or :multiplicative decomposition is required
- `shift::Float64 = 0.02`: shift value for :additive decomposition
- `multiplier::Float64 = 1.01`: multiplier value for :multiplicative decomposition
- `output_format::Symbol = :auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64 = 0.05`: the ratio to decide whether a matrix is sparse
- `verbose::Bool = false`: if true, more information will be printed out

# Examples
```julia
# additive mode
JudiLing.make_transform_matrix(
    C,
    S,
    method = :additive,
    shift = 0.02,
    verbose = false)

# multiplicative mode
JudiLing.make_transform_matrix(
    C,
    S,
    method=:multiplicative,
    multiplier = 1.01,
    verbose = false)

# further control of sparsity ratio
JudiLing.make_transform_matrix(
    ...
    output_format = :auto,
    sparse_ratio = 0.05,
    ...)
```
"""
function make_transform_matrix(
    X::Matrix,
    Y::Union{SparseMatrixCSC,Matrix};
    method = :additive,
    shift = 0.02,
    multiplier = 1.01,
    output_format = :auto,
    sparse_ratio = 0.05,
    verbose = false,
)

    XtX = X'X

    if method == :additive
        fac = cholesky(XtX + shift * I)
    else
        XtX = convert(SparseMatrixCSC{Float64,Int64}, XtX)
        for i = 1:size(XtX, 2)
            XtX[i, i] *= multiplier
        end
        fac = cholesky(XtX)
    end

    # M is in sparse format
    # but sometimes it is actually a dense matrix
    M = fac \ (X'Y)
    format_matrix(
        M,
        output_format,
        sparse_ratio = sparse_ratio,
        verbose = verbose,
    )
end

"""
    make_transform_matrix(X::SparseMatrixCSC, Y::SparseMatrixCSC)

Use the Cholesky decomposition to calculate the transformation matrix from X to Y,
where X is a sparse matrix and Y is a sparse matrix.

# Obligatory Arguments
- `X::SparseMatrixCSC`: the X matrix, where X is a sparse matrix
- `Y::SparseMatrixCSC`: the Y matrix, where Y is a sparse matrix

# Optional Arguments
- `method::Symbol = :additive`: whether :additive or :multiplicative decomposition is required
- `shift::Float64 = 0.02`: shift value for :additive decomposition
- `multiplier::Float64 = 1.01`: multiplier value for :multiplicative decomposition
- `output_format::Symbol = :auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64 = 0.05`: the ratio to decide whether a matrix is sparse
- `verbose::Bool = false`: if true, more information will be printed out

# Examples
```julia
# additive mode
JudiLing.make_transform_matrix(
    C,
    S,
    method = :additive,
    shift = 0.02,
    verbose = false)

# multiplicative mode
JudiLing.make_transform_matrix(
    C,
    S,
    method = :multiplicative,
    multiplier = 1.01,
    verbose = false)

# further control of sparsity ratio
JudiLing.make_transform_matrix(
    ...
    output_format = :auto,
    sparse_ratio = 0.05,
    ...)
```
"""
function make_transform_matrix(
    X::SparseMatrixCSC,
    Y::SparseMatrixCSC;
    method = :additive,
    shift = 0.02,
    multiplier = 1.01,
    output_format = :auto,
    sparse_ratio = 0.05,
    verbose = false,
)

    XtX = X'X

    if method == :additive
        fac = cholesky(XtX, shift = shift)
    else
        # convert value to Float64
        # otherwise multiplier would raise error
        XtX = convert(SparseMatrixCSC{Float64,Int64}, XtX)
        for i = 1:size(XtX, 2)
            XtX[i, i] *= multiplier
        end
        fac = cholesky(XtX)
    end

    # M is in sparse format
    # but sometimes it is actually a dense matrix
    M = fac \ (X'Y)
    format_matrix(
        M,
        output_format,
        sparse_ratio = sparse_ratio,
        verbose = verbose,
    )
end

"""
    make_transform_matrix(X::Union{SparseMatrixCSC,Matrix},
                            Y::Union{SparseMatrixCSC,Matrix},
                            freq::Union{Array{Int64, 1}, Array{Float64,1}})

Weight X and Y using the frequencies in freq. Then use the Cholesky
decomposition to calculate the transformation matrix from X to Y,
where X is a sparse matrix and Y is a sparse matrix.

# Obligatory Arguments
- `X::SparseMatrixCSC`: the X matrix, where X is a sparse matrix
- `Y::SparseMatrixCSC`: the Y matrix, where Y is a sparse matrix
- `freq::Union{Array{Int64, 1}, Array{Float64,1}}`: list of frequencies of the wordforms in X and Y

# Optional Arguments
- `method::Symbol = :additive`: whether :additive or :multiplicative decomposition is required
- `shift::Float64 = 0.02`: shift value for :additive decomposition
- `multiplier::Float64 = 1.01`: multiplier value for :multiplicative decomposition
- `output_format::Symbol = :auto`: to force output format to dense(:dense) or sparse(:sparse), make it auto(:auto) to determined by the program
- `sparse_ratio::Float64 = 0.05`: the ratio to decide whether a matrix is sparse
- `verbose::Bool = false`: if true, more information will be printed out

# Examples
```julia
JudiLing.make_transform_matrix_frequency(
    C,
    S,
    data.Frequency)
```
"""
function make_transform_matrix(X::Union{SparseMatrixCSC,Matrix},
                             Y::Union{SparseMatrixCSC,Matrix},
                             freq::Union{Array{Int64, 1}, Array{Float64,1}};
                             method = :additive,
                             shift = 0.02,
                             multiplier = 1.01,
                             output_format = :auto,
                             sparse_ratio = 0.05,
                             verbose = false,)
    max_freq, _ = findmax(freq)
    p_sqrt = sqrt.(freq ./ max_freq)

    P_sqrt = spdiagm(0 => p_sqrt)

    X_sch = P_sqrt * X
    Y_sch = P_sqrt * Y
    F_f = make_transform_matrix(X_sch, Y_sch, method=method,
                                shift=shift, multiplier=multiplier,
                                output_format=output_format,
                                sparse_ratio=sparse_ratio,
                                verbose=verbose)
    F_f
end

"""
    format_matrix(M::Union{SparseMatrixCSC, Matrix}, output_format=:auto)

Convert output matrix format to either a dense matrix or a sparse matrix.
"""
function format_matrix(
    M::Union{SparseMatrixCSC,Matrix},
    output_format = :auto;
    sparse_ratio = 0.05,
    verbose = false,
)

    if output_format == :dense
        verbose && println("Returning a dense matrix format")
        Array(M)
    elseif output_format == :sparse
        verbose && println("Returning a sparse matrix format")
        sparse(M)
    else
        verbose && print("Auto mode: ")
        if is_truly_sparse(M, threshold = sparse_ratio, verbose = verbose)
            verbose && println("Returning a sparse matrix format")
            return M
        else
            verbose && println("Returning a dense matrix format")
            return Array(M)
        end
    end
end
