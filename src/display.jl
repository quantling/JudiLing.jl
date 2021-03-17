"""
display_matrix(M, rownames, colnames)

Display matrix with rownames and colnames.
"""
function display_matrix(M, rownames, colnames; nrow=6, ncol=6)
    M_part = M[1:nrow, 1:ncol]
    rownames_part = rownames[1:nrow]
    colnames_part = colnames[1:ncol]

    df = DataFrame()
    df[!, :Data] = rownames_part
    for (i,coln) in enumerate(colnames_part)
        df[!, coln] = M_part[:,i]
    end

    display(df)
end

"""
display_matrix(data, target_col, cue_obj, M, M_type)

Display matrix with rownames and colnames.

...
# Obligatory Arguments
- `data::DataFrame`: the dataset
- `target_col::Union{String, Symbol}`: the target column name
- `cue_obj::Cue_Matrix_Struct`: the cue matrix structure
- `M::Union{SparseMatrixCSC, Matrix}`: the matrix
- `M_type::Union{String, Symbol}`: the type of the matrix, currently support :C, :S, :F, :G, :Chat, :Shat, :A and :R

# Optional Arguments
- `nrow::Int64 = 6`: the number of rows to display
- `ncol::Int64 = 6`: the number of columns to display

# Examples
```julia
JudiLing.display_matrix(latin, :Word, cue_obj, cue_obj.C, :C)
JudiLing.display_matrix(latin, :Word, cue_obj, S, :S)
JudiLing.display_matrix(latin, :Word, cue_obj, G, :G)
JudiLing.display_matrix(latin, :Word, cue_obj, Chat, :Chat)
JudiLing.display_matrix(latin, :Word, cue_obj, F, :F)
JudiLing.display_matrix(latin, :Word, cue_obj, Shat, :Shat)
JudiLing.display_matrix(latin, :Word, cue_obj, A, :A)
JudiLing.display_matrix(latin, :Word, cue_obj, R, :R)
```
...
"""
function display_matrix(
    data,
    target_col,
    cue_obj,
    M,
    M_type;
    nrow = 6,
    ncol = 6
    )
    
    if M_type == :C || M_type == "C"
        rownames = data[:,target_col]
        colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
    elseif M_type == :Chat || M_type == "Chat"
        rownames = data[:,target_col]
        colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
    elseif M_type == :S || M_type == "S"
        rownames = data[:,target_col]
        colnames = ["S$i" for i in 1:size(M,2)]
    elseif M_type == :Shat || M_type == "Shat"
        rownames = data[:,target_col]
        colnames = ["S$i" for i in 1:size(M,2)]
    elseif M_type == :F || M_type == "F"
        rownames = colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
        colnames = ["S$i" for i in 1:size(M,2)]
    elseif M_type == :G || M_type == "G"
        rownames = ["S$i" for i in 1:size(M,2)]
        colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
    elseif M_type == :A || M_type == "A"
        rownames = [cue_obj.i2f[i] for i in 1:size(M,2)]
        colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
    elseif M_type == :R || M_type == "R"
        rownames = data[:,target_col]
        colnames = data[:,target_col]
    else
        throw(ArgumentError("type is incorrect, using :C," * 
            " :S, :Chat, :Shat, :F, :G, :A and :R"))
    end

    display_matrix(M, rownames, colnames, nrow=nrow, ncol=ncol)
end