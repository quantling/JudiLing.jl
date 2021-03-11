function display_matrix(M, rownames, colnames; nrow=6, ncol=6)
    M_part = M[1:nrow, 1:ncol]
    rownames_part = rownames[1:nrow]
    colnames_part = colnames[1:ncol]

    df = DataFrame()
    df[!, "Data"] = rownames_part
    for (i,coln) in enumerate(colnames_part)
        df[!, coln] = M_part[:,i]
    end

    display(df)
end

function display_matrix(
    data,
    target_col,
    cue_obj,
    M,
    type;
    nrow=6,
    ncol=6
    )
    
    if type == :C || type == ["C"]
        rownames = data[:,target_col]
        colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
    elseif type == :Chat || type == ["Chat"]
        rownames = data[:,target_col]
        colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
    elseif type == :S || type == ["S"]
        rownames = data[:,target_col]
        colnames = ["S$i" for i in 1:size(M,2)]
    elseif type == :Shat || type == ["Shat"]
        rownames = data[:,target_col]
        colnames = ["S$i" for i in 1:size(M,2)]
    elseif type == :F || type == ["F"]
        rownames = colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
        colnames = ["S$i" for i in 1:size(M,2)]
    elseif type == :G || type == ["G"]
        rownames = ["S$i" for i in 1:size(M,2)]
        colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
    elseif type == :A || type == ["A"]
        rownames = [cue_obj.i2f[i] for i in 1:size(M,2)]
        colnames = [cue_obj.i2f[i] for i in 1:size(M,2)]
    elseif type == :R || type == ["R"]
        rownames = data[:,target_col]
        colnames = data[:,target_col]
    else
        throw(ArgumentError("type is incorrect, using :C," * 
            " :S, :Chat, :Shat, :F, :G, :A and :R"))
    end

    display_matrix(M, rownames, colnames, nrow=nrow, ncol=ncol)
end