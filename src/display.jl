function display_SC(M, rownames, colnames; nrow=6, ncol=6)
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

function display_SC(data, cue_obj::Cue_Matrix_Struct; nrow=6, ncol=6)
    rownames = data
    colnames = [cue_obj.i2f[i] for i in 1:size(cue_obj.C,2)]

    display_SC(cue_obj.C, rownames, colnames, nrow=nrow, ncol=ncol)
end

function display_SC(data, S::Matrix; nrow=6, ncol=6)
    rownames = data
    colnames = ["S$i" for i in 1:size(S,2)]

    display_SC(S, rownames, colnames, nrow=nrow, ncol=ncol)
end