function display_SC(M, rownames, colnames; nrow=6, ncol=6)
    M_part = M[1:nrow, 1:ncol]
    rownames_part = rownames[1:nrow]
    colnames_part = colnames[1:ncol]

    df = DataFrame()
    df["Data"] = rownames_part
    for (i,coln) in enumerate(colnames_part)
        df[coln] = M_part[:,i]
    end

    display(df)
end