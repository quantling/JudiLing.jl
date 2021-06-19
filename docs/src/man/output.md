```@meta
CurrentModule = JudiLing
```

# Output

```@docs
    write2csv
    write2df
    write2csv(res, data, cue_obj_train, cue_obj_val, filename)
    write2csv(gpi::Vector{Gold_Path_Info_Struct}, filename)
    write2csv(ts::Threshold_Stat_Struct, filename)
    write2df(res, data, cue_obj_train, cue_obj_val)
    write2df(gpi::Vector{Gold_Path_Info_Struct})
    write2df(ts::Threshold_Stat_Struct)
    save_L_matrix(L, filename)
    load_L_matrix(filename)
    save_S_matrix(S, filename, data, target_col)
    load_S_matrix(filename)
```