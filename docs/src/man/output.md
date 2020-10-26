```@meta
CurrentModule = JudiLing
```

# Output

```@docs
  write2csv
  write2df
  save_L_matrix
  load_L_matrix
  write2csv(::Array{Array{Result_Path_Info_Struct,1},1}, ::DataFrame, ::Cue_Matrix_Struct, ::Cue_Matrix_Struct, ::String)
  write2csv(::Vector{Gold_Path_Info_Struct}, ::String)
  write2df(::Array{Array{Result_Path_Info_Struct,1},1}, ::DataFrame, ::Cue_Matrix_Struct, ::Cue_Matrix_Struct)
  write2df(::gpi::Vector{Gold_Path_Info_Struct})
  save_L_matrix(::L_Matrix_Struct,::String)
  load_L_matrix(::String)

```