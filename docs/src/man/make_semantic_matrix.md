```@meta
CurrentModule = JudiLing
```

# Make Semantic Matrix

```@docs
  PS_Matrix_Struct
  L_Matrix_Struct
  make_pS_matrix
  make_S_matrix
  make_L_matrix
  make_combined_S_matrix
  make_combined_L_matrix
  make_pS_matrix(::DataFrame)
  make_pS_matrix(::DataFrame, ::PS_Matrix_Struct)
  make_S_matrix(::DataFrame, ::Vector, ::Vector)
  make_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector)
  make_S_matrix(::DataFrame, ::Vector)
  make_S_matrix(::DataFrame, ::DataFrame, ::Vector)
  make_S_matrix(::DataFrame, ::Vector, ::Vector, ::L_Matrix_Struct)
  make_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::L_Matrix_Struct)
  make_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector, ::L_Matrix_Struct)
  make_S_matrix(::DataFrame, ::Vector, ::L_Matrix_Struct)
  make_L_matrix(::DataFrame, ::Vector, ::Vector)
  make_L_matrix(::DataFrame, ::Vector)
  make_combined_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector, ::L_Matrix_Struct)
  make_combined_S_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector)
  make_combined_L_matrix(::DataFrame, ::DataFrame, ::Vector)
  make_combined_L_matrix(::DataFrame, ::DataFrame, ::Vector, ::Vector)
  L_Matrix_Struct(L, sd_base, sd_base_mean, sd_inflection, sd_inflection_mean, base_f, infl_f, base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)
  L_Matrix_Struct(L, sd_base, sd_inflection, base_f, infl_f, base_f2i, infl_f2i, n_base_f, n_infl_f, ncol)
  process_features(data, feature_cols)
  comp_f_M!(L, sd, sd_mean, n_f, ncol, n_b)
  comp_f_M!(L, sd, n_f, ncol, n_b)
  merge_f2i(base_f2i, infl_f2i, n_base_f, n_infl_f)
  lexome_sum(L, features)
  make_St(L, n, data, base, inflections)
  make_St(L, n, data, base)
  add_St_noise!(St, sd_noise)
  normalize_St!(St, n_base, n_infl)
  normalize_St!(St, n_base)
```