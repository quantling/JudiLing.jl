using SafeTestsets

@safetestset "make cue matrix tests" begin
  include("make_cue_matrix_tests.jl")
end

@safetestset "make semantic matrix tests" begin
  include("make_semantic_matrix_tests.jl")
end