using SafeTestsets

@safetestset "make cue matrix tests" begin
  include("make_cue_matrix_tests.jl")
end

@safetestset "make semantic matrix tests" begin
  include("make_semantic_matrix_tests.jl")
end

@safetestset "cholesky tests" begin
  include("cholesky_tests.jl")
end

@safetestset "make adjacency matrix tests" begin
  include("make_adjacency_matrix_tests.jl")
end

@safetestset "make yt matrix tests" begin
  include("make_yt_matrix_tests.jl")
end

@safetestset "find path tests" begin
  include("find_path_tests.jl")
end

@safetestset "eval tests" begin
  include("eval_tests.jl")
end

@safetestset "output tests" begin
  include("output_tests.jl")
end

@safetestset "test_combo tests" begin
  include("test_combo_tests.jl")
end