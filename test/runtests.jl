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