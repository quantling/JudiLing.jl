using SafeTestsets

@safetestset "cholesky tests" begin
    include("cholesky_tests.jl")
end

@safetestset "display tests" begin
    include("display_tests.jl")
end

@safetestset "eval tests" begin
    include("eval_tests.jl")
end

@safetestset "find_path tests" begin
    include("find_path_tests.jl")
end

@safetestset "make_adjacency_matrix tests" begin
    include("make_adjacency_matrix_tests.jl")
end

@safetestset "make_cue_matrix tests" begin
    include("make_cue_matrix_tests.jl")
end

@safetestset "make_semantic_matrix tests" begin
    include("make_semantic_matrix_tests.jl")
end

@safetestset "make_yt_matrix tests" begin
    include("make_yt_matrix_tests.jl")
end

@safetestset "output_matrix tests" begin
    include("output_tests.jl")
end

@safetestset "preprocess tests" begin
    include("preprocess_tests.jl")
end

@safetestset "test_combo tests" begin
    include("test_combo_tests.jl")
end

@safetestset "wh tests" begin
    include("wh_tests.jl")
end
