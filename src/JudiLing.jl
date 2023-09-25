module JudiLing

using DataFrames
using Random, Distributions
using SparseArrays, LinearAlgebra, Statistics, SuiteSparse
using BenchmarkTools
using DataStructures
using ProgressMeter
using CSV
using GZip
using Embeddings
using Requires

function __init__()
    @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin
     using .PyCall
     @eval include("pickle.jl")
     @eval include("pyndl.jl")
 end
end

include("utils.jl")
include("wh.jl")
include("make_cue_matrix.jl")
include("make_semantic_matrix.jl")
include("cholesky.jl")
include("make_adjacency_matrix.jl")
include("make_yt_matrix.jl")
include("find_path.jl")
include("eval.jl")
include("output.jl")
include("preprocess.jl")
include("test_combo.jl")
include("display.jl")

end
