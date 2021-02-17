struct Pyndl_Weight_Struct
    cues::Vector{String}
    outcomes::Vector{String}
    weight::Matrix{Float64}
end

function pyndl(
    data_path::String;
    alpha = 0.1::Float64,
    betas = (0.1, 0.1)::Tuple{Float64,Float64},
)::Pyndl_Weight_Struct

    ndl = pyimport("pyndl.ndl")

    weights_py = ndl.ndl(
        events = data_path,
        alpha = alpha,
        betas = betas,
        method = "openmp",
        remove_duplicates = true,
    )

    unwrap_xarray(weights_py)

end

function unwrap_xarray(weights::PyObject)Tuple{Vector,Vector,Matrix}
    coords = weights.coords.to_dataset()
    cues = [i for i in coords.cues.data]
    outcomes = [i for i in coords.outcomes.data]
    weights = weights.data

    Pyndl_Weight_Struct(cues, outcomes, weights)
end
