"""
Pyndl object.
"""
struct Pyndl_Weight_Struct
    cues::Vector{String}
    outcomes::Vector{String}
    weight::Matrix{Float64}
end

"""
    pyndl(data_path)

Perform pyndl.
"""
function pyndl(
    data_path;
    alpha = 0.1,
    betas = (0.1, 0.1),
)

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

function unwrap_xarray(weights)
    coords = weights.coords.to_dataset()
    cues = [i for i in coords.cues.data]
    outcomes = [i for i in coords.outcomes.data]
    weights = weights.data

    Pyndl_Weight_Struct(cues, outcomes, weights)
end
