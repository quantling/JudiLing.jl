"""
originally from:
https://gist.github.com/RobBlackwell/10a1aeabeb85bbf1a17cc334e5e60acf
"""

using PyCall
pickle = pyimport("pickle")


"""
Save pickle from python pickle file.
"""
function save_pickle end

"""
Load pickle from python pickle file.
"""
function load_pickle end

function save_pickle(filename, obj)
    out = open(filename, "w")
    pickle.dump(obj, out)
    close(out)
end

function load_pickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename, "rb") as f begin
        r = pickle.load(f)
    end
    return r
end
