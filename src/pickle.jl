"""
originally from:
https://gist.github.com/RobBlackwell/10a1aeabeb85bbf1a17cc334e5e60acf
"""

using PyCall
pickle = pyimport("pickle")


"""
save pickle from python pickle file
"""
function mypickle end

"""
load pickle from python pickle file
"""
function myunpickle end

function mypickle(filename, obj)
    out = open(filename,"w")
    pickle.dump(obj, out)
    close(out)
 end

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end
