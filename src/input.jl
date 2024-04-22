"""
    load_dataset(filepath::String;
                delim::String=",",
                kargs...)

Load a dataset from file, usually comma- or tab-separated.
Returns a DataFrame.

# Obligatory arguments
- `filepath::String`: Path to file to be loaded.

# Optional arguments
- `delim::String=","`: Delimiter in the file (usually either `","` or `"\\t"`).
- `kargs...`: Further keyword arguments are passed to `CSV.File()`.

# Example
```julia
latin = JudiLing.load_dataset("latin.csv")
first(latin, 10)
```
"""
function load_dataset(filepath::String;
                        delim::String=",",
                        kargs...)
    return(DataFrame(CSV.File(filepath, stringtype=String, delim=delim; kargs...)))
end
