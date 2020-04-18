using NPZ


"""
    load_numpy(path)

Load a numpy array and transpose it so that it is
in column-major form. This lets us make use of the 
same data files for our models in Python.

Say you created a numpy array created by:

np.arange(10).reshape((2, 5))
>>> array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

loading that array with `load_numpy`, we would get

5×2 LinearAlgebra.Transpose{Int64,Array{Int64,2}}:
 0  5
 1  6
 2  7
 3  8
 4  9
"""
function load_numpy(path)
    arry = npzread(path)
    transpose(arry)
end
