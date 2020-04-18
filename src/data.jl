import NPZ
import MLDataPattern

"""
    load_numpy(path)

Load a numpy array and transpose it so that it is
in column-major form. This lets us make use of the 
same data files for our models in Python.

Say you created a numpy array created by, and saved
it as `my_array.npy`:

```python
np.arange(10).reshape((2, 5))

>>> array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
```

loading that array with `load_numpy("my_array.npy")`, 
we would get

```julia
5×2 LinearAlgebra.Transpose{Int64,Array{Int64,2}}:
 0  5
 1  6
 2  7
 3  8
 4  9
```
"""
function load_numpy(path)
    arry = NPZ.npzread(path)
    transpose(arry)
end


"""
    split_data(x, y, ratio=0.85)

Split a dataset of x, y pairs into two sets with a given ratio.
This will return a pair of tuples with the `ratio` percent of
of the data belonging to the first tuple:

```
(x_split1, y_split1), (x_split2, y_split2)
```

# Examples
```julia-repl
julia> x = transpose(reshape(collect(1:10), (2,5)))
julia> y = transpose(reshape(collect(0:1), (2,1)))
julia> split_data(x, y, 0.5)
(([1; 3; … ; 7; 9], [0]), ([2; 4; … ; 8; 10], [1]))
```
"""
function split_data(x::Array{Real}, y::Array{Real}, ratio::Float64=0.85)
    MLDataPattern.splitobs((x, y); at = ratio)
end

