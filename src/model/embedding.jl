using Flux


struct Embedding
    table
end


@Flux.treelike Embedding


(e::Embedding)(x) = e.table[x, :]


"""
    Embedding(num_vocab, embed_dim)

Construct an embedding layer for Flux models. This 
is a mapping from the dimension of the model's 
`num_vocab`, the number of unique vocabulary
terms in a corpus, to some embedding dimension, `embed_dim`.

# Examples
```julia-repl
julia> m = Embeddig(10, 5)

```
"""
Embedding(num_vocab, embed_dim) = Embedding(Flux.glorot_normal(num_vocab, embed_dim))


"""
Forward pass of the Embedding layer. 

Given an array of size (input_dim, batch_size), this will 
embed the information to (input_dim, embed_dim, batch_size).
"""
function (e::Embedding)(x::AbstractArray{T, 2}) where {T}
    out = e.table[x, :]
    permutedims(out, (1, 3, 2))
end
