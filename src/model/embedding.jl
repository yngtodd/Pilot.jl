using Flux


struct Embedding
    table
end


Embedding(num_vocab, embed_dim) = Embedding(Flux.glorot_normal(num_vocab, embed_dim))


(e::Embedding)(x) = e.table[x, :]


function (e::Embedding)(x::AbstractArray{T, 2}) where {T}
    out = e.table[x, :]
    return(permutedims(out, (1, 3, 2)))
end


@Flux.treelike Embedding
