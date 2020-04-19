using Flux


struct Embedding
    table
end


Embedding(voc_size, feature_size) = Embedding(Flux.glorot_normal(voc_size, feature_size))


(e::Embedding)(x) = e.table[x, :]


function (e::Embedding)(x::AbstractArray{T, 2}) where {T}
    out = e.table[x, :]
    return(permutedims(out, (1, 3, 2)))
end


@Flux.treelike Embedding
