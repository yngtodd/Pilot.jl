using Flux
using Tracker

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

model = Embedding(100, 16) # feature-size=16
input = rand(1:100, (10,32)) # input-size=10, batch-size=32

out = model(input) # input-size x feature-size x batch-size
println("$out")
#loss(x) = sum(model(x)) # not actually meaningful, just as a test
#loss(input)

#Tracker.gradient(params(model)) do
#    loss(input)
#end 
