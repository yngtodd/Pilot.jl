using  Languages, TextAnalysis, Flux, PyPlot, Statistics

Arr = ["well done",
     "good work",
     "great effort",
     "nice work",
     "excellent",
     "weak",
     "poor effort",
     "not good",
     "poor work",
     "could have done better"]

# positve or negative sentiment to each 'document' string
y = [true true true true true false false false false false]


docs=[]
for i in 1:length(Arr)
    push!(docs, StringDocument(Arr[i]))
end

crps=Corpus(docs)    
update_lexicon!(crps)
doc_term_matrix=DocumentTermMatrix(crps)
word_dict=doc_term_matrix.column_indices

tk_idx(s) = haskey(word_dict, s) ? i=word_dict[s] : i=0

function pad_corpus(c, pad_size)
    M=[]
    for doc in 1:length(c)
        tks = tokens(c[doc])
        if length(tks)>=pad_size
            tk_indexes=[tk_idx(w) for w in tks[1:pad_size]]
        end
        if length(tks)<pad_size
            tk_indexes=zeros(Int64,pad_size-length(tks))
            tk_indexes=vcat(tk_indexes, [tk_idx(w) for w in tks])
        end
        doc==1 ? M=tk_indexes' : M=vcat(M, tk_indexes')
    end
    return M
end

pad_size =4
doc_pad_size=4
padded_docs = pad_corpus(crps, pad_size)
x = padded_docs'
data = [(x, y)]

N = size(padded_docs,1)  #Number of documents (10)
max_features = 8
vocab_size = 20

embedding_matrix=Flux.glorot_normal(max_features, vocab_size)

m = Chain(x -> embedding_matrix * Flux.onehotbatch(reshape(x, doc_pad_size*size(x,2)), 0:vocab_size-1),
          x -> reshape(x, max_features, doc_pad_size, trunc(Int64(size(x,2)/doc_pad_size))),
          x -> mean(x, dims=2),
          x -> reshape(x, max_features, :),
          Dense(max_features, 2),
          softmax
)

loss_h=[]
accuracy_train=[]
accuracy(x, y) = mean(x .== y)

loss(x, y) = sum(Flux.binarycrossentropy.(m(x), y))
optimizer = Flux.Descent(0.01)

for epoch in 1:400
    Flux.train!(loss, Flux.params(m), data, optimizer)
    println(loss(x, y).data, " ", accuracy(m(x).>0.5,y))
    push!(loss_h, loss(x, y).data)
    push!(accuracy_train, accuracy(m(x).>0.5,y))
end

println(m(x).>0.5)
accuracy(m(x).>0.5,y)
