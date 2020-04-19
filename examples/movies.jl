using CSV, DataFrames, Random, TextAnalysis, Languages, Statistics, PyPlot, Flux, BSON


df_all=CSV.read("/Users/ygx/data/songdata.csv")
categorical!(df_all, :artist)
show(by(df_all, :artist, nrow))

artists=["Queen", "The Beatles", "Michael Jackson", "Eminem", "INXS"]
df=df_all[[x in artists for x in df_all[:artist]],:]
df_all=nothing
Random.seed!(1000);
df=df[shuffle(1:size(df, 1)),:]
df[1,:]


docs=Any[]
for i in 1:size(df,1)
    txt=df.text
    txt=replace(df[i,:].text, "\n" => " ")
    txt=replace(df[i,:].text, "'" => "")
    dm=TextAnalysis.DocumentMetadata(Languages.English(), df[i,:].song,"","")
    doc=StringDocument(txt, dm)
    push!(docs, doc)
end

crps=Corpus(docs)
orig_corpus=deepcopy(crps);
prepare!(crps, strip_non_letters | strip_punctuation | strip_case | strip_stopwords | strip_whitespace)

update_lexicon!(crps)
update_inverse_index!(crps)

m_dtm=DocumentTermMatrix(crps)
word_dict=m_dtm.column_indices

tk_idx(s) = haskey(word_dict, s) ? i=word_dict[s] : i=0

function pad_corpus(c, size)
    M=[]
    for doc in 1:length(c)
        tks = tokens(c[doc])
        if length(tks)>=size
            tk_indexes=[tk_idx(w) for w in tks[1:size]]
        end
        if length(tks)<size
            tk_indexes=zeros(Int64,size-length(tks))
            tk_indexes=vcat(tk_indexes, [tk_idx(w) for w in tks])
        end
        doc==1 ? M=tk_indexes' : M=vcat(M, tk_indexes')
    end
    return M
end

num_terms_in_songs=[length(tokens(crps[i])) for i in 1:length(crps)]

doc_pad_size=200
padded_docs = pad_corpus(crps, doc_pad_size)
X = padded_docs'

artist_dict = Dict()
for (n, a) in enumerate(unique(df.artist))
   artist_dict["$a"] = n
end
artist_dict

artist_indexes=[artist_dict[df[:artist][i]] for i in 1:size(df,1)]
y = Flux.onehotbatch(artist_indexes, 1:5)

X_train = X[:, 1:649]
y_train = y[:,1:649]
X_test = X[:, 650:end]
y_test = y[:, 650:end]

train_set = [(X_train, y_train)]


function load_embeddings(embedding_file)
    local LL, indexed_words, index
    indexed_words = Vector{String}()
    LL = Vector{Vector{Float32}}()
    open(embedding_file) do f
        index = 1
        for line in eachline(f)
            xs = split(line)
            word = xs[1]
            push!(indexed_words, word)
            push!(LL, parse.(Float32, xs[2:end]))
            index += 1
        end
    end
    return reduce(hcat, LL), indexed_words
end

embeddings, vocab = load_embeddings("glove.6B.300d.txt")
embed_size, max_features = size(embeddings)

#Function to return the index of the word 's' in the embedding (returns 0 if the word is not found)
function vec_idx(s)
    i=findfirst(x -> x==s, vocab)
    i==nothing ? i=0 : i 
end

#Function to return the word vector for string 's'
wvec(s) = embeddings[:, vec_idx(s)]


max_features = 300
vocab_size = 8450

embedding_matrix=Flux.glorot_normal(max_features, vocab_size)

for term in m_dtm.terms
    if vec_idx(term)!=0
       embedding_matrix[:,word_dict[term]+1]=wvec(term)
    end
end

m = Chain(x -> embedding_matrix * Flux.onehotbatch(reshape(x, doc_pad_size*size(x,2)), 0:vocab_size-1),
    x -> reshape(x, max_features, doc_pad_size, trunc(Int64(size(x,2)/doc_pad_size))),
    x -> mean(x, dims=2),
    x -> reshape(x, max_features, :),
    Dense(max_features, 5),
    softmax
)

loss_h=[]
accuracy_train=[]
accuracy_test=[]
accuracy(x, y) = mean(Flux.onecold(x) .== Flux.onecold(y))
loss(x, y) = sum(Flux.crossentropy(m(x), y))
optimizer = Flux.Momentum(0.2)

Momentum(0.2, 0.9, IdDict{Any,Any}())


for epoch in 1:400
    Flux.train!(loss, Flux.params(m), train_set, optimizer)
    l = loss(X_train, y_train).data
    push!(loss_h, l)
    accuracy_trn=accuracy(m(X_train).data, y_train)
    push!(accuracy_train, accuracy_trn)
    accuracy_tst=accuracy(m(X_test).data, y_test)
    push!(accuracy_test, accuracy_tst)
    println("$epoch -> loss= $l accuracy train=$accuracy_trn accuracy test=$accuracy_tst")
end
