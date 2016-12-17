import gensim


# ===========================================
# Load dictionary
Dictionary =

# ===========================================
# instantiate our DM and DBOW models
size = 400
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
# build vocab
model_dm.build_vocab(Dictionary)
model_dbow.build_vocab(Dictionary)

# Store initialized model
model_dm.save("model_dm")
model_dbow.save("model_dbow")

# ===========================================
# Initialize Word2Vecmodel
n_dim = 300
model_w2v = gensim.models.word2vec.Word2Vec(size=n_dim, min_count=10)
# build vocab
model_w2v.build_vocab(Dictionary)
# Store initialized model
model_w2v.save("model_w2v")
