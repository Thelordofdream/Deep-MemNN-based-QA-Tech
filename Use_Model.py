import gensim

model_dbow = gensim.models.Word2Vec.load('model_dbow')
model_dm = gensim.models.Word2Vec.load('model_dm')
# print model_dm.doesnt_match("breakfast cereal dinner lunch".split())
# print model_dm['is']
# print model_dbow['is']
print model_dm.similarity('is', 'was')
#print model_dbow.similarity('is', 'was')