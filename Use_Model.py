import gensim

# ===========================================
# Doc2Vec
model_dbow = gensim.models.Doc2Vec.load('model_dbow')
model_dm = gensim.models.Doc2Vec.load('model_dm')
# print model_dm.doesnt_match("breakfast cereal dinner lunch".split())
# print model_dm['is']
# print model_dbow['is']
word1 = "huge"
word2 = "large"
print "The similarity between %s and %s is:" % (word1, word2)
print model_dm.similarity(word1, word2)
model_dbow.similarity('is', 'was')

# ===========================================
# Word2Vec
model_w2v = gensim.models.Word2Vec.load('model_w2v')
print "The similarity between %s and %s is:" % (word1, word2)
print model_w2v.similarity(word1, word2)