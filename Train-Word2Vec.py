import gensim
import pymysql.cursors
import Word2Vec
import numpy as np

# ===========================================
# load data
connection = pymysql.connect(user='root', password='root', database='GRE')
cursor = connection.cursor()
commit = "select * from GRES"
cursor.execute(commit)
Sentences = [each[1] for each in cursor.fetchall()]

# ===========================================
# Initialize model
n_dim = 300
model_w2v = gensim.models.word2vec.Word2Vec(size=n_dim, min_count=10)
# build vocab
model_w2v.build_vocab(Sentences)

# ===========================================
# Train model
Word2Vec.Train_Wrod2VEc(Sentences, model_w2v)

# ===========================================
# Generalize words
train_vecs = np.concatenate([Word2Vec.buildWordVector(model_w2v, z, n_dim) for z in Sentences])
Word2Vec.storeVecs(train_vecs, 'w2v_vecs.txt')

