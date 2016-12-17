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
# Train model
model_w2v = gensim.models.Word2Vec.load('model_w2v')
Word2Vec.Train_Wrod2VEc(Sentences, model_w2v)

# ===========================================
# Generalize words
train_vectors = np.concatenate([Word2Vec.buildWordVector(model_w2v, z, n_dim = 300) for z in Sentences])
Word2Vec.storeVecs(train_vectors, 'w2v_vecs.txt')

