import Word2Vec
import gensim
import numpy as np
import pymysql.cursors

# ===========================================
# load data
connection = pymysql.connect(user='root', password='root', database='GRE')
cursor = connection.cursor()
commit = "select * from GRES"
cursor.execute(commit)
Sentences = [each[1] for each in cursor.fetchall()]

# ===========================================
# Load model
model_google= gensim.models.Word2Vec.load_word2vec_format('../model/GoogleNews-vectors-negative300.bin', binary=True)
# Word2Vec.Train_Wrod2VEc(Sentences, model_google)

# ===========================================
# Generalize words
n_dim = 300
train_vectors = np.concatenate([Word2Vec.buildWordVector(model_google, z, n_dim) for z in Sentences])
Word2Vec.storeVecs(train_vectors, '../vectors/google_vecs.txt')
