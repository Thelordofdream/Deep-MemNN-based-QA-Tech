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
Sentences = Word2Vec.cleanText(Sentences)

# ===========================================
# Train model
model_w2v = gensim.models.Word2Vec.load('../model/model_w2v')
Word2Vec.Train_Wrod2VEc(Sentences, model_w2v)

# ===========================================
# Generalize words
n_dim = 300
train_vectors = [Word2Vec.buildWordVector(model_w2v, z, n_dim) for z in Sentences]
Word2Vec.storeVecs(train_vectors, '../model/w2v_vecs.txt')

