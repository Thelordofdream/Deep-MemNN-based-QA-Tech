import numpy as np
import random
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
import pymysql.cursors
import Doc2Vec

# ===========================================
# load data
connection = pymysql.connect(user='root', password='root', database='GRE')
cursor = connection.cursor()
commit = "select * from GRES"
cursor.execute(commit)
Sentences = [each[1] for each in cursor.fetchall()]

# ===========================================
# instantiate our DM and DBOW models
size = 400
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

# ===========================================
# Train models
Sentences = Doc2Vec.Train_Doc2Vec(Sentences, model_dm, model_dbow)

# ===========================================
# Generalize words
train_vecs = Doc2Vec.buildtrainvecs(model_dm, model_dbow, Sentences, size)
print train_vecs[0]
Doc2Vec.storeVecs(train_vecs, 'd2v_vecs.txt')

