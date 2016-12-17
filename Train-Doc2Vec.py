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
Sentences = Doc2Vec.Preprocessing(Sentences)

# ===========================================
# Train models
model_dbow = gensim.models.Doc2Vec.load('model_dbow')
model_dm = gensim.models.Doc2Vec.load('model_dm')
Doc2Vec.Train_Doc2Vec(Sentences, model_dm, model_dbow)

# ===========================================
# Generalize words
train_vecs = Doc2Vec.buildtrainvecs(model_dm, model_dbow, Sentences, size = 400)git rm -r --cached .
Doc2Vec.storeVecs(train_vecs, 'd2v_vecs.txt')

