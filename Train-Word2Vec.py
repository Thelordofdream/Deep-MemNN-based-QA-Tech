import gensim
import numpy as np
LabeledSentence = gensim.models.doc2vec.LabeledSentence
import pymysql.cursors
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale

# load data
connection = pymysql.connect(user='root', password='root',
                             database='GRE')

cursor = connection.cursor()
commit = "select * from GRES"
cursor.execute(commit)
Sentences = [each[1] for each in cursor.fetchall()]
number = len(Sentences)


# ===========================================
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except:
            continue
    if count != 0:
        vec /= count
    return vec


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()
# ===========================================
# Pretreatment
x_train = cleanText(Sentences[:int(number * 98 / 100)])
unsup_reviews = cleanText(Sentences[int(number * 98 / 100):])

# ===========================================
# Initialize model and build vocab
n_dim = 300

model_w2v = Word2Vec(size=n_dim, min_count=10)
model_w2v.build_vocab(x_train)

# ===========================================
# Train the model over train_reviews (this may take several minutes)
model_w2v.train(x_train)

# Store model and result
model_w2v.save("model_w2v")

train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
print train_vecs[0]
train_vecs = scale(train_vecs)
print train_vecs[0]
storeVecs(train_vecs, 'w2v_vecs.txt')

