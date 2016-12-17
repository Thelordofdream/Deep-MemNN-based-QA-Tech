import numpy as np
import random
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
import pymysql.cursors


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
        corpus = [z.replace(c, ' ') for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


# Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
# We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
# a dummy index of the review.
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


def getVecs(model, corpus, size):
    vecs = []
    for z in corpus:
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for each in np.array(model[z.words]):
            try:
                vec += each.reshape((1, size))
                count += 1.
            except:
                continue
        if count != 0:
            vec /= count
        vecs.extend(vec)
    return np.array(vecs)


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()


def buildtrainvecs(model_dm, model_dbow, sentences, iteration):
    # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    # all_train_reviews = np.concatenate((x_train))
    print "+==== Training =====+"
    for epoch in range(iteration):
        # perm = np.random.permutation(all_train_reviews.shape[0])
        print "+=== Iteration %d ===+" % epoch
        random.shuffle(sentences)
        model_dm.train(sentences)
        model_dbow.train(sentences)
    train_vecs_dm = getVecs(model_dm, sentences, size)
    train_vecs_dbow = getVecs(model_dbow, sentences, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    return train_vecs


# ===========================================
# Pretreatment
x_train = cleanText(Sentences[:int(number * 98 / 100)])
unsup_reviews = cleanText(Sentences[int(number * 98 / 100):])

x_train = labelizeReviews(x_train, 'TRAIN')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')
# ===========================================
# Train Doc2Vec models
size = 400
iteration = 10

# instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

# build vocab over all reviews
temp1 = x_train[:]
temp1.extend(unsup_reviews)
model_dm.build_vocab(temp1)
model_dbow.build_vocab(temp1)

# get train_vecs
train_vecs = buildtrainvecs(model_dm, model_dbow, temp1, iteration)

# Store model and result
model_dm.save("model_dm")
model_dbow.save("model_dbow")

print train_vecs[0]
storeVecs(train_vecs, 'd2v_vecs.txt')

