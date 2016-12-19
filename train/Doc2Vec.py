import random
import numpy as np
import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence

# ===========================================
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, '') for z in corpus]
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
        for each in np.array(model[z.words]):
            try:
                vec = each.reshape((1, size))
                vecs.extend(vec)
            except:
                continue
    return np.array(vecs)


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()


def buildtrainvecs(model_dm, model_dbow, sentences, size):
    # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    # all_train_reviews = np.concatenate((x_train))
    train_vecs_dm = getVecs(model_dm, sentences, size)
    train_vecs_dbow = getVecs(model_dbow, sentences, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    return train_vecs


def Preprocessing(Sentences):
    # ===========================================
    # Pretreatment
    number = len(Sentences)
    x_train = cleanText(Sentences[:int(number * 98 / 100)])
    unsup_reviews = cleanText(Sentences[int(number * 98 / 100):])

    x_train = labelizeReviews(x_train, 'TRAIN')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

    # ===========================================
    # build vocab over all reviews
    temp = x_train[:]
    temp.extend(unsup_reviews)
    return temp


def Train_Doc2Vec(Sentences, model_dm, model_dbow):
    # ===========================================
    # Train Doc2Vec models
    iteration = 10
    print "+==== Training =====+"
    for epoch in range(iteration):
        # perm = np.random.permutation(all_train_reviews.shape[0])
        print "+=== Iteration %d ===+" % epoch
        random.shuffle(Sentences)
        model_dm.train(Sentences)
        model_dbow.train(Sentences)

    # ===========================================
    # Store model and result
    model_dm.save("../model/model_dm")
    model_dbow.save("../model/model_dbow")
