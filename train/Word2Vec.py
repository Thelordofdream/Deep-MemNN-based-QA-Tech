import gensim
import numpy as np

LabeledSentence = gensim.models.doc2vec.LabeledSentence

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


def buildWordVector(model_w2v, text, size):
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


def Train_Wrod2VEc(Sentences, model_w2v):
    # ===========================================
    # Pretreatment
    x_train = cleanText(Sentences)

    # ===========================================
    # Train the model over train_reviews (this may take several minutes)
    model_w2v.train(x_train)

    # ===========================================
    # Store model and result
    model_w2v.save("./model/model_w2v")