def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


train_vecs = grabVecs('./model/d2v_vecs.txt')