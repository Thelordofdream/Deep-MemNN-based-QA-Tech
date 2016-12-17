def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


train_vecs = grabVecs('d2v_vecs.txt')