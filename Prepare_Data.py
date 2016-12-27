import numpy as np
import pymysql.cursors


def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()

goole_vecs = grabVecs('google_vecs.txt')
dataset = []
for a in goole_vecs:
    sentence = np.zeros((41, 300))
    m = len(a)
    start = int((41 - m) / 2)
    sentence[start:start + m] = a
    dataset.append(sentence)
storeVecs(dataset, './data/dataset.txt')


connection = pymysql.connect(user='root', password='root', database='GRE')
cursor = connection.cursor()
commit = "select * from GRES1"
cursor.execute(commit)
label = []
for each in cursor.fetchall():
    single = [0, 0]
    single[each[2]] = 1
    label.append(single)
storeVecs(label, './data/label.txt')