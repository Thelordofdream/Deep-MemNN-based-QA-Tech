import gensim
import pymysql.cursors
import Word2Vec
import Doc2Vec

# ===========================================
# Load dictionary
connection = pymysql.connect(user='root', password='root', database='GRE')
cursor = connection.cursor()
commit = "select * from GRES"
cursor.execute(commit)
Sentences = [each[1] for each in cursor.fetchall()]
Dictionary1 = Word2Vec.cleanText(Sentences)
Dictionary2 = Doc2Vec.Preprocessing(Sentences)

# ===========================================
# instantiate our DM and DBOW models
size = 400
model_dm = gensim.models.Doc2Vec(min_count=0, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=0, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
# build vocab
model_dm.build_vocab(Dictionary2)
model_dbow.build_vocab(Dictionary2)

# Store initialized model
model_dm.save("../model/model_dm")
model_dbow.save("../model/model_dbow")

# ===========================================
# Initialize Word2Vecmodel
n_dim = 300
model_w2v = gensim.models.word2vec.Word2Vec(size=n_dim, min_count=0)
# build vocab
model_w2v.build_vocab(Dictionary1)
# Store initialized model
model_w2v.save("../model/model_w2v")
