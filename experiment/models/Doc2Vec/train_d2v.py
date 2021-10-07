from gensim.models import KeyedVectors
from configs.D2VConfigs import args 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_d2v(sents):
    l_sents = [TaggedDocument(line,'1') for line in sents]
    m = Doc2Vec(l_sents,vector_size = args.vector_size,eopchs = args.eopchs,min_count =1)
    save_d2v(m)
    return m
    
def save_d2v(model):
    model.save(args.d2v_path)

def load_model():
    return Doc2Vec.load(args.d2v_path)



    