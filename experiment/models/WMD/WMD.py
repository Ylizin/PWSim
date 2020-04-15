from gensim.models import KeyedVectors
from configs.WMDConfigs import args 



def __loadModel():
    return KeyedVectors.load_word2vec_format(args.pret_embeddings, binary=True)

def cal_WMD(in_seq1,in_seq2):
    if not hasattr(cal_WMD,'model'):
        cal_WMD.model = __loadModel()
    model = cal_WMD.model
    return model.wmdistance(in_seq1,in_seq2)


    