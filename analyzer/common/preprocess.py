import gensim
import numpy as np
class Word2Vec(object):
    def __init__(self, model_name = None):
        if model_name is None: model_name = "./data/word2vec.model"
        self.model = gensim.models.Word2Vec.load(model_name)

    def vectorize_word(self, word):
        assert(word in self.model.wv)
        return self.model.wv[word]

    def vectorize_sentence(self, sentence):
        return np.concatenate([self.vectorize_word(word)[np.newaxis,:] for word in sentence],0)


if __name__=="__main__":
    sentence = ["hello","world"]
    w2v = Word2Vec()
    print(w2v.vectorize_sentence(sentence))