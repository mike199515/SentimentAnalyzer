import numpy as np
import pickle
from definitions import verify_cwd

class Word2Vec(object):
    def __init__(self, model_name = None):
        import gensim
        if model_name is None: model_name = "./data/word2vec.model"
        self.model = gensim.models.Word2Vec.load(model_name)

    def vectorize_word(self, word):
        assert(word in self.model.wv)
        return self.model.wv[word]

    def vectorize_sentence(self, sentence):
        return np.concatenate([self.vectorize_word(word)[np.newaxis,:] for word in sentence],0)

class GloVe(object):
    def __init__(self, model_name = None, default_vector = None):
        if model_name is None: model_name = "./data/glove_300_aug.pkl"
        if default_vector is None: default_vector = np.zeros(300)
        self.dict = pickle.load(open(model_name,"rb"))
        self.default_vector = default_vector

    def vectorize_word(self, word):
        if word == "": return self.default_vector
        return self.dict[word.lower()]

    def vectorize_sentence(self, sentence, required_size = None):
        if required_size is not None:
            assert(len(sentence)<=required_size)," sentence {} > required_size {}".format(sentence, required_size)
            sentence = [""]* (required_size - len(sentence)) + sentence
        return np.concatenate([self.vectorize_word(word)[np.newaxis,:] for word in sentence],0)


if __name__=="__main__":
    verify_cwd()
    sentence = ["jack","is","play"]
    w2v = Word2Vec()
    #print(w2v.vectorize_sentence(sentence))
    glv = GloVe()
    print(glv.vectorize_sentence(sentence, 8))