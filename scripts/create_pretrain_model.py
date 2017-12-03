from gensim.models import word2vec
import logging
import itertools
import glove
import multiprocessing
from definitions import verify_cwd

def train_word2vec(sentences = None, nr_feature = None, save_name= None):
    verify_cwd()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #gensim.models.Word2Vec, may be we need to train it later
    if sentences is None:
        sentences = word2vec.Text8Corpus("./data/text8")
    if save_name is None:
        save_name = "./data/word2vec.model"
    if nr_feature is None:
        nr_feature = 200
    model  = word2vec.Word2Vec(sentences, size = nr_feature)
    model.save(save_name)

def train_glove(sentences = None, nr_feature = None, save_name = None):
    verify_cwd()
    if sentences is None:
        print("preprocessing sentences...")
        sentences = list(itertools.islice(word2vec.Text8Corpus('./data/text8'),None))
        print("{} sentences found.".format(len(sentences)))
    if save_name is None:
        save_name = "./data/glove.model"
    if nr_feature is None:
        nr_feature = 200

    corpus = glove.Corpus()
    print("start fiting sentences...")
    corpus.fit(sentences, window = 10)
    gl = glove.Glove(no_components=nr_feature, learning_rate=0.05)
    print("start training glove...")
    gl.fit(corpus.matrix, epochs=10, no_threads= multiprocessing.cpu_count() , verbose=True)
    corpus.save("./data/corpus.model")
    gl.save("./data/glove.model")


if __name__=="__main__":
    train_glove()