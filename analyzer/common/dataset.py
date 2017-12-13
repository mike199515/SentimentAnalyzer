import pytreebank
import numpy as np
from tqdm import tqdm
from IPython import embed
import pickle
import random
from keras.utils import to_categorical
# include helper functions for dataset
class Dataset(object):
    @staticmethod
    def get_raw_train_dataset(path="./data/"):
        train_data = pytreebank.import_tree_corpus("{}/trees/train.txt".format(path))
        return train_data

    @staticmethod
    def get_raw_val_dataset(path="./data/"):
        val_data = pytreebank.import_tree_corpus("{}/trees/dev.txt".format(path))
        return val_data
    @staticmethod
    def get_raw_test_dataset(path="./data/"):
        test_data = pytreebank.import_tree_corpus("{}/trees/test.txt".format(path))
        return test_data

    @staticmethod
    def flattened_datapoint(tree_data, vectorizer, filter_missing=False, binarize=False, sample_all=False, max_size = 60):

        missing = None
        if filter_missing:
            missing=pickle.load(open("./data/missing.pkl","rb"))
        # return data point with only the whole sentence
        datas = []
        labels = []
        #max_size = max( len(node.to_lines()[0].split()) for node in tqdm(tree_data))
        #print("max_size = {}".format(max_size))
        for root in tree_data:
            if sample_all:
                sample_list = [random.choice(list(root.all_children()))]
                sample_list.append(root)
            else:
                sample_list = [root]
            for node in sample_list:
                if binarize:
                    label = node.label
                    if label == 2:
                        continue
                    elif label < 2:
                        label = 0
                    else:
                        label = 1
                else:
                    label = node.label
                sentence = node.to_lines()[0].split()
                if filter_missing:
                    skip = False
                    for word in sentence:
                        if word in missing:
                            skip=True
                            break
                    if skip: continue
                data = vectorizer.vectorize_sentence(sentence, max_size)
                datas.append(data)
                labels.append(label)

        return np.array(datas), np.array(labels)

    @staticmethod
    def flattened_single_generator(tree_data, vectorizer, filter_missing=False, binarize=False, sample_all=False):
        nr_class = 2 if binarize else 5
        missing = None
        if filter_missing:
            missing=pickle.load(open("./data/missing.pkl","rb"))
        # return data point with only the whole sentence
        datas = []
        labels = []
        #max_size = max( len(node.to_lines()[0].split()) for node in tqdm(tree_data))
        #print("max_size = {}".format(max_size))
        for root in tree_data:
            if sample_all:
                sample_list = [random.choice(list(root.all_children()))]
                sample_list.append(root)
            else:
                sample_list = [root]
            for node in sample_list:
                if binarize:
                    label = node.label
                    if label == 2:
                        continue
                    elif label < 2:
                        label = 0
                    else:
                        label = 1
                else:
                    label = node.label
                sentence = node.to_lines()[0].split()
                if filter_missing:
                    skip = False
                    for word in sentence:
                        if word in missing:
                            skip=True
                            break
                    if skip: continue
                data = vectorizer.vectorize_sentence(sentence, None)
                yield data[None,:], to_categorical(label, nr_class)

    @staticmethod
    def flattened_generator(tree_data, vectorizer, batch_size=32, filter_missing=False, binarize=False, sample_all=False):
        nr_class = 2 if binarize else 5
        while True:
            datas, labels = Dataset.flattened_datapoint(tree_data, vectorizer, filter_missing, binarize, sample_all)
            # do shuffle here
            datas, labels = Dataset.unison_shuffled_copies(datas, labels)

            for i in range(0, datas.shape[0], batch_size):
                yield datas[i:i+batch_size], to_categorical(labels[i:i+batch_size],nr_class)

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    @staticmethod
    def get_nr_sample(tree_data, vectorizer, batch_size=32, filter_missing=False, binarize=False, sample_all=False):
        datas, labels = Dataset.flattened_datapoint(tree_data, vectorizer, filter_missing, binarize, sample_all)
        count = 0
        for i in range(0, datas.shape[0],  batch_size):
            count+=1
        return count

if __name__=="__main__":
    from definitions import verify_cwd
    from analyzer.common.preprocess import GloVe
    verify_cwd()
    #datas, labels = Dataset.flattened_datapoint(Dataset.get_raw_train_dataset(), GloVe())
    for data, label in Dataset.flattened_generator(Dataset.get_raw_train_dataset(), GloVe(), binarize=False, sample_all=True):
        print(data.shape)
        break
    embed()