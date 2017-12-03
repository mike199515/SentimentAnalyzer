import pytreebank
import numpy as np
from tqdm import tqdm
from IPython import embed
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
    def flattened_datapoint(tree_data, vectorizer):
        # return data point with only the whole sentence
        datas = []
        labels = []
        #max_size = max( len(node.to_lines()[0].split()) for node in tqdm(tree_data))
        #print("max_size = {}".format(max_size))
        max_size = 52
        for node in tqdm(tree_data):
            sentence = node.to_lines()[0].split()
            data = vectorizer.vectorize_sentence(sentence, max_size)
            label = node.label
            datas.append(data)
            labels.append(label)
        return np.array(datas), np.array(labels)

    @staticmethod
    def batch_generator(data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset. yield twice for both data and label
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

if __name__=="__main__":
    from definitions import verify_cwd
    from analyzer.common.preprocess import GloVe
    verify_cwd()
    datas, labels = Dataset.flattened_datapoint(Dataset.get_raw_train_dataset(), GloVe())
    embed()