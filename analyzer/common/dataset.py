import pytreebank

def get_raw_train_dataset(self, path="./data/"):
    train_data = pytreebank.import_tree_corpus("{}/stanford_sentiment_treebank/trees/train.txt".format(path))
    return train_data

def get_raw_val_dataset(self, path="./data/"):
    val_data = pytreebank.import_tree_corpus("{}/stanford_sentiment_treebank/trees/dev.txt".format(path))
    return val_data