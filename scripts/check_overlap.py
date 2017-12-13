from definitions import verify_cwd
from analyzer.common.dataset import Dataset
from IPython import embed
from tqdm import tqdm
import pickle
import numpy as np
np.random.seed(42)

if __name__=="__main__":
    verify_cwd()
    words = set()
    for vals in [Dataset.get_raw_train_dataset, Dataset.get_raw_val_dataset, Dataset.get_raw_test_dataset]:
        for tree in tqdm(vals()):
            text = tree.to_lines()[0].split()
            for word in text:
                words.add(word.lower())
    pickle.dump(words, open("./data/words.pkl","wb"))
    glove_dict = pickle.load(open("./data/glove_300.pkl","rb"))
    print(len(words))
    missing = set()
    for word in words:
        if word not in glove_dict:
            missing.add(word)
            glove_dict[word] = np.random.rand(300) * 0.1 - 0.05
            #glove_dict[word] = np.zeros((300,))
    pickle.dump(glove_dict,open("./data/glove_300_aug.pkl", "wb"))
    print(len(missing))
    pickle.dump(missing, open("./data/missing.pkl","wb"))
