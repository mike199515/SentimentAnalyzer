from definitions import verify_cwd
from analyzer.common import dataset
from IPython import embed
from tqdm import tqdm
import pickle
import numpy as np
np.random.seed(42)

if __name__=="__main__":
    verify_cwd()
    words = set()
    for vals in [dataset.get_raw_train_dataset, dataset.get_raw_val_dataset]:
        for tree in tqdm(vals()):
            text = tree.to_lines()[0].split()
            for word in text:
                words.add(word.lower())
    pickle.dump(words, open("./data/words.pkl","wb"))
    glove_dict = pickle.load(open("./data/glove_300.pkl","rb"))
    print(len(words))
    missing = []
    for word in words:
        if word not in glove_dict:
            missing.append(word)
            glove_dict[word] = np.random.rand(300) * 0.1 - 0.05
    pickle.dump(glove_dict,open("./data/glove_300_aug.pkl", "wb"))
    print(len(missing))