from definitions import verify_cwd
import numpy as np
import pickle
from tqdm import tqdm
import os
def gen_glove():
    f = open("./data/glove.840B.300d.txt","r",encoding="utf-8")
    words = pickle.load(open("./data/words.pkl","rb"))
    d = dict()
    for line in tqdm(f):
        try:
            vals = line.split()
            if vals[0] not in words: continue
            d[vals[0]] = np.array(list(map(float,vals[1:])))
        except Exception as e:
            print(vals[0],vals,e)
            continue
    o = open("./data/glove_300.pkl","bw")
    pickle.dump(d,o)
if __name__=="__main__":
    verify_cwd()
    if not os.path.exists("./data/glove_300.pkl"):
        gen_glove()