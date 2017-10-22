import sys
import os
import urllib.request
import zipfile
import pytreebank
import time
from scripts.create_pretrain_model import train_word2vec, train_glove
def main():
    print("make sure this is run under root!")
    time.sleep(5) # give you time to prepare lol
    if not os.path.exists("./data/"):
        print("creating folder...")
        os.mkdir("./data/")
    if not os.path.exists("./data/trees/"):
        print("preparing sentiment treebank...")
        try:
            pytreebank.load_sst("./data/")
        except:
            pass  # pytreebank downloader seems not robust under windows env. Actually we just want the data and the parser, so ignored.

    if not os.path.exists("./data/text8.zip"):
        print("retrieving text8...")
        urllib.request.urlretrieve("http://mattmahoney.net/dc/text8.zip","./data/text8.zip")
    if not os.path.exists("./data/text8"):
        print("extracting text8...")
        with zipfile.ZipFile("./data/text8.zip", "r") as zip_ref:
            zip_ref.extractall("./data/")
    if not os.path.exists("./data/word2vec.model"):
        print("training word2vec...")
        train_word2vec()
    #if not os.path.exists("./data/glove.model"): glove training is slow. You should call it manually on create_pretrain_model.py
    #    print("training glove...")
    #    train_glove()
    print("=== ALL CLEAR! ===")


if __name__=="__main__":
    main()