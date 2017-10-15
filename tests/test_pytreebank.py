import pytreebank
from IPython import embed
TARGET_STRING = "(3 (2 (2 The) (2 Rock) )(4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's) )(2 (3 new) (2 (2 ``) (2 Conan) )))))))(2 '') )(2 and) )(3 (2 that) (3 (2 he) (3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash) )(2 (2 even) (3 greater) )))(2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger) )(2 ,) )(2 (2 Jean-Claud) (2 (2 Van) (2 Damme) )))(2 or) )(2 (2 Steven) (2 Segal) ))))))))))))(2 .) ))"
def main():
    print("make sure you run the test at root.")
    pytreebank.load_sst("./data/")
    train_data = pytreebank.import_tree_corpus("./data/stanford_sentiment_treebank/trees/train.txt")
    assert(str(train_data[0])==TARGET_STRING),"test fail for pytreebank."
    print("Correctness verified.")

if __name__ == "__main__":
    main()