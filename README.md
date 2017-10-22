# SentimentAnalyzer

A deep learning project on sentiment analysis using stanford's sentiment bank.

## Requirement

Using python 3.6

[keras](https://github.com/fchollet/keras/tree/master/keras) with underlying tensorflow(latest, preferrable gpu version) backend

[pytreebank](https://github.com/JonathanRaiman/pytreebank) for dataset

(future use)[glove-python](https://github.com/maciejkula/glove-python) to use glove and [gensim](https://radimrehurek.com/gensim/install.html) for many NLP tools including word2vec

with anaconda recommended as python library & version control

After correct setup, you should pass tests. Use 

- test_keras.py 
- test_pytreebank.py 

to check if your requirement is met.

## Installation

You need to make sure Requirements are installed beforehand.

run "pip install -e SentimentAnalyzer"(preferrable in anaconda environment) to install it in developer mode, then you can import analyzer.* in python from anywhere and change source code as you wish. 

Run scripts/initialize.py to get training data and preprocessed models ready. GloVe training is slow(hours of train time), so you should run create_pretrain_model.py instead.

## Tutorial and tools

For windows installation, we have
[installation of tensorflow on windows](https://github.com/antoniosehk/keras-tensorflow-windows-installation) &  [tensorflow self-check script](https://gist.github.com/mrry/ee5dbcfdd045fa48a27d56664411d41c)

[Linux and Mac tutorial](https://www.tensorflow.org/install/)