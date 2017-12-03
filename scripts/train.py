from keras.layers import GRU, LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import to_categorical

from analyzer.common.dataset import Dataset
from analyzer.common.preprocess import GloVe
from IPython import embed

def baseline_model():
    model = Sequential()
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(52,300)))
    model.add(Activation('relu'))
    model.add(Dense(5, activation='softmax'))

    return model

def prepare_train_data(vectorizer):
    return Dataset.flattened_datapoint(Dataset.get_raw_train_dataset(), vectorizer)

def prepare_val_data(vectorizer):
    return Dataset.flattened_datapoint(Dataset.get_raw_val_dataset(), vectorizer)

def main():
    print("Building model...")
    model = baseline_model()
    print(model.summary())
    vectorizer = GloVe()
    print("Preparing data...")
    train_data, train_label = prepare_train_data(vectorizer)
    train_label = to_categorical(train_label ,5)
    val_data, val_label = prepare_val_data(vectorizer)
    val_label = to_categorical(val_label)
    print(train_label.shape)
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(train_data, train_label,
              batch_size=32,
              epochs=15,
              validation_data=(val_data, val_label))
    score, acc = model.evaluate(val_data, val_label,
                                batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__=="__main__":
    from definitions import verify_cwd
    verify_cwd()
    main()