from keras.layers import GRU, LSTM, Conv1D, Bidirectional, BatchNormalization, MaxPooling1D, Conv1D, AtrousConv1D
from keras.layers import Concatenate, Flatten, Dropout, Add, Dot, Subtract, Multiply
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Dense, Lambda
from keras.layers import Activation
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.activations import softmax

from keras import regularizers
from analyzer.common.dataset import Dataset
from analyzer.common.preprocess import GloVe
from analyzer.common.loss import focal_loss
from analyzer.common.attention import attention_3d_block,attention_MT_LSTM
from analyzer.common.layer_norm import LayerNorm

from IPython import embed
from keras import optimizers
import keras.backend as K
from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
MAX_SIZE = 60
BINARIZE = False
NR_CLASS = 2 if BINARIZE else 5
SAMPLE_ALL = False

def baseline_model():
    model = Sequential()
    #model.add(BatchNormalization(axis=2,input_shape=(MAX_SIZE,300)))
    model.add(Bidirectional(LSTM(300, dropout=0.2, recurrent_dropout=0.2), input_shape=(MAX_SIZE,300)))
    #model.add(Bidirectional(GRULN(output_dim=128), input_shape=(None, 300)))
    model.add(Dense(NR_CLASS, activation='softmax'))
    return model

def attention_model():
    input = Input((MAX_SIZE,300))
    bi_lstm = Bidirectional(LSTM(300,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(input)
    #bi_lstm2 = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(bi_lstm)
    attention_mul = attention_3d_block(bi_lstm)
    summed_attention = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
    fc =  Dense(400, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(summed_attention)
    fc = Dropout(0.2)(fc)
    output = Dense(NR_CLASS, activation="softmax")(fc)
    model = Model(inputs=input, outputs=output)
    return model

def MT_attention_model():
    input = Input((MAX_SIZE, 300))
    # encode with MT_attention
    transformed_input = attention_MT_LSTM(input, MAX_SIZE)
    result = Dropout(0.2)(Bidirectional(LSTM(300, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(transformed_input))
    attention_mul = attention_3d_block(result, MAX_SIZE)
    summed_attention = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
    fc =  Dropout(0.2)(Dense(400, activation="relu", kernel_regularizer=regularizers.l2(0.01))(summed_attention))
    output = Dense(NR_CLASS, activation="softmax")(fc)
    model = Model(inputs=input, outputs=output)
    return model, "MT_attention"

def multiply(x):
    x_transpose = tf.transpose(x, perm=[0,2, 1])
    return K.batch_dot(x,x_transpose)


def biattention_model():
    input = Input((MAX_SIZE, 300))
    transformed_input = attention_MT_LSTM(input, MAX_SIZE)
    size = int(input.shape[1])
    # we add MT_attention later
    result = Bidirectional(LSTM(300, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(transformed_input)
    # has shape ( MAX_SIZE, 256)
    affinity = Lambda(lambda x: multiply(x), output_shape =(MAX_SIZE, MAX_SIZE))(result)
    A_x = Lambda(lambda x: softmax(x,axis=1))(affinity)
    A_y = Lambda(lambda x: softmax(x,axis=2))(affinity)
    c_x = Lambda(lambda inp : tf.matmul(inp[0],inp[1]))([A_x,result])
    c_y = Lambda(lambda inp : tf.matmul(inp[0],inp[1]))([A_y,result])
    x_y = Bidirectional(LSTM(150, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(c_x)
    y_x = Bidirectional(LSTM(150, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(c_y)
    def get_feat(inp):
        attention_mul = attention_3d_block(inp, 60)
        self_attention = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
        return self_attention

    x_feat = get_feat(x_y)
    y_feat = get_feat(y_x)
    concate_feat = Dropout(0.2)(Concatenate()([x_feat, y_feat]))
    fc = Dense(400, activation="relu", kernel_regularizer=regularizers.l2(0.01))(concate_feat)
    output = Dense(NR_CLASS, activation="softmax")(fc)
    model = Model(inputs=input, outputs=output)
    return model

def res_conv_pool(inp, kernel_size, filter_size):
    #inp = BatchNormalization()(inp)
    conv1 = Conv1D(
        filters=filter_size,
        kernel_size=kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(0.01)
    )(inp)
    residual = Add()([conv1, inp])
    conv2 = Conv1D(
        filters=filter_size,
        kernel_size=kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(0.01)
    )(residual)
    pool = MaxPooling1D(pool_size=2)(conv2)
    return pool

def convolution_model():
    inp = Input(shape=(MAX_SIZE, 300))
    conv1 = Conv1D(256, 3, border_mode='same')(inp)
    conv2 = Conv1D(128, 3, border_mode='same')(conv1)
    conv3 = Conv1D(64, 3, border_mode='same')(conv2)
    flatten = Flatten()(conv1)
    dense = Dropout(0.2)(Dense(400, activation='relu')(Dropout(0.2)(flatten)))
    out = Dense(5, activation='softmax')(dense)
    return Model(inputs=inp, outputs=out)

def convolution_attention_model():
    inp = Input(shape=(MAX_SIZE,300))
    conv_in =  inp
    conv_outs = []

    for _ in range(4):
        conv_in = res_conv_pool(conv_in, 5, 300)
        conv_outs.append(conv_in)
    # apply attention for each layer
    attentions = []
    for conv_out in conv_outs:
        attention_mul = attention_3d_block(conv_out, conv_out._keras_shape[1])
        summed_attention = Dropout(0.2)(Lambda(lambda x: K.sum(x, axis=1))(attention_mul))
        attentions.append(summed_attention)
    concate = Concatenate()(attentions)
    fc = Dense(1200, activation="relu", kernel_regularizer=regularizers.l2(0.01))(concate)
    out = Dense(5, activation="softmax")(fc)
    model = Model(inputs = inp, outputs = out)
    return model

def conv_lstm_model():
    inp = Input(shape=(MAX_SIZE,300))
    dropout_inp = Dropout(0.2)(inp)
    # get from high-level feat
    pool1 = res_conv_pool(dropout_inp, 3, 300)
    pool2 = res_conv_pool(pool1, 3, 300)
    pool3 = res_conv_pool(pool2, 3, 300)

    attentions = []
    for pool in [pool1, pool2, pool3]:
        lstm = Bidirectional(LSTM(300, dropout=0.2, recurrent_dropout=0.2, return_sequences=True,  kernel_regularizer=regularizers.l2(0.01)))(pool)
        attention_mul = attention_3d_block(lstm,lstm._keras_shape[1])
        summed_attention = Dropout(0.2)(Lambda(lambda x: K.sum(x, axis=1))(attention_mul))
        attentions.append(summed_attention)

    concate_attention = Concatenate()(attentions)

    fc = Dense(1200, activation="relu",  kernel_regularizer=regularizers.l2(0.01))(concate_attention)
    out = Dense(NR_CLASS, activation="softmax")(fc)
    model = Model(inputs = inp, outputs = out)
    return model


def bytenet_resblock(inp, rate, size=3):
    assert(len(inp.shape)==3)
    in_dim = int(inp.shape[2])
    #norm_inp = BatchNormalization()(inp)
    #activated = Activation("relu")(norm_inp)
    conved = Conv1D(filters= in_dim//2, kernel_size=3, padding = "same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(inp)
    out = Conv1D(filters = in_dim, kernel_size=size, padding = "same", dilation_rate=rate, activation="relu", kernel_regularizer=regularizers.l2(0.01))(conved)
    summed = Add()([out, inp])
    return summed

def bytenet_model():
    inp = Input((60,300))
    out = inp
    outs = []
    for dilation in [1,2,4,8]:
        out = Dropout(0.2)(bytenet_resblock(out, dilation))
        outs.append(out)
    attentions = []
    for out in outs:
        attention_mul = attention_3d_block(out, out._keras_shape[1])
        summed_attention = Dropout(0.2)(Lambda(lambda x: K.sum(x, axis=1))(attention_mul))
        attentions.append(summed_attention)
    concate_attention = Concatenate()(attentions)
    fc = Dense(400, activation="relu", kernel_regularizer=regularizers.l2(0.01))(concate_attention)
    out = Dense(5, activation="softmax")(fc)
    model = Model(inputs=inp, outputs=out)
    return model


def prepare_train_data(vectorizer):
    return Dataset.flattened_datapoint(Dataset.get_raw_train_dataset(), vectorizer, filter_missing=False, binarize=BINARIZE, sample_all=SAMPLE_ALL)

def prepare_train_gen(vectorizer):
    return Dataset.flattened_single_generator(Dataset.get_raw_train_dataset(), vectorizer, filter_missing=False, binarize=BINARIZE, sample_all=SAMPLE_ALL)


def prepare_val_data(vectorizer):
    return Dataset.flattened_datapoint(Dataset.get_raw_val_dataset(), vectorizer, filter_missing=False, binarize=BINARIZE)

def prepare_test_data(vectorizer):
    return Dataset.flattened_datapoint(Dataset.get_raw_test_dataset(), vectorizer, binarize=BINARIZE)

def main():
    print("Building model...")
    #model = baseline_model()
    #model = attention_model()
    #model = biattention_model()
    #model = MT_attention_model()
    #model  = convolution_model()
    #model = convolution_attention_model()
    #model = conv_lstm_model()
    model = bytenet_model()

    print(model.summary())
    vectorizer = GloVe()
    print("Preparing data...")
    train_data, train_label = prepare_train_data(vectorizer)
    train_label = to_categorical(train_label)
    val_data, val_label = prepare_val_data(vectorizer)
    val_label = to_categorical(val_label)


    # try using different optimizers and different optimizer configs
    model.compile(
        loss=[focal_loss(2,2)],
        #loss="categorical_crossentropy",
        optimizer=optimizers.Adam(),
        metrics=['accuracy'])

    print('Train...')

    reduce_lr = ReduceLROnPlateau(verbose=1)
    early_stop = EarlyStopping(patience=3, verbose=1)
    model.fit(train_data, train_label,
              batch_size=32,
              epochs=100,
              #validation_split=0.2,
              validation_data=(val_data, val_label),
              verbose=2,
              callbacks=[reduce_lr, early_stop])

    test_data, test_label = prepare_test_data(vectorizer)
    test_label = to_categorical(test_label)
    score, acc = model.evaluate(test_data, test_label,
                                batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__=="__main__":
    from definitions import verify_cwd
    verify_cwd()
    main()