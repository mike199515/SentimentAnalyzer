from keras.layers import merge, Permute, Reshape, Lambda, RepeatVector, Dense, Flatten
from keras.layers import Add, Multiply, Concatenate
from keras.layers import Bidirectional, LSTM
SINGLE_ATTENTION_VECTOR = False

def attention_3d_block(inputs, time_steps = 60):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_MT_LSTM(inputs, time_steps = 60):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    bi_lstm = Bidirectional(LSTM(input_dim, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(inputs)
    a = Permute((2, 1))(bi_lstm)
    a = Reshape((input_dim*2, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([bi_lstm, a_probs])
    transformed_attention = Dense(input_dim,activation="relu")(output_attention_mul)
    concated = Concatenate()([transformed_attention, inputs])
    return concated


def attention_after_lstm(lstm_output):
    attention_mul = attention_3d_block(lstm_output)
    attention_mul = Flatten()(attention_mul)
