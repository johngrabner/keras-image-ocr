
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers import Dropout
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
import keras.callbacks
import ctc_drop_first_2


def make_model(img_w, img_h, output_size, absolute_max_string_len):

    # Network parameters
    conv_filters = 16
    #conv_filters = 32 # experiment 2
    kernel_size = (3, 3)
    time_dense_size = 32
    rnn_size = 512
    pool_size = 2

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    type_of_model = "original" # "https://keras.io/examples/mnist_cnn/"
    if type_of_model == "https://keras.io/examples/mnist_cnn/":
        inner = Conv2D(32, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
        inner = Conv2D(64, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
        inner = Dropout(0.25)(inner) # Fraction of the input units to drop
        conv_filters = 64
    else:
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='afilter'+str(conv_filters))(input_data)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='apool'+str(pool_size)+"by"+str(pool_size))(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name='bfilter'+str(conv_filters))(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='bpool'+str(pool_size)+"by"+str(pool_size))(inner)
        # experiment 3b ... add dropout
        #inner = Dropout(0.5)(inner) # Fraction of the input units to drop
        # experiment 3c ... add dropout
        inner = Dropout(0.25)(inner) # Fraction of the input units to drop

    # image is down sampled by MaxPooling twice, hence pool_size ** 2
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    # experiment 3 removes this reduction
    #inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    if type_of_model == "https://keras.io/examples/mnist_cnn/":
        inner = Dropout(0.5)(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(output_size, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)

    # this intermediate point is usefull for predictions without training
    model_p = Model(inputs=input_data, outputs=y_pred)
 
    labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
   
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    
    # K.Lambda wraps arbitrary expression as a Layer object.
    # Q then its called ?
    loss_out = Lambda(ctc_drop_first_2.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])



    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    
    return (model, model_p, input_data, y_pred)



    

    
