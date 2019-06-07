from keras import backend as K

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    
    print(labels)
    
    # K.ctc_batch_cost runs CTC loss algorithm on each batch element
    #    y_true: tensor (samples, max_string_length) containing the truth labels.
    #    y_pred: tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.
    #    input_length: tensor (samples, 1) containing the sequence length for each batch item in y_pred.
    #    label_length: tensor (samples, 1) containing the sequence length for each batch item in y_true.
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)