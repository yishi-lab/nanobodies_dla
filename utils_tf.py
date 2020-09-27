"""
 Copyright 2020 - by Lirane Bitton (liranebitton@gmail.com)
     All rights reserved

     Permission is granted for anyone to copy, use, or modify this
     software for any uncommercial purposes, provided this copyright
     notice is retained, and note is made of any changes that have
     been made. This software is distributed without any warranty,
     express or implied. In no event shall the author or contributors be
     liable for any damage arising out of the use of this software.

     The publication of research using this software, modified or not, must include
     appropriate citations to:
"""
from tensorflow.keras.utils import to_categorical
from config import le, aa, le_align, aa_align
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def to_dataset(dict_to_parse, data, labels, max_seq_size):
    for key, value in dict_to_parse.items():
        for i, seq in enumerate(value['sequence']):
            to_add = to_one_hot_prot(seq, max_seq_size)
            data[value['origin'][i]].append(to_add)
            labels[value['origin'][i]].append(value['labels'][i])


def to_one_hot_prot(seq, max_seq_size=140):
    integer_encoded = le.transform(list(seq))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    encoded = to_categorical(integer_encoded, num_classes=aa.__len__())
    to_add = np.zeros((max_seq_size, aa.__len__()))
    to_add[:encoded.shape[0], :encoded.shape[1]] = encoded
    return to_add


def to_one_hot_prot_align(seq, max_seq_size=140):
    integer_encoded = le_align.transform(list(seq))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    encoded = to_categorical(integer_encoded, num_classes=aa_align.__len__())
    to_add = np.zeros((max_seq_size, aa_align.__len__()))
    to_add[:encoded.shape[0], :encoded.shape[1]] = encoded
    return to_add


def plot_history(history, _loss_fn=None, _acc_fn=None):
    history_dict = history.history
    history_dict.keys()

    import matplotlib.pyplot as plt

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.figure()
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if None == _loss_fn:
        plt.show()
    else:
        plt.savefig(_loss_fn)
        # plt.show()

    plt.figure()  # new figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    if None == _acc_fn:
        plt.show()
    else:
        plt.savefig(_acc_fn)
        # plt.show()


def split_data(data, labels, per):
    indices = np.array(range(len(data)))
    random.shuffle(indices)
    ind = math.floor(len(data) * per)

    return {
        "train_data": data[indices[:ind]],
        "train_labels": labels[indices[:ind]],
        "test_data": data[indices[ind:]],
        "test_labels": labels[indices[ind:]],
    }


def load_data(path_to_train, path_to_test):
    npzfile_train = np.load(path_to_train)
    npzfile_test = np.load(path_to_test)
    return {
        "train_data": npzfile_train['data'],
        "train_labels": npzfile_train['labels'],
        "test_data": npzfile_test['data'],
        "test_labels": npzfile_test['labels'],
    }


def predict(model_path, data_path):
    npzfile_test = np.load(data_path)
    _test_data = npzfile_test['data']
    _test_labels = npzfile_test['labels']

    _classifier = tf.keras.models.load_model(model_path)

    _layer_outputs = [layer.output for layer in _classifier.layers[:]]
    activation_model = tf.keras.models.Model(
        inputs=_classifier.input, outputs=_layer_outputs)

    _activations = activation_model.predict(_test_data)
    return _test_data, _test_labels, _classifier, _layer_outputs, _activations
