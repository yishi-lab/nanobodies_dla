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
import argparse
import numpy as np
import tensorflow as tf

BEST_TO_SHOW = 10
FILTER_LENGTH = 7
NUM_OF_FILTER = 20
BETTER_THAN = 10
GREATER_INTER = 2


FULLY_DENSE_LAYER=6
ACTIVATION_BEFORE_FULLY=5
OUTPUT_ACTIVATION=7
ACTIVATION_CONV_LAYER=2
ACTIVATION_MAXPOOL_LAYER=3

def retrieve_activation_path(_activations):
    data_to_ret = dict()
    for index, activ in enumerate(_activations[2]):
        activ[activ == 0] = -1
        all_active_filter = np.in1d(activ, _activations[3][index])
        all_reshape = np.reshape(all_active_filter, activ.shape)
        max_indices_positions = np.argmax(activ * all_reshape, axis=0)
        best_activ = activ[max_indices_positions, range(activ.shape[1])]
        best_filter_sort = np.argsort(best_activ)
        prot_pos = max_indices_positions[best_filter_sort]
        data_to_ret[index] = list(zip(best_filter_sort, prot_pos))
    return data_to_ret


def retrieve_full_path_activation(_activations, _layers):
    fully, bias_fully = _layers[FULLY_DENSE_LAYER].get_weights()
    fully_mul = fully.T.reshape((fully.shape[1], 1, fully.shape[0]))
    signal_layer = np.multiply(np.array(_activations[ACTIVATION_BEFORE_FULLY]), fully_mul)
    percentiles = np.zeros((signal_layer.shape[0], signal_layer.shape[1], 1))
    for i in range(signal_layer.shape[0]):
        percentiles[i][:] = np.percentile(signal_layer[i], 97, axis=1).reshape((signal_layer.shape[1], 1))

    ind = signal_layer >= percentiles
    ind = ind.reshape(signal_layer.shape[0], signal_layer.shape[1], _activations[3].shape[1], _activations[3].shape[2])
    data_to_ret = dict()
    for index, activ in enumerate(_activations[2]):
        activ[activ == 0] = -1
        all_active_filter = np.in1d(activ, _activations[3][index][ind[np.argmax(_activations[7][index])][index]])
        all_reshape = np.reshape(all_active_filter, activ.shape)
        max_indices_positions = np.argmax(activ * all_reshape, axis=0)
        best_activ = activ[max_indices_positions, range(activ.shape[1])]
        best_filter_sort = np.argsort(best_activ)
        prot_pos = max_indices_positions[best_filter_sort]
        data_to_ret[index] = list(zip(best_filter_sort, prot_pos, np.sort(best_activ)))
    return data_to_ret


def retrieve_full_path_activation_fullyconnected(_activations, _layers):
    fully_3, bias_fully_3 = _layers[9].get_weights()
    fully_3mul = fully_3.T.reshape((fully_3.shape[1], 1, fully_3.shape[0]))
    signal_3layer = np.multiply(np.array(_activations[8]), fully_3mul)
    percentiles3 = np.zeros((signal_3layer.shape[0], signal_3layer.shape[1], 1))
    for i in range(signal_3layer.shape[0]):
        percentiles3[i][:] = np.percentile(signal_3layer[i], 97, axis=1).reshape((signal_3layer.shape[1], 1))

    ind3 = signal_3layer >= percentiles3

    fully, bias_fully = _layers[FULLY_DENSE_LAYER].get_weights()
    fully_mul = fully.T.reshape((fully.shape[1], 1, fully.shape[0]))
    signal_layer = np.multiply(np.array(_activations[ACTIVATION_BEFORE_FULLY]), fully_mul)
    percentiles = np.zeros((signal_layer.shape[0], signal_layer.shape[1], 1))
    for i in range(signal_layer.shape[0]):
        percentiles[i][:] = np.percentile(signal_layer[i], 97, axis=1).reshape((signal_layer.shape[1], 1))

    ind_flat = signal_layer >= percentiles

    ind3_roll = np.rollaxis(ind3, axis=1)
    ind3_flat_roll = np.rollaxis(ind_flat, axis=1)
    matmul_ind = np.matmul(ind3_roll, ind3_flat_roll)

    ind = matmul_ind.reshape(signal_3layer.shape[1], signal_3layer.shape[0], _activations[3].shape[1],
                            _activations[3].shape[2])

    data_to_ret3 = dict()
    for index, activ in enumerate(_activations[2]):
        activ[activ == 0] = -1
        all_active_filter = np.in1d(activ, _activations[3][index][ind[index][np.argmax(_activations[10][index])]])
        all_reshape = np.reshape(all_active_filter, activ.shape)
        max_indices_positions = np.argmax(activ * all_reshape, axis=0)
        best_activ = activ[max_indices_positions, range(activ.shape[1])]
        best_filter_sort = np.argsort(best_activ)
        prot_pos = max_indices_positions[best_filter_sort]
        data_to_ret3[index] = list(zip(best_filter_sort, prot_pos, np.sort(best_activ)))
    return data_to_ret3

def dense_activation(before_activation, curr_layer, percentile):
    fully, bias_fully = curr_layer.get_weights()
    fully_mul = fully.T.reshape((fully.shape[1], 1, fully.shape[0]))
    _signal_layer = np.multiply(np.array(before_activation), fully_mul)
    _percentiles = np.zeros((_signal_layer.shape[0], _signal_layer.shape[1], 1))
    for i in range(_signal_layer.shape[0]):
        _percentiles[i][:] = np.percentile(_signal_layer[i], percentile, axis=1).reshape((_signal_layer.shape[1], 1))

    return _signal_layer, _percentiles
    
def conv1d_activation():
    pass

def output_activation(curr_activation):
    return np.argmax(curr_activation, axis=1)

def flatten_activation(before_activation, last):
    ind_to_ret = last.reshape(last.shape[0], last.shape[1], before_activation.shape[1], before_activation.shape[2])
    return ind_to_ret

def activation_path_model_conv1d_max_flatten_dense(_activations, _layers, percentile):
    signal_layer, percentiles = dense_activation(_activations[ACTIVATION_BEFORE_FULLY], _layers[FULLY_DENSE_LAYER], percentile)
    indices_above_percentile = signal_layer >= percentiles
    best_pred_label = output_activation(_activations[OUTPUT_ACTIVATION])
    indices_above_percentile_max_reshape = flatten_activation(_activations[ACTIVATION_MAXPOOL_LAYER], indices_above_percentile)
    best_filter_pos_to_ret = dict()
    all_filter_pos_to_ret = dict()
    
    
    input_max_layer =np.expand_dims(np.array(_activations[ACTIVATION_CONV_LAYER], copy=True), axis=3)
        
    stride = _layers[ACTIVATION_MAXPOOL_LAYER].strides[0]
    pool = _layers[ACTIVATION_MAXPOOL_LAYER].pool_size[0]
    output_max_layer_with_args = tf.nn.max_pool_with_argmax(input_max_layer,
                                                            ksize=(1,pool,1,1),
                                                            strides=(1,stride,1,1),
                                                            padding='SAME')
    all_values_maxpool = np.squeeze(output_max_layer_with_args[0],axis=3)
    all_indices_maxpool = np.squeeze(output_max_layer_with_args[1],axis=3)
    for index, activ in enumerate(_activations[ACTIVATION_CONV_LAYER]):
        curr_pred = best_pred_label[index]
        curr_mask = indices_above_percentile_max_reshape[curr_pred][index]
        curr_signal =  signal_layer[curr_pred,index,indices_above_percentile[curr_pred][index]]
        curr_ind = np.where(curr_mask == True)
        curr_signal_sorted = np.sort(curr_signal)
        curr_signal_argsorted = np.argsort(curr_signal)
        
        
        
        orig_indices = np.unravel_index(all_indices_maxpool[index], activ.shape)
        activ_with_signal_values_argmax = np.zeros(activ.shape)
        # activ_with_signal_values_argmax[orig_indices] = all_values_maxpool[index]
        
        # in order to update only the best one we apply the mask then we can retrieve the max one which will be the same without the mask
        # or all the filter even duplicate activation then we need the mask
        activ_with_signal_values_argmax[orig_indices[0][curr_mask],orig_indices[1][curr_mask]] = all_values_maxpool[index][curr_mask]
        # activ[activ == 0] = -1
        
        # all_active_filter = np.in1d(activ, _activations[ACTIVATION_MAXPOOL_LAYER][index][indices_above_percentile_max_reshape[best_pred_label[index]][index]])
        # all_reshape = np.reshape(all_active_filter, activ.shape)
        # activ_with_signal_values = np.zeros(activ.shape)
        
        
        
        # activ_with_signal_values[all_reshape] = signal_layer[best_pred_label[index],index][indices_above_percentile[best_pred_label[index],index]]
        # max_indices_positions = np.argmax(activ * all_reshape, axis=0)
        # max_indices_positions = np.argmax(activ_with_signal_values,axis=0)
        
        all_indices_positions = np.nonzero(activ_with_signal_values_argmax)
        all_activ = activ[all_indices_positions[0], all_indices_positions[1]]
        
        all_indices_positions_sorted_activation_path = (all_indices_positions[0][curr_signal_argsorted], all_indices_positions[1][curr_signal_argsorted])
        all_activ_sorted_activation_path = activ[all_indices_positions_sorted_activation_path[0],all_indices_positions_sorted_activation_path[1]]
        
        best_signal_activ = np.zeros(activ_with_signal_values_argmax.shape[1])
        prot_pos=np.zeros(activ_with_signal_values_argmax.shape[1])
        best_activ=np.zeros(activ_with_signal_values_argmax.shape[1])
        for i, filt_pos in enumerate(all_indices_positions_sorted_activation_path[1]):
            prot_pos[filt_pos] = all_indices_positions_sorted_activation_path[0][i]
            best_activ[filt_pos] = all_activ_sorted_activation_path[i]
            best_signal_activ[filt_pos] = curr_signal_sorted[i]
        # max_indices_positions = np.argmax(activ_with_signal_values_argmax,axis=0)
        # best_activ = activ[max_indices_positions, range(activ.shape[1])]
        
        # NEED TP REARRANGE FOLLOWING ORIGINAL ORDER
        
        best_filter_sort = np.argsort(best_activ)
        prot_pos = prot_pos[best_filter_sort]
        best_signal_activ = best_signal_activ[best_filter_sort]
        # prot_pos = max_indices_positions[best_filter_sort]
        # here I want to return best filter ordered by contribution in fully for pred with also the score of filter after conv
        best_filter_pos_to_ret[index] = list(zip(best_filter_sort, prot_pos, np.sort(best_activ), best_signal_activ))
        # here I want to return all filter ordered by contribution in fully for pred with also the score of filter after conv
        # new format (filter number, protein position, filter activity, signal contribution) order crescendo following signal contrib 
        all_filter_pos_to_ret[index] = list(zip(all_indices_positions_sorted_activation_path[1], 
                                                all_indices_positions_sorted_activation_path[0], 
                                                all_activ_sorted_activation_path, 
                                                curr_signal_sorted))
    return best_filter_pos_to_ret, all_filter_pos_to_ret

def path_activation(_activations, _layers, percentile):
    if _activations.__len__() != _layers.__len__():
        raise Exception("not same length of activation and layers")
    full_path = []
    for i in range(_layers.__len__()-1, -1, -1):
        if type(_layers[i]) == tf.keras.layers.Dense:
            if i-1 >= 0:
                signal_layer, percentiles = dense_activation(_activations[i-1], _layers[i], percentile)
                ind = signal_layer >= percentiles
                full_path.append(ind)
            else:
                raise Exception("needs data input for analysis")
        elif type(_layers[i]) == tf.keras.layers.Conv1D:
            conv1d_activation()
        elif type(_layers[i]) == tf.keras.layers.Flatten:
            if i-1 >= 0:
                flatten_activation(_activations[i-1], full_path[-1])
            else:
                raise Exception("needs data input for analysis")
        elif type(_layers[i]) == tf.keras.layers.Activation and i == _layers.__len__() - 1: #last layer
            full_path.append(output_activation(_activations[i]))
