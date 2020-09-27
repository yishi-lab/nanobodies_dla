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
import logging
import numpy as np
import utils
from os import path
from math import floor
import activation_path
import utils_tf
logger = logging.getLogger('model_analysis')

RIGHT_LABEL=0
PREDICTED_LABEL=1
CDR_NUM = 3
BEST_FILTER_ACTIV_ORDER=0
BEST_FILTER_SIGNAL_ORDER=0


def print_filter_true_res(txt, cdr_num, aa, test, label, best_to_show):
    denominateur = np.sum(test == label)
    if denominateur != 0:
        a = np.array(aa) / denominateur
        logger.info("{} filter for cdr {} in prediction {}: ".format(txt, cdr_num, int(label)))
        logger.info(np.argsort(a)[-best_to_show:])
        logger.info(np.sort(a)[-best_to_show:])
    else:
        logger.info("NO {} filter for cdr {} in prediction {}: ".format(txt, cdr_num, int(label)))

def print_filter_false_res(txt, cdr_num, aa, test, label, best_to_show):
    denominateur = np.sum(test != label)
    if denominateur != 0:
        a = np.array(aa) / denominateur
        logger.info("{} filter for cdr {} in prediction {}: ".format(txt, cdr_num, int(label)))
        logger.info(np.argsort(a)[-best_to_show:])
        logger.info(np.sort(a)[-best_to_show:])
    else:
        logger.info("NO {} filter for cdr {} in prediction {}: ".format(txt, cdr_num, int(label)))



def init_stats(num_cdr, num_label, num_of_filter):
    stats=dict()
    for cdr_num in range(1, num_cdr+1):
        stats['total_filters_cdr{}'.format(cdr_num)] = [0] * num_of_filter
        for label in range(num_label):
            stats['total_filters_cdr{}_l{}'.format(cdr_num, label)] = [0] * num_of_filter
            stats['total_filters_cdr{}_l{}_best'.format(cdr_num, label)] = [0] * num_of_filter
            stats['total_filters_cdr{}_l{}_worst'.format(cdr_num, label)] = [0] * num_of_filter
            stats['total_filters_cdr{}_l{}_bad'.format(cdr_num, label)] = [0] * num_of_filter
    return stats

def init_id_res(num_label):
    id_results = {}
    for l1 in range(num_label):
        for l2 in range(num_label):
            id_results['label_{}_predict_{}'.format(l1, l2)] = []
    return id_results

def update_stats(stats, cdr_num, compare, index, ind, f, better_than, labels, score):
    for label in labels:
        if compare[index][PREDICTED_LABEL] == label:
            if compare[index][RIGHT_LABEL] != label:
                # wrong prediction
                if ind > better_than:
                    stats['total_filters_cdr{}_l{}_worst'.format(cdr_num, int(label))][f] = stats['total_filters_cdr{}_l{}_worst'.format(cdr_num, int(label))][f] + score
                stats['total_filters_cdr{}_l{}_bad'.format(cdr_num, int(label))][f] = stats['total_filters_cdr{}_l{}_bad'.format(cdr_num, int(label))][f] + score
            else:
                if ind > better_than:
                    stats['total_filters_cdr{}_l{}_best'.format(cdr_num, int(label))][f] = stats['total_filters_cdr{}_l{}_best'.format(cdr_num, int(label))][f] + score
                stats['total_filters_cdr{}_l{}'.format(cdr_num, int(label))][f] = stats['total_filters_cdr{}_l{}'.format(cdr_num, int(label))][f] + score
                
def log_general_stats(_comb, test_labels, compare, labels):
    
    logger.info("STATS for {}".format(_comb))
    logger.info("total labels: ", )
    logger.info(test_labels.__len__())
    for label in labels:
        logger.info("total {} labels: ".format(int(label)))
        logger.info(np.sum(test_labels == int(label)))
        logger.info("total {} predict: ".format(int(label)))
        logger.info(np.sum(compare[:, 1] == label))

    # diff_0_predict = compare[:, 1][compare[:, 0] == 0.0]
    # diff_1_predict = compare[:, 1][compare[:, 0] == 1.0]
    # diff_2_predict = compare[:, 1][compare[:, 0] == 2.0]
    diff = dict()
    for label in labels:
        diff['while_predict_{}'.format(int(label))] = compare[:, PREDICTED_LABEL][compare[:, RIGHT_LABEL] == label]
        for pred in labels:
            if pred == label:
                continue
            logger.info("total predict {} while label {}:".format(pred,label))
            logger.info(np.sum(diff['while_predict_{}'.format(int(label))] == pred))
    return diff

def model_analysis(_test, _input, _model, all_data_ids_fn, _comb, _ids, filename, labels=[0.0,1.0,2.0], percentile=97):
    logger.setLevel(logging.INFO)
    # filename = _test.split('/')[-1].split('.')[0]
    handler = logging.FileHandler(path.join(_input, './cluster_analysis/{}.log'.format(filename)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('Start Cluster Analysis')
    logger.info('run arguments')
    logger.info(_test)
    logger.info(_input)
    logger.info(_model)
    logger.info(_comb)
    logger.info(_ids)

    test_data, test_labels, classifier, layer_outputs, activations = utils_tf.predict(_model, _test)

    best_filter_pos = activation_path.activation_path_model_conv1d_max_flatten_dense(activations, classifier.layers, percentile)
    compare = utils.join_labels_n_predict(test_labels, np.argmax(activations[-1], axis=1))

    all_data_ids = utils.load_pickle(all_data_ids_fn)
    test_ids = utils.load_pickle(_ids)
    cdr_list_keys = [e for e in list(all_data_ids[list(all_data_ids.keys())[0]].keys()) if e.startswith('cdr')] 
    filter_size, _, num_of_filter = classifier.layers[0].kernel.shape
    
    label_num = labels.__len__()
    stats = init_stats(CDR_NUM, label_num, num_of_filter)
    id_results = init_id_res(label_num)
    better_than = floor(0.6*num_of_filter) #0.6 is defined  by user for threshold of best activity

    id_dict_result = dict()
    for index, test in enumerate(test_data):
        id = test_ids[index]
        test_seq = utils.le.inverse_transform(np.argmax(test, axis=1))
        id_seq_str = all_data_ids[id]['sequence'][0]
        id_seq = np.array(list(id_seq_str))
        best_fil_pos = best_filter_pos[BEST_FILTER_SIGNAL_ORDER][index]
        # check it's the same sequence
        if np.array_equal(test_seq[:id_seq.__len__()], id_seq):

            id_results['label_{}_predict_{}'.format(int(compare[index][0]), int(compare[index][1]))].append(id)
            id_dict_result[id] = {
                'bad_filters': [],
                'good_filters': [],
            }

            cdr1 = all_data_ids[id]['cdr1'][0]
            cdr2 = all_data_ids[id]['cdr2'][0]
            cdr3 = all_data_ids[id]['cdr3'][0]
            # cdr3 = cdr3[1:cdr3.find(':')]
            range_cdr1 = (id_seq_str.find(cdr1), id_seq_str.find(cdr1) + cdr1.__len__())
            range_cdr2 = (id_seq_str.find(cdr2), id_seq_str.find(cdr2) + cdr2.__len__())
            range_cdr3 = (id_seq_str.find(cdr3), id_seq_str.find(cdr3) + cdr3.__len__())
            filt = [0] * num_of_filter
            remark = []
            for ind, pos_pair in enumerate(best_fil_pos):
                f, pos_d, conv_activ, signal = pos_pair
                if signal < 0.1:
                    continue
                pos=int(pos_d)
                end_pos = pos + filter_size
                if end_pos >= id_seq.__len__():
                    end_pos = id_seq.__len__() - 1
                    remark.append('filter#{} activate outside seq in padding region'.format(f))
                fil_range = range(pos, end_pos)
                filt[f] = ''.join(id_seq[fil_range])

                cdr1_inter = set(list(range(range_cdr1[0], range_cdr1[1]))).intersection(list(fil_range))
                cdr2_inter = set(list(range(range_cdr2[0], range_cdr2[1]))).intersection(list(fil_range))
                cdr3_inter = set(list(range(range_cdr3[0], range_cdr3[1]))).intersection(list(fil_range))
                if cdr1_inter.__len__() > 0:
                    update_stats(stats,1,compare,index,ind, f,better_than,labels,cdr1_inter.__len__()/filter_size)
                    # update_stats(stats,1,compare,index,ind, f,better_than,labels,1)
                    stats['total_filters_cdr1'][f] = stats['total_filters_cdr1'][f] + 1
                    
                elif cdr2_inter.__len__() > 0:
                    update_stats(stats,2,compare,index,ind, f,better_than,labels,cdr2_inter.__len__()/filter_size)
                    # update_stats(stats,2,compare,index,ind, f,better_than,labels,1)
                    stats['total_filters_cdr2'][f] = stats['total_filters_cdr2'][f] + 1
                    
                elif cdr3_inter.__len__() > 0:
                    update_stats(stats,3,compare,index,ind, f,better_than,labels,cdr3_inter.__len__()/filter_size)
                    # update_stats(stats,3,compare,index,ind, f,better_than,labels,1)
                    stats['total_filters_cdr3'][f] = stats['total_filters_cdr3'][f] + 1
        else:
            logger.info(id)

    diff = log_general_stats(_comb, test_labels, compare, labels)
    for cdr_num in range(1, 4):
        for label in labels:
            logger.info('')
            print_filter_true_res('', cdr_num, stats['total_filters_cdr{}_l{}'.format(cdr_num, int(label))], compare[:, 0], label, better_than)
            print_filter_true_res('best', cdr_num, stats['total_filters_cdr{}_l{}_best'.format(cdr_num, int(label))], compare[:, 0], label, better_than)
            print_filter_false_res('bad', cdr_num, stats['total_filters_cdr{}_l{}_bad'.format(cdr_num, int(label))], diff['while_predict_{}'.format(int(label))], label, better_than)
            print_filter_false_res('worst', cdr_num, stats['total_filters_cdr{}_l{}_worst'.format(cdr_num, int(label))], diff['while_predict_{}'.format(int(label))], label, better_than)
            
    handler.flush()
    handler.close()
    logger.removeHandler(handler)
