import argparse
import datetime
import numpy as np
import os
import pandas as pd
import random

from utils import to_fasta, to_npz, to_pickle, load_pickle, load_multiple_seq_align_fasta
from utils_tf import to_dataset, to_one_hot_prot
ID_COL = 0
SEQ_COL = 1
CR1_COL = 2
CR2_COL = 3
CR3_COL = 4
LABEL_COL = 5

TIMESTAMP = format(f"{datetime.datetime.now():%Y_%m_%d_%H_%M}")
max_seq_size = 0


def parse_pd_ids(all_data, dataframe, cond=None):
    cols = dataframe.columns
    bad_label = 0
    for index, row in dataframe.iterrows():
        if row[cols[LABEL_COL]] == -1:
            bad_label+=1
            continue
        seq = row[cols[ID_COL]]
        global max_seq_size
        max_seq_size = max(max_seq_size, len(row[cols[SEQ_COL]]))
        if seq in all_data:
            all_data[seq]['count'] = all_data[seq]['count'] + 1
            all_data[seq]['cdr1'].append(row[cols[CR1_COL]])
            all_data[seq]['cdr2'].append(row[cols[CR2_COL]])
            all_data[seq]['cdr3'].append(row[cols[CR3_COL]])
            all_data[seq]['labels'].append(row[cols[LABEL_COL]])
            all_data[seq]['sequence'].append(row[cols[SEQ_COL]])
            if cond != None:
                all_data[seq]['cond'].append(cond)
        else:
            if cond != None:
                all_data[seq] = {
                    'sequence': [row[cols[SEQ_COL]]],
                    'count': 1,
                    'cdr1': [row[cols[CR1_COL]]],
                    'cdr2': [row[cols[CR2_COL]]],
                    'cdr3': [row[cols[CR3_COL]]],
                    'labels': [row[cols[LABEL_COL]]],
                    'cond': [cond]
                }
            else:
                all_data[seq] = {
                    'sequence': [row[cols[SEQ_COL]]],
                    'count': 1,
                    'cdr1': [row[cols[CR1_COL]]],
                    'cdr2': [row[cols[CR2_COL]]],
                    'cdr3': [row[cols[CR3_COL]]],
                    'labels': [row[cols[LABEL_COL]]]
                }
    print('max_seq_size: ', max_seq_size)
    return bad_label


def parse_pd_cdr(all_cdr, dataframe, idx_1, idx_2, idx_3):
    cols = dataframe.columns
    bad_label = 0
    for index, row in dataframe.iterrows():
        if row[cols[LABEL_COL]] == -1:
            bad_label += 1
            continue
        if idx_1 == idx_2 == idx_3:
            seq = row[cols[idx_1]]
        elif idx_1 == idx_2:
            seq = row[cols[idx_1]] + row[cols[idx_3]]
        elif idx_1 == idx_3:
            seq = row[cols[idx_1]] + row[cols[idx_2]]
        elif idx_2 == idx_3:
            seq = row[cols[idx_1]] + row[cols[idx_2]]
        else:
            seq = row[cols[idx_1]] + row[cols[idx_2]] + row[cols[idx_3]]

        if seq in all_cdr:
            all_cdr[seq]['count'] = all_cdr[seq]['count'] + 1
            all_cdr[seq]['sequence'].append(row[cols[SEQ_COL]])
            all_cdr[seq]['labels'].append(row[cols[LABEL_COL]])
            all_cdr[seq]['id'].append(row[cols[ID_COL]])
            all_cdr[seq]['cdr1'].append(row[cols[CR1_COL]])
            all_cdr[seq]['cdr2'].append(row[cols[CR2_COL]])
            all_cdr[seq]['cdr3'].append(row[cols[CR3_COL]])
        else:
            all_cdr[seq] = {
                'id': [row[cols[ID_COL]]],
                'count': 1,
                'sequence': [row[cols[SEQ_COL]]],
                'labels': [row[cols[LABEL_COL]]],
                'cdr1': [row[cols[CR1_COL]]],
                'cdr2': [row[cols[CR2_COL]]],
                'cdr3': [row[cols[CR3_COL]]],
            }
    return bad_label



def generate_cdr_combinations(high_ph, low_ph, salt):
    cdr_comb = {
        'high':{
            '123': {},
            '1': {},
            '2': {},
            '3': {},
            '12': {},
            '13': {},
            '23': {}
        },
        'low':{
            '123': {},
            '1': {},
            '2': {},
            '3': {},
            '12': {},
            '13': {},
            '23': {}
        },
        'salt':{
            '123': {},
            '1': {},
            '2': {},
            '3': {},
            '12': {},
            '13': {},
            '23': {}
        }
    }

    parse_pd_cdr(cdr_comb['high']['123'], high_ph, CR1_COL, CR2_COL, CR3_COL)
    parse_pd_cdr(cdr_comb['low']['123'], low_ph, CR1_COL, CR2_COL, CR3_COL)
    parse_pd_cdr(cdr_comb['salt']['123'], salt, CR1_COL, CR2_COL, CR3_COL)

    parse_pd_cdr(cdr_comb['high']['23'], high_ph, CR3_COL, CR2_COL, CR3_COL)
    parse_pd_cdr(cdr_comb['low']['23'], low_ph, CR3_COL, CR2_COL, CR3_COL)
    parse_pd_cdr(cdr_comb['salt']['23'], salt, CR3_COL, CR2_COL, CR3_COL)

    parse_pd_cdr(cdr_comb['high']['13'], high_ph, CR3_COL, CR1_COL, CR3_COL)
    parse_pd_cdr(cdr_comb['low']['13'], low_ph, CR3_COL, CR1_COL, CR3_COL)
    parse_pd_cdr(cdr_comb['salt']['13'], salt, CR3_COL, CR1_COL, CR3_COL)

    parse_pd_cdr(cdr_comb['high']['12'], high_ph, CR2_COL, CR1_COL, CR2_COL)
    parse_pd_cdr(cdr_comb['low']['12'], low_ph, CR2_COL, CR1_COL, CR2_COL)
    parse_pd_cdr(cdr_comb['salt']['12'], salt, CR2_COL, CR1_COL, CR2_COL)

    parse_pd_cdr(cdr_comb['high']['1'], high_ph, CR1_COL, CR1_COL, CR1_COL)
    parse_pd_cdr(cdr_comb['low']['1'], low_ph, CR1_COL, CR1_COL, CR1_COL)
    parse_pd_cdr(cdr_comb['salt']['1'], salt, CR1_COL, CR1_COL, CR1_COL)

    parse_pd_cdr(cdr_comb['high']['2'], high_ph, CR2_COL, CR2_COL, CR2_COL)
    parse_pd_cdr(cdr_comb['low']['2'], low_ph, CR2_COL, CR2_COL, CR2_COL)
    parse_pd_cdr(cdr_comb['salt']['2'], salt, CR2_COL, CR2_COL, CR2_COL)

    parse_pd_cdr(cdr_comb['high']['3'], high_ph, CR3_COL, CR3_COL, CR3_COL)
    parse_pd_cdr(cdr_comb['low']['3'], low_ph, CR3_COL, CR3_COL, CR3_COL)
    parse_pd_cdr(cdr_comb['salt']['3'], salt, CR3_COL, CR3_COL, CR3_COL)



    print("combination of cdr 123 in high ph: ",cdr_comb['high']['123'].__len__())
    print("combination of cdr 13 in high ph: ",cdr_comb['high']['13'].__len__())
    print("combination of cdr 12 in high ph: ",cdr_comb['high']['12'].__len__())
    print("combination of cdr 23 in high ph: ",cdr_comb['high']['23'].__len__())
    print("combination of cdr 1 in high ph: ",cdr_comb['high']['1'].__len__())
    print("combination of cdr 2 in high ph: ",cdr_comb['high']['2'].__len__())
    print("combination of cdr 3 in high ph: ",cdr_comb['high']['3'].__len__())

    print("combination of cdr 123 in low ph: ",cdr_comb['low']['123'].__len__())
    print("combination of cdr 13 in low ph: ",cdr_comb['low']['13'].__len__())
    print("combination of cdr 12 in low ph: ",cdr_comb['low']['12'].__len__())
    print("combination of cdr 23 in low ph: ",cdr_comb['low']['23'].__len__())
    print("combination of cdr 1 in low ph: ",cdr_comb['low']['1'].__len__())
    print("combination of cdr 2 in low ph: ",cdr_comb['low']['2'].__len__())
    print("combination of cdr 3 in low ph: ",cdr_comb['low']['3'].__len__())

    print("combination of cdr 123 in salt: ",cdr_comb['salt']['123'].__len__())
    print("combination of cdr 13 in salt: ",cdr_comb['salt']['13'].__len__())
    print("combination of cdr 12 in salt: ",cdr_comb['salt']['12'].__len__())
    print("combination of cdr 23 in salt: ",cdr_comb['salt']['23'].__len__())
    print("combination of cdr 1 in salt: ",cdr_comb['salt']['1'].__len__())
    print("combination of cdr 2 in salt: ",cdr_comb['salt']['2'].__len__())
    print("combination of cdr 3 in salt: ",cdr_comb['salt']['3'].__len__())



    cdr_123 = dict()
    parse_pd_cdr(cdr_123, high_ph, CR1_COL, CR2_COL, CR3_COL)
    parse_pd_cdr(cdr_123, low_ph, CR1_COL, CR2_COL, CR3_COL)
    parse_pd_cdr(cdr_123, salt, CR1_COL, CR2_COL, CR3_COL)

    print("combination of cdr over all condition 1 2 & 3: ", cdr_123.__len__())

    cdr_23 = dict()

    parse_pd_cdr(cdr_23, high_ph, CR3_COL, CR2_COL, CR3_COL)
    parse_pd_cdr(cdr_23, low_ph, CR3_COL, CR2_COL, CR3_COL)
    parse_pd_cdr(cdr_23, salt, CR3_COL, CR2_COL, CR3_COL)

    print("combination of cdr over all condition 2 & 3: ", cdr_23.__len__())

    cdr_13 = dict()

    parse_pd_cdr(cdr_13, high_ph, CR3_COL, CR1_COL, CR3_COL)
    parse_pd_cdr(cdr_13, low_ph, CR3_COL, CR1_COL, CR3_COL)
    parse_pd_cdr(cdr_13, salt, CR3_COL, CR1_COL, CR3_COL)

    print("combination of cdr over all condition 1 & 3: ", cdr_13.__len__())

    cdr_12 = dict()

    parse_pd_cdr(cdr_12, high_ph, CR2_COL, CR1_COL, CR2_COL)
    parse_pd_cdr(cdr_12, low_ph, CR2_COL, CR1_COL, CR2_COL)
    parse_pd_cdr(cdr_12, salt, CR2_COL, CR1_COL, CR2_COL)

    print("combination of cdr over all condition 1 & 2: ", cdr_12.__len__())

    cdr_1 = dict()

    parse_pd_cdr(cdr_1, high_ph, CR1_COL, CR1_COL, CR1_COL)
    parse_pd_cdr(cdr_1, low_ph, CR1_COL, CR1_COL, CR1_COL)
    parse_pd_cdr(cdr_1, salt, CR1_COL, CR1_COL, CR1_COL)

    print("num of diff cdr over all condition 1: ", cdr_1.__len__())

    cdr_2 = dict()

    parse_pd_cdr(cdr_2, high_ph, CR2_COL, CR2_COL, CR2_COL)
    parse_pd_cdr(cdr_2, low_ph, CR2_COL, CR2_COL, CR2_COL)
    parse_pd_cdr(cdr_2, salt, CR2_COL, CR2_COL, CR2_COL)

    print("num of diff cdr over all condition 2: ", cdr_2.__len__())

    cdr_3 = dict()

    parse_pd_cdr(cdr_3, high_ph, CR3_COL, CR3_COL, CR3_COL)
    parse_pd_cdr(cdr_3, low_ph, CR3_COL, CR3_COL, CR3_COL)
    parse_pd_cdr(cdr_3, salt, CR3_COL, CR3_COL, CR3_COL)

    print("num of diff cdr over all condition 3: ", cdr_3.__len__())

    high = dict()
    low = dict()
    salt_d = dict()

    print("Bad Labeled high ph cdr: ", parse_pd_cdr(high, high_ph, 0, 0, 0))
    print("Bad Labeled low ph cdr: ", parse_pd_cdr(low, low_ph, 0, 0, 0))
    print("Bad Labeled salt cdr: ", parse_pd_cdr(salt_d, salt, 0, 0, 0))
    return cdr_comb

