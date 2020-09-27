import numpy as np
import pickle
from Bio import AlignIO, SeqIO
import matplotlib.pyplot as plt
import argparse
import csv
import random
import string
from config import le, le_align

def to_one_hot(data, num_class):
    binary_data = np.zeros([data.shape[0], data.shape[1], num_class])
    for c in range(num_class):
        binary_data[:, :, c] = (data == c)
    binary_data = binary_data.reshape([data.shape[0], data.shape[1] * num_class])
    return binary_data


def plot_components(principal_comp, cluster=None):
    plt.scatter(principal_comp[:,0],principal_comp[:,1], c=cluster);
    plt.show()


def to_fasta(dict_to_parse, fasta_name, key_seq=False):
    with open(fasta_name, 'w') as the_file:
        if key_seq:
            for key, value in dict_to_parse.items():
                the_file.write('> {}\n{}\n'.format(value['id'][0], key))
        else:
            for key, value in dict_to_parse.items():
                the_file.write('> {}\n{}\n'.format(key, value['sequence'][0]))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_fasta(fasta_name):
    return SeqIO.to_dict(SeqIO.parse(fasta_name, "fasta"))


def load_multiple_seq_align_fasta(fasta_name, encoding=None, with_labels=False):
    alignment = AlignIO.read(fasta_name, 'fasta')
    if with_labels:
        labels = [row.id for row in alignment]
    if encoding is None:
        try:
            sequences = np.asarray([le_align.transform(list(str(row.seq))) for row in alignment])
            if with_labels:
                return sequences, labels
            return sequences
        except:
            raise
    sequences = np.asarray([np.frombuffer(bytes(str(row.seq), encoding=encoding), dtype=np.uint8) for row in alignment])
    if with_labels:
        return sequences, labels
    return sequences

def to_npz(filename, data, labels):
    np.savez_compressed(filename, data=np.array(data), labels=np.array(labels))


def to_pickle(dict_to_pickle, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dict_to_pickle, handle, protocol=4)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        loaded_pickle = pickle.load(handle)
        return loaded_pickle


def join_labels_n_predict(test_labels, layer):
    compare = np.zeros([test_labels.shape[0], 2])
    compare[:, 0] = test_labels
    compare[:, 1] = layer
    return compare


def plot_3d(data, name='default title', target=None):
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.figure())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=target, cmap=plt.cm.Spectral)
    plt.title(name)
    plt.show()


def from_clust_to_dict(clust_file, dict_to, comb_high, comb_low, comb_salt):
    with open(clust_file, 'r') as my_clust:
        line = my_clust.readline()
        while line:
            if line.startswith('>Cluster'):
                curr_cluster = line.rstrip('\n').split(' ')[1]
            else:
                curr_id = line.rstrip('\n').split(' > ')[1].split('...')[0]
                cdr_chain = dict_to[curr_id]['cdr1'][0]+ dict_to[curr_id]['cdr2'][0]+ dict_to[curr_id]['cdr3'][0]
                if cdr_chain in comb_high:
                    for all_id in comb_high[cdr_chain]['id']:
                        dict_to[all_id]['cluster'] = curr_cluster
                if cdr_chain in comb_low:
                    for all_id in comb_low[cdr_chain]['id']:
                        dict_to[all_id]['cluster'] = curr_cluster
                if cdr_chain in comb_salt:
                    for all_id in comb_salt[cdr_chain]['id']:
                        dict_to[all_id]['cluster'] = curr_cluster
            line = my_clust.readline()


def dict_to_csv(d, csv_file):
    with open(csv_file, mode="w") as csv_out:
        my_writer = csv.writer(csv_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        my_writer.writerow(['id','sequence','cdr1', 'cdr2','cdr3','high','low','salt', 'cluster'])
        for curr_id, val in d.items():
            line_to_add=[curr_id, val['sequence'][0], val['cdr1'][0],val['cdr2'][0],val['cdr3'][0], '', '','']
            for i, c in enumerate(val['cond']):
                if c=='high':
                    line_to_add[5]=val['labels'][i]
                if c=='low':
                    line_to_add[6]=val['labels'][i]
                if c=='salt':
                    line_to_add[7]=val['labels'][i]
            line_to_add.append(val['cluster'])
            if line_to_add.__len__() == 9:
                my_writer.writerow(line_to_add)
            else:
                print(curr_id, line_to_add.__len__())

def cluster_res_to_dataset(filename, thr=80):
    train_id = []
    test_id = []
    with open(filename) as my_clust:
        line = my_clust.readline()
        while line:
            if line.startswith('>Cluster'):
                curr_cluster = line.rstrip('\n').split(' ')[1]
            else:
                res_line = line.rstrip('\n').split(' > ')[1].split('... at ')
                if res_line.__len__() == 1:
                    train_id.append(line.rstrip('\n').split(' > ')[1].split('...')[0])
                else:
                    _id = res_line[0]
                    sim_score = res_line[1]
                    if int(sim_score.split('.')[0]) > thr:
                        train_id.append(_id)
                    else:
                        test_id.append(_id)
            line = my_clust.readline()
    return train_id, test_id

def get_run_name(dataset):
    uid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return dataset + '_' + uid , uid
