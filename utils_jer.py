"""
 Copyright 2018 - by Jerome Tubiana (jertubiana@@gmail.com)
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
import pandas as pd
import numpy as np
from config import aadict, amino_acids, list_aa, chemistry_aromatic
from utils import load_multiple_seq_align_fasta
import numba
from numba import njit, prange, vectorize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.text import TextPath
from matplotlib.font_manager import FontProperties


curr_float = np.float32
curr_int = np.int16
signature = "(float32[:])(int16[:,:],float32[:,:])"
fp = FontProperties(family="Arial", weight="bold")
globscale = 1.35
LETTERS = dict([ (letter, TextPath((-0.30, 0), letter, size=1, prop=fp) )   for letter in list_aa] )

def aa_color(letter):
    if letter in ['$\\boxminus$']:
        return 'black'
    return chemistry_aromatic[letter]

COLOR_SCHEME = dict( [(letter,aa_color(letter)) for letter in list_aa] )

def build_scores(matrix,epsilon = 1e-4):
    n_sites = matrix.shape[0]
    n_colors = matrix.shape[1]
    all_scores = []
    for site in range(n_sites):
        conservation = np.log2(21) + (np.log2(matrix[site]+epsilon) * matrix[site]).sum()
        liste = []
        order_colors = np.argsort(matrix[site])
        for c in order_colors:
            liste.append( (list_aa[c],matrix[site,c] * conservation) )
        all_scores.append(liste)
    return all_scores

def build_scores2(matrix):
    n_sites = matrix.shape[0]
    n_colors = matrix.shape[1]
    epsilon = 1e-4
    all_scores = []
    for site in range(n_sites):
        liste = []
        c_pos = np.nonzero(matrix[site] >= 0)[0]
        c_neg = np.nonzero(matrix[site] < 0)[0]

        order_colors_pos = c_pos[np.argsort(matrix[site][c_pos])]
        order_colors_neg = c_neg[np.argsort(-matrix[site][c_neg])]
        for c in order_colors_pos:
            liste.append( (list_aa[c],matrix[site,c],'+') )
        for c in order_colors_neg:
            liste.append( (list_aa[c],-matrix[site,c],'-') )
        all_scores.append(liste)
    return all_scores

def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def sequence_logo(matrix, ax = None,data_type=None,figsize=None,ylabel = None,title=None,epsilon=1e-4,show=True,ticks_every=1,ticks_labels_size=14,title_size=20):
    if data_type is None:
        if matrix.min()>=0:
            data_type='mean'
        else:
            data_type = 'weights'

    if data_type == 'mean':
        all_scores = build_scores(matrix,epsilon=epsilon)
    elif data_type =='weights':
        all_scores = build_scores2(matrix)
    else:
        print('data type not understood')
        return -1

    if ax is not None:
        show = False
        return_fig = False
    else:
        if figsize is None:
            figsize = (  max(int(0.3 * matrix.shape[0]), 2)  ,  3)        
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True

    x = 1
    maxi = 0
    mini = 0
    for scores in all_scores:
        if data_type == 'mean':
            y = 0
            for base, score in scores:
                if score > 0.01:
                    letterAt(base, x,y, score, ax)
                y += score
            x += 1
            maxi = max(maxi, y)


        elif data_type =='weights':
            y_pos = 0
            y_neg = 0
            for base,score,sign in scores:
                if sign == '+':
                    letterAt(base, x,y_pos, score, ax)
                    y_pos += score
                else:
                    y_neg += score
                    letterAt(base, x,-y_neg, score, ax)
            x += 1
            maxi = max(y_pos,maxi)
            mini = min(-y_neg,mini)

    if data_type == 'weights':
        maxi = max(  maxi, abs(mini) )
        mini = -maxi

    if ticks_every > 1:
        xticks = range(1,x)
        xtickslabels = ['%s'%k if k%ticks_every==0 else '' for k in xticks]
        ax.set_xticks(xticks,xtickslabels)
    else:
        ax.set_xticks(range(1,x))
    ax.set_xlim((0, x))
    ax.set_ylim((mini, maxi))
    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Conservation (bits)'
        elif data_type =='weights':
            ylabel = 'Weights'
    ax.set_ylabel(ylabel,fontsize=title_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='major', labelsize=ticks_labels_size)
    ax.tick_params(axis='both', which='minor', labelsize=ticks_labels_size)
    if title is not None:
        ax.set_title(title,fontsize=title_size)
    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
        return fig
    
def shorten_alignement(data_aligned_path, data_short_path):
    sequences, labels = load_FASTA(data_aligned_path, with_labels=True,drop_duplicates=False)
    empty_columns = (sequences==20).mean(0)>0.9#0.99
    sequences_cleaned = sequences[:,~empty_columns]
    write_FASTA(data_short_path,sequences_cleaned,all_labels=labels)

def print_alignement(data):
    all_data = load_multiple_seq_align_fasta(data)
    all_data_for_jer = (all_data-1)%21
    a= all_data_for_jer.astype(np.int16)
    mu = average_C(a, 21)
    sequence_logo(mu, ticks_every=1);
    
    
signature = "(float32[:,:])(int16[:,:],int64)"

@njit(signature,parallel=False,cache=True,nogil=False)
def average_C(config,q):
    B = config.shape[0]
    N = config.shape[1]
    out = np.zeros((N,q) ,dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[n, config[b,n]] +=1
    out/=B
    return out

def load_FASTA(filename,with_labels=False, remove_insertions = True,drop_duplicates=True):
    count = 0
    current_seq = ''
    all_seqs = []
    if with_labels:
        all_labels = []
    with open(filename,'r') as f:
        for line in f:
            if line[0] == '>':
                all_seqs.append(current_seq)
                current_seq = ''
                if with_labels:
                    all_labels.append(line[1:].replace('\n','').replace('\r','').strip())
            else:
                current_seq+=line.replace('\n','').replace('\r','')
                count+=1
        all_seqs.append(current_seq)
        all_seqs=np.array(list(map(lambda x: [aadict[y] for y in x],all_seqs[1:])),dtype=curr_int,order="c")

    if remove_insertions:
        all_seqs = np.asarray(all_seqs[:, ((all_seqs == -1).max(0) == False) ],dtype=curr_int,order='c')

    if drop_duplicates:
        all_seqs = pd.DataFrame(all_seqs).drop_duplicates()
        if with_labels:
            all_labels = np.array(all_labels)[all_seqs.index]
        all_seqs = np.array(all_seqs)

    if with_labels:
        return all_seqs, np.array(all_labels)
    else:
        return all_seqs

def binarize(all_sequences,n_c=21):
    binary_data = np.concatenate([(all_sequences == c)[:, :, np.newaxis]
                                  for c in range(n_c)], axis=-1)
    return binary_data


def write_FASTA(filename,all_data,all_labels=None):
    sequences = num2seq(all_data)
    if all_labels is None:
        all_labels = ['S%s'%k for k in range(len(sequences))]
    with open(filename,'w') as fil:
        for seq, label in zip(sequences,all_labels):
            fil.write('>%s\n'%label)
            fil.write('%s\n'%seq)
    return 'done'

def num2seq(num):
    return [''.join([amino_acids[x] for x in num_seq]) for num_seq in num]

