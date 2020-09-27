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
import os
import argparse
import shutil

def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True,  help='folder to clean')
    parser.add_argument('--thr', required=True, help='threshold to cut off')
    parser.add_argument('--name_parts', default=2, help='threshold to cut off')
    argums = parser.parse_args()
    return argums

def cleanup(folder, patterns):
    for root, dirs, files in os.walk(folder):
        for name in files:
            for pat in patterns:
                if pat in name:
                    print("remove file: ", os.path.join(root, name))
                    os.remove(os.path.join(root, name))
        for name in dirs:
            for pat in patterns:
                if pat in name:
                    print("remove dir: ", os.path.join(root, name))
                    shutil.rmtree(os.path.join(root, name))
    
    
def cleanup_folder(folder, patterns):
    os.walk(folder)
    
if __name__ == "__main__":
    args = myargs()
    log_folder = os.path.join(args.input_path, 'logs')
    files = os.listdir(log_folder)
    to_del=[]
    for f in files:
        logfilename = f.split('.log')
        if logfilename.__len__()<2:
            continue
        parse_filename= logfilename[0].split('_')
        if parse_filename.__len__() == int(args.name_parts):
            to_del.append(parse_filename[-1])
        else:
            acc = float(logfilename[0].split('_acc_')[1])
            if acc < float(args.thr):
                to_del.append(logfilename[0].split('_')[1])
            
    if to_del.__len__() > 0:
        cleanup(os.path.join(args.input_path, 'logs'), to_del)
        cleanup(os.path.join(args.input_path, 'best_models'), to_del)
        cleanup(os.path.join(args.input_path, 'filters'), to_del)
        cleanup(os.path.join(args.input_path, 'cluster_analysis'), to_del)
        cleanup(os.path.join(args.input_path, 'graph'), to_del)