import argparse
import logging
import math
from os import mkdir
from os import path
import shutil
import json
import pickle
import utils_tf
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import utils
import argparse
import random
import string
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import utils_tf

from model import DeepNano


logger = logging.getLogger('learning_log.model_train')


def train(data, best_folder, run_name, graph_log_dir, param_path, num_classes=3):
    with open(param_path, 'r') as f:
        param = json.load(f)
    batch_size, lr, epoch, filt, ks, dp_out, per, stride, rel_l1 = param.values()
    logger.info(param)
    train_data, train_labels, test_data, test_labels = data.values()
    logger.info('Param to model')
    logger.info(train_data.shape[1])
    logger.info(train_data.shape[2])
    logger.info(filt)
    logger.info(ks)

    model = DeepNano.build(
        train_data.shape[1],
        train_data.shape[2],
        filt,
        ks,
        dp_out,
        stride,
        rel_l1,
        num_classes)

    adam = tf.keras.optimizers.Adam(lr=lr,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-8,
                                    decay=0.0,
                                    amsgrad=False)

    # if num_classes == 2:
    #     model.add(tf.keras.layers.Dense(1))
    #     model.add(tf.keras.layers.Activation('sigmoid'))
    #     model.compile(optimizer=adam,
    #                   loss='binary_crossentropy',
    #                   metrics=['acc',
    #                            tf.keras.metrics.TruePositives(name='tp'),
    #                            tf.keras.metrics.FalsePositives(name='fp'),
    #                            tf.keras.metrics.TrueNegatives(name='tn'),
    #                            tf.keras.metrics.FalseNegatives(name='fn'),
    #                            tf.keras.metrics.BinaryAccuracy(),
    #                            tf.keras.metrics.BinaryCrossentropy(),
    #                            tf.keras.metrics.Precision(),
    #                            tf.keras.metrics.Recall(),
    #                            tf.keras.metrics.AUC()
    #                            ]
    #                   )
    
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                    metrics=['acc',
                            tf.keras.metrics.TruePositives(name='tp'),
                            tf.keras.metrics.FalsePositives(name='fp'),
                            tf.keras.metrics.TrueNegatives(name='tn'),
                            tf.keras.metrics.FalseNegatives(name='fn'),
                            tf.keras.metrics.CategoricalAccuracy(),
                            tf.keras.metrics.CategoricalCrossentropy(),
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            tf.keras.metrics.AUC()
                            ]
                    )
    model.summary(print_fn=logger.info)
    model.summary()

    save_best = path.join(best_folder, "DeepNano_{}.h5".format(run_name))

    check_pointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_best, verbose=1, save_best_only=True)
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    patience=10,
                                                    mode='auto', 
                                                    verbose=1)#adjust lr by the validation loss

    tf_board = tf.keras.callbacks.TensorBoard(log_dir=graph_log_dir,
                                              histogram_freq=0,
                                              write_graph=True,
                                              write_grads=False,
                                              write_images=True,
                                              embeddings_freq=0,
                                              embeddings_layer_names=None,
                                              embeddings_metadata=None)

    x_train, y_train, x_val, y_val = utils_tf.split_data(
        train_data, train_labels, per).values()

    logger.info("train on {} samples".format(x_train.__len__()))
    logger.info("valid on {} samples".format(x_val.__len__()))
    logger.info("test on {} samples".format(test_data.__len__()))
    class_weight=None
    if num_classes==2:
        total = train_labels.__len__()
        pos = np.count_nonzero(train_labels)
        neg = total-pos

        weight_for_0 = (1 / neg) * (total)/2.0
        weight_for_1 = (1 / pos) * (total)/2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
    # elif num_classes>2:
        
    y_train =  tf.keras.utils.to_categorical(
                        y_train, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(
                        y_val, num_classes=num_classes)
    test_labels = tf.keras.utils.to_categorical(
                            test_labels, num_classes=num_classes)
    history = model.fit(x_train,
                        y_train,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_data=(x_val,y_val),
                        callbacks=[tf_board, check_pointer, reduce_lr, early_stopper],
                        class_weight=class_weight,
                        verbose=2)

    results = model.evaluate(test_data, test_labels)
    logger.info("results")
    logger.info(results)
    return history, model, results


def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True,
                        help='npz filename with data')
    parser.add_argument('--train_file', required=True,
                        help='npz filename with data')
    parser.add_argument('--test_file', required=True,
                        help='npz filename with data')
    parser.add_argument('--env', required=True, help='npz filename with data')
    parser.add_argument('--dataset',  required=True, help='dataset to handle')
    parser.add_argument('--num_classes', default=3,
                        help='number of classes output')
    parser.add_argument('--gpu', type=utils.str2bool, nargs='?',
                        const=True,  default=True, help="flag for gpu training")
    argums = parser.parse_args()
    return argums


def main(name, input_path, train_path, test_path, param_path, gpu, num_classes):
    logger.setLevel(logging.INFO)
    if gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # tf.compat.v1.keras.backend.set_session(sess)
        K.set_session(sess)
    # else:
    #     config = tf.ConfigProto(device_count={'GPU': 0})
    #     sess = tf.Session(config=config)
    #     # tf.compat.v1.keras.backend.set_session(sess)
    #     K.set_session(sess)
    run_name, uid = utils.get_run_name(name)
    # create a file handler
    handler = logging.FileHandler(
        path.join(input_path, './logs/{}.log'.format(run_name)))
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.info('Start Training!')
    logger.info(run_name)
    logger.info('run arguments')
    logger.info(param_path)
    # I inversed order to notice
    # my_data = load_data(args.test, args.train)
    logger.info("train dataset:")
    logger.info(train_path)
    logger.info("test dataset:")
    logger.info(test_path)
    my_data = utils_tf.load_data(train_path, test_path)

    best_folder = path.join(input_path, 'best_models', uid)
    mkdir(best_folder)
    logger.info("RESULTS FOLDER:")
    logger.info(best_folder)
    graph_dir = path.join(input_path, 'graph', uid)
    mkdir(graph_dir)
    logger.info("TENSORBOARD MONITORING:")
    logger.info(graph_dir)

    history_ret, model_trained, res = train(
        my_data, best_folder, run_name, graph_dir, param_path, num_classes)

    loss_fn = path.join(best_folder, "loss_{}.png".format(run_name))
    acc_fn = path.join(best_folder, "accuracy_{}.png".format(run_name))
    utils_tf.plot_history(history_ret, loss_fn, acc_fn)

    utils.to_pickle(history_ret.history, path.join(best_folder, '{}_history.pickle'.format(run_name)))

    handler.flush()
    handler.close()
    logger.removeHandler(handler)
    shutil.move(path.join(input_path,
                          './logs/{}.log'.format(run_name)),
                path.join(input_path,
                          './logs/{}_l_{}_acc_{}.log'.format(run_name,
                                                             np.around(
                                                                 res[0], decimals=4),
                                                             np.around(res[1], decimals=4))))

    return res, run_name


if __name__ == "__main__":
    args = myargs()
    train_path = path.join(args.input_path, args.train_file)
    test_path = path.join(args.input_path, args.test_file)
    ROOT_DIR = path.dirname(path.abspath(__file__))
    param_path = path.join(
        ROOT_DIR, './params/param_{}.json'.format(args.dataset))
    _res, _run_name = main(args.env,
                           args.input_path,
                           train_path,
                           test_path,
                           param_path,
                           args.gpu,
                           int(args.num_classes))
