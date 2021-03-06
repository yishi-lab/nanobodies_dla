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
import tensorflow as tf
import logging
logger = logging.getLogger('learning_log.model_build')


class DeepNano:

    @staticmethod
    def baseline(dim, l, f_num, ks, dp_out=0.3, stride=1, reg_l1=0.01, out=3):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(dim, l)))
        model.add(tf.keras.layers.Dense(16, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l1(reg_l1)))
        model.add(tf.keras.layers.Dropout(dp_out))

        model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(out, activation='softmax'))

        return model
    
    @staticmethod
    def build(dim, l, f_num, ks, dp_out=0.3, stride=1, reg_l1=0.01, out=3):
        logger.info('DeepNano.build()')
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Convolution1D(input_shape=(dim, l),
                                                filters=f_num,
                                                kernel_size=ks,
                                                strides=stride,
                                                padding='valid',
                                                kernel_regularizer=tf.keras.regularizers.l1(l=reg_l1)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=4,
                                                strides=4,
                                                padding='same'))

        model.add(tf.keras.layers.Dropout(dp_out))
        model.add(tf.keras.layers.Flatten())

        # model.add(tf.keras.layers.Dense(out))
        # model.add(tf.keras.layers.Activation('softmax'))
        return model