'''
Using our oun dataset to train a classifier that recognizes people
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import agrparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC


def main(args):

    with tf.Graph().as_default():
        with tf.Session() as sess:

            np.random.seed(seed = args.seed)

            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset((dataset_tmp, args.min_nrof_images_per_class))
                if args.mode == 'TRAIN':
                    dataset = train_set
                elif args.mode == 'CLASSIFY':
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths))
