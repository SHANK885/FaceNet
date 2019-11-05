'''
Creates 128 D face embeddings for all the images
present in database/cropped_faces folder
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import cv2
import tensorflow as tf
import numpy as np
import facenet
import argparse
import copy
import sys
import os
import json


def main(args):

    images, image_paths = load_images(args.image_files,
                                      args.image_size,
                                      args.margin,
                                      args.gpu_memory_fraction)

    # print('image files: ', args.image_files)
    print('image size: ', args.image_size)
    print('image margin: ', args.margin)
    # print('model: ', args.model)
    # print('gpu memory fraction: ', args.gpu_memory_fraction)

    with tf.Session() as sess:

        # load model
        facenet.load_model(args.model)

        # get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Run forward pass to compute embeddings
        feed_dict = {images_placeholder: images,
                     phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)

    print('embeddings shape: ', emb.shape)
    # print('Embeddings type: ', type(emb))

    nrof_images = len(image_paths)
    print('number of images: ', nrof_images)

    print("Images:")
    for i in range(nrof_images):
        print('%ld: %s' % (i, image_paths[i]))
    print('')

    # storing embeddings to a dictionary

    embeddings_dict = {}
    for i, filename in enumerate(image_paths):
        # print('{}: {}'.format(i, filename))
        # print(filename.split('/')[-2])
        person = filename.split('/')[-2]
        embeddings_dict[person] = emb[i, :].tolist()
        # print(type(embeddings_dict[person]), len(embeddings_dict))

    # saving embeddings to json file
    with open('../database/embeddings/face_embeddings.json', 'w+') as f:
        json.dump(embeddings_dict, f)
        print('Embeddings written to json file')


def load_images(image_paths, image_size, margin, gpu_memory_fraction):

    # print('image paths: ', image_paths)
    image_paths = os.path.expanduser(image_paths)
    print('image path expanduser: ', image_paths)
    image_classes = [path for path in os.listdir(image_paths) \
                     if os.path.isdir(os.path.join(image_paths, path))]
    print('Image classes:\n', image_classes, end='\n')

    final_image_paths = []
    nrof_image_classes = len(image_classes)
    for i in range(nrof_image_classes):
        class_name = image_classes[i]
        facedir = os.path.join(image_paths, class_name)
        print('face dir: ', facedir)
        image_path_list = get_image_paths(facedir)
        for item in image_path_list:
            final_image_paths.append(item)

    # print("Number of final image paths: ", len(final_image_paths))
    # print('final image paths:', end='\n')
    tmp_image_paths = copy.copy(final_image_paths)
    img_list = []
    print(tmp_image_paths)

    for image in tmp_image_paths:
        print(image)
        # write code to make sure face is present in the image from comapre.py
        img = cv2.imread(os.path.expanduser(image))
        resized = cv2.resize(img,
                             (image_size, image_size),
                             interpolation=cv2.INTER_LINEAR)
        prewhitened = facenet.prewhiten(resized)
        img_list.append(prewhitened)
    images = np.stack(img_list)

    return images, final_image_paths


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        help='could be a directory containing meta/ckpt file',
                        default='../models/20180402-114759/')
    parser.add_argument('--image_files',
                        type=str,
                        help='images to create embeddings',
                        default='../database/cropped_faces/')
    parser.add_argument('--image_size',
                        type=int,
                        help='image size (height, width) in pixels.',
                        default=160)
    parser.add_argument('--margin',
                        type=int,
                        help='margin for crop around the bounding box',
                        default=44)
    parser.add_argument('--gpu_memory_fraction',
                        type=float,
                        help='upper bound on the GPU memory to be used',
                        default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
