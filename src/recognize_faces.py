''' recognize faces in the provided directory
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import json
import sys
import os
import copy
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face


def main(args):
    images, image_paths = load_and_align_data(args.image_files,
                                                    args.image_size,
                                                    args.margin,
                                                    args.gpu_memory_fraction)

    print('image files: ', args.image_files)
    print('image size: ', args.image_size)
    print('image margin: ', args.margin)
    print('model: ', args.model)
    print('gpu memory fraction: ', args.gpu_memory_fraction)

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
        test_emb = sess.run(embeddings, feed_dict=feed_dict)

    print('embeddings shape: ', test_emb.shape)
    print('Embeddings type: ', type(test_emb))
    # print("Test Embeddings: \n", test_emb)

    # test_emb is a class instance of np.ndarray

    nrof_images = len(image_paths)
    print('number of images: ', nrof_images)

    print("Images:")
    for i in range(nrof_images):
        print('%ld: %s' % (i, image_paths[i]))
    print('')

    with open('../database/embeddings/face_embeddings.json', 'r') as f:
        base_emb = json.load(f)  # covert ot numpy array , it is a list
    # print("Base Embeddings: \n", base_emb)
    # print("Base embedding shape: ", base_emb.shape)

    for i, embedding in enumerate(test_emb):

        original_name = image_paths[i].split('/')[-2]

        identity, similarity = get_identity(embedding, base_emb, original_name)
        print("original: {}   predicted: {}".format(original_name, identity))
        print("Similarity: ", similarity)
        print("")
        '''
        min_dist = 100
        for (name, db_emb) in base_emb.items():
            dist = np.linalg.norm(embedding - np.array(db_emb))

            if dist < min_dist:
                min_dist = dist
                identity = name

        print("original: {}   predicted: {}".format(original_name, identity))
        '''


def get_identity(embedding, base_emb, original_name):

    min_dist = 100
    for (name, db_emb) in base_emb.items():
        dist = np.linalg.norm(embedding - np.array(db_emb))

        if dist < min_dist:
            min_dist = dist
            identity = name

        print("L2 Distance between {} and {} => {}".format(original_name, name, dist))

    return identity, min_dist


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20                    # minimum size of face
    threshold = [0.6, 0.7, 0.7]     # three steps's threshold
    factor = 0.709                  # scale factor

    print('image paths: ', image_paths)
    image_paths = os.path.expanduser(image_paths)
    # print('image path expanduser: ', image_paths)
    image_classes = [path for path in os.listdir(image_paths) \
                     if os.path.isdir(os.path.join(image_paths, path))]
    print('\nImage classes:\n')
    print("\n".join(image_classes))

    final_image_paths = []
    nrof_image_classes = len(image_classes)
    for i in range(nrof_image_classes):
        class_name = image_classes[i]
        facedir = os.path.join(image_paths, class_name)
        print('face dir: ', facedir)
        image_path_list = get_image_paths(facedir)
        for item in image_path_list:
            final_image_paths.append(item)

    print('\nCreating networks and loading parameters\n')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    print("Number of final image paths: ", len(final_image_paths))
    # print('final image paths:', end='\n')
    tmp_image_paths = copy.copy(final_image_paths)
    img_list = []
    print(tmp_image_paths)

    for image in tmp_image_paths:
        print(image)
        img = cv2.imread(os.path.expanduser(image))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img,
                                                          minsize,
                                                          pnet,
                                                          rnet,
                                                          onet,
                                                          threshold,
                                                          factor)
        if len(bounding_boxes) < 1:
            final_image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = cv2.resize(cropped,
                             (image_size, image_size),
                             interpolation=cv2.INTER_LINEAR)
        prewhitened = facenet.prewhiten(aligned)
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
                        default='../database/test_data/')
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
