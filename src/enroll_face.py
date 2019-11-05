
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import align.detect_face
import json
import cv2
import sys
import os
import facenet
import argparse


gpu_memory_fraction = 1.0


def main(args):

    raw_path = "../database/raw_images/"
    cropped_path = "../database/cropped_faces/"
    embedding_path = "../database/embeddings/face_embeddings.json"
    facenet_model_checkpoint = "../models/20180402-114759"
    face_detector_path = "../classifiers/haarcascade_frontalface_default.xml"

    # print(raw_path)
    raw_path = os.path.join(raw_path, args.name)
    aligned_path = os.path.join(cropped_path, args.name)
    aligned_path = os.path.join(cropped_path, args.name)
    raw_img_path = os.path.join(raw_path, args.name + ".png")
    aligned_image_path = os.path.join(aligned_path, args.name + ".png")

    video_capture = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(face_detector_path)

    print("*****Initializing face enrollment*****\n")

    while True:
        while True:
            if video_capture.isOpened():
                ret, frame = video_capture.read()

            raw_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray,
                                                   scaleFactor=1.1,
                                                   minNeighbors=3,
                                                   minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print("Face Captured for: {}".format(args.name))
                break

        # cv2.imshow("Captured Frame", frame)

        print("Press 'C' to confirm this image")
        print("Press 'R' to retake the picture")

        response = input("\nEnter Your Response: ")

        if response == "C" or response == "c":
            print("\nImage finalized\n")
            video_capture.release()
            cv2.destroyAllWindows()
            break
        if response == "R" or response == "r":
            cv2.destroyAllWindows()
            continue

    # video_capture.release()
    # cv2.destroyAllWindows()

    # detect,align and get cropped face
    detect = Detection()
    aligned_face = detect.find_faces(raw_frame)

    # encode face
    encoder = Encoder(facenet_model_checkpoint)
    embedding = encoder.generate_embedding(aligned_face)
    embedding = embedding.tolist()
    # print("Embedding: ", embedding)
    print("Embedding Shape: ", len(embedding))

    if os.path.exists(raw_path):
        print("Member with name: {} already exists!!".format(args.name))
        print("Press 'C' to overwrite or 'R' to return")
        val = input("Enter response:")
        if val == 'r' or val == 'R':
            return
        elif val == 'c' or val == 'C':
            cv2.imwrite(raw_img_path, raw_frame)

    else:
        os.makedirs(raw_path)
        cv2.imwrite(raw_img_path, raw_frame)

    if not os.path.exists(aligned_path):
        os.makedirs(aligned_path)

    cv2.imwrite(aligned_image_path, aligned_face)

    try:
        with open(embedding_path, 'r') as rf:
            base_emb = json.load(rf)
    except IOError:
        print("Embeddibg file empty!! Creating a new embedding file")
        with open(embedding_path, 'w+') as rf:
            base_emb = {}
    with open(embedding_path, 'w') as wf:
        base_emb[args.name] = embedding
        json.dump(base_emb, wf)
        print("embedding written")

    print("face enrolled with name => {}".format(args.name))


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):

        bounding_boxes, _ = align.detect_face.detect_face(image,
                                                          self.minsize,
                                                          self.pnet,
                                                          self.rnet,
                                                          self.onet,
                                                          self.threshold,
                                                          self.factor)

        area = 0
        for i, bb in enumerate(bounding_boxes):

            bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            face_area = abs(bounding_box[3] - bounding_box[1]) * \
                        abs(bounding_box[2] - bounding_box[0])

            if face_area > area:
                cropped = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
                image = cv2.resize(cropped,
                                   (self.face_crop_size, self.face_crop_size),
                                   interpolation=cv2.INTER_LINEAR)

        return image


class Encoder:
    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(ckpt)

    def generate_embedding(self, image):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}

        # return self.sess.run(embeddings, feed_dict=feed_dict)[0]

        with self.sess as ses:
            return ses.run(embeddings, feed_dict=feed_dict)[0]


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('name',
                        type=str,
                        help='Add the name of member to be added.')
    parser.add_argument('--image_size',
                        type=int,
                        help='Image size (height, width), in pixels.',
                        default=182)
    parser.add_argument('--margin',
                        type=int,
                        help='margin for the crop around the bounding box.',
                        default=44)
    parser.add_argument('--gpu_memory_fraction',
                        type=float,
                        help='upper boun of amount of gpu memory to allocate.',
                        default=1.0)
    parser.add_argument('--detect_multiple_faces',
                        type=bool,
                        help='Detect and allign multiple faces per image.',
                        default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
