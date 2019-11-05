# coding=utf-8
"""Face Detection and Recognition"""
import json
import time
import cv2
import numpy as np
import tensorflow as tf
import facenet

gpu_memory_fraction = 1.0

facenet_model_checkpoint = "../models/20180402-114759"
embedding_path = "../database/embeddings/face_embeddings.json"
face_detector_path = "../classifiers/haarcascade_frontalface_default.xml"
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):

        det_time = time.time()
        faces = self.detect.find_faces(image)
        enc_time = time.time()
        print("Detection time: {} ms".format(1000 * (enc_time-det_time)))

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)
        print("Encoding and recognition time: {} ms".format(1000 * (time.time()- enc_time)))
        return faces


class Identifier:
    def __init__(self):
        with open(embedding_path, 'r') as infile:
            self.base_emb = json.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            min_dist = 100
            for (name, db_emb) in self.base_emb.items():
                dist = np.linalg.norm(face.embedding - np.array(db_emb))

                if dist < min_dist:
                    min_dist = dist
                    identity = name

            if min_dist > 1.07:
                identity = "Unknown"
            print("Identity: {} L2 Distance: {}".format(identity, min_dist))

            return identity


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 40   # minimum size of face
    factor = 1.2  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=16):
        self.face_detector = cv2.CascadeClassifier("../classifiers/haarcascade_frontalface_default.xml")
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def get_boundding_box(self, image, minsize, factor):
        # bounding_box = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray,
                                                    scaleFactor=factor,
                                                    minNeighbors=5,
                                                    minSize=(40, 40))
        print("Faces", faces)
        return faces

    def find_faces(self, image):
        faces = []

        bounding_boxes = self.get_boundding_box(image,
                                                self.minsize,
                                                self.factor)

        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            # print('image size: ', img_size)
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[0] + bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[1] + bb[3] + self.face_crop_margin / 2, img_size[0])
            # print("bb0:{}, bb1:{}, bb2:{}, bb:{}".format(face.bounding_box[0], face.bounding_box[1], face.bounding_box[2], face.bounding_box[3]))
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]

            # print('cropped image size: ', np.asarray(cropped.shape)[0:2])
            face.image = cv2.resize(cropped,
                                    (self.face_crop_size, self.face_crop_size),
                                    interpolation=cv2.INTER_LINEAR)

            faces.append(face)

        return faces
