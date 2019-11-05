# coding=utf-8
"""Face Detection and Recognition"""
import json
import cv2
import numpy as np
import tensorflow as tf
import facenet
import datetime


gpu_memory_fraction = 1.0

facenet_model_checkpoint = "../models/20180402-114759"
embedding_path = "../database/embeddings/face_embeddings.json"
face_detector_path = "../classifiers/haarcascade_frontalface_default.xml"
eye_detector_path = "../classifiers/haarcascade_eye.xml"
age_prot_path = "../age_gender/deploy_age.prototxt"
age_caffe_path = "../age_gender/age_net.caffemodel"
gender_prot_path = "../age_gender/deploy_gender.prototxt"
gender_caffe_path = "../age_gender/gender_net.caffemodel"


debug = False


class Face:
    def __init__(self):
        self.name = None
        self.age = None
        self.gender = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.timestamp = None


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
        faces = self.detect.find_faces(image)
        faces = self.detect.predictAge(faces)
        faces = self.detect.predictGender(faces)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)
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

            if min_dist > 1.06:
                identity = "Unknown"
            face.timestamp = datetime.datetime.now()
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
    factor = 1.1  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=16):
        self.face_detector = cv2.CascadeClassifier(face_detector_path)
        self.eye_detector = cv2.CascadeClassifier(eye_detector_path)
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.age_net = cv2.dnn.readNetFromCaffe(age_prot_path, age_caffe_path)
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_prot_path, gender_caffe_path)
        self.age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
        self.gender_list = ['Male', 'Female']
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def get_boundding_box(self, image, minsize, factor):
        bounding_box = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray,
                                                    scaleFactor=factor,
                                                    minNeighbors=5,
                                                    minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_frame = image[y:y+h, x:x+w]
            eyes = self.eye_detector.detectMultiScale(roi_frame)
            if len(eyes) > 0:
                bounding_box.append([x, y, w, h])
        return bounding_box

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

            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[0] + bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[1] + bb[3] + self.face_crop_margin / 2, img_size[0])

            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]

            face.image = cv2.resize(cropped,
                                    (self.face_crop_size, self.face_crop_size),
                                    interpolation=cv2.INTER_LINEAR)

            faces.append(face)

        return faces

    def predictAge(self, faces):
        for face in faces:
            blob = cv2.dnn.blobFromImage(face.image,
                                         1,
                                         (227, 227),
                                         self.MODEL_MEAN_VALUES,
                                         swapRB=False)

            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            face.gender = gender
        return faces

    def predictGender(self, faces):
        for face in faces:
            blob = cv2.dnn.blobFromImage(face.image,
                                         1,
                                         (227, 227),
                                         self.MODEL_MEAN_VALUES,
                                         swapRB=False)
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            # print("age_preds", age_preds)
            age = self.age_list[age_preds[0].argmax()]
            face.age = age
        return faces
