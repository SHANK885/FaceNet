'''
Perform face detection in real-time using webcam
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import numpy as np
import os
import cv2
import har_face_v2
import sys
import argparse
import time
import copy
import csv


def add_overlays(frame, faces, frame_rate):

    if faces is not None:
        for id_face in faces:

            name_gender = '%s %s' % (id_face.name, id_face.gender)
            age = 'Age: %s Yr.' % (id_face.age)
            face_bb = id_face.bounding_box.astype(int)

            if id_face.name == 'Unknown':
                box_color = (0, 0, 255)
                text_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)
                text_color = (255, 10, 0)

            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]),
                          (face_bb[2], face_bb[3]),
                          box_color,
                          2)

            labelSize_n, baseLine_n = cv2.getTextSize(name_gender,
                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.7,
                                                      2)
            label_ymin_n = max(face_bb[1], labelSize_n[1] + 10)

            cv2.rectangle(frame,
                          (face_bb[0], label_ymin_n-labelSize_n[1]-10),
                          (face_bb[0]+labelSize_n[0], label_ymin_n+baseLine_n-10),
                          (255, 255, 255),
                          cv2.FILLED)

            labelSize_a, baseLine_a = cv2.getTextSize(age,
                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.7,
                                                      2)
            label_ymin_a = max(face_bb[3], labelSize_a[1] - 10)

            cv2.rectangle(frame,
                          (face_bb[0], label_ymin_a-labelSize_a[1]-10),
                          (face_bb[0]+labelSize_a[0], label_ymin_a+baseLine_a-10),
                          (255, 255, 255),
                          cv2.FILLED)

            if id_face.name is not None:
                cv2.putText(frame,
                            name_gender,
                            (face_bb[0], label_ymin_n-7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.70,
                            text_color,
                            thickness=2,
                            lineType=2)
                cv2.putText(frame,
                            age,
                            (face_bb[0], label_ymin_a-7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.70,
                            text_color,
                            thickness=2,
                            lineType=2)

    cv2.putText(frame,
                str(frame_rate) + ' fps',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                thickness=2,
                lineType=2)
    return frame


def main(args):
    # number of frame after which to run face detection
    frame_interval = 3

    fps_display_interval = 5    # second
    frame_rate = 0
    frame_count = 0
    faces = []
    face_data = []
    prev_faces = []
    last_faces = []

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    face_recognition = har_face_v2.Recognition()

    if args.debug:
        # print("Debug Enabled")
        har_face_v2.debug = True

    if not os.path.exists('../output/'):
        os.makedirs('../output/')

    with open("../output/face_data.csv", 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(['Time', 'Identified_Person', 'Age', 'Gender'])

    start_time = time.time()
    start_time_data = time.time()

    while True:
        # capture frame
        if video_capture.isOpened():
            ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            if len(faces) > 0:
                last_faces = copy.deepcopy(faces)
            faces = face_recognition.identify(frame, last_faces)

            # check current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
        if len(faces) > 0:
            frame = add_overlays(frame, faces, frame_rate)
        frame_count += 1
        cv2.imshow('Realtime Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(prev_faces) == 0:
            prev_faces = copy.deepcopy(faces)
            for face in faces:
                face_data.append(face)
            continue

        face_data, prev_faces = get_detected_faces(faces,
                                                   prev_faces,
                                                   face_data)

        if (time.time() - start_time_data) > 900:
            with open("../output/face_data.csv", 'a', newline='') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                for face in face_data:
                    ti = '{:%Y-%m-%d %H:%M:%S}'.format(face.timestamp)
                    face_info = [ti, face.name, face.age, face.gender]
                    wr.writerow(face_info)
                face_data = []
                start_time_data = time.time()

    # when everything is done
    video_capture.release()
    cv2.destroyAllWindows()

    with open("../output/face_data.csv", 'a', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        for face in face_data:
            ti = '{:%Y-%m-%d %H:%M:%S}'.format(face.timestamp)
            face_info = [ti, face.name, face.age, face.gender]
            wr.writerow(face_info)


def get_detected_faces(faces, prev_faces, face_data):
    if len(prev_faces) > 0 and len(faces) > 0:
        for face in faces:
            for prev_face in prev_faces:
                time_delta = face.timestamp - prev_face.timestamp
                if face.name == prev_face.name and face.name == 'Unknown':
                    l2_dist = np.linalg.norm(face.embedding - prev_face.embedding)
                    # face_centroid = bb_centroid(face)
                    # prev_face_centroid = bb_centroid(prev_face)
                    centroid_dist = dist_centroid(bb_centroid(face),
                                                  bb_centroid(prev_face))
                    threshold = get_threshold(prev_face)

                    if l2_dist < 1.10 and centroid_dist < threshold:
                        if (time_delta.total_seconds() > 60):
                            face_data.append(face)
                        prev_face.timestamp = datetime.datetime.now()

                elif face.name == prev_face.name and face.name != 'Unknown':
                    if (time_delta.total_seconds() > 60):
                        face_data.append(face)
                    prev_face.timestamp = datetime.datetime.now()

            if face.name not in [f.name for f in prev_faces]:
                face_data.append(face)
                prev_faces.append(face)

    return face_data, prev_faces


def bb_centroid(face):
    x = face.bounding_box[0] + face.bounding_box[2]//2
    y = face.bounding_box[1] + face.bounding_box[3]//3
    return (x, y)


def dist_centroid(c1, c2):
    x1, y1 = c1[0], c1[1]
    x2, y2 = c2[0], c2[1]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def get_threshold(t_face):
    w = t_face.bounding_box[2]
    h = t_face.bounding_box[3]
    return 0.75 * np.sqrt((w/2)**2 + (h/2)**2)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
