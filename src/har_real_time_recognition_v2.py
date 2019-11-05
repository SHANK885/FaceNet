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
            face_bb = id_face.bounding_box.astype(int)
            if id_face.name == 'Unknown':
                box_color = (0, 0, 255)
                text_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)
                text_color = (255, 0, 0)

            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]),
                          (face_bb[2], face_bb[3]),
                          box_color,
                          2)
            if id_face.name is not None:
                cv2.putText(frame,
                            id_face.name+" "+id_face.gender,
                            (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            text_color,
                            thickness=2,
                            lineType=2)
                cv2.putText(frame,
                            "Age: " + id_face.age + " Yrs",
                            (face_bb[0], face_bb[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
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


def main(args):

    # number of frame after which to run face detection
    frame_interval = 3

    fps_display_interval = 5    # second
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = har_face_v2.Recognition()

    if args.debug:
        # print("Debug Enabled")
        har_face_v2.debug = True

    faces = []
    face_data = []

    if not os.path.exists('../output/'):
        os.makedirs('../output/')

    with open("../output/face_data.csv", 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(['Time', 'Identified_Person', 'Age', 'Gender'])

    prev_faces = []
    last_faces = []

    start_time = time.time()
    start_time_data = time.time()

    while True:
        # capture frame-by-frame
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

        if len(prev_faces) == 0:
            prev_faces = copy.deepcopy(faces)
            for face in faces:
                face_data.append(face)
            continue

        if len(prev_faces) > 0 and len(faces) > 0:
            for face in faces:
                for prev_face in prev_faces:
                    if face.name == prev_face.name and face.name == 'Unknown':
                        # if np.linalg.norm(face.embedding - prev_face.embedding) < 1.10
                        if ((face.timestamp - prev_face.timestamp).total_seconds() > 15):
                            face_data.append(face)
                            prev_face.timestamp = datetime.datetime.now()

                    elif face.name == prev_face.name and face.name != 'Unknown':
                        if ((face.timestamp - prev_face.timestamp).total_seconds() > 60):
                            face_data.append(face)
                            prev_face.timestamp = datetime.datetime.now()

                if face.name not in [f.name for f in prev_faces]:
                    face_data.append(face)
                    prev_faces.append(face)

        add_overlays(frame, faces, frame_rate)
        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
