'''
Perform face detection in real-time using webcam
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import face
import sys
import argparse
import time


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for id_face in faces:
            face_bb = id_face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]),
                          (face_bb[2], face_bb[3]),
                          (0, 255, 0),
                          2)
            if id_face.name is not None:
                cv2.putText(frame,
                            id_face.name,
                            (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2,
                            lineType=2)
    cv2.putText(frame,
                str(frame_rate) + ' fps',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2,
                lineType=2)


def main(args):

    # number of frame after which to run face detection
    frame_interval = 3

    fps_display_interval = 5    # second
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    start_time = time.time()

    if args.debug:
        print("Debug Enabled")
        face.debug = True

    while True:
        # capture frame-by-frame
        iter_time = time.time()

        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # check current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
        # print("Frame Rate: ", frame_rate)
        add_overlays(frame, faces, frame_rate)

        frame_count += 1
        cv2.imshow('Video', frame)
        print("One iteration time: {} ms\n\n".format(1000 * (time.time()-iter_time)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # when everything is done
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
