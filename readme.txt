# readme file for FaceNet

Platform Secification:
  Ubuntu 18.04

Requirements:
  Python==3.5.6 or above
  OpenCV==4.1.1
  NumPy==1.14.2
  SciPy==1.1.0


How to enroll a new face using web cam ?

  1. Go to the FaceNet/src directory.
  2. run "python enroll_face.py <name_of_new_member>
  3. Web cam will oen up with a window to capture face.
  4. Press s by selectiong the video window to capture the image.
  5. If you want to recapture the image:
        select the terminal window and enter "R" or "r".
    else
        enter "C" or "c"

  It will enroll the new face with the name provided in the command line.

  The captured image will be saved to:
        FaceNet/database/raw_images/<name_of_new_member> directory
  The cropped and aligned face will be saved to:
        FaceNet/database/cropped_faces/<name_of_new_member> directory
  The 512 D face embedding will be appended to:
        FaceNet/database/embeddings/face_embeddings.json

How to enroll a face manually ?

  1. Go to FaceNet/database/raw_images directory.
  2. Create a folder with the name of new user.
  3. Add 1 image of the person in the created directory.
  3. Go to FaceNet/src/ directory.
  4. Run python get_aligned_faces.py
  5. Run create face embeddings.

  It will enroll the new face added manually in the database and
  will add create the embeddings.

Where the image is stored ?

  The raw images of all the enrolled members is stored in:
    FaceNet/database/raw_images/<name> directory
  The cropped and alligned faces of all te enrolled members is stored in:
    FaceNet/database/cropped_faces/<name>
  The embeddings of all the enrolled faces is present in:
    FaceNet/database/embeddings/face_embeddings.json

FaceNet Realtime: What is does?

  Out facenet realtime is able to recognize the faces of all the members
  that is enrolled in the database. However, the feature for recognizing
  some faces as unknown which is not in the database is still to be added.

Unknown faces?

  For now as we have not added the feature to classify never seen faces as
  unknown. It will classify it to the most similar face in the database and
  give that name is predicted.

How to run realtime recognition.

  After enrolling the face. You can go to the FaceNet/src directory and Run
      python real_time_recognition.py

  It will be able to recognize the faceq that are present in the database.

How to recognize faces stored in the FaceNet/database/test_data directory ?

  1. Go to FaceNet/src directory.
  2. run python recognize_face.py

  It will show up the actual and predicted name of the person
  present in the test_data directory.
