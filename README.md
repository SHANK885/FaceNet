# FaceNet
## Realtime recognition using FaceNet


### Platform Secification:

  * Ubuntu 18.04


### Requirements:
  
  * tensorflow==1.7
  * sklearn==1.17.2
  * Python==3.5.6
  * OpenCV==4.1.1
  * NumPy==1.14.2
  * SciPy==1.3.0

### Setup

  * Clone this [repository](https://github.com/gyrusai/FaceNet/archive/master.zip).
  * Download the [models.zip](https://drive.google.com/open?id=1LfsMvRRdiWvjgWS8Ufw_Q8RXmwrUcncj).
  * Downlaod the [age_gender.zip](https://drive.google.com/open?id=1aFQGU1FoBwW6qsdMvjFqdEsq7rboCo-r).
  * Unzip both **models.zip** and **age_gender.zip** inside the FaceNet directory.
  
### Enroll a new face using webcam?

  1. Go to the FaceNet/src directory.
  2. run "python enroll_face.py <name_of_new_member>
  3. Web cam will oen up with a window to capture face.
  4. Press s by selectiong the video window to capture the image.
  5. If you want to recapture the image:
        select the terminal window and enter "R" or "r" else enter "C" or "c".

  It will enroll the new face with the name provided in the command line.

  The captured image will be saved to:
        FaceNet/database/raw_images/<name_of_new_member> directory
  
  The cropped and aligned face will be saved to:
        FaceNet/database/cropped_faces/<name_of_new_member> directory
  
  The 512 D face embedding will be appended to:
        FaceNet/database/embeddings/face_embeddings.json


### Where the image is stored ?

  * The raw images of all the enrolled members is stored in:
    [FaceNet/database/raw_images/<name> directory](https://github.com/gyrusai/FaceNet/tree/master/database/raw_images)
  * The cropped and alligned faces of all te enrolled members is stored in:
    [FaceNet/database/cropped_faces/<name>](https://github.com/gyrusai/FaceNet/tree/master/database/cropped_faces)
  * The embeddings of all the enrolled faces is present in:
    [FaceNet/database/embeddings/face_embeddings.json](https://github.com/gyrusai/FaceNet/tree/master/database/embeddings)

### FaceNet Realtime: What is does?

Our facenet realtime is able to recognize the faces of all the members that is enrolled in the database. However, if a face is not enrolled it will make it as unknown.


### How to run realtime recognition.

  * Enroll the faces you want by following the above steps.
  * Go to the FaceNet/src directory.
  * run har_real_time_recognition_v2.py.
  * It will be able to recognize the faceq that are present in the database and will mark a face unknown if it is not             registered.

### How to recognize faces stored in the FaceNet/database/test_data directory ?

  1. Go to FaceNet/src directory.
  2. run recognize_face.py

  It will show up the actual and predicted name of the person
  present in the test_data directory.
