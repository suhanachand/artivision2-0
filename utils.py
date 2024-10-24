import dlib
from glob import glob 
import cv2
import numpy as np
import os
import pytesseract

# load the face detector, landmark predictor, and face recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

age_model = cv2.dnn.readNetFromCaffe(
             "models/age_deploy.prototxt",
             "models/age_net.caffemodel")

gender_model = cv2.dnn.readNetFromCaffe(
             "models/gender_deploy.prototxt",
             "models/gender_net.caffemodel")


# change this to include other image formats you want to support (e.g. .webp)
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def get_age_gender (face_image):
    gender_labels = ["Male", "Female"]
    #gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_img_copy = cv2.resize(face_image, (256,256))
    print(face_img_copy.shape)
    cv2.imshow("face", face_img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blob = cv2.dnn.blobFromImage(face_img_copy, 1.0, (227,227),
                        (78.4263377603, 87.7689143744, 114.895847746),
                        swapRB=False)

    age_model.setInput(blob)
    gender_model.setInput(blob)
    
    age_pred = age_model.forward()
    gender_pred = gender_model.forward()

    print("Age:",age_pred[0])
    print("gender:", gender_pred)

    return

def get_image_paths(root_dir, class_names):
    """ grab the paths to the images in our dataset"""
    image_paths = []

        # loop over the class names
    for class_name in class_names:
        # grab the paths to the files in the current class directory
        class_dir = os.path.sep.join([root_dir, class_name])
        class_file_paths = glob(os.path.sep.join([class_dir, '*.*']))

        # loop over the file paths in the current class directory
        for file_path in class_file_paths:
            # extract the file extension of the current file
            ext = os.path.splitext(file_path)[1]

            # if the file extension is not in the valid extensions list, ignore the file
            if ext.lower() not in VALID_EXTENSIONS:
                print("Skipping file: {}".format(file_path))
                continue

            # add the path to the current image to the list of image paths
            image_paths.append(file_path)

    return image_paths

def face_rects(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = face_detector(gray, 1)
    # return the bounding boxes
    return rects

def face_landmarks(image):
    return [shape_predictor(image, face_rect) for face_rect in face_rects(image)]

def face_encodings(image):
    #print("utils--face_encodings:",image.shape)
    # compute the facial embeddings for each face 
    # in the input image. the `compute_face_descriptor` 
    # function returns a 128-d vector that describes the face in an image
    return [np.array(face_encoder.compute_face_descriptor(image, face_landmark)) 
            for face_landmark in face_landmarks(image)]

def nb_of_matches(known_encodings, unknown_encoding):
    # compute the Euclidean distance between the current face encoding 
    # and all the face encodings in the database
    #print(known_encodings - unknown_encoding)
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    # keep only the distances that are less than the threshold
    small_distances = distances <= 0.6
    # return the number of matches
    return sum(small_distances)


if __name__ == "__main__":
     image = cv2.imread("girl.jpg")
     print(image.shape)
     get_age_gender(image)
