import platform
import sys
import os
plat = platform.release()


try:
    import math
    import glob
    import os
    from sklearn import neighbors
    import os.path
    import pickle
    from PIL import Image, ImageDraw, ImageFont
    import shutil
    import face_recognition
    import PIL
    from face_recognition.face_recognition_cli import image_files_in_folder
    import warnings
    import cv2
    import random
    import datetime
    import time
    import tkinter as tk
    from tkinter import ttk
    import numpy as np
    import imutils
    from colorama import init, Fore
    from tkinter import *
    import getpass
    import requests
    import copy
    import argparse
    import cv2 as cv
    from tkinter import messagebox
    from clint.textui import progress
    from zipfile import ZipFile
    from requests.structures import CaseInsensitiveDict
    import subprocess
    from github import Auth
    from github import Github
except ModuleNotFoundError:
    os.system("pip install -r data/Libs/Libs.txt")
import math
import glob
import os
from sklearn import neighbors
import os.path
import pickle
from PIL import Image, ImageDraw, ImageFont
import shutil
import face_recognition
import PIL
from face_recognition.face_recognition_cli import image_files_in_folder
import warnings
import cv2
import random
import datetime
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import imutils
from colorama import init, Fore
from tkinter import *
import getpass
import requests
import copy
import argparse
import cv2 as cv
from tkinter import messagebox
from clint.textui import progress
from zipfile import ZipFile
from requests.structures import CaseInsensitiveDict
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
init()

def ccam():
    global camcapi
    cm = input("choose your camera mode(ip/internal) ==>> ")
    if cm == "ip":
        camcapi = input("write your ip ==>> ")
        camcapi = str(camcapi)
    elif cm == "internal":
        camcapi = input("write your camera number(write 0 if you don't know what to choose, 0 is default) ==>> ")
        camcapi = int(camcapi)
ccam()
def recognize(path):
    warnings.filterwarnings("ignore")
    def getFaceBox(net, frame, conf_threshold=0.7):
        warnings.filterwarnings("ignore")
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes
    faceProto = "data/ag/ag_models/opencv_face_detector.pbtxt"
    faceModel = "data/ag/ag_models/opencv_face_detector_uint8.pb"
    ageProto = "data/ag/ag_models/age_deploy.prototxt"
    ageModel = "data/ag/ag_models/age_net.caffemodel"
    genderProto = "data/ag/ag_models/gender_deploy.prototxt"
    genderModel = "data/ag/ag_models/gender_net.caffemodel"
    warnings.filterwarnings("ignore")
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    warnings.filterwarnings("ignore")
    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    warnings.filterwarnings("ignore")
    # Open a video file or an image file or a camera stream
    warnings.filterwarnings("ignore")
    cap = cv.VideoCapture(path)
    warnings.filterwarnings("ignore")
    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            t.insert("1.0", "No face Detected, Please Try Again")
            warnings.filterwarnings("ignore")
        for bbox in bboxes:
            warnings.filterwarnings("ignore")
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            warnings.filterwarnings("ignore")
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            warnings.filterwarnings("ignore")
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            warnings.filterwarnings("ignore")
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            t.insert("1.0", f"Gender : {gender}")
            t.insert("1.0", f"Age : {age}")
            t.insert("1.0", f"--------------------------------------------------------------------------------------")
            label = "{},{}".format(gender, age)
            for bbox in bboxes:
                # print(bbox)
                face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
def fr():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    root2.withdraw()
    warnings.filterwarnings("ignore")
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}
    warnings.filterwarnings("ignore")
    def submit():
        root3.withdraw()
        def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
            X = []
            y = []
            # Loop through each person in the training set
            for class_dir in os.listdir(train_dir):
                if not os.path.isdir(os.path.join(train_dir, class_dir)):
                    continue
                # Loop through each training image for the current person
                for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                    image = face_recognition.load_image_file(img_path)
                    face_bounding_boxes = face_recognition.face_locations(image)
                    if len(face_bounding_boxes) != 1:
                        # If there are no people (or too many people) in a training image, skip the image.
                        if verbose:
                            print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                    else:
                        # Add face encoding for current image to the training set
                        X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                        y.append(class_dir)
            # Determine how many neighbors to use for weighting in the KNN classifier
            if n_neighbors is None:
                n_neighbors = int(round(math.sqrt(len(X))))
                if verbose:
                    print("Chose n_neighbors automatically:", n_neighbors)
            # Create and train the KNN classifier
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
            knn_clf.fit(X, y)
            # Save the trained KNN classifier
            if model_save_path is not None:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(knn_clf, f)
            return knn_clf
        def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
            if knn_clf is None and model_path is None:
                raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
            # Load a trained KNN model (if one was passed in)
            if knn_clf is None:
                with open(model_path, 'rb') as f:
                    knn_clf = pickle.load(f)
            global X_face_locations
            X_face_locations = face_recognition.face_locations(X_frame)
            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []
            # Find encodings for faces in the test image
            faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
            # Predict classes and remove classifications that aren't within the threshold
            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        def show_prediction_labels_on_image(frame, predictions):
            from PIL import Image, ImageDraw
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)
            for name, (top, right, bottom, left) in predictions:
                # enlarge the predictions for the full sized image.
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                key = cv2.waitKey(1)
                if ord('s') == key:
                    cropi = frame[top:bottom, left:right]
                    pi = Image.fromarray(cropi)
                    pi.show()
                    pi.save("database/saved faces/unknown face {}.jpg".format(random.random()))
                    n = input(f"{Fore.GREEN}Enter face name ==>> ")
                    print("face is encoded and you can see the face name when you press r on your keyboard[face name: {}]".format(n))
                    os.mkdir("database/recognized_faces/train/{}".format(n))
                    pi.save("database/recognized_faces/train/{}/{} {}.jpg".format(n, n, random.random()))
                elif ord('r') == key:
                    print(f"{Fore.YELLOW} Getting requirements ready to retrain all models, this may take some minutes if you have 99+ trained faces pictures")
                    classifier = train("database/recognized_faces/train", model_save_path="data/prx_models/trained_faces.prx", n_neighbors=2)
                    print(f"{Fore.GREEN}All faces got Retrained!")
                elif ord('a') == key:
                    mod = os.path.isfile("data/ag/ag_models/gender_net.caffemodel")
                    mod2 = os.path.isfile("data/ag/ag_models/age_net.caffemodel")
                    if mod == False and mod2 == False:
                        moi = messagebox.askyesno("Age & Gender Recognition", "you have not installed age & gender recognition models, do you want to download them right now?")
                        if moi == True:
                            url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel"
                            r = requests.get(url, stream=True)
                            path = 'data/ag/ag_models/gender_net.caffemodel'
                            with open(path, 'wb') as f:
                                total_length = int(r.headers.get('content-length'))
                                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                                    if chunk:
                                        f.write(chunk)
                                        f.flush()
                            url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"
                            r = requests.get(url, stream=True)
                            path = 'data/ag/ag_models/age_net.caffemodel'
                            with open(path, 'wb') as f:
                                total_length = int(r.headers.get('content-length'))
                                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                                    if chunk:
                                        f.write(chunk)
                                        f.flush()
                    elif mod == False and mod2 == True:
                        moi = messagebox.askyesno("Age & Gender Recognition", "one of age & gender recognition models are missing from your program, do you want to download them right now?")
                        if moi == True:
                            url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel"
                            r = requests.get(url, stream=True)
                            path = 'data/ag/ag_models/gender_net.caffemodel'
                            with open(path, 'wb') as f:
                                total_length = int(r.headers.get('content-length'))
                                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                                    if chunk:
                                        f.write(chunk)
                                        f.flush()
                    elif mod == True and mod2 == False:
                        moi = messagebox.askyesno("Age & Gender Recognition", "one of age & gender recognition models are missing from your program, do you want to download them right now?")
                        if moi == True:
                            url = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"
                            r = requests.get(url, stream=True)
                            path = 'data/ag/ag_models/age_net.caffemodel'
                            with open(path, 'wb') as f:
                                total_length = int(r.headers.get('content-length'))
                                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                                    if chunk:
                                        f.write(chunk)
                                        f.flush()
                    else:
                        os.system("python data/ag/ag.prx")
                        current_time = datetime.datetime.now()
                        with open('data/ag/{}{}.prx'.format(current_time.day, current_time.hour), 'r') as f:
                            lines = f.readlines()
                            print(lines)
                elif key == ord('e'):
                    current_time = datetime.datetime.now()
                    cropi = frame[top - 20:bottom + 20, left - 20:right + 20]
                    pe = Image.fromarray(cropi)
                    pe.save("data/ed/{}{}.jpg".format(current_time.day, current_time.hour))
                    os.system("python data/ed/ed.prx")
                    with open('data/ed/{}{}.prx'.format(current_time.day, current_time.hour), 'r') as f:
                        lines = []
                        for line in f:
                            lines.append(line)
                            print(line)
                elif key == ord('o'):
                    os.system("python data/recog/recog.py {}".format(name))
                warnings.filterwarnings("ignore")
                # Draw a box around the face using the Pillow module
                if name != "unknown":
                    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
                elif name == "unknown":
                    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                warnings.filterwarnings("ignore")
                # There's a bug in Pillow where it blows up with non-UTF-8 text
                # when using the default bitmap font
                warnings.filterwarnings("ignore")
                # Draw a label with a name below the face
                text_bbox = draw.textbbox((0, 0), name)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                warnings.filterwarnings("ignore")
                if name != "unknown":
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
                elif name == "unknown":
                    draw.text((10, 50), 'warning!', fill=(0, 0, 255))
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                current_time = datetime.datetime.now()
                day = current_time.strftime("%A")  # Get the full name of the day
                hour = current_time.hour
                current_time_str = f"{day}_{hour}"
                with open(f"database/Logs/{current_time_str}", "a") as f:
                    current_time = datetime.datetime.now()
                    day = current_time.strftime("%A")  # Get the full name of the day
                    hour = current_time.hour
                    current_time_str = f"{day}_{hour}"
                    f.write(f"{name} Logined at {current_time}\n")
                    with open(f"database/Logs/{current_time_str}", 'r') as f:
                        lines = f.readlines()
                    name_is_the_same = True
                    if name_is_the_same:
                        lines = [line for line in lines if not line.startswith(f"{name} Logined at")]
                    with open(f"database/Logs/{current_time_str}", 'w') as f:
                        f.writelines(lines)
                name = name.encode("UTF-8")
                nof = len(X_face_locations)
                if nof == None or nof == 0:
                    draw.text((10, 30), 'number of faces: 0', fill=(0, 255, 0))
                else:
                    draw.text((10, 30), 'number of faces:' + str(nof), fill=(0, 255, 0))
                warnings.filterwarnings("ignore")
                draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 255, 0))
                warnings.filterwarnings("ignore")
            # Remove the drawing library from memory as per the Pillow docs.
            del draw
            # Save image in open-cv format to be able to show it.
            opencvimage = np.array(pil_image)
            return opencvimage
        if __name__ == "__main__":
            train_a = messagebox.askyesno("BW face recognition", "do you want to retrain all faces again?(click yes if you have add new faces)")
            if train_a == True:
                print(f"{Fore.YELLOW} Getting requirements ready to train all models, this may take some minutes if you have 99+ trained faces pictures")
                classifier = train("database/recognized_faces/train", model_save_path="data/prx_models/trained_faces.prx", n_neighbors=2)
            print(f"{Fore.GREEN}Training complete!")
            print(f"{Fore.WHITE} ")
            # process one frame in every 30 frames for speed
            process_this_frame = 29
            cap = cv2.VideoCapture(camcapi)
            while 1:
                ret, frame = cap.read()
                if ret:
                    # Different resizing options can be chosen based on desired program runtime.
                    # Image resizing for more stable streaming
                    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    process_this_frame = process_this_frame + 1
                    if process_this_frame % 30 == 0:
                        predictions = predict(img, model_path="data/prx_models/trained_faces.prx")
                    frame = show_prediction_labels_on_image(frame, predictions)
                    cv2.imshow('face recognition', frame)
                    if 27 == cv2.waitKey(1):
                        root3.deiconify()
                        cap.release()
                        cv2.destroyAllWindows()
                        break
    def submit2():
        root3.withdraw()
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        warnings.filterwarnings("ignore")
        def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
            X = []
            y = []
            # Loop through each person in the training set
            for class_dir in os.listdir(train_dir):
                if not os.path.isdir(os.path.join(train_dir, class_dir)):
                    continue
                # Loop through each training image for the current person
                for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                    image = face_recognition.load_image_file(img_path)
                    face_bounding_boxes = face_recognition.face_locations(image)
                    if len(face_bounding_boxes) != 1:
                        # If there are no people (or too many people) in a training image, skip the image.
                        if verbose:
                            print(" Image {} not suitable for training: {}".format(img_path, " Didn't find a face" if len(face_bounding_boxes) < 1 else " Found more than one face"))
                    else:
                        # Add face encoding for current image to the training set
                        X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                        y.append(class_dir)
            # Determine how many neighbors to use for weighting in the KNN classifier
            if n_neighbors is None:
                n_neighbors = int(round(math.sqrt(len(X))))
                if verbose:
                    print(" Chose n_neighbors automatically:", n_neighbors)
            # Create and train the KNN classifier
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
            knn_clf.fit(X, y)
            # Save the trained KNN classifier
            if model_save_path is not None:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(knn_clf, f)
            return knn_clf
        def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
            if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
                raise Exception(" Invalid image path: {}".format(X_img_path))
            if knn_clf is None and model_path is None:
                raise Exception(" Must supply knn classifier either thourgh knn_clf or model_path")
            # Load a trained KNN model (if one was passed in)
            if knn_clf is None:
                with open(model_path, 'rb') as f:
                    knn_clf = pickle.load(f)
            # Load image file and find face locations
            global X_img
            X_img = face_recognition.load_image_file(X_img_path)
            global X_face_locations
            X_face_locations = face_recognition.face_locations(X_img)
            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []
            # Find encodings for faces in the compare iamge
            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
            # Use the KNN model to find the best matches for the compare face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
            # Predict classes and remove classifications that aren't within the threshold
            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        def show_prediction_labels_on_image(img_path, predictions):
            from PIL import Image, ImageDraw, ImageFont
            pil_image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_image)
            
            # Initialize `nof` to the number of faces found
            nof = len(predictions)
        
            for name, (top, right, bottom, left) in predictions:
                # Draw a box around the face using the Pillow module
                if name != "unknown":
                    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                else:
                    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
                
                # Load a font and calculate text size
                font = ImageFont.load_default()
                text_bbox = draw.textbbox((left, bottom), name, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw label with name below the face
                if name != "unknown":
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                else:
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 0, 0), outline=(255, 0, 0))
                
                draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 255, 0), font=font)
            
            # Insert the number of faces found into the Tkinter text widget
            t.insert('1.0', 'Found {} face(s) in this photograph.\n\n'.format(nof))
            
            # Display the resulting image
            pil_image.show()
            
        if __name__ == "__main__":
            train_a = messagebox.askyesno("BW face recognition", "do you want to retrain all faces again?(click yes if you have add new faces)")
            if train_a == True:
                print(f"{Fore.YELLOW} Getting requirements ready to train all models, this may take some minutes if you have 99+ trained faces pictures")
                classifier = train("database/recognized_faces/train", model_save_path="data/prx_models/trained_faces.prx", n_neighbors=2)
            print(f"{Fore.GREEN} Training complete!")
            print(f"{Fore.WHITE} ")
            warnings.filterwarnings("ignore")
            # STEP 2: Using the trained classifier, make predictions for unknown images
        from tkinter import filedialog as fd
        from tkinter.messagebox import showinfo
        root=tk.Tk()
        def disable_event():
            root3.deiconify()
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", disable_event)
        root.title('BW face recognition -  Choose Image File')
        root.geometry("600x350")
        name_var=tk.StringVar()
        def ret():
            t.insert("1.0", " Getting requirements ready to retrain all models, this may take some minutes if you have 99+ trained faces pictures\n")
            time.sleep(1)
            classifier = train("database/recognized_faces/train", model_save_path="data/prx_models/trained_faces.prx", n_neighbors=2)
            t.insert("1.0", "all faces got retrained!\n")
            time.sleep(1)
        def submit2():
            image_file = filename
            full_file_path = filename
            root.title("BW face recognition - " + filename)
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predict(full_file_path, model_path="data/prx_models/trained_faces.prx")
            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                t.insert("1.0", " Found {} at left: {}, top: {}, right: {}, bottom: {} In {} \n".format(name, left, top, right, bottom, image_file))
                time.sleep(1)
                warnings.filterwarnings("ignore")
            # Display results overlaid on an image
            show_prediction_labels_on_image(image_file, predictions)
            warnings.filterwarnings("ignore")
        # Let's Create some buttons to help the user and make the program more optional & friendly for a beginner user!
        name_var.set("")
        def submit3():
            global name
            name=name_var.get()
            root.title("BW face recognition - " + filename)
            # Load the jpg file into a numpy array
            image = face_recognition.load_image_file("{}".format(filename))
            image2 = face_recognition.load_image_file("{}".format(filename))
            image3 = face_recognition.load_image_file("{}".format(filename))
            # Find all facial features in all the faces in the image
            face_landmarks_list = face_recognition.face_landmarks(image)
            face_landmarks_list2 = face_recognition.face_landmarks(image3)
            face_locations = face_recognition.face_locations(image)
            face_locations2 = face_recognition.face_locations(image2)
            t.insert('1.0', 'founded {} face(s) in this photograph. \n'.format(len(face_landmarks_list)))
            time.sleep(1)
            # Create a PIL imagedraw object so we can draw on the picture
            from PIL import Image, ImageDraw
            pil_image = Image.fromarray(image)
            pil_image4 = Image.fromarray(image2)
            pil_image5 = Image.fromarray(image3)
            from PIL import Image, ImageDraw
            d = ImageDraw.Draw(pil_image)
            d2 = ImageDraw.Draw(pil_image4)
            d3 = ImageDraw.Draw(pil_image5)
            for (top, right, bottom, left) in face_locations:
                d.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                t.insert("1.0", f" A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}, saving face")
                time.sleep(1)
            for (top, right, bottom, left) in face_locations2:
                d2.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                nof = len(face_landmarks_list)
                d2.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
                face_image = image[top:bottom, left:right]
                from PIL import Image, ImageDraw
                pil_image3 = Image.fromarray(face_image)
                pil_image3.show()
                
                pil_image3.save("database/saved faces/{} cropped face {}.jpg".format(name, random.random()))
            for face_landmarks in face_landmarks_list2:
                for facial_feature in face_landmarks.keys():
                    d3.line(face_landmarks[facial_feature], width=4)
                    d3.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
            for face_landmarks in face_landmarks_list:
                # Print the location of each facial feature in this image
                for facial_feature in face_landmarks.keys():
                    t.insert('1.0', 'The {} in this face has the following points: {} \n'.format(facial_feature, face_landmarks[facial_feature]))
                    time.sleep(1)
                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
                    d.line(face_landmarks[facial_feature], width=4)
                    d.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
            # Show the picture
            pil_image.show()
            
            pil_image4.show()
            
            pil_image5.show()
            
            pil_image4.save("database/saved faces/{} full image(detection only) {}.jpg".format(name, random.random()))
            pil_image.save("database/saved faces/{} full image(biometrics + detection) {}.jpg".format(name, random.random()))
            pil_image5.save("database/saved faces/{} full image(biometrics only) {}.jpg".format(name, random.random()))
        # lets do some definition for buttons that we just got 
        def ed():
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
            from deepface import DeepFace
            # Load the image
            image_path = filename
            # Analyze the image
            root.title("BW face recognition - " + filename)
            try:
                demography = DeepFace.analyze(image_path, actions=['age', 'gender', 'emotion'])
                t.insert("1.0", f"result: {demography}")
            except ValueError:
                t.insert("1.0", f"Cannot find any face in {filename}")
        def sf():
            mn = input("please write the name of the face ==>> ")
            mn = mn.replace(" ", "_")
            cmn = os.path.isfile("database/recognized_faces/train/{}".format(mn))
            if cmn == True:
                shutil.copy(filename, "database/recognized_faces/train/{}/{}".format(cmn, random.random()))
                print(f"{Fore.GREEN}Face saved in database as {mn}")
            else:
                os.mkdir("database/recognized_faces/train/{}".format(mn))
                shutil.copy(filename, "database/recognized_faces/train/{}".format(mn))
                print(f"{Fore.GREEN}Face saved in database as {mn}")
        # create the root window
        text1 = ttk.Label(root, text = 'Choose your File')
        def select_file():
            global filename
            filename = fd.askopenfilename(filetypes=(("All Files", "*.*"),))
        # open button
        open_button = ttk.Button(
            root,
            text='Open a Image',
            command=select_file
        )
        def cos():
            t.delete('1.0', END)
        def scdef():
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save("database/saved faces/screenshot.png")
            filename = "database/saved faces/screenshot.png"
            image_file = filename
            full_file_path = filename
            root.title("BW face recognition - " + filename)
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predict(full_file_path, model_path="data/prx_models/trained_faces.prx")
            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                t.insert("1.0", " Found {} at left: {}, top: {}, right: {}, bottom: {} In {} \n".format(name, left, top, right, bottom, image_file))
                time.sleep(1)
                warnings.filterwarnings("ignore")
            # Display results overlaid on an image
            show_prediction_labels_on_image(image_file, predictions)
            warnings.filterwarnings("ignore")
            # Let's Create some buttons to help the user and make the program more optional & friendly for a beginner user!
            name_var.set("")
            name=name_var.get()
            root.title("BW face recognition - " + filename)
            # Load the jpg file into a numpy array
            image = face_recognition.load_image_file("{}".format(filename))
            image2 = face_recognition.load_image_file("{}".format(filename))
            image3 = face_recognition.load_image_file("{}".format(filename))
            # Find all facial features in all the faces in the image
            face_landmarks_list = face_recognition.face_landmarks(image)
            face_landmarks_list2 = face_recognition.face_landmarks(image3)
            face_locations = face_recognition.face_locations(image)
            face_locations2 = face_recognition.face_locations(image2)
            t.insert('1.0', 'founded {} face(s) in this photograph. \n'.format(len(face_landmarks_list)))
            time.sleep(1)
            # Create a PIL imagedraw object so we can draw on the picture
            from PIL import Image, ImageDraw
            pil_image = Image.fromarray(image)
            pil_image4 = Image.fromarray(image2)
            pil_image5 = Image.fromarray(image3)
            from PIL import Image, ImageDraw
            d = ImageDraw.Draw(pil_image)
            d2 = ImageDraw.Draw(pil_image4)
            d3 = ImageDraw.Draw(pil_image5)
            for (top, right, bottom, left) in face_locations:
                d.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                t.insert("1.0", f" A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}, saving face")
                time.sleep(1)
            for (top, right, bottom, left) in face_locations2:
                d2.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
                nof = len(face_landmarks_list)
                d2.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
                face_image = image[top:bottom, left:right]
                from PIL import Image, ImageDraw
                pil_image3 = Image.fromarray(face_image)
                pil_image3.show()
                
                pil_image3.save("database/saved faces/{} cropped face {}.jpg".format(name, random.random()))
            for face_landmarks in face_landmarks_list2:
                for facial_feature in face_landmarks.keys():
                    d3.line(face_landmarks[facial_feature], width=4)
                    d3.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
            for face_landmarks in face_landmarks_list:
                # Print the location of each facial feature in this image
                for facial_feature in face_landmarks.keys():
                    t.insert('1.0', 'The {} in this face has the following points: {} \n'.format(facial_feature, face_landmarks[facial_feature]))
                    time.sleep(1)
                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
                    d.line(face_landmarks[facial_feature], width=4)
                    d.text((10, 30), 'number of faces:' + str(nof), fill=(255, 0, 255))
            # Show the picture
            pil_image.show()
            
            pil_image4.show()
            
            pil_image5.show()
            
            pil_image4.save("database/saved faces/{} full image(detection only) {}.jpg".format(name, random.random()))
            pil_image.save("database/saved faces/{} full image(biometrics + detection) {}.jpg".format(name, random.random()))
            pil_image5.save("database/saved faces/{} full image(biometrics only) {}.jpg".format(name, random.random()))
        def dmc():
            command = ["python", "data/mesh/mesh.py", "-file", filename]
            subprocess.run(command, check=True)
        def fsc():
            root.withdraw()
            def disable_event():
                root.deiconify()
                root4.destroy()
            def face1f():
                global face1
                face1 = fd.askopenfilename(filetypes=(("All Files", "*.*"),))
            def face2f():
                global face2
                face2 = fd.askopenfilename(filetypes=(("All Files", "*.*"),))
            def swap():
                import cv2
                import dlib
                import numpy as np
                
                # Load the models
                predictor_path = "data/Swap_Models/shape_predictor_68_face_landmarks.dat"
                face_detector = dlib.get_frontal_face_detector()
                shape_predictor = dlib.shape_predictor(predictor_path)
                
                def get_landmarks(image):
                    faces = face_detector(image, 1)
                    if len(faces) == 0:
                        return None
                    return np.array([[p.x, p.y] for p in shape_predictor(image, faces[0]).parts()])
                
                def apply_affine_transform(src, src_tri, dst_tri, size):
                    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
                    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    return dst
                
                def warp_triangle(img1, img2, t1, t2):
                    r1 = cv2.boundingRect(np.float32([t1]))
                    r2 = cv2.boundingRect(np.float32([t2]))
                    
                    t1_rect = []
                    t2_rect = []
                    t2_rect_int = []
                
                    for i in range(3):
                        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
                        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
                        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
                
                    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
                
                    size = (r2[2], r2[3])
                    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
                
                    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
                    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
                
                    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (1 - mask) + img2_rect * mask
                
                def face_swap(img1, img2):
                    landmarks1 = get_landmarks(img1)
                    landmarks2 = get_landmarks(img2)
                
                    if landmarks1 is None or landmarks2 is None:
                        print("Face not detected in one or both images.")
                        return None
                
                    img1 = np.float32(img1)
                    img2 = np.float32(img2)
                
                    # Triangulation
                    rect = (0, 0, img1.shape[1], img1.shape[0])
                    subdiv = cv2.Subdiv2D(rect)
                    for p in landmarks1:
                        subdiv.insert((int(p[0]), int(p[1])))
                
                    triangles = subdiv.getTriangleList()
                    triangles = np.array(triangles, dtype=np.int32)
                
                    indices_triangles = []
                    for t in triangles:
                        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
                        indices = []
                        for pt in pts:
                            for k in range(len(landmarks1)):
                                if abs(pt[0] - landmarks1[k][0]) < 1.0 and abs(pt[1] - landmarks1[k][1]) < 1.0:
                                    indices.append(k)
                        if len(indices) == 3:
                            indices_triangles.append(indices)
                
                    for indices in indices_triangles:
                        t1 = [landmarks1[indices[0]], landmarks1[indices[1]], landmarks1[indices[2]]]
                        t2 = [landmarks2[indices[0]], landmarks2[indices[1]], landmarks2[indices[2]]]
                        warp_triangle(img1, img2, t1, t2)
                
                    return np.uint8(img2)
                
                # Load images
                img1 = cv2.imread(face1)
                img2 = cv2.imread(face2)
                
                # Perform face swap
                result = face_swap(img1, img2)
                
                if result is not None:
                    cv2.imshow('Face Swap', result)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Face swap failed.")
                
            root4=tk.Tk()
            root4.protocol("WM_DELETE_WINDOW", disable_event)
            root4.title('Face Swap')
            face1b = ttk.Button(root4, text = "Face no.1", command = face1f)
            face2b = ttk.Button(root4 , text = "Face no.2", command = face2f)
            Swapb = ttk.Button(root4, text = "Swap Faces", command = swap)
            face1b.pack()
            face2b.pack()
            Swapb.pack()
            root4.mainloop()
        
        # Example: Assuming you are using Tkinter for your GUI setup
        text1.pack()
        open_button.pack()
        text2 = ttk.Label(root, text='Choose your Operation')
        name_entry = ttk.Entry(root, textvariable=name_var, font=('calibre', 10, 'normal'))
        sub_btnn = ttk.Button(root, text='Recognize all Faces', command=submit2)
        sub_btn2 = ttk.Button(root, text='Find all biometrics on faces & save cropped faces', command=submit3)
        sub_btn5 = ttk.Button(root, text='Recognize emotions/age/gender', command=ed)
        screendef = ttk.Button(root, text='Recognize Faces From Screen', command=scdef)
        dm = ttk.Button(root, text="3D face mesh", command=dmc)
        fs = ttk.Button(root, text = "Face Swap", command = fsc)
        sub_btn6 = ttk.Button(root, text='Save this face in database', command=sf)
        l = ttk.Label(root, text='Refresh Train Faces (Do This If You Have Add New Faces OR This is the first time you are using this program)')
        ret = ttk.Button(root, text='Retrain all faces', command=ret)
        t = Text(root, height=20, width=40)
        l2 = ttk.Label(root, text='Output Shell')
        cos = ttk.Button(root, text='Clear Shell', command=cos)
        

        text2.pack()
        sub_btnn.pack()
        sub_btn2.pack()
        sub_btn5.pack()
        screendef.pack()
        dm.pack()
        fs.pack()
        sub_btn6.pack()
        l.pack()
        ret.pack()
        l2.pack()
        t.pack()
        cos.pack()
        t.update()
        root.mainloop()
    def disable_event():
        root2.deiconify()
        root3.destroy()
    root3=tk.Tk()
    root3.protocol("WM_DELETE_WINDOW", disable_event)
    root3.title('Choose the Program')
    root3.geometry("268x110")
    root3.resizable(0, 0)
    name_var=tk.StringVar()
    text2 = ttk.Label(root3, text = 'Choose the Program')
    name_entry = tk.Entry(root3,textvariable = name_var, font=('calibre',10,'normal'))
    sub_btnn=ttk.Button(root3,text = 'face recognition from real time camera', command = submit)
    sub_btn2=ttk.Button(root3,text = 'face recognition from picture', command = submit2)
    text2.pack()
    sub_btnn.pack()
    sub_btn2.pack()
    root3.mainloop()
def fd():
    root2.withdraw()
    def submit():
        def main():
            import mediapipe as mp
            cap = cv.VideoCapture(camcapi)
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection= 1,
                min_detection_confidence= 0.5,
            )
            while 1:
                ret, image = cap.read()
                if not ret:
                    break
                
                global debug_image
                debug_image = copy.deepcopy(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                results = face_detection.process(image)
                if results.detections is not None:
                    for detection in results.detections:
                        debug_image = draw_detection(debug_image, detection)
                cv.imshow("BW face recognition - face detection", debug_image)
                key = cv.waitKey(1)
                if 27 == cv.waitKey(1):
                    break
            cap.release()
            cv.destroyAllWindows()
        def draw_detection(image, detection):
            cv.flip(image, 1)
            image_width, image_height = image.shape[1], image.shape[0]
            bbox = detection.location_data.relative_bounding_box
            bbox.xmin = int(bbox.xmin * image_width)
            bbox.ymin = int(bbox.ymin * image_height)
            bbox.width = int(bbox.width * image_width)
            bbox.height = int(bbox.height * image_height)
            cv.rectangle(image, (int(bbox.xmin), int(bbox.ymin)),
                         (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)),
                         (0, 0, 0), 2)
            xleft, ytop, xright, ybot  = int(bbox.xmin), int(bbox.ymin), int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)
            crop_img = image[ytop: ybot, xleft: xright]
            key = cv.waitKey(1)
            current_time = datetime.datetime.now()
            if key == ord('c'):
                os.chdir("database/saved faces")
                crop_img = imutils.resize(crop_img, width=150)
                cv.imshow("cropped {}".format(random.random()), crop_img)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
            elif key == ord('s'):
                os.chdir("database/saved faces")
                crop_img = imutils.resize(crop_img, width=150)
                print("[FACE SAVED] face saved in database/saved faces")
                cv.imwrite("cropped face {} {} {} {} {}.jpg".format(current_time.day, current_time.hour, current_time.minute, current_time.second, random.random()), crop_img)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
            return image
        if __name__ == '__main__':
            main()
    def submit2():
        def main():
            import mediapipe as mp
            cap = cv.VideoCapture(camcapi)
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(
                model_selection= 1,
                min_detection_confidence= 0.5,
            )
            while 1:
                ret, image = cap.read()
                if not ret:
                    break
                
                global debug_image
                debug_image = copy.deepcopy(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                results = face_detection.process(image)
                if results.detections is not None:
                    for detection in results.detections:
                        debug_image = draw_detection(debug_image, detection)
                cv.imshow("BW face recognition - face hider", debug_image)
                key = cv.waitKey(1)
                if 27 == cv.waitKey(1):
                    break
            cap.release()
            cv.destroyAllWindows()
        def draw_detection(image, detection):
            cv.flip(image, 1)
            image_width, image_height = image.shape[1], image.shape[0]
            bbox = detection.location_data.relative_bounding_box
            bbox.xmin = int(bbox.xmin * image_width)
            bbox.ymin = int(bbox.ymin * image_height)
            bbox.width = int(bbox.width * image_width)
            bbox.height = int(bbox.height * image_height)
            cv.rectangle(image, (int(bbox.xmin), int(bbox.ymin)),
                         (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)),
                         (0, 0, 0), -1)
            xleft, ytop, xright, ybot  = int(bbox.xmin), int(bbox.ymin), int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)
            crop_img = image[ytop: ybot, xleft: xright]
            key = cv.waitKey(1)
            current_time = datetime.datetime.now()
            if key == ord('c'):
                os.chdir("database/saved faces")
                crop_img = imutils.resize(crop_img, width=150)
                cv.imshow("cropped {}".format(random.random()), crop_img)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
            elif key == ord('s'):
                os.chdir("database/saved faces")
                crop_img = imutils.resize(crop_img, width=150)
                print("[FACE SAVED] face saved in database/saved faces")
                cv.imwrite("cropped face {} {} {} {} {}.jpg".format(current_time.day, current_time.hour, current_time.minute, current_time.second, random.random()), crop_img)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
                pd = os.path.dirname(os.getcwd())
                os.chdir(pd)
            return image
        if __name__ == '__main__':
            main()
    def submit3():
        def main():
            import mediapipe as mp
            init()
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_face_mesh = mp.solutions.face_mesh
            # For webcam input:
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            cap = cv2.VideoCapture(camcapi)
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
              while cap.isOpened():
                success, image = cap.read()
                if not success:
                  print("Ignoring empty camera frame.")
                  # If loading a video, use 'break' instead of 'continue'.
                  continue
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = face_mesh.process(image)
                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                if results.multi_face_landmarks:
                  for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                cv2.imshow('Face Mesh', image)
                if cv2.waitKey(5) & 0xFF == 27:
                  break
            cap.release()
            cv2.destroyAllWindows()
        if __name__ == '__main__':
            main()
    root=tk.Tk()
    def disable_event():
        root2.deiconify()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", disable_event)
    root.title('Choose the Program')
    root.geometry("200x120")
    root.resizable(0, 0)
    name_var=tk.StringVar()
    text2 = ttk.Label(root, text = 'Choose the Program')
    name_entry = tk.Entry(root,textvariable = name_var, font=('calibre',10,'normal'))
    sub_btnn=ttk.Button(root,text = 'face detection', command = submit)
    sub_btn2=ttk.Button(root,text = 'face hider', command = submit2)
    sub_btn3=ttk.Button(root,text = 'face mesh', command = submit3)
    text2.pack()
    sub_btnn.pack()
    sub_btn2.pack()
    sub_btn3.pack()
    root.mainloop()
def cs():
    os.system("clear")
def cc():
    for clean_up in glob.glob('data/ed/*.*'):
        if not clean_up.endswith('ed.prx'):    
            os.remove(clean_up)
    for clean_up in glob.glob('data/ag/*.*'):
        if not clean_up.endswith('ag.prx'):    
            os.remove(clean_up)
import tkinter as tk
root2=tk.Tk()
root2.title('BW face recognition')
root2.geometry("250x170")
root2.resizable(0, 0)
name_var=tk.StringVar()
text2 = ttk.Label(root2, text = 'Choose the Program')
name_entry = tk.Entry(root2,textvariable = name_var, font=('calibre',10,'normal'))
sub_btnn=ttk.Button(root2,text = 'Face Recognition', command = fr)
sub_btn2=ttk.Button(root2,text = 'Face Detection', command = fd)
recam=ttk.Button(root2,text = 'Choose Another Camera Input', command = ccam)
csb=ttk.Button(root2, text = "Clear Shell", command = cs)
ccb=ttk.Button(root2, text = "Clear Cache", command = cc)
text2.pack()
sub_btnn.pack()
sub_btn2.pack()
recam.pack()
csb.pack()
ccb.pack()
root2.mainloop()
