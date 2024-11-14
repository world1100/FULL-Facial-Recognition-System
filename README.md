# FULL-Facial-Recognition-System

## WARNING

### The creator of this project is Kasra Moradi. If you are a content creator or want to share this project, please give credit.

### How to Install:

<!DOCTYPE html>
<html>
<head>
  
</head>
<body>

<h3>Operating System Compatibility</h3>

<table>
    <tr>
        <th>Operating System</th>
        <th>Status</th>
        <th>Recommendation</th>
    </tr>
    <tr>
        <td>Windows</td>
        <td class="supported">Supported</td>
        <td class="not-recommended">Not recommended</td>
    </tr>
    <tr>
        <td>Linux</td>
        <td class="supported">Supported</td>
        <td class="supported">Recommended</td>
    </tr>
    <tr>
        <td>Mac</td>
        <td class="not-tested">Not tested yet</td>
        <td class="not-tested">Unknown</td>
    </tr>
</table>
</body>
</html>
<h3>Python Version Compatibility</h3>
<h4>Any Python version from 3.9 to 3.12 is recommended</h4>

### First Step:
<code>git clone https://github.com/KasraMoradi-0/FULL-Facial-Recognition-System</code>

Then run:

<code>pip install -r requirements.txt</code>

By running this, pip will automatically install every library needed to run the program. Remember, if you are on Windows, some libraries like dlib or pillow may cause issues.

### To Install dlib on Windows:

Download the compatible dlib wheel file for your Python version from <a href="https://github.com/z-mahmud22/Dlib_Windows_Python3.x">here</a>, And then do `pip install 'dlib_wheel.whl`.

## Problem with installation?

### I know that installation for this program might be a little bit complex especially if you are new to python, just put on an<a href="https://github.com/KasraMoradi-0/FULL-Facial-Recognition-System/issues"> issue </a>on this repository and i will answer it as soon as possible

### After Installation

Once all required libraries are installed, you’re ready to run the program.

### Run the Program:

#### Linux: `python3 "facial recognition system.py"`
#### Windows: `py "facial recognition system.py"`

After running the code, you’ll choose your camera, whether it’s wireless or internal. If using an internal camera, select "internal" for the first question. For the second question, choose the camera number (e.g., if you have two cameras, the first is 0, and the second is 1). For IP mode, simply enter your camera IP, whether local, port-forwarded, or part of a home security setup.

#### Start Using It

To get started, perform simple face detection by clicking the face detection button on the GUI page.

![image](https://github.com/user-attachments/assets/aad30e45-2802-48c1-8701-9dd1522ed612)

In this section, there are several options, such as face detection, face hider, and face mesh. For example, by selecting face detection, you can access features only for the camera.

![image](https://github.com/user-attachments/assets/2768ac9d-b32c-400f-a8b9-05326a787bec)

Here is how real-time face detection from a camera looks (I am holding my phone in front of the camera):

![image](https://github.com/user-attachments/assets/83a114a7-9a18-4d7c-ba30-ac39c93b25a8)

Here’s how face hider looks:

![image](https://github.com/user-attachments/assets/a518d58e-95dd-44a3-a8b4-176e5ddf4042)

And finally, Face Mesh:

![image](https://github.com/user-attachments/assets/4bcbd58f-6eb3-4306-bb11-fae9e514f885)

### Features:

In face detection, pressing `c` shows a cropped image of your face, pressing `s` saves the cropped face in `database/saved faces`, and pressing/holding the escape key closes the window.

Now, let’s move to Face Recognition. Just click on the Face Recognition button on the main screen.

This time, I’m choosing face recognition from a picture to recognize a face from an image.

## Face Train

When running Face Recognition from a picture, there’s a y/n question.

![image](https://github.com/user-attachments/assets/9cd70663-009a-443e-acc5-d0a22eed3991)

Here’s how it works:

Add faces with names in `database/recognized_faces/train`. For example, Joe Biden is already there. Add folders with names like "Andrew" and place images of that person’s face in them (100 images is an example; more images lead to better recognition).

The program can’t recognize faces by itself initially, so you need to train it using the faces from the database. The program creates a file named `trained_faces.prx` in `data/prx_models` for this purpose.

![image](https://github.com/user-attachments/assets/81bb456d-f62b-4ab4-b029-3d58049ccdef)

When training is complete, it looks like this:

![image](https://github.com/user-attachments/assets/5c2ef36d-0d87-47cd-a48b-99f5b8bcf0e4)

## Warning

### NEVER share your personal `trained_faces.prx` file with anyone you don’t trust; it contains sensitive information that could be useful to hackers.

Note that you don’t always need to retrain faces. Only retrain when adding new faces or using the program for the first time.

## Face Recognition from Picture

![image](https://github.com/user-attachments/assets/8561aba9-f085-4a6e-bfa2-cb155b1ef693)

There are various options here. To start, open an image.

Click on "Find all Biometrics on Face & Save Cropped Faces."

![image](https://github.com/user-attachments/assets/4d5425bf-b9ae-4f10-9dde-29917d62df2c)

As you can see, all information about the face appears in the shell:

![image](https://github.com/user-attachments/assets/3737fa6e-53c8-4bb0-8de0-3417e15d3575)

Files are saved in `database/saved faces`.

For Face Recognition, click "Recognize all Faces."

![image](https://github.com/user-attachments/assets/9a9b5016-0c75-457d-a2a5-2dd0f41c23dc)

## Age, Gender, and Emotion Detection

These features require specific models, which are optional, so we’ll skip details here.

## Recognize Faces From Screen

Takes a screenshot and recognizes all faces in it. Nothing special, so we’ll skip details.

## 3D Face Mesh

This feature captures the face biometrics and presents them in a 3D environment.

![ezgif-2-949caffc84](https://github.com/user-attachments/assets/d3c02199-c3d9-40c2-abcf-ff599e7456ea)

## Face Swap

Clicking the Face Swap button closes the main window and opens this one:

![image](https://github.com/user-attachments/assets/c1a376c2-6f34-419d-9a35-6c924b4a59d9)

Choose Face No. 1 and Face No. 2, then click "swap". For example:

Face No. 1 is the face we’re placing on Face No. 2, and Face No. 2 is the face that will be replaced.

## Warning

Face Swap requires `shape_predictor_68_face_landmarks.dat` from `data/Swap_Models`. Get it <a href="https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat">here</a>.

![image](https://github.com/user-attachments/assets/356108c6-eaf2-4404-8f2d-9d8051ae0c2f)

## This feature is still in Beta, so it may not work well.

## Other Features

Clicking "Save this Face in Database" adds a new face and prompts for a name in the terminal.

"Retrain all Faces" retrains when new faces are added or when first using the program.

"Clear Shell" clears the GUI shell.

## Face Recognition from Real-Time Camera

![image](https://github.com/user-attachments/assets/faf2d8e3-a766-4572-9051-60dfc09ebad4)

![image](https://github.com/user-attachments/assets/ed2da06b-7097-4669-ba6c-7c5a0fa16334)

### Features:

- Press `s` to save an unknown face in the database
- Press `r` to retrain all faces
- Press `a` for age and gender detection

## Other Options:

### Choose another camera input:

Just in case you want to change your camera input without restarting the program

### Clear shell:

Clears terminal

### Clear cache:

Clears Cache files

## Still having issue?

- `@kasra_moradi_1` (on telegram)
- `kasramoradi517@gmail.com`
