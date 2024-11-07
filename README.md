# FULL-Facial-Recognition-System

## WARNING

### The Creator of this project is Kasra Moradi, If you are a content creator or if you wanna share this project with someone, please put the credits

### How to install:

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
        <td class="not-tested">Haven't tested yet</td>
        <td class="not-tested">Unknown</td>
    </tr>
</table>
</body>
</html>
<h3>Python Version Compatibility</h3>
<h4>Any python version from 3.9 to 3.12 is recommended</h4>

### First Step:
<code>git clone https://github.com/KasraMoradi-0/FULL-Facial-Recognition-System</code>

And then run:

<code>pip install -r requirement.txt</code>

By running this code, pip will automatically install every single Library that is needed to run the program, But always remember, if you are on windows, some of them may have problems such as dlib or pillow
### To install dlib on windows:

download the compatible dlib wheel file for your python version from <a href="https://github.com/z-mahmud22/Dlib_Windows_Python3.x">here</a>

### After Installation

After you have successfully installed all libraries that are needed, You are ready to run the program

### Run the program:

#### Linux: python3 "facial recognition system.py"
#### Windows: py "facial recognition system.py"

After running the code, We are going to choose our camera, wether it's wireless or it's an internal camera, If you are Using an internal camera, then you have to choose internal on the first question, on the second question, depends on how many cameras are connected to your computer, you have to choose the number, for example i have 2 cameras connected to my computer, the camera number one is 0 and the camera number two is 1, if it's in IP mode, you just need to write your camera ip, wether it's local or you have a port forwarding or you want to create a home security system setup

#### Start using it

so to get started, lets do a simple face detection, click on face detection button that you can see on the GUI page

![image](https://github.com/user-attachments/assets/aad30e45-2802-48c1-8701-9dd1522ed612)

In here, there are Several options, Such as face detection, face hider and face mesh, for example i'm going to click on face detection, by the way, all the options are ONLY for camera in this section

![image](https://github.com/user-attachments/assets/2768ac9d-b32c-400f-a8b9-05326a787bec)

And here is how real time face detection from camera look like(i am holding my phone in my hand in fron tof the camera):

![image](https://github.com/user-attachments/assets/83a114a7-9a18-4d7c-ba30-ac39c93b25a8)

Here is how face hider look like:

![image](https://github.com/user-attachments/assets/a518d58e-95dd-44a3-a8b4-176e5ddf4042)

And finally, Face mesh:

![image](https://github.com/user-attachments/assets/4bcbd58f-6eb3-4306-bb11-fae9e514f885)


### Features:

### In face detection, by pressing c button, you can see your face cropped, by pressing s, the cropped face is saved in datqabase/saved faces, but pressing/holding escape button for several times the window closes

Now lets get to Face Recognition, just click on the Face Recognition button on the main screen

Now this time, i'm going to choose face recognition from picture to recognize a face from an image

## Face Train

Yes, you have noticed that when you run Face Recognition form picture there is a y/n question

![image](https://github.com/user-attachments/assets/9cd70663-009a-443e-acc5-d0a22eed3991)

So, let me explain how it works:

you can put faces + names in database/recognized_faces/train

For example. you can see Joe Biden is already there, You can add Names like: Andrew, and in Andrew folder, put like 100 images of Andrew face(100 is example, more pictures = better recognition)

but the program is not able to recognize this all by it self, so you need to train the program to recognize every single face from database, the program creates a file in data/prx_models named trained_faces.prx, here how it looks like to train faces:

![image](https://github.com/user-attachments/assets/81bb456d-f62b-4ab4-b029-3d58049ccdef)

And here how it looks like when the training is done:

![image](https://github.com/user-attachments/assets/5c2ef36d-0d87-47cd-a48b-99f5b8bcf0e4)

## Warning

### NEVER EVER GIVE YOUR VERY OWN trained_faces.prx FILE TO SOMEONE ELSE UNTIL YOU TRUST THEM, THERE ARE A LOT OF USEFUL INFO IN THAT FILE ESPECIALLY FOR HACKERS

### by the way, be aware that you don't always need to train faces again, you only need to do it when you add more faces, or this is the first time you are using the program

## Face Recognition from picutre
