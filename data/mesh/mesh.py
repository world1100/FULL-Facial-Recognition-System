import argparse
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

def main(image_path):
    # Use a dark background style
    style.use('dark_background')

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    # Load the image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and extract face mesh
    results = face_mesh.process(rgb_image)

    # Check if any face is detected
    if not results.multi_face_landmarks:
        print("No face detected.")
        exit()

    # Extract landmarks
    face_landmarks = results.multi_face_landmarks[0]
    landmark_coords = np.array([(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark])

    # Plot face mesh in 3D
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Plot the wireframe
    mesh = ax.plot_trisurf(landmark_coords[:, 0], landmark_coords[:, 1], landmark_coords[:, 2], linewidth=0.2, antialiased=True, color='cyan')

    # Set the view angle to the front view with 0 degrees azimuth and 180 degrees elevation
    ax.view_init(elev=0, azim=0)

    # Display the Matplotlib plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image for face mesh visualization.')
    parser.add_argument('-file', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()
    main(args.file)
