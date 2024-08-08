import os
import cv2

def load_video(input_video, output_folder, reshape_scale=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        height, width = frame.shape[:2]
        new_dim = (int(width * reshape_scale), int(height * reshape_scale))
        resized_frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
        
        filename = os.path.join(output_folder, f'{frame_number:05d}.jpg')
        cv2.imwrite(filename, resized_frame)
        
        frame_number += 1
    
    cap.release()
    print("Frames have been extracted and saved.")
