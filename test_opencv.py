import cv2

video_path = "data/celebdf/Celeb-real/id0_0000.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
else:
    print(f"Success: Video {video_path} opened successfully")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame rate: {fps} fps")
    cap.release()
