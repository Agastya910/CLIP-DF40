import cv2
import os
import glob

def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extract frames from a video and save them as images.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save the extracted frames.
        frame_interval (int): Extract 1 frame every `frame_interval` seconds.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}. Check if the file exists and is a valid video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Invalid frame rate (0) for video {video_path}. The video might be corrupted.")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(fps * frame_interval)  # Number of frames to skip

    print(f"Processing video: {video_path}")
    print(f"Frame rate: {fps} fps")
    print(f"Total frames: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} seconds (every {frame_step} frames)")

    # Extract frames
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at the specified interval
        if frame_count % frame_step == 0:
            # Generate a unique filename using the video name and frame count
            video_name = os.path.basename(video_path).replace(".mp4", "")
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {output_dir}")

def process_all_videos(video_dir, output_dir, frame_interval=10):
    """
    Process all videos in a directory and extract frames.

    Args:
        video_dir (str): Directory containing video files.
        output_dir (str): Directory to save extracted frames.
        frame_interval (int): Extract 1 frame every `frame_interval` seconds.
    """
    # Get all video files in the directory
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

    # Process each video
    for video_path in video_files:
        extract_frames(video_path, output_dir, frame_interval)

if __name__ == "__main__":
    # Example usage
    video_dir = "data/celebdf/Celeb-real"          # Directory containing real videos
    output_dir = "data/celebdf/frames/real"  # Directory to save all frames
    process_all_videos(video_dir, output_dir, frame_interval=1)  # Extract 1 frame every 10 seconds