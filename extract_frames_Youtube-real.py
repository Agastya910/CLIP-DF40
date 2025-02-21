import cv2
import os
import glob

def get_next_frame_number(output_dir, video_name):
    """Get the next available frame number for a video to avoid overwriting existing frames"""
    existing_frames = glob.glob(os.path.join(output_dir, f"{video_name}_frame_*.jpg"))
    if not existing_frames:
        return 0
    # Extract numbers and find the maximum
    numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing_frames]
    return max(numbers) + 1

def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extract frames from a video and save them as images without overwriting existing files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Invalid frame rate for video {video_path}")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(fps * frame_interval)
    
    video_name = os.path.basename(video_path).replace(".mp4", "")
    start_number = get_next_frame_number(output_dir, video_name)

    print(f"Processing: {video_path}")
    print(f"Starting frame number: {start_number}")

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{start_number + saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Added {saved_count} new frames to {output_dir}\n")

def process_all_videos(video_dir, output_dir, frame_interval=10):
    """Process all videos in a directory with proper output folder handling"""
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    print(f"Processing {len(video_files)} videos from {video_dir}")
    print(f"Output directory: {output_dir}")
    
    for video_path in video_files:
        extract_frames(video_path, output_dir, frame_interval)

if __name__ == "__main__":
    # YouTube-real (real videos)
    process_all_videos(
        video_dir="data/celebdf/YouTube-real",
        output_dir="data/celebdf/frames/real",
        frame_interval=1
    )
