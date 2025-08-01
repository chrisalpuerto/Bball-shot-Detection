import os
import cv2
import shutil
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-pro")

# Directories
INPUT_VIDEO = "videoDataset/meshooting.mp4"
CLIP_DIR = "temp_clips"
OUTPUT_VIDEO = "output/highlights.mp4"
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

def extract_clips(video_path, clip_length=5):
    print("Splitting video into clips...")
    video = VideoFileClip(video_path)
    duration = int(video.duration)
    clips = []
    for start in range(0, duration, clip_length):
        clip_path = os.path.join(CLIP_DIR, f"clip_{start}.mp4")
        clip = video.subclip(start, min(start + clip_length, duration))
        clip.write_videofile(clip_path, codec="libx264", audio=False, verbose=False, logger=None)
        clips.append((clip_path, start))
    return clips

def extract_collage_from_clip(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    frame_paths = []

    for i in range(num_frames):
        time_sec = (duration / (num_frames + 1)) * (i + 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        success, frame = cap.read()
        if success:
            frame_paths.append(frame)
    cap.release()

    if len(frame_paths) != num_frames:
        return None

    collage = np.concatenate(frame_paths, axis=1)  # Side-by-side collage
    collage_path = video_path.replace(".mp4", "_collage.jpg")
    cv2.imwrite(collage_path, collage)
    return collage_path

def analyze_frame_with_gemini(image_path):
    prompt = """
These three images are from a 5-second basketball video clip, shown in order from left to right.
Do they show a player making a shot — like a layup, 3-pointer, or jump shot — where the ball goes into the basket?
If yes, describe the shot briefly. If not, say "No made shot".
"""
    try:
        img = genai.upload_file(image_path)
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return "No made shot"

def build_highlight_reel(selected_clips, output_path):
    print("Building final highlight reel...")
    final_clips = [VideoFileClip(path) for path, _ in selected_clips]
    final = concatenate_videoclips(final_clips)
    final.write_videofile(output_path, codec="libx264")

def main():
    all_clips = extract_clips(INPUT_VIDEO)
    selected = []

    print("Analyzing clips with Gemini (collage)...")
    for clip_path, start_time in all_clips:
        collage_path = extract_collage_from_clip(clip_path)
        if not collage_path:
            continue

        description = analyze_frame_with_gemini(collage_path)
        print(f"[{clip_path}] → Gemini says: {description}")

        if "no made shot" not in description.lower():
            selected.append((clip_path, description))

        os.remove(collage_path)

    if not selected:
        print("No highlights found.")
        return

    with open("output/highlight_descriptions.txt", "w") as f:
        for path, desc in selected:
            f.write(f"{os.path.basename(path)}: {desc}\n")

    build_highlight_reel([(p, 0) for p, _ in selected], OUTPUT_VIDEO)
    print("✅ Highlight video saved to:", OUTPUT_VIDEO)

    # Optional: keep temp_clips if you want to debug them

    shutil.rmtree(CLIP_DIR)
    os.makedirs(CLIP_DIR)
def delete_files(file_path_1):
    CLIP_DIR = file_path_1
    shutil.rmtree(CLIP_DIR)
    os.makedirs(CLIP_DIR)

if __name__ == "__main__":
    main()
