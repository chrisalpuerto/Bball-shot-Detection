from VideoInputTest import process_video_and_summarize
import json
import os
from moviepy.editor import VideoFileClip
from datetime import datetime

# === CONFIG ===
VIDEO_PATH = "videoDataset/meshooting.mp4"
OUTPUT_FOLDER = "clips"
PRE_CLIP = 2  # seconds before the shot
POST_CLIP = 2  # seconds after the shot

# === UTILS ===
def create_json(res):
    try:
        parsed_json = json.loads(res)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def time_to_seconds(ts):
    dt = datetime.strptime(ts, "%H:%M:%S")
    return dt.hour * 3600 + dt.minute * 60 + dt.second

def slice_clips_from_gemini_json(video_path, gemini_data, output_folder="clips"):
    shot_events = gemini_data.get("shot_events", [])
    os.makedirs(output_folder, exist_ok=True)
    video = VideoFileClip(video_path)

    for idx, shot in enumerate(shot_events):
        ts = shot.get("TS")
        label = shot.get("MM", "unknown").lower()

        if not ts:
            continue

        timestamp = time_to_seconds(ts)
        start = max(0, timestamp - PRE_CLIP)
        end = min(video.duration, timestamp + POST_CLIP)
        filename = f"{label}_{idx:03}.mp4"
        output_path = os.path.join(output_folder, filename)

        print(f"Clipping {filename} from {start}s to {end}s")
        video.subclip(start, end).write_videofile(output_path, codec="libx264", audio=False)

    video.close()

# === MAIN ===
if __name__ == "__main__":
    response_text = process_video_and_summarize(VIDEO_PATH)
    parsed_json = create_json(response_text)

    if parsed_json:
        slice_clips_from_gemini_json(VIDEO_PATH, parsed_json, OUTPUT_FOLDER)
